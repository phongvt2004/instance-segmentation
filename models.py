import torch
import torch.nn as nn
from torch import Tensor
import torchvision
from torchvision.models import (
    resnet50, ResNet50_Weights,
    swin_t, Swin_T_Weights,
    resnext101_32x8d # Keep this import
    # REMOVED: ResNeXt101_32x8d_Weights
)
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.models.detection.backbone_utils import (
    _resnet_fpn_extractor, _validate_trainable_layers, BackboneWithFPN
)
# Import MaskRCNN_ResNet50_FPN_Weights if using the pretrained option in maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import MultiScaleRoIAlign, misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool, ExtraFPNBlock

from collections import OrderedDict
import warnings
from typing import List, Dict, Tuple, Optional, Any, Callable

# --- Constants ---
DEFAULT_FPN_OUT_CHANNELS = 256
DEFAULT_NUM_CLASSES = 91  # Default for COCO

# --- Custom MaskRCNN Class (Keep as is) ---
class MyMaskRCNN(MaskRCNN):
    # ... (Your MyMaskRCNN code - no changes needed here) ...
    def __init__(
        self,
        backbone,
        num_classes=None,
        # transform parameters
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # RPN parameters
        rpn_anchor_generator=None,
        rpn_head=None,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters
        box_roi_pool=None,
        box_head=None,
        box_predictor=None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        # Mask parameters
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        **kwargs,
    ):

        if not isinstance(mask_roi_pool, (MultiScaleRoIAlign, type(None))):
            raise TypeError(
                f"mask_roi_pool should be of type MultiScaleRoIAlign or None instead of {type(mask_roi_pool)}"
            )

        # Handle num_classes default if not provided
        if num_classes is None and box_predictor is None and mask_predictor is None:
             num_classes = DEFAULT_NUM_CLASSES
        elif num_classes is None and (box_predictor is not None or mask_predictor is not None):
             # Let MaskRCNN handle the error if predictor is given but num_classes is None
             pass

        # Ensure backbone has out_channels before proceeding
        if not hasattr(backbone, "out_channels"):
            raise ValueError("Backbone must have an attribute 'out_channels'.")
        out_channels = backbone.out_channels

        if mask_roi_pool is None:
            # Check if backbone output has multiple feature maps (like FPN)
            # Heuristic: Check if output names look like FPN levels ('0', '1', ...) or 'pool'
            # This might need adjustment depending on backbone specifics
            featmap_names = ["0", "1", "2", "3"] # Default for FPN P2-P5
            # A more robust check would be needed if backbones output different keys
            # if hasattr(backbone, 'fpn'): # Or check isinstance(backbone, BackboneWithFPN) etc.
            #     featmap_names = list(backbone.fpn.inner_blocks.keys()) + list(backbone.fpn.layer_blocks.keys())
            #     if hasattr(backbone.fpn, 'extra_blocks'):
            #         featmap_names += list(backbone.fpn.extra_blocks.keys()) # e.g. ['pool']
            mask_roi_pool = MultiScaleRoIAlign(featmap_names=featmap_names, output_size=14, sampling_ratio=2)


        if mask_head is None:
            mask_layers = (256, 256, 256, 256)
            mask_dilation = 1
            mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)

        if mask_predictor is None:
            # Avoid error if num_classes is None because predictor was provided
            if num_classes is not None:
                 mask_predictor_in_channels = 256  # == mask_layers[-1]
                 mask_dim_reduced = 256
                 mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels, mask_dim_reduced, num_classes)
            else:
                 mask_predictor = None # Let superclass handle predictor logic

        super().__init__(
            backbone,
            num_classes,
            # transform parameters
            min_size,
            max_size,
            image_mean,
            image_std,
            # RPN-specific parameters
            rpn_anchor_generator,
            rpn_head,
            rpn_pre_nms_top_n_train,
            rpn_pre_nms_top_n_test,
            rpn_post_nms_top_n_train,
            rpn_post_nms_top_n_test,
            rpn_nms_thresh,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_score_thresh,
            # Box parameters
            box_roi_pool,
            box_head,
            box_predictor,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            # Mask parameters specific to MaskRCNN (handled below)
            **kwargs,
        )

        self.roi_heads.mask_roi_pool = mask_roi_pool
        self.roi_heads.mask_head = mask_head
        self.roi_heads.mask_predictor = mask_predictor

    # Keep your forward method as is
    def forward(self, images, targets=None):
        # Your forward implementation here...
        # --- Start Copied Forward ---
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))
        images, targets = self.transform(images, targets)

        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not hasattr(self, "_has_warned"): self._has_warned = False # Ensure attribute exists
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)
        # --- End Copied Forward ---

# --- Custom Swin FPN Backbone (Keep as is) ---
class CustomSwinFPN(nn.Module):
    # ... (Your CustomSwinFPN code - no changes needed here) ...
    def __init__(self, body, return_layer_keys, fpn_in_channels_list, fpn_out_channels, extra_blocks=None):
        super().__init__()
        self.body = body
        self.return_layer_keys = return_layer_keys
        self.fpn_map = {key: str(i) for i, key in enumerate(return_layer_keys)}
        print(f"CustomSwinFPN: Mapping body keys {list(self.fpn_map.keys())} to FPN keys {list(self.fpn_map.values())}")

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=fpn_in_channels_list,
            out_channels=fpn_out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = fpn_out_channels # Required by MaskRCNN

    def forward(self, x):
        features = self.body(x) # Expected B, H, W, C
        fpn_input = OrderedDict()
        for body_key, fpn_key in self.fpn_map.items():
            if body_key in features:
                tensor_bhwc = features[body_key]
                tensor_bchw = tensor_bhwc.permute(0, 3, 1, 2) # Permute B,H,W,C -> B,C,H,W
                fpn_input[fpn_key] = tensor_bchw
            else:
                raise KeyError(f"Expected key '{body_key}' not found in extractor output. Found keys: {features.keys()}")
        fpn_output = self.fpn(fpn_input)
        return fpn_output

# --- Combined Swin + ResNet-Like FPN Backbone (Keep as is) ---
class CombinedSwinResNetLikeFPN(nn.Module):
    # ... (Your CombinedSwinResNetLikeFPN code - no changes needed here) ...
    def __init__(self,
                 resnet_like_base: nn.Module,
                 resnet_like_channels: List[int],
                 pretrained_swin: bool = True,
                 projection_channels: int = 256,
                 fpn_out_channels: int = 256,
                 extra_blocks: Optional[ExtraFPNBlock] = None):
        super().__init__()

        # Swin Setup
        swin_weights = Swin_T_Weights.DEFAULT if pretrained_swin else None
        swin_base = swin_t(weights=swin_weights).cpu()
        swin_return_nodes = {
            'features.1.1.add_1': 'swin_feat0', # Stride 4 -> 96 C
            'features.3.1.add_1': 'swin_feat1', # Stride 8 -> 192 C
            'features.5.5.add_1': 'swin_feat2', # Stride 16-> 384 C
            'features.7.1.add_1': 'swin_feat3', # Stride 32-> 768 C
        }
        self.swin_output_keys = list(swin_return_nodes.values())
        swin_channels = [96, 192, 384, 768]
        print("Creating Swin feature extractor for combined model...")
        self.body_swin = create_feature_extractor(swin_base, return_nodes=swin_return_nodes)

        # ResNet-like Setup
        self.resnet_like_base = resnet_like_base.cpu()
        resnet_like_return_nodes = {
            'layer1': 'res_feat0', # Stride 4
            'layer2': 'res_feat1', # Stride 8
            'layer3': 'res_feat2', # Stride 16
            'layer4': 'res_feat3', # Stride 32
        }
        self.resnet_like_output_keys = list(resnet_like_return_nodes.values())
        print("Creating ResNet-like feature extractor for combined model...")
        self.body_resnet_like = create_feature_extractor(self.resnet_like_base, return_nodes=resnet_like_return_nodes)

        # Projection Layers
        self.projection_convs = nn.ModuleList()
        fpn_in_channels_list = []
        print("Creating projection layers for combined model:")
        for i in range(len(swin_channels)):
            concat_channels = swin_channels[i] + resnet_like_channels[i]
            proj_conv = nn.Conv2d(concat_channels, projection_channels, kernel_size=1)
            self.projection_convs.append(proj_conv)
            fpn_in_channels_list.append(projection_channels)
            print(f"  Stage {i}: Concat Channels={concat_channels} -> Proj Conv -> Out Channels={projection_channels}")

        # FPN
        print(f"Creating FPN for combined model with input channels: {fpn_in_channels_list}, output channels: {fpn_out_channels}")
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=fpn_in_channels_list,
            out_channels=fpn_out_channels,
            extra_blocks=extra_blocks,
        )
        self.out_channels = fpn_out_channels # Required by MaskRCNN

    def forward(self, x):
        swin_features = self.body_swin(x)
        resnet_like_features = self.body_resnet_like(x)
        fpn_input = OrderedDict()
        for i in range(len(self.swin_output_keys)):
            swin_key = self.swin_output_keys[i]
            res_key = self.resnet_like_output_keys[i]
            fpn_key = str(i)
            swin_f = swin_features[swin_key]
            res_f = resnet_like_features[res_key]
            swin_f_permuted = swin_f.permute(0, 3, 1, 2) # B,H,W,C -> B,C,H,W
            if swin_f_permuted.shape[2:] != res_f.shape[2:]:
                swin_f_permuted = nn.functional.interpolate(swin_f_permuted, size=res_f.shape[2:], mode='bilinear', align_corners=False)
            combined_f = torch.cat([swin_f_permuted, res_f], dim=1)
            projected_f = self.projection_convs[i](combined_f)
            fpn_input[fpn_key] = projected_f
        fpn_output = self.fpn(fpn_input)
        return fpn_output


# --- Factory Function for ResNet50 Backbone (Keep as is) ---
def maskrcnn_resnet50_fpn(
    *,
    num_classes: Optional[int] = None,
    pretrained: bool = False, # Whether to use pretrained MaskRCNN weights (if available)
    pretrained_backbone: bool = True, # Kept for consistency, but primarily controlled by `pretrained` logic now
    trainable_backbone_layers: Optional[int] = None, # 0-5, how many ResNet layers to train (higher means more)
    progress: bool = True,
    **kwargs: Any,
) -> MyMaskRCNN:
    """
    Constructs a Mask R-CNN model with a ResNet-50-FPN backbone.

    Args:
        num_classes (Optional[int]): Number of classes (including background). Defaults to 91 (COCO).
        pretrained (bool): If True, attempts to load weights pretrained on COCO from torchvision.
        pretrained_backbone (bool): If True (and pretrained=False), loads ImageNet weights for the ResNet backbone.
        trainable_backbone_layers (Optional[int]): Number of trainable (unfrozen) layers starting from the top.
                                                  Valid values are 0 to 5. If None, defaults based on `pretrained`.
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    weights = None # Initialize weights variable for the underlying call
    weights_backbone = None # Initialize backbone weights variable

    if pretrained:
         # Set conditions for using the official torchvision pretrained model
         if trainable_backbone_layers is None: trainable_backbone_layers = 3
         if num_classes is None: num_classes = DEFAULT_NUM_CLASSES

         if num_classes == DEFAULT_NUM_CLASSES and trainable_backbone_layers == 3:
             print("Loading pre-trained MaskRCNN ResNet50 FPN from torchvision...")
             # Get the weights enum
             weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
             # Filter kwargs to avoid conflict with the 'weights' enum parameter
             kwargs_filtered = {k: v for k, v in kwargs.items() if k not in ['weights', 'pretrained', 'pretrained_backbone', 'weights_backbone']}
             model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                 weights=weights,
                 progress=progress,
                 # num_classes and trainable_backbone_layers are implicitly handled by the weights enum
                 **kwargs_filtered # Pass only non-conflicting kwargs
             )
             print("Note: Returning standard torchvision MaskRCNN when using exact pretrained weights.")
             # Decide if you need to cast to MyMaskRCNN or return directly
             # If MyMaskRCNN has no critical overrides, returning torchvision's is fine.
             # If MyMaskRCNN is needed, you'd have to rebuild and load state_dict manually.
             return model # Return torchvision model directly for simplicity here
         else:
             print("Warning: Pretrained weights requested but num_classes or trainable_layers differ from torchvision defaults. Building model from scratch with pretrained backbone.")
             pretrained_backbone = True # Ensure backbone weights are loaded if pretrained=True failed
             # Continue to build from scratch below

    # --- Build model from scratch ---
    # Determine backbone weights only if not loading full pretrained model above
    if weights is None: # Ensure we are in the 'build from scratch' path
        weights_backbone = ResNet50_Weights.DEFAULT if pretrained_backbone else None

    # Validate trainable layers based on whether backbone weights are loaded
    trainable_backbone_layers = _validate_trainable_layers(weights_backbone is not None, trainable_backbone_layers, 5, 3)
    norm_layer = misc_nn_ops.FrozenBatchNorm2d if weights_backbone is not None else nn.BatchNorm2d

    print(f"Building ResNet50 backbone (trainable layers: {trainable_backbone_layers})...")
    backbone_model = resnet50(weights=weights_backbone, progress=progress, norm_layer=norm_layer)
    # Use torchvision's helper to create ResNet+FPN
    backbone = _resnet_fpn_extractor(backbone_model, trainable_backbone_layers, norm_layer=norm_layer)

    # Use the provided num_classes, defaulting if necessary
    if num_classes is None:
        num_classes = DEFAULT_NUM_CLASSES

    print("Creating MyMaskRCNN with ResNet50 FPN backbone...")
    # Pass potentially relevant kwargs (excluding those handled by backbone creation)
    model = MyMaskRCNN(backbone, num_classes=num_classes, **kwargs)

    return model

# --- Factory Function for Swin-T Backbone (Keep as is) ---
def maskrcnn_swin_t_fpn(
    *,
    num_classes: Optional[int] = None,
    pretrained_backbone: bool = True,
    fpn_out_channels: int = DEFAULT_FPN_OUT_CHANNELS,
    progress: bool = True,
    **kwargs: Any,
) -> MyMaskRCNN:
    # ... (Your maskrcnn_swin_t_fpn code - no changes needed here) ...
    print("Creating Swin-T backbone...")
    # Use string alias for weights
    weights_swin_param = "DEFAULT" if pretrained_backbone else None
    swin_model = swin_t(weights=weights_swin_param, progress=progress).cpu()

    return_nodes_for_extractor = {
        'features.1.1.add_1': 'feat0',
        'features.3.1.add_1': 'feat1',
        'features.5.5.add_1': 'feat2',
        'features.7.1.add_1': 'feat3',
    }
    extractor_output_keys = list(return_nodes_for_extractor.values())
    fpn_in_channels_list = [96, 192, 384, 768] # Swin-T specific

    print("Creating Swin feature extractor body...")
    body = create_feature_extractor(swin_model, return_nodes=return_nodes_for_extractor)

    print("Creating CustomSwinFPN wrapper...")
    custom_backbone = CustomSwinFPN(
        body=body,
        return_layer_keys=extractor_output_keys,
        fpn_in_channels_list=fpn_in_channels_list,
        fpn_out_channels=fpn_out_channels,
        extra_blocks=LastLevelMaxPool()
    )

    print("Creating MyMaskRCNN with Swin-T FPN backbone...")
    model = MyMaskRCNN(custom_backbone, num_classes=num_classes, **kwargs)
    return model


# --- Factory Function for ResNeXt101 Backbone ---
def maskrcnn_resnext101_fpn(
    *,
    num_classes: Optional[int] = None,
    pretrained_backbone: bool = True, # Whether to use pretrained ResNeXt weights
    trainable_backbone_layers: Optional[int] = None, # 0-5
    progress: bool = True,
    **kwargs: Any,
) -> MyMaskRCNN:
    """
    Constructs a Mask R-CNN model with a ResNeXt-101-32x8d FPN backbone.
    """
    # Use string alias for weights
    weights_backbone_param = "DEFAULT" if pretrained_backbone else None
    is_pretrained = weights_backbone_param is not None

    trainable_backbone_layers = _validate_trainable_layers(is_pretrained, trainable_backbone_layers, 5, 3)

    print(f"Building ResNeXt101 backbone (trainable layers: {trainable_backbone_layers})...")
    # Pass the string or None to the weights parameter
    backbone_model = resnext101_32x8d(weights=weights_backbone_param, progress=progress)

    # Freeze layers manually
    if is_pretrained and trainable_backbone_layers < 5:
        layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_backbone_layers]
        # Note: ResNeXt typically doesn't have a top-level 'bn1' like ResNet,
        # the first norm is inside 'conv1'. Freezing handled by not including 'conv1' etc.
        for name, parameter in backbone_model.named_parameters():
             # Freeze if the layer name doesn't start with any of the layers to train
            if not any(name.startswith(layer) for layer in layers_to_train):
                 parameter.requires_grad_(False)
                 # print(f"Freezing: {name}") # Optional: verify frozen layers

    # FPN setup
    extra_blocks = LastLevelMaxPool()
    returned_layers = [1, 2, 3, 4]
    return_layers_dict = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}
    in_channels_list = [256, 512, 1024, 2048] # ResNeXt101 specific
    fpn_out_channels = DEFAULT_FPN_OUT_CHANNELS

    print("Creating ResNeXt101 BackboneWithFPN...")
    backbone = BackboneWithFPN(
        backbone_model, return_layers_dict, in_channels_list, fpn_out_channels, extra_blocks=extra_blocks
    )

    print("Creating MyMaskRCNN with ResNeXt101 FPN backbone...")
    model = MyMaskRCNN(backbone, num_classes=num_classes, **kwargs)
    return model


# --- Factory Function for Combined Swin + ResNet50 Backbone ---
def maskrcnn_swin_resnet50_fpn(
    *,
    num_classes: Optional[int] = None,
    pretrained_swin: bool = True,
    pretrained_resnet: bool = True,
    projection_channels: int = DEFAULT_FPN_OUT_CHANNELS,
    fpn_out_channels: int = DEFAULT_FPN_OUT_CHANNELS,
    progress: bool = True,
    **kwargs: Any,
) -> MyMaskRCNN:
    """
    Constructs Mask R-CNN with a combined Swin-T + ResNet50 FPN backbone.
    """
    print("Loading ResNet50 base for combined model...")
    resnet_weights_param = ResNet50_Weights.DEFAULT if pretrained_resnet else None
    resnet_base = resnet50(weights=resnet_weights_param, progress=progress)
    resnet_channels = [256, 512, 1024, 2048] # ResNet50 specific

    print("Instantiating CombinedSwinResNetLikeFPN (with ResNet50)...")
    combined_backbone = CombinedSwinResNetLikeFPN(
        resnet_like_base=resnet_base,
        resnet_like_channels=resnet_channels,
        pretrained_swin=pretrained_swin, # Pass boolean flag
        projection_channels=projection_channels,
        fpn_out_channels=fpn_out_channels,
        extra_blocks=LastLevelMaxPool()
    )

    print("Creating MyMaskRCNN with combined Swin+ResNet50 backbone...")
    model = MyMaskRCNN(combined_backbone, num_classes=num_classes, **kwargs)
    return model


# --- Factory Function for Combined Swin + ResNeXt101 Backbone ---
def maskrcnn_swin_resnext101_fpn(
    *,
    num_classes: Optional[int] = None,
    pretrained_swin: bool = True,
    pretrained_resnext: bool = True,
    projection_channels: int = DEFAULT_FPN_OUT_CHANNELS,
    fpn_out_channels: int = DEFAULT_FPN_OUT_CHANNELS,
    progress: bool = True,
    **kwargs: Any,
) -> MyMaskRCNN:
    """
    Constructs Mask R-CNN with a combined Swin-T + ResNeXt-101 FPN backbone.
    """
    print("Loading ResNeXt101 base for combined model...")
    # Use string alias for weights
    resnext_weights_param = "DEFAULT" if pretrained_resnext else None
    resnext_base = resnext101_32x8d(weights=resnext_weights_param, progress=progress)
    resnext_channels = [256, 512, 1024, 2048] # ResNeXt101 specific

    print("Instantiating CombinedSwinResNetLikeFPN (with ResNeXt101)...")
    combined_backbone = CombinedSwinResNetLikeFPN(
        resnet_like_base=resnext_base,
        resnet_like_channels=resnext_channels,
        pretrained_swin=pretrained_swin, # Pass boolean flag
        projection_channels=projection_channels,
        fpn_out_channels=fpn_out_channels,
        extra_blocks=LastLevelMaxPool()
    )

    print("Creating MyMaskRCNN with combined Swin+ResNeXt101 backbone...")
    model = MyMaskRCNN(combined_backbone, num_classes=num_classes, **kwargs)
    return model


# --- Example Usage (Keep as is) ---
if __name__ == '__main__':
    # ... (Your example usage code - no changes needed here) ...
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dummy_input = torch.randn(1, 3, 512, 512).to(device) # Smaller size for quicker testing

    # Test ResNet50
    print("\n--- Testing ResNet50 ---")
    # Use pretrained=False here to force building from scratch with MyMaskRCNN
    model_resnet50 = maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=True, num_classes=10)
    model_resnet50.to(device)
    model_resnet50.eval()
    with torch.no_grad():
        out = model_resnet50(dummy_input)
    print("ResNet50 OK")

    # Test Swin-T
    print("\n--- Testing Swin-T ---")
    model_swin = maskrcnn_swin_t_fpn(pretrained_backbone=True, num_classes=10)
    model_swin.to(device)
    model_swin.eval()
    with torch.no_grad():
        out = model_swin(dummy_input)
    print("Swin-T OK")

    # Test ResNeXt101
    print("\n--- Testing ResNeXt101 ---")
    model_resnext = maskrcnn_resnext101_fpn(pretrained_backbone=True, num_classes=10)
    model_resnext.to(device)
    model_resnext.eval()
    with torch.no_grad():
        out = model_resnext(dummy_input)
    print("ResNeXt101 OK")

    # Test Swin + ResNet50
    print("\n--- Testing Swin + ResNet50 ---")
    model_swin_resnet = maskrcnn_swin_resnet50_fpn(pretrained_swin=True, pretrained_resnet=True, num_classes=10)
    model_swin_resnet.to(device)
    model_swin_resnet.eval()
    with torch.no_grad():
        out = model_swin_resnet(dummy_input)
    print("Swin + ResNet50 OK")

    # Test Swin + ResNeXt101
    print("\n--- Testing Swin + ResNeXt101 ---")
    model_swin_resnext = maskrcnn_swin_resnext101_fpn(pretrained_swin=True, pretrained_resnext=True, num_classes=10)
    model_swin_resnext.to(device)
    model_swin_resnext.eval()
    with torch.no_grad():
        out = model_swin_resnext(dummy_input)
    print("Swin + ResNeXt101 OK")