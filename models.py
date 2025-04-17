import torch
import torch.nn as nn
from torch import Tensor
import torchvision
from torchvision.models import (
    resnet50, ResNet50_Weights,
    swin_t, Swin_T_Weights,
    resnext101_32x8d, ResNeXt101_32x8d_Weights
)
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.models.detection.backbone_utils import (
    _resnet_fpn_extractor, _validate_trainable_layers, BackboneWithFPN
)
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops import MultiScaleRoIAlign, misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool, ExtraFPNBlock

from collections import OrderedDict
import warnings
from typing import List, Dict, Tuple, Optional, Any, Callable

# --- Constants ---
DEFAULT_FPN_OUT_CHANNELS = 256
DEFAULT_NUM_CLASSES = 91  # Default for COCO

# --- Custom MaskRCNN Class (Keep as is if needed) ---
class MyMaskRCNN(MaskRCNN):
    """
    Your custom MaskRCNN subclass (inherits all functionality).
    You can override methods here if needed in the future.
    """
    # (Your existing MyMaskRCNN code here - no changes needed from the snippet provided)
    # ... (constructor and forward method as provided in your code) ...
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

        out_channels = backbone.out_channels

        if mask_roi_pool is None:
            mask_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)

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
        # Add the features print statement if needed for debugging
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
        # Debug print - remove if not needed
        # print("--- Backbone Features ---")
        # for k, v in features.items():
        #     print(k, v.shape)
        # print("-----------------------")
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


# --- Custom Swin FPN Backbone ---
class CustomSwinFPN(nn.Module):
    """
    A custom module combining a Swin feature extractor with an FPN,
    handling the channel permutation.
    """
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


# --- Combined Swin + ResNet-Like FPN Backbone ---
class CombinedSwinResNetLikeFPN(nn.Module):
    """
    Combines features from Swin Transformer and a ResNet-like backbone,
    projects them, and feeds them into an FPN.
    """
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
                # print(f"Warning: Resizing Swin features at stage {i} from {swin_f_permuted.shape[2:]} to {res_f.shape[2:]}")
                swin_f_permuted = nn.functional.interpolate(swin_f_permuted, size=res_f.shape[2:], mode='bilinear', align_corners=False)
            combined_f = torch.cat([swin_f_permuted, res_f], dim=1)
            projected_f = self.projection_convs[i](combined_f)
            fpn_input[fpn_key] = projected_f
        fpn_output = self.fpn(fpn_input)
        return fpn_output


# --- Factory Function for ResNet50 Backbone ---
def maskrcnn_resnet50_fpn(
    *,
    num_classes: Optional[int] = None,
    pretrained: bool = True, # Whether to use pretrained MaskRCNN weights (if available)
    pretrained_backbone: bool = True, # Whether to use pretrained ResNet weights
    trainable_backbone_layers: Optional[int] = None, # 0-5, how many ResNet layers to train (higher means more)
    progress: bool = True,
    **kwargs: Any,
) -> MyMaskRCNN:
    """
    Constructs a Mask R-CNN model with a ResNet-50-FPN backbone.

    Args:
        num_classes (Optional[int]): Number of classes (including background). Defaults to 91 (COCO).
        pretrained (bool): If True, loads weights pretrained on COCO from torchvision (if available for this exact config).
        pretrained_backbone (bool): If True, loads ImageNet weights for the ResNet backbone.
        trainable_backbone_layers (Optional[int]): Number of trainable (unfrozen) layers starting from the top.
                                                  Valid values are 0 to 5. If None, defaults based on `pretrained`.
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    if pretrained:
         # Try loading torchvision's pretrained model directly if args match
         if trainable_backbone_layers is None: trainable_backbone_layers = 3 # Default for pretrained COCO model
         if num_classes is None: num_classes = DEFAULT_NUM_CLASSES

         if num_classes == DEFAULT_NUM_CLASSES and trainable_backbone_layers == 3:
             print("Loading pre-trained MaskRCNN ResNet50 FPN from torchvision...")
             # Note: Using torchvision's direct function if possible
             weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
             weights_backbone = None # Loaded by the main weights
             model = torchvision.models.detection.maskrcnn_resnet50_fpn(
                 weights=weights, progress=progress, num_classes=num_classes,
                 trainable_backbone_layers=trainable_backbone_layers, **kwargs
             )
             # If you need MyMaskRCNN specifically, you might need to rebuild it
             # and load the state_dict carefully, or just use the torchvision one.
             # For simplicity here, returning the torchvision model if pretrained matches.
             # If you *must* use MyMaskRCNN, set pretrained=False and load state_dict later.
             return model # Return torchvision model directly
         else:
             print("Warning: Pretrained weights requested but num_classes or trainable_layers differ from torchvision defaults. Building model from scratch with pretrained backbone.")
             pretrained = False # Fallback to building from scratch

    # --- Build model from scratch ---
    weights_backbone = ResNet50_Weights.DEFAULT if pretrained_backbone else None
    trainable_backbone_layers = _validate_trainable_layers(weights_backbone is not None, trainable_backbone_layers, 5, 3)
    norm_layer = misc_nn_ops.FrozenBatchNorm2d if weights_backbone is not None else nn.BatchNorm2d

    print(f"Building ResNet50 backbone (trainable layers: {trainable_backbone_layers})...")
    backbone_model = resnet50(weights=weights_backbone, progress=progress, norm_layer=norm_layer)
    # Use torchvision's helper to create ResNet+FPN
    backbone = _resnet_fpn_extractor(backbone_model, trainable_backbone_layers, norm_layer=norm_layer)

    print("Creating MyMaskRCNN with ResNet50 FPN backbone...")
    model = MyMaskRCNN(backbone, num_classes=num_classes, **kwargs)

    return model


# --- Factory Function for Swin-T Backbone ---
def maskrcnn_swin_t_fpn(
    *,
    num_classes: Optional[int] = None,
    pretrained_backbone: bool = True, # Whether to use pretrained Swin weights
    fpn_out_channels: int = DEFAULT_FPN_OUT_CHANNELS,
    progress: bool = True, # Progress for weight download
    **kwargs: Any,
) -> MyMaskRCNN:
    """
    Constructs a Mask R-CNN model with a Swin Transformer (Swin-T) FPN backbone.
    """
    print("Creating Swin-T backbone...")
    weights_swin = Swin_T_Weights.DEFAULT if pretrained_backbone else None
    # Load on CPU initially
    swin_model = swin_t(weights=weights_swin, progress=progress).cpu()

    return_nodes_for_extractor = {
        'features.1.1.add_1': 'feat0', # Stride 4
        'features.3.1.add_1': 'feat1', # Stride 8
        'features.5.5.add_1': 'feat2', # Stride 16
        'features.7.1.add_1': 'feat3', # Stride 32
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
        extra_blocks=LastLevelMaxPool() # Add P6 layer
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
    weights_backbone = ResNeXt101_32x8d_Weights.DEFAULT if pretrained_backbone else None
    trainable_backbone_layers = _validate_trainable_layers(weights_backbone is not None, trainable_backbone_layers, 5, 3)
    # norm_layer = misc_nn_ops.FrozenBatchNorm2d if weights_backbone is not None else nn.BatchNorm2d
    # Note: ResNeXt in torchvision might not expose norm_layer in constructor easily, handle freezing manually.

    print(f"Building ResNeXt101 backbone (trainable layers: {trainable_backbone_layers})...")
    backbone_model = resnext101_32x8d(weights=weights_backbone, progress=progress)

    # Freeze layers manually if FrozenBatchNorm not applicable directly
    if weights_backbone is not None and trainable_backbone_layers < 5:
        layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_backbone_layers]
        if trainable_backbone_layers == 5: layers_to_train.append("bn1") # Should match ResNet freezing logic
        for name, parameter in backbone_model.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                 parameter.requires_grad_(False)

    # FPN setup (similar to ResNet helper)
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
    projection_channels: int = DEFAULT_FPN_OUT_CHANNELS, # Project concatenated features TO this dim
    fpn_out_channels: int = DEFAULT_FPN_OUT_CHANNELS,    # Final FPN output channels
    progress: bool = True,
    **kwargs: Any,
) -> MyMaskRCNN:
    """
    Constructs Mask R-CNN with a combined Swin-T + ResNet50 FPN backbone.
    """
    print("Loading ResNet50 base for combined model...")
    resnet_weights = ResNet50_Weights.DEFAULT if pretrained_resnet else None
    resnet_base = resnet50(weights=resnet_weights, progress=progress)
    resnet_channels = [256, 512, 1024, 2048] # ResNet50 specific

    print("Instantiating CombinedSwinResNetLikeFPN (with ResNet50)...")
    combined_backbone = CombinedSwinResNetLikeFPN(
        resnet_like_base=resnet_base,
        resnet_like_channels=resnet_channels,
        pretrained_swin=pretrained_swin,
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
    resnext_weights = ResNeXt101_32x8d_Weights.DEFAULT if pretrained_resnext else None
    resnext_base = resnext101_32x8d(weights=resnext_weights, progress=progress)
    resnext_channels = [256, 512, 1024, 2048] # ResNeXt101 specific

    print("Instantiating CombinedSwinResNetLikeFPN (with ResNeXt101)...")
    combined_backbone = CombinedSwinResNetLikeFPN(
        resnet_like_base=resnext_base,
        resnet_like_channels=resnext_channels,
        pretrained_swin=pretrained_swin,
        projection_channels=projection_channels,
        fpn_out_channels=fpn_out_channels,
        extra_blocks=LastLevelMaxPool()
    )

    print("Creating MyMaskRCNN with combined Swin+ResNeXt101 backbone...")
    model = MyMaskRCNN(combined_backbone, num_classes=num_classes, **kwargs)
    return model


# --- Example Usage ---
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dummy_input = torch.randn(1, 3, 512, 512).to(device)

    # Test ResNet50
    print("\n--- Testing ResNet50 ---")
    model_resnet50 = maskrcnn_resnet50_fpn(pretrained_backbone=True, num_classes=10)
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