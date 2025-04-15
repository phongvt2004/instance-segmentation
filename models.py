import numpy as np
import torchvision
from torchvision.models import ResNet50_Weights, resnet50
from huggingface_hub import login
import os
from torchvision.models.detection import MaskRCNN, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision.ops import MultiScaleRoIAlign, misc as misc_nn_ops
import torch
import warnings
from collections import OrderedDict
from typing import List, Dict, Tuple, Optional, Any
from torch import Tensor, nn
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor, _validate_trainable_layers, \
    BackboneWithFPN
from transformers import ViTModel, ViTConfig, AutoProcessor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from dotenv import load_dotenv

load_dotenv()
class MyBackbone(nn.Module):
    def init(self, cnn_backbone):
        super(MyBackbone, self).init()
        self.cnn_backbone = cnn_backbone

    def forward(self, x):
        return self.backbone(x)

    @property
    def out_channels(self):
        return self.backbone.out_channels
class MyMaskRCNN(MaskRCNN):
    """
    Implements Mask R-CNN.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending on if it is in training or evaluation mode.

    During training, the model expects both the input tensors and targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN, and the mask loss.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction
        - masks (UInt8Tensor[N, 1, H, W]): the predicted masks for each instance, in 0-1 range. In order to
          obtain the final segmentation masks, the soft masks can be thresholded, generally
          with a value of 0.5 (mask >= 0.5)

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain an out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
        min_size (int): Images are rescaled before feeding them to the backbone:
            we attempt to preserve the aspect ratio and scale the shorter edge
            to ``min_size``. If the resulting longer edge exceeds ``max_size``,
            then downscale so that the longer edge does not exceed ``max_size``.
            This may result in the shorter edge beeing lower than ``min_size``.
        max_size (int): See ``min_size``.
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        rpn_score_thresh (float): only return proposals with an objectness score greater than rpn_score_thresh
        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
        box_head (nn.Module): module that takes the cropped feature maps as input
        box_predictor (nn.Module): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
        mask_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
             the locations indicated by the bounding boxes, which will be used for the mask head.
        mask_head (nn.Module): module that takes the cropped feature maps as input
        mask_predictor (nn.Module): module that takes the output of the mask_head and returns the
            segmentation mask logits

    Example::

        >>> import torch
        >>> import torchvision
        >>> from torchvision.models.detection import MaskRCNN
        >>> from torchvision.models.detection.anchor_utils import AnchorGenerator
        >>>
        >>> # load a pre-trained model for classification and return
        >>> # only the features
        >>> backbone = torchvision.models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
        >>> # MaskRCNN needs to know the number of
        >>> # output channels in a backbone. For mobilenet_v2, it's 1280
        >>> # so we need to add it here,
        >>> backbone.out_channels = 1280
        >>>
        >>> # let's make the RPN generate 5 x 3 anchors per spatial
        >>> # location, with 5 different sizes and 3 different aspect
        >>> # ratios. We have a Tuple[Tuple[int]] because each feature
        >>> # map could potentially have different sizes and
        >>> # aspect ratios
        >>> anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
        >>>                                    aspect_ratios=((0.5, 1.0, 2.0),))
        >>>
        >>> # let's define what are the feature maps that we will
        >>> # use to perform the region of interest cropping, as well as
        >>> # the size of the crop after rescaling.
        >>> # if your backbone returns a Tensor, featmap_names is expected to
        >>> # be ['0']. More generally, the backbone should return an
        >>> # OrderedDict[Tensor], and in featmap_names you can choose which
        >>> # feature maps to use.
        >>> roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
        >>>                                                 output_size=7,
        >>>                                                 sampling_ratio=2)
        >>>
        >>> mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
        >>>                                                      output_size=14,
        >>>                                                      sampling_ratio=2)
        >>> # put the pieces together inside a MaskRCNN model
        >>> model = MaskRCNN(backbone,
        >>>                  num_classes=2,
        >>>                  rpn_anchor_generator=anchor_generator,
        >>>                  box_roi_pool=roi_pooler,
        >>>                  mask_roi_pool=mask_roi_pooler)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
    """

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

        if num_classes is not None:
            if mask_predictor is not None:
                raise ValueError("num_classes should be None when mask_predictor is specified")

        out_channels = backbone.out_channels

        if mask_roi_pool is None:
            mask_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)

        if mask_head is None:
            mask_layers = (256, 256, 256, 256)
            mask_dilation = 1
            mask_head = MaskRCNNHeads(out_channels, mask_layers, mask_dilation)

        if mask_predictor is None:
            mask_predictor_in_channels = 256  # == mask_layers[-1]
            mask_dim_reduced = 256
            mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels, mask_dim_reduced, num_classes)

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
            **kwargs,
        )

        self.roi_heads.mask_roi_pool = mask_roi_pool
        self.roi_heads.mask_head = mask_head
        self.roi_heads.mask_predictor = mask_predictor

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """

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

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        features = self.backbone(images.tensors)
        for k, v in features.items():
            print(k, v.shape)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)
def my_maskrcnn_resnet50_fpn(
    *,
    weights: Optional[str] = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs: Any,
    ) -> MyMaskRCNN:

    weights_backbone = ResNet50_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = 91
    elif num_classes is None:
        num_classes = 91

    is_trained = weights is not None or weights_backbone is not None
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)
    norm_layer = misc_nn_ops.FrozenBatchNorm2d if is_trained else nn.BatchNorm2d

    backbone = resnet50(weights=weights_backbone, progress=progress, norm_layer=norm_layer)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
    model = MyMaskRCNN(backbone, num_classes=num_classes, **kwargs)

    if weights is not None:
        model.load_state_dict(weights)

    return model

class CustomSwinFPN(nn.Module):
    """
    A custom module that combines a feature extractor (like one from
    create_feature_extractor) with a Feature Pyramid Network (FPN).

    Handles the potential mismatch in dictionary keys between the
    feature extractor output and the FPN input.
    """
    def __init__(self, body, return_layer_keys, fpn_in_channels_list, fpn_out_channels, extra_blocks=None):
        """
        Args:
            body (nn.Module): The feature extractor module (e.g., from create_feature_extractor).
                              Expected to return a dict[str, Tensor].
            return_layer_keys (List[str]): The keys that 'body' returns in its output dict,
                                           in the order corresponding to strides (e.g., ['feat0', 'feat1', 'feat2', 'feat3']).
            fpn_in_channels_list (List[int]): List of input channels for each FPN level, matching the channels
                                              of the tensors returned by 'body' for the keys in 'return_layer_keys'.
            fpn_out_channels (int): The number of output channels for each FPN feature map.
            extra_blocks (nn.Module, optional): An optional extra block to apply after the last FPN level
                                                (e.g., LastLevelMaxPool for P6). Defaults to None.
        """
        super().__init__()
        self.body = body # The model created by create_feature_extractor
        self.return_layer_keys = return_layer_keys # e.g., ['feat0', 'feat1', 'feat2', 'feat3']
        # Map the body output keys to the keys FPN expects ('0', '1', '2', '3')
        self.fpn_map = {key: str(i) for i, key in enumerate(return_layer_keys)}

        print(f"CustomSwinFPN: Mapping body keys {list(self.fpn_map.keys())} to FPN keys {list(self.fpn_map.values())}")

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=fpn_in_channels_list,
            out_channels=fpn_out_channels,
            extra_blocks=extra_blocks,
        )
        # IMPORTANT: MaskRCNN and FasterRCNN expect the backbone to have this attribute
        self.out_channels = fpn_out_channels

    def forward(self, x):
        # 1. Get features from the body (feature extractor)
        features = self.body(x) # Expected output: {'feat0': tensor, 'feat1': tensor, ...}

        # 2. Rename keys for FPN input
        fpn_input = OrderedDict()
        for body_key, fpn_key in self.fpn_map.items():
            if body_key in features:
                fpn_input[fpn_key] = features[body_key]
            else:
                # This indicates a problem with create_feature_extractor or key definition
                raise KeyError(f"Expected key '{body_key}' not found in feature extractor output. Found keys: {features.keys()}")

        # 3. Pass renamed features to FPN
        fpn_output = self.fpn(fpn_input) # FPN expects {'0': tensor, '1': tensor, ...}
        return fpn_output

def my_maskrcnn_swin_t_fpn(
        *,
        weights: Optional[str] = None,
        progress: bool = True,
        num_classes: Optional[int] = None,
        weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
        trainable_backbone_layers: Optional[int] = None,
        **kwargs: Any,
) -> MyMaskRCNN:
    NUM_CLASSES = 91
    PRETRAINED_SWIN = True
    FPN_OUT_CHANNELS = 256

    # --- 1. Load Swin Model ---
    weights = torchvision.models.Swin_T_Weights.DEFAULT if PRETRAINED_SWIN else None
    swin_model = torchvision.models.swin_t(weights=weights).cpu()  # Load on CPU

    # --- 2. Define Nodes and Output Keys for Extractor ---
    # Use the correct internal node names as keys, and assign simple output keys as values
    return_nodes_for_extractor = {
        'features.1.1.add_1': 'feat0',  # Stage 1 out
        'features.3.1.add_1': 'feat1',  # Stage 2 out
        'features.5.5.add_1': 'feat2',  # Stage 3 out
        'features.7.1.add_1': 'feat3',  # Stage 4 out
    }
    # Store the output keys in order, corresponding to increasing stride
    extractor_output_keys = ['feat0', 'feat1', 'feat2', 'feat3']
    print(f"Nodes for feature extractor: {return_nodes_for_extractor}")
    print(f"Extractor output keys (ordered): {extractor_output_keys}")

    # --- 3. Create the Feature Extractor Body ---
    print("Creating feature extractor body...")
    try:
        body = create_feature_extractor(swin_model, return_nodes=return_nodes_for_extractor)
        print("Feature extractor body created successfully.")
    except Exception as e:
        import traceback
        print(f"\nError creating feature extractor body: {e}")
        print(traceback.format_exc())
        exit()

    # --- 4. Define Input Channels for FPN ---
    # Corresponds channel count for 'feat0', 'feat1', 'feat2', 'feat3' outputs
    fpn_in_channels_list = [96, 192, 384, 768]  # For Swin-T

    # --- 5. Create the Custom Backbone + FPN ---
    print("Creating CustomSwinFPN...")
    try:
        # Create an instance of our custom wrapper
        custom_backbone = CustomSwinFPN(
            body=body,
            return_layer_keys=extractor_output_keys,  # Pass the keys body will return
            fpn_in_channels_list=fpn_in_channels_list,
            fpn_out_channels=FPN_OUT_CHANNELS,
            # Optional: Add extra_blocks for P6 if needed for your MaskRCNN config
            # extra_blocks=LastLevelMaxPool()
        )
        print("CustomSwinFPN created successfully.")
    except Exception as e:
        import traceback
        print(f"\nError creating CustomSwinFPN: {e}")
        print(traceback.format_exc())
        exit()

    # --- 6. Create the Mask R-CNN Model ---
    print("Creating MaskRCNN model...")
    try:
        # Pass our custom backbone to MaskRCNN
        model = MaskRCNN(
            custom_backbone,  # Use our custom wrapper
            num_classes=NUM_CLASSES,
        )
        print("MaskRCNN model created successfully.")
    except Exception as e:
        import traceback
        print(f"\nError creating MaskRCNN: {e}")
        print(traceback.format_exc())
        print("Ensure custom_backbone has the '.out_channels' attribute set correctly.")
        exit()

    # if weights is not None:
    #     model.load_state_dict(weights)

    return model

if __name__ == "__main__":
    # Example usage
    login(token=os.getenv("HUGGINGFACE_TOKEN"))
    model = my_maskrcnn_swin_t_fpn()
    model.eval()
    print(model)
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)
    print(predictions)