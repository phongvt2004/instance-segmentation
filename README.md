# Instance Segmentation with Mask R-CNN
 
This folder contains reference training scripts for object detection and instance segmentation.
They serve as a log of how to train specific models, to provide baseline
training and evaluation scripts to quickly bootstrap research.

To execute the example commands below you must install the following:

```
cython
pycocotools
matplotlib
```

You must modify the following flags:
`--model=maskrcnn_resnet50_fpn/maskrcnn_resnext101_fpn/maskrcnn_swin_t_fpn/maskrcnn_swin_resnet50_fpn/maskrcnn_swin_resnext101_fpn`
`--data-path=/path/to/coco/dataset`

`--nproc_per_node=<number_of_gpus_available>`

### Resources
| Link            |
| ------------------ |
| [Our COCO subset](https://www.kaggle.com/datasets/phongvt2k4/coco-2017-subset) |
| [Checkpoints](https://www.kaggle.com/models/phongvt2k4/checkpoints-maskrcnn) |

### Training
```
./train.sh
```

### Evaluation
```
./eval.sh
```

### References
[Torchvision references](https://github.com/pytorch/vision/tree/main/references)