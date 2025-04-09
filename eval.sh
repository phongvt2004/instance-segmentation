!torchrun --nproc_per_node=2 train.py --output-dir /kaggle/working/output \
    --dataset coco --model maskrcnn_resnet50_fpn --epochs 20 --data-path /kaggle/input/coco-2017-subset/coco2017_subset \
    --lr 0.005 --world-size 2 --test-only\
    --lr-steps 8 16 --aspect-ratio-group-factor 3 --weights MaskRCNN_ResNet50_FPN_Weights.DEFAULT --weights-backbone ResNet50_Weights.IMAGENET1K_V1 --use-v2 -b 8 --workers 2 --opt sgd --print-freq 200