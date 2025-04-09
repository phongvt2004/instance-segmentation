torchrun --nproc_per_node=2 train.py --output-dir output \
    --dataset coco --model maskrcnn_resnet50_fpn --epochs 20 --data-path /workspace/coco2017_subset \
    --lr 0.005 --world-size 2 \
    --lr-steps 8 16 --aspect-ratio-group-factor 3 --weights-backbone ResNet50_Weights.IMAGENET1K_V1 --use-v2 -b 16 --workers 12 --opt sgd --print-freq 200