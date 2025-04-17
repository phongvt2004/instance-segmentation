torchrun --nproc_per_node=4 train.py --output-dir /workspace/output_resnext \
    --dataset coco --model maskrcnn_resnext101_fpn --epochs 12 --data-path /workspace/coco2017_subset \
    --lr 0.01 --world-size 4 \
    --lr-steps 4 8 --aspect-ratio-group-factor 3 --weights-backbone ResNet50_Weights.IMAGENET1K_V1 --use-v2 -b 8 --workers 10 --opt sgd --print-freq 400

torchrun --nproc_per_node=4 train.py --output-dir /workspace/output_swin_resnext \
    --dataset coco --model maskrcnn_swin_resnext101_fpn --epochs 12 --data-path /workspace/coco2017_subset \
    --lr 0.01 --world-size 4 \
    --lr-steps 4 8 --aspect-ratio-group-factor 3 --weights-backbone ResNet50_Weights.IMAGENET1K_V1 --use-v2 -b 2 --workers 10 --opt sgd --print-freq 400