torchrun --nproc_per_node=4 train.py --output-dir /workspace/output_swin_resnext \
    --dataset coco --model maskrcnn_swin_resnext101_fpn --epochs 12 --resume /workspace/output_swin_resnext/checkpoint.pth --data-path /workspace/coco2017_subset \
    --lr 0.01 --world-size 4 \
    --lr-steps 4 8 --aspect-ratio-group-factor 3  --use-v2 -b 2 --workers 10 --opt sgd --print-freq 400