torchrun --nproc_per_node 3 \
    --rdzv_endpoint=$HOSTE_NODE_ADDR \
    -m main \
    --imagenet-val '/nas1/datasets/ImageNet1k/val' \
    --logs ../eval/ \
    --pretrained '/home/kecheng/ECCV2024/epoch_32.pt' \
    --batch-size=16 \
    --model "ViT-B-16" \
