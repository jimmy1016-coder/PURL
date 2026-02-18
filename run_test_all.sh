#!/bin/bash

CKPT_DIR="outputs/ncu-Softloss-alpha0.5-1ep/checkpoints"

for ckpt_file in "$CKPT_DIR"/epoch=*.ckpt; do
    ckpt_name=$(basename "$ckpt_file" .ckpt)
    echo "=========================================="
    echo "Testing: $ckpt_name"
    echo "=========================================="
    python run_test.py --config configs/ncu.yaml --ckpt_name "$ckpt_name"
done
