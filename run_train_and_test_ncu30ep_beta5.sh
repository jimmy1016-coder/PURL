#!/bin/bash

set -euo pipefail

CONFIG="configs/ncu_30ep.yaml"
CKPT_DIR="outputs/ncu-Softloss-beta0.3-exp-30ep/checkpoints"
MAX_PARALLEL=10
pids=()

echo "[TRAIN] Start NCU training with beta=5.0"
python run_train.py --config "$CONFIG" --ckpt_path "/home/sooyoung/interspeech/chns/outputs/supcon/checkpoints/epoch=199_step=333400.ckpt"
echo "[TRAIN] Finished"

echo "[TEST] Evaluate checkpoints epoch 200..229"
for epoch in {200..229}; do
  ckpt_file=$(ls "$CKPT_DIR"/epoch=${epoch}_*.ckpt 2>/dev/null | head -1)
  if [ -z "$ckpt_file" ]; then
    echo "[SKIP] Checkpoint for epoch $epoch not found"
    continue
  fi

  ckpt_name=$(basename "$ckpt_file" .ckpt)
  echo "[START] Testing: $ckpt_name"
  python run_test.py --config "$CONFIG" --ckpt_name "$ckpt_name" --no_wandb &
  pids+=($!)

  if [ ${#pids[@]} -ge $MAX_PARALLEL ]; then
    wait "${pids[@]}"
    pids=()
    echo "[DONE] Batch completed"
  fi
done

if [ ${#pids[@]} -gt 0 ]; then
  wait "${pids[@]}"
fi

echo "[DONE] All training/testing finished"
