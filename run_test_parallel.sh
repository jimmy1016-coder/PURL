#!/bin/bash

CKPT_DIR="outputs/ncu-Hardloss-30ep/checkpoints"
CONFIG="configs/ncu_30ep.yaml"
MAX_PARALLEL=10
pids=()

# epoch 210 ~ 223
for epoch in {220..229}; do
  ckpt_file=$(ls "$CKPT_DIR"/epoch=${epoch}_*.ckpt 2>/dev/null | head -1)
  if [ -z "$ckpt_file" ]; then
    echo "[SKIP] Checkpoint for epoch $epoch not found"
    continue
  fi
  ckpt_name=$(basename "$ckpt_file" .ckpt)

  echo "[START] Testing: $ckpt_name"
  python run_test.py --config "$CONFIG" --ckpt_name "$ckpt_name" --no_wandb &
  pid=$!
  pids+=($pid)

  # 4개 꽉 차면 대기
  if [ ${#pids[@]} -ge $MAX_PARALLEL ]; then
    wait "${pids[@]}"
    pids=()
    echo "[DONE] Batch completed"
  fi
done

# 남은 작업 대기
if [ ${#pids[@]} -gt 0 ]; then
  wait "${pids[@]}"
  echo "[DONE] All finished"
fi
