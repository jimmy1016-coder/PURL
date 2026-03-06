#!/bin/bash
# VoxSRC2023 test set 평가 (voxsrc2023_final_fixed.txt, VoxSRC2023_test)

CKPT_DIR="outputs/supcon-seedfix/checkpoints"
CONFIG="configs/supcon_voxsrc2023.yaml"
MAX_PARALLEL=10
pids=()

for epoch in {180..199}; do
  ckpt_file=$(ls "$CKPT_DIR"/epoch=${epoch}_*.ckpt 2>/dev/null | head -1)
  if [ -z "$ckpt_file" ]; then
    echo "[SKIP] Checkpoint for epoch $epoch not found"
    continue
  fi
  ckpt_name=$(basename "$ckpt_file" .ckpt)

  echo "[START] VoxSRC2023 Testing: $ckpt_name"
  python run_test.py --config "$CONFIG" --ckpt_name "$ckpt_name" --no_wandb &
  pid=$!
  pids+=($pid)

  if [ ${#pids[@]} -ge $MAX_PARALLEL ]; then
    wait "${pids[@]}"
    pids=()
    echo "[DONE] Batch completed"
  fi
done

if [ ${#pids[@]} -gt 0 ]; then
  wait "${pids[@]}"
  echo "[DONE] All finished"
fi
