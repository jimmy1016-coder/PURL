#!/bin/bash
# SupCon ResNet34 - voxceleb1_E, voxceleb1_O 테스트
# Config: supcon_resnet34.yaml
# Checkpoints: epoch 199, epoch 229

cd "$(dirname "$0")"  # chns 디렉터리에서 실행

CONFIG="configs/supcon_resnet34.yaml"
CKPT_DIR="outputs/supcon_resnet34/checkpoints"
DATASET_DIR="/home/sooyoung/interspeech/dataset"

# 테스트할 checkpoint (epoch)
EPOCHS=(199 229)

# 테스트할 dataset (voxceleb1_0 -> voxceleb1_O 사용)
DATASETS=("voxceleb1_E.txt" "voxceleb1_O.txt")

for trials_file in "${DATASETS[@]}"; do
  trials_path="${DATASET_DIR}/${trials_file}"
  if [ ! -f "$trials_path" ]; then
    echo "[SKIP] Dataset not found: $trials_path"
    continue
  fi

  for epoch in "${EPOCHS[@]}"; do
    ckpt_file=$(ls "$CKPT_DIR"/epoch=${epoch}_*.ckpt 2>/dev/null | head -1)
    if [ -z "$ckpt_file" ]; then
      echo "[SKIP] Checkpoint for epoch $epoch not found"
      continue
    fi
    ckpt_name=$(basename "$ckpt_file" .ckpt)

    echo "[START] $trials_file | $ckpt_name"
    python run_test.py \
      --config "$CONFIG" \
      --ckpt_name "$ckpt_name" \
      --data.init_args.test_config.trials_file_path="$trials_path" \
      --no_wandb
    echo "[DONE] $trials_file | $ckpt_name"
    echo ""
  done
done

echo "[DONE] All tests completed"
