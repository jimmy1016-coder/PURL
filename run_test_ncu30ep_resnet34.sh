#!/bin/bash
# NCU 30ep ResNet34 - Voxceleb1-O, Voxceleb1-E, VoxSRC2023 테스트
# Config: ncu_30ep_resnet34.yaml
# Checkpoint: epoch 229 only

cd "$(dirname "$0")"

CONFIG="configs/ncu_30ep_resnet34.yaml"
CKPT_DIR="outputs/ncu-Softloss-alpha0.5-30ep-resnet34/checkpoints"
DATASET_DIR="/home/sooyoung/interspeech/dataset"
DEV1_DIR="/home/sooyoung/interspeech/dev1"
VOXSRC_DIR="/home/sooyoung/interspeech/VoxSRC2023_test"

EPOCH=229

# epoch 229 checkpoint 확인
ckpt_file=$(ls "$CKPT_DIR"/epoch=${EPOCH}_*.ckpt 2>/dev/null | head -1)
if [ -z "$ckpt_file" ]; then
  echo "[ERROR] Checkpoint for epoch $EPOCH not found in $CKPT_DIR"
  exit 1
fi
ckpt_name=$(basename "$ckpt_file" .ckpt)

echo "=== NCU 30ep ResNet34 | $ckpt_name ==="
echo ""

# 1. Voxceleb1-O
if [ -f "${DATASET_DIR}/voxceleb1_O.txt" ]; then
  echo "[START] Voxceleb1-O | $ckpt_name"
  python run_test.py \
    --config "$CONFIG" \
    --ckpt_name "$ckpt_name" \
    --data.init_args.test_config.trials_file_path="${DATASET_DIR}/voxceleb1_O.txt" \
    --data.init_args.test_config.data_dir="$DEV1_DIR" \
    --no_wandb
  echo "[DONE] Voxceleb1-O | $ckpt_name"
  echo ""
else
  echo "[SKIP] Voxceleb1-O: ${DATASET_DIR}/voxceleb1_O.txt not found"
fi

# 2. Voxceleb1-E
if [ -f "${DATASET_DIR}/voxceleb1_E.txt" ]; then
  echo "[START] Voxceleb1-E | $ckpt_name"
  python run_test.py \
    --config "$CONFIG" \
    --ckpt_name "$ckpt_name" \
    --data.init_args.test_config.trials_file_path="${DATASET_DIR}/voxceleb1_E.txt" \
    --data.init_args.test_config.data_dir="$DEV1_DIR" \
    --no_wandb
  echo "[DONE] Voxceleb1-E | $ckpt_name"
  echo ""
else
  echo "[SKIP] Voxceleb1-E: ${DATASET_DIR}/voxceleb1_E.txt not found"
fi

# 3. VoxSRC2023 (data_dir, trials_file_path 모두 변경)
VOXSRC_TRIALS="${DATASET_DIR}/voxsrc2023_final_fixed.txt"
if [ -f "$VOXSRC_TRIALS" ]; then
  if [ -d "$VOXSRC_DIR" ]; then
    echo "[START] VoxSRC2023 | $ckpt_name"
    python run_test.py \
      --config "$CONFIG" \
      --ckpt_name "$ckpt_name" \
      --data.init_args.test_config.trials_file_path="$VOXSRC_TRIALS" \
      --data.init_args.test_config.data_dir="$VOXSRC_DIR" \
      --no_wandb
    echo "[DONE] VoxSRC2023 | $ckpt_name"
    echo ""
  else
    echo "[SKIP] VoxSRC2023: data_dir $VOXSRC_DIR not found"
  fi
else
  echo "[SKIP] VoxSRC2023: trials file $VOXSRC_TRIALS not found"
fi

echo "[DONE] All tests completed"
