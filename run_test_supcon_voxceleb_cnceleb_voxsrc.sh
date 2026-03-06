#!/bin/bash
# SupCon (ECAPA) - VoxCeleb1-H, CN-Celeb(E), VoxSRC2023 테스트
# Config: supcon.yaml
# Checkpoint: epoch 229 only

cd "$(dirname "$0")"

CONFIG="configs/supcon.yaml"
CKPT_DIR="outputs/supcon/checkpoints"
DATASET_DIR="/home/sooyoung/interspeech/dataset"
DEV1_DIR="/home/sooyoung/interspeech/dev1"
VOXSRC_DIR="/home/sooyoung/interspeech/VoxSRC2023_test"
CNCELEB_DIR="/home/sooyoung/interspeech/CN-Celeb_flac/eval"

EPOCH=229

# epoch 229 checkpoint 확인
ckpt_file=$(ls "$CKPT_DIR"/epoch=${EPOCH}_*.ckpt 2>/dev/null | head -1)
if [ -z "$ckpt_file" ]; then
  echo "[ERROR] Checkpoint for epoch $EPOCH not found in $CKPT_DIR"
  exit 1
fi
ckpt_name=$(basename "$ckpt_file" .ckpt)

echo "=== SupCon (ECAPA) | $ckpt_name ==="
echo ""

# 1. VoxCeleb1-H (data_dir=dev1)
if [ -f "${DATASET_DIR}/voxceleb1_H.txt" ]; then
  echo "[START] VoxCeleb1-H | $ckpt_name"
  python run_test.py \
    --config "$CONFIG" \
    --ckpt_name "$ckpt_name" \
    --data.init_args.test_config.trials_file_path="${DATASET_DIR}/voxceleb1_H.txt" \
    --data.init_args.test_config.data_dir="$DEV1_DIR" \
    --no_wandb
  echo "[DONE] VoxCeleb1-H | $ckpt_name"
  echo ""
else
  echo "[SKIP] VoxCeleb1-H: ${DATASET_DIR}/voxceleb1_H.txt not found"
fi

# 2. CN-Celeb(E) - data_dir, trials_file_path 모두 VoxCeleb와 다름
CNCELEB_TRIALS="${DATASET_DIR}/cnceleb_eval_pairs.txt"
if [ ! -f "$CNCELEB_TRIALS" ]; then
  CNCELEB_TRIALS="/home/sooyoung/interspeech/chns/resources/cnceleb_eval_pairs.txt"
fi
if [ -f "$CNCELEB_TRIALS" ] && [ -d "$CNCELEB_DIR" ]; then
  echo "[START] CN-Celeb(E) | $ckpt_name"
  python run_test.py \
    --config "$CONFIG" \
    --ckpt_name "$ckpt_name" \
    --data.init_args.test_config.trials_file_path="$CNCELEB_TRIALS" \
    --data.init_args.test_config.data_dir="$CNCELEB_DIR" \
    --no_wandb
  echo "[DONE] CN-Celeb(E) | $ckpt_name"
  echo ""
else
  echo "[SKIP] CN-Celeb(E): trials=$CNCELEB_TRIALS data_dir=$CNCELEB_DIR"
fi

# 3. VoxSRC2023 (data_dir, trials_file_path 모두 변경)
VOXSRC_TRIALS="${DATASET_DIR}/voxsrc2023_final_fixed.txt"
if [ ! -f "$VOXSRC_TRIALS" ]; then
  VOXSRC_TRIALS="/home/sooyoung/interspeech/voxsrc2023_final_fixed.txt"
fi
if [ -f "$VOXSRC_TRIALS" ] && [ -d "$VOXSRC_DIR" ]; then
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
  echo "[SKIP] VoxSRC2023: trials=$VOXSRC_TRIALS data_dir=$VOXSRC_DIR"
fi

echo "[DONE] All tests completed"
