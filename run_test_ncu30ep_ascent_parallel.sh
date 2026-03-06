#!/bin/bash
# NCU 30ep ascent - Voxceleb1-O, Voxceleb1-E, VoxSRC2023, CN-Celeb(E) 병렬 테스트
# Config: ncu_30ep_ascent.yaml
# Checkpoint: last

cd "$(dirname "$0")"

#export OMP_NUM_THREADS=4

CONFIG="configs/ncu_30ep_ascent.yaml"
CKPT_NAME="last"
DATASET_DIR="/home/sooyoung/interspeech/dataset"
DEV1_DIR="/home/sooyoung/interspeech/dev1"
VOXSRC_DIR="/home/sooyoung/interspeech/VoxSRC2023_test"
CNCELEB_DIR="/home/sooyoung/interspeech/CN-Celeb_flac/eval"

echo "=== NCU 30ep ascent | $CKPT_NAME | parallel ==="
echo ""

pids=()

# 1. Voxceleb1-O
if [ -f "${DATASET_DIR}/voxceleb1_O.txt" ]; then
  echo "[START] Voxceleb1-O | $CKPT_NAME"
  python run_test.py \
    --config "$CONFIG" \
    --ckpt_name "$CKPT_NAME" \
    --data.init_args.test_config.trials_file_path="${DATASET_DIR}/voxceleb1_O.txt" \
    --data.init_args.test_config.data_dir="$DEV1_DIR" \
    --no_wandb &
  pids+=($!)
else
  echo "[SKIP] Voxceleb1-O: ${DATASET_DIR}/voxceleb1_O.txt not found"
fi

# 2. Voxceleb1-E
if [ -f "${DATASET_DIR}/voxceleb1_E.txt" ]; then
  echo "[START] Voxceleb1-E | $CKPT_NAME"
  python run_test.py \
    --config "$CONFIG" \
    --ckpt_name "$CKPT_NAME" \
    --data.init_args.test_config.trials_file_path="${DATASET_DIR}/voxceleb1_E.txt" \
    --data.init_args.test_config.data_dir="$DEV1_DIR" \
    --no_wandb &
  pids+=($!)
else
  echo "[SKIP] Voxceleb1-E: ${DATASET_DIR}/voxceleb1_E.txt not found"
fi

# 3. VoxSRC2023
VOXSRC_TRIALS="${DATASET_DIR}/voxsrc2023_final_fixed.txt"
[ ! -f "$VOXSRC_TRIALS" ] && VOXSRC_TRIALS="/home/sooyoung/interspeech/voxsrc2023_final_fixed.txt"
if [ -f "$VOXSRC_TRIALS" ] && [ -d "$VOXSRC_DIR" ]; then
  echo "[START] VoxSRC2023 | $CKPT_NAME"
  python run_test.py \
    --config "$CONFIG" \
    --ckpt_name "$CKPT_NAME" \
    --data.init_args.test_config.trials_file_path="$VOXSRC_TRIALS" \
    --data.init_args.test_config.data_dir="$VOXSRC_DIR" \
    --no_wandb &
  pids+=($!)
else
  echo "[SKIP] VoxSRC2023: trials=$VOXSRC_TRIALS data_dir=$VOXSRC_DIR"
fi

# 4. CN-Celeb(E)
CNCELEB_TRIALS="${DATASET_DIR}/cnceleb_eval_pairs.txt"
[ ! -f "$CNCELEB_TRIALS" ] && CNCELEB_TRIALS="/home/sooyoung/interspeech/chns/resources/cnceleb_eval_pairs.txt"
if [ -f "$CNCELEB_TRIALS" ] && [ -d "$CNCELEB_DIR" ]; then
  echo "[START] CN-Celeb(E) | $CKPT_NAME"
  python run_test.py \
    --config "$CONFIG" \
    --ckpt_name "$CKPT_NAME" \
    --data.init_args.test_config.trials_file_path="$CNCELEB_TRIALS" \
    --data.init_args.test_config.data_dir="$CNCELEB_DIR" \
    --no_wandb &
  pids+=($!)
else
  echo "[SKIP] CN-Celeb(E): trials=$CNCELEB_TRIALS data_dir=$CNCELEB_DIR"
fi

echo ""
echo "[WAIT] Running ${#pids[@]} tests in parallel..."
wait "${pids[@]}"
echo "[DONE] All tests completed"
