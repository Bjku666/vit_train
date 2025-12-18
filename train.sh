#!/bin/bash
set -e

# Usage examples:
#   GPU_ID=0 MODEL_NAME="swin_base_patch4_window12_384" STAGE=1 RUN_ID=run1 ./train.sh
#   GPU_ID=0 MODEL_NAME="swin_base_patch4_window12_384" STAGE=2 RUN_ID=run1 INIT_CKPT="models/run_run1/..._stage1_best.pth" ./train.sh

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

STAGE="${STAGE:-1}"
export CURRENT_STAGE="${STAGE}"

# Use same RUN_ID across stages if you want stage2 to share the same output folder
if [ -n "$RUN_ID" ]; then
  export RUN_ID="$RUN_ID"
fi

# Default to Swin; override if needed
export MODEL_NAME="${MODEL_NAME:-swin_base_patch4_window12_384}"

# Default: single split
export USE_KFOLD="${USE_KFOLD:-0}"

mkdir -p logs

LOG_FILE="logs/train_${RUN_ID:-auto}_stage${STAGE}.log"

EXTRA_ARGS=""

# Stage2 warm start
if [ "$STAGE" = "2" ]; then
  # If user passed INIT_CKPT, use it
  if [ -n "$INIT_CKPT" ]; then
    EXTRA_ARGS="--init_ckpt ${INIT_CKPT}"
  else
    # Try auto-detect from same run folder
    if [ -n "$RUN_ID" ] && [ -d "models/run_${RUN_ID}" ]; then
      CAND=$(ls -1 models/run_${RUN_ID}/*_stage1_*_best.pth 2>/dev/null | head -n 1 || true)
      if [ -z "$CAND" ]; then
        CAND=$(ls -1 models/run_${RUN_ID}/*_stage1_*_epoch*.pth 2>/dev/null | tail -n 1 || true)
      fi
      if [ -n "$CAND" ]; then
        EXTRA_ARGS="--init_ckpt ${CAND}"
        echo "[Auto] init_ckpt => ${CAND}"
      else
        echo "[Warn] No stage1 checkpoint found for warm-start. Training stage2 from scratch."
      fi
    fi
  fi
fi

echo "========================================"
echo "Training"
echo "  GPU_ID      : ${GPU_ID}"
echo "  STAGE       : ${STAGE}"
echo "  RUN_ID      : ${RUN_ID:-auto}"
echo "  MODEL_NAME  : ${MODEL_NAME}"
echo "  USE_KFOLD   : ${USE_KFOLD}"
echo "  EXTRA_ARGS  : ${EXTRA_ARGS}"
echo "  LOG_FILE    : ${LOG_FILE}"
echo "========================================"

nohup python -u train_vit.py ${EXTRA_ARGS} > "${LOG_FILE}" 2>&1 &
PID=$!
echo "Started. PID=${PID}"
echo "tail -f ${LOG_FILE}"
tail -f "${LOG_FILE}"
