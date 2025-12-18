#!/usr/bin/env bash
set -e

# ============================
#  Interactive Train Launcher
#  single-split, no pseudo, no local benchmark
# ============================

echo "========================================"
echo "           Train (single-split)"
echo "========================================"

# ---- GPU
read -p "GPU_ID [default 0]: " GPU_ID
GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# ---- Stage
echo "Select STAGE:"
echo "  1) Stage 1 (base resolution, e.g. 384)"
echo "  2) Stage 2 (higher resolution, Swin must be divisible by 48 if window12/patch4)"
read -p "STAGE [default 1]: " STAGE
STAGE=${STAGE:-1}
export CURRENT_STAGE="${STAGE}"

# ---- RUN_ID (recommended: reuse the same RUN_ID across stage1/stage2)
read -p "RUN_ID (empty=auto timestamp, reuse same id for stage2) [default auto]: " RUN_ID
if [ -z "${RUN_ID}" ]; then
  RUN_ID=$(date +"%Y%m%d_%H%M%S")
fi

# ---- model name
read -p "MODEL_NAME [default swin_base_patch4_window12_384]: " MODEL_NAME
MODEL_NAME=${MODEL_NAME:-swin_base_patch4_window12_384}

# ---- epochs (optional)
read -p "EPOCHS (optional, empty=use config default): " EPOCHS

# ---- extra args (optional)
read -p "EXTRA_ARGS (optional): " EXTRA_ARGS

mkdir -p logs "models/run_${RUN_ID}"

LOG_FILE="logs/train_${RUN_ID}_stage${STAGE}.log"

# ---- Stage2 warm start: auto-detect stage1 best ckpt in same run
INIT_CKPT=""
if [ "${STAGE}" = "2" ]; then
  CAND=$(ls -1 "models/run_${RUN_ID}/"*"_stage1_best.pth" 2>/dev/null | head -n 1 || true)
  if [ -n "${CAND}" ]; then
    INIT_CKPT="--init_ckpt ${CAND}"
    echo "[Init] Found stage1 best ckpt: ${CAND}"
  else
    echo "[Warn] No stage1 best ckpt found under models/run_${RUN_ID}/. Stage2 will train from scratch."
  fi
fi

# ---- epochs arg
EPOCH_ARG=""
if [ -n "${EPOCHS}" ]; then
  EPOCH_ARG="--epochs ${EPOCHS}"
fi

# ---- print summary
echo "----------------------------------------"
echo "GPU_ID     : ${GPU_ID}"
echo "STAGE      : ${STAGE}"
echo "RUN_ID     : ${RUN_ID}"
echo "MODEL_NAME : ${MODEL_NAME}"
echo "LOG_FILE   : ${LOG_FILE}"
echo "INIT_CKPT  : ${INIT_CKPT}"
echo "EPOCHS     : ${EPOCHS:-<config default>}"
echo "EXTRA_ARGS : ${EXTRA_ARGS}"
echo "----------------------------------------"
echo "Command:"
echo "python train_vit.py --run_id ${RUN_ID} --stage ${STAGE} --model_name ${MODEL_NAME} ${INIT_CKPT} ${EPOCH_ARG} ${EXTRA_ARGS}"
echo "----------------------------------------"

# ---- launch (nohup + tail hint)
nohup python train_vit.py \
  --run_id "${RUN_ID}" \
  --stage "${STAGE}" \
  --model_name "${MODEL_NAME}" \
  ${INIT_CKPT} \
  ${EPOCH_ARG} \
  ${EXTRA_ARGS} \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "Started. PID=${PID}"
echo "tail -f ${LOG_FILE}"
