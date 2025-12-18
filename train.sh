#!/bin/bash
set -e

# 用法示例：
#   GPU_ID=0 MODEL_NAME="swin_base_patch4_window12_384" STAGE=1 RUN_ID=run1 ./train.sh
#   GPU_ID=0 MODEL_NAME="swin_base_patch4_window12_384" STAGE=2 RUN_ID=run1 INIT_CKPT="models/run_run1/..._stage1_best.pth" ./train.sh

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

STAGE="${STAGE:-1}"
export CURRENT_STAGE="${STAGE}"

# 如果希望 Stage2 共享同一输出目录，请在两个阶段使用相同的 RUN_ID
if [ -n "$RUN_ID" ]; then
  export RUN_ID="$RUN_ID"
fi

# 默认使用 Swin；需要时可覆盖
export MODEL_NAME="${MODEL_NAME:-swin_base_patch4_window12_384}"

# 默认：单划分
export USE_KFOLD="${USE_KFOLD:-0}"

mkdir -p logs

LOG_FILE="logs/train_${RUN_ID:-auto}_stage${STAGE}.log"

EXTRA_ARGS=""

# Stage2 预热（warm start）
if [ "$STAGE" = "2" ]; then
  # 若用户传入 INIT_CKPT，则直接使用
  if [ -n "$INIT_CKPT" ]; then
    EXTRA_ARGS="--init_ckpt ${INIT_CKPT}"
  else
    # 尝试从同一 run 目录自动查找
    if [ -n "$RUN_ID" ] && [ -d "models/run_${RUN_ID}" ]; then
      CAND=$(ls -1 models/run_${RUN_ID}/*_stage1_*_best.pth 2>/dev/null | head -n 1 || true)
      if [ -z "$CAND" ]; then
        CAND=$(ls -1 models/run_${RUN_ID}/*_stage1_*_epoch*.pth 2>/dev/null | tail -n 1 || true)
      fi
      if [ -n "$CAND" ]; then
        EXTRA_ARGS="--init_ckpt ${CAND}"
        echo "[自动] 使用 init_ckpt => ${CAND}"
      else
        echo "[警告] 未找到 Stage1 权重用于 warm-start，将从头训练 Stage2。"
      fi
    fi
  fi
fi

echo "========================================"
echo "训练"
echo "  GPU_ID      : ${GPU_ID}"
echo "  阶段(STAGE) : ${STAGE}"
echo "  RUN_ID      : ${RUN_ID:-auto}"
echo "  模型名称    : ${MODEL_NAME}"
echo "  使用K折     : ${USE_KFOLD}"
echo "  额外参数    : ${EXTRA_ARGS}"
echo "  日志文件    : ${LOG_FILE}"
echo "========================================"

nohup python -u train_vit.py ${EXTRA_ARGS} > "${LOG_FILE}" 2>&1 &
PID=$!
echo "已启动. PID=${PID}"
echo "tail -f ${LOG_FILE}"
tail -f "${LOG_FILE}"
