#!/usr/bin/env bash
set -e

# ============================
#  交互式训练启动器（中文提示）
#  - 自动生成 RUN_ID（时间戳）
#  - Stage1/Stage2 无需手动输入 RUN_ID
#  - 与 train_vit.py 参数严格对齐（仅使用支持的参数）
# ============================

echo "========================================"
echo "               训练启动器"
echo "========================================"

# 选择 GPU（仅设置可见卡）
read -p "GPU 编号 [默认 0]: " GPU_ID
GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# 选择阶段（分辨率在 config.py 内由 CURRENT_STAGE 控制）
echo "选择阶段："
echo "  1) Stage 1（低分辨率暖启动）"
echo "  2) Stage 2（高分辨率微调）"
read -p "输入 1 或 2 [默认 1]: " STAGE
STAGE=${STAGE:-1}
export CURRENT_STAGE="${STAGE}"

# 选择模型名（与 config.MODEL_NAME 保持一致，默认 swin_base_patch4_window7_224）
read -p "模型名（timm）[swin_base_patch4_window7_224]: " MODEL_NAME
MODEL_NAME=${MODEL_NAME:-swin_base_patch4_window7_224}
export MODEL_NAME

# 是否使用 KFold
read -p "使用 KFold 训练？[y/N]: " USE_KFOLD_INPUT
USE_KFOLD_INPUT=${USE_KFOLD_INPUT:-N}
if [[ "$USE_KFOLD_INPUT" =~ ^[Yy]$ ]]; then
  export USE_KFOLD=1
else
  export USE_KFOLD=0
fi

EXTRA_ARGS=""
if [ "${STAGE}" = "1" ]; then
  RUN_ID=$(date +"%Y%m%d_%H%M%S")
  export RUN_ID
else
  # Stage2：交互式承接 Stage1 并复用 RUN_ID
  # 1) 收集含有 Stage1 产物的 run 目录
  mapfile -t CAND_RUNS < <(ls -1dt models/run_* 2>/dev/null || true)
  STAGE1_RUNS=()
  for d in "${CAND_RUNS[@]}"; do
    if ls -1 "$d"/*_stage1_best.pth >/dev/null 2>&1 || ls -1 "$d"/*_stage1_fold*_best.pth >/dev/null 2>&1 || ls -1 "$d"/*_stage1_epoch*.pth >/dev/null 2>&1; then
      STAGE1_RUNS+=("$d")
    fi
  done

  if [ ${#STAGE1_RUNS[@]} -eq 0 ]; then
    RUN_ID=$(date +"%Y%m%d_%H%M%S")
    export RUN_ID
    echo "[提示] 未找到任何 Stage1 产物。Stage2 将从预训练开始训练。"
  else
    echo "可用的 Stage1 运行目录："
    idx=1
    for d in "${STAGE1_RUNS[@]}"; do
      echo "  [$idx] $d"
      idx=$((idx+1))
    done
    read -p "选择要承接的 RUN（数字）[1]: " RUN_SEL
    RUN_SEL=${RUN_SEL:-1}
    STAGE1_DIR="${STAGE1_RUNS[$((RUN_SEL-1))]}"
    RUN_ID=$(basename "$STAGE1_DIR" | sed 's/^run_//')
    export RUN_ID

    if [ "$USE_KFOLD" = "1" ]; then
      # KFold：传目录，由训练脚本逐 fold 读取 *_stage1_fold{k}_best
      EXTRA_ARGS="--stage1_models_dir ${STAGE1_DIR}"
      echo "[Warm-Start] Stage2(KFold) 复用 RUN_ID=${RUN_ID} | dir=${STAGE1_DIR}"
    else
      # 单划分：列出可选 checkpoint（best 或具体 epoch）
      echo "可选的 Stage1 checkpoint："
      mapfile -t BESTS < <(ls -1t "$STAGE1_DIR"/*_stage1_best.pth 2>/dev/null || true)
      mapfile -t EPOCHS < <(ls -1t "$STAGE1_DIR"/*_stage1_epoch*.pth 2>/dev/null || true)
      OPTIONS=()
      if [ ${#BESTS[@]} -gt 0 ]; then
        OPTIONS+=("BEST:${BESTS[0]}")
      fi
      for ep in "${EPOCHS[@]}"; do
        OPTIONS+=("EPOCH:${ep}")
      done
      if [ ${#OPTIONS[@]} -eq 0 ]; then
        echo "  (未找到 *_stage1_best 或 *_stage1_epoch*.pth，将从预训练开始)"
      else
        i=1
        for opt in "${OPTIONS[@]}"; do
          disp=$(echo "$opt" | cut -d':' -f2)
          echo "  [$i] $disp"
          i=$((i+1))
        done
        read -p "选择 checkpoint（数字）[1]: " CK_SEL
        CK_SEL=${CK_SEL:-1}
        CHOSEN=$(echo "${OPTIONS[$((CK_SEL-1))]}" | cut -d':' -f2)
        EXTRA_ARGS="--init_ckpt ${CHOSEN}"
        echo "[Warm-Start] Stage2(single) 复用 RUN_ID=${RUN_ID} | init_ckpt=${CHOSEN}"
      fi
    fi
  fi
fi

mkdir -p logs "models/run_${RUN_ID}"

LOG_FILE="logs/train_${RUN_ID}_stage${STAGE}.log"

echo "----------------------------------------"
echo "GPU_ID     : ${GPU_ID}"
echo "STAGE      : ${STAGE}"
echo "RUN_ID     : ${RUN_ID}"
echo "MODEL_NAME : ${MODEL_NAME}"
echo "EXTRA_ARGS : ${EXTRA_ARGS:-<none>}"
echo "日志文件   : ${LOG_FILE}"
echo "----------------------------------------"

# 启动训练（与 train_vit.py 参数保持一致：仅传 --stage1_models_dir 或 --init_ckpt）
nohup python -u train_vit.py ${EXTRA_ARGS} > "${LOG_FILE}" 2>&1 &

PID=$!
echo "已启动，PID=${PID}"
echo "tail -f ${LOG_FILE}"
