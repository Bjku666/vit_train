#!/bin/bash

# --- 配置区 ---
# 指定 GPU ID (例如 0, 1 或 0,1)
GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 确保日志目录存在
mkdir -p logs

STAGE="${STAGE:-1}"
export CURRENT_STAGE="$STAGE"

# 可选：手动指定 RUN_ID 以实现 Stage1/Stage2 TensorBoard 无缝衔接
# 示例：
#   RUN_ID=20251217_120000 STAGE=1 ./train.sh
#   RUN_ID=20251217_120000 STAGE=2 ./train.sh
if [ -n "${RUN_ID:-}" ]; then
	export RUN_ID
fi

# 默认不启用伪标签（你当前不需要这一步）
export USE_PSEUDO_LABELS="${USE_PSEUDO_LABELS:-0}"

STAGE1_MODELS_DIR="${STAGE1_MODELS_DIR:-}"

# 若 Stage2 且你指定了 RUN_ID，但没显式给 Stage1 模型目录，则默认使用同一 run 目录
if [ "$CURRENT_STAGE" = "2" ] && [ -z "$STAGE1_MODELS_DIR" ] && [ -n "${RUN_ID:-}" ]; then
	STAGE1_MODELS_DIR="models/run_${RUN_ID}"
fi

echo "[ViT-Classification] 启动 5-Fold 训练 (Progressive Resizing)..."
echo "使用的 GPU: $GPU_ID"
echo "Stage: $CURRENT_STAGE | USE_PSEUDO_LABELS=$USE_PSEUDO_LABELS"
if [ -n "${RUN_ID:-}" ]; then
	echo "RUN_ID: $RUN_ID"
fi
if [ "$CURRENT_STAGE" = "2" ] && [ -n "$STAGE1_MODELS_DIR" ]; then
	echo "Stage1 模型目录: $STAGE1_MODELS_DIR"
fi

LOG_FILE="./train_stage${CURRENT_STAGE}.log"
echo "日志将写入: $LOG_FILE"

# --- 核心命令 ---
# 1. nohup: 后台运行，关闭终端不退出
# 2. python -u: 禁用输出缓冲，让日志实时写入
# 3. > ./...: 重定向 stdout 和 stderr 到文件
# 4. &: 放入后台
EXTRA_ARGS=""
if [ "$CURRENT_STAGE" = "2" ] && [ -n "$STAGE1_MODELS_DIR" ]; then
	EXTRA_ARGS="--stage1_models_dir $STAGE1_MODELS_DIR"
fi

nohup python -u train_vit.py $EXTRA_ARGS > "$LOG_FILE" 2>&1 &

PID=$!
echo "任务已启动！PID: $PID"
echo "正在追踪日志 (按 Ctrl+C 退出追踪，训练会继续)..."
echo "---------------------------------------------------"
tail -f "$LOG_FILE"