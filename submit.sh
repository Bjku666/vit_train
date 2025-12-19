#!/usr/bin/env bash
set -e

echo "========================================"
echo "              提交脚本（交互式）"
echo "========================================"

read -p "RUN_ID（例如 20251218_210000）[自动生成，无需填写直接回车]: " RUN_ID
RUN_ID=${RUN_ID:-auto}

read -p "阶段 (1=224, 2=448) [2]: " STAGE
STAGE=${STAGE:-2}

read -p "模型名（timm）[swin_base_patch4_window7_224]: " MODEL_NAME
MODEL_NAME=${MODEL_NAME:-swin_base_patch4_window7_224}

# 先提示阈值（可直接回车使用默认），避免后续再输入
echo "----------------------------------------"
echo "如果你已用 ./benchmark.sh 在本地带标签集上评测过，"
echo "请将那里得到的 best_thr 填进来；否则直接回车使用默认（0.5 或 meta）。"
read -p "最佳阈值（可留空）: " THRESHOLD

# 选择一个或多个模型路径/通配符
echo "----------------------------------------"
echo "请输入一个或多个模型路径（支持通配符）"
echo "示例:"
echo "  models/run_${RUN_ID}/*_stage${STAGE}_best.pth"
echo "  models/run_${RUN_ID}/*_stage${STAGE}_epoch*.pth"
read -p "模型路径: " MPATH_INPUT

if [ -z "$MPATH_INPUT" ]; then
  # 默认使用本阶段的 best
  MPATH_INPUT="models/run_${RUN_ID}/*_stage${STAGE}_best.pth"
fi

# 是否使用 TTA
read -p "启用 TTA 水平翻转? [Y/n]: " USE_TTA
USE_TTA=${USE_TTA:-Y}

# 推断 image size
if [ "$STAGE" = "1" ]; then
  IMAGE_SIZE=224
else
  IMAGE_SIZE=448
fi

echo "=================================="
echo "RUN_ID     : $RUN_ID"
echo "阶段        : $STAGE (image_size=$IMAGE_SIZE)"
echo "模型名      : $MODEL_NAME"
echo "模型列表    : $MPATH_INPUT"
echo "TTA         : $USE_TTA"
echo "阈值        : ${THRESHOLD:-<默认>}"
echo "=================================="

# 展开通配符
eval "MODEL_PATHS=( $MPATH_INPUT )"
if [ ${#MODEL_PATHS[@]} -eq 0 ]; then
  echo "❌ No model path resolved. Abort."
  exit 1
fi

ARGS=(
  --model_paths
)
for m in "${MODEL_PATHS[@]}"; do
  ARGS+=("$m")
done

ARGS+=(--image_size "$IMAGE_SIZE" --stage "$STAGE")

if [[ "$USE_TTA" =~ ^[Yy]$ ]]; then
  ARGS+=(--tta)
fi

if [ -n "$THRESHOLD" ]; then
  ARGS+=(--threshold "$THRESHOLD")
fi

# 将关键信息传递给 Python 端（inference.py 使用 config.MODEL_NAME/CURRENT_STAGE）
export MODEL_NAME
export CURRENT_STAGE="$STAGE"

python submit.py "${ARGS[@]}"
