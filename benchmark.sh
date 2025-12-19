#!/usr/bin/env bash
set -e

echo "========================================"
echo "        本地评测（交互式）"
echo "========================================"

# 1) 带标签测试集目录
read -p "带标签测试集目录 [data/2-MedImage-TestSet]: " LABELED_TEST_DIR
LABELED_TEST_DIR=${LABELED_TEST_DIR:-data/2-MedImage-TestSet}

# 2) 模型名（timm 名称）
read -p "模型名（timm）[swin_base_patch4_window7_224]: " MODEL_NAME
MODEL_NAME=${MODEL_NAME:-swin_base_patch4_window7_224}

# 3) 阶段选择（自动匹配分辨率）
read -p "阶段 (1=224, 2=448) [2]: " STAGE
STAGE=${STAGE:-2}
if [ "$STAGE" = "1" ]; then
  IMG_SIZE=224
else
  IMG_SIZE=448
fi

# 4) 是否使用 2x TTA（水平翻转）
read -p "启用 TTA 水平翻转? [Y/n]: " USE_TTA
USE_TTA=${USE_TTA:-Y}

# 5) 选择 ckpt：支持通配符或空格分隔多个路径
echo "----------------------------------------"
echo "示例："
echo "  models/run_20251218_210000/*_stage${STAGE}_best.pth"
echo "  models/run_20251218_210000/*_stage${STAGE}_epoch*.pth"
echo "  /abs/path/to/xxx.pth /abs/path/to/yyy.pth"
read -p "输入 ckpt 路径（可多个）: " CKPT_INPUT

if [ -z "$CKPT_INPUT" ]; then
  echo "❌ No checkpoint provided. Abort."
  exit 1
fi

# 展开通配符到数组
eval "CKPTS=( $CKPT_INPUT )"
if [ ${#CKPTS[@]} -eq 0 ]; then
  echo "❌ No checkpoint resolved from input. Abort."
  exit 1
fi

echo "=================================="
echo "带标签测试集 : $LABELED_TEST_DIR"
echo "模型名       : $MODEL_NAME"
echo "输入尺寸     : $IMG_SIZE (stage $STAGE)"
echo "TTA          : ${USE_TTA}"
echo "检查点       : ${CKPTS[*]}"
echo "=================================="

ARGS=(
  --data_dir "$LABELED_TEST_DIR"
  --model_name "$MODEL_NAME"
  --img_size "$IMG_SIZE"
  --ckpts
)
for ck in "${CKPTS[@]}"; do
  ARGS+=("$ck")
done

if [[ "$USE_TTA" =~ ^[Yy]$ ]]; then
  ARGS+=(--tta)
fi

python local_eval.py "${ARGS[@]}"
