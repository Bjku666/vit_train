#!/usr/bin/env bash
set -e

echo "========================================"
echo "              提交脚本（交互式）"
echo "========================================"

read -p "阶段 (1=224, 2=448) [2]: " STAGE
STAGE=${STAGE:-2}

read -p "模型名（timm）[swin_base_patch4_window7_224]: " MODEL_NAME
MODEL_NAME=${MODEL_NAME:-swin_base_patch4_window7_224}

# 先提示阈值（可直接回车使用默认），避免后续再输入
echo "----------------------------------------"
echo "如果你已用 ./benchmark.sh 在本地带标签集上评测过，"
echo "请将那里得到的 best_thr 填进来；否则直接回车使用默认（0.5 或 meta）。"
read -p "最佳阈值（可留空）: " THRESHOLD

echo "----------------------------------------"
echo "查找可用 RUN：models/run_*"
mapfile -t RUN_DIRS < <(find models -maxdepth 1 -type d -name "run_*" | sort -r)
if [ ${#RUN_DIRS[@]} -eq 0 ]; then
  echo "❌ 未找到任何 models/run_* 目录。请先训练或将权重放入 models/ 下。"
  exit 1
fi
echo "可选择的 RUN:"
i=1
for d in "${RUN_DIRS[@]}"; do
  echo "  [$i] $d"
  i=$((i+1))
done
read -p "选择要使用的 RUN（数字）[1]: " RUN_IDX
RUN_IDX=${RUN_IDX:-1}
if ! [[ "$RUN_IDX" =~ ^[0-9]+$ ]] || [ "$RUN_IDX" -lt 1 ] || [ "$RUN_IDX" -gt ${#RUN_DIRS[@]} ]; then
  echo "❌ 非法 RUN 选择: $RUN_IDX"
  exit 1
fi
TARGET_RUN_DIR=${RUN_DIRS[$((RUN_IDX-1))]}
RUN_ID=${TARGET_RUN_DIR#models/run_}

echo "----------------------------------------"
echo "可选的 Stage${STAGE} checkpoint:"
mapfile -t STAGE_CKPTS < <(find "$TARGET_RUN_DIR" -maxdepth 1 -type f -name "*_stage${STAGE}_*.pth" | sort)
if [ ${#STAGE_CKPTS[@]} -eq 0 ]; then
  echo "⚠️ 该 RUN 下未找到 *_stage${STAGE}_*.pth，可手动输入路径/通配。"
else
  j=1
  for p in "${STAGE_CKPTS[@]}"; do
    echo "  [$j] $p"
    j=$((j+1))
  done
fi
echo "----------------------------------------"
echo "输入编号（可多个，空格分隔），或直接输入路径/通配符"
echo "示例：'1 3 5' 或 'models/run_*/swin*_stage${STAGE}_epoch*.pth'"
read -p "选择: " SELECTION

if [ -z "$SELECTION" ]; then
  # 默认使用本阶段的 best
  SELECTION="${TARGET_RUN_DIR}/*_stage${STAGE}_best.pth"
fi

MODEL_PATHS=()
TOKENS=( $SELECTION )
for t in "${TOKENS[@]}"; do
  if [[ "$t" =~ ^[0-9]+$ ]]; then
    if [ ${#STAGE_CKPTS[@]} -eq 0 ]; then
      echo "❌ 无可编号的 ckpt，且输入为数字。"
      exit 1
    fi
    if [ "$t" -lt 1 ] || [ "$t" -gt ${#STAGE_CKPTS[@]} ]; then
      echo "❌ 编号越界: $t"
      exit 1
    fi
    MODEL_PATHS+=( "${STAGE_CKPTS[$((t-1))]}" )
  else
    eval "EXPANDED=( $t )"
    if [ ${#EXPANDED[@]} -eq 0 ]; then
      echo "❌ 路径/通配未匹配: $t"
      exit 1
    fi
    MODEL_PATHS+=( "${EXPANDED[@]}" )
  fi
done

if [ ${#MODEL_PATHS[@]} -eq 0 ]; then
  echo "❌ No model path resolved. Abort."
  exit 1
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
echo "模型列表    : ${MODEL_PATHS[*]}"
echo "TTA         : $USE_TTA"
echo "阈值        : ${THRESHOLD:-<默认>}"
echo "=================================="

ARGS=(
  --ckpts
)
for m in "${MODEL_PATHS[@]}"; do
  ARGS+=("$m")
done

if [[ ! "$USE_TTA" =~ ^[Yy]$ ]]; then
  # 关闭 TTA：inference.py 以 --no_tta 控制
  ARGS+=(--no_tta)
fi

if [ -n "$THRESHOLD" ]; then
  ARGS+=(--threshold "$THRESHOLD")
fi

# 将关键信息传递给 Python 端（inference.py 使用 config.MODEL_NAME/CURRENT_STAGE/RUN_ID）
export MODEL_NAME
export CURRENT_STAGE="$STAGE"
export RUN_ID

python submit.py "${ARGS[@]}"
