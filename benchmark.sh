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

echo "----------------------------------------"
echo "查找可用 RUN：models/run_*"
# 收集 run 目录并排序（最新在前）
mapfile -t RUN_DIRS < <(find models -maxdepth 1 -type d -name "run_*" | sort -r)
if [ ${#RUN_DIRS[@]} -eq 0 ]; then
  echo "未找到任何 models/run_* 目录。请先训练或将权重放入 models/ 下。"
  exit 1
fi
echo "可选择的 RUN:"
idx=1
for d in "${RUN_DIRS[@]}"; do
  echo "  [$idx] $d"
  idx=$((idx+1))
done
read -p "选择要使用的 RUN（数字）[1]: " RUN_IDX
RUN_IDX=${RUN_IDX:-1}
if ! [[ "$RUN_IDX" =~ ^[0-9]+$ ]] || [ "$RUN_IDX" -lt 1 ] || [ "$RUN_IDX" -gt ${#RUN_DIRS[@]} ]; then
  echo "非法 RUN 选择: $RUN_IDX"
  exit 1
fi
TARGET_RUN_DIR=${RUN_DIRS[$((RUN_IDX-1))]}

# 在选定 RUN 下列出本阶段可用 ckpt
echo "----------------------------------------"
echo "可选的 Stage${STAGE} checkpoint:"
mapfile -t STAGE_CKPTS < <(find "$TARGET_RUN_DIR" -maxdepth 1 -type f -name "*_stage${STAGE}_*.pth" | sort)
if [ ${#STAGE_CKPTS[@]} -eq 0 ]; then
  echo "该 RUN 下未找到 *_stage${STAGE}_*.pth，可手动输入路径/通配。"
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
  echo "未选择任何 checkpoint。"
  exit 1
fi

# 解析输入：优先当作编号；否则当作路径/通配符
CKPTS=()
TOKENS=( $SELECTION )
ALL_NUM=1
for t in "${TOKENS[@]}"; do
  if [[ "$t" =~ ^[0-9]+$ ]]; then
    if [ ${#STAGE_CKPTS[@]} -eq 0 ]; then
      echo "无可编号的 ckpt，且输入为数字。"
      exit 1
    fi
    if [ "$t" -lt 1 ] || [ "$t" -gt ${#STAGE_CKPTS[@]} ]; then
      echo "编号越界: $t"
      exit 1
    fi
    CKPTS+=( "${STAGE_CKPTS[$((t-1))]}" )
  else
    # 作为路径/通配符展开
    eval "EXPANDED=( $t )"
    if [ ${#EXPANDED[@]} -eq 0 ]; then
      echo "路径/通配未匹配: $t"
      exit 1
    fi
    CKPTS+=( "${EXPANDED[@]}" )
    ALL_NUM=0
  fi
done

if [ ${#CKPTS[@]} -eq 0 ]; then
  echo "未解析到任何 checkpoint"
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

# 避免在评测阶段意外创建新的 models/run_* 目录
export CONFIG_INIT_DIRS=0

python local_eval.py "${ARGS[@]}"
