#!/bin/bash

# ！！！关键环境变量！！！
# 告诉 Python 脚本不要创建新的 logs/run_xxx 目录，保持“只读”模式
export CONFIG_INIT_DIRS=0 

# 默认不启用伪标签（推理不需要，但保持一致）
export USE_PSEUDO_LABELS="${USE_PSEUDO_LABELS:-0}"

# --- 1. 自动列出最近的 Run ID ---
echo "========================================"
echo " 最近的训练记录 (models/run_*):"
ls -1t models/ | grep "run_" | head -n 5
echo "========================================"

# --- 2. 交互式输入 ---
read -p "请输入要使用的 RUN_ID (例如 20231217_153000): " RUN_ID

if [ -z "$RUN_ID" ]; then
    echo " 错误: RUN_ID 不能为空"
    exit 1
fi

MODEL_DIR="models/run_${RUN_ID}"

# 检查模型目录是否存在
if [ ! -d "$MODEL_DIR" ]; then
    echo " 错误: 找不到模型目录 $MODEL_DIR"
    exit 1
fi

echo ""
echo " 目标 Run ID: $RUN_ID"
echo "----------------------------------------"
echo "请选择推理所用模型阶段（必须与该 RUN_ID 训练分辨率一致）:"
echo "  1) Stage 1 (384)"
echo "  2) Stage 2 (512)"
read -p "请输入数字 [1-2]，默认 2: " STAGE
STAGE=${STAGE:-2}
export CURRENT_STAGE="$STAGE"

echo ""
echo " [Inference] 将执行：阈值搜索(读取 output/oof/oof_fold_*.csv) + 5fold×8TTA 推理"
echo " Stage: $CURRENT_STAGE"

python inference.py \
  --model_paths "${MODEL_DIR}/vit_fold*.pth"

echo ""
echo " 所有任务执行完毕！"