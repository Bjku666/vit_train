#!/bin/bash

export CONFIG_INIT_DIRS=0 
export USE_PSEUDO_LABELS="${USE_PSEUDO_LABELS:-0}"

echo "========================================"
echo "    全流程评测与提交 (OOF版)"
echo "========================================"

# --- 1. 选择 RUN_ID ---
runs=($(ls -1t models/ | grep "run_" | head -n 5))
if [ ${#runs[@]} -eq 0 ]; then
    echo " ❌ 错误: models/ 目录下没有找到任何 run_ 文件夹。"
    exit 1
fi

echo "最近的训练记录:"
for i in "${!runs[@]}"; do
    echo "  $((i+1))) ${runs[$i]}"
done
echo "========================================"
read -p "选择版本 (输入序号 1-5，默认 1): " RUN_INDEX
RUN_INDEX=${RUN_INDEX:-1}

if [[ "$RUN_INDEX" =~ ^[0-9]+$ ]] && [ "$RUN_INDEX" -le "${#runs[@]}" ] && [ "$RUN_INDEX" -ge 1 ]; then
    RUN_ID=${runs[$((RUN_INDEX-1))]}
else
    RUN_ID=$RUN_INDEX
fi
MODEL_DIR="models/run_${RUN_ID}"

# --- 2. 选择 GPU ---
echo "----------------------------------------"
echo "请选择推理所用模型阶段（必须与该 RUN_ID 训练分辨率一致）:"
echo "  1) Stage 1 (384)"
echo "  2) Stage 2 (512)"
read -p "请输入数字 [默认 2]: " STAGE
export CURRENT_STAGE="${STAGE:-2}"

echo "----------------------------------------"
read -p "指定 GPU ID (例如 0 或 1) [默认 0]: " GPU_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"

# =======================================================
#   Step 1: OOF 评测 (秒级)
# =======================================================
echo ""
echo "📊 [Step 1] 正在读取 OOF 文件进行评测..."
echo "你想用哪些折的 OOF 数据来计算最佳阈值？"
echo "  1) 全部 5 折 (推荐，最稳)"
echo "  2) 指定单折 (例如只跑完 Fold 1 时用)"
read -p "选择 [1/2]: " EVAL_OPT

OOF_PATTERN="output/oof/oof_fold_*.csv" # 默认全部

if [ "$EVAL_OPT" == "2" ]; then
    read -p "请输入折数 (例如 1): " FOLD_NUM
    OOF_PATTERN="output/oof/oof_fold_${FOLD_NUM}.csv"
fi

# 调用 inference.py 的 --only_eval 模式
# 注意：这里 model_paths 随便填一个通配符即可，因为 eval 模式下不会加载模型
python inference.py \
    --model_paths "${MODEL_DIR}/vit_fold*.pth" \
    --oof_paths "$OOF_PATTERN" \
    --only_eval

# =======================================================
#   Step 2: 决策与提交
# =======================================================
echo ""
read -p "🚀 满意这个分数吗？是否生成提交文件? [y/N]: " DO_SUBMIT

if [ "$DO_SUBMIT" == "y" ] || [ "$DO_SUBMIT" == "Y" ]; then
    echo ""
    echo "📦 [Step 2] 正在加载模型进行推理..."
    echo "你想用哪些模型？"
    echo "  1) 全部 5 折集成"
    echo "  2) 指定单折模型"
    read -p "选择 [1/2]: " INF_OPT

    MODEL_PATTERN="${MODEL_DIR}/vit_fold*.pth"

    if [ "$INF_OPT" == "2" ]; then
        read -p "请输入折数 (例如 1): " MODEL_FOLD_NUM
        MODEL_PATTERN="${MODEL_DIR}/vit_fold${MODEL_FOLD_NUM}.pth"
    fi

    # 再次调用 inference.py (这次没有 --only_eval)
    python inference.py \
        --model_paths "$MODEL_PATTERN" \
        --oof_paths "$OOF_PATTERN"
else
    echo "已取消提交。"
fi

echo ""
echo "✅ 任务结束！"