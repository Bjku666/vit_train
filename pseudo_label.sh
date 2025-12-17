#!/bin/bash

# 环境变量：防止脚本意外创建不需要的日志目录
export CONFIG_INIT_DIRS=0

echo "========================================"
echo "  伪标签生成工具 (Pseudo-Labeling)"
echo "========================================"

# --- 1. 自动列出最近的 Run ID ---
echo " 最近的训练记录 (models/run_*):"
ls -1t models/ | grep "run_" | head -n 5
echo "----------------------------------------"

# --- 2. 交互式选择模型 ---
read -p "请输入要使用的 RUN_ID (完整文件夹名后缀): " RUN_ID

if [ -z "$RUN_ID" ]; then
    echo " 错误: RUN_ID 不能为空"
    exit 1
fi

MODEL_DIR="models/run_${RUN_ID}"
# 检查模型目录
if [ ! -d "$MODEL_DIR" ]; then
    echo " 错误: 找不到模型目录 $MODEL_DIR"
    exit 1
fi

# --- 3. 交互式选择阶段 (分辨率对齐) ---
echo "----------------------------------------"
echo "请选择模型对应的训练阶段 (用于对齐分辨率):"
echo "  1) Stage 1 (384) [默认]"
echo "  2) Stage 2 (512)"
read -p "请输入数字 [1-2]: " STAGE
STAGE=${STAGE:-1}
export CURRENT_STAGE="$STAGE"
echo "已选择 Stage: $STAGE"

# --- 3. 交互式选择阈值 ---
# 伪标签的关键在于“宁缺毋滥”，所以默认值设得很高 (0.99)
read -p "请输入置信度阈值 [默认 0.99]: " THRESHOLD
THRESHOLD=${THRESHOLD:-0.99}

echo ""
echo "     配置确认:"
echo "   - 使用模型: $MODEL_DIR/vit_fold*.pth"
echo "   - 分辨率阶段: Stage $CURRENT_STAGE"
echo "   - 置信度阈值: $THRESHOLD"
echo "   - 目标目录: data/pseudo_labeled_set (由 config.py 定义)"
echo "----------------------------------------"
read -p "按回车键开始生成 (Ctrl+C 取消)..."

# --- 4. 执行核心命令 ---
# 注意：这里我们直接在前台运行，因为通常需要看进度条 (tqdm)
# 如果数据量巨大，可以在 python 命令前加 nohup 并去掉 -u
python generate_pseudo_labels.py \
    --model_paths "${MODEL_DIR}/vit_fold*.pth" \
    --threshold $THRESHOLD \
    --batch_size 64

echo ""
echo "伪标签生成完成！"
echo "下一步建议："
echo "1. 检查 data/pseudo_labeled_set 下的文件数量"
echo "2. 若要让训练加载伪标签，请显式启用："
echo "   USE_PSEUDO_LABELS=1 STAGE=1 ./train.sh"