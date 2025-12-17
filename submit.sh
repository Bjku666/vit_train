#!/bin/bash

# ！！！关键环境变量！！！
# 告诉 Python 脚本不要创建新的 logs/run_xxx 目录，保持“只读”模式
export CONFIG_INIT_DIRS=0 

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
BENCHMARK_JSON=""

# 检查模型目录是否存在
if [ ! -d "$MODEL_DIR" ]; then
    echo " 错误: 找不到模型目录 $MODEL_DIR"
    exit 1
fi

echo ""
echo " 目标 Run ID: $RUN_ID"
echo "----------------------------------------"
echo "请选择操作模式:"
echo "  1) 仅 Benchmark (搜索最佳阈值)"
echo "  2) 仅 Inference (生成提交文件 - 需已有 JSON)"
echo "  3) 一键全流程 (Benchmark + Inference) [推荐]"
echo "----------------------------------------"
read -p "请输入数字 [1-3]: " MODE

# --- 函数定义 ---

run_benchmark() {
    echo ""
    echo " [Step 1] 正在运行 Benchmark 搜索最佳阈值..."
    # 通配符匹配该次训练的所有 fold 模型
    python benchmark_vit.py --model_paths "${MODEL_DIR}/vit_fold*.pth"

    # benchmark_vit.py 使用当前时间戳命名输出，这里自动取最新生成的结果文件
    BENCHMARK_JSON=$(ls -1t output/benchmark_result_*.json 2>/dev/null | head -n 1)
    if [ -z "$BENCHMARK_JSON" ] || [ ! -f "$BENCHMARK_JSON" ]; then
        echo " Benchmark 失败：未在 output/ 下找到 benchmark_result_*.json"
        exit 1
    fi
    echo " Benchmark 完成！阈值已保存。"
}

run_inference() {
    echo ""
    echo " [Step 2] 正在运行 Inference 生成提交文件..."
    
    if [ -z "$BENCHMARK_JSON" ] || [ ! -f "$BENCHMARK_JSON" ]; then
        # 兼容“仅 Inference”：尝试自动选择最新的 benchmark 结果
        BENCHMARK_JSON=$(ls -1t output/benchmark_result_*.json 2>/dev/null | head -n 1)
        if [ -z "$BENCHMARK_JSON" ] || [ ! -f "$BENCHMARK_JSON" ]; then
            echo " 错误: 找不到 benchmark_result_*.json，请先运行 Benchmark。"
            exit 1
        fi
        echo " [Info] 未指定 RUN_ID 对应的 JSON，已自动使用最新文件: $BENCHMARK_JSON"
    fi
    
    python inference.py \
        --model_paths "${MODEL_DIR}/vit_fold*.pth" \
        --benchmark_json "$BENCHMARK_JSON"
}

# --- 执行逻辑 ---

if [ "$MODE" == "1" ]; then
    run_benchmark

elif [ "$MODE" == "2" ]; then
    run_inference

elif [ "$MODE" == "3" ]; then
    run_benchmark
    run_inference

else
    echo " 无效输入"
    exit 1
fi

echo ""
echo " 所有任务执行完毕！"