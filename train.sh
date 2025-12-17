#!/bin/bash

# --- 配置区 ---
# 指定 GPU ID (例如 0, 1 或 0,1)
GPU_ID="${GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 确保日志目录存在
mkdir -p logs

echo "[ViT-Classification] 启动 5-Fold 全流程训练..."
echo "使用的 GPU: $GPU_ID"
echo "日志将写入: ./train_launcher.log"

# --- 核心命令 ---
# 1. nohup: 后台运行，关闭终端不退出
# 2. python -u: 禁用输出缓冲，让日志实时写入
# 3. > ./...: 重定向 stdout 和 stderr 到文件
# 4. &: 放入后台
nohup python -u train_vit.py > ./train_launcher.log 2>&1 &

PID=$!
echo "任务已启动！PID: $PID"
echo "正在追踪日志 (按 Ctrl+C 退出追踪，训练会继续)..."
echo "---------------------------------------------------"
tail -f ./train_launcher.log