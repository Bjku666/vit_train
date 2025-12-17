#!/bin/bash

# --- 配置区 ---
LOG_DIR="./logs"
PORT=6006

echo " 清理旧的 TensorBoard 进程..."
pkill -f "tensorboard"

echo " 启动 TensorBoard..."
echo "监控目录: $LOG_DIR"
echo "端口: $PORT"

# --bind_all 允许远程访问 (SSH Tunnel)
# 日志重定向到 /dev/null 保持清爽
nohup tensorboard --logdir=$LOG_DIR --port $PORT --bind_all > /dev/null 2>&1 &

echo " TensorBoard 已在后台运行！"
echo "请在浏览器访问: http://localhost:$PORT"