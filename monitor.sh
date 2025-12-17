#!/bin/bash

# --- 配置区 ---
LOG_DIR="./logs"
PORT=6006
MONITOR_LOG="./monitor.log"

echo " 清理旧的 TensorBoard 进程..."
pkill -f "tensorboard"

echo " 启动 TensorBoard..."
echo "监控目录: $LOG_DIR"
echo "端口: $PORT"
echo "Monitor Log: $MONITOR_LOG"

# --bind_all 允许远程访问 (SSH Tunnel)
# 日志写入项目根目录 monitor.log，便于排查问题
nohup tensorboard --logdir=$LOG_DIR --port $PORT --bind_all > "$MONITOR_LOG" 2>&1 &

echo " TensorBoard 已在后台运行！"
echo "请在浏览器访问: http://localhost:$PORT"