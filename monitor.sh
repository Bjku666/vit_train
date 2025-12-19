#!/usr/bin/env bash
set -e

echo "========================================"
#!/usr/bin/env bash
set -e

echo "========================================"
echo "        启动 TensorBoard（交互式）"
echo "========================================"

read -p "日志目录 [./logs]: " LOG_DIR
LOG_DIR=${LOG_DIR:-./logs}

read -p "端口 [6006]: " PORT
PORT=${PORT:-6006}

read -p "是否结束已存在的 tensorboard 进程? [Y/n]: " KILL
KILL=${KILL:-Y}
if [[ "$KILL" =~ ^[Yy]$ ]]; then
  pkill -f tensorboard || true
fi

MONITOR_LOG="monitor_tb.log"
echo "----------------------------------------"
echo "logdir : ${LOG_DIR}"
echo "port   : ${PORT}"
echo "log    : ${MONITOR_LOG}"
echo "----------------------------------------"

nohup tensorboard --logdir "${LOG_DIR}" --port "${PORT}" --bind_all > "${MONITOR_LOG}" 2>&1 &
PID=$!
echo "TensorBoard 已启动，PID=${PID}"
echo "访问地址: http://localhost:${PORT}"
