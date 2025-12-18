#!/usr/bin/env bash
set -e

echo "========================================"
echo "              Monitor Logs"
echo "========================================"

# list recent logs
logs=( $(ls -1t logs/train_*_stage*.log 2>/dev/null | head -n 10 || true) )
if [ ${#logs[@]} -eq 0 ]; then
  echo "❌ No logs found under logs/"
  exit 1
fi

echo "Recent logs:"
for i in "${!logs[@]}"; do
  echo "  $((i+1))) ${logs[$i]}"
done

read -p "Choose log [1-${#logs[@]}] (default 1): " IDX
IDX=${IDX:-1}

if ! [[ "$IDX" =~ ^[0-9]+$ ]] || [ "$IDX" -lt 1 ] || [ "$IDX" -gt "${#logs[@]}" ]; then
  echo "❌ Invalid selection."
  exit 1
fi

LOG_FILE="${logs[$((IDX-1))]}"
echo "----------------------------------------"
echo "Tailing: ${LOG_FILE}"
echo "----------------------------------------"
tail -n 200 -f "${LOG_FILE}"
