#!/bin/bash
set -e

# 提交脚本：仅对“无标签测试集”进行推理并写出提交 CSV。
#
# 用法示例：
#   RUN_ID=run1 STAGE=2 MODEL_NAME="swin_base_patch4_window12_384" SELECT="epoch:006" ./submit.sh
#   RUN_ID=run1 STAGE=2 AVG_LAST_K=3 ./submit.sh
#
# 可选参数：
#   THRESHOLD=0.62   # 覆盖阈值（推荐基于验证集决定）
#   NO_TTA=1         # 关闭 TTA

export CONFIG_INIT_DIRS=0

RUN_ID="${RUN_ID:?请设置 RUN_ID=...}"
STAGE="${STAGE:-2}"
export CURRENT_STAGE="${STAGE}"
export MODEL_NAME="${MODEL_NAME:-swin_base_patch4_window12_384}"

CKPT_DIR="models/run_${RUN_ID}"
if [ ! -d "${CKPT_DIR}" ]; then
  echo "[错误] 未找到权重目录: ${CKPT_DIR}"
  exit 1
fi

# ckpt pattern: modelname_stageX_*.pth
PATTERN="${PATTERN:-*_stage${STAGE}_*.pth}"

SELECT="${SELECT:-}"
AVG_LAST_K="${AVG_LAST_K:-0}"

# 解析阈值：
# 1) 环境变量 THRESHOLD
# 2) 读取 train_vit.py 写入的最新 *_meta.json 中的 best_thr
# 3) 兜底为 0.5
THRESHOLD_VAL=""
META_JSON=""
if [ -n "${THRESHOLD:-}" ]; then
  THRESHOLD_VAL="${THRESHOLD}"
else
  META_JSON=$(ls -1t "${CKPT_DIR}"/*_meta.json 2>/dev/null | head -n 1 || true)
  if [ -n "${META_JSON}" ]; then
    THRESHOLD_VAL=$(python - "${META_JSON}" <<'PY'
import json, sys
p = sys.argv[1]
try:
    j = json.load(open(p, "r", encoding="utf-8"))
    v = j.get("best_thr", j.get("best_threshold", j.get("threshold", 0.5)))
    print(float(v))
except Exception:
    print(0.5)
PY
)
  else
    THRESHOLD_VAL="0.5"
  fi
fi

NO_TTA_FLAG=""
if [ "${NO_TTA:-0}" = "1" ]; then
  NO_TTA_FLAG="--no_tta"
fi

echo "========================================"
echo "仅对无标签测试集进行推理"
echo "  RUN_ID      : ${RUN_ID}"
echo "  阶段(STAGE) : ${STAGE}"
echo "  模型名称    : ${MODEL_NAME}"
echo "  权重目录    : ${CKPT_DIR}"
echo "  匹配模式    : ${PATTERN}"
echo "  选择策略    : ${SELECT}"
echo "  均值最后K   : ${AVG_LAST_K}"
echo "  阈值        : ${THRESHOLD_VAL}"
echo "  元数据JSON  : ${META_JSON}"
echo "  TTA         : $([ -z "${NO_TTA_FLAG}" ] && echo 启用 || echo 关闭)"
echo "========================================"

python inference.py \
  --ckpt "${CKPT_DIR}" \
  --pattern "${PATTERN}" \
  --select "${SELECT}" \
  --avg_last_k "${AVG_LAST_K}" \
  --threshold "${THRESHOLD_VAL}" \
  ${NO_TTA_FLAG}
