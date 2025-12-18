#!/bin/bash
set -e

# Submit script.
# It only runs inference on the **unlabeled** test set and writes submission csv.
#
# Usage:
#   RUN_ID=run1 STAGE=2 MODEL_NAME="swin_base_patch4_window12_384" SELECT="epoch:006" ./submit.sh
#   RUN_ID=run1 STAGE=2 AVG_LAST_K=3 ./submit.sh
#
# Optional:
#   THRESHOLD=0.62   # override threshold (recommended: decided on validation)
#   NO_TTA=1         # disable TTA

export CONFIG_INIT_DIRS=0

RUN_ID="${RUN_ID:?Please set RUN_ID=...}"
STAGE="${STAGE:-2}"
export CURRENT_STAGE="${STAGE}"
export MODEL_NAME="${MODEL_NAME:-swin_base_patch4_window12_384}"

CKPT_DIR="models/run_${RUN_ID}"
if [ ! -d "${CKPT_DIR}" ]; then
  echo "[Error] checkpoint dir not found: ${CKPT_DIR}"
  exit 1
fi

# ckpt pattern: modelname_stageX_*.pth
PATTERN="${PATTERN:-*_stage${STAGE}_*.pth}"

SELECT="${SELECT:-}"
AVG_LAST_K="${AVG_LAST_K:-0}"

# Resolve threshold:
# 1) THRESHOLD env var
# 2) best_thr in latest *_meta.json written by train_vit.py
# 3) fallback 0.5
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
echo "Inference (unlabeled test set only)"
echo "  RUN_ID      : ${RUN_ID}"
echo "  STAGE       : ${STAGE}"
echo "  MODEL_NAME  : ${MODEL_NAME}"
echo "  CKPT_DIR    : ${CKPT_DIR}"
echo "  PATTERN     : ${PATTERN}"
echo "  SELECT      : ${SELECT}"
echo "  AVG_LAST_K  : ${AVG_LAST_K}"
echo "  THRESHOLD   : ${THRESHOLD_VAL}"
echo "  META_JSON   : ${META_JSON}"
echo "  TTA         : $([ -z "${NO_TTA_FLAG}" ] && echo enabled || echo disabled)"
echo "========================================"

python inference.py \
  --ckpt "${CKPT_DIR}" \
  --pattern "${PATTERN}" \
  --select "${SELECT}" \
  --avg_last_k "${AVG_LAST_K}" \
  --threshold "${THRESHOLD_VAL}" \
  ${NO_TTA_FLAG}
