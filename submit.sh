#!/bin/bash
set -e

# Usage examples:
#   RUN_ID=run1 STAGE=2 MODEL_NAME="swin_base_patch4_window12_384" SELECT="epoch:006" ./submit.sh
#   RUN_ID=run1 STAGE=2 AVG_LAST_K=3 ./submit.sh

export CONFIG_INIT_DIRS=0

RUN_ID="${RUN_ID:?Please set RUN_ID=...}"
STAGE="${STAGE:-2}"
export CURRENT_STAGE="${STAGE}"

export MODEL_NAME="${MODEL_NAME:-swin_base_patch4_window12_384}"

CKPT_DIR="models/run_${RUN_ID}"
if [ ! -d "${CKPT_DIR}" ]; then
  echo "Checkpoint dir not found: ${CKPT_DIR}"
  exit 1
fi

PATTERN="${PATTERN:-*_stage${STAGE}_*.pth}"
SELECT="${SELECT:-best}"
AVG_LAST_K="${AVG_LAST_K:-0}"

# 1) Benchmark on labeled test set (if available)
echo "========================================"
echo "Benchmark (labeled test set)"
echo "  RUN_ID     : ${RUN_ID}"
echo "  STAGE      : ${STAGE}"
echo "  MODEL_NAME : ${MODEL_NAME}"
echo "  CKPT_DIR   : ${CKPT_DIR}"
echo "  PATTERN    : ${PATTERN}"
echo "  SELECT     : ${SELECT}"
echo "  AVG_LAST_K : ${AVG_LAST_K}"
echo "========================================"

python benchmark_vit.py \
  --ckpt "${CKPT_DIR}" \
  --pattern "${PATTERN}" \
  --select "${SELECT}" \
  --avg_last_k "${AVG_LAST_K}"

THR_JSON="output/benchmark_result_${RUN_ID}.json"
if [ ! -f "${THR_JSON}" ]; then
  # fallback: current env RUN_ID might differ; pick latest benchmark json
  THR_JSON=$(ls -1t output/benchmark_result_*.json | head -n 1 || true)
fi

# 2) Inference for submission
echo "========================================"
echo "Inference (unlabeled test set)"
echo "  threshold_json: ${THR_JSON}"
echo "========================================"

python inference.py \
  --ckpt "${CKPT_DIR}" \
  --pattern "${PATTERN}" \
  --select "${SELECT}" \
  --avg_last_k "${AVG_LAST_K}" \
  --threshold_json "${THR_JSON}"
