#!/usr/bin/env bash
set -e

echo "========================================"
echo "        Submit (single-split)"
echo "========================================"

# ---- choose run folder
runs=( $(ls -1t models/ | grep "^run_" | head -n 8 || true) )
if [ ${#runs[@]} -eq 0 ]; then
  echo "No run_ folders found under models/"
  exit 1
fi

echo "Recent runs:"
for i in "${!runs[@]}"; do
  echo "  $((i+1))) ${runs[$i]}"
done

read -p "Choose RUN [1-${#runs[@]}] (default 1): " RUN_IDX
RUN_IDX=${RUN_IDX:-1}
RUN_DIR="models/${runs[$((RUN_IDX-1))]}"
RUN_ID_FROM_DIR="${runs[$((RUN_IDX-1))]#run_}"
export RUN_ID="${RUN_ID_FROM_DIR}"

# ---- stage
echo "Select STAGE for inference (must match training resolution logic):"
echo "  1) Stage 1"
echo "  2) Stage 2"
read -p "STAGE [default 2]: " STAGE
STAGE=${STAGE:-2}
export CURRENT_STAGE="${STAGE}"

# ---- GPU
read -p "GPU_ID [default 0]: " GPU_ID
GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# ---- select checkpoint
echo "----------------------------------------"
echo "Choose checkpoint:"
echo "  1) use *_stage${STAGE}_best.pth"
echo "  2) choose an epoch ckpt (*_stage${STAGE}_epochXXX.pth)"
read -p "Option [1/2] (default 1): " OPT
OPT=${OPT:-1}

CKPT=""
if [ "${OPT}" = "1" ]; then
  CKPT=$(ls -1 "${RUN_DIR}/"*"_stage${STAGE}_best.pth" 2>/dev/null | head -n 1 || true)
  if [ -z "${CKPT}" ]; then
    echo "best checkpoint not found under ${RUN_DIR}/"
    exit 1
  fi
else
  cands=( $(ls -1 "${RUN_DIR}/"*"_stage${STAGE}_epoch"*.pth 2>/dev/null | tail -n 20 || true) )
  if [ ${#cands[@]} -eq 0 ]; then
    echo "No epoch checkpoints found under ${RUN_DIR}/"
    exit 1
  fi
  echo "Recent epoch checkpoints:"
  for i in "${!cands[@]}"; do
    echo "  $((i+1))) ${cands[$i]}"
  done
  read -p "Choose ckpt [1-${#cands[@]}] (default ${#cands[@]}): " CIDX
  CIDX=${CIDX:-${#cands[@]}}
  CKPT="${cands[$((CIDX-1))]}"
fi

echo "----------------------------------------"
echo "RUN_DIR : ${RUN_DIR}"
echo "STAGE   : ${STAGE}"
echo "GPU_ID  : ${GPU_ID}"
echo "CKPT    : ${CKPT}"
echo "----------------------------------------"

CKPT_BASE=$(basename "${CKPT}" .pth)
OUTPUT_CSV_DEFAULT="output/submission_${RUN_ID}_stage${STAGE}_${CKPT_BASE}.csv"
OUTPUT_CSV="${OUTPUT_CSV:-$OUTPUT_CSV_DEFAULT}"
echo "OUTPUT  : ${OUTPUT_CSV}"
echo "----------------------------------------"

# ---- run inference
# 你当前版本应该已经没有 benchmark / pseudo 逻辑了，所以 inference.py 只负责：
#   读取测试集 -> 推理 -> 输出 submission.csv
python inference.py --ckpt "${CKPT}" --output_csv "${OUTPUT_CSV}"

echo "Done. Check output/ for submission csv."
