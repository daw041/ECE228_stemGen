#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="${CHECKPOINT:-outputs/audio_token/runpod_e5_2cb/checkpoints/best.pt}"
DATA_CONFIG="${DATA_CONFIG:-configs/runpod_data_config.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model_config.yaml}"
OUT_DIR="${OUT_DIR:-outputs/audio_token/runpod_e5_2cb/diagnostics}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}"
  exit 1
fi

"${PYTHON_BIN}" scripts/diagnose_audio_token.py \
  --checkpoint "${CHECKPOINT}" \
  --data_config "${DATA_CONFIG}" \
  --model_config "${MODEL_CONFIG}" \
  --device cuda \
  --mask_ratios 0.15,0.30,0.50,0.75,1.00 \
  --iterations_per_codebook 32,16 \
  --temperature 0.8 \
  --top_k 50 \
  --output_dir "${OUT_DIR}"

echo "Diagnostics saved to ${OUT_DIR}"
