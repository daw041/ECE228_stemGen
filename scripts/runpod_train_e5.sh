#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_CONFIG="${DATA_CONFIG:-configs/runpod_data_config.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model_config.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/runpod_train_config.yaml}"
RESUME="${RESUME:-}"
LOG_DIR="${LOG_DIR:-outputs/audio_token/runpod_e5_2cb/logs}"

mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/train_${STAMP}.log"

echo "Training log: ${LOG_FILE}"
echo "Data config: ${DATA_CONFIG}"
echo "Model config: ${MODEL_CONFIG}"
echo "Train config: ${TRAIN_CONFIG}"

if [[ -n "${RESUME}" ]]; then
  "${PYTHON_BIN}" scripts/train.py \
    --data_config "${DATA_CONFIG}" \
    --model_config "${MODEL_CONFIG}" \
    --train_config "${TRAIN_CONFIG}" \
    --device cuda \
    --resume "${RESUME}" 2>&1 | tee "${LOG_FILE}"
else
  "${PYTHON_BIN}" scripts/train.py \
    --data_config "${DATA_CONFIG}" \
    --model_config "${MODEL_CONFIG}" \
    --train_config "${TRAIN_CONFIG}" \
    --device cuda 2>&1 | tee "${LOG_FILE}"
fi
