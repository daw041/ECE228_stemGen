#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${PYTHON_BIN:-}" && -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi
DATA_CONFIG="${DATA_CONFIG:-configs/runpod_data_config.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/model_config.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/runpod_train_config.yaml}"
RESUME="${RESUME:-}"
LOG_DIR="${LOG_DIR:-outputs/audio_token/runpod_e5_2cb/logs}"
TRAIN_ARGS=()

if [[ "${RESUME_MODEL_ONLY:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--resume_model_only)
fi
if [[ "${RESET_BEST_ON_RESUME:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--reset_best_on_resume)
fi
if [[ "${RESET_EPOCH_ON_RESUME:-0}" == "1" ]]; then
  TRAIN_ARGS+=(--reset_epoch_on_resume)
fi

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
    --resume "${RESUME}" \
    "${TRAIN_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
else
  "${PYTHON_BIN}" scripts/train.py \
    --data_config "${DATA_CONFIG}" \
    --model_config "${MODEL_CONFIG}" \
    --train_config "${TRAIN_CONFIG}" \
    --device cuda \
    "${TRAIN_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
fi
