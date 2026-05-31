#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-dataset/audio_subset}"

echo "== Synthetic E5 smoke test =="
"${PYTHON_BIN}" scripts/smoke_test_e5.py --device cuda --seq_len 64 --batch_size 2

echo "== Real-data E5 smoke test =="
"${PYTHON_BIN}" scripts/smoke_test_e5_data.py \
  --data_root "${DATA_ROOT}" \
  --clip_duration 10.0 \
  --device cuda

echo "RunPod smoke tests passed."
