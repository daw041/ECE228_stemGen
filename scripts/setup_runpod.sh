#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "== Python =="
"${PYTHON_BIN}" - <<'PY'
import sys
print(sys.executable)
print(sys.version)
PY

echo "== Install requirements =="
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -r requirements.txt

echo "== Environment check =="
"${PYTHON_BIN}" - <<'PY'
import importlib.util
pkgs = ["torch", "torchaudio", "encodec", "yaml", "librosa", "soundfile", "matplotlib", "tensorboard"]
for pkg in pkgs:
    print(f"{pkg}: {bool(importlib.util.find_spec(pkg))}")

import torch
print("cuda:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "== E5 synthetic smoke test =="
"${PYTHON_BIN}" scripts/smoke_test_e5.py --device cuda --seq_len 64 --batch_size 2

echo "RunPod setup complete."
