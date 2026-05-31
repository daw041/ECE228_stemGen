#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python}"
ARCHIVE_PATH="${ARCHIVE_PATH:-dataset/archive.zip}"
OUT_DIR="${OUT_DIR:-dataset/audio_subset}"
N_TRACKS="${N_TRACKS:-200}"
START_FROM="${START_FROM:-0}"

if [[ ! -f "${ARCHIVE_PATH}" ]]; then
  echo "Archive not found: ${ARCHIVE_PATH}"
  echo "Upload or mount dataset/archive.zip first, or set ARCHIVE_PATH=/path/to/archive.zip"
  exit 1
fi

echo "Extracting ${N_TRACKS} rendered-bass tracks from ${ARCHIVE_PATH}"
"${PYTHON_BIN}" scripts/extract_audio_subset.py \
  --archive "${ARCHIVE_PATH}" \
  --out_dir "${OUT_DIR}" \
  --n_tracks "${N_TRACKS}" \
  --start_from "${START_FROM}"

echo "Prepared audio subset at ${OUT_DIR}"
