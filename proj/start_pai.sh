#!/usr/bin/env bash
set -eu
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/csi-data-aly/user/ziroujiang/model/pai-diagnosis-qwen35-27b-5193/}"
DATA_ROOT="${DATA_ROOT:-/mnt/csi-data-aly/user/ziroujiang/datasets/train_data_v2}"
nvidia-smi || true
mkdir -p "$OUTPUT_DIR"
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -e "$PROJ/output" ]; then
  ln -sf "$OUTPUT_DIR" "$PROJ/output"
fi
cd "$PROJ"
pwd
ls -la
if [ ! -f "$DATA_ROOT/dataset.jsonl" ]; then
  echo "ERROR: missing $DATA_ROOT/dataset.jsonl" >&2
  exit 1
fi
ls -lh "$DATA_ROOT/dataset.jsonl"
ls "$DATA_ROOT/images" 2>/dev/null | head -3 || true
echo '[debug-env] disabled'
export DATA_ROOT
chmod +x train.sh
if [ "${SKIP_TRAIN:-0}" = "1" ]; then
  echo "[start_pai] SKIP_TRAIN=1, skip train.sh"
  exit 0
fi
exec bash train.sh
