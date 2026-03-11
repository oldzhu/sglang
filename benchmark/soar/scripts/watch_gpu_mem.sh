#!/usr/bin/env bash
set -euo pipefail

OUT_DIR=${1:-benchmark/soar/results/mem_watch}
INTERVAL_SECS=${2:-2}
GPU_ID=${3:-0}

mkdir -p "$OUT_DIR"
TS=$(date +%Y%m%d_%H%M%S)
OUT_FILE="$OUT_DIR/gpu_mem_${TS}.csv"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. This script requires NVIDIA driver tools." >&2
  exit 1
fi

echo "timestamp,gpu_index,mem_used_mb,mem_total_mb,util_gpu,util_mem,temperature,power_w" > "$OUT_FILE"
echo "[watch_gpu_mem] logging to $OUT_FILE (interval=${INTERVAL_SECS}s, gpu=${GPU_ID})"
echo "[watch_gpu_mem] stop with Ctrl+C"

while true; do
  NOW=$(date +"%Y-%m-%d %H:%M:%S")
  LINE=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw --format=csv,noheader,nounits -i "$GPU_ID")
  echo "$NOW,$LINE" >> "$OUT_FILE"
  sleep "$INTERVAL_SECS"
done
