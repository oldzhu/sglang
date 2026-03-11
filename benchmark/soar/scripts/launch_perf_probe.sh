#!/usr/bin/env bash
set -euo pipefail

source /root/sglang/sglang_minicpm_sala_env/bin/activate

export MODEL_PATH=${MODEL_PATH:-/root/models/openbmb/MiniCPM-SALA}
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-30000}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}

pkill -f "sglang.launch_server" || true

python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --trust-remote-code \
  --disable-radix-cache \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 32768 \
  --prefill-max-requests 1 \
  --max-running-requests 20 \
  --mem-fraction-static 0.84 \
  --schedule-conservativeness 1.0 \
  --skip-server-warmup
