#!/usr/bin/env bash
set -euo pipefail

source /root/sglang/sglang_minicpm_sala_env/bin/activate

export MODEL_PATH=${MODEL_PATH:-/root/models/openbmb/MiniCPM-SALA}
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-30000}
export CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-32768}

extra_args=()
if [[ -n "${MEM_FRACTION_STATIC:-}" ]]; then
  extra_args+=(--mem-fraction-static "$MEM_FRACTION_STATIC")
fi

pkill -f "sglang.launch_server" || true

# SOAR toolkit documented default example:
# --disable-radix-cache --attention-backend flashinfer --chunked-prefill-size 32768
python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  --trust-remote-code \
  --disable-radix-cache \
  --attention-backend flashinfer \
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
  --skip-server-warmup \
  "${extra_args[@]}"
