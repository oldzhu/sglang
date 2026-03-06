#!/usr/bin/env bash
set -euo pipefail

echo "[prepare_env] start $(date '+%F %T')"

uv pip install --no-deps -e ./sglang/python

# Optional: install extra dependencies for quant-prep here when needed.
# Example:
# uv pip install gptqmodel

QUANT_MODE="${SOAR_QUANT_MODE:-copy}"

if [[ "$QUANT_MODE" == "gptq" ]]; then
	export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS:-} --quantization gptq"
fi

export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS:-} --log-level info"

echo "[prepare_env] SOAR_QUANT_MODE=${QUANT_MODE}"
echo "[prepare_env] SGLANG_SERVER_ARGS=${SGLANG_SERVER_ARGS}"
echo "[prepare_env] done"
