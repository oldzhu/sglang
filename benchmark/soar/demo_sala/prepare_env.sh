#!/usr/bin/env bash
#set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLASH_ATTN_WHL="${SCRIPT_DIR}/flash_attn-2.8.3+cu128sm120-cp310-cp310-linux_x86_64.whl"

echo "[prepare_env] start $(date '+%F %T')"

uv pip install --no-deps -e ./sglang/python

uv pip install gptqmodel --no-build-isolation -v

if [[ ! -f "${FLASH_ATTN_WHL}" ]]; then
	echo "[prepare_env] missing flash-attn wheel: ${FLASH_ATTN_WHL}" >&2
	exit 1
fi

uv pip install "${FLASH_ATTN_WHL}" --no-build-isolation -v

uv pip uninstall torchao
uv pip install torchao==0.9.0

export SOAR_QUANT_MODE="${SOAR_QUANT_MODE:-gptq}"
QUANT_MODE="${SOAR_QUANT_MODE}"

export SOAR_GPTQ_CALIBRATION_FILE="${SOAR_GPTQ_CALIBRATION_FILE:-$(pwd)/perf_public_set.jsonl}"
export SOAR_GPTQ_CALIBRATION_SAMPLES="${SOAR_GPTQ_CALIBRATION_SAMPLES:-32}"
export SOAR_GPTQ_BATCH_SIZE="${SOAR_GPTQ_BATCH_SIZE:-1}"
export SOAR_GPTQ_BITS="${SOAR_GPTQ_BITS:-4}"
export SOAR_GPTQ_GROUP_SIZE="${SOAR_GPTQ_GROUP_SIZE:-128}"
export SOAR_GPTQ_ATTN_IMPL="${SOAR_GPTQ_ATTN_IMPL:-flash_attention_2}"
export SOAR_TRUST_REMOTE_CODE="${SOAR_TRUST_REMOTE_CODE:-true}"
export SOAR_GPTQ_LAYER_AWARE="${SOAR_GPTQ_LAYER_AWARE:-1}"
export SOAR_GPTQ_INCLUDE_MODULES="${SOAR_GPTQ_INCLUDE_MODULES:-self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj}"
export SOAR_GPTQ_EXCLUDE_MODULES="${SOAR_GPTQ_EXCLUDE_MODULES:-self_attn.o_gate,self_attn.z_proj}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.6}"

if [[ "$QUANT_MODE" == "gptq" ]]; then
	export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS:-} --trust-remote-code --disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 32768 --max-prefill-tokens 32768 --prefill-max-requests 1 --max-running-requests 20 --mem-fraction-static 0.84 --schedule-conservativeness 1.0 --skip-server-warmup --dense-as-sparse --quantization gptq_marlin --kv-cache-dtype fp8_e5m2 --force-dense-minicpm"
fi

# export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS:-} --log-level info"

echo "[prepare_env] SOAR_QUANT_MODE=${QUANT_MODE}"
echo "[prepare_env] SOAR_GPTQ_CALIBRATION_FILE=${SOAR_GPTQ_CALIBRATION_FILE}"
echo "[prepare_env] SOAR_GPTQ_CALIBRATION_SAMPLES=${SOAR_GPTQ_CALIBRATION_SAMPLES}"
echo "[prepare_env] SOAR_GPTQ_BATCH_SIZE=${SOAR_GPTQ_BATCH_SIZE}"
echo "[prepare_env] SGLANG_SERVER_ARGS=${SGLANG_SERVER_ARGS}"
echo "[prepare_env] done"
