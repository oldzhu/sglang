#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

profile="${1:-}"
mode="${2:-run}"

if [[ -z "$profile" ]]; then
  echo "Usage: bash benchmark/soar/scripts/launch_profile.sh <safe|probe|default> [run|status|--dry-run]"
  exit 1
fi

if [[ "$mode" == "--dry-run" ]]; then
  mode="status"
fi

if [[ "$mode" != "run" && "$mode" != "status" ]]; then
  echo "Invalid mode: $mode"
  echo "Supported modes: run | status | --dry-run"
  exit 3
fi

target_script=""
resolved_cmd=""

case "$profile" in
  safe)
    target_script="$SCRIPT_DIR/launch_eval_safe.sh"
    resolved_cmd='python3 -m sglang.launch_server --model-path "$MODEL_PATH" --host "$HOST" --port "$PORT" --trust-remote-code --disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 4096 --max-prefill-tokens 8192 --prefill-max-requests 1 --max-running-requests 12 --mem-fraction-static 0.80 --schedule-conservativeness 1.2 --skip-server-warmup'
    ;;
  probe)
    target_script="$SCRIPT_DIR/launch_perf_probe.sh"
    resolved_cmd='python3 -m sglang.launch_server --model-path "$MODEL_PATH" --host "$HOST" --port "$PORT" --trust-remote-code --disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 8192 --max-prefill-tokens 16384 --prefill-max-requests 1 --max-running-requests 20 --mem-fraction-static 0.84 --schedule-conservativeness 1.0 --skip-server-warmup'
    ;;
  default)
    target_script="$SCRIPT_DIR/launch_toolkit_default.sh"
    resolved_cmd='python3 -m sglang.launch_server --model-path "$MODEL_PATH" --host "$HOST" --port "$PORT" --trust-remote-code --disable-radix-cache --attention-backend flashinfer --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" --skip-server-warmup [--mem-fraction-static "$MEM_FRACTION_STATIC" if set]'
    ;;
  *)
    echo "Invalid profile: $profile"
    echo "Supported profiles: safe | probe | default"
    exit 2
    ;;
esac

if [[ "$mode" == "status" ]]; then
  echo "[launch_profile] profile=$profile mode=status"
  echo "[launch_profile] script=$target_script"
  echo "[launch_profile] MODEL_PATH=${MODEL_PATH:-/root/models/openbmb/MiniCPM-SALA}"
  echo "[launch_profile] HOST=${HOST:-0.0.0.0}"
  echo "[launch_profile] PORT=${PORT:-30000}"
  if [[ "$profile" == "safe" || "$profile" == "probe" ]]; then
    echo "[launch_profile] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"
  fi
  if [[ "$profile" == "default" ]]; then
    echo "[launch_profile] CHUNKED_PREFILL_SIZE=${CHUNKED_PREFILL_SIZE:-32768}"
    echo "[launch_profile] MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-<unset>}"
  fi
  echo "[launch_profile] command=$resolved_cmd"
  exit 0
fi

exec bash "$target_script"
