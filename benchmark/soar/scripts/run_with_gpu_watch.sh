#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash benchmark/soar/scripts/run_with_gpu_watch.sh '<command>' [watch_out_dir] [interval_secs] [gpu_id]"
  exit 1
fi

CMD="$1"
WATCH_OUT_DIR="${2:-benchmark/soar/results/mem_watch}"
INTERVAL_SECS="${3:-2}"
GPU_ID="${4:-0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WATCH_SCRIPT="$SCRIPT_DIR/watch_gpu_mem.sh"

if [[ ! -f "$WATCH_SCRIPT" ]]; then
  echo "watch script not found: $WATCH_SCRIPT" >&2
  exit 2
fi

mkdir -p "$WATCH_OUT_DIR"

echo "[run_with_gpu_watch] starting watcher..."
bash "$WATCH_SCRIPT" "$WATCH_OUT_DIR" "$INTERVAL_SECS" "$GPU_ID" &
WATCH_PID=$!

cleanup() {
  if kill -0 "$WATCH_PID" >/dev/null 2>&1; then
    echo "[run_with_gpu_watch] stopping watcher pid=$WATCH_PID"
    kill "$WATCH_PID" >/dev/null 2>&1 || true
    wait "$WATCH_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[run_with_gpu_watch] running command: $CMD"
set +e
bash -lc "$CMD"
CMD_EXIT=$?
set -e

echo "[run_with_gpu_watch] command exit code: $CMD_EXIT"
echo "[run_with_gpu_watch] watcher logs under: $WATCH_OUT_DIR"

exit "$CMD_EXIT"
