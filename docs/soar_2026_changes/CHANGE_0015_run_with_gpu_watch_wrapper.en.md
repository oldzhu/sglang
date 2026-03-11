# CHANGE_0015_run_with_gpu_watch_wrapper

## 1) Background & Motivation
- We added a standalone GPU watcher script, but manual start/stop around each benchmark command is inconvenient.
- Goal: provide a wrapper that starts watcher, runs target command, then auto-stops watcher.

## 2) SOAR Rule-Compliance Check
- Tooling-only orchestration change.
- No model/runtime algorithm changes.

## 3) Plan Before Code Change
- Add one script under `benchmark/soar/scripts`.
- Keep behavior generic by accepting any command string.

## 4) Actual Code Change
- Added `benchmark/soar/scripts/run_with_gpu_watch.sh`.
- Script behavior:
  - starts `watch_gpu_mem.sh` in background,
  - runs target command,
  - traps EXIT/INT/TERM to stop watcher,
  - returns target command exit code.

## 5) Usage
```bash
# Wrap one S1 run
bash benchmark/soar/scripts/run_with_gpu_watch.sh \
  "python3 benchmark/soar/run_soar_suite.py --api-base http://127.0.0.1:30000 --model-path /root/models/openbmb/MiniCPM-SALA --speed-data-s1 /root/soar_fast_data/speed_s1.jsonl" \
  /root/soar_logs 1 0
```

## 6) Risks
- Very low; shell wrapper only.

## 7) Rollback
1. Delete `benchmark/soar/scripts/run_with_gpu_watch.sh`.

## 8) Next-Step Suggestions
- Optionally add mode to also tail server logs in parallel in a future change.
