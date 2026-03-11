# CHANGE_0014_gpu_memory_watcher

## 1) Background & Motivation
- OOM diagnosis benefits from timeline-level GPU memory/utilization logs during eval and benchmark.
- Goal: provide a lightweight, reusable watcher script that logs `nvidia-smi` stats to CSV.

## 2) SOAR Rule-Compliance Check
- Tooling/observability change only.
- No model/kernel/runtime path modification.

## 3) Plan Before Code Change
- Add one script under `benchmark/soar/scripts`.
- Keep dependencies minimal (`nvidia-smi` only).

## 4) Actual Code Change
- Added `benchmark/soar/scripts/watch_gpu_mem.sh`.
- Script behavior:
  - writes CSV with timestamp, memory used/total, GPU/memory utilization, temperature, power.
  - configurable output dir, sampling interval, and GPU id.

## 5) Usage
```bash
# default (out dir, 2s interval, gpu 0)
bash benchmark/soar/scripts/watch_gpu_mem.sh

# custom output dir + 1s interval + gpu 0
bash benchmark/soar/scripts/watch_gpu_mem.sh /root/soar_logs 1 0
```

## 6) Risks
- Very low. Monitoring-only script.

## 7) Rollback
1. Delete `benchmark/soar/scripts/watch_gpu_mem.sh`.

## 8) Next-Step Suggestions
- In future change, optionally add a wrapper to launch watcher + bench together and auto-stop watcher on completion.
# CHANGE_0014_gpu_memory_watcher

## 1) Background & Motivation
- OOM events can be intermittent and hard to diagnose from final traceback alone.
- We need a lightweight way to record GPU memory/utilization over time during eval/bench runs.

## 2) SOAR Rule-Compliance Check
- Tooling-only diagnostics change.
- No model/kernel/runtime algorithm modifications.

## 3) Plan Before Code Change
- Add one script under `benchmark/soar/scripts`.
- Script logs `nvidia-smi` metrics to CSV at a fixed interval.

## 4) Actual Code Change
- Added: `benchmark/soar/scripts/watch_gpu_mem.sh`
- Inputs:
  1. output directory (optional)
  2. interval seconds (optional)
  3. GPU id (optional)
- Output:
  - timestamped CSV with memory used/total, gpu/mem utilization, temperature, power.

## 5) Usage
```bash
# default
bash benchmark/soar/scripts/watch_gpu_mem.sh

# custom output + 1s interval + GPU 0
bash benchmark/soar/scripts/watch_gpu_mem.sh /root/soar_logs 1 0
```

## 6) Risks
- Very low; read-only monitoring.

## 7) Rollback
1. Delete `benchmark/soar/scripts/watch_gpu_mem.sh`.

## 8) Next-Step Suggestions
- Correlate CSV timestamps with scheduler/OOM logs to identify peak patterns.
