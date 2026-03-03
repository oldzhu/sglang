# CHANGE_0003_run_suite_live_progress

## 1) Background & Motivation
- Problem statement: when `run_soar_suite.py` is launched via shell scripts, users cannot see stage progress until completion.
- Root cause:
  1) sub-process output was fully buffered via `subprocess.run(..., stdout=PIPE)`.
  2) bench command always forced `--disable-tqdm`.
- Goal: show live progress in terminal while preserving per-stage log files.

## 2) SOAR Rule-Compliance Check
- This is tooling/observability improvement only; no model algorithm or inference kernel behavior change.
- No impact on official forbidden/required settings.

## 3) Plan Before Code Change
- Modify one script only: `benchmark/soar/run_soar_suite.py`.
- Replace buffered capture with streaming capture.
- Make bench tqdm behavior configurable; default to enabled (visible progress).

## 4) Actual Code Change (After Approval)
- Updated `run_and_capture(...)` to use `subprocess.Popen` and stream stdout line-by-line to both terminal and log.
- Added labeled stage headers (e.g., `correctness`, `speed/s1`).
- Added optional `--disable-tqdm` flag to `run_soar_suite.py`.
- Removed unconditional `--disable-tqdm` from bench invocation.

## 5) Expected Behavior
- Running through `.sh` now shows real-time stage outputs.
- Logs are still persisted in run directory.
- If quiet mode is needed, pass `--disable-tqdm`.

## 6) Validation Commands
```bash
python3 benchmark/soar/run_soar_suite.py \
  --api-base http://127.0.0.1:30000 \
  --model-path /root/models/openbmb/MiniCPM-SALA \
  --eval-script /root/data/eval_model.py \
  --public-data /root/data/perf_public_set.jsonl \
  --speed-data-s1 /root/data/speed_s1.jsonl
```

Optional quiet benchmark progress:
```bash
python3 benchmark/soar/run_soar_suite.py \
  --api-base http://127.0.0.1:30000 \
  --model-path /root/models/openbmb/MiniCPM-SALA \
  --speed-data-s1 /root/data/speed_s1.jsonl \
  --disable-tqdm
```

## 7) Risks
- Very low. Change is limited to process output handling and bench CLI passthrough.

## 8) Rollback
1. Revert `benchmark/soar/run_soar_suite.py`.
2. Re-run a small correctness+S1 job to confirm original behavior.

## 9) Next Steps
- If needed, add per-stage timeout and retry options as a separate change.
