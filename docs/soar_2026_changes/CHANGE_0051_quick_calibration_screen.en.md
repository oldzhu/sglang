# CHANGE_0051 Quick Calibration Screen

## Background and Motivation

Full public evaluation now takes about one hour per calibration candidate, which makes manual subset search too slow. Recent bucket analysis already showed the main instability sits in `mcq|len_0_4k`, `qa|len_4k_32k`, and `qa|len_32k_128k`. The next useful step is a smaller, fixed public micro-eval that preserves those weak-bucket signals while cutting turnaround time.

## Rule-Compliance Statement

This change remains within current SOAR 2026 rules.

- It only reads rows from the public `perf_public_set.jsonl` file.
- It does not alter model weights, runtime kernels, concurrency rules, or official evaluation logic.
- It is an offline diagnostic and ranking tool for deciding which calibration candidates deserve a full run.

## Detailed Implementation Plan

Before change:

- Define a fixed 20-row public subset centered on the current weak buckets.
- Keep the subset reproducible by storing indices in a config file instead of copying a second large JSONL dataset.
- Reuse the existing `eval_model_001.py` generation and scoring logic so quick-screen results stay aligned with the full evaluator.
- Report both overall subset accuracy and focused weak-bucket metrics.

## Actual Code Changes

- Added `benchmark/soar/demo_sala/quick_screen_public_subset.json`.
- Added `benchmark/soar/demo_sala/quick_calibration_screen.py`.

Subset design:

- 20 total public rows
- fixed indices: `1, 2, 4, 8, 11, 17, 23, 25, 61, 63, 65, 66, 68, 70, 71, 76, 80, 81, 85, 90`
- intended focus buckets:
  - `task=mcq|len_0_4k`
  - `task=qa|len_4k_32k`
  - `task=qa|len_32k_128k`

Quick-screen outputs include:

- `quick_screen_accuracy`: raw accuracy on the 20-row subset
- `focus_bucket_average`: mean accuracy across the three target task-length buckets
- `focus_bucket_min`: weakest target-bucket accuracy
- per-task, per-length, and per-task-length bucket tables

The script also supports `--check` so the fixed subset can be validated without calling the model.

## Validation Commands

```bash
python3 -m py_compile benchmark/soar/demo_sala/quick_calibration_screen.py
python3 benchmark/soar/demo_sala/quick_calibration_screen.py --check
python3 benchmark/soar/demo_sala/quick_calibration_screen.py \
  --model_path openbmb/MiniCPM-SALA \
  --api_base http://127.0.0.1:30000
```

## Result Summary Table

| Artifact | Purpose | Status |
|---|---|---|
| `quick_screen_public_subset.json` | Fixed reproducible 20-row quick-screen definition | Added |
| `quick_calibration_screen.py` | Fast bucket-aware proxy evaluator | Added |

## Rollback Instructions

- Stop using `quick_calibration_screen.py` and return to `eval_model_001.py` only.
- Ignore the subset config if manual or full public evaluation is preferred.
- No model or runtime rollback is required because this change only adds offline tooling.

## Next-Step Suggestions

1. Run the quick screen on 2-4 strongest calibration candidates first.
2. Promote only the top quick-screen candidates to the 1-hour full evaluation.
3. If quick-screen ranking correlates poorly with full evaluation, the next upgrade should be a BF16-vs-GPTQ output-drift screen rather than more manual subset guessing.