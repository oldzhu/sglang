# CHANGE_0048 Eval Bucket Accuracy Diagnostics

## Background and Motivation

Total `ori_accuracy` alone is too coarse for the current GPTQ debugging stage. We need to distinguish whether post-quantization regression is concentrated in specific task families, prompt-length regions, or both. This change adds a diagnostic evaluation variant that preserves the existing compatibility outputs while exposing bucketed accuracy summaries.

## Rule-Compliance Statement

This change is diagnostic-only and remains compliant with current SOAR 2026 guidance.

- It does not change the model, runtime serving path, or official evaluation logic.
- It keeps the legacy `ori_accuracy` and `overall_accuracy` fields intact.
- It operates only on the local/public evaluation flow to support debugging and calibration-set design.

## Detailed Implementation Plan

Before change:

- Copy `benchmark/soar/demo_sala/eval_model.py` to a new variant file.
- Keep current stdout and summary compatibility for total accuracy.
- Add per-task, per-length-bucket, and task-by-length-bucket aggregation.
- Save the new bucket diagnostics into both console output and `summary.json`.

## Actual Code Changes

- Added `benchmark/soar/demo_sala/eval_model_001.py`.
- Preserved compatibility outputs:
  - `Average Score` in stdout
  - `ori_accuracy` in `summary.json`
  - `overall_accuracy` in `summary.json`
- Added new bucket statistics:
  - `task`
  - `length_bucket`
  - `task_length_bucket`
- Added per-sample fields in prediction output:
  - `length_bucket`
  - `task_length_bucket`
- Added machine-readable `bucket_accuracy` object to `summary.json` and a single-line console print for log scraping.

Length buckets follow the same boundaries used in calibration analysis:

- `len_0_4k`
- `len_4k_32k`
- `len_32k_128k`
- `len_128k_plus`

## Validation Commands

```bash
python3 -m py_compile benchmark/soar/demo_sala/eval_model_001.py
python3 benchmark/soar/demo_sala/eval_model_001.py --help
```

Example run:

```bash
python3 benchmark/soar/demo_sala/eval_model_001.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_DIR> \
  --data_path benchmark/soar/demo_sala/perf_public_set.jsonl \
  --concurrency 32
```

## Result Summary Table

| Variant | Total compatibility output | Bucket diagnostics | Status |
|---|---|---|---|
| Original `eval_model.py` | Yes | No | Existing |
| New `eval_model_001.py` | Yes | Yes | Added |

## Rollback Instructions

- Continue using the original `benchmark/soar/demo_sala/eval_model.py`.
- Ignore or delete `benchmark/soar/demo_sala/eval_model_001.py` if the extra diagnostics are not needed.
- No runtime or preprocessing rollback is required because this is an isolated local evaluation helper.

## Next-Step Suggestions

1. Run the same model with both your current sequential calibration set and the new semantic-priority sets.
2. Compare which task and length buckets recover accuracy first.
3. Use those bucket deltas to decide whether the next calibration iteration should add more `qa`, more long-context samples, or a different mix entirely.