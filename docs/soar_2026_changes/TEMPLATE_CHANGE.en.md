# CHANGE_XXXX_<short_title>

## 1) Background & Motivation
- Problem statement:
- Why this should improve speed:
- Target stage(s): (prefill / decode / memory / kernel launch / scheduling)

## 2) SOAR Rule-Compliance Check
- Allowed by rules because:
- Not violating constraints (prefix cache/concurrency/reproducibility):
- Expected impact on correctness coefficient C:

## 3) Plan Before Code Change
- Files/functions to modify:
- Minimal diff strategy:
- Rollback plan:

## 4) Actual Code Change (After Approval)
- Commit/patch summary:
- Final modified files:
- Key logic differences:

## 5) Validation Commands
### Correctness
```bash
# Example
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_PATH> \
  --data_path <DATA_PATH>/perf_public_set.jsonl \
  --concurrency 32
```

### Speed
```bash
# Example
export SPEED_DATA_S1=<path_to_s1.jsonl>
export SPEED_DATA_S8=<path_to_s8.jsonl>
export SPEED_DATA_SMAX=<path_to_smax.jsonl>
bash bench_serving.sh http://127.0.0.1:30000
```

## 6) Results Summary
| Metric | Baseline | New | Delta |
|---|---:|---:|---:|
| Accuracy / overall_accuracy |  |  |  |
| S1 benchmark_duration (s) |  |  |  |
| S8 benchmark_duration (s) |  |  |  |
| S∞ benchmark_duration (s) |  |  |  |

## 7) Risk Assessment
- Accuracy risk:
- Stability risk:
- Reproducibility risk:

## 8) Rollback Instructions
1.
2.

## 9) Next-Step Suggestions
-
