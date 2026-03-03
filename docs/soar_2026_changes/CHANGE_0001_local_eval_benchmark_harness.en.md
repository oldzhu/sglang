# CHANGE_0001_local_eval_benchmark_harness

## 1) Background & Motivation
- Problem statement: Iteration speed is limited by manual correctness/speed runs and ad-hoc log parsing.
- Why this should improve speed: Faster and more reliable experiment loops reduce time-to-valid-optimization and avoid invalid submissions.
- Target stage(s): scheduling / benchmarking workflow (not runtime engine kernels).

## 2) SOAR Rule-Compliance Check
- Allowed by rules because: this change only automates local evaluation/benchmark orchestration.
- Not violating constraints (prefix cache/concurrency/reproducibility): this script does not modify server behavior and explicitly benchmarks S1/S8/S∞ tiers with fixed concurrency options.
- Expected impact on correctness coefficient C: no direct model behavior change; only improves observability and regression control.

## 3) Plan Before Code Change
- Files/functions to modify:
  - `benchmark/soar/run_soar_suite.py` (new)
  - This EN/ZH change-document pair
- Minimal diff strategy: add one standalone script and docs only; no inference engine source touched.
- Rollback plan: remove the new script and this document pair.

## 4) Actual Code Change (After Approval)
- Patch summary:
  - Added `benchmark/soar/run_soar_suite.py`.
  - Added support for one-command local suite:
    - Optional correctness run through toolkit `eval_model.py`.
    - Speed benchmark runs for S1/S8/S∞ via `python -m sglang.bench_serving`.
    - Auto-adapt speed dataset from SOAR format (`question` + `model_response`) to bench custom format.
    - Emit structured `summary.json` and tier logs.
- Final modified files:
  - `benchmark/soar/run_soar_suite.py`
  - `docs/soar_2026_changes/CHANGE_0001_local_eval_benchmark_harness.en.md`
  - `docs/soar_2026_changes/CHANGE_0001_local_eval_benchmark_harness.zh.md`
- Key logic differences: standardizes run commands and result collection; no runtime generation path changes.

## 5) Validation Commands
### Correctness + Speed (single command)
```bash
python3 benchmark/soar/run_soar_suite.py \
  --api-base http://127.0.0.1:30000 \
  --model-path /root/models/openbmb/MiniCPM-SALA \
  --eval-script /path/to/SOAR-Toolkit/eval_model.py \
  --public-data /path/to/perf_public_set.jsonl \
  --speed-data-s1 /path/to/s1.jsonl \
  --speed-data-s8 /path/to/s8.jsonl \
  --speed-data-smax /path/to/smax.jsonl
```

### Speed only
```bash
python3 benchmark/soar/run_soar_suite.py \
  --api-base http://127.0.0.1:30000 \
  --model-path /root/models/openbmb/MiniCPM-SALA \
  --speed-data-s1 /path/to/s1.jsonl \
  --speed-data-s8 /path/to/s8.jsonl \
  --speed-data-smax /path/to/smax.jsonl
```

## 6) Results Summary
| Metric | Baseline | New | Delta |
|---|---:|---:|---:|
| Accuracy / overall_accuracy | N/A | N/A | N/A |
| S1 benchmark_duration (s) | N/A | N/A | N/A |
| S8 benchmark_duration (s) | N/A | N/A | N/A |
| S∞ benchmark_duration (s) | N/A | N/A | N/A |

Notes:
- This change is workflow tooling only; no direct model-runtime acceleration is claimed.
- Fill baseline/new numbers after running on fcloud.

## 7) Risk Assessment
- Accuracy risk: none expected (no inference logic change).
- Stability risk: low; script may fail if dataset format is unsupported.
- Reproducibility risk: low; script improves reproducibility by recording logs and summary JSON.

## 8) Rollback Instructions
1. Delete `benchmark/soar/run_soar_suite.py`.
2. Delete this EN/ZH document pair.

## 9) Next-Step Suggestions
- Use this harness to establish baseline S1/S8/S∞ + correctness, then proceed to first engine-level optimization (attention backend micro-optimization) with strict A/B comparison.
