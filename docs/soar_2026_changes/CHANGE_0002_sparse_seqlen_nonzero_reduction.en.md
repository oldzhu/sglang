# CHANGE_0002_sparse_seqlen_nonzero_reduction

## 1) Background & Motivation
- Problem statement: Evaluation-phase OOM occurred in MiniCPM sparse attention extend path when processing long-context and concurrent requests.
- Root location: `python/sglang/srt/layers/attention/minicpm_backend.py` around sparse cache seqlen computation.
- Why this should help: reduce temporary tensor memory footprint during per-batch sparse sequence length reduction.

## 2) SOAR Rule-Compliance Check
- Allowed by rules: runtime memory optimization in inference engine implementation.
- No forbidden behavior: does not enable prefix cache tricks and does not alter official concurrency protocol.
- Reproducibility: deterministic code-path replacement with equivalent semantics.

## 3) Plan Before Code Change
- One focused change only:
  - Replace `(metadata.sparse_page_table != 0).sum(dim=1)`
  - With `torch.count_nonzero(metadata.sparse_page_table, dim=1)`
- Keep dtype/device cast unchanged downstream.
- Rollback: revert this one expression.

## 4) Speed Impact Assessment (Documenting User Q/A)
- Expected inference speed impact: no meaningful slowdown.
- Why: `count_nonzero` avoids explicit materialization of a large boolean temporary tensor before reduction.
- Practical expectation:
  - Decode throughput: essentially unchanged.
  - Prefill-heavy long context: usually unchanged to slightly better.
  - Main benefit: lower peak memory and fewer OOM interruptions.
- Risk note: tiny variance-level regression is still possible and should be checked by A/B repetition.

## 5) Actual Code Change (After Approval)
- Modified file:
  - `python/sglang/srt/layers/attention/minicpm_backend.py`
- Logic change:
  - Old: boolean temporary + sum reduction.
  - New: direct nonzero-count reduction.

## 6) Validation Commands (fcloud)
### Correctness stress ramp
```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path /root/models/openbmb/MiniCPM-SALA \
  --data_path <DATA_DIR>/perf_public_set.jsonl \
  --concurrency 8

python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path /root/models/openbmb/MiniCPM-SALA \
  --data_path <DATA_DIR>/perf_public_set.jsonl \
  --concurrency 16

python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path /root/models/openbmb/MiniCPM-SALA \
  --data_path <DATA_DIR>/perf_public_set.jsonl \
  --concurrency 32
```

### A/B speed sanity check
```bash
python3 benchmark/soar/run_soar_suite.py \
  --api-base http://127.0.0.1:30000 \
  --model-path /root/models/openbmb/MiniCPM-SALA \
  --speed-data-s1 <S1_JSONL> \
  --speed-data-s8 <S8_JSONL> \
  --speed-data-smax <SMAX_JSONL> \
  --num-prompts 64
```

## 7) Results Summary
| Metric | Baseline | New | Delta |
|---|---:|---:|---:|
| OOM occurrence during eval |  |  |  |
| Accuracy / overall_accuracy |  |  |  |
| S1 benchmark_duration (s) |  |  |  |
| S8 benchmark_duration (s) |  |  |  |
| S∞ benchmark_duration (s) |  |  |  |

## 8) Risk Assessment
- Accuracy risk: very low (equivalent counting semantics).
- Stability risk: low.
- Performance risk: low; any change expected to be within noise unless memory pressure was limiting throughput.

## 9) Rollback Instructions
1. Revert the single line change in `python/sglang/srt/layers/attention/minicpm_backend.py`.
2. Re-run eval concurrency ramp (8/16/32) to confirm behavior restored.

## 10) Next-Step Suggestions
- If OOM still appears, next low-risk option is scheduler-side guard tuning (`--prefill-max-requests`, `--max-running-requests`) before deeper kernel changes.
