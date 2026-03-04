# CHANGE_0017_sparse_empty_max_guard

## 1) Background & Motivation
- An early scheduler crash occurred in MiniCPM backend metadata initialization:
  - `RuntimeError: max(): Expected reduction dim to be specified for input.numel() == 0`
- Root cause: `seqlen_q_sparse_tensor.max()` called when sparse tensor is empty.

## 2) SOAR Rule-Compliance Check
- Runtime robustness fix only.
- No model/kernel algorithm changes beyond safe edge-case handling.

## 3) Plan Before Code Change
- Modify one location in `python/sglang/srt/layers/attention/minicpm_backend.py`.
- Guard empty tensor before calling `max()`.

## 4) Actual Code Change
- Added conditional:
  - if `seqlen_q_sparse_tensor.numel() == 0` -> `metadata.max_seqlen_q_adjusted = 0`
  - else keep original `max().item() * heads_per_group` logic.

## 5) Validation Commands
```bash
# Relaunch and rerun the previously failing path
python3 benchmark/soar/run_soar_suite.py \
  --api-base http://127.0.0.1:30000 \
  --model-path /root/models/openbmb/MiniCPM-SALA \
  --speed-data-s8 /root/soar_fast_data/speed_s8.jsonl
```

## 6) Risks
- Very low. Only affects empty-sparse edge case.

## 7) Rollback
1. Revert the guard block in `minicpm_backend.py`.

## 8) Next-Step Suggestions
- If needed, add one debug counter for empty sparse-batch occurrences in future change.
