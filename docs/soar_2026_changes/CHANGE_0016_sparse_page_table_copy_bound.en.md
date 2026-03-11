# CHANGE_0016_sparse_page_table_copy_bound

## 1) Background & Motivation
- S8 benchmark run triggered a runtime error in MiniCPM backend:
  - tensor size mismatch while copying dense page table entries into sparse page table.
- Error pattern indicated `kv_len` can exceed one of target/source table widths in this branch.

## 2) SOAR Rule-Compliance Check
- Runtime robustness fix in inference code path.
- No forbidden evaluation behavior.
- No model replacement or external trick.

## 3) Plan Before Code Change
- Modify one location in `python/sglang/srt/layers/attention/minicpm_backend.py`.
- Bound copy length to valid capacity intersection.

## 4) Actual Code Change
- Replaced direct `:kv_len` assignment with bounded `copy_len`:
  - `copy_len = min(kv_len, sparse_page_table.shape[1], page_table.shape[1])`
- Applied `copy_len` to both dense-head-group copy assignments.

## 5) Why This Fix
- Prevents assignment shape mismatch when `kv_len` is larger than table width in either source or destination.
- Keeps behavior identical for valid in-range cases.

## 6) Validation Commands
```bash
# Relaunch and run S8 tier
python3 benchmark/soar/run_soar_suite.py \
  --api-base http://127.0.0.1:30000 \
  --model-path /root/models/openbmb/MiniCPM-SALA \
  --speed-data-s8 /root/soar_fast_data/speed_s8.jsonl
```

## 7) Risks
- Low. Copy is now safely clipped to tensor capacity.

## 8) Rollback
1. Revert this change in `minicpm_backend.py`.

## 9) Next-Step Suggestions
- If needed, add debug counters for clipped-copy occurrences in a separate change.
