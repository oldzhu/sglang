# CHANGE_0071: Fix MiniCPM Sparse Triton int64 Guard Mismatch

## 1) Background and Motivation

- **Problem**: Test 2 (`MiniCPM-SALA-Copy` + sparse attention + bf16 KV cache) failed during CUDA graph capture with a Triton compilation error in `compress_k_complete_kernel_new`.
- **Observed error**: `AssertionError('Mismatched type for token_k_indices between then block (int64) and else block (int32)')`.
- **Impact**: Sparse decode preparation could not compile, so the non-FP8 diagnostic path was blocked before correctness testing.
- **Additional operational issue**: Manual Test 1 showed that omitting `source /root/submission_sim/prepare_env.sh` before launching the server can trigger CUDA OOM and produce misleading low-accuracy results.

## 2) Rule Compliance

- Re-checked official pages: `https://soar.openbmb.cn/competition` and `https://soar.openbmb.cn/toolkit`.
- This is a correctness and test-enablement fix. It does not alter model weights, evaluation concurrency, or use restricted techniques.
- The change only makes Triton branch types consistent and records the required fcloud test procedure.
- Risk: Low. The kernel already intended to use 64-bit indices in this path; the fix only removes a compilation-time type inconsistency.

## 3) Detailed Implementation Plan

### Fix A: Normalize Triton branch dtypes
1. In `compress_k_complete_kernel_new` and `compress_k_complete_kernel_new_padded`, replace `token_k_indices = 0` in the out-of-range branch.
2. Use `tl.zeros([], dtype=tl.int64)` so both branches produce the same dtype as the `tl.load(...).to(tl.int64)` branch.

### Fix B: Record mandatory fcloud test notes
1. Update `.github/copilot-instructions.md` to state that new fcloud server terminals must source `/root/submission_sim/prepare_env.sh` before launch.
2. Record that `eval_model_001.py` should use `/root/data/perf_public_set.jsonl` as the eval dataset.

## 4) Actual Code Changes

### File: `python/sglang/srt/layers/attention/minicpm_sparse_kernels.py`

- Replaced four `token_k_indices = 0` fallback assignments with:

```python
token_k_indices = tl.zeros([], dtype=tl.int64)
```

- Applied in both:
  - `compress_k_complete_kernel_new`
  - `compress_k_complete_kernel_new_padded`

### File: `.github/copilot-instructions.md`

- Added explicit fcloud testing notes:
  - always run `source /root/submission_sim/prepare_env.sh` before starting `sglang` in a new terminal
  - `eval_model_001.py` uses `--data_path /root/data/perf_public_set.jsonl`

## 5) Why This Fix Is Correct

- Triton requires the same variable to have the same dtype on all control-flow branches.
- The in-range branch already cast `token_k_indices` to `tl.int64` because later address arithmetic can exceed 32-bit range.
- The out-of-range branch setting a plain `0` made the variable `int32`, which caused compilation to fail before runtime.
- Using `tl.zeros([], dtype=tl.int64)` preserves the intended semantics while matching Triton type requirements.

## 6) Validation Commands

```bash
# Sync latest commit to fcloud
python3 scripts/fcloud/fcloud_workflow.py sync

# Launch Test 2 server
source /root/submission_sim/prepare_env.sh
python3 -m sglang.launch_server \
  --model-path /root/models/openbmb/MiniCPM-SALA-Copy \
  --trust-remote-code --disable-radix-cache \
  --attention-backend minicpm_flashinfer \
  --dense-as-sparse \
  --chunked-prefill-size 32768 --max-prefill-tokens 32768 \
  --prefill-max-requests 1 --max-running-requests 20 \
  --mem-fraction-static 0.84 --schedule-conservativeness 1.0 \
  --port 30000

# Accuracy evaluation
cd /root/data
python3 eval_model_001.py \
  --api_base http://127.0.0.1:30000 \
  --model_path openbmb/MiniCPM-SALA \
  --data_path /root/data/perf_public_set.jsonl \
  --concurrency 8
```

## 7) Result Summary Table

| Item | Before | After | Change | Notes |
|---|---|---|---|---|
| Test 2 startup | Triton compile failure | Can proceed to launch test | Fix | Needs user validation on fcloud |
| `token_k_indices` fallback dtype | int32 | int64 | Fix | Matches in-range branch |
| fcloud launch procedure | Implicit | Documented | Safety | Avoids CUDA OOM from missing env |
| eval dataset path | Easy to mis-specify | Documented | Safety | Uses `/root/data/perf_public_set.jsonl` |

## 8) Rollback Instructions

```bash
git revert <commit>
```

Or manually restore the four fallback assignments in `minicpm_sparse_kernels.py` back to `0` and remove the added fcloud notes from `.github/copilot-instructions.md`.

## 9) Next-Step Suggestions

1. Re-run Test 2 to confirm the Triton compile failure is gone.
2. If Test 2 still has low accuracy, continue root-cause analysis in sparse prefill/decode logic rather than quantization.
3. If Test 2 succeeds, compare bf16 KV vs FP8 KV accuracy to isolate the remaining loss source.