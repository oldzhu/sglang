# CHANGE_0070: Fix Sparse Attention CUDA Graph kv_indptr Plan Mismatch

## 1) Background and Motivation

- **Problem**: Sparse attention crashes under CUDA graph replay because the FlashInfer workspace plan (computed outside the graph via `begin_forward()`) uses a static `kv_indptr = [0, K, 2K, ...]` that does not match the actual cumulative sparse sequence lengths. When the captured graph runs `convert_sparse_page_table_to_flashinfer()`, it overwrites kv_indptr with the correct values, but the plan is already wrong → workspace out-of-bounds → crash.
- **Workaround that was in place**: `--force-dense-minicpm` flag bypasses sparse attention entirely by switching `minicpm_flashinfer` → standard `flashinfer` backend. This disables the model's native sparse routing, forcing full-context dense attention on every layer.
- **Impact**: Dense attention is significantly slower on long-context inputs (32K-160K tokens = 50% of SOAR speed eval distribution). Enabling sparse attention is estimated to yield 20-40% speedup.
- **Root cause**: Bug 4 in the sparse attention CUDA graph path — the replay function in `init_forward_metadata_replay_cuda_graph()` passes static `kv_indptr` to `begin_forward()` instead of computing the correct values from the current batch's `sparse_cache_seqlens`.

## 2) Rule Compliance

- Re-checked official pages: `https://soar.openbmb.cn/competition` and `https://soar.openbmb.cn/toolkit`.
- This is a correctness fix, not a new optimization technique. Sparse attention is the model's native architecture.
- No forbidden techniques used. No model weight changes. No prefix cache re-enabling.
- Risk: Low. This restores the intended execution path. Dense attention remains available via `--force-dense-minicpm` for rollback.

## 3) Detailed Implementation Plan

### Fix A: Correct kv_indptr in replay (`minicpm_backend.py`)
1. In `init_forward_metadata_replay_cuda_graph()`, replace the static kv_indptr with correct values from `forward_batch.sparse_cu_seqlens_k_cpu` (already available — it's `pad(cumsum(sparse_cache_seqlens), (1,0))`).
2. Pad tail entries (for `real_bs < bs` padding) with the last valid cumsum value so FlashInfer sees zero-length sequences.
3. Zero stale `sparse_cache_seqlens_int32` tail entries so the in-graph `convert_sparse_page_table_to_flashinfer()` computes matching kv_indptr.

### Fix B: Enable sparse attention (`prepare_env.sh`)
1. Remove `--force-dense-minicpm` from `SGLANG_SERVER_ARGS`.

## 4) Actual Code Changes

### File: `python/sglang/srt/layers/attention/minicpm_backend.py`

**Change 1** — Zero stale sparse_cache_seqlens tail (after line ~1733):
```python
metadata.sparse_cache_seqlens_int32[: 2 * real_bs].copy_(
    forward_batch.sparse_cache_seqlens_int32_cpu
)
# Zero stale tail entries so in-graph convert_sparse_page_table_to_flashinfer
# computes the same kv_indptr that replay passes to begin_forward.
metadata.sparse_cache_seqlens_int32[2 * real_bs :].fill_(0)
```

**Change 2** — Write correct kv_indptr in replay (replaces static pattern):
```python
# Bug 4 fix: Write correct kv_indptr from actual sparse cache
# seqlens instead of the static [0, K, 2K, ...] pattern.
kv_indptr_view[: sparse_real_bs + 1].copy_(
    forward_batch.sparse_cu_seqlens_k_cpu[: sparse_real_bs + 1]
)
# Pad remaining entries so FlashInfer sees zero-length sequences
if sparse_real_bs < sparse_bs:
    kv_indptr_view[sparse_real_bs + 1 :].fill_(
        kv_indptr_view[sparse_real_bs].item()
    )
```

### File: `benchmark/soar/demo_sala/prepare_env.sh`

Removed `--force-dense-minicpm` from `SGLANG_SERVER_ARGS`.

## 5) Why This Fix Is Correct

The data flow for FlashInfer CUDA graph sparse attention:

1. **Replay** (outside graph): Write correct `kv_indptr` from `sparse_cu_seqlens_k_cpu` → `begin_forward()` computes workspace plan matching actual batch
2. **Graph execution**: `convert_sparse_page_table_to_flashinfer()` overwrites same buffer with same values (from `sparse_cache_seqlens_int32` which was set from same source) — redundant but harmless
3. **Graph execution**: `wrapper.forward()` uses the plan (correct) and reads buffers (correct) → no OOB

The static pattern `[0, K, 2K, ...]` assumed every batch entry always has exactly `num_sparse_topk_tokens` entries. In reality, `sparse_cache_seqlens` varies with sequence length and block alignment:
```python
sparse_cache_seqlens = where(
    seq_lens <= K, seq_lens,
    (topk-1)*block_size + seq_lens % block_size
)
```

## 6) Validation Commands

```bash
# Test without --force-dense-minicpm (sparse attention enabled)
# Correctness
python benchmark/soar/demo_sala/eval/eval.py --preset sparse_qkv_w8

# Speed
python benchmark/soar/demo_sala/eval/speed_eval.py
```

## 7) Result Summary

| Item | Baseline (dense) | New (sparse) | Change | Notes |
|---|---|---|---|---|
| --force-dense-minicpm | Yes | Removed | — | Sparse attention enabled |
| kv_indptr source | Static [0,K,2K,...] | From sparse_cu_seqlens_k_cpu | Fix | Matches in-graph computation |
| Stale seqlens zeroing | Not done | Added | Safety | Ensures plan/graph consistency |
| Expected speedup | — | 20-40% on long inputs | — | Pending user validation |

## 8) Rollback

Add `--force-dense-minicpm` back to `SGLANG_SERVER_ARGS` in `benchmark/soar/demo_sala/prepare_env.sh`:
```bash
--kv-cache-dtype fp8_e5m2 --force-dense-minicpm${FUSED_QK_NORM_ROPE_ARG}
```

Revert `minicpm_backend.py` changes via `git revert`.

## 9) Next Steps

1. Test correctness — accuracy must stay ≥99% for C=1.0
2. Test speed — measure s1, s8, smax improvements
3. If sparse attention crashes with a different error, investigate further (Bug 3: FP8 descaling in scratch kernel may need attention)
4. If stable, proceed to Path C.1 (fused scaled-residual+RMSNorm) for additional 3-8% speedup
