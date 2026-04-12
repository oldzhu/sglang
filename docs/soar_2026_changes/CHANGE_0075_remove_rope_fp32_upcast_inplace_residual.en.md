# CHANGE_0075: Remove RoPE Float32 Upcast & In-place Residual Scale

## Background and Motivation

Profiling the MiniCPM-SALA model forward pass revealed two sources of unnecessary overhead:

1. **Float32 RoPE upcast**: All attention layers (both 8 standard `minicpm4` and 24 lightning) convert Q/K tensors from bf16 to float32 before applying rotary position embeddings, then convert back. This is defensive code from the original HuggingFace implementation. The sgl-kernel's `apply_rope_with_cos_sin_cache_inplace` CUDA kernel already handles bf16 inputs natively with an internal float32 cos_sin_cache, making the explicit conversion unnecessary.

2. **Tensor-allocating residual scale**: The `hidden_states = hidden_states * self.residual_scale` pattern allocates a new output tensor for a simple scalar multiplication. Since the result is consumed immediately by the next operation, in-place `*=` avoids the allocation.

## Rule-Compliance Statement

- **No accuracy risk**: RoPE kernel already operates correctly in bf16; in-place multiply is mathematically identical.
- **No submission constraint impact**: No model size change, no additional dependencies.
- **Reproducible**: Standard code optimization, fully deterministic.

## Implementation

### Change 1: Remove float32 upcast in `MiniCPMAttention.forward()`

**Before** (8 standard attention layers):
```python
if self.attn_use_rope:
    orig_dtype = q.dtype
    q, k = q.float(), k.float()
    q, k = self.rotary_emb(positions, q, k)
    q, k = q.to(orig_dtype), k.to(orig_dtype)
```

**After**:
```python
if self.attn_use_rope:
    q, k = self.rotary_emb(positions, q, k)
```

### Change 2: Remove float32 upcast in `MiniCPMLightningMixer._apply_qk_norm_rope()` (non-fused path)

Same pattern — removes the `.float()` / `.to(orig_dtype)` wrapping when the fused QK-norm-rope kernel is not active.

### Change 3: In-place residual scale in `MiniCPMDecoderLayer.forward()`

**Before** (all 32 layers, 2× per layer):
```python
hidden_states = hidden_states * self.residual_scale
```

**After**:
```python
hidden_states *= self.residual_scale
```

## Files Changed

- `python/sglang/srt/models/minicpm.py` — main model (all 3 changes)
- `benchmark/soar/demo_sala/sglang/python/sglang/srt/models/minicpm.py` — demo_sala copy (RoPE changes only; this copy doesn't have residual_scale)

## Validation Commands

```bash
# Accuracy test (must remain ≥99% normalized for C=1.0)
python3 scripts/fcloud/fcloud_workflow.py accuracy

# Speed benchmark (all 3 tiers)
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

## Expected Impact

- **RoPE fix**: Eliminates 4 dtype cast ops × 32 layers = 128 unnecessary ops per forward pass. Expected 2-5% speedup on prefill-heavy workloads where the standard attention layers are most expensive. For lightning layers, the fused path is already active when `--enable-fused-qk-norm-rope` is set, but the fallback path is also fixed.
- **In-place residual scale**: Eliminates 64 tensor allocations per forward pass (2 per layer × 32 layers). Expected 1-2% improvement from reduced memory allocation pressure.

## Result Summary

| Metric | Baseline | After CHANGE_0075 | Delta |
|--------|----------|--------------------|-------|
| S1 | TBD | TBD | TBD |
| S8 | TBD | TBD | TBD |
| Smax | TBD | TBD | TBD |
| Accuracy | TBD | TBD | TBD |

## Rollback Instructions

Revert the three code changes:
1. Restore float32 upcast in `MiniCPMAttention.forward()`
2. Restore float32 upcast in `MiniCPMLightningMixer._apply_qk_norm_rope()` non-fused path
3. Restore `hidden_states = hidden_states * self.residual_scale` (2 occurrences in `MiniCPMDecoderLayer.forward()`)

## Next Steps

- Test on fcloud with GPTQ+FP8+dense config to measure actual speed improvement
- If stable, combine with `--enable-torch-compile` for compound gains
- Consider folding `residual_scale` into `o_proj`/`down_proj` weights at load time for zero-cost scaling (higher complexity, deferred)
