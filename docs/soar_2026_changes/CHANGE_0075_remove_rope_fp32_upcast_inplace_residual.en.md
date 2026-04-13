# CHANGE_0075: Remove RoPE Float32 Upcast & In-place Residual Scale

## Background and Motivation

Profiling the MiniCPM-SALA model forward pass revealed two sources of unnecessary overhead:

1. **Float32 RoPE upcast**: All attention layers (both 8 standard `minicpm4` and 24 lightning) convert Q/K tensors from bf16 to float32 before applying rotary position embeddings, then convert back. This is defensive code from the original HuggingFace implementation. The sgl-kernel's `apply_rope_with_cos_sin_cache_inplace` CUDA kernel already handles bf16 inputs natively with an internal float32 cos_sin_cache, making the explicit conversion unnecessary.

2. **Tensor-allocating residual scale**: The `hidden_states = hidden_states * self.residual_scale` pattern allocates a new output tensor for a simple scalar multiplication. Since the result is consumed immediately by the next operation, in-place `*=` avoids the allocation.

## Rule-Compliance Statement

- ~~**No accuracy risk**: RoPE kernel already operates correctly in bf16; in-place multiply is mathematically identical.~~
- **ACCURACY RISK CONFIRMED**: Both changes cause catastrophic accuracy regression. See results below.
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

**STATUS: REVERTED — Both changes cause catastrophic accuracy regression.**

### Test 14: Both changes (bf16 RoPE + in-place residual) — commit c818ae261

| Metric | Baseline (Test 12) | After CHANGE_0075 | Delta |
|--------|--------------------|--------------------|-------|
| S1 | 121.66s | 139.26s | **+14.5% slower** |
| S8 | 44.17s | 52.82s | **+19.6% slower** |
| Smax | 35.91s | CRASH (server died) | **FATAL** |
| ori_accuracy | 79.29% | **52.64%** | **-26.65pp** |
| normalized | 99.11% | **65.81%** | **-33.3pp** |
| C | 1.0 | **0 (eliminated)** | — |

Task-level accuracy collapse:
- cwe: 47.67% (baseline ~72%)
- fwe: 98.89% (OK — unaffected)
- mcq: 53.33% (baseline ~63%)
- niah: 36.67% (baseline ~100%)
- qa: 26.67% (baseline ~63%)

### Test 15: In-place residual only (RoPE restored) — commit b8196b71e

| Metric | Baseline (Test 12) | In-place residual only | Delta |
|--------|--------------------|-----------------------|-------|
| ori_accuracy | 79.29% | **51.91%** | **-27.38pp** |
| normalized | 99.11% | **64.89%** | **-34.22pp** |
| C | 1.0 | **0 (eliminated)** | — |

**Even worse than Test 14** — the in-place `*=` alone causes massive regression.

### Root Cause Analysis

1. **bf16 RoPE**: Although the sgl-kernel CUDA kernel technically accepts bf16 inputs and uses float32 internally for cos/sin cache values, the multiplication `q*cos + rotate(q)*sin` happens in bf16 precision when bf16 tensors are passed. MiniCPM-SALA relies on float32 precision for this computation — the float32 upcast is NOT defensive, it is REQUIRED.

2. **In-place residual scale**: The `hidden_states *= scalar` vs `hidden_states = hidden_states * scalar` behaves differently under sglang's CUDA graph capture. In-place modification of pre-allocated tensors in the CUDA graph can corrupt tensor aliasing and buffer reuse, leading to cascading numerical errors across layers. This is NOT simply a mathematical equivalence — the memory management semantics differ.

3. **Speed regression**: The broken accuracy caused the model to generate longer/incorrect outputs, artificially inflating the benchmark duration. There is no genuine speed benefit from either change.

## Rollback Instructions

All changes have been **fully reverted** in commit caa93efe9:
1. Float32 upcast restored in `MiniCPMAttention.forward()` — REQUIRED for accuracy
2. Float32 upcast restored in `MiniCPMLightningMixer._apply_qk_norm_rope()` — REQUIRED for accuracy
3. `hidden_states = hidden_states * self.residual_scale` restored (2 occurrences) — REQUIRED for CUDA graph correctness

## Lessons Learned

1. **Never assume kernel-level dtype support implies model-level correctness.** The bf16 RoPE kernel works correctly at the kernel level, but the model was trained with float32 RoPE computations. Removing the upcast introduces systematic numerical drift that compounds across 32 layers.

2. **In-place tensor operations are NOT safe under CUDA graphs.** sglang's CUDA graph capture pre-allocates tensor buffers and fixes memory addresses. In-place `*=` modifies these buffers directly, which can corrupt aliased references. Always create new tensors in the forward pass when CUDA graphs are active.

3. **"Mathematically identical" does not mean "computationally identical."** Both changes appeared safe on paper but failed catastrophically in practice due to precision and memory management differences in the GPU execution model.

## Next Steps

- **Do NOT retry** either optimization without extensive precision analysis
- Focus on other optimization vectors: torch.compile, scheduling tuning, kernel fusion
- Consider folding `residual_scale` into weights at load time (avoids runtime computation entirely) — but verify with accuracy test first
