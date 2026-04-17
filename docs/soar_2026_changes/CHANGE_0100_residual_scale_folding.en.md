# CHANGE_0100: Residual Scale Folding

## Background and Motivation

MiniCPM-SALA uses three runtime scaling operations applied per-token at every forward pass:

1. **`residual_scale`** (`scale_depth / √num_hidden_layers` = 0.247487): Applied twice per decoder layer — after self-attention o_proj output and after MLP down_proj output. With 56 layers, this means **112 scalar multiply kernel launches** per forward pass.

2. **`scale_emb`** (12): Applied to embedding lookup output before entering the decoder stack. **1 multiply** per forward pass.

3. **`1/scale_width`** (`1/16.0`): Applied to the final hidden states before logits computation. **1 division** per forward pass.

Each of these is a trivial scalar-tensor operation, but each launches a separate CUDA kernel with its own overhead. Total: **114 unnecessary kernel launches per forward pass**.

Since these are all fixed scalars known at model load time, they can be folded (pre-multiplied) into the relevant weight matrices at load time, eliminating the runtime overhead entirely.

## Rule-Compliance Statement

- **Mathematically exact**: Scalar multiplication commutes with all linear operations (GEMM, Marlin dequantization, permutation). No approximation is introduced.
- **No accuracy impact**: The computation is identical; only the timing of when the scalar is applied changes.
- **No submission constraint impact**: No additional files, no size increase, no extra quantization time.
- **Compliant with SOAR rules**: No forbidden tricks; purely an algebraic optimization.

## Implementation Plan

### Folding Strategy

| Scale Factor | Where Folded | Target Attribute |
|---|---|---|
| `residual_scale` (0.247487) | o_proj Marlin dequant scales | `layer.self_attn.o_proj.scales` × 56 layers |
| `residual_scale` (0.247487) | down_proj Marlin dequant scales | `layer.mlp.down_proj.scales` × 56 layers |
| `scale_emb` (12) | embed_tokens weight | `model.embed_tokens.weight` |
| `1/scale_width` (1/16) | lm_head weight | `lm_head.weight` |

### GPTQ Marlin Compatibility

The Marlin kernel computes: `output = X @ (scales * Q_int)`. Multiplying `scales` by a scalar `s` gives `output' = X @ (s * scales * Q_int) = s * output`. This is exactly what the runtime `hidden_states *= residual_scale` was doing. The scalar multiply also commutes with `marlin_permute_scales()` since that is a pure permutation operation.

### Timing

The folding happens in `post_load_weights()`, which is called by the model loader **after** `process_weights_after_loading()` has already run on all quantized layers. This means the Marlin scales are fully materialized and permuted before we multiply the scalar in.

## Actual Code Changes

### File: `python/sglang/srt/models/minicpm.py`

1. **Added `logging` import and logger** at top of file.

2. **`MiniCPMDecoderLayer.__init__`**: Added `self.scaling_folded = False` flag.

3. **`MiniCPMDecoderLayer.forward()`**: Wrapped both `hidden_states *= self.residual_scale` lines with `if not self.scaling_folded:` guard.

4. **`MiniCPMModel.__init__`**: Added `self._scaling_folded = False` flag.

5. **`MiniCPMModel.forward()`**: Made `* self.config.scale_emb` conditional on `not self._scaling_folded`.

6. **`MiniCPMForCausalLM.__init__`**: Added `self._scaling_folded = False` flag.

7. **`MiniCPMForCausalLM.forward()`**: Made `/ self.scale_width` conditional on `not self._scaling_folded`.

8. **`MiniCPMForCausalLM._fold_scaling_factors()`**: New method that:
   - Multiplies `o_proj.scales` and `down_proj.scales` by `residual_scale` for all 56 layers
   - Multiplies `embed_tokens.weight` by `scale_emb`
   - Divides `lm_head.weight` by `scale_width`
   - Sets all `scaling_folded` flags to `True`

9. **`MiniCPMForCausalLM.post_load_weights()`**: New method (called by sglang model loader after weight processing) that invokes `_fold_scaling_factors()`.

### Backward Compatibility

- The `if not self.scaling_folded:` guards ensure the code works correctly even if `post_load_weights()` is not called (e.g., different model loaders). In that case, the original runtime scaling is used.
- The `input_embeds` external path in `MiniCPMForCausalLM.forward()` still applies `* self.config.scale_emb` at runtime, since external embeddings don't go through `embed_tokens`.

## Validation Commands

### Accuracy Test
```bash
python3 scripts/fcloud/fcloud_workflow.py accuracy
```

### Speed Test
```bash
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

### Expected Behavior
- Server log should show: `Folded scaling factors: residual_scale=0.247487, scale_emb=12, scale_width=16.0`
- Accuracy should be identical to baseline (within floating-point noise)
- Speed improvement: 1-3% (elimination of 114 kernel launches per forward pass)

## Result Summary

| Metric | Baseline (Test 20) | After M1 | Delta |
|---|---|---|---|
| S1 | 113.67s | TBD | TBD |
| S8 | 41.07s | TBD | TBD |
| Smax | 34.15s | TBD | TBD |
| Accuracy | 80.64% | TBD | TBD |
| Normalized | 100.80% | TBD | TBD |
| C | 1.0 | TBD | TBD |

## Rollback Instructions

Revert the single commit that implements this change:
```bash
git revert <commit-hash>
```

Or manually: remove the `post_load_weights`, `_fold_scaling_factors` methods, remove all `scaling_folded` flags and guards, restore the unconditional `*= residual_scale`, `* scale_emb`, `/ scale_width` lines.

## Next-Step Suggestions

- **K4**: Fuse RMSNorm + residual_scale into a single CUDA kernel (eliminates the norm + scale as two separate operations)
- **A1**: SimpleGLA state contiguity guarantee (5-8% decode improvement)
- **A3**: Fuse state I/O into FLA kernel (10-15% decode improvement)
