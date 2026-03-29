# CHANGE_0052 Force-Dense Lightning Runtime Optimization

## Background and Motivation

The current submission-style runtime keeps `--force-dense-minicpm` enabled, which disables the sparse MiniCPM path and leaves the lightning layers as the dominant model-specific runtime target. In this configuration, the highest-value first kernel/runtime feature is not sparse-topk optimization, but reducing overhead around the SimpleGLA lightning backend that still covers most decoder layers.

This iteration therefore focuses on one cohesive force-dense runtime feature only:

- improve lightning backend dispatch policy for 6000D-oriented serving
- reduce recurrent state gather/scatter overhead in the SimpleGLA path

The math and model outputs are intentionally left unchanged.

## Rule-Compliance Statement

This change was reviewed against the latest SOAR competition and toolkit pages before implementation.

- It stays inside the officially allowed runtime optimization scope: kernel/runtime scheduling, memory/KV read-write optimization, and inference backend tuning.
- It does not replace the MiniCPM-SALA base model.
- It does not depend on forbidden prefix-cache behavior.
- It does not alter the fixed evaluation concurrency model.
- It preserves the `prepare_env.sh` and `prepare_model.sh` submission contract.

## Detailed Implementation Plan

Before change:

1. Reconfirm that current force-dense serving disables sparse MiniCPM layers.
2. Trace the lightning path through `MiniCPMLightningMixer` into `SimpleGLAAttnBackend`.
3. Keep the optimization tightly scoped to the SimpleGLA boundary that is controlled inside this repo.
4. Use environment-variable tuning rather than broad server-arg plumbing so rollback remains simple.

## Actual Code Changes

Changed files:

- `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- `benchmark/soar/demo_sala/prepare_env.sh`

Runtime changes in `hybrid_linear_attn_backend.py`:

1. Added force-dense-aware recurrent-threshold tuning:
   - new env var: `SGLANG_MINICPM_LIGHTNING_RECURRENT_THRESHOLD`
   - default is `128` under force-dense MiniCPM, instead of the previous hardcoded `64`
2. Added a fast state-IO path:
   - new env var: `SGLANG_MINICPM_LIGHTNING_FAST_STATE_IO`
   - uses `torch.index_select` for state gather and `index_copy_` for state writeback
3. Cached lightning layer-to-cache mapping locally inside the backend to avoid repeated dictionary lookups on the hot path.
4. Centralized the mode-selection logic into a helper so the recurrent/chunk policy is explicit and easy to tune.
5. Removed duplicate state-index resolution and duplicate layer-cache lookup inside the forward path.

Environment defaults in `prepare_env.sh`:

1. `SGLANG_MINICPM_LIGHTNING_FAST_STATE_IO=1`
2. `SGLANG_MINICPM_LIGHTNING_RECURRENT_THRESHOLD=128`
3. Added log output for both variables so fcloud runs show the effective tuning.

## Design Notes

### Why this targets force-dense runtime first

With `--force-dense-minicpm`, sparse MiniCPM kernels are not the current bottleneck candidate. The remaining model-specific runtime path with the highest expected ROI is the lightning backend used by most layers.

### Why change the recurrent threshold

The old lightning path used a fixed `seq_len < 64` decision boundary for choosing recurrent versus chunk mode. That threshold is generic rather than hardware-tuned. For the current 6000D-oriented force-dense setup, this iteration promotes a slightly more aggressive recurrent fast path for medium-length lightning extends while keeping chunk mode for larger inputs.

### Why state IO is part of the feature

Every lightning layer repeatedly gathers temporal state before the kernel call and writes it back after the call. This is repeated across most layers and every decode step. Reducing overhead in this gather/scatter path is a better first force-dense feature than deeper changes to dense flashinfer attention.

## Validation Commands

Syntax check:

```bash
python3 -m py_compile python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
```

Correctness check:

```bash
python3 benchmark/soar/demo_sala/eval_model_001.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_DIR> \
  --data_path benchmark/soar/demo_sala/perf_public_set.jsonl \
  --concurrency 32
```

Serving benchmark:

```bash
bash SOAR/bench_serving.sh http://127.0.0.1:30000
```

Rollback comparison:

```bash
export SGLANG_MINICPM_LIGHTNING_FAST_STATE_IO=0
export SGLANG_MINICPM_LIGHTNING_RECURRENT_THRESHOLD=64
```

## Result Summary Table

| Item | Before | After |
| --- | --- | --- |
| Lightning recurrent threshold | hardcoded `64` | env-tunable, default `128` in force-dense setup |
| State gather path | advanced indexing + contiguous copy | `index_select` fast path |
| State writeback path | advanced indexing assignment | `index_copy_` fast path |
| Rollback | code edit required | env-only rollback |
| Speed result | pending validation | pending validation |

## Rollback Instructions

If the new lightning tuning is unstable or not beneficial:

1. set `SGLANG_MINICPM_LIGHTNING_FAST_STATE_IO=0`
2. set `SGLANG_MINICPM_LIGHTNING_RECURRENT_THRESHOLD=64`
3. rerun the same serving and correctness validation commands

No model re-quantization or packaging layout change is required for rollback.

## Next-Step Suggestions

1. Measure S1 and S8 first, because this feature should help decode-heavy paths most directly.
2. If the gain is real and correctness remains stable, the next feature should target the external SimpleGLA kernel boundary more aggressively.
3. Only return to sparse-path optimization after the current sparse crash is fixed.