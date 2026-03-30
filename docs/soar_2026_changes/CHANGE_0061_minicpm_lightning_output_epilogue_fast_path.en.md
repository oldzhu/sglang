# CHANGE_0061 MiniCPM Lightning Output Epilogue Fast Path

## Background and Motivation

The current local baseline improved materially after the lightning-only fused `qk_norm_rope` path landed, but the MiniCPM lightning mixer still performs a repeated post-attention epilogue in Python/Torch tensor ops:

1. optional output RMSNorm
2. `z_proj(hidden_states)`
3. `sigmoid(z)`
4. elementwise gating multiply
5. `o_proj`

Under the current `--force-dense-minicpm` runtime, lightning layers dominate the active model-specific path. That makes even modest reductions in epilogue overhead worth testing locally before moving to higher-risk kernel work.

This iteration keeps the exact same math and limits itself to a low-risk fast path that reduces temporary tensor overhead in the lightning output gate epilogue.

## Rule-Compliance Statement

This change remains compliant with the latest SOAR competition and toolkit pages checked on 2026-03-30.

- It is a pure runtime-path optimization.
- It does not replace the MiniCPM-SALA base model.
- It does not change submission concurrency rules or prefix-cache behavior.
- It preserves the existing GPTQ + Marlin + FP8 KV cache path.
- It keeps rollback simple through one environment variable.

## Detailed Implementation Plan

Before change:

1. Keep the lightning output math unchanged.
2. Isolate the post-attention epilogue into a helper.
3. Add a guarded fast path only for inference-style execution.
4. Use an environment variable so the feature can be disabled without code revert.

## Actual Code Changes

Changed files:

- `python/sglang/srt/models/minicpm.py`
- `benchmark/soar/demo_sala/prepare_env.sh`

What changed:

1. Added `SGLANG_MINICPM_LIGHTNING_FAST_OUTPUT_GATE` as a local runtime toggle.
2. Added `_apply_output_epilogue()` to `MiniCPMLightningMixer`.
3. Moved the lightning post-attention tail into that helper.
4. Added a guarded fast path for the output gate sequence:
   - only when `SGLANG_MINICPM_LIGHTNING_FAST_OUTPUT_GATE=1`
   - only when autograd is disabled
   - only when `z_proj` output dtype matches the attention output dtype
5. In the fast path:
   - `z.sigmoid_()` runs in-place
   - `attn_output.mul_(z)` applies the gate in-place
6. If any guard is not satisfied, the code falls back to the original non-inplace behavior.
7. Logged the new environment variable in `prepare_env.sh`.

## Design Notes

### Why this is low risk

The math order is unchanged:

1. `o_norm`
2. `z_proj`
3. `sigmoid`
4. gate multiply
5. `o_proj`

The only optimization is using in-place tensor ops when the tensor contract is safe.

### Why this can still help speed

This feature does not remove either of the two linear layers in the epilogue, so it is not expected to be a large win. The target is smaller but repeated overhead:

- one less temporary sigmoid output tensor
- one less out-of-place gate-multiply result tensor
- reduced memory traffic in a hot path repeated across many lightning layers

### Why use a toggle

The gain may be modest and hardware-dependent. A dedicated env toggle makes A/B measurement straightforward on the current local baseline.

## Validation Commands

Python syntax:

```bash
python3 -m py_compile python/sglang/srt/models/minicpm.py
```

Shell syntax:

```bash
bash -n benchmark/soar/demo_sala/prepare_env.sh
```

Correctness validation:

```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_DIR> \
  --data_path <DATA_DIR>/perf_public_set.jsonl \
  --concurrency 32
```

Speed validation:

```bash
bash SOAR/bench_serving.sh http://127.0.0.1:30000
```

Rollback toggle:

```bash
export SGLANG_MINICPM_LIGHTNING_FAST_OUTPUT_GATE=0
```

## Result Summary Table

| Item | Before | After |
| --- | --- | --- |
| Lightning output gate path | generic out-of-place sigmoid + multiply | guarded in-place fast path with fallback |
| Math order | unchanged | unchanged |
| Rollback | code edit required | env toggle |
| Expected gain | baseline | small repeated epilogue overhead reduction |

## Rollback Instructions

If the fast epilogue path is unstable or does not help:

1. set `SGLANG_MINICPM_LIGHTNING_FAST_OUTPUT_GATE=0`
2. rerun correctness and speed checks
3. keep the helper structure if useful, but leave the fast path disabled for later A/B work

## Next-Step Suggestions

1. Measure this against the new local baseline first, especially S1.
2. If the gain is too small, move next to deeper Marlin-side epilogue fusion or output-projection layout tuning.
3. If the gain is measurable and correctness is stable, keep this enabled as part of the new local baseline before further kernel work.