# CHANGE_0060 MiniCPM Lightning Fused QK-Norm-RoPE

## Background and Motivation

The current submission keeps `--force-dense-minicpm` enabled because the sparse MiniCPM path has unresolved stability issues. Under this runtime mode, the active model-specific hot path is the MiniCPM-SALA lightning attention stack rather than the sparse `minicpm4` path.

The shipped MiniCPM-SALA config uses `qk_norm=true`, `lightning_use_rope=true`, and a 75% lightning / 25% sparse layer mix. With `--force-dense-minicpm`, the sparse path is disabled as a workaround, so a lightning-only fusion feature becomes a good low-risk next step.

This repository already contains an existing SRT `fused_qk_norm_rope` kernel path used by another model family. This iteration reuses that existing kernel contract for the MiniCPM lightning mixer instead of introducing a brand-new custom kernel first.

## Rule-Compliance Statement

This change remains compliant with the latest SOAR competition and toolkit pages checked on 2026-03-29.

- It stays within the allowed runtime-optimization scope: operator fusion and inference-path optimization.
- It does not replace the MiniCPM-SALA base model.
- It preserves the submission contract based on `prepare_env.sh` and `prepare_model.sh`.
- It does not modify benchmark concurrency rules or re-enable forbidden prefix-cache behavior.
- It keeps the existing GPTQ W4A16 + Marlin + FP8 KV cache path intact.

## Detailed Implementation Plan

Before change:

1. Reuse the existing fused `qk_norm_rope` kernel rather than writing a new MiniCPM-only kernel first.
2. Limit the feature to the lightning mixer path used in force-dense mode.
3. Keep a conservative fallback to the original unfused MiniCPM lightning implementation.
4. Enable the feature through an existing server arg, but allow rollback with one environment variable.

## Actual Code Changes

Changed files:

- `python/sglang/srt/models/minicpm.py`
- `benchmark/soar/demo_sala/prepare_env.sh`

What changed:

1. Added a MiniCPM-local `_compute_yarn_parameters()` helper to provide the fused kernel with the same RoPE scaling parameters expected by existing SRT fused paths.
2. Imported the existing `sgl_kernel.fused_qk_norm_rope` operator only on CUDA.
3. Added compatibility guards in `MiniCPMLightningMixer`:
   - CUDA only
   - `qk_norm` enabled
   - RoPE enabled
   - supported `head_dim` values (`64`, `128`, `256`)
   - reject `MRotaryEmbedding`
4. Added `_apply_qk_norm_rope()` to `MiniCPMLightningMixer`.
5. Replaced the old explicit lightning sequence:
   - split q/k/v
   - `q_norm`
   - `k_norm`
   - RoPE
   with a fused-or-fallback helper call.
6. Wired the submission default to enable the feature with:
   - `SOAR_ENABLE_FUSED_QK_NORM_ROPE=1`
   - `--enable-fused-qk-norm-rope`
7. Logged the new environment toggle in `prepare_env.sh`.

## Design Notes

### Why only the lightning path

MiniCPM-SALA sparse `minicpm4` layers use `attn_use_rope=false` in the shipped model config, while lightning layers use `lightning_use_rope=true` and `qk_norm=true`. The fused `qk_norm_rope` operator matches the lightning pattern, not the sparse path.

### Why reuse the existing fused op first

The repository already has a functioning fused kernel path used by another SRT model. Reusing it minimizes scope and reduces kernel-maintenance risk. A MiniCPM-specific custom kernel is only justified if this reuse path proves incompatible or too fragile.

### Why keep a strict fallback

The original MiniCPM lightning implementation remains the correctness reference. Falling back automatically when the tensor contract is not suitable is safer than forcing the fused path on every request.

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
export SOAR_ENABLE_FUSED_QK_NORM_ROPE=0
```

## Result Summary Table

| Item | Before | After |
| --- | --- | --- |
| MiniCPM lightning Q/K norm + RoPE | separate ops | fused fast path with guarded fallback |
| Kernel source | unfused MiniCPM-only Python path | reused existing SRT fused kernel |
| Force-dense compatibility | N/A | targeted explicitly |
| Rollback | code change required | env toggle in `prepare_env.sh` |

## Rollback Instructions

If the fused lightning path is unstable or fails to improve speed:

1. set `SOAR_ENABLE_FUSED_QK_NORM_ROPE=0`
2. rerun the same correctness and speed checks
3. if needed, keep the code but submit with the feature disabled while evaluating whether a MiniCPM-specific fused op is justified

## Next-Step Suggestions

1. Validate correctness first because the public correctness gate remains the primary blocker.
2. If correctness holds and speed improves, keep this feature as the new force-dense lightning baseline.
3. If speed gain is small, profile whether the remaining bottleneck is inside the SimpleGLA backend or the quantized Marlin path before deciding between `CHANGE_0052` follow-up work and further `sgl-kernel` tuning.