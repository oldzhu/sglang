# CHANGE_0068 GPTQ Mixed Precision With Sparse-QKV W8 Plus O-Proj W8

## Background and Motivation

The previous two mixed-precision branches produced incomplete gains when tested separately.

- `o_proj_w8` showed some positive signal on `mcq`, but it did not recover `qa` enough to restore the overall score.
- `sparse_qkv_w8` was made load-compatible through the `CHANGE_0067` loader-alignment correction, but by itself it still did not provide a convincing accuracy recovery.

This iteration combines those two already-vetted hypotheses into one controlled preset: sparse/full-attention-layer QKV at 8-bit together with global `self_attn.o_proj` at 8-bit. The goal is to test whether the remaining loss is split across both attention formation and post-attention output projection, rather than being dominated by only one of them.

## Rule-Compliance Statement

This change remains compliant with the latest SOAR competition and toolkit pages checked on 2026-04-03, including the toolkit `技术路径指引` and `提交说明` sections.

- It only changes preprocessing quantization configuration.
- It preserves the official `prepare_env.sh` and `prepare_model.sh --input/--output` workflow.
- It does not replace the MiniCPM-SALA base model.
- It does not alter fixed concurrency settings, prefix-cache rules, or the serving contract.
- It keeps the runtime on the existing `gptq_marlin` + FP8 KV-cache route.

## Detailed Implementation Plan

Before change:

1. Keep the existing `o_proj_w8` and `sparse_qkv_w8` behaviors intact.
2. Add one new combined preset, `sparse_qkv_w8_o_proj_w8`.
3. Compose the existing sparse-layer QKV W8 overrides with the global `self_attn.o_proj` W8 override inside the same dynamic-rule builder.
4. Make the combined preset the default for the next experiment so it can be tested directly on fcloud without extra env edits.

## Actual Code Changes

Changed files:

- `benchmark/soar/demo_sala/preprocess_model.py`
- `benchmark/soar/demo_sala/prepare_env.sh`

What changed:

1. Refactored mixed-precision preset handling so presets can enable one or both override groups.
2. Added support for `SOAR_GPTQ_MIXED_PRECISION_PRESET=sparse_qkv_w8_o_proj_w8`.
3. Kept the existing `o_proj_w8`, `sparse_qkv_w8`, and `off` values valid.
4. Switched the default mixed-precision preset in `prepare_env.sh` to the new combined preset.

## Design Notes

### Why a combined preset instead of comma-separated preset names

The preset parser is designed to accept one controlled feature name at a time. A dedicated combined preset is simpler to validate, document, and compare than free-form comma-separated composition.

### Why combine these two branches before pivoting away

This is the last clean mixed-precision composition before moving to a different hypothesis class. If this combined preset still does not recover accuracy enough, the branch has much stronger evidence of exhaustion.

### Why keep the older presets available

Retaining the individual presets preserves clean A/B comparisons against the combined branch without forcing another code rollback.

## Validation Commands

Shell syntax:

```bash
bash -n benchmark/soar/demo_sala/prepare_env.sh
```

Python syntax:

```bash
python3 -m py_compile benchmark/soar/demo_sala/preprocess_model.py
```

Preprocess validation:

```bash
bash benchmark/soar/demo_sala/prepare_model.sh --input <RAW_MODEL_DIR> --output <OUTPUT_MODEL_DIR>
```

Correctness validation:

```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_DIR> \
  --data_path <DATA_DIR>/perf_public_set.jsonl \
  --concurrency 32
```

Serving benchmark:

```bash
bash SOAR/bench_serving.sh http://127.0.0.1:30000
```

Preset comparisons:

```bash
export SOAR_GPTQ_MIXED_PRECISION_PRESET=off
export SOAR_GPTQ_MIXED_PRECISION_PRESET=o_proj_w8
export SOAR_GPTQ_MIXED_PRECISION_PRESET=sparse_qkv_w8
export SOAR_GPTQ_MIXED_PRECISION_PRESET=sparse_qkv_w8_o_proj_w8
```

## Result Summary Table

| Item | Before | After |
| --- | --- | --- |
| Mixed-precision preset | single-branch only | single-branch or combined |
| W8 attention target | sparse QKV or `o_proj` | sparse QKV plus `o_proj` |
| Default preset | `sparse_qkv_w8` | `sparse_qkv_w8_o_proj_w8` |
| Expected risk | moderate | moderate-to-high speed regression |
| Expected benefit | partial or unclear recovery | last combined accuracy-recovery test in this branch |

## Rollback Instructions

If the combined preset does not improve correctness enough, or the speed cost is too high:

1. set `SOAR_GPTQ_MIXED_PRECISION_PRESET=off`
2. rerun preprocess, correctness, and serving checks

If only one branch looks promising, switch back to `o_proj_w8` or `sparse_qkv_w8` without changing code.

## Next-Step Suggestions

1. Compare aggregate score, but prioritize `qa` and `mcq` because the combined preset is meant to join the strongest signal from each earlier branch.
2. If this still fails, stop extending the mixed-precision branch and move to a different hypothesis class.
3. If it helps accuracy but hurts speed too much, narrow sparse layers first before introducing any new W8 targets.