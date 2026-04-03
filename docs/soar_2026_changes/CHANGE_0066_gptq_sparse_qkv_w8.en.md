# CHANGE_0066 GPTQ Mixed Precision With Sparse-Layer QKV W8

## Background and Motivation

The previous mixed-precision experiment promoted `self_attn.o_proj` to 8-bit while keeping the rest of the GPTQ path at the W4A16 baseline. That experiment improved some `mcq` behavior but did not recover `qa` enough to restore the overall score to the target band.

This result suggests the remaining loss is more likely tied to attention formation than to the post-attention output projection alone. At the same time, promoting QKV globally would impose a broader runtime cost than necessary.

MiniCPM-SALA already separates sparse/full-attention layers from lightning-attention layers through the model `mixer_types` configuration. That architecture split gives a more principled subset than hand-picking arbitrary layer IDs. This iteration therefore promotes `q_proj`, `k_proj`, and `v_proj` together to 8-bit only on sparse/full-attention layers, while leaving lightning layers and the rest of the quantized path unchanged.

## Rule-Compliance Statement

This change remains compliant with the latest SOAR competition and toolkit pages checked on 2026-04-03, including the toolkit `技术路径指引` and `提交说明` sections.

- It only changes preprocessing quantization configuration.
- It preserves the official `prepare_env.sh` and `prepare_model.sh --input/--output` workflow.
- It does not replace the MiniCPM-SALA base model.
- It does not alter fixed concurrency settings, prefix-cache rules, or the serving contract.
- It keeps the runtime on the existing `gptq_marlin` + FP8 KV-cache route.

## Detailed Implementation Plan

Before change:

1. Keep the current GPTQ include/exclude module scope unchanged.
2. Extend the mixed-precision preset logic so it can target layer-qualified module regexes, not only global module names.
3. Add one supported preset, `sparse_qkv_w8`, that promotes `self_attn.q_proj`, `self_attn.k_proj`, and `self_attn.v_proj` together to 8-bit only on sparse/full-attention layers.
4. Resolve sparse layer IDs automatically from `config.mixer_types`, while allowing an explicit env override for debugging or narrower follow-up tests.

## Actual Code Changes

Changed files:

- `benchmark/soar/demo_sala/preprocess_model.py`
- `benchmark/soar/demo_sala/prepare_env.sh`

What changed:

1. Added parsing support for optional integer layer-id lists from environment variables.
2. Added sparse-layer resolution from `config.mixer_types`.
3. Extended GPTQ dynamic-rule generation to support a new `sparse_qkv_w8` preset.
4. Added env controls for `SOAR_GPTQ_SPARSE_QKV_BITS`, `SOAR_GPTQ_SPARSE_QKV_GROUP_SIZE`, and optional `SOAR_GPTQ_SPARSE_LAYER_IDS`.
5. Enabled `sparse_qkv_w8` as the default mixed-precision preset in `prepare_env.sh`.

## Design Notes

### Why use all sparse layers instead of arbitrary layer IDs

The sparse/full-attention layers are the more likely place to recover `qa`-style routing quality, while lightning layers are more tightly connected to the optimized recurrent fast path. Using the architecture-defined sparse subset is more defensible than guessing manual layer IDs before collecting another round of evidence.

### Why promote Q, K, and V together

MiniCPM runtime serves Q, K, and V through a fused QKV boundary. Promoting the full QKV group together avoids the structural inconsistency that made earlier selective rollback ideas risky.

### Why keep an explicit layer-id override

If all sparse layers improve accuracy but cost too much speed, the next iteration may want to narrow the subset without rewriting code. The env override makes that follow-up cheaper.

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

Baseline comparison:

```bash
export SOAR_GPTQ_MIXED_PRECISION_PRESET=off
```

Manual subset override example:

```bash
export SOAR_GPTQ_SPARSE_LAYER_IDS=0,4,8,12
```

## Result Summary Table

| Item | Before | After |
| --- | --- | --- |
| Mixed-precision preset | `o_proj_w8` | `sparse_qkv_w8` |
| W8 attention target | `o_proj` only | fused Q + K + V on sparse layers |
| Layer selection | global module match | config-driven sparse-layer subset |
| Loader safety | safe | safe if QKV stays promoted together |
| Expected risk | low-to-moderate speed regression | moderate speed regression |
| Expected benefit | helped `mcq` more than `qa` | target better `qa` recovery |

## Rollback Instructions

If sparse-layer QKV W8 does not improve correctness enough, or the speed cost is too high:

1. set `SOAR_GPTQ_MIXED_PRECISION_PRESET=off`
2. rerun preprocess, correctness, and serving checks

If accuracy improves but speed cost is too high, keep the preset and narrow the affected sparse layers using `SOAR_GPTQ_SPARSE_LAYER_IDS`.

## Next-Step Suggestions

1. Compare `qa` first, especially the `len_4k_32k` and `len_32k_128k` buckets, because that is where the previous W8 `o_proj` experiment remained weak.
2. If sparse-layer QKV W8 helps accuracy but hurts speed too much, the next feature should reduce the sparse-layer subset rather than reverting immediately.
3. If it still does not recover accuracy, the next direction should move away from calibration and weight precision toward generation-behavior control or other architecture-aware hypotheses.