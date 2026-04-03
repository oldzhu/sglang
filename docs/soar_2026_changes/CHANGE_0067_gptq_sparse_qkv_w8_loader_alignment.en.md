# CHANGE_0067 GPTQ Sparse-QKV W8 Loader Alignment

## Background and Motivation

CHANGE_0066 introduced a mixed-precision preset that promoted `q_proj`, `k_proj`, and `v_proj` to 8-bit on sparse/full-attention layers. The intent was sound, but the first end-to-end run failed during SGLang model loading with a fused-QKV shape mismatch:

```text
AssertionError: param_data.shape=torch.Size([512, 256]), loaded_weight.shape=torch.Size([1024, 256])
```

That failure showed the sparse-layer QKV W8 rule reached GPTQ preprocess at the unfused shard names, but did not reach the SGLang runtime at the fused `qkv_proj` name. As a result, the checkpoint shard was produced with W8 packing while the runtime destination parameter was still created with W4 packing.

This corrective iteration keeps the sparse-layer QKV W8 direction, but fixes the preprocess/runtime naming mismatch by emitting dynamic overrides for both the unfused shard names and the fused `qkv_proj` runtime name.

## Rule-Compliance Statement

This change remains compliant with the latest SOAR competition and toolkit pages checked on 2026-04-03, including the toolkit `技术路径指引` and `提交说明` sections.

- It only changes preprocessing quantization configuration.
- It preserves the official `prepare_env.sh` and `prepare_model.sh --input/--output` workflow.
- It does not replace the MiniCPM-SALA base model.
- It does not alter fixed concurrency settings, prefix-cache rules, or the serving contract.
- It keeps the runtime on the existing `gptq_marlin` + FP8 KV-cache route.

## Detailed Implementation Plan

Before change:

1. Keep the sparse-layer mixed-precision preset name and env interface unchanged.
2. Preserve shard-level overrides for GPTQ preprocess on `q_proj`, `k_proj`, and `v_proj`.
3. Add matching fused-name overrides on `qkv_proj` for the same sparse layers.
4. Leave the rest of the quantization scope unchanged so the fix only targets the load-shape mismatch.

## Actual Code Changes

Changed files:

- `benchmark/soar/demo_sala/preprocess_model.py`

What changed:

1. The `sparse_qkv_w8` preset still emits positive overrides for sparse-layer `self_attn.q_proj`, `self_attn.k_proj`, and `self_attn.v_proj`.
2. It now also emits matching positive overrides for sparse-layer `self_attn.qkv_proj`.
3. The bits and group-size values remain controlled by the existing sparse-QKV env variables.

## Design Notes

### Why emit both unfused and fused names

GPTQ preprocess operates on the unfused checkpoint module tree, while SGLang runtime loads the same logical weights into a fused QKV parameter. The same mixed-precision decision must therefore be visible under both naming schemes.

### Why the shape mismatch indicates bit-width disagreement

The reported destination shape `[512, 256]` versus source shape `[1024, 256]` is consistent with a packing-factor mismatch between 4-bit and 8-bit GPTQ Marlin weights. That means the runtime parameter and the checkpoint shard were not instantiated with the same quantization format.

### Why not abandon sparse-QKV W8 yet

This failure occurred before accuracy could be measured. It is a loader-alignment issue, not yet evidence that the sparse-layer QKV W8 hypothesis is ineffective.

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

## Result Summary Table

| Item | CHANGE_0066 | CHANGE_0067 |
| --- | --- | --- |
| Sparse-layer W8 target | Q + K + V shards | Q + K + V shards plus fused `qkv_proj` |
| Preprocess/runtime naming alignment | incomplete | aligned |
| Loader outcome | fused-QKV shape mismatch | target load-compatible sparse-QKV W8 |
| Accuracy signal | not measurable | measurable after load succeeds |

## Rollback Instructions

If the corrected sparse-QKV W8 path still does not load or hurts speed/accuracy too much:

1. set `SOAR_GPTQ_MIXED_PRECISION_PRESET=off`
2. rerun preprocess, correctness, and serving checks

If load succeeds but speed cost is high, keep the preset and narrow sparse layers using `SOAR_GPTQ_SPARSE_LAYER_IDS` rather than removing the correction.

## Next-Step Suggestions

1. Confirm first that the model now loads successfully; that is the gating check for this correction.
2. If loading succeeds, compare `qa` before focusing on aggregate score.
3. If accuracy improves but speed regresses too much, reduce the sparse-layer subset before changing the target modules again.