# CHANGE_0065 GPTQ Mixed Precision With W8 Attention Output Projection

## Background and Motivation

The latest calibration-only experiment increased `qa`, `mcq`, and `cwe` coverage to all 90 available public records across those three tasks, but correctness still remained around the same band. That result materially weakens the hypothesis that the remaining gap is mainly caused by calibration sampling variance.

At this point, the more plausible explanation is structural quantization sensitivity in a subset of modules. The goal of this iteration is therefore not to broaden calibration again, but to add one narrow mixed-precision experiment that is compatible with the current GPTQ + Marlin serving path.

The selected target is `self_attn.o_proj` at 8-bit, while keeping the existing GPTQ W4A16 baseline for the rest of the quantized attention and MLP projections. This target is intentionally conservative:

- it is outside the merged `qkv_proj` loader boundary that previously caused compatibility risk
- it affects both short and long-context behavior after attention routing is formed
- it stays within the current `gptq_marlin` format support, which in this code path supports 4-bit and 8-bit symmetric formats

## Rule-Compliance Statement

This change remains compliant with the latest SOAR competition and toolkit pages checked on 2026-04-02, including the toolkit `技术路径指引` and `提交说明` sections.

- It only changes preprocessing quantization configuration.
- It preserves the official `prepare_env.sh` and `prepare_model.sh --input/--output` workflow.
- It does not replace the MiniCPM-SALA base model.
- It does not alter fixed concurrency settings, prefix-cache rules, or the serving contract.
- It keeps the runtime on the existing `gptq_marlin` + FP8 KV-cache route.

## Detailed Implementation Plan

Before change:

1. Keep the current GPTQ include/exclude module scope unchanged.
2. Extend the preprocess dynamic-rule builder so it can emit positive per-module overrides, not only negative skip rules.
3. Add one supported mixed-precision preset, `o_proj_w8`, that raises `self_attn.o_proj` to 8-bit with group size 128.
4. Keep the preset env-controlled so it can be disabled cleanly for A/B comparison without another code rollback.

## Actual Code Changes

Changed files:

- `benchmark/soar/demo_sala/preprocess_model.py`
- `benchmark/soar/demo_sala/prepare_env.sh`

What changed:

1. `preprocess_model.py` now builds positive GPTQ dynamic overrides in addition to exclusion rules.
2. Added support for `SOAR_GPTQ_MIXED_PRECISION_PRESET=o_proj_w8`.
3. Added env-controlled override values for `SOAR_GPTQ_O_PROJ_BITS` and `SOAR_GPTQ_O_PROJ_GROUP_SIZE`.
4. Enabled the `o_proj_w8` preset by default in `prepare_env.sh`.

## Design Notes

### Why choose `o_proj` first

`o_proj` is a safer mixed-precision target than Q, K, or V in this repository. MiniCPM runtime serves Q, K, and V through the merged `qkv_proj` boundary, which is where previous selective rollback work became structurally risky. `o_proj` is independent of that fused loader path.

### Why not use 3-bit MLP weights first

The current `gptq_marlin` runtime path in this codebase supports 4-bit and 8-bit symmetric formats, not a 3-bit Marlin serving format. Introducing 3-bit here would require a different runtime path or deeper kernel/runtime work, which is out of scope for this iteration.

### Why keep the preset env-controlled

This is still an experiment. Making it switchable with environment variables allows a clean A/B comparison between baseline W4A16 and the W8 `o_proj` variant without further source edits.

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

Disable the preset for baseline comparison:

```bash
export SOAR_GPTQ_MIXED_PRECISION_PRESET=off
```

## Result Summary Table

| Item | Before | After |
| --- | --- | --- |
| GPTQ dynamic overrides | exclude-only | exclude + positive overrides |
| Mixed-precision preset | none | `o_proj_w8` |
| Attention QKV format | W4 baseline | unchanged |
| `self_attn.o_proj` format | W4 baseline | W8 |
| Expected risk | low | low-to-moderate speed regression |
| Expected benefit | current 78-level plateau | target improved accuracy stability |

## Rollback Instructions

If the W8 `o_proj` experiment does not improve correctness enough, or the speed cost is too high:

1. set `SOAR_GPTQ_MIXED_PRECISION_PRESET=off`
2. rerun preprocess, correctness, and serving checks

If the feature should be fully removed from source later, delete the preset env variables and the positive override branch in `preprocess_model.py`.

## Next-Step Suggestions

1. Measure whether `mcq` improves first, because the latest results suggest the remaining loss is not only long-context routing noise.
2. If `o_proj_w8` helps but not enough, the next mixed-precision experiment should stay fused-group-safe and consider a narrow `down_proj` or layer-subset extension.
3. If it does not help, shift attention away from calibration and mixed precision toward runtime-side or architecture-aware accuracy hypotheses.