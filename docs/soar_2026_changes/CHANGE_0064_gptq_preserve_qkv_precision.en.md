# CHANGE_0064 GPTQ Preserve Full QKV Precision

## Background and Motivation

CHANGE_0063 attempted to preserve only `self_attn.k_proj` in higher precision while continuing to quantize `q_proj` and `v_proj`. That hypothesis was accuracy-motivated, but it turned out to be structurally incompatible with the MiniCPM runtime loader used by this repository.

MiniCPM rewrites `q_proj`, `k_proj`, and `v_proj` weights into a merged `qkv_proj` parameter during load. Preserving only K created a mixed checkpoint layout that quantized successfully but failed at serving time with a missing `qkv_proj.weight` path. This means the previous feature was not just ineffective or risky; it was not load-compatible.

This corrective iteration replaces the unsupported K-only policy with a loader-compatible policy: preserve the full QKV projection in higher precision and keep GPTQ focused on the remaining attention output and MLP projections.

## Rule-Compliance Statement

This change remains compliant with the latest SOAR competition and toolkit pages checked on 2026-03-31, including the toolkit `技术路径指引` and submission workflow expectations.

- It only changes preprocessing quantization scope.
- It preserves the official `prepare_env.sh` and `prepare_model.sh --input/--output` workflow.
- It does not change the serving contract, concurrency rules, or prefix-cache behavior.
- It keeps the current SGLang runtime path, GPTQ/Marlin backend, and FP8 KV-cache configuration for the rest of the quantized model.
- It removes a known incompatible quantization layout rather than introducing a forbidden runtime trick.

## Detailed Implementation Plan

Before change:

1. Remove the K-only preservation policy from default GPTQ include/exclude lists.
2. Replace it with a full-QKV preservation policy so all three projections follow the same format.
3. Apply the same policy in the module-mismatch retry path so retry behavior cannot silently reintroduce the incompatible layout.
4. Keep the rest of the quantization scope unchanged to isolate the fix to the QKV compatibility issue.

## Actual Code Changes

Changed files:

- `benchmark/soar/demo_sala/prepare_env.sh`
- `benchmark/soar/demo_sala/preprocess_model.py`

What changed:

1. Updated default `SOAR_GPTQ_INCLUDE_MODULES` to remove `self_attn.q_proj`, `self_attn.k_proj`, and `self_attn.v_proj` from GPTQ quantization.
2. Updated default `SOAR_GPTQ_EXCLUDE_MODULES` to explicitly exclude the full QKV trio.
3. Updated `preprocess_model.py` internal defaults so direct preprocess execution uses the same full-QKV preservation policy.
4. Updated the GPTQ module-mismatch retry path to preserve the same include/exclude behavior.

## Design Notes

### Why preserve full QKV instead of only K

In this MiniCPM implementation, Q, K, and V are not independent loader endpoints at serving time. They are stacked into `qkv_proj`. A partial rollback inside that merged structure produces an inconsistent checkpoint representation. Preserving full QKV is the narrowest change that restores structural consistency.

### Why this is still a useful accuracy experiment

The original accuracy hypothesis was that attention projection precision is one of the remaining weak points for `qa`, `mcq`, and `cwe`. Full-QKV preservation tests that broader hypothesis while staying compatible with the runtime.

### Why not also preserve `o_proj`

That would increase the rollback scope further and make it harder to attribute any accuracy or speed changes. This corrective iteration only fixes the broken QKV boundary.

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

Rollback to the pre-0064 scope:

```bash
export SOAR_GPTQ_INCLUDE_MODULES=self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj
export SOAR_GPTQ_EXCLUDE_MODULES=self_attn.o_gate,self_attn.z_proj
```

## Result Summary Table

| Item | CHANGE_0063 | CHANGE_0064 |
| --- | --- | --- |
| Preserved attention projections | K only | full Q + K + V |
| Loader compatibility | broken for merged `qkv_proj` | aligned with merged `qkv_proj` loader |
| Preprocess outcome | quantization may finish | quantization layout should remain loadable |
| Speed expectation | slight regression possible | higher regression than K-only, still bounded |
| Accuracy expectation | targeted K sensitivity | broader attention precision recovery |

## Rollback Instructions

If full-QKV preservation does not improve correctness enough, or the speed cost is too large:

1. restore `self_attn.q_proj`, `self_attn.k_proj`, and `self_attn.v_proj` to the GPTQ include list
2. remove those three modules from the GPTQ exclude list
3. rerun preprocess, model load, correctness, and serving checks

Do not roll back to the CHANGE_0063 K-only layout, because that layout is known to be incompatible with the current MiniCPM merged-QKV loader.

## Next-Step Suggestions

1. Confirm first that the quantized model now loads successfully in SGLang; that is the gating validation for this fix.
2. If loading succeeds, compare `qa`, `mcq`, and `cwe` against the current baseline before making any further calibration changes.
3. If the speed regression is too large, the next accuracy experiment should target a different boundary than QKV preservation rather than revisiting the broken K-only split.

## Status Update Before Adoption

This feature is also temporarily rolled back and not used in the active code path.

Why:

1. The investigation after CHANGE_0063 confirmed the direct cause of the serving failure: `KeyError: 'model.layers.0.self_attn.qkv_proj.weight'`.
2. Although full-QKV preservation is structurally safer than K-only preservation at the checkpoint level, it was not fully validated against the current SGLang MiniCPM runtime quantization-selection path.
3. Before spending more iterations on projection-preservation logic, the current plan is to return to the last known loadable GPTQ scope and run a cleaner calibration-only experiment.

Temporary rollback rationale:

- The team wants more time to review the merged-QKV loader boundary and the runtime quantization mapping in detail.
- The immediate next experiment is lower risk: keep `qa,mcq,cwe` task focus and raise calibration samples to 90 so all available records from those three tasks are used.
- This document remains as a design note for a possible future revisit, not as an adopted optimization feature.