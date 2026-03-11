# CHANGE_0030_003: GPTQ Conservative Unblocker via Module-Tree Sanitization

## Background and Motivation
After CHANGE_0030_002, fcloud logs showed the failure was not coming from the tentative `QuantizeConfig` fields, but from the loaded GPTQ model instance itself.

Observed evidence:

- GPTQ AutoCompat inferred a shared `module_tree` from `layer0`
- that tree contained `self_attn.o_gate`
- later lightning-style layers do not always contain `o_gate`
- quantization therefore failed with:
  - `ValueError: layer module item self_attn.o_gate not found in model`

This continuation adopts the conservative unblocker first.

## Decision Recorded From Review
Two review questions were resolved as follows:

1. Should `o_gate` be removed only for lightning layers or globally?
- For this first unblocker, remove it globally from the GPTQ quantization target tree.
- This does **not** remove `o_gate` from the model itself.
- It only prevents GPTQ from trying to quantize/calibrate that module.

2. Does that affect calibration?
- Yes, but only for `o_gate`.
- `o_gate` will no longer contribute GPTQ calibration statistics.
- Other target modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) still participate normally.

Why this choice was accepted:

- it is the lowest-risk unblocker
- it preserves quantization of the stable cross-layer modules
- it avoids fighting deeper GPTQ internals before we have a successful end-to-end path

## Rule-Compliance Statement (SOAR)
This change remains compliant because it:

- only modifies offline preprocessing
- does not alter official evaluation-time behavior
- preserves the `prepare_model.sh --input/--output` submission contract
- still requires correctness and speed validation after quant-prep succeeds

## Actual Code Changes
Updated file:

- `benchmark/soar/demo_sala/preprocess_model.py`

Implemented logic:

1. Added `_sanitize_module_tree_node(...)`
- recursively walks lists, tuples, and dicts
- removes banned leaf module names such as `o_gate`
- prunes empty containers after removal

2. Added `_sanitize_model_module_tree(...)`
- sanitizes `model.module_tree`
- sanitizes `model.module_tree_overrides` if present
- logs before/after tree values when changes occur

3. Applied sanitization after `GPTQModel.load(...)`
- the loaded model instance is now sanitized before the first `model.quantize(...)`
- retry path also sanitizes the reloaded model

## Risk to Accuracy/Stability
Tradeoffs:

- `o_gate` quantization is skipped globally in this conservative path
- some compression/speed opportunity may be left on the table for sparse layers where `o_gate` exists
- however, this is safer than forcing a mixed-layer plan that fails before quantization completes

Expected practical impact:

- small reduction in quant coverage
- much higher chance of successful preprocessing

## Validation Commands
1. Local syntax check
```bash
python3 -m py_compile benchmark/soar/demo_sala/preprocess_model.py
```

2. fcloud smoke run
```bash
cd benchmark/soar/demo_sala
SOAR_QUANT_MODE=gptq \
SOAR_TRUST_REMOTE_CODE=true \
SOAR_GPTQ_ATTN_IMPL=flash_attention_2 \
SOAR_GPTQ_LAYER_AWARE=1 \
SOAR_GPTQ_INCLUDE_MODULES=self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj \
SOAR_GPTQ_EXCLUDE_MODULES=self_attn.o_gate \
SOAR_GPTQ_CALIBRATION_FILE=/path/to/calib.jsonl \
SOAR_GPTQ_CALIBRATION_SAMPLES=16 \
SOAR_GPTQ_BATCH_SIZE=1 \
./prepare_model.sh --input /path/to/raw_model --output /path/to/quant_smoke 2>&1 | tee /path/to/quant_smoke.log
```

3. fcloud quality pass
```bash
cd benchmark/soar/demo_sala
SOAR_QUANT_MODE=gptq \
SOAR_GPTQ_CALIBRATION_FILE=/path/to/calib.jsonl \
SOAR_GPTQ_CALIBRATION_SAMPLES=128 \
SOAR_GPTQ_BATCH_SIZE=2 \
./prepare_model.sh --input /path/to/raw_model --output /path/to/quant_full 2>&1 | tee /path/to/quant_full.log
```

## Result Summary Table
Status: implementation complete locally; fcloud validation pending.

| Metric | Before CHANGE_0030_003 | After CHANGE_0030_003 |
|---|---:|---:|
| GPTQ preprocess completion | Failed on `self_attn.o_gate` mismatch | TBD |
| Correctness score | TBD | TBD |
| Speed S1 | TBD | TBD |
| Speed S8 | TBD | TBD |
| Speed Smax | TBD | TBD |

## Rollback Instructions
1. Revert the CHANGE_0030_003 commit.
2. Or temporarily revert to `SOAR_QUANT_MODE=copy`.
3. If needed, remove module-tree sanitization while keeping earlier compatibility helpers.

## Next-Step Suggestions
1. Run the 16-sample fcloud smoke and share the full log.
2. Confirm log lines show sanitized `module_tree` without `o_gate`.
3. If successful, continue with the 128-sample run and then correctness/speed validation.
