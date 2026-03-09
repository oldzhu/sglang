# CHANGE_0030_005: GPTQ Official Dynamic Exclusion Implementation

## Background and Motivation
CHANGE_0030_004 research confirmed that GPTQModel 5.7.0 exposes an official per-module exclusion mechanism through `QuantizeConfig.dynamic`.

fcloud evidence showed:

- `module_tree` still contains `self_attn.o_gate`
- `full_layer_modules` still contains `self_attn.o_gate`
- `simple_layer_modules` excludes `self_attn.o_gate`

This means the correct fix path is to drive GPTQ's official filtering path, not mutate internal trees.

## Rule-Compliance Statement (SOAR)
This change remains compliant because it:

- only modifies offline preprocessing behavior
- does not change evaluation-time behavior
- preserves the official `prepare_model.sh --input/--output` contract
- keeps correctness and speed validation as required follow-up work

## Actual Code Changes
Updated file:

- `benchmark/soar/demo_sala/preprocess_model.py`

Implemented changes:

1. Replaced heuristic module-tree sanitization path with official dynamic control
- `QuantizeConfig` is now created with `dynamic` negative-match rules when layer-aware mode is enabled
- exclusion uses regex rules such as:
  - `-:.*self_attn\.o_gate.*`

2. Added `_build_dynamic_rules(...)`
- builds negative exclusion regex rules from `exclude_modules`

3. Preserved GPTQ API compatibility wrappers
- `GPTQModel.load(...)` still tolerates optional kwarg differences
- `model.quantize(...)` still tolerates optional kwarg differences

4. Added module-resolution debug logs
- prints `dynamic_rules`
- prints `simple_layer_modules(...)` from the loaded GPTQ model to confirm official filtering took effect
- retry path prints the same information

5. Removed reliance on internal module-tree mutation as the intended fix direction
- tree mutation was not the authoritative control path in GPTQModel 5.7.0

## Risk to Accuracy/Stability
Tradeoffs:

- `self_attn.o_gate` remains globally excluded in the conservative path
- regex specificity may need one small adjustment if GPTQModel's internal module names differ in some runs

Why this is lower risk than prior attempts:

- it uses GPTQModel's documented public control path
- it matches observed `simple_layer_modules(...)` behavior from fcloud

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

3. Success indicators
- startup log prints `dynamic_rules`
- startup log prints `simple_layer_modules=` without `self_attn.o_gate`
- quantization proceeds past the prior `o_gate` mismatch point

## Result Summary Table
Status: code implemented locally; fcloud validation pending.

| Metric | Before CHANGE_0030_005 | After CHANGE_0030_005 |
|---|---:|---:|
| Official module exclusion path used | No | Yes |
| GPTQ preprocess completion | Fails on `o_gate` mismatch | TBD |
| Correctness score | TBD | TBD |
| Speed S1 | TBD | TBD |
| Speed S8 | TBD | TBD |
| Speed Smax | TBD | TBD |

## Rollback Instructions
1. Revert the CHANGE_0030_005 commit.
2. Or temporarily return to `SOAR_QUANT_MODE=copy`.
3. If needed, keep the GPTQ API compatibility helpers while reverting the `dynamic` rule construction.

## Next-Step Suggestions
1. Run the 16-sample fcloud smoke first.
2. Confirm `simple_layer_modules` excludes `o_gate` in the new log.
3. If smoke passes, continue to the 128-sample run and then correctness/speed validation.
