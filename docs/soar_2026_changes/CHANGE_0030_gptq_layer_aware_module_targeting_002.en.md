# CHANGE_0030_002: GPTQ Layer-Aware Module Targeting (Implementation)

## Background and Motivation
This continuation records the actual code implementation for CHANGE_0030 after proposal approval and fcloud API introspection confirmed `gptqmodel==5.7.0` uses instance-level `quantize(...)` on `BaseQModel`.

The implementation goal remains the same:

- reduce mixed-layer module mismatch failures
- preserve submission compatibility
- keep the GPTQ path debuggable with explicit logs and controlled retry behavior

## Rule-Compliance Statement (SOAR)
This feature remains compliant because it:

- only changes offline preprocessing behavior
- does not introduce evaluation-side shortcuts
- preserves the official `prepare_model.sh --input/--output` contract
- keeps correctness verification as a required follow-up before any submission decision

## Actual Code Changes
Updated file:

- `benchmark/soar/demo_sala/preprocess_model.py`

Implemented changes:

1. Added compatibility helpers
- `_env_truthy(...)`
- `_parse_csv_env(...)`
- `_call_with_supported_kwargs(...)`
- `_apply_quant_module_controls(...)`
- `_is_module_mismatch_error(...)`

2. Added layer-aware GPTQ defaults
- `SOAR_GPTQ_LAYER_AWARE=1` by default
- include defaults:
  - `self_attn.q_proj`
  - `self_attn.k_proj`
  - `self_attn.v_proj`
  - `self_attn.o_proj`
  - `mlp.gate_proj`
  - `mlp.up_proj`
  - `mlp.down_proj`
- exclude default:
  - `self_attn.o_gate`

3. Added version-tolerant GPTQ dispatch
- `GPTQModel.load(...)` now drops unsupported optional kwargs such as `attn_implementation` when needed
- `model.quantize(...)` now drops unsupported optional kwargs such as `batch_size` when needed

4. Added mixed-layer retry path
- if quantization fails with a module-mismatch style error, the script rebuilds `QuantizeConfig`, reapplies conservative common targets, reloads the model, and retries once
- conservative retry still excludes `self_attn.o_gate`

5. Added diagnostic logging
- logs active include/exclude module lists
- logs which quant-config attributes were successfully applied
- logs retry reason and retry controls

## Risk to Accuracy/Stability
Known tradeoffs:

- If `gptqmodel` ignores all attempted quant-config module control attributes, this change may not fully eliminate the underlying mismatch in all versions.
- Conservative targeting may reduce quant coverage compared with a perfectly model-aware target planner.
- Runtime behavior still depends on real fcloud `gptqmodel` internals, so this needs fcloud validation.

## Validation Commands
1. Local syntax validation
```bash
python3 -m py_compile benchmark/soar/demo_sala/preprocess_model.py
```

2. fcloud smoke run
```bash
cd benchmark/soar/demo_sala
SOAR_QUANT_MODE=gptq SOAR_GPTQ_CALIBRATION_FILE=/path/to/calib.jsonl SOAR_GPTQ_CALIBRATION_SAMPLES=16 SOAR_GPTQ_BATCH_SIZE=1 ./prepare_model.sh --input /path/raw_model --output /path/quant_model
```

3. fcloud quality pass
```bash
cd benchmark/soar/demo_sala
SOAR_QUANT_MODE=gptq SOAR_GPTQ_CALIBRATION_FILE=/path/to/calib.jsonl SOAR_GPTQ_CALIBRATION_SAMPLES=128 SOAR_GPTQ_BATCH_SIZE=2 ./prepare_model.sh --input /path/raw_model --output /path/quant_model
```

4. Optional explicit module override test
```bash
SOAR_GPTQ_LAYER_AWARE=1 SOAR_GPTQ_INCLUDE_MODULES=self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj SOAR_GPTQ_EXCLUDE_MODULES=self_attn.o_gate ./prepare_model.sh --input /path/raw_model --output /path/quant_model
```

## Result Summary Table
Status: code implemented locally; runtime metrics pending fcloud execution.

| Metric | Before CHANGE_0030_002 | After CHANGE_0030_002 |
|---|---:|---:|
| GPTQ preprocess completion | Failed on mixed-layer mismatch | TBD |
| Correctness score | TBD | TBD |
| Speed S1 | TBD | TBD |
| Speed S8 | TBD | TBD |
| Speed Smax | TBD | TBD |

## Rollback Instructions
1. Revert the CHANGE_0030 implementation commit.
2. Or temporarily use `SOAR_QUANT_MODE=copy`.
3. If needed, set `SOAR_GPTQ_LAYER_AWARE=0` to bypass new targeting behavior while retaining other compatibility fixes.

## Next-Step Suggestions
1. Run the small-sample fcloud smoke first and share the full log.
2. If mismatch persists, capture the printed applied include/exclude attrs to identify which `QuantizeConfig` controls are actually honored.
3. Once quant-prep completes, run correctness and 3x speed evaluation and backfill results into CHANGE_0030 docs.
