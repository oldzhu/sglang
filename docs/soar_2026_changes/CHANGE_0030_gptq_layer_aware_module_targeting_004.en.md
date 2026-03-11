# CHANGE_0030_004: GPTQ Official Dynamic Control Research and Proposal

## Background and Motivation
After repeated CHANGE_0030 attempts, the team paused implementation to inspect GPTQModel 5.7.0 source and documentation directly instead of continuing with unsupported guesses.

The concrete goal of this research step was:

- identify the exact Python source files responsible for the failure stack
- determine how GPTQModel builds the per-layer module list for calibration/quantization
- confirm whether GPTQModel exposes an official include/exclude control path

This continuation records the findings.

## Rule-Compliance Statement (SOAR)
This is a research and proposal-only iteration.

- no model behavior was changed in this step
- no evaluation-time behavior was altered
- submission contract remains unchanged

## Problem Stack: Installed GPTQModel 5.7.0 Source on fcloud
Relevant Python source files from the failing environment:

- `/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/gptqmodel/quantization/config.py`
- `/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/gptqmodel/models/base.py`
- `/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/gptqmodel/looper/module_looper.py`
- `/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/gptqmodel/looper/stage_layer.py`
- `/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/gptqmodel/utils/model.py`

Failure path seen in traceback:

- `BaseQModel.quantize(...)`
- `ModuleLooper.loop(...)`
- `ModuleLooper._loop_impl(...)`
- `run_layer_stage(...)`
- `create_named_modules(...)`
- raises `ValueError: layer module item self_attn.o_gate not found in model`

## Research Findings
### 1. Official per-module control exists
GPTQModel README and source indicate the official mechanism is:

- `QuantizeConfig.dynamic`

Documented behavior:

- positive regex match: override per-module quantization config
- negative regex match: skip matching module from quantization

### 2. Official filtering path is in BaseQModel
Relevant logic in `gptqmodel/models/base.py`:

- `simple_layer_modules(...)`
- `filter_not_quantize_module(...)`

`filter_not_quantize_module(...)` removes modules when:

- `dynamic_get(quantize_config.dynamic, module_name=m) is False`

This is a strong signal that `dynamic` is the intended official way to exclude modules.

### 3. Secondary enforcement exists in quant module creation
Relevant logic in `gptqmodel/utils/model.py`:

- `create_quant_module(...)`

It also consults `dynamic` and skips module creation when the dynamic match is negative.

### 4. fcloud runtime evidence confirms the intended control flow
Observed output from the loaded model:

- `module_tree = ['model', 'layers', '#', {'self_attn': ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'o_gate'), 'mlp': ('gate_proj', 'up_proj', 'down_proj')}]`
- `simple_layer_modules = [['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj'], ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']]`
- `full_layer_modules = [['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'self_attn.o_gate'], ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']]`

Interpretation:

- `module_tree` is structural and still contains `o_gate`
- `full_layer_modules` is the broader structural expansion and still contains `o_gate`
- `simple_layer_modules` is the filtered quantization-oriented list and already excludes `o_gate`

This is the clearest evidence so far that GPTQModel expects module exclusion to happen through the official filtering path, not by mutating `module_tree`.

## Proposal: Official Fix Direction
### Objective and Expected Gain
Replace current heuristic internal mutations with a documented `QuantizeConfig.dynamic` negative-match rule so `self_attn.o_gate` is excluded through GPTQModel's intended control path.

Expected gain:

- stop relying on unsupported guesses such as manual `module_tree` sanitization
- align with GPTQModel 5.7.0 official per-module exclusion flow
- reduce time spent on unstable trial-and-error patches

### Exact File to Change
- `benchmark/soar/demo_sala/preprocess_model.py`

### Planned Code Change
Construct `QuantizeConfig` with `dynamic` negative match, for example:

```python
dynamic = {
    r"-:.*self_attn\.o_gate.*": {},
}
quant_config = QuantizeConfig(bits=bits, group_size=group_size, dynamic=dynamic)
```

Fallback variant if exact path matching is needed:

```python
dynamic = {
    r"-:.*o_gate.*": {},
}
```

Planned cleanup:

- remove heuristic `module_tree` mutation path added in earlier CHANGE_0030 continuation
- keep API-compat wrappers for `GPTQModel.load(...)` and `model.quantize(...)`
- log the final `dynamic` rules used for reproducibility

## Risk to Accuracy/Stability
Risks:

- `o_gate` remains excluded globally from quantization in the conservative path
- exact regex may need one adjustment if GPTQModel matches module names differently than expected

Mitigations:

- use the documented `dynamic` mechanism rather than private internals
- validate by checking that `simple_layer_modules(...)` excludes `o_gate`
- run 16-sample smoke before any larger quantization pass

## Validation Commands
1. Verify official filtering path directly on fcloud
```bash
python3 - <<'PY'
from gptqmodel import GPTQModel, QuantizeConfig

qcfg = QuantizeConfig(
    bits=4,
    group_size=128,
    dynamic={r"-:.*self_attn\\.o_gate.*": {}},
)

model = GPTQModel.load(
    "/path/to/raw_model",
    qcfg,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)

print("module_tree =", getattr(model, "module_tree", None))
print("simple_layer_modules =", model.simple_layer_modules(model.model.config, model.quantize_config))
print("full_layer_modules =", model.full_layer_modules(model.model.config))
PY
```

2. fcloud smoke after implementation
```bash
cd benchmark/soar/demo_sala
SOAR_QUANT_MODE=gptq \
SOAR_GPTQ_CALIBRATION_FILE=/path/to/calib.jsonl \
SOAR_GPTQ_CALIBRATION_SAMPLES=16 \
SOAR_GPTQ_BATCH_SIZE=1 \
./prepare_model.sh --input /path/to/raw_model --output /path/to/quant_smoke
```

## Result Summary Table
Status: research complete; code change not yet applied in this continuation.

| Metric | Before CHANGE_0030_004 | After CHANGE_0030_004 |
|---|---:|---:|
| Official module exclusion path identified | No | Yes |
| GPTQ preprocess completion | Fails on `o_gate` mismatch | TBD |
| Correctness score | TBD | TBD |
| Speed S1 | TBD | TBD |
| Speed S8 | TBD | TBD |
| Speed Smax | TBD | TBD |

## Rollback Instructions
No rollback needed for this continuation because it documents research and proposal only.

## Next-Step Suggestions
1. Approve one focused implementation that replaces heuristic tree mutation with `QuantizeConfig.dynamic`.
2. Validate `simple_layer_modules(...)` behavior on fcloud immediately after the patch.
3. Only if official `dynamic` still fails should we revisit deeper internal hooks.