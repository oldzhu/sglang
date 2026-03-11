# CHANGE_0030_001: GPTQ Layer-Aware Module Targeting (Introspection-Based Continuation)

## Purpose
This continuation document captures the long API-introspection-based reply for CHANGE_0030, so the main feature document remains readable while preserving full decision traceability.

## New Evidence from fcloud API Introspection
From user-provided fcloud output:

- `gptqmodel` version: `5.7.0`
- Loaded instance type: `gptqmodel.models.base.BaseQModel`
- Instance methods available include:
  - `quantize(...)`
  - `save(...)`
  - `save_quantized(...)`
- Earlier introspection error (`GPTQModel has no attribute quantize`) is explained by class-level inspection; quantization is exposed on loaded model instance.

Additional runtime clues from logs:

- Module Tree AutoCompat used `layer0` to infer targets and included `self_attn.o_gate`.
- This remains risky for mixed layer families where some layers do not contain `o_gate`.

## Interpretation
Confirmed compatibility facts:

1. Quantization dispatch should rely on loaded instance API (not class static assumptions).
2. Current call pattern `model.quantize(calibration_texts, batch_size=...)` is valid for this version.
3. Failure mode is still tied to mixed-layer module targeting (AutoCompat deriving from non-representative layer subset).

## Implementation-Ready Proposal (Approved Direction)
### Objective
Make `benchmark/soar/demo_sala/preprocess_model.py` robust for `gptqmodel==5.7.0` and heterogeneous layer module layouts.

### Expected Gain
- Avoid hard failure from `self_attn.o_gate not found in model` in mixed layer architectures.
- Keep preprocessing reproducible with explicit target selection logs.

### Rule Compliance
- Offline preprocess only.
- No evaluation-time forbidden tricks.
- Submission interface contract unchanged.

### Risks and Mitigations
Risks:
- Conservative module targeting may reduce quant coverage.
- Aggressive targeting can still trigger missing-module errors.

Mitigations:
- Environment-gated layer-aware behavior.
- Explicit target list logging before quantization.
- Controlled retry path with conservative module subset.

## Exact Files / Functions to Change
Primary file:

- `benchmark/soar/demo_sala/preprocess_model.py`

Planned updates inside `run_gptq_quantization(...)` and helper utilities:

1. Add env knobs:
- `SOAR_GPTQ_LAYER_AWARE` (default `1`)
- `SOAR_GPTQ_INCLUDE_MODULES` (CSV override)
- `SOAR_GPTQ_EXCLUDE_MODULES` (CSV)

2. Add module list parsing helper:
- normalize CSV into stable list
- ignore empty tokens

3. Configure quant targets safely:
- preferred set from include/exclude + safe defaults
- omit `o_gate` from conservative fallback set

4. Add one-pass retry on module-mismatch errors:
- first attempt with configured target set
- on specific missing-module errors, retry with conservative common modules only:
  - `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`

5. Improve diagnostic messages:
- print active target set and retry reason
- fail with explicit actionable guidance when both attempts fail

## Validation Commands
1. Small-sample smoke:
```bash
SOAR_QUANT_MODE=gptq \
SOAR_GPTQ_CALIBRATION_FILE=/path/to/calib.jsonl \
SOAR_GPTQ_CALIBRATION_SAMPLES=16 \
SOAR_GPTQ_BATCH_SIZE=1 \
./prepare_model.sh --input /path/raw_model --output /path/quant_model
```

2. Larger-sample quality pass:
```bash
SOAR_QUANT_MODE=gptq \
SOAR_GPTQ_CALIBRATION_FILE=/path/to/calib.jsonl \
SOAR_GPTQ_CALIBRATION_SAMPLES=128 \
SOAR_GPTQ_BATCH_SIZE=2 \
./prepare_model.sh --input /path/raw_model --output /path/quant_model
```

3. Then correctness and speed evaluation:
```bash
python benchmark/soar/eval_model.py --help
python benchmark/soar/run_soar_suite.py --help
```

## Result Summary Table (To Fill After Run)
| Metric | Before continuation changes | After continuation changes |
|---|---:|---:|
| GPTQ preprocess completion | Fails on mixed module mismatch | TBD |
| Correctness | TBD | TBD |
| Speed S1 | TBD | TBD |
| Speed S8 | TBD | TBD |
| Speed Smax | TBD | TBD |

## Rollback
1. Disable layer-aware env gate (if introduced) and retry.
2. Use `SOAR_QUANT_MODE=copy` as temporary fallback.
3. Revert continuation commit if needed and restore prior behavior.

## Next Steps
1. Implement this continuation as the concrete CHANGE_0030 code patch.
2. Run fcloud smoke and share logs.
3. Refine module include/exclude defaults using observed failures.
