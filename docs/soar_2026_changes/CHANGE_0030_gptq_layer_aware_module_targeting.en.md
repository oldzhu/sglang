# CHANGE_0030: GPTQ Layer-Aware Module Targeting (Proposal)

## Background and Motivation
Recent `preprocess_model.py` GPTQ runs progressed past earlier blockers (`trust_remote_code`, unsupported internal attention kwarg), but now fail with:

- `ValueError: layer module item self_attn.o_gate not found in model`

User-provided structure evidence shows MiniCPM-SALA uses heterogeneous attention block types across layers:

- Some sparse layers include `self_attn.o_gate`
- Some lightning/other layers do not include `o_gate` and instead expose modules such as `z_proj` and related norms

This means a single global GPTQ module target list is not architecture-safe for this hybrid model.

## Objective and Expected Gain
Implement a layer-aware GPTQ target selection policy in `benchmark/soar/demo_sala/preprocess_model.py` so quantization:

- avoids hard failures on missing modules
- preserves as much valid quant coverage as possible across heterogeneous layers
- becomes reproducible and debuggable with explicit plan/summary logging

Expected gain:

- unblock end-to-end quant-prep on fcloud
- reduce iteration time lost to architecture mismatch exceptions
- provide stable foundation for later quality/speed tuning

## Rule-Compliance Statement (SOAR)
This change is compliant with SOAR constraints because it:

- only modifies offline preprocessing behavior
- does not use forbidden online-eval tricks
- keeps correctness as first-class validation target
- preserves submission contract (`prepare_model.sh --input/--output` flow)

Rule freshness note:

- Compliance assumptions should continue to be checked against latest official pages before final submission decisions:
  - `https://soar.openbmb.cn/competition`
  - `https://soar.openbmb.cn/toolkit` (especially `技术路径指引` and `提交说明`)

## Risk to Accuracy/Stability
Potential risks:

- If target set is too conservative, quant coverage may drop and speed gain can be limited.
- If target set is too aggressive, calibration may become unstable or quality may degrade.
- Per-layer variability can make behavior version-sensitive across model revisions.

Mitigations:

- use explicit inclusion policy + deterministic filtering by actual module existence
- print quant target plan and per-group counts before quantization
- run staged validation (small calibration smoke -> larger calibration quality pass)

## Detailed Implementation Plan (Before Change)
Planned code changes in `benchmark/soar/demo_sala/preprocess_model.py`:

1. Add optional env controls for target strategy
- `SOAR_GPTQ_LAYER_AWARE=1` (default on)
- `SOAR_GPTQ_INCLUDE_MODULES` (comma-separated allowlist, optional override)
- `SOAR_GPTQ_EXCLUDE_MODULES` (comma-separated denylist, optional)
- `SOAR_GPTQ_STRICT_TARGETS=0/1` (strict fail if no usable target)

2. Build module inventory by introspecting model layers
- collect real module paths available per layer
- separate common modules (across all layers) from conditional modules (subset only)

3. Construct effective target set
- start from safe defaults (q/k/v/o and known MLP projections where present)
- include sparse-specific and lightning-specific modules only when present
- apply exclude list last

4. Quantization retry/guard logic
- if quantizer rejects some targets, retry with reduced set only when safe and logged
- fail clearly when effective target set becomes empty

5. Add transparent logs
- print selected targets, skipped targets, and counts by layer type/presence
- persist summary JSON next to output model for reproducibility (optional)

## Proposed Targeting Policy from Previous Reply (for review/refine)
Use a two-tier target strategy:

- Tier A (common first): modules that exist broadly and are usually high-value for GPTQ
- Tier B (conditional): modules only included when verified present in given layer family (e.g., sparse-only `o_gate`, lightning-only projections)

Operational rule:

- never pass a module name to GPTQ quantization unless it is observed in the current model inventory

## Validation Commands (Correctness + Speed)
After implementation, run in fcloud:

1. Quant-prep smoke (small calibration)
```bash
SOAR_QUANT_MODE=gptq \
SOAR_GPTQ_CALIBRATION_FILE=/path/to/calib.jsonl \
SOAR_GPTQ_CALIBRATION_SAMPLES=16 \
SOAR_GPTQ_BATCH_SIZE=1 \
./prepare_model.sh --input /path/raw_model --output /path/quant_model
```

2. Quant-prep quality pass (larger calibration)
```bash
SOAR_QUANT_MODE=gptq \
SOAR_GPTQ_CALIBRATION_FILE=/path/to/calib.jsonl \
SOAR_GPTQ_CALIBRATION_SAMPLES=128 \
SOAR_GPTQ_BATCH_SIZE=2 \
./prepare_model.sh --input /path/raw_model --output /path/quant_model
```

3. Serve + correctness check
```bash
# launch with quantized model then run official/local correctness eval
python benchmark/soar/eval_model.py --help
```

4. 3x speed benchmark (S1/S8/Smax)
```bash
python benchmark/soar/run_soar_suite.py --help
```

## Result Summary Table (Baseline vs New)
Status: pending implementation and fcloud results.

| Metric | Baseline (before CHANGE_0030) | New (after CHANGE_0030) |
|---|---:|---:|
| GPTQ preprocess completion rate | Fails on mixed-module mismatch | TBD |
| Correctness score | TBD | TBD |
| Speed S1 | TBD | TBD |
| Speed S8 | TBD | TBD |
| Speed Smax | TBD | TBD |

## Rollback Instructions
If regressions occur:

1. Disable layer-aware logic with env gate (if implemented) or revert commit for CHANGE_0030.
2. Fall back to `SOAR_QUANT_MODE=copy` to keep submission path functional.
3. Re-run correctness and speed baselines to confirm rollback safety.

## Next-Step Suggestions
1. Implement the plan as one cohesive feature in `preprocess_model.py`.
2. Run small-sample fcloud smoke first and share logs.
3. Refine target allow/exclude defaults based on real module inventory output.
4. Then run full correctness + 3x speed and update this doc with measured numbers.
