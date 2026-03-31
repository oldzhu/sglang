# CHANGE_0062 GPTQ QA+MCQ+CWE Calibration Scope

## Background and Motivation

Recent local and official runs show that the current speed-optimized package has become competitive on benchmark duration, but correctness remains the main score limiter. The largest unstable or degraded task groups are now consistently:

1. `qa`
2. `mcq`
3. `cwe`

The previous calibration focus used only `qa,cwe`. That leaves `mcq` outside the default GPTQ calibration scope even though recent local evaluations show `mcq` is one of the three weakest categories.

This iteration changes only one variable: widen the default calibration task scope from `qa,cwe` to `qa,mcq,cwe` while keeping the existing `32`-sample budget unchanged.

## Rule-Compliance Statement

This change remains compliant with the latest SOAR competition and toolkit pages checked on 2026-03-31.

- It only changes calibration-record selection during allowed preprocessing.
- It preserves the existing GPTQ W4A16 + Marlin path.
- It does not change runtime concurrency rules or prefix-cache behavior.
- It does not replace the MiniCPM-SALA base model.
- It keeps rollback to the previous task scope trivial through an environment variable.

## Detailed Implementation Plan

Before change:

1. Keep the existing calibration selector implementation unchanged.
2. Keep the total calibration sample count at `32`.
3. Keep stratified selection enabled.
4. Change only the default task include list from `qa,cwe` to `qa,mcq,cwe`.
5. Defer any `48`-sample experiment until after this narrower A/B is measured.

## Actual Code Changes

Changed files:

- `benchmark/soar/demo_sala/prepare_env.sh`

What changed:

1. Updated the default value of `SOAR_GPTQ_CALIBRATION_TASK_INCLUDE`.
2. Previous default:
   - `qa,cwe`
3. New default:
   - `qa,mcq,cwe`
4. Sample count remains:
   - `SOAR_GPTQ_CALIBRATION_SAMPLES=32`
5. Sampling mode remains:
   - `SOAR_GPTQ_CALIBRATION_SAMPLING=stratified`

## Design Notes

### Why not move to 48 samples immediately

Moving directly to `48` samples would change two variables at once:

1. add `mcq`
2. increase the sample budget

That makes the result harder to interpret, and recent tests already show `64` samples cause OOM consistently. Starting with `qa,mcq,cwe` at the existing `32`-sample budget is the cleaner one-feature step.

### Why add mcq now

Recent local runs show `mcq` is one of the three main weak categories, alongside `qa` and `cwe`. Since the selector already supports task filtering, the lowest-risk next move is to ensure all three weak categories participate in calibration.

### Why keep stratified sampling

The existing stratified flow still provides bucket diversity within the chosen tasks. That is useful because the current instability is not isolated to a single task only, but also interacts with prompt-length buckets.

## Validation Commands

Shell syntax:

```bash
bash -n benchmark/soar/demo_sala/prepare_env.sh
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

Rollback to previous scope:

```bash
export SOAR_GPTQ_CALIBRATION_TASK_INCLUDE=qa,cwe
```

Optional next-step experiment if this is not enough:

```bash
export SOAR_GPTQ_CALIBRATION_SAMPLES=48
export SOAR_GPTQ_CALIBRATION_TASK_INCLUDE=qa,mcq,cwe
```

## Result Summary Table

| Item | Before | After |
| --- | --- | --- |
| Default calibration task scope | `qa,cwe` | `qa,mcq,cwe` |
| Sample count | `32` | `32` |
| Sampling mode | `stratified` | `stratified` |
| Main changed variable | none | include `mcq` in default calibration focus |

## Rollback Instructions

If adding `mcq` does not improve correctness enough:

1. set `SOAR_GPTQ_CALIBRATION_TASK_INCLUDE=qa,cwe`
2. rerun the same preprocess and correctness validation
3. then move to the next feature: `48` samples with `qa,mcq,cwe`, or a memory-safer calibration expansion if OOM appears

## Next-Step Suggestions

1. Measure whether `mcq` improves without harming `qa` and `cwe`.
2. If accuracy remains unstable, test `48` samples with the same task scope as the next separate feature.
3. If `48` causes OOM, the next iteration should target calibration-memory stability rather than simply adding more samples.