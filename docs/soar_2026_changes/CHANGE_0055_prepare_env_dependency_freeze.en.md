# CHANGE_0055 Prepare-Env Dependency Freeze

## Background and Motivation

The official diagnostic submission showed that the evaluation environment was resolving a newer dependency set than the user's long-lived fcloud instance:

- official: `gptqmodel==5.8.0`, `transformers==5.3.0`, `torchao==0.16.0`
- fcloud: `gptqmodel==5.7.0`, `transformers==4.57.1`, `torchao==0.9.0`

That mismatch makes GPTQ debugging noisy because failures may come from dependency drift instead of the submitted code. Before attempting more compatibility work against the newer stack, this iteration freezes the submission environment to the exact dependency versions already exercised in fcloud.

## Rule-Compliance Statement

This change remains compliant with the latest SOAR toolkit submission workflow.

- It preserves the `prepare_env.sh` + `prepare_model.sh --input/--output` contract.
- It only pins Python package versions inside `prepare_env.sh`, which is explicitly allowed.
- It does not replace the MiniCPM-SALA base model or alter scoring logic.
- It does not modify prefix cache, concurrency, or hidden evaluation behavior.

## Detailed Implementation Plan

Before change:

1. Replace the floating `gptqmodel` install with exact version pins.
2. Pin `transformers` explicitly instead of relying on dependency resolution.
3. Reinstall `torchao==0.9.0` so the runtime package set matches the fcloud-tested environment.
4. Restore `prepare_model.sh` from the diagnostic probe back to the real preprocess entrypoint.
5. Print the effective installed versions in `prepare_env.sh` so official logs confirm the pins actually took effect.

## Actual Code Changes

Changed files:

- `benchmark/soar/demo_sala/prepare_env.sh`
- `benchmark/soar/demo_sala/prepare_model.sh`

What changed:

1. Replaced floating `uv pip install gptqmodel ...` with an exact, force-reinstalled dependency set:
   - `gptqmodel==5.7.0`
   - `transformers==4.57.1`
2. Switched `torchao` installation to a force-reinstalled exact pin at `0.9.0`.
3. Added a short post-install Python probe that prints the effective versions and import paths of:
   - `torch`
   - `gptqmodel`
   - `transformers`
   - `torchao`
4. Restored `prepare_model.sh` to call `preprocess_model.py` instead of the temporary diagnostic probe.

## Design Notes

### Why pin the full tested set instead of only `gptqmodel`

The observed mismatch was not limited to `gptqmodel`. `transformers` and `torchao` also differed, and those packages can directly affect model-loading code paths and quantization support behavior. Pinning only one package would leave dependency drift unresolved.

### Why prefer the older tested stack first

The fcloud environment already exercised this stack during the earlier GPTQ debugging cycle. Freezing to that known stack is lower risk than simultaneously upgrading to new major versions and debugging new behavior at the same time.

### What newer versions might change later

Newer `gptqmodel` / `transformers` versions may include loader fixes, API changes, kernel integration changes, or quantization-related bug fixes. However, they do not automatically imply better quantization accuracy. Accuracy is usually dominated by quantization method, calibration data, grouping, and model-specific handling. Any claim that `gptqmodel==5.8.0` and `transformers==5.3.0` improve accuracy needs a controlled A/B comparison on the same model, same calibration set, and same eval set.

## Validation Commands

Environment setup validation:

```bash
bash benchmark/soar/demo_sala/prepare_env.sh
```

Check pinned versions explicitly:

```bash
python3 - <<'PY'
import importlib
import json
for name in ["torch", "gptqmodel", "transformers", "torchao"]:
    module = importlib.import_module(name)
    print(json.dumps({
        "module": name,
        "version": getattr(module, "__version__", "unknown"),
        "file": getattr(module, "__file__", None),
    }, ensure_ascii=False))
PY
```

Resume preprocess validation:

```bash
bash benchmark/soar/demo_sala/prepare_model.sh --input <RAW_MODEL_DIR> --output <OUTPUT_MODEL_DIR>
```

## Result Summary Table

| Item | Before | After |
| --- | --- | --- |
| `gptqmodel` install | floating latest resolution | pinned to `5.7.0` |
| `transformers` install | dependency-resolved by installer | pinned to `4.57.1` |
| `torchao` install | manually restored but not force-frozen with the full set | pinned to `0.9.0` |
| Official/fcloud reproducibility | drifting dependency set | aligned to known fcloud stack |
| `prepare_model.sh` target | diagnostic probe | real preprocess entrypoint |

## Rollback Instructions

If the pinned older stack stops installing or proves incompatible with the current official image:

1. restore floating installs or update the pins to a new validated set
2. rerun the dependency probe to confirm the effective versions
3. if needed, move to a second iteration that upgrades both fcloud and official runs to the newer `gptqmodel==5.8.0` / `transformers==5.3.0` stack

## Next-Step Suggestions

1. Re-run the official submission with the pinned dependency set and confirm the log shows the expected versions.
2. If preprocessing still fails, continue debugging without the dependency-drift variable.
3. Later, if time permits, run a controlled A/B between the pinned stack and the newer stack to see whether any loader behavior or accuracy changes are actually beneficial.