# CHANGE_0056 Prepare-Env Local Wheel Installs

## Background and Motivation

After freezing the submission environment to the known fcloud dependency set, the next bottleneck was package installation latency and instability. Installing `gptqmodel==5.7.0`, `transformers==4.57.1`, and `torchao==0.9.0` from remote indexes can be slow on weak networks, and `gptqmodel==5.7.0` does not provide a directly downloadable prebuilt wheel for the target environment.

This iteration converts the pinned dependency installation path to local wheel installs. The user prepared:

- a locally built `gptqmodel-5.7.0-*.whl`
- a downloaded `transformers-4.57.1-*.whl`
- a downloaded `torchao-0.9.0-*.whl`

The goal is to make submission environment setup deterministic and less dependent on remote package availability.

## Rule-Compliance Statement

This change remains compliant with the latest SOAR toolkit submission workflow.

- It keeps the required `prepare_env.sh` + `prepare_model.sh --input/--output` contract.
- It only changes how dependencies are installed inside `prepare_env.sh`.
- It does not replace the MiniCPM-SALA base model or change evaluation behavior.
- It does not alter concurrency, prefix cache, or runtime scoring logic.

## Detailed Implementation Plan

Before change:

1. Stop installing the pinned GPTQ stack from remote indexes.
2. Look for the required local wheels in the demo submission directory.
3. Fail early with a clear message if any expected wheel is missing or ambiguous.
4. Install the local wheels with `uv pip install --force-reinstall --no-deps ...`.
5. Keep the post-install version probe so official logs confirm the effective versions.

## Actual Code Changes

Changed files:

- `benchmark/soar/demo_sala/prepare_env.sh`

What changed:

1. Added local-wheel glob discovery for:
   - `gptqmodel-5.7.0-*.whl`
   - `transformers-4.57.1-*.whl`
   - `torchao-0.9.0-*.whl`
2. Added strict checks that require exactly one matching wheel for each package.
3. Replaced remote package installs with local wheel installs for:
   - `gptqmodel`
   - `transformers`
   - `torchao`
4. Preserved the existing flash-attn and sgl-kernel local-wheel install flow.
5. Preserved the post-install Python probe that prints the effective versions and import paths of `torch`, `gptqmodel`, `transformers`, and `torchao`.

## Design Notes

### Why use local wheel installs instead of remote exact pins

Exact pins solve dependency drift, but they do not solve slow or unstable network fetches. Local wheels remove both sources of nondeterminism: version drift and network variability.

### Why only `gptqmodel` needed a local build

`transformers` is a pure-Python wheel and `torchao` already had a suitable prebuilt wheel. `gptqmodel==5.7.0` did not expose a directly downloadable matching wheel, so the user built a local wheel from the source archive on fcloud.

### Why not rebuild `transformers` or `torchao`

There is no expected 6000D-specific performance benefit from rebuilding `transformers`, and `torchao` is being used here as a compatibility-pinned runtime package rather than a custom hardware-tuned kernel package. This differs from the `sgl-kernel` wheel, where target-specific native code generation matters directly.

## Validation Commands

Place the three wheels in the demo directory, then run:

```bash
bash benchmark/soar/demo_sala/prepare_env.sh
```

Check effective versions:

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
| `gptqmodel` install | remote pinned install | local wheel install |
| `transformers` install | remote pinned install | local wheel install |
| `torchao` install | remote pinned install | local wheel install |
| Network dependence during setup | required | removed for the pinned GPTQ stack |
| Dependency reproducibility | pinned but still network-coupled | pinned and artifact-coupled |

## Rollback Instructions

If local wheel packaging becomes inconvenient or one of the wheels is unavailable:

1. restore the previous remote exact-pin installs in `prepare_env.sh`
2. remove the local-wheel presence checks
3. rerun `prepare_env.sh` and confirm the effective versions from the post-install probe

## Next-Step Suggestions

1. Copy the three wheel files into `benchmark/soar/demo_sala/` before packaging the submission archive.
2. Re-run the official submission and confirm the log shows the local wheel installs and expected pinned versions.
3. If preprocessing still fails after this change, continue debugging with both dependency drift and network fetch variability removed.