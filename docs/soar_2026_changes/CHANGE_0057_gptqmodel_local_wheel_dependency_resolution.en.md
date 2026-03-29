# CHANGE_0057 GPTQModel Local-Wheel Dependency Resolution

## Background and Motivation

After switching the pinned GPTQ stack to local wheel installs, official submission setup still failed before quantization began. The failure happened when importing `gptqmodel` and showed that the local wheel had been installed without its required lightweight Python dependencies:

```text
ModuleNotFoundError: No module named 'logbar'
```

The root cause was the local `gptqmodel` wheel being installed with `--no-deps`, which prevented its transitive Python dependencies from being resolved. At the same time, `prepare_env.sh` had fail-fast mode disabled, so the dependency-probe failure did not stop setup early.

This iteration fixes that specific packaging issue while keeping `transformers` and `torchao` pinned from local wheels.

## Rule-Compliance Statement

This change remains compliant with the latest SOAR toolkit submission workflow.

- It preserves the `prepare_env.sh` + `prepare_model.sh --input/--output` contract.
- It only changes dependency installation behavior inside `prepare_env.sh`.
- It does not alter the MiniCPM-SALA base model, runtime scoring logic, concurrency, or prefix-cache behavior.

## Detailed Implementation Plan

Before change:

1. Re-enable fail-fast shell behavior in `prepare_env.sh`.
2. Keep local wheel installs for the heavy pinned packages `transformers` and `torchao`.
3. Change local `gptqmodel` installation to allow dependency resolution.
4. Ensure `gptqmodel` is installed after the pinned local wheels so its dependency resolution does not become the primary source of version drift.
5. Keep the post-install import probe so missing dependencies are caught inside `prepare_env.sh`.

## Actual Code Changes

Changed files:

- `benchmark/soar/demo_sala/prepare_env.sh`

What changed:

1. Restored `set -euo pipefail` so setup stops on import-probe or install failures.
2. Reordered installs so:
   - local `transformers` wheel is installed first
   - local `torchao` wheel is installed second
   - local `gptqmodel` wheel is installed last
3. Changed `gptqmodel` local-wheel installation to allow dependency resolution by removing `--no-deps`.
4. Added an explicit uninstall step before reinstalling `gptqmodel`, mirroring the existing `torchao` handling.
5. Kept the post-install Python probe for `torch`, `gptqmodel`, `transformers`, and `torchao`.

## Design Notes

### Why only `gptqmodel` should resolve dependencies

`transformers` and `torchao` are already pinned and provided as local wheels, so those two should remain deterministic and network-free. `gptqmodel`, however, depends on several lightweight Python packages such as `logbar`. Blocking dependency resolution for `gptqmodel` made the local wheel incomplete at runtime.

### Why install `gptqmodel` after the pinned wheels

Installing the pinned local wheels first reduces the chance that `gptqmodel` dependency resolution will override the explicitly chosen versions of `transformers` or `torchao`. The intended effect is to let `gptqmodel` fetch only its missing lightweight dependencies.

### Why re-enable fail-fast behavior

The previous script allowed `prepare_env.sh` to continue after the import probe failed, which delayed the real error until `prepare_model.sh`. Failing inside `prepare_env.sh` makes packaging problems much easier to diagnose from official logs.

## Validation Commands

Environment setup validation:

```bash
bash benchmark/soar/demo_sala/prepare_env.sh
```

Check imports explicitly:

```bash
python3 - <<'PY'
import importlib
import json
for name in ["torch", "gptqmodel", "transformers", "torchao", "logbar"]:
    module = importlib.import_module(name)
    print(json.dumps({
        "module": name,
        "version": getattr(module, "__version__", "unknown"),
        "file": getattr(module, "__file__", None),
    }, ensure_ascii=False))
PY
```

Then retry preprocess:

```bash
bash benchmark/soar/demo_sala/prepare_model.sh --input <RAW_MODEL_DIR> --output <OUTPUT_MODEL_DIR>
```

## Result Summary Table

| Item | Before | After |
| --- | --- | --- |
| `gptqmodel` local wheel install | local wheel with `--no-deps` | local wheel with dependency resolution enabled |
| Missing lightweight deps such as `logbar` | possible | expected to be resolved during install |
| `prepare_env.sh` error handling | fail-fast disabled | fail-fast enabled |
| Packaging failures visibility | deferred into `prepare_model.sh` | surfaced during `prepare_env.sh` |

## Rollback Instructions

If allowing `gptqmodel` dependency resolution causes unwanted upgrades of pinned packages:

1. revert to the previous local-wheel install logic
2. vendor the missing lightweight dependencies explicitly as local wheels or exact pins
3. keep fail-fast shell mode enabled so dependency issues remain visible early

## Next-Step Suggestions

1. Re-run the official submission and confirm `logbar` no longer fails to import.
2. Check whether `gptqmodel` dependency resolution preserved the pinned `transformers==4.57.1` and `torchao==0.9.0` versions.
3. If another lightweight dependency is missing, vendor or pin that smaller dependency set explicitly rather than abandoning the local-wheel approach.