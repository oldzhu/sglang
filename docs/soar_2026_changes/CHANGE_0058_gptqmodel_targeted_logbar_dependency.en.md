# CHANGE_0058 GPTQModel Targeted Lightweight Dependencies

## Background and Motivation

Allowing the local `gptqmodel` wheel to resolve its dependencies fixed the first observed missing module (`logbar`), but it also triggered large remote downloads and broad dependency re-resolution for heavyweight packages such as `torch`, `triton`, and multiple NVIDIA CUDA wheels. That reintroduced both setup-time risk and compatibility drift.

The first confirmed missing dependency from official logs was `logbar`, followed by `accelerate`, `threadpoolctl`, `tokenicer`, the `pcre` module provided by the `pypcre` package, and then the `device_smi` module provided by the `device-smi` package. A later run also showed that unconstrained `accelerate` installation upgraded `huggingface-hub` to an incompatible `1.x` version for `transformers==4.57.1`. This iteration keeps the narrower path: explicitly install those lightweight packages, pin `huggingface-hub` back to a compatible `<1.0` version, then install the local `gptqmodel` wheel with `--no-deps` again.

## Rule-Compliance Statement

This change remains compliant with the latest SOAR toolkit submission workflow.

- It preserves the required `prepare_env.sh` + `prepare_model.sh --input/--output` contract.
- It only changes dependency installation behavior in `prepare_env.sh`.
- It does not alter the MiniCPM-SALA base model, runtime scoring logic, concurrency, or prefix-cache behavior.

## Detailed Implementation Plan

Before change:

1. Keep fail-fast shell behavior enabled.
2. Preserve pinned local-wheel installs for `transformers` and `torchao`.
3. Add explicit installs for the lightweight dependencies `logbar`, `accelerate`, `threadpoolctl`, `tokenicer`, `pypcre`, and `device-smi`.
4. Pin `huggingface-hub` back to a `transformers==4.57.1` compatible version after installing `accelerate`.
4. Revert the local `gptqmodel` wheel installation back to `--no-deps`.
5. Keep the import probe so the next missing lightweight dependency, if any, is surfaced immediately.

## Actual Code Changes

Changed files:

- `benchmark/soar/demo_sala/prepare_env.sh`

What changed:

1. Added explicit lightweight dependency installs for:
    - `logbar`
    - `accelerate`
    - `threadpoolctl`
    - `tokenicer`
    - `pypcre`
    - `device-smi`
2. Changed `accelerate` installation to use `--no-deps`.
3. Added an explicit `huggingface-hub==0.34.4` pin to preserve compatibility with `transformers==4.57.1`.
4. Reverted the local `gptqmodel` wheel install back to `--no-deps`.
5. Kept the local wheel installs for:
   - `transformers==4.57.1`
   - `torchao==0.9.0`
6. Kept fail-fast shell mode and the post-install import/version probe.

## Design Notes

### Why explicitly install `logbar`, `accelerate`, `threadpoolctl`, `tokenicer`, `pypcre`, and `device-smi`

`logbar`, `accelerate`, `threadpoolctl`, `tokenicer`, `pypcre`, and `device-smi` are the lightweight dependencies confirmed missing in official logs so far. Installing only confirmed missing lightweight packages is safer than allowing `gptqmodel` to resolve its full dependency graph from remote indexes.

### Why pin `huggingface-hub`

An official run showed `huggingface-hub==1.7.2`, which is incompatible with `transformers==4.57.1` because that version requires `huggingface-hub>=0.34.0,<1.0`. Installing `accelerate` without dependency guards can pull in a `1.x` hub release, so the script now restores a compatible `huggingface-hub==0.34.4` pin.

### Why revert `gptqmodel` back to `--no-deps`

The previous dependency-resolution approach triggered downloads of many heavyweight packages and risked changing the pinned runtime stack. Reverting to `--no-deps` restores determinism for the large packages while still allowing targeted fixes for small missing modules.

### Why use an incremental approach for future missing modules

If the next official run reveals another small missing Python dependency, it can be added in the same explicit way. This is slower than a full resolver but much safer for a submission environment that must stay reproducible.

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
for name in ["torch", "gptqmodel", "transformers", "torchao", "logbar", "accelerate", "threadpoolctl", "tokenicer", "pcre", "device_smi", "huggingface_hub"]:
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
| `gptqmodel` install strategy | local wheel with dependency resolution | local wheel with `--no-deps` |
| lightweight `gptqmodel` extras | missing at runtime | `logbar`, `accelerate`, `threadpoolctl`, `tokenicer`, `pypcre`, and `device-smi` installed explicitly |
| `huggingface-hub` compatibility | could drift to incompatible `1.x` | pinned to `0.34.4` for `transformers==4.57.1` |
| Heavy package re-resolution risk | high | minimized |
| Future missing small deps | implicit resolver behavior | explicit incremental vendoring |

## Rollback Instructions

If explicit lightweight dependency installation is insufficient and the manual dependency list grows too much:

1. inspect the next official missing-module error
2. decide whether to add another small explicit dependency or switch to a controlled vendoring bundle for all lightweight `gptqmodel` extras
3. keep avoiding unconstrained heavyweight dependency resolution unless absolutely necessary

## Next-Step Suggestions

1. Re-run the official submission and confirm `logbar`, `accelerate`, `threadpoolctl`, `tokenicer`, `pcre`, and `device_smi` are no longer missing.
2. If another small dependency is reported, add it explicitly instead of reopening full resolver behavior.
3. Confirm `huggingface-hub` remains `<1.0` so `transformers==4.57.1` continues to import cleanly.
3. If the list of missing small packages becomes long, consider bundling that lightweight dependency set as local wheels too.