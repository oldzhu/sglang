# CHANGE_0054 Submission Environment Version Probe

## Background and Motivation

Recent official submissions started failing in a way that suggests the official evaluation environment may differ from the user's fcloud instance on key preprocessing dependencies. The current suspicion is that `gptqmodel`, `transformers`, or related runtime packages changed in the official environment even though local behavior did not.

Before continuing with more compatibility code changes, this iteration adds a focused diagnostic submission feature: a standalone preprocess probe that prints the effective Python and dependency versions seen by the official platform, then fails intentionally so those lines appear in the platform's final error log.

## Rule-Compliance Statement

This change remains compliant with the latest SOAR competition and toolkit guidance.

- It preserves the required `prepare_env.sh` + `prepare_model.sh --input/--output` submission contract.
- It does not replace the MiniCPM-SALA base model or modify evaluation logic.
- It does not alter runtime concurrency, prefix-cache behavior, or serving configuration.
- It is a troubleshooting-only submission feature used to inspect the official preprocessing environment.

## Detailed Implementation Plan

Before change:

1. Keep the existing GPTQ preprocess logic untouched.
2. Add a separate diagnostic preprocess entrypoint instead of editing the current quantization flow.
3. Log the versions and import locations of `torch`, `gptqmodel`, and `transformers` as early as possible.
4. Raise an intentional error immediately after logging so the official platform exposes those lines in the last 50 log lines.
5. Temporarily point `prepare_model.sh` to the diagnostic entrypoint for the probe submission.

## Actual Code Changes

Changed files:

- `benchmark/soar/demo_sala/preprocess_model_001.py`
- `benchmark/soar/demo_sala/prepare_model.sh`

What changed:

1. Added a new standalone script `preprocess_model_001.py`.
2. The new script parses `--input` and `--output` to preserve the official submission interface.
3. It prints:
   - invocation context (`argv`, cwd, input/output paths)
   - Python version and interpreter path
   - `torch` version and import path
   - `gptqmodel` version and import path
   - `transformers` version and import path
4. Each dependency probe is resilient: if import fails, the script logs the exception type and message instead of hiding the failure.
5. The script raises an intentional `RuntimeError` immediately after logging so the submission fails early and surfaces the official environment details.
6. `prepare_model.sh` now points to `preprocess_model_001.py` temporarily for this diagnostic submission.

## Design Notes

### Why use a separate preprocess script

The current `preprocess_model.py` already contains active GPTQ compatibility work. A separate probe keeps the troubleshooting submission isolated, avoids mixing diagnostic failure logic into the working preprocess path, and makes rollback trivial.

### Why fail intentionally

The objective of this iteration is not to produce a processed model. It is to force the official platform to emit the relevant version lines in its error log. Failing immediately after logging minimizes noise and keeps the last 50 lines focused on the environment mismatch question.

### Why log import file paths in addition to versions

Version strings alone may not fully explain mismatches when editable installs, shadowed modules, or multiple site-packages paths are involved. Logging module file paths provides stronger evidence about what the official environment is actually importing.

## Validation Commands

Local probe execution:

```bash
bash benchmark/soar/demo_sala/prepare_model.sh --input <RAW_MODEL_DIR> --output <OUTPUT_MODEL_DIR>
```

Direct Python execution:

```bash
python3 benchmark/soar/demo_sala/preprocess_model_001.py \
  --input <RAW_MODEL_DIR> \
  --output <OUTPUT_MODEL_DIR>
```

Expected result:

- version probe lines are printed for Python, `torch`, `gptqmodel`, and `transformers`
- the script exits with the intentional diagnostic `RuntimeError`

## Result Summary Table

| Item | Before | After |
| --- | --- | --- |
| Official preprocess visibility | only failure trace from active GPTQ path | explicit environment probe lines before failure |
| Dependency comparison | inferred indirectly | direct `torch` / `gptqmodel` / `transformers` version logging |
| Import-path evidence | none | module file paths logged |
| Active GPTQ preprocess logic | would run and fail later | untouched, bypassed for this diagnostic submission |

## Rollback Instructions

After collecting the official environment information:

1. restore `benchmark/soar/demo_sala/prepare_model.sh` to call `preprocess_model.py`
2. keep or remove `preprocess_model_001.py` depending on whether future environment probes are still needed
3. resume fixes only after comparing the official versions against the user's fcloud instance

## Next-Step Suggestions

1. Submit this probe package and capture the official last-50-lines output.
2. Compare the logged `torch`, `gptqmodel`, and `transformers` versions against the fcloud instance.
3. If versions differ, ask the official-site owner about that exact delta.
4. If versions match, continue debugging with the existing GPTQ compatibility instrumentation instead of assuming an environment mismatch.