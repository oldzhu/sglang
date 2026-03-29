# CHANGE_0059 GPTQ QA+CWE Calibration Focus

## Background and Motivation

Recent official submissions are operationally stable but still fail the SOAR correctness gate because quantized accuracy remains below the required 97% threshold. Local non-quant versus quant comparisons show that the largest accuracy losses are concentrated in the `qa` and `cwe` tasks, especially in the `len_4k_32k` and `len_32k_128k` prompt-length buckets.

The latest SOAR toolkit `技术路径指引` still explicitly recommends the W4A16 GPTQ + Marlin path, and the submission instructions still allow on-platform preprocessing through `prepare_model.sh --input --output`. That means calibration-data composition remains a compliant and low-risk lever for accuracy recovery.

This iteration narrows GPTQ calibration to the two task families with the largest observed quantization loss first: `qa` and `cwe`. The goal is to spend the fixed 32-sample calibration budget on the weakest buckets before moving to higher-risk changes such as selective de-quantization.

## Rule-Compliance Statement

This change is compliant with the latest SOAR competition and toolkit pages checked on 2026-03-28.

- It keeps the official `prepare_env.sh` and `prepare_model.sh --input/--output` submission contract unchanged.
- It stays within the officially encouraged quantization path: GPTQ W4A16 + Marlin.
- It does not modify the MiniCPM-SALA base model, benchmark concurrency logic, or forbidden prefix-cache behavior.
- It only changes calibration-record selection inside the allowed preprocessing stage.

## Detailed Implementation Plan

Before change:

1. Keep the existing 32-sample GPTQ calibration flow.
2. Preserve stratified sampling so prompt-length diversity remains represented.
3. Add a task-include filter driven by environment variable.
4. Default the submission configuration to `qa,cwe` for this experiment.
5. Emit the active task filter in setup logs so official runs are auditable.

## Actual Code Changes

Changed files:

- `benchmark/soar/demo_sala/preprocess_model.py`
- `benchmark/soar/demo_sala/prepare_env.sh`

What changed:

1. Added `_filter_calibration_records_by_task()` in `preprocess_model.py`.
2. Added task normalization so `SOAR_GPTQ_CALIBRATION_TASK_INCLUDE` is matched case-insensitively.
3. Applied the task filter before the existing sequential, shuffled, or stratified sample selection logic.
4. Extended the calibration summary to report:
   - whether task filtering was applied
   - included tasks
   - record count before filtering
   - record count after filtering
5. Set the submission default in `prepare_env.sh` to:
   - `SOAR_GPTQ_CALIBRATION_TASK_INCLUDE=qa,cwe`
6. Logged the active task filter in `prepare_env.sh` output.

## Design Notes

### Why filter before stratified sampling

The current code already preserves task and prompt-length structure during `stratified` mode. Filtering first lets the remaining 32-sample budget focus on `qa` and `cwe`, while still keeping length-aware coverage inside those tasks.

### Why keep 32 samples

The latest evidence suggests accuracy is the blocker, not preprocessing cost. Keeping the larger 32-sample budget avoids mixing two variables at once.

### Why use an environment variable

An environment variable keeps the change reversible without further code edits. Future runs can broaden the task set or disable the filter by overriding `SOAR_GPTQ_CALIBRATION_TASK_INCLUDE`.

## Validation Commands

Shell syntax:

```bash
bash -n benchmark/soar/demo_sala/prepare_env.sh
```

Python syntax:

```bash
python3 -m py_compile benchmark/soar/demo_sala/preprocess_model.py
```

Quick calibration-selection probe:

```bash
SOAR_GPTQ_CALIBRATION_TASK_INCLUDE=qa,cwe \
python3 benchmark/soar/demo_sala/preprocess_model.py \
  --input <RAW_MODEL_DIR> \
  --output <OUTPUT_MODEL_DIR> \
  --mode copy
```

Full preprocess run:

```bash
bash benchmark/soar/demo_sala/prepare_model.sh --input <RAW_MODEL_DIR> --output <OUTPUT_MODEL_DIR>
```

Correctness check after quantization:

```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <OUTPUT_MODEL_DIR> \
  --data_path <DATA_DIR>/perf_public_set.jsonl \
  --concurrency 32
```

## Result Summary Table

| Item | Before | After |
| --- | --- | --- |
| Calibration task scope | all tasks in calibration file | only `qa` and `cwe` by default |
| Sample budget | 32 | 32 |
| Sampling mode | stratified | stratified |
| Length-bucket representation | enabled | preserved inside filtered task set |
| Quantization feature scope | unchanged | unchanged |

## Rollback Instructions

If QA+CWE-focused calibration does not improve correctness enough:

1. override `SOAR_GPTQ_CALIBRATION_TASK_INCLUDE` to a broader task list such as `mcq,qa,cwe`
2. or remove the filter by setting a different default in `prepare_env.sh`
3. keep the current code path and move to the next lower-risk step, such as adding `mcq` back while retaining length-aware stratification

## Next-Step Suggestions

1. Run one local quantization + correctness pass with the new `qa,cwe` focus and compare the four known weak buckets first.
2. If the gain is limited, try `qa,cwe,mcq` next before moving to selective de-quantization.
3. If QA improves but CWE still lags, consider a second iteration that biases calibration allocation toward the `len_4k_32k` and `len_32k_128k` buckets within the filtered tasks.