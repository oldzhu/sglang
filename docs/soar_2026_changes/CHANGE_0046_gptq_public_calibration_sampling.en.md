# CHANGE_0046: GPTQ Public Calibration Sampling Strategy

## Background and Motivation

The current GPTQ preprocess path uses `perf_public_set.jsonl` as the calibration source, which is allowed by the latest SOAR toolkit page because the public correctness dataset is explicitly published for self-checking.

However, the previous implementation selected calibration samples by reading the JSONL file in order and taking the first `N` rows only. That created two problems:

- increasing `SOAR_GPTQ_CALIBRATION_SAMPLES` from `32` to `64` did not necessarily improve accuracy because sample composition stayed biased
- the selected calibration slice could over-represent whichever task and prompt-length blocks appear earliest in `perf_public_set.jsonl`

For MiniCPM-SALA, this is especially risky because the public set mixes short MCQ prompts and much longer information-retrieval prompts. GPTQ quality depends on activation coverage, so sample composition matters at least as much as sample count.

This iteration keeps the quantization method unchanged and focuses on a single low-risk optimization feature: make calibration sampling configurable and reproducible, with a stratified mode that better covers the public task and prompt-length distribution.

## Rule-Compliance Statement

This change remains compliant with the latest official competition and toolkit pages:

- it stays within the officially encouraged `路径一：量化加速` workflow
- it does not replace the MiniCPM-SALA base model
- it still performs quantization on the evaluation machine through `prepare_model.sh`
- it uses only the published `perf_public_set.jsonl` for public-set calibration and does not use hidden or private data
- it does not modify serving concurrency rules, prefix-cache behavior, or the official correctness script

## Detailed Implementation Plan

Before change:

1. Re-check the latest `competition` and `toolkit` pages to confirm that public-set self-checking and GPTQ path-one optimization remain allowed.
2. Inspect the current preprocess code to confirm that calibration sample loading is sequential and non-random.
3. Add a deterministic selection layer before text extraction so sampling can be changed without changing quantization logic.
4. Keep a rollback-safe `sequential` mode so the previous behavior remains available.

## Actual Code Changes

Changed files:

- `benchmark/soar/demo_sala/preprocess_model.py`
- `benchmark/soar/demo_sala/prepare_env.sh`

Changes made in `preprocess_model.py`:

1. Added deterministic calibration sampling modes:
   - `sequential`
   - `shuffled`
   - `stratified`
2. Added a fixed-seed path so shuffled and stratified selection are reproducible.
3. Added stratification helpers based on:
   - task type
   - `prompt_tokens` length bucket when available
4. Added a largest-remainder allocation step so the selected sample budget is distributed across buckets instead of collapsing onto the earliest rows.
5. Extended GPTQ start logging to print the effective calibration sampling summary, including:
   - mode
   - seed
   - available rows
   - selected rows
   - selected bucket counts

Changes made in `prepare_env.sh`:

1. Added new exported defaults for calibration selection:
   - `SOAR_GPTQ_CALIBRATION_SAMPLING=stratified`
   - `SOAR_GPTQ_CALIBRATION_SEED=20260320`
   - `SOAR_GPTQ_CALIBRATION_TASK_BALANCE=1`
   - `SOAR_GPTQ_CALIBRATION_USE_PROMPT_TOKENS=1`
2. Added startup logging for these environment variables so remote validation logs clearly show the effective strategy.

## Design Notes

### Why not just increase calibration sample count

The earlier `32 -> 64` experiment did not improve accuracy. The root issue is that the code previously selected the first `N` rows only. If the dataset is block-ordered by task or prompt length, increasing `N` can simply expand the same bias instead of improving calibration coverage.

### Why stratified sampling is the default

The public set contains both short MCQ examples and long-context retrieval examples. A calibration slice that covers both task families and multiple prompt-length bands is more likely to preserve general correctness after W4A16 quantization than a pure prefix slice from the file.

### Why the feature still keeps rollback safety

The previous behavior is preserved through:

- `SOAR_GPTQ_CALIBRATION_SAMPLING=sequential`

This keeps rollback easy if the new sampling policy underperforms on the public or private set.

## Validation Commands

Quantize with the new default strategy:

```bash
bash prepare_model.sh --input <raw_model_dir> --output <processed_model_dir>
```

Force sequential rollback for A/B comparison:

```bash
export SOAR_GPTQ_CALIBRATION_SAMPLING=sequential
bash prepare_model.sh --input <raw_model_dir> --output <processed_model_dir>
```

Correctness check:

```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_DIR> \
  --data_path ./perf_public_set.jsonl \
  --concurrency 32
```

Local summary harness:

```bash
python3 benchmark/soar/run_soar_suite.py \
  --api-base http://127.0.0.1:30000 \
  --model-path <MODEL_DIR> \
  --eval-script benchmark/soar/demo_sala/eval_model.py \
  --public-data benchmark/soar/demo_sala/perf_public_set.jsonl
```

## Result Summary Table

| Item | Before | After |
| --- | --- | --- |
| Calibration selection | first `N` rows only | configurable selection mode |
| Reproducibility | implicit file order only | explicit seed-controlled |
| Task coverage | depends on file prefix | stratified by task |
| Length coverage | depends on file prefix | stratified by `prompt_tokens` bucket |
| Rollback path | not configurable | `sequential` mode |
| Accuracy result | pending validation | pending validation |

## Rollback Instructions

If the new sampling strategy does not improve correctness:

1. set `SOAR_GPTQ_CALIBRATION_SAMPLING=sequential`
2. optionally disable task and prompt-length balancing
3. rerun `prepare_model.sh` to regenerate the GPTQ artifact

Rollback example:

```bash
export SOAR_GPTQ_CALIBRATION_SAMPLING=sequential
export SOAR_GPTQ_CALIBRATION_TASK_BALANCE=0
export SOAR_GPTQ_CALIBRATION_USE_PROMPT_TOKENS=0
bash prepare_model.sh --input <raw_model_dir> --output <processed_model_dir>
```

## Next-Step Suggestions

1. Run an A/B comparison between `sequential` and `stratified` at the same sample count of `64`.
2. If stratified improves correctness, then test `SOAR_GPTQ_GROUP_SIZE=64` as the next isolated feature.
3. If stratified does not help, inspect per-task differences on MCQ versus long-context rows before changing any quantized module coverage.