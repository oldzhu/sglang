# CHANGE_0047 Semantic-Priority Calibration Subsets

## Background and Motivation

Recent local results indicate that GPTQ W4A16 quantization quality is now the main limiter for MiniCPM-SALA submission quality. Increasing calibration sample count alone did not improve stability. A smaller but more semantically dense subset is the next low-risk variable to test.

The current hypothesis is that calibration rows dominated by low-semantic long-context extraction patterns can distort GPTQ scale fitting for this workload. This change adds two curated subsets that emphasize `mcq` and `qa`, while keeping only minimal `niah` coverage.

## Rule-Compliance Statement

This change remains within the official SOAR 2026 guidance.

- It uses only rows from the official public calibration/evaluation file `perf_public_set.jsonl`.
- It does not modify the base model, evaluation logic, or online serving path.
- It fits the official submission model where preprocessing logic and auxiliary resources are included in the submission package and executed on the evaluation platform.
- It stays aligned with the toolkit technical path for GPTQ W4A16 preprocessing.

## Detailed Implementation Plan

Before change:

- Add a small reproducible builder script under `benchmark/soar/demo_sala/`.
- Encode two fixed index lists representing 8-sample and 16-sample semantic-priority subsets.
- Generate ready-to-use JSONL files beside `perf_public_set.jsonl`.
- Add a `--check` mode so the files can be revalidated after copy, packaging, or transfer.

## Actual Code Changes

- Added `benchmark/soar/demo_sala/build_semantic_priority_calibration_sets.py`.
- Added `benchmark/soar/demo_sala/calib_semantic_priority_8.jsonl`.
- Added `benchmark/soar/demo_sala/calib_semantic_priority_16.jsonl`.

Subset design:

- 8-sample subset: `2, 11, 17, 23, 25, 61, 76, 90`
- 16-sample subset: `2, 5, 8, 11, 17, 23, 25, 30, 31, 60, 61, 63, 66, 76, 81, 90`

Task mix:

- Prioritize `mcq` and `qa` as the main semantic carriers.
- Keep `niah` only in the 16-sample set, and only with limited coverage.
- Exclude `cwe` and `fwe` from this first curated round.

## Validation Commands

```bash
python3 -m py_compile benchmark/soar/demo_sala/build_semantic_priority_calibration_sets.py
python3 benchmark/soar/demo_sala/build_semantic_priority_calibration_sets.py
python3 benchmark/soar/demo_sala/build_semantic_priority_calibration_sets.py --check
wc -l benchmark/soar/demo_sala/calib_semantic_priority_8.jsonl
wc -l benchmark/soar/demo_sala/calib_semantic_priority_16.jsonl
```

Example quantization runs:

```bash
export SOAR_GPTQ_CALIBRATION_FILE=benchmark/soar/demo_sala/calib_semantic_priority_8.jsonl
bash benchmark/soar/demo_sala/prepare_env.sh
```

```bash
export SOAR_GPTQ_CALIBRATION_FILE=benchmark/soar/demo_sala/calib_semantic_priority_16.jsonl
bash benchmark/soar/demo_sala/prepare_env.sh
```

## Result Summary Table

| Variant | Calibration set | Public correctness | Local duration | Status |
|---|---|---:|---:|---|
| Baseline quantized | Sequential 8 | 79.31 / 75.96 | 3535.36 / 3054.16 | Existing reference |
| Baseline quantized | Sequential 16 | 77.58 / 74.18 | 3755.73 / 3707.54 | Existing reference |
| New candidate | Semantic-priority 8 | TBD | TBD | Pending validation |
| New candidate | Semantic-priority 16 | TBD | TBD | Pending validation |

## Rollback Instructions

- Stop using the new files by restoring `SOAR_GPTQ_CALIBRATION_FILE` to the previous source.
- If needed, regenerate the files from the builder script to ensure they were not modified during transfer.
- This change does not alter runtime code paths, so rollback is only a calibration-file selection change.

## Next-Step Suggestions

1. Compare semantic-priority 8 against the current sequential 8 baseline first.
2. If semantic-priority 8 improves accuracy without hurting runtime materially, test whether semantic-priority 16 is more stable across repeated runs.
3. If both fail, move to a lightweight pre-quantization screening tool instead of further manual subset guessing.