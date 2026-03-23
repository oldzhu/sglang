# CHANGE_0049 Targeted MCQ-QA Calibration V2

## Background and Motivation

Bucketed evaluation showed that the main remaining accuracy weaknesses are not `niah` or `fwe`, but the combination of:

- `mcq|len_0_4k`
- `qa|len_4k_32k`

The previous semantic-priority 8-sample subset improved some long-context buckets, but introduced instability in exactly these two critical regions. This change adds a second 8-sample curated subset that is explicitly targeted at those weak buckets.

## Rule-Compliance Statement

This change remains compliant with SOAR 2026 requirements.

- It only selects rows from the public `perf_public_set.jsonl` file.
- It does not alter model weights directly, runtime kernels, or evaluation logic.
- It remains an offline GPTQ calibration-data optimization, which is allowed by the official preprocessing workflow.

## Detailed Implementation Plan

Before change:

- Keep the existing semantic-priority 8 and 16 subsets unchanged.
- Extend the existing builder script with one additional named output.
- Generate a new `8_v2` file that emphasizes short `mcq` anchors and mid-length `qa` anchors, with only one long `qa` anchor retained.

## Actual Code Changes

- Updated `benchmark/soar/demo_sala/build_semantic_priority_calibration_sets.py`.
- Added `benchmark/soar/demo_sala/calib_semantic_priority_8_v2.jsonl`.

New subset indices:

- `1, 2, 4, 8, 61, 63, 66, 76`

Design rationale:

- `1, 2, 4, 8`: preserve the stable short-`mcq` anchor pattern from the original sequential 8 baseline.
- `61, 63, 66`: explicitly increase `qa|len_4k_32k` coverage.
- `76`: retain one `qa|len_32k_128k` anchor so long `qa` does not collapse completely.
- Exclude `niah`, `fwe`, and `cwe` from this targeted repair iteration.

## Validation Commands

```bash
python3 -m py_compile benchmark/soar/demo_sala/build_semantic_priority_calibration_sets.py
python3 benchmark/soar/demo_sala/build_semantic_priority_calibration_sets.py
python3 benchmark/soar/demo_sala/build_semantic_priority_calibration_sets.py --check
wc -l benchmark/soar/demo_sala/calib_semantic_priority_8_v2.jsonl
```

Example run:

```bash
export SOAR_GPTQ_CALIBRATION_FILE=benchmark/soar/demo_sala/calib_semantic_priority_8_v2.jsonl
bash benchmark/soar/demo_sala/prepare_env.sh
```

## Result Summary Table

| Variant | Focus | Status |
|---|---|---|
| `semantic_priority_8` | General semantic subset | Existing |
| `semantic_priority_16` | Larger semantic subset | Existing |
| `semantic_priority_8_v2` | Repair `mcq|len_0_4k` and `qa|len_4k_32k` | Added |

## Rollback Instructions

- Continue using `calib_semantic_priority_8.jsonl` or the original calibration source.
- Regenerate the files with the builder script if needed.
- No codepath rollback is required beyond switching the calibration file path.

## Next-Step Suggestions

1. Compare `semantic_priority_8_v2` directly against the original sequential 8 baseline.
2. Use `eval_model_001.py` to confirm whether `mcq|len_0_4k` and `qa|len_4k_32k` recover together.
3. Only consider a further v3 if the new subset improves both weak buckets without pushing total accuracy back under the safety margin.