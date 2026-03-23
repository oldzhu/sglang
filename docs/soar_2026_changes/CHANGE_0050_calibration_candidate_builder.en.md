# CHANGE_0050 Calibration Candidate Builder

## Background and Motivation

Manual one-off calibration subset guessing has reached diminishing returns. Recent experiments showed that different 8-sample subsets mainly move error between `mcq|len_0_4k`, `qa|len_4k_32k`, `qa|len_32k_128k`, and occasionally `niah`, rather than producing a stable global improvement. The next step is to make candidate generation reproducible and comparisons consistent.

## Rule-Compliance Statement

This change remains within current SOAR 2026 rules.

- It only uses rows from the public `perf_public_set.jsonl` file.
- It does not modify runtime inference logic, kernels, or official scoring logic.
- It is a tooling layer for offline GPTQ calibration-data experiments under the existing preprocessing contract.

## Detailed Implementation Plan

Before change:

- Move candidate definitions out of hardcoded Python literals into a JSON config file.
- Extend the existing builder script to read that config file.
- Add multiple 8-sample candidates for controlled A/B testing.
- Add a template for recording repeated evaluation results with the same decision criteria.

## Actual Code Changes

- Updated `benchmark/soar/demo_sala/build_semantic_priority_calibration_sets.py`.
- Added `benchmark/soar/demo_sala/calibration_candidates.json`.
- Added `benchmark/soar/demo_sala/calibration_candidate_results_template.md`.

Supported workflows:

- `--list`: print candidate metadata without generating files
- default run: generate all candidate JSONL files
- `--check`: verify generated files against source data and configured indices

Initial candidate set includes:

- historical subsets:
  - `semantic_priority_8`
  - `semantic_priority_8_v2`
  - `semantic_priority_16`
- new experimental candidates:
  - `candidate_baseline_like_8`
  - `candidate_mcq_qa_mid_8`
  - `candidate_mcq_heavy_8`
  - `candidate_qa_balanced_8`

## Validation Commands

```bash
python3 -m py_compile benchmark/soar/demo_sala/build_semantic_priority_calibration_sets.py
python3 benchmark/soar/demo_sala/build_semantic_priority_calibration_sets.py --list
python3 benchmark/soar/demo_sala/build_semantic_priority_calibration_sets.py
python3 benchmark/soar/demo_sala/build_semantic_priority_calibration_sets.py --check
```

## Result Summary Table

| Artifact | Purpose | Status |
|---|---|---|
| `calibration_candidates.json` | Source of truth for candidate definitions | Added |
| `build_semantic_priority_calibration_sets.py` | Candidate generator and validator | Updated |
| `calibration_candidate_results_template.md` | Standardized experiment recording | Added |

## Rollback Instructions

- Continue using previously generated calibration files only.
- Ignore the new candidate config and template if manual experiments are preferred.
- No runtime rollback is required because this change only affects offline experiment tooling.

## Next-Step Suggestions

1. Generate all candidate JSONL files once.
2. Evaluate only 2-4 highest-value candidates first rather than all of them.
3. Use the same template for repeated runs so stability can be compared directly.