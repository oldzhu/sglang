# CHANGE_0025_workflow_one_feature_per_iteration

## 1) Background and Motivation
- Current "one tiny change" rhythm is too slow for practical iteration speed.
- Team requested a faster cadence while preserving traceability and review quality.

## 2) Rule-Compliance Statement
- This is a process-policy update in project instructions.
- It does not alter model/runtime behavior or competition evaluation logic.

## 3) Planned Change (Before Edit)
- Update workflow wording in `.github/copilot-instructions.md`:
  - from one small change per iteration
  - to one improving feature per iteration
- Keep bilingual documentation requirement, but map one doc pair to one feature iteration.

## 4) Actual Code Changes
- Updated `.github/copilot-instructions.md`:
  - `One change at a time` -> `One improving feature at a time`
  - clarified that one feature can include cohesive edits across multiple files
  - clarified docs pair maps to one feature iteration

## 5) Validation
```bash
git diff .github/copilot-instructions.md
```

## 6) Risks
- Low. Slightly larger per-iteration diffs may increase review load.
- Mitigation: keep feature scope cohesive and avoid unrelated edits.

## 7) Rollback
1. Revert updated workflow section in `.github/copilot-instructions.md`.

## 8) Next-Step Suggestion
- Apply this new feature-level cadence to upcoming quant-prep implementation.
