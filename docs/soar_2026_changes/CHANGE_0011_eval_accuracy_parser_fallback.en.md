# CHANGE_0011_eval_accuracy_parser_fallback

## 1) Background & Motivation
- In several fcloud runs, `run_soar_suite.py` produced `null` for `ori_accuracy` / `overall_accuracy` in `summary.json`.
- Root cause: parser expected `ori_accuracy` and `overall_accuracy` labels, while some eval outputs only print `Average Score: xx.xx%`.
- Goal: make summary parsing robust so baseline tracking can use `summary.json` directly.

## 2) SOAR Rule-Compliance Check
- Tooling-only change (result parsing), no model/kernel/runtime behavior change.
- No impact on official evaluation constraints.

## 3) Plan Before Code Change
- Modify one file: `benchmark/soar/run_soar_suite.py`.
- Add fallback regex patterns for common score lines.
- Fill missing field from the available one when only one score format is present.

## 4) Actual Code Change
- Updated `parse_accuracy_from_text(...)`:
  - `ori_accuracy` now also parses `Average Score` and `Average Accuracy`.
  - `overall_accuracy` now also parses `Overall Score` and `Relative Score`.
  - If only one value is found, it is mirrored into the missing field to avoid `null` summary fields.

## 5) Validation Commands
```bash
python3 benchmark/soar/run_soar_suite.py \
  --api-base http://127.0.0.1:30000 \
  --model-path /root/models/openbmb/MiniCPM-SALA \
  --eval-script /root/data/eval_model.py \
  --public-data /root/data/perf_public_set.jsonl
```

Check:
```bash
cat benchmark/soar/results/<timestamp>/summary.json
```

Expected: `correctness.ori_accuracy` and `correctness.overall_accuracy` are no longer `null` when eval log has score lines.

## 6) Risks
- Low. Parsing is more permissive and may map one score into both fields when only one metric is available.

## 7) Rollback
1. Revert `benchmark/soar/run_soar_suite.py`.

## 8) Next-Step Suggestions
- If needed, add a `correctness.score_source` field in a future change to indicate exact parsed line type.
