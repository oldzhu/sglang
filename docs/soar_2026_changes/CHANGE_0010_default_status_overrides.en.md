# CHANGE_0010_default_status_overrides

## 1) Background & Motivation
- `CHANGE_0009` added `CHUNKED_PREFILL_SIZE` and `MEM_FRACTION_STATIC` overrides to toolkit-default launch.
- `launch_profile.sh status` did not show these effective values, which made troubleshooting harder.

## 2) SOAR Rule-Compliance Check
- Script-only visibility improvement.
- No model/inference behavior changes.

## 3) Plan Before Code Change
- Update one file: `benchmark/soar/scripts/launch_profile.sh`.
- Show default profile override values in status mode.

## 4) Actual Code Change
- In `default` profile status output, now prints:
  - `CHUNKED_PREFILL_SIZE` (default `32768`)
  - `MEM_FRACTION_STATIC` (`<unset>` when not provided)
- Updated resolved command text to reflect env-driven chunked prefill and optional mem-fraction argument.

## 5) Validation
```bash
bash benchmark/soar/scripts/launch_profile.sh default status
CHUNKED_PREFILL_SIZE=8192 MEM_FRACTION_STATIC=0.80 bash benchmark/soar/scripts/launch_profile.sh default --dry-run
```

## 6) Risks
- Very low (status output only).

## 7) Rollback
1. Revert `benchmark/soar/scripts/launch_profile.sh`.

## 8) Next-Step Suggestions
- Next change candidate: improve `run_soar_suite.py` accuracy parsing fallback for eval logs where `ori_accuracy/overall_accuracy` are missing.
