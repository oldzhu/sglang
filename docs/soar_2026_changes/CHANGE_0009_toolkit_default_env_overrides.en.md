# CHANGE_0009_toolkit_default_env_overrides

## 1) Background & Motivation
- Toolkit-default launch profile failed at startup with `Not enough memory` in some environments.
- We need simple parameterization to test whether startup OOM is caused by:
  - too aggressive `chunked-prefill-size`, and/or
  - too low automatically chosen static memory fraction.

## 2) SOAR Rule-Compliance Check
- Script-level operational change only.
- No model/kernel/inference logic modifications.
- No conflict with official evaluation constraints.

## 3) Plan Before Code Change
- Modify one script only: `benchmark/soar/scripts/launch_toolkit_default.sh`.
- Keep default behavior unchanged when env vars are unset.

## 4) Actual Code Change
- Added optional env parameters:
  - `CHUNKED_PREFILL_SIZE` (default: `32768`)
  - `MEM_FRACTION_STATIC` (optional; only passed if set)
- Launch command now uses:
  - `--chunked-prefill-size "$CHUNKED_PREFILL_SIZE"`
  - `--mem-fraction-static "$MEM_FRACTION_STATIC"` when provided

## 5) Usage Examples
```bash
# Keep toolkit default behavior
bash benchmark/soar/scripts/launch_toolkit_default.sh

# Test lower chunked prefill to reduce startup pressure
CHUNKED_PREFILL_SIZE=8192 bash benchmark/soar/scripts/launch_toolkit_default.sh

# Test static memory fraction override
MEM_FRACTION_STATIC=0.80 bash benchmark/soar/scripts/launch_toolkit_default.sh

# Test both together
CHUNKED_PREFILL_SIZE=8192 MEM_FRACTION_STATIC=0.80 bash benchmark/soar/scripts/launch_toolkit_default.sh
```

## 6) Risks
- Very low; startup-script parameterization only.

## 7) Rollback
1. Revert `benchmark/soar/scripts/launch_toolkit_default.sh`.

## 8) Next-Step Suggestions
- If this stabilizes startup, sync the same tunable behavior into profile status output in a separate change.
