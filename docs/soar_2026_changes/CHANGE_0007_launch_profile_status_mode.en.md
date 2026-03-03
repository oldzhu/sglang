# CHANGE_0007_launch_profile_status_mode

## 1) Background & Motivation
- The unified launcher exists, but users still need a way to verify resolved command/env before actual launch.
- Goal: add a safe inspection mode (`status` / `--dry-run`) without starting the server.

## 2) SOAR Rule-Compliance Check
- Script-only operational enhancement.
- No inference/runtime algorithm change.

## 3) Plan Before Code Change
- Modify one file: `benchmark/soar/scripts/launch_profile.sh`.
- Add mode argument with default `run`.

## 4) Actual Code Change
- New usage:
  - `bash benchmark/soar/scripts/launch_profile.sh <safe|probe|default> [run|status|--dry-run]`
- Added `status`/`--dry-run` behavior:
  - print selected profile
  - print target script path
  - print resolved env defaults (`MODEL_PATH`, `HOST`, `PORT`, and `PYTORCH_CUDA_ALLOC_CONF` where applicable)
  - print resolved launch command
  - exit without launching
- Default behavior remains unchanged (`run`).

## 5) Validation Commands
```bash
bash benchmark/soar/scripts/launch_profile.sh safe status
bash benchmark/soar/scripts/launch_profile.sh probe --dry-run
bash benchmark/soar/scripts/launch_profile.sh default status
```

## 6) Risks
- Very low; shell control-flow change only.

## 7) Rollback
1. Revert `benchmark/soar/scripts/launch_profile.sh`.

## 8) Next-Step Suggestions
- Optionally print whether the server port is currently occupied in status mode (separate change).
