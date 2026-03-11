# CHANGE_0006_unified_profile_launcher

## 1) Background & Motivation
- We already have three launch scripts (`eval-safe`, `perf-probe`, `toolkit-default`).
- Switching scripts manually is error-prone during repeated experiments.
- Goal: provide one consistent entrypoint with profile argument.

## 2) SOAR Rule-Compliance Check
- Script-only operational improvement.
- No model/kernel/runtime algorithm changes.
- No impact on official evaluation constraints.

## 3) Plan Before Code Change
- Add one shell script:
  - `benchmark/soar/scripts/launch_profile.sh`
- It dispatches to existing scripts by profile name.

## 4) Actual Code Change
- Added `launch_profile.sh` with profiles:
  - `safe` -> `launch_eval_safe.sh`
  - `probe` -> `launch_perf_probe.sh`
  - `default` -> `launch_toolkit_default.sh`
- Includes usage/help and invalid profile handling.

## 5) Usage
```bash
bash benchmark/soar/scripts/launch_profile.sh safe
bash benchmark/soar/scripts/launch_profile.sh probe
bash benchmark/soar/scripts/launch_profile.sh default
```

## 6) Risks
- Very low; delegates to existing validated scripts.

## 7) Rollback
1. Delete `benchmark/soar/scripts/launch_profile.sh`.

## 8) Next-Step Suggestions
- Optionally add profile-specific env-file loading in a separate change.
