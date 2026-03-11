# CHANGE_0005_launch_profiles

## 1) Background & Motivation
- We need reproducible startup profiles for different testing goals:
  - Eval-safe (OOM-resistant correctness runs)
  - Perf-probe (higher throughput exploration)
  - Toolkit-default (alignment check with SOAR toolkit documented defaults)

## 2) SOAR Rule-Compliance Check
- This change only adds launch helper scripts.
- No inference kernel/model code modifications.
- Does not bypass official evaluation constraints.

## 3) Plan Before Code Change
- Add three shell scripts under `benchmark/soar/scripts`.
- Keep script logic simple and explicit.

## 4) Actual Code Change
- Added:
  - `benchmark/soar/scripts/launch_eval_safe.sh`
  - `benchmark/soar/scripts/launch_perf_probe.sh`
  - `benchmark/soar/scripts/launch_toolkit_default.sh`

## 5) Notes on Toolkit Default
- Based on SOAR toolkit documentation examples, the default argument style includes:
  - `--disable-radix-cache --attention-backend flashinfer --chunked-prefill-size 32768`
- Real platform behavior is controlled by official evaluation pipeline, but this script mirrors the documented default example.

## 6) Validation Commands
```bash
bash benchmark/soar/scripts/launch_eval_safe.sh
bash benchmark/soar/scripts/launch_perf_probe.sh
bash benchmark/soar/scripts/launch_toolkit_default.sh
```

## 7) Risks
- Very low. Script-only operational convenience change.

## 8) Rollback
1. Delete the three scripts above.

## 9) Next-Step Suggestions
- Add a small wrapper to choose profile by argument (`safe|probe|default`) in a separate change.
