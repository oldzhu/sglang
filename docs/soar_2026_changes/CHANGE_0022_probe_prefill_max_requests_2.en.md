# CHANGE_0022_probe_prefill_max_requests_2

## 1) Background and Motivation
- After CHANGE_0021, weighted duration improved slightly and remained stable.
- Next low-risk step is to improve prefill batching utilization under queue pressure.

## 2) Rule-Compliance Statement
- Runtime launch-parameter tuning only.
- No base model replacement and no forbidden rule bypass.

## 3) Planned Change (Before Edit)
- In probe profile only:
  - `--prefill-max-requests`: `1 -> 2`
- Keep all other parameters unchanged for single-factor attribution.

## 4) Actual Code Changes
- Updated `benchmark/soar/scripts/launch_perf_probe.sh`.
- Updated probe `resolved_cmd` in `benchmark/soar/scripts/launch_profile.sh`.

## 5) Validation Commands
```bash
bash -n benchmark/soar/scripts/launch_perf_probe.sh
bash -n benchmark/soar/scripts/launch_profile.sh

bash benchmark/soar/scripts/launch_profile.sh probe
python3 benchmark/soar/run_soar_suite.py ...
```

## 6) Expected Gain / Risk
- Expected: modest S8/Smax gains via better prefill overlap.
- Risk: slightly higher memory pressure and occasional instability on long-context bursts.

## 7) Result Table Template
| Metric | Baseline | New | Delta |
|---|---:|---:|---:|
| S1 duration (s) | TBD | TBD | TBD |
| S8 duration (s) | TBD | TBD | TBD |
| Smax duration (s) | TBD | TBD | TBD |
| Weighted proxy (s) | TBD | TBD | TBD |
| Correctness overall_accuracy | TBD | TBD | TBD |

## 8) Rollback
1. Revert `--prefill-max-requests` to `1` in `benchmark/soar/scripts/launch_perf_probe.sh`.
2. Revert matching probe command display in `benchmark/soar/scripts/launch_profile.sh`.

## 9) Next Step Suggestion
- If this remains stable and helpful, proceed to one additional scheduler/memory tuning step; otherwise rollback and switch to quantization path.
