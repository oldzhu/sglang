# CHANGE_0021_probe_prefill_32768_tuning

## 1) Background and Motivation
- Baseline (3x prompts) shows S1 is the dominant bottleneck in weighted duration.
- We apply a low-risk serving-arg optimization first to improve long-context prefill throughput.

## 2) Rule-Compliance Statement
- This is a runtime launch-parameter optimization only.
- No base model replacement, no forbidden prefix-cache behavior, no concurrency-rule bypass.

## 3) Planned Change (Before Edit)
- In `probe` profile only:
  - `--chunked-prefill-size`: `8192 -> 32768`
  - `--max-prefill-tokens`: `16384 -> 32768`
- Keep all other knobs unchanged for isolated attribution.

## 4) Actual Code Changes
- Updated `benchmark/soar/scripts/launch_perf_probe.sh` with the new prefill values.
- Updated the probe `resolved_cmd` string in `benchmark/soar/scripts/launch_profile.sh` to stay consistent with status/dry-run output.

## 5) Validation Commands
```bash
bash -n benchmark/soar/scripts/launch_perf_probe.sh
bash -n benchmark/soar/scripts/launch_profile.sh

bash benchmark/soar/scripts/launch_profile.sh probe
python3 benchmark/soar/run_soar_suite.py ...
```

## 6) Expected Gain / Risk
- Expected: S1 and weighted duration improvement under long input loads.
- Risk: potential memory pressure increase and occasional instability under high queue pressure.

## 7) Result Table Template
| Metric | Baseline | New | Delta |
|---|---:|---:|---:|
| S1 duration (s) | TBD | TBD | TBD |
| S8 duration (s) | TBD | TBD | TBD |
| Smax duration (s) | TBD | TBD | TBD |
| Weighted proxy (s) | TBD | TBD | TBD |
| Correctness overall_accuracy | TBD | TBD | TBD |

## 8) Rollback
1. Revert `benchmark/soar/scripts/launch_perf_probe.sh` prefill values to previous settings.
2. Revert matching probe command string in `benchmark/soar/scripts/launch_profile.sh`.

## 9) Next Step Suggestion
- If this change is stable and improves weighted duration, continue with one additional low-risk serving-arg tuning in a separate iteration.
