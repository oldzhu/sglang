# CHANGE_0023_rollback_prefill_max_requests_2

## 1) Background and Motivation
- CHANGE_0022 increased probe `--prefill-max-requests` from 1 to 2.
- New test showed only tiny speed gain but notable correctness drop (`ori_accuracy` to 79.18), so rollback is required.

## 2) Rule-Compliance Statement
- This is a rollback of runtime launch arguments.
- No model replacement or forbidden strategy is introduced.

## 3) Planned Change (Before Edit)
- Revert probe setting:
  - `--prefill-max-requests`: `2 -> 1`
- Keep all other probe settings unchanged.

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

## 6) Result Summary
- Decision: reject CHANGE_0022 and rollback.
- Reason: correctness risk outweighs tiny speed gain.

## 7) Rollback (This Change)
- Completed by restoring probe prefill request limit to 1.

## 8) Next Step Suggestion
- Continue with a different single low-risk knob or shift to quantization path planning if runtime gains plateau.
