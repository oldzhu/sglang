# CHANGE_0041_minicpm_fp8_force_dense_submission_profile

## 1) Background & Motivation
- Problem statement: `fp8 + gpu graph` remained unstable for MiniCPM sparse decode and continued to fail at CUDA graph replay, while `fp8 + gpu graph + --force-dense-minicpm` completed correctness evaluation without the replay crash.
- Why this matters: this provides a submission-ready runtime profile for the current weekly ranking while preserving the main FP8 KV-cache and CUDA graph speedups.
- Objective of this iteration: switch the submission launch profile to `--kv-cache-dtype fp8_e5m2 --force-dense-minicpm` so this week's package uses the fastest currently validated stable configuration.

## 2) SOAR Rule-Compliance Check
- Latest official references re-checked before this change: `https://soar.openbmb.cn/competition` and `https://soar.openbmb.cn/toolkit`.
- Allowed by rules because: this is a runtime serving-argument adjustment within the official `prepare_env.sh` submission flow, and it stays within the encouraged KV-cache optimization direction.
- Constraints respected: no base model replacement, no prefix-cache re-enable, no concurrency changes, no submission interface changes.
- Accuracy/stability risk: low for this iteration, because the selected launch profile already passed the user's latest fcloud correctness run.

## 3) Detailed Implementation Plan Before Change
Scope for this iteration:

1. Update the submission launch args in `prepare_env.sh` to enable FP8 KV cache.
2. Force MiniCPM dense mode to bypass the currently unstable sparse decode graph path.
3. Stop appending the explicit `--log-level info` override in the submission script.

Files/functions to change:

1. `benchmark/soar/demo_sala/prepare_env.sh`

Expected gain:

1. Preserve `fp8 + gpu graph` for the current weekly submission.
2. Avoid the sparse replay crash by disabling MiniCPM sparse attention.
3. Keep the submission script aligned with the official `prepare_env.sh` environment-variable contract.

## 4) Actual Code Changes
Changed file:

1. `benchmark/soar/demo_sala/prepare_env.sh`

Actual edits:

1. Added `--kv-cache-dtype fp8_e5m2` to the exported `SGLANG_SERVER_ARGS` for the GPTQ serving profile.
2. Added `--force-dense-minicpm` to the same launch profile.
3. Commented out the extra `--log-level info` append line.

What stays unchanged:

1. GPTQ Marlin remains enabled for the weekly submission profile.
2. CUDA graph remains enabled.
3. Existing serving limits and memory-related launch arguments remain unchanged.

## 5) Validation Commands
Syntax validation:

```bash
bash -n benchmark/soar/demo_sala/prepare_env.sh
```

Recommended correctness validation on fcloud:

```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_PATH> \
  --data_path benchmark/soar/demo_sala/perf_public_set.jsonl \
  --concurrency 32
```

Recommended speed validation on fcloud:

```bash
python3 benchmark/soar/run_soar_suite.py \
  --base-url http://127.0.0.1:30000 \
  --dataset-profile heavy
```

## 6) Result Summary Table
| Area | Previous stable candidate | New submission profile | Delta | Notes |
|---|---:|---:|---:|---|
| KV-cache dtype | non-FP8 or unstable FP8 sparse path | FP8 E5M2 | positive | keeps FP8 decode optimization |
| MiniCPM sparse path | enabled | disabled with `--force-dense-minicpm` | trade-off | avoids current replay crash |
| Correctness run | crashing under FP8 sparse graph | `acc_ori: 80.84%` | positive | user-reported fcloud result |
| Speed proxy S1 | `142.00` | `136.32` | better | user-reported |
| Speed proxy S8 | `31.21` | `28.65` | better | user-reported |
| Speed proxy Smax | `25.68` | `23.61` | better | user-reported |

## 7) Rollback Instructions
If this iteration is not beneficial or causes regressions:

1. Remove `--kv-cache-dtype fp8_e5m2` and `--force-dense-minicpm` from `benchmark/soar/demo_sala/prepare_env.sh`.
2. Restore the explicit `--log-level info` export if needed for debugging.
3. Re-run correctness and speed tests on the previous launch profile.

## 8) Next-Step Suggestions
1. Use this profile for the current weekly submission if the final fcloud rerun remains stable.
2. Continue root-cause debugging of the `fp8 + gpu graph + sparse` replay crash so sparse MiniCPM can be re-enabled later.
3. Re-check whether force-dense remains optimal on the hidden speed set once sparse replay is fixed.