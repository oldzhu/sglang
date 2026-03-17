# CHANGE_0044: DeepGEMM Pin Build Unblock

## Background and Motivation

`CHANGE_0043` added an SM120-specific Marlin tuning path and a submission-side local `sgl-kernel` wheel install workflow. The next required step is to build the patched `sgl-kernel` wheel on the user's fcloud instance without Docker.

That Docker-free build path is present and valid (`make build` / `uv build --wheel --no-build-isolation`), but the build currently fails during CMake dependency population before the Marlin code is compiled. The failure is caused by a stale `repo-deepgemm` git pin in `sgl-kernel/CMakeLists.txt`:

- old pin: `54f99a8af537b3c6eb4819b69907ccbe2b600792`
- observed error: `fatal: reference is not a tree`

This iteration unblocks wheel generation by replacing the dead DeepGEMM revision with a live immutable commit from the same fork.

## Rule-Compliance Statement

This change remains within the officially encouraged `路径一：量化加速` scope on the latest toolkit page:

- it does not replace the base MiniCPM-SALA model
- it does not alter evaluation concurrency rules or re-enable forbidden prefix cache behavior
- it preserves the current `prepare_env.sh`-based submission interface described in the latest `提交说明`
- it improves reproducibility by pinning to a valid immutable git commit rather than a floating branch

## Detailed Implementation Plan

Before change:

1. Confirm the latest official toolkit and competition pages still allow custom environment setup through `prepare_env.sh` and still recommend the path-one Marlin optimization direction.
2. Confirm the DeepGEMM pin in `sgl-kernel/CMakeLists.txt` is the direct cause of the build failure.
3. Replace the dead DeepGEMM commit with a live immutable commit from `sgl-project/DeepGEMM`, preferring the fork's `sgl-release` branch head over `main` to reduce integration risk.
4. Re-run the Docker-free wheel build on fcloud.

## Actual Code Changes

Changed file:

- `sgl-kernel/CMakeLists.txt`

Change made:

- Updated `repo-deepgemm` `GIT_TAG` from the removed commit `54f99a8af537b3c6eb4819b69907ccbe2b600792`
- New pinned commit: `ffe2b6b97420a9f8c58268ca55755168e6e2f360`

Rationale:

- the old pin no longer exists upstream and blocks all local source builds
- the new pin is a concrete immutable commit on the fork's `sgl-release` ref
- the change is isolated to third-party fetch configuration and does not modify runtime serving logic or the SM120 Marlin kernel implementation itself

## Validation Commands

Dependency pin verification:

```bash
cd /root/sglang-minicpm/sgl-kernel
git ls-remote https://github.com/sgl-project/DeepGEMM | grep ffe2b6b97420a9f8c58268ca55755168e6e2f360
```

Docker-free wheel build:

```bash
cd /root/sglang-minicpm/sgl-kernel
python -m pip install -U uv scikit-build-core ninja
make build MAX_JOBS=2 CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1"
```

Submission-side install check after copying the built wheel into the demo package directory:

```bash
cd /root/sglang-minicpm
bash -n benchmark/soar/demo_sala/prepare_env.sh
```

Correctness and speed validation after wheel install:

```bash
python3 eval_model.py --api_base http://127.0.0.1:30000 --model_path <MODEL_DIR> --data_path <DATA_DIR>/perf_public_set.jsonl --concurrency 32
```

```bash
bash SOAR/bench_serving.sh http://127.0.0.1:30000
```

Runtime verification of the SM120 Marlin path:

```bash
grep -F "[sgl-kernel] SM120 Marlin auto-config enabled:" <server_log_file>
```

## Result Summary Table

| Item | Before | After |
| --- | --- | --- |
| DeepGEMM source fetch | Fails at removed commit | Uses live immutable commit |
| Docker-free `sgl-kernel` wheel build | Blocked during CMake populate | Unblocked at dependency-fetch stage |
| Serving profile | `CHANGE_0041` baseline + `CHANGE_0043` wheel workflow | unchanged |
| SM120 Marlin runtime logic | present | unchanged |

## Rollback Instructions

If the newer DeepGEMM pin introduces an incompatibility:

1. revert the `repo-deepgemm` `GIT_TAG` line in `sgl-kernel/CMakeLists.txt`
2. delete the failed `sgl-kernel/build` directory
3. retry with another immutable `sgl-project/DeepGEMM` commit after validating that the ref exists upstream

Rollback command:

```bash
git revert <commit>
```

## Next-Step Suggestions

1. Build the patched `sgl-kernel` wheel on fcloud with the low-parallelism command above.
2. Copy the produced `sgl-kernel` wheel into `benchmark/soar/demo_sala/` and re-run correctness first.
3. If correctness holds, run `S1`, `S8`, and `Smax` and compare only against the `CHANGE_0041` baseline.