# CHANGE_0043: SM120 Marlin Kernel Tuning With Submission Wheel Install

## Background and motivation

The current stable MiniCPM-SALA submission path remains `gptq_marlin + fp8_e5m2 + force-dense-minicpm`. The official toolkit explicitly recommends a path-one optimization that targets Marlin tile and warp tuning for 6000D / SM120 GPUs.

The existing Marlin launcher still uses a small generic execution-config table and does not branch on SM120 hardware. That leaves a plausible low-risk kernel optimization opportunity in the hottest quantized linear path.

During the previous serving-side experiment (`CHANGE_0042`), correctness regressed to `acc_ori 79.40`, so this iteration restores the last known-good serving baseline while moving the optimization effort to the Marlin kernel path.

## Rule-compliance statement

This change is compliant with the latest official SOAR 2026 guidance:

- It follows path one from the toolkit: GPTQ W4A16 + Marlin + FP8 KV cache.
- It implements kernel scheduling optimization for the official RTX PRO / SM120 environment.
- It does not replace the official base model.
- It does not enable forbidden prefix cache behavior.
- It preserves the official evaluation concurrency model.
- It uses `prepare_env.sh` to install a patched local wheel artifact, which is explicitly allowed by the submission model.

## Detailed implementation plan before change

1. Keep the stable serving profile from `CHANGE_0041` as the accuracy-safe baseline.
2. Patch repo-root `sgl-kernel` Marlin launcher logic, not the submission mirror.
3. Add an SM120-specific candidate thread-config table for Marlin auto-config.
4. Keep existing kernel math and reduction behavior unchanged.
5. Build a patched `sgl-kernel` wheel from the repo root.
6. Copy that wheel into `benchmark/soar/demo_sala/` manually.
7. Update `prepare_env.sh` so the submission installs exactly one local `sgl-kernel` wheel with `--force-reinstall --no-deps`.
8. Emit a one-time runtime log line from the patched Marlin launcher so test logs can confirm SM120 auto-config was actually used.

## Actual code changes

### 1. SM120-specific Marlin auto-config

Updated [sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu](/home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu) to:

- add SM120-specific small-batch and large-batch candidate config tables
- detect SM120-or-later GPUs in the Marlin launcher
- choose the SM120 candidate table during auto-config on SM120 GPUs
- print a one-time line to stderr showing the selected auto-config for the first launch

The new SM120 candidate ordering broadens the search to include wider `thread_n` and higher-occupancy `256`-thread options before falling back to the older generic configurations.

### 2. Distinguishable patched wheel version

Updated [sgl-kernel/pyproject.toml](/home/oldzhu/sglang/sgl-kernel/pyproject.toml) version from `0.3.20` to `0.3.20.post1` so the patched wheel is easy to identify and copy into the submission directory.

### 3. Submission installs the local patched kernel wheel

Updated [benchmark/soar/demo_sala/prepare_env.sh](/home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh) to:

- require exactly one local `sgl-kernel` wheel in the submission directory
- install that wheel with `uv pip install --force-reinstall --no-deps`
- restore the last known-good serving settings from `CHANGE_0041`

This avoids shipping the entire `sgl-kernel` source tree in the submission package while still making the patched kernel active during official runs.

## How to verify the SM120 config was used

After the patched wheel is installed and the server starts, inspect the server log for a one-time line like:

```text
[sgl-kernel] SM120 Marlin auto-config enabled: M=... N=... K=... thread_m_blocks=... thread_n=... thread_k=... num_threads=...
```

If that line appears during correctness or `S1/S8/Smax` runs, the patched SM120 config path was entered.

Also verify that `prepare_env.sh` reports the local wheel path it installs.

## Wheel build and copy commands

Build the patched wheel from the repo root:

```bash
cd sgl-kernel
./build.sh 3.10 12.8 x86_64
```

Copy the resulting wheel into the submission directory:

```bash
cp dist/*/sgl_kernel-0.3.20.post1-*.whl ../benchmark/soar/demo_sala/
```

If your build output layout differs, copy the generated `sgl_kernel-0.3.20.post1-*.whl` file manually into `benchmark/soar/demo_sala/`.

## Validation commands

Shell validation:

```bash
bash -n benchmark/soar/demo_sala/prepare_env.sh
```

Correctness:

```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_DIR> \
  --data_path <DATA_DIR>/perf_public_set.jsonl \
  --concurrency 32
```

Speed:

```bash
export SPEED_DATA_S1=<PATH_TO_S1_JSONL>
export SPEED_DATA_S8=<PATH_TO_S8_JSONL>
export SPEED_DATA_SMAX=<PATH_TO_SMAX_JSONL>

bash SOAR/bench_serving.sh http://127.0.0.1:30000
```

## Result summary table

| Metric | Baseline (`CHANGE_0041`) | New (`CHANGE_0043`) |
| --- | --- | --- |
| `acc_ori` | 80.07 official second score | pending |
| `S1` | 458.78 official second score | pending |
| `S8` | 634.35 official second score | pending |
| `Smax` | 1140.66 official second score | pending |
| Local wheel install | no | yes |
| SM120 Marlin log | no | pending |

## Rollback instructions

1. Revert [sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu](/home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu) to the generic config tables.
2. Revert [sgl-kernel/pyproject.toml](/home/oldzhu/sglang/sgl-kernel/pyproject.toml) version bump.
3. Remove the local `sgl-kernel` wheel install block from [benchmark/soar/demo_sala/prepare_env.sh](/home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh).
4. Fall back to the original pip-provided `sgl-kernel` package.

## Next-step suggestions

1. Build the patched wheel and run correctness first to confirm the kernel-only change keeps accuracy stable.
2. Compare `S1/S8/Smax` against the `CHANGE_0041` baseline, not the invalid `CHANGE_0042` branch.
3. If the log confirms SM120 auto-config but the gain is small, run one more SM120 table-order iteration before considering sparse FP8 recovery.