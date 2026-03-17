# CHANGE_0042: Dense Profile Scheduler And Warmup Tuning

## Background and motivation

The current weekly submission profile uses the stable `fp8 + gpu graph + force-dense-minicpm` path, but its serving configuration is conservative for throughput. In particular, the profile was limited to `--prefill-max-requests 1`, `--max-running-requests 20`, and `--skip-server-warmup`.

This combination protects stability, but it can leave throughput on the table in the official `S8` and `Smax` settings. The goal of this iteration is to improve request packing and remove first-request startup cost while keeping the stable dense MiniCPM path unchanged.

## Rule-compliance statement

This change remains within SOAR 2026 allowed optimization scope:

- It only tunes inference runtime behavior and submission startup behavior.
- It does not modify the base model architecture or replace the official model.
- It does not enable forbidden prefix cache behavior.
- It preserves the official evaluation concurrency model.
- It keeps the existing stable quantization path: `gptq_marlin` weights and `fp8_e5m2` KV cache.

## Detailed implementation plan before change

1. Keep `--force-dense-minicpm` so MiniCPM sparse attention stays disabled.
2. Keep `--kv-cache-dtype fp8_e5m2` and `--quantization gptq_marlin` unchanged.
3. Increase `--prefill-max-requests` from `1` to `2` to allow modest prefill batching.
4. Increase `--max-running-requests` from `20` to `28` to improve overlap and queue utilization.
5. Reduce `--schedule-conservativeness` from `1.0` to `0.85` to make scheduler token admission less conservative.
6. Remove `--skip-server-warmup` so the server executes its warmup request before evaluation traffic arrives.

## Actual code changes

Updated [benchmark/soar/demo_sala/prepare_env.sh](/home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh) to:

- change `--prefill-max-requests 1` to `--prefill-max-requests 2`
- change `--max-running-requests 20` to `--max-running-requests 28`
- change `--schedule-conservativeness 1.0` to `--schedule-conservativeness 0.85`
- remove `--skip-server-warmup`

No code path changes were made to MiniCPM dense attention, Lightning attention, sparse attention, or Marlin kernels in this iteration.

## Validation commands

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

Shell validation:

```bash
bash -n benchmark/soar/demo_sala/prepare_env.sh
```

## Result summary table

| Metric | Baseline (`CHANGE_0041`) | New (`CHANGE_0042`) |
| --- | --- | --- |
| `acc_ori` | 80.07 official second score | pending |
| `S1` | 458.78 official second score | pending |
| `S8` | 634.35 official second score | pending |
| `Smax` | 1140.66 official second score | pending |
| Stability | stable | pending validation |

## Rollback instructions

If this tuning regresses throughput or stability, revert the four serving-side changes in [benchmark/soar/demo_sala/prepare_env.sh](/home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh):

- restore `--prefill-max-requests 1`
- restore `--max-running-requests 20`
- restore `--schedule-conservativeness 1.0`
- re-add `--skip-server-warmup`

## Next-step suggestions

1. Run correctness and speed validation on fcloud to determine whether the gains appear mainly in `S8` and `Smax`.
2. If this profile improves but remains short of target, make the next feature a controlled scheduler sweep around `prefill-max-requests 2/4` and `max-running-requests 28/36`.
3. If serving-side tuning saturates, move to SM120-specific Marlin kernel tuning before attempting sparse FP8 recovery or EAGLE3.