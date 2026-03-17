# CHANGE_0045: SM120 Marlin Scored Runtime Selection

## Background and Motivation

`CHANGE_0043` proved that the SM120-specific Marlin path was being used during serving by emitting:

```text
[sgl-kernel] SM120 Marlin auto-config enabled: ...
```

However, the first end-to-end validation after rebuilding the patched `sgl-kernel` wheel showed that enabling the SM120 config table alone did not produce a meaningful speed improvement:

- `acc_ori`: `79.31%`
- `S1`: `124.07`
- `S8`: `26.44`
- `Smax`: `21.86`

Further inspection of the launcher showed why the previous change had limited upside:

- the Marlin launcher was still choosing the **first valid** config from a small precompiled kernel set
- it was **not** evaluating all valid candidates for the current GEMM shape
- it was **not** caching a better shape-specific decision once found

This iteration keeps the kernel family unchanged and focuses only on a safer next step: score all valid SM120 candidates at runtime, select the best one for the current shape, and cache that decision per shape.

## Rule-Compliance Statement

This change remains within the officially encouraged `路径一：量化加速` direction on the latest toolkit page:

- it tunes the Marlin W4A16 inference kernel launch path only
- it does not replace the base MiniCPM-SALA model
- it does not modify competition concurrency rules or prefix-cache behavior
- it remains compatible with the current `prepare_env.sh` + local `sgl-kernel` wheel workflow described by the latest `提交说明`

## Detailed Implementation Plan

Before change:

1. Re-check the official toolkit and competition pages to ensure path-one Marlin tuning remains allowed and submission still relies on `prepare_env.sh`.
2. Confirm that the existing SM120 launcher still returns the first valid config instead of comparing candidates.
3. Add a shape-keyed cache so config scoring is paid once per unique hot shape rather than on every GEMM launch.
4. Restrict the new scoring path to SM120 so non-SM120 behavior stays unchanged.

## Actual Code Changes

Changed file:

- `sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu`

Changes made:

1. Added a scored SM120 runtime-selection path over the existing precompiled Marlin kernel family.
2. Added a per-shape cache keyed by runtime-relevant GEMM attributes so repeated shapes reuse the selected config.
3. Kept non-SM120 behavior unchanged: it still uses the previous first-valid selection flow.
4. Extended the one-time SM120 log line to report whether the selected config came from scoring or cache reuse, together with the estimated occupancy and score.

Scoring inputs used across all valid SM120 candidates:

- M tile coverage
- N tile coverage through total tile count
- estimated parallelism from total tiles versus SM capacity
- shared-memory fit
- optional occupancy signal from `cudaOccupancyMaxActiveBlocksPerMultiprocessor`

Important implementation decision from the design discussion:

- **selection happens at runtime, per GEMM shape**
- it is **not** a single build-time or one-time global initialization choice
- the extra scoring cost is intentionally amortized through the shape cache, so the goal is to pay it once per hot shape rather than on every launch

## Discussion Notes for Later Reference

This section records the key reasoning from the implementation discussion.

### 1. What “auto-scored” means here

The launcher now evaluates all valid precompiled SM120 candidates for the current shape instead of returning the first valid config in the candidate list.

This means the selection is based on runtime shape properties such as:

- `M`, `N`, `K`
- tile coverage
- expected parallelism
- shared-memory pressure
- occupancy estimate

### 2. When the selection happens

The selection happens at **runtime**, not at build time.

- build time decides which Marlin kernel variants exist in the wheel
- runtime decides which of those already-compiled variants to launch for the current GEMM shape

In practice, this means warmup, CUDA graph capture, and any uncaptured runtime shape can each trigger a selection for that particular shape. Repeated shapes should then hit the cache.

### 3. Why not keep a single static config

A single static Marlin config is unlikely to be optimal across all serving shapes because the workload varies with:

- batch-dependent `M`
- prefill/decode path differences
- graph-captured versus uncaptured shape mix
- tile count and occupancy behavior

The design assumption is therefore:

- no single static config will consistently deliver the best performance for all hot shapes

### 4. Is the scoring overhead worth paying

There is some extra host-side cost compared with a purely static config.

The intended tradeoff is:

- scoring all valid candidates once per new shape is cheap relative to the GEMM itself
- caching the chosen config avoids repaying that cost in steady-state serving

So the practical target is **cached runtime selection**, not repeated dynamic search for every individual launch.

## Validation Commands

Build the patched wheel:

```bash
cd /root/sglang-minicpm/sgl-kernel
rm -rf build dist
python -m pip install -U uv scikit-build-core ninja
make build MAX_JOBS=2 CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1"
```

Correctness check:

```bash
python3 eval_model.py --api_base http://127.0.0.1:30000 --model_path <MODEL_DIR> --data_path <DATA_DIR>/perf_public_set.jsonl --concurrency 32
```

Speed check:

```bash
bash SOAR/bench_serving.sh http://127.0.0.1:30000
```

Runtime verification from server log:

```bash
grep -F "[sgl-kernel] SM120 Marlin auto-config enabled:" <server_log_file>
```

## Result Summary Table

| Item | Before | After |
| --- | --- | --- |
| SM120 config choice | first valid candidate | scored best valid candidate |
| Selection timing | runtime | runtime |
| Repeated-shape overhead | repeated first-valid scan | cached selection reuse |
| Non-SM120 behavior | first valid candidate | unchanged |
| Speed result | pending validation | pending validation |

## Rollback Instructions

If the scored runtime selector adds instability or fails to improve serving speed:

1. revert `sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu`
2. rebuild the `sgl-kernel` wheel
3. re-copy the wheel into the submission package directory

Rollback command:

```bash
git revert <commit>
```

## Next-Step Suggestions

1. Rebuild the patched `sgl-kernel` wheel on fcloud and confirm the new SM120 log line appears.
2. Re-run correctness first because the current observed `acc_ori` is still below a safe submission target.
3. Only if correctness is stable, compare `S1`, `S8`, and `Smax` against the `CHANGE_0041` baseline and the previous `CHANGE_0043` wheel.