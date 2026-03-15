## Background and motivation

MiniCPM FP8 decode now reaches the CUDA-graph replay stage, where failures appear as illegal memory access during replay rather than an earlier Python-side sparse scorer error. While investigating the decode metadata path, the decode-time sparse table write path in `python/sglang/srt/mem_cache/common.py` was found to use tuple indexing `(start, end)` instead of the intended range write `slice(start, end)`.

That difference matters for PyTorch indexing semantics:

- `table[row, slice(3, 4)] = [999]` writes exactly one slot at index `3`
- `table[row, (3, 4)] = [999]` selects two explicit positions, `3` and `4`, and broadcasts the value

For decode, `token_per_req=1`, so sparse decode updates often intend to append a single new sparse entry. Tuple indexing can therefore duplicate-write the next slot and silently corrupt `req_to_sparse_k1_token` / `req_to_sparse_k2_token`. Once those corrupted tables are consumed by replayed decode kernels, the failure can surface as a CUDA illegal access inside graph replay.

## Rule-compliance statement

This change is a runtime correctness fix in sparse decode metadata handling. It does not alter model weights, evaluation concurrency, forbidden caching behavior, or any submission interface. It remains within SOAR competition guardrails.

Toolkit and rule references were previously reviewed for this optimization thread, and this change does not depend on any rule-sensitive workaround.

## Detailed implementation plan

1. Fix decode-time sparse table writes to use `slice(start, end)` rather than tuple indexing.
2. Add temporary bounds checks for sparse k1/k2 decode writes so overflow fails early with a deterministic Python-side error instead of a later asynchronous CUDA illegal access.
3. Keep the bounds checks only as a stabilization/debugging aid during FP8 decode investigation.
4. If the crash is resolved and the path is performance-stable, consider removing the explicit bounds checks in a later cleanup iteration to avoid any unnecessary Python overhead in the hot path.

## Actual code changes

File changed:

- `python/sglang/srt/mem_cache/common.py`

Behavioral changes:

- Replaced decode sparse k1 writes from tuple indexing to `slice(k1_len, k1_end)`.
- Replaced decode sparse k2 writes from tuple indexing to `slice(k2_len, k2_end)`.
- Added temporary explicit bounds checks against `req_to_sparse_k1_token.shape[1]` and `req_to_sparse_k2_token.shape[1]`.
- Added error messages that include request index, computed sparse offset, appended token count, and table width.

## Validation commands

Correctness with blocking stack traces from the server process:

```bash
CUDA_LAUNCH_BLOCKING=1 python3 -m sglang.launch_server ...
```

Then run the correctness client separately:

```bash
python3 eval_model.py --api_base http://127.0.0.1:30000 --model_path <MODEL_PATH> --data_path benchmark/soar/demo_sala/perf_public_set.jsonl --concurrency 32
```

Static validation in workspace:

```bash
python3 -m compileall python/sglang/srt/mem_cache/common.py
```

Serving sanity after correctness is stable:

```bash
python3 benchmark/soar/demo_sala/run_serving_bench.py ...
```

## Result summary table

| Item | Baseline before CHANGE_0040 | After CHANGE_0040 |
| --- | --- | --- |
| Decode sparse write semantics | Tuple indexing, may duplicate-write adjacent slots | Explicit slice write of intended range |
| Sparse decode overflow handling | May surface later as CUDA illegal access | Early deterministic Python-side bounds failure |
| Correctness eval stability | Pending repro, currently crashes in decode CUDA graph replay | Pending user validation |
| Serving speed | FP8 path already promising | Pending post-fix benchmark |

## Rollback instructions

If this change is shown to be harmful:

1. Revert the `slice(...)` and temporary bounds-check edits in `python/sglang/srt/mem_cache/common.py`.
2. Re-run the same correctness command with `CUDA_LAUNCH_BLOCKING=1` to confirm the behavior difference.
3. Only keep the rollback if the new evidence shows this change itself introduced a regression.

Default guidance: do not rollback automatically just because the crash persists. If the fix is logically correct and harmless, keep it while continuing to narrow the remaining decode CUDA-graph fault.

## Next-step suggestions

1. Reproduce the FP8 correctness run with this fix and `CUDA_LAUNCH_BLOCKING=1` still enabled on the sglang server.
2. If the crash persists, compare the new blocking stack with the pre-CHANGE_0040 replay stack to decide whether the remaining issue is metadata-related or inside a replayed sparse kernel.
3. Once the crash is fully resolved, evaluate whether the temporary bounds checks should be removed for final competition tuning.