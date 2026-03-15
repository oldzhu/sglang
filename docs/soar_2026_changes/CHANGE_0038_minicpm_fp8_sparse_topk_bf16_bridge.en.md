# CHANGE_0038_minicpm_fp8_sparse_topk_bf16_bridge

## 1) Background & Motivation
- Problem statement: after fixing the MiniCPM FlashInfer FP8 dtype/backend contract, the first real request still failed during sparse top-k scoring with `RuntimeError: FlashAttention only support fp16 and bf16 data type`.
- Why this matters: this showed the active blocker was no longer FlashInfer startup or wrapper planning, but MiniCPM's separate sparse-routing scorer.
- Objective of this iteration: preserve FP8 KV cache for the actual cache storage and attention path, while bridging the sparse top-k scorer through BF16 so the unsupported kernel path can run.

## 2) SOAR Rule-Compliance Check
- Latest official references re-checked before this change: `https://soar.openbmb.cn/competition` and `https://soar.openbmb.cn/toolkit`.
- Allowed by rules because: this is a runtime/kernel-compatibility optimization in the inference stack and stays within the officially encouraged KV-cache optimization direction.
- Constraints respected: no model-weight changes, no prefix-cache re-enable, no concurrency changes, no submission-interface changes.
- Accuracy/stability risk: low to medium. The bridge changes only the sparse scoring scratch path, but correctness must still be revalidated.

## 3) Detailed Implementation Plan Before Change
Scope for this iteration:

1. Detect FP8 KV-cache mode in MiniCPM sparse scoring.
2. Cast only the sparse top-k scorer inputs to BF16.
3. Keep the actual KV cache and FlashInfer attention path in FP8.

Files/functions to change:

1. `python/sglang/srt/layers/attention/minicpm_backend.py`
   - `sparse_get_topk_impl(...)`

Expected gain:

1. Unblock prefill/decode sparse routing under FP8 KV cache.
2. Preserve the core FP8 KV-cache memory/decode benefits.
3. Keep the fallback narrowly scoped to a kernel path that does not support FP8.

## 4) Actual Code Changes
Changed file:

1. `python/sglang/srt/layers/attention/minicpm_backend.py`

Actual edit:

1. In `sparse_get_topk_impl(...)`, when `kv_cache_dtype` starts with `fp8`, cast:
   - `query_layer`
   - `compressed_k`
   - `compressed_k2`
   to `torch.bfloat16` before calling the sparse top-k scoring kernels.

What stays unchanged:

1. Real KV-cache storage remains FP8.
2. FlashInfer prefill/decode attention path remains FP8-aware.
3. The sparse scorer still returns the same routing output shape and semantics.

## 5) Why No Cast Back To FP8 Is Needed
The BF16 bridge is only for sparse scoring scratch inputs.

1. The scorer consumes query/compressed-key tensors.
2. It outputs `topk_idx` block selections.
3. It does not rewrite the KV cache.

Therefore the dataflow is:

1. KV cache stored in FP8
2. Sparse scoring temporarily uses BF16 tensors
3. Routing indices are produced
4. Actual attention continues to use the FP8 KV-cache path

This is not a `FP8 -> BF16 -> FP8` round-trip on the real cache contents.

## 6) Validation Commands
Syntax / import validation:

```bash
python3 -m compileall \
  python/sglang/srt/layers/attention/minicpm_backend.py
```

Recommended first runtime smoke test:

```bash
export SGLANG_MINICPM_FLASHINFER_PREFILL_BACKEND=auto
python3 -m sglang.launch_server \
  --model-path <MODEL_PATH> \
  --attention-backend minicpm_flashinfer \
  --kv-cache-dtype fp8_e5m2 \
  --disable-cuda-graph
```

Correctness check after successful launch:

```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_PATH> \
  --data_path benchmark/soar/demo_sala/perf_public_set.jsonl \
  --concurrency 32
```

Speed proxy after successful launch:

```bash
python3 benchmark/soar/run_soar_suite.py \
  --base-url http://127.0.0.1:30000 \
  --dataset-profile heavy
```

## 7) Result Summary Table
| Area | Baseline | New | Delta | Notes |
|---|---:|---:|---:|---|
| FlashInfer startup with FP8 KV | reached first request | unchanged | n/a | prior iteration fixed this |
| Sparse top-k scorer dtype | may receive FP8 tensors | forced BF16 under FP8 KV mode | n/a | local compatibility bridge |
| Real KV-cache storage | FP8 | FP8 | none | unchanged |
| First-request sparse scorer crash | present before patch | pending runtime validation | pending | target of this change |

## 8) Rollback Instructions
If this iteration is not beneficial or causes regressions:

1. Revert the BF16 bridge block in `python/sglang/srt/layers/attention/minicpm_backend.py`.
2. Re-run the FP8 launch to confirm the old sparse scorer failure returns.

## 9) Next-Step Suggestions
1. Re-test the same first request that previously failed in sparse top-k scoring.
2. If the scorer still fails, inspect the next failing kernel to see whether another compression subpath still receives FP8.
3. If it passes, move directly to correctness and decode-heavy speed validation to determine whether FP8 KV cache remains worthwhile end to end.