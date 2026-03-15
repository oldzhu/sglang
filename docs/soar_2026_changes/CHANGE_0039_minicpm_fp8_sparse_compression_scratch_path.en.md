# CHANGE_0039_minicpm_fp8_sparse_compression_scratch_path

## 1) Background & Motivation
- Problem statement: after enabling MiniCPM FP8 KV cache and fixing the FlashInfer / sparse-scoring dtype issues, correctness evaluation still crashed with an asynchronous `CUDA illegal memory access` during decode preparation.
- Diagnosis: the strongest remaining suspect was MiniCPM's compressed-K maintenance kernel, which reads and writes through the shared KV-cache pointer. Under FP8 KV mode, that shared pointer refers to FP8-backed global GPU storage, but the compression kernel was written around BF16-style behavior.
- Objective of this iteration: stop mutating the shared FP8 KV-cache path for sparse compressed-K maintenance, and instead recompute compressed K1/K2 directly into BF16 scratch buffers when FP8 KV cache is enabled.

## 2) SOAR Rule-Compliance Check
- Latest official references re-checked before this change: `https://soar.openbmb.cn/competition` and `https://soar.openbmb.cn/toolkit`.
- Allowed by rules because: this is a runtime/kernel stability optimization inside the allowed inference-path and KV-cache optimization scope.
- Constraints respected: no model weight change, no prefix-cache re-enable, no concurrency change, no submission-interface change.
- Accuracy/stability risk: medium-low. The scratch path changes how compressed sparse routing features are materialized, so correctness must still be revalidated.

## 3) Design Decision Record
Two candidate designs were considered:

1. **BF16 scratch compressed-K path**
   - Keep real KV cache in FP8.
   - Recompute compressed K1/K2 into BF16 scratch buffers.
   - Feed sparse top-k routing from those BF16 scratch buffers.
   - Advantage: smallest implementation risk and easiest stabilization.

2. **True FP8-aware compressed-K path**
   - Keep compressed K1/K2 in FP8 whenever KV cache is FP8.
   - Requires scale-aware quantize/store/dequantize semantics end to end.
   - Requires every compressed-K reader and writer to be FP8-aware.
   - Deferred for future design/implementation because it is broader and riskier.

Decision for this iteration:

1. Implement the BF16 scratch compressed-K path first.
2. Preserve the true FP8-aware compressed-K design as a future option.

## 4) What Compressed-K Is Used For
MiniCPM sparse attention does not attend densely over the entire historical context for every token. Instead, it first builds compressed K1/K2 summaries over the context and uses them for coarse routing:

1. compressed K1/K2 are pooled representations of historical keys at two granularities
2. sparse top-k scoring uses them to choose which blocks are likely relevant
3. only the selected sparse blocks are then used for the downstream sparse attention path

Therefore compressed K is a routing / block-selection aid, not the final KV-cache format used by the main attention kernel.

## 5) Why Dtype Matters For Accuracy
Changing compressed K from BF16 to FP8 can affect accuracy even if the main KV cache already uses FP8:

1. compressed K participates in block selection, not just final value lookup
2. routing is sensitive to score ordering, especially around top-k boundaries
3. extra quantization noise in compressed K can select slightly different sparse blocks
4. different sparse blocks can change the final attention context and downstream answer quality

This is why a future true-FP8 compressed-K design must be evaluated carefully for correctness, not only speed.

## 6) Detailed Implementation Plan Before Change
Files/functions to change:

1. `python/sglang/srt/layers/attention/minicpm_sparse_kernels.py`
   - add scratch-only compression kernels that write only to `full_compressed_k`
2. `python/sglang/srt/layers/attention/minicpm_sparse_utils.py`
   - add a `scratch_only` switch to compression helpers
3. `python/sglang/srt/layers/attention/minicpm_backend.py`
   - enable scratch-only compression automatically under FP8 KV mode

Expected gain:

1. remove the most suspicious FP8 shared-KV mutation path
2. improve correctness stability under FP8 KV cache
3. preserve the current strong serving performance as much as possible

## 7) Actual Code Changes
Changed files:

1. `python/sglang/srt/layers/attention/minicpm_sparse_kernels.py`
2. `python/sglang/srt/layers/attention/minicpm_sparse_utils.py`
3. `python/sglang/srt/layers/attention/minicpm_backend.py`

Actual edits:

1. Added scratch-only Triton kernels that recompute compressed K directly from the base token table into `full_compressed_k`.
2. Added `scratch_only` support to `get_compress_k_v2`, `get_compress_k_v2_padded`, and `allocate_and_compress_keys`.
3. Enabled the scratch-only path automatically when MiniCPM runs with FP8 KV cache.

## 8) Validation Commands
Syntax / import validation:

```bash
python3 -m compileall \
  python/sglang/srt/layers/attention/minicpm_sparse_kernels.py \
  python/sglang/srt/layers/attention/minicpm_sparse_utils.py \
  python/sglang/srt/layers/attention/minicpm_backend.py
```

Recommended runtime checks:

```bash
CUDA_LAUNCH_BLOCKING=1 python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_PATH> \
  --data_path benchmark/soar/demo_sala/perf_public_set.jsonl \
  --concurrency 32
```

```bash
python3 benchmark/soar/run_soar_suite.py \
  --base-url http://127.0.0.1:30000 \
  --dataset-profile heavy
```

## 9) Result Summary Table
| Area | Baseline | New | Delta | Notes |
|---|---:|---:|---:|---|
| Shared KV mutation by sparse compression | present in FP8 mode | removed for FP8 mode | n/a | main design change |
| Compressed-K materialization | shared-KV backed | BF16 scratch-backed in FP8 mode | n/a | stability-oriented |
| Correctness eval crash | present before patch | pending validation | pending | target issue |
| Future FP8 compressed-K design | conceptual only | documented decision record | n/a | deferred |

## 10) Rollback Instructions
1. Revert the scratch-only compression kernels and helper switches.
2. Revert the MiniCPM backend flag that enables scratch-only compression in FP8 mode.
3. Re-run the FP8 path to confirm the previous behavior returns.

## 11) Next-Step Suggestions
1. Re-run correctness evaluation with `CUDA_LAUNCH_BLOCKING=1` to confirm the illegal access is gone.
2. Compare serving benchmarks before and after this scratch-path change to quantify the overhead.
3. If stable and fast enough, keep the BF16 scratch path.
4. If speed regresses too much, revisit the deferred true FP8-aware compressed-K design.