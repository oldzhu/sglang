# CHANGE_0033_minicpm_cudagraph_sparse_length_sync

## Status Update
- Current status: rolled back in the runtime branch after local bench review.
- Rollback reason: the shared-dataset smoke comparison did not show a meaningful speed win, so the code path was restored to baseline behavior and this note is kept for possible re-evaluation later.

## 1) Background & Motivation
- Problem statement: MiniCPM CUDA-graph decode was capturing FlashInfer sparse-wrapper storage once, but replay still planned with capture-shaped top-k lengths instead of the current batch's real sparse cumulative lengths.
- Why this should improve speed: when padded or shorter sparse rows are planned as full top-k rows, FlashInfer sees more work than the live batch actually contains, which can erode the decode fast path most noticeably at S8 and Smax.
- Target stage(s): decode / kernel launch / scheduling

## 2) SOAR Rule-Compliance Check
- Allowed by rules because: this is a runtime metadata synchronization fix inside the existing CUDA-graph decode path; it does not change model outputs intentionally, evaluation concurrency, or submission interfaces.
- Not violating constraints (prefix cache/concurrency/reproducibility): no prefix cache is introduced, official concurrency settings remain untouched, and the replay metadata is derived deterministically from the current batch.
- Expected impact on correctness coefficient C: neutral in intent; the change only aligns FlashInfer planning metadata with already computed sparse lengths and marks empty padded rows as empty.

## 3) Plan Before Code Change
- Files/functions to modify: `python/sglang/srt/layers/attention/minicpm_backend.py` in CUDA-graph replay metadata setup, and `python/sglang/srt/layers/attention/minicpm_sparse_kernels.py` in sparse-page-table to FlashInfer conversion.
- Minimal diff strategy: update only the replay-time FlashInfer planning tensors and the empty-row serialization behavior; keep model logic, scheduling policy, and public interfaces unchanged.
- Rollback plan: revert this patch to restore the previous placeholder `kv_indptr` replay behavior and unconditional `kv_last_page_len=1` conversion.

## 4) Actual Code Change (After Approval)
- Commit/patch summary: replay now copies `forward_batch.sparse_cu_seqlens_k_cpu` into the FlashInfer planning `kv_indptr` buffer before `begin_forward()`, pads the tail with the final live offset instead of capture capacity, and keeps padded rows at `kv_last_page_len=0`.
- Final modified files: `python/sglang/srt/layers/attention/minicpm_backend.py`, `python/sglang/srt/layers/attention/minicpm_sparse_kernels.py`.
- Key logic differences:
  - FlashInfer CUDA-graph planning no longer assumes every sparse row uses `num_sparse_topk_tokens`.
  - Empty padded sparse rows are serialized as empty rows during sparse-to-FlashInfer conversion.
  - The inline CUDA-graph comments now match the actual MiniCPM graph flow.

## 5) Validation Commands
### Correctness
```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_PATH> \
  --data_path benchmark/soar/demo_sala/perf_public_set.jsonl \
  --concurrency 32
```

### Speed
```bash
export SPEED_DATA_S1=<shared_representative_speed_set.jsonl>
export SPEED_DATA_S8=<shared_representative_speed_set.jsonl>
export SPEED_DATA_SMAX=<shared_representative_speed_set.jsonl>
python3 benchmark/soar/run_soar_suite.py \
  --base-url http://127.0.0.1:30000 \
  --dataset-profile heavy
```

## 6) Results Summary
| Metric | Baseline | New | Delta |
|---|---:|---:|---:|
| Accuracy / overall_accuracy | not re-run | not re-run | n/a |
| Shared-data S1 throughput | 143.04 | 145.29 | +1.6% smoke-only |
| Shared-data S8 throughput | 31.44 | 31.50 | +0.2% smoke-only |
| Shared-data S∞ throughput | 25.90 | 25.97 | +0.3% smoke-only |

- Decision: these deltas were too small to justify keeping the runtime change, so the code was rolled back and the document remains as an experiment record.

## 7) Risk Assessment
- Accuracy risk: low; the change only synchronizes replay metadata with already computed sparse lengths.
- Stability risk: medium-low; it touches the CUDA-graph replay path, so validation should emphasize padded-batch decode cases and mixed short/long sequences.
- Reproducibility risk: low; no randomization or environment-sensitive branching was added.

## 8) Rollback Instructions
1. Revert the edits in `python/sglang/srt/layers/attention/minicpm_backend.py` that copy live sparse cumulative lengths into FlashInfer replay buffers.
2. Revert the edits in `python/sglang/srt/layers/attention/minicpm_sparse_kernels.py` that serialize empty rows with `kv_last_page_len=0`.
3. Keep this document as a historical note so the sparse-length sync idea can be revisited if future profiling shows wrapper planning mismatch again.

## 9) Next-Step Suggestions
- Run the shared-dataset S1/S8/Smax suite and compare graph hit rate and benchmark duration against the current branch baseline.
- If S8/Smax still diverge materially from expectation, inspect whether MiniCPM decode falls back because of graph batch-size selection or other replay-shape guards.