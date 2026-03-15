# CHANGE_0036_minicpm_fp8_kv_cache_benchmark_matrix

## 1) Background & Motivation
- Problem statement: there was confusion about whether `--kv-cache-dtype` affects only MiniCPM prefill or also decode, and how to measure its real value once FP8-compatible paths are available.
- Why this matters: FP8 KV cache is primarily a memory-bandwidth and KV-capacity optimization. If evaluated with the wrong workload mix, it is easy to misread the result or blame the wrong stage.
- Target stage(s): KV-cache write path, KV-cache read path, prefill/decode decomposition, benchmark methodology

## 2) SOAR Rule-Compliance Check
- Latest official references reviewed before this document: `https://soar.openbmb.cn/competition` and `https://soar.openbmb.cn/toolkit`.
- Allowed by rules because: the competition page explicitly allows inference-path optimization, memory/KV optimization, and quantization-based acceleration, while the toolkit `技术路径指引` explicitly lists FP8 KV cache as a valid optimization direction.
- Not violating constraints: this document does not change model weights, concurrency settings, prefix cache behavior, or submission interfaces; it only clarifies runtime behavior and local benchmarking procedure.
- Expected impact on correctness coefficient C: none by itself; however, FP8 KV cache experiments must still be validated with the public correctness set because cache quantization can affect model accuracy.

## 3) Technical Explanation
- `--kv-cache-dtype` is not a prefill-only setting. It changes the storage format of the KV cache itself, so it affects both:
  - prefill/extend: writing newly produced K/V tensors into cache
  - decode: reading previously stored K/V tensors back during attention
- In MiniCPM FlashInfer today, prefill and decode use different wrappers:
  - decode uses `BatchDecodeWithPagedKVCacheWrapper`
  - prefill uses `BatchPrefillWithPagedKVCacheWrapper`
- Both wrappers receive `kv_data_type=self.data_type`, where `self.data_type` is derived from `model_runner.kv_cache_dtype`.
- Therefore, KV-cache dtype is conceptually a whole-cache format decision, not a phase-local knob.

## 4) Why the Current Failure Looks Like a Prefill Issue
- The current MiniCPM FlashInfer implementation creates the prefill wrapper with `backend="fa2"`.
- FlashInfer rejects the FP8 tensor-core path for this backend combination, so the runtime fails during prefill planning or the first prefill execution.
- This does **not** mean decode is unaffected by KV-cache dtype. It means the current incompatible wrapper is encountered before decode benefits can even be measured.

## 5) Expected Impact by Resource and Runtime Stage
### Memory capacity
- Usually the largest immediate benefit.
- FP8 KV cache can reduce KV-storage bytes to roughly half of BF16/FP16 cache bytes.
- Total process memory reduction will be smaller than 2x because weights, activations, workspace, and fragmentation do not scale the same way.

### Prefill / extend
- Benefit is often smaller and may even be flat.
- Reason: prefill is frequently dominated by prompt-side GEMMs and prompt attention computation, while KV dtype mainly changes cache write cost and storage footprint.

### Decode
- This is usually where FP8 KV cache helps most.
- Reason: decode repeatedly rereads the stored KV cache, so reducing bytes per cached element lowers memory-bandwidth pressure.
- Long-context and higher-concurrency cases are the most likely to show a visible gain.

### Accuracy and stability
- FP8 KV cache can introduce accuracy risk because cached activations are more aggressively quantized than BF16/FP16.
- Any speed or memory win must be validated against `perf_public_set.jsonl` before treating it as contest-safe.

## 6) Concrete Benchmark Matrix
### Matrix A: startup and compatibility
Goal: determine whether a given `attention-backend + kv-cache-dtype` combination is actually runnable.

Test combinations:
1. `attention-backend=flashinfer`, `kv-cache-dtype=bfloat16`
2. `attention-backend=flashinfer`, `kv-cache-dtype=fp8_e5m2`
3. `attention-backend=flashinfer`, `kv-cache-dtype=fp8_e4m3`
4. If available for MiniCPM, a non-FlashInfer attention backend with the same KV dtype settings

Record:
1. Whether server launch succeeds
2. Whether failure occurs at startup, first prefill request, or first decode request
3. Exact assertion/error text

Interpretation:
1. If only the FP8 + FlashInfer path fails, this is a compatibility problem, not a performance result.
2. If another backend launches with FP8, the blocker is likely MiniCPM FlashInfer integration rather than FP8 KV cache in general.

### Matrix B: memory impact
Goal: measure how much KV-cache dtype changes actual memory usage.

Workload:
1. One fixed long-context request or a small batch of long-context requests
2. Keep model, prompt, output length, and concurrency fixed

Compare:
1. `bfloat16`
2. `fp8_e4m3`
3. `fp8_e5m2` if launchable

Record:
1. Peak GPU memory
2. Steady-state GPU memory after prefill
3. Any observable KV allocator size or cache residency stats

Expected outcome:
1. KV-storage footprint should shrink materially under FP8
2. Total process memory will shrink less than the KV-only ratio

### Matrix C: prefill-heavy sensitivity
Goal: isolate whether FP8 KV cache helps prompt ingestion.

Workload:
1. Long inputs
2. Very short outputs, such as 1 to 16 generated tokens

Record:
1. TTFT
2. End-to-end latency
3. Peak GPU memory

Expected outcome:
1. Small gain, no gain, or slight regression are all plausible
2. Prefill is often not KV-bandwidth-bound enough for large benefit

### Matrix D: decode-heavy sensitivity
Goal: isolate the phase where FP8 KV cache is most likely to pay off.

Workload:
1. Moderate/long inputs so the cache is nontrivial
2. Long outputs so decode repeatedly rereads cached K/V

Record:
1. TPOT or output tokens/sec
2. Latency tail
3. Peak GPU memory
4. Optional profiler evidence for memory-bandwidth pressure

Expected outcome:
1. This is the most likely place to see measurable FP8 KV-cache speedup
2. Larger gains are more likely under longer context and higher concurrency

### Matrix E: end-to-end SOAR proxy
Goal: determine whether the memory and decode-phase effects survive into the actual score proxy.

Workload:
1. Shared-dataset S1
2. Shared-dataset S8
3. Shared-dataset Smax

Record:
1. `benchmark_duration`
2. `acc_ori` / `overall_accuracy`
3. Peak GPU memory
4. Any instability under concurrency

Interpretation:
1. If decode bandwidth is the real bottleneck, gains should usually show up more clearly in S8 and Smax than in S1.
2. If memory drops but Smax remains flat, other overheads such as scheduling, wrapper planning, or weight-side compute may dominate.

## 7) Suggested Commands
### Compatibility
```bash
python3 -m sglang.launch_server \
  --model-path <MODEL_PATH> \
  --attention-backend flashinfer \
  --kv-cache-dtype bfloat16
```

```bash
python3 -m sglang.launch_server \
  --model-path <MODEL_PATH> \
  --attention-backend flashinfer \
  --kv-cache-dtype fp8_e5m2
```

```bash
python3 -m sglang.launch_server \
  --model-path <MODEL_PATH> \
  --attention-backend flashinfer \
  --kv-cache-dtype fp8_e4m3
```

### Correctness
```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_PATH> \
  --data_path benchmark/soar/demo_sala/perf_public_set.jsonl \
  --concurrency 32
```

### End-to-end speed proxy
```bash
export SPEED_DATA_S1=<shared_representative_speed_set.jsonl>
export SPEED_DATA_S8=<shared_representative_speed_set.jsonl>
export SPEED_DATA_SMAX=<shared_representative_speed_set.jsonl>
python3 benchmark/soar/run_soar_suite.py \
  --base-url http://127.0.0.1:30000 \
  --dataset-profile heavy
```

### Memory observation
```bash
nvidia-smi dmon -s mu -d 1
```

## 8) Results Summary Template
| Area | BF16 baseline | FP8 candidate | Delta | Notes |
|---|---:|---:|---:|---|
| Launch success | pending | pending | n/a | startup / first prefill / first decode |
| Peak GPU memory | pending | pending | pending | |
| Prefill-heavy TTFT | pending | pending | pending | |
| Decode-heavy TPOT | pending | pending | pending | |
| S1 benchmark_duration | pending | pending | pending | |
| S8 benchmark_duration | pending | pending | pending | |
| S∞ benchmark_duration | pending | pending | pending | |
| Accuracy / overall_accuracy | pending | pending | pending | |

## 9) Rollback / Decision Guidance
1. If FP8 launch fails before decode starts, treat the issue as backend compatibility, not benchmark underperformance.
2. If memory improves but speed does not, keep FP8 only if the extra capacity is strategically useful.
3. If decode improves but correctness drops, do not keep the path for SOAR submission until the accuracy issue is understood.
4. If S8/Smax improve consistently while correctness stays safe, FP8 KV cache becomes a meaningful candidate for further tuning.

## 10) Next-Step Suggestions
- Use this matrix before changing MiniCPM FP8 logic so compatibility, memory, and speed effects do not get conflated.
- If the current FlashInfer path still fails at prefill, the next feature should add an explicit MiniCPM compatibility guardrail or supported fallback instead of letting the runtime fail deep inside FlashInfer.