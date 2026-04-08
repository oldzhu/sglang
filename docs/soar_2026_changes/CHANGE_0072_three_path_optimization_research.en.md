# CHANGE_0072: Three-Path Optimization Research & Planning

**Date:** 2026-04-08  
**Type:** Research / Planning Document  
**Status:** Complete — Awaiting path selection decision

---

## 1. Background and Motivation

After systematic testing (Tests 1–7), we established a clear picture of MiniCPM-SALA accuracy across configurations:

| Test | Config | Accuracy | Key Finding |
|------|--------|----------|-------------|
| 1 | Non-quant + FP8 + sparse | 53.18% | FP8 hurts sparse |
| 2 | Non-quant + bf16 + sparse (HEAD w/ CHANGE_0070) | 57.11% | CHANGE_0070 regression |
| 3 | Non-quant + bf16 + sparse (pre-0070) | **83.02%** | Baseline confirmed |
| 4 | Non-quant + bf16 + sparse (HEAD−0070) | **83.02%** | Bug fixes safe |
| 5 | GPTQ + FP8 + sparse | 50.36% | GPTQ+FP8 breaks sparse |
| 6 | GPTQ + bf16 + sparse | 57.47% | GPTQ alone breaks sparse |
| 7 | GPTQ + FP8 + dense | 78.84% | Dense tolerates GPTQ+FP8 |

**2×2 Matrix (GPTQ model):**

|  | Dense | Sparse |
|--|-------|--------|
| **bf16 KV** | ~99.47% (no GPTQ baseline) | 57.47% |
| **FP8 KV** | 78.84% | 50.36% |

**Core problem:** GPTQ quantization breaks sparse attention's topk scoring, while dense attention is robust to GPTQ+FP8. Three parallel optimization paths were researched.

**Scoring formula:** `final_score = (S1×40% + S8×30% + Smax×30%) × C`  
where C = 0 if relative accuracy < 97%, C = 0.8 if 97–100%, C = 1.0 if ≥ 100%.

---

## 2. Path 1: Dense + GPTQ + FP8 Speed Optimization

### Concept
Accept dense attention (78.84% accuracy with GPTQ+FP8) and focus on maximizing throughput via runtime tuning and kernel optimization.

### Key Findings (Subagent Research)

**Call chain:** `ModelRunner → MiniCPMForCausalLM → MiniCPMDecoderLayer × 32 → GPTQ Marlin GEMM + FlashInfer attention`

**Top bottlenecks:**
1. Attention backend choice: 15–25% of runtime
2. Marlin GEMM kernel config: 10–20%
3. KV cache FP8 dequant overhead: 5–10%

**Optimization phases:**

| Phase | Action | Effort | Expected Gain |
|-------|--------|--------|---------------|
| A (Quick wins) | CUDA graph tuning, `chunked_prefill`, `max_running_requests`, `schedule_conservativeness` | 1–2h | 5–15% speed |
| B (Kernel) | KV dequant fusion, Lightning vs dense tradeoff | 4–8h | 10–20% speed |
| C (Accuracy) | Per-layer mixed precision, extended GPTQ calibration | 8–16h | +1–2% accuracy |

### Assessment
- **Effort:** Low–Medium (days)
- **Risk:** Low
- **Accuracy ceiling:** ~78–80%
- **Speed ceiling:** 10–20% faster dense
- **Key dependency:** NCU profiling for bottleneck quantification

---

## 3. Path 2: Non-quant + Sparse + EAGLE3 Speculative Decoding

### Concept
Use non-quantized model (83.02% accuracy) with sparse attention, and add EAGLE3 speculative decoding for 1.3–2.2× throughput speedup.

### Key Findings (Subagent Research)

**Critical blocker:** Lightning Attention's recurrent state computation is **incompatible with tree verification masks** needed for EAGLE3 tree-based speculation.

**Workaround options:**

| Option | Approach | Speedup | Risk | Effort |
|--------|----------|---------|------|--------|
| A | Flat draft (no tree) | 1.3–1.5× | Low | 3–4 weeks |
| B | Branch state tracking | 1.5–2.0× | HIGH | 6–8 weeks |
| C | Skip Lightning in draft | 1.4–1.8× | Low | 4–5 weeks |
| D | Algorithmic innovation | 2.0–2.5× | Very High | 8+ weeks |

**Additional requirements:**
- Training draft head: ~50K–100K examples, 12–24h training
- Draft head size: ~50–100MB
- Key reference files: `llama_eagle3.py`, `hybrid_linear_attn_backend.py`

### Assessment
- **Effort:** Very High (6–9 weeks)
- **Risk:** High (Lightning Attention blocker)
- **Accuracy ceiling:** ~83% (C=1.0 bracket possible)
- **Speed ceiling:** 1.3–2.2× throughput
- **Key dependency:** Training infrastructure, Lightning Attention workaround

---

## 4. Path 3: Fix GPTQ + Sparse Incompatibility

### Concept
Fix the root cause of GPTQ breaking sparse attention—precision-sensitive topk scoring—with low-effort changes to recover accuracy while keeping both GPTQ compression and sparse speed benefits.

### Root Cause Analysis (Subagent Research)

**Error propagation chain:**
1. Q projection: GPTQ int4→bf16 dequant introduces ~1–2% per-element error
2. K projection: Same GPTQ error, then optionally FP8 quantization adds more
3. K compression: `compress_k_to_scratch_kernel` averages K over `kernel_size=32` tokens — helps but doesn't eliminate error
4. Score computation: `infllmv2_attn_stage1` computes `Q @ K^T / sqrt(head_dim)` — errors amplify (sqrt(128) = ~11.3× multiplier)
5. TopK selection: `topk=8` from ~2000 candidates (0.4% selection rate) — **binary winner-takes-all, no tolerance for score errors**

**Why NIAH collapses (100%→37%):** The needle must be in the exact top-8 blocks. With GPTQ error shifting scores by 3–5%, the needle block easily falls out of top-8.

**Why FWE is robust (97–98% always):** Facts appear redundantly across many blocks, so even with wrong top-8, some relevant blocks are included.

### Proposed Solutions

| Solution | Effort | Expected Accuracy (from 57.47%) | Speed Impact |
|----------|--------|---------------------------------|--------------|
| **1. Increase topk (8→16)** | 1–2h | →70–75% | −10–15% |
| **2. FP32 scoring** | 2–3h | →80–85% | −5–10% |
| **3. Disable sparse for GPTQ (fallback)** | 1h | →78.84% (dense) | Dense speed |
| **4. topk=16 + FP32 combined** | 3–4h | →85%+ | −15–20% |

**Implementation details:**

**Solution 1 (topk increase):** Change `self.sparse_topk = topk + (window_size // block_size)` → double the topk value in `minicpm_backend.py`.

**Solution 2 (FP32 scoring):** Cast Q and compressed K to float32 before `infllmv2_attn_stage1()` in `minicpm_sparse_utils.py`. Reduces score computation error from 3–5% to 0.3–0.5%.

**Solution 3 (dense fallback):** Detect GPTQ config and set `--force-dense-minicpm` automatically.

### Assessment
- **Effort:** Very Low (1–4 hours)
- **Risk:** Low
- **Accuracy ceiling:** ~80–85%
- **Speed ceiling:** Sparse speed (faster than dense, minus topk overhead)
- **Key dependency:** None — can test immediately

---

## 5. Cross-Path Comparison

| Criterion | Path 1 (Dense Speed) | Path 2 (EAGLE3) | Path 3 (Fix GPTQ+Sparse) |
|-----------|----------------------|------------------|---------------------------|
| **Effort** | Low–Medium | Very High (6–9wk) | Very Low (1–4h) |
| **Risk** | Low | High | Low |
| **Accuracy ceiling** | ~78–80% | ~83% (C=1.0) | ~80–85% |
| **Speed ceiling** | 10–20% faster dense | 1.3–2.2× | Sparse speed |
| **Timeline** | Days | Weeks–Months | Hours |
| **Dependencies** | NCU profiling | Training infra | None |

---

## 6. Recommended Strategy

**Priority 1 — Path 3 (Immediate, hours):**  
Try topk increase and/or fp32 scoring with GPTQ+sparse. If accuracy recovers to ≥80%, we get both GPTQ compression and sparse speed benefits.

**Priority 2 — Path 1 (Parallel, days):**  
Tune runtime parameters and profile with NCU for speed gains. These apply to any configuration.

**Priority 3 — Path 2 (Only if timeline allows, weeks):**  
EAGLE3 has the highest theoretical payoff but Lightning Attention blocker makes it risky. Defer unless significant time remains.

**Quick win combo:** Path 3 recovers GPTQ+sparse to ~83% accuracy + Path 1 speed tuning → competitive score without EAGLE3.

---

## 7. Decision Record

**Decision (2026-04-08):** Proceed with Path 3 first, then Path 1. Path 2 deferred.

---

## 8. Next Steps

1. Implement Path 3 Solution 1 (increase topk=8→16) — test with GPTQ+FP8+sparse
2. If insufficient, add Solution 2 (fp32 scoring) — test combination
3. Run speed benchmarks to measure throughput impact
4. Begin Path 1 runtime parameter tuning in parallel
