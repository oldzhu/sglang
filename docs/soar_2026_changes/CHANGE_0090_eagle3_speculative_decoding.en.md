# CHANGE_0090: EAGLE3 Speculative Decoding for MiniCPM-SALA

> **Status**: PAUSED — untrained draft model results not viable; needs trained draft to continue  
> **Branch**: `eagle3-spec-decode` (latest commit: `548c8c153`)  
> **Date**: 2026-04-17  

---

## Background and Motivation

EAGLE3 is a speculative decoding algorithm that uses a lightweight draft model to predict multiple tokens, then verifies them against the target model in a single forward pass. For large models (13B+), this typically yields 2-5x speedup at zero accuracy loss.

We implemented EAGLE3 for MiniCPM-SALA to explore whether speculative decoding could improve inference speed for the SOAR 2026 competition.

### MiniCPM-SALA Architecture Challenges
- **Hybrid architecture**: 8 standard attention layers + 24 SimpleGLA (recurrent) layers
- **Recurrent state**: SimpleGLA maintains internal state that must be rolled back after rejected draft tokens — unlike standard transformer KV cache
- **Small model (~4B params)**: The per-token latency of the target model is already low, meaning draft overhead may not be offset by fewer target forward passes

---

## Rule-Compliance Statement

- Speculative heads are **explicitly allowed** per SOAR rules ("Speculative heads allowed, count toward 2GB")
- Draft model (612MB) fits within the 2GB total submission limit alongside GPTQ model
- No accuracy loss from speculative decoding (verification guarantees target model distribution)

---

## Implementation Summary

### Draft Model Architecture (`minicpm_eagle3.py`)
- FC fusion layer: 3×4096 → 4096 (fuses embeddings + aux hidden states from low/mid/high layers)
- 1 decoder layer with standard GQA attention (32 Q heads, 8 KV heads)
- Shared embed_tokens and lm_head weights from target model
- Total: 293M params, 612MB

### Target Model Changes (`minicpm.py`)
- Added `layers_to_capture` list and aux hidden states capture in forward()
- Added `get_embed_and_head()`, `set_embed_and_head()`, `set_eagle3_layers_to_capture()`

### SimpleGLA State Management
- Reused GDR (Gated Delta Rule) update kernel with `SKIP_DELTA_RULE=True` flag
- State shape (16, 128, 64) matches existing MambaPool
- Sequential verify + state rollback via `update_mamba_state_after_mtp_verify()`

### Bug Fixes (7 total, all committed)
1. Draft model config: changed model_type to "minicpm_sala" with auto_map
2. `set_eagle3_layers_to_capture`: added to MiniCPMForCausalLM
3. `get_embed_and_head`: return `.weight` tensor not Module
4. RadixAttention scaling: `head_dim**-0.5` float not rotary_emb function
5. Mamba state rollback: use `mambaish_config` (covers minicpm_hybrid_config)
6. `update_mamba_state_after_mtp_verify`: guard conv_states with `has_conv` check
7. `_compute_retrieve_parent_token`: standalone helper for topk>1

---

## Test Results (Test 22)

| Metric | EAGLE3 (untrained) | Baseline (Test 20) | Change |
|--------|--------------------|--------------------|--------|
| Config | GPTQ+FP8+dense+EAGLE3 | GPTQ+FP8+dense | — |
| mem-fraction-static | 0.72 | 0.84 | -14% less KV cache |
| S1 | 187.01s | 113.67s | **+65% slower** |
| ori_accuracy | 74.33% | 80.64% | -6.3pp |
| normalized_accuracy | 92.92% | 100.80% | -7.9pp |
| C | **0 (eliminated)** | 1.0 | — |
| Accept rate | 0.26 | — | Random draft |
| Accept length | 1.03-1.07 | — | ~0 bonus tokens |

### Per-Task Breakdown
| Task | EAGLE3 | Baseline |
|------|--------|----------|
| MCQ | 56.67% (avg_out=8527) | ~76.67% (avg_out=~1442) |
| NIAH | 100% | 100% |
| CWE | 78.33% | ~80% |
| FWE | 73.33% | ~80% |
| QA | 63.33% | ~70% |

---

## Analysis

### Why It Failed
1. **Untrained draft model** → accept rate 0.26 (random prediction), adding pure overhead
2. **EAGLE3 scheduling penalties**: Disables mixed-chunk, disables overlap scheduler, caps max_running_requests=24
3. **Memory pressure**: Draft model (612MB) + aux hidden states (~1GB) consume ~10GB → mem-fraction-static reduced from 0.84 to 0.72 → less KV cache
4. **MCQ overgeneration**: 8527 avg output tokens vs baseline ~1442 — model fails to stop properly (unclear if scheduling or draft model related)

### What's Needed to Make It Work
1. **Train the draft model** using SpecForge or EAGLE training scripts
   - Requires running BF16 target model on text data to extract hidden states
   - Then train FC fusion + decoder layer to predict next-token hidden states
   - Training data: ShareGPT + domain data (QA/CWE/MCQ/NIAH tasks)
   - GPU time: estimated 2-4 hours on 1× RTX 6000D
2. **Achieve accept rate > 0.5** to offset draft model overhead
3. **Investigate MCQ overgeneration** — may be a separate bug

### Cost-Benefit Assessment
Even with a well-trained draft model:
- MiniCPM-SALA is a small model (~4B) — per-token latency is already low
- Draft model adds 293M params forward per step
- EAGLE3 disables mixed-chunk and overlap scheduler
- Net speedup likely modest (10-20%) vs the effort required
- **Recommendation: Pause and pursue other optimization vectors first**

---

## Rollback Instructions

To disable EAGLE3, set `SOAR_ENABLE_EAGLE3=0` (or remove) in `prepare_env.sh` and restore `mem-fraction-static` to 0.84.

---

## Next Steps (When Resuming)

1. Set up SpecForge training environment on fcloud
2. Generate training data from BF16 model using ShareGPT + domain text
3. Train draft model to achieve >0.5 accept rate on eval data
4. Retest with trained draft model
5. If accept rate > 0.6 and speed improves, proceed; otherwise abandon
