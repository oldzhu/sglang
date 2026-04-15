# CHANGE_0090: EAGLE3 Speculative Decoding for MiniCPM-SALA

## Background and Motivation

MiniCPM-SALA uses 24 SimpleGLA (lightning attention) layers as its recurrent backbone.
The current bottleneck is decode speed — each token requires a full forward pass through
all 32 layers (8 MLA + 24 SimpleGLA). EAGLE3 speculative decoding can generate multiple
candidate tokens with a lightweight draft model and verify them in parallel, potentially
achieving 2-4x speedup on S1 (single concurrency) and significant gains on S8/Smax.

The gap to top 5 is ~40% (team-beta score 52.94 vs #5 score ≥79.41). Config tweaks alone
cannot close this gap. EAGLE3 is the only viable strategic path to top 5.

## Rule-Compliance Statement

- EAGLE3 is explicitly allowed per the toolkit guidance ("speculative heads allowed, count
  toward 2GB")
- Draft model weights (~300M params at FP16 = ~600MB) are well within the 2GB limit
- On-site training is feasible within the 5-hour window
- No accuracy regression expected (verify step ensures exact match with target model)

## Implementation Plan

### Phase 1: Infrastructure (COMPLETED - commit 4059c2420)

#### Target Model Changes (minicpm3.py)
- Added `layers_to_capture` list to `MiniCPM3Model` for intermediate hidden state capture
- Modified forward loop to collect `aux_hidden_states` at configurable layer indices
- Added `capture_aux_hidden_states` flag to `MiniCPM3ForCausalLM`
- Added `get_embed_and_head()`, `set_embed_and_head()`, `set_embed()` for weight sharing
- Added `set_eagle3_layers_to_capture(layer_ids)` — defaults to [2, num_layers//2, num_layers-3]

#### Draft Model (minicpm_eagle3.py — NEW FILE)
- `MiniCPMEagle3Attention`: Standard QKV attention (not MLA), takes 2×hidden_size input
  (concatenation of embeddings + hidden states), uses RadixAttention with RoPE
- `MiniCPMEagle3MLP`: Standard SiLU+Mul MLP with gate/up/down projections
- `MiniCPMEagle3DecoderLayer`: input_layernorm on embeds, hidden_norm on hidden,
  concatenate, then attention + MLP
- `MiniCPMEagle3Model`: embed_tokens, FC(3×hidden_size → hidden_size), 1 decoder layer, norm
- `MiniCPMForCausalLMEagle3`: Full draft model with load_weights, get_hot_token_id,
  shared embed/lm_head with target model

#### GDR Kernel SKIP_DELTA_RULE (fused_recurrent.py)
- Added `SKIP_DELTA_RULE: tl.constexpr` to `fused_recurrent_gated_delta_rule_update_fwd_kernel`
- When True: skips delta rule subtraction (`v -= k^T @ h`) and beta scaling
- Effectively reduces GDR recurrence to SimpleGLA: `h = exp(g)·h + k⊗v`
- Propagated through wrapper function, autograd Function, and public API

#### SimpleGLA Verify Path (hybrid_linear_attn_backend.py)
- `SimpleGLAAttnBackend.forward()`: Added `is_target_verify` branch that:
  - Expands per-head slope `g_gamma` to (1, seq_len, num_heads) for kernel
  - Calls `fused_recurrent_gated_delta_rule_update` with `skip_delta_rule=True`,
    `disable_state_update=True`, and `intermediate_states_buffer`
  - Reuses the complete tree-verify infrastructure (parent-token state loading,
    intermediate state caching, per-step state saving)
- `update_mamba_state_after_mtp_verify()`: Guarded conv_states access with `has_conv`
  check for models without conv layers (SimpleGLA-only)
- Added `_compute_retrieve_parent_token()`: Standalone helper to compute parent tokens
  from next_token/next_sibling (needed since SimpleGLA has no causal_conv1d side-effect)

### Phase 2: Draft Model Training (PENDING)

Requirements:
- Training script to collect hidden states from target model at 3 capture layers
- Distillation training: FC(3×3584 → 3584) → 1 decoder layer → lm_head
- ~300M parameter draft model (~600MB at FP16)
- Training data: perf_public_set.jsonl (90 samples) or collected additional data

### Phase 3: End-to-End Testing (PENDING)

Server launch with EAGLE3:
```bash
--speculative-algorithm EAGLE3
--speculative-draft-model-path /path/to/draft_model
--speculative-num-steps 5
--speculative-num-draft-tokens 64
--speculative-eagle-topk 1  # start with greedy, then try topk=4
```

Draft model config.json requirements:
```json
{
  "architectures": ["MiniCPMForCausalLMEagle3"],
  "num_hidden_layers": 1,
  "eagle_config": {
    "use_aux_hidden_state": true,
    "eagle_aux_hidden_state_layer_ids": [2, 20, 37]
  }
}
```

## Validation Commands

```bash
# Accuracy (must maintain > 97% normalized, ideally > 99%)
python3 /root/data/eval_model_001.py --data_path /root/data/perf_public_set.jsonl

# Speed benchmarks
# S1 speed
python3 /root/data/eval_model.py --data_path /root/data/speed_s1.jsonl --max-concurrent 1
# S8 speed
python3 /root/data/eval_model.py --data_path /root/data/speed_s8.jsonl --max-concurrent 8
# Smax speed
python3 /root/data/eval_model.py --data_path /root/data/speed_smax.jsonl
```

## Expected Results

| Metric | Baseline (no spec decode) | Expected (EAGLE3) |
|--------|--------------------------|-------------------|
| S1     | ~113s                    | ~45-60s (2-2.5x) |
| S8     | ~42s                     | ~25-35s (1.5-2x)  |
| Smax   | ~36s                     | ~25-30s (1.2-1.4x)|
| Accuracy | 79.29% (C=1.0)         | 79.29% (C=1.0)    |

Note: S1 benefits most from spec decode (decode-dominated). S8/Smax have more
prefill overlap, so gains are smaller.

## Rollback Instructions

The EAGLE3 changes are on a separate branch `eagle3-spec-decode`. To rollback:
```bash
git checkout mixed_minicpm_cudagraph  # back to main branch
```
No changes to the main branch are needed.

## Next Steps

1. Create draft model training script
2. Train draft model on fcloud A800
3. Test end-to-end with EAGLE3
4. Tune num_steps, num_draft_tokens, topk for optimal speed/accuracy
5. Add CUDA graph support for verify mode
