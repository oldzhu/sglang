# CHANGE_0100: EAGLE3 Draft Model Training Script

## Background and Motivation

EAGLE3 speculative decoding is the highest-impact optimization path for the SOAR 2026 competition. Top teams (#1 FlashSALA at 94.29, #2 智算一队 at 90.22) likely use speculative decoding. Our Phase 1 (inference infrastructure) is complete on the `eagle3-spec-decode` branch — this change implements Phase 2: training the draft head.

The draft model is a lightweight single-layer transformer that predicts the target model's output distribution from auxiliary hidden states captured at intermediate layers, enabling 2-4x decode speedup with mathematically exact outputs (rejection sampling).

## Rule Compliance

- **Speculative heads explicitly allowed** by competition rules (count toward 2GB)
- Draft model is ~586MB FP16 (~293M params) — well within 2GB total
- No forbidden tricks — standard knowledge distillation training
- Apache 2.0 compatible
- Speculative decoding preserves output distribution exactly (C = 1.0)

## Implementation Plan

### Training Script: `benchmark/soar/demo_sala/train_eagle3_draft.py`

**Architecture** (matches sglang inference `MiniCPMForCausalLMEagle3`):
- `FC(3×4096 → 4096)`: combines 3 auxiliary hidden states
- `DraftDecoderLayer`: standard QKV attention (2×H input) + SiLU-gated MLP
- `RMSNorm`: final normalization
- Shared `embed_tokens` and `lm_head` from target model (frozen)

**Training Flow**:
1. Load target model (MiniCPM-SALA BF16) with HuggingFace transformers
2. Register forward pre-hooks on layers [2, 16, 29] to capture hidden states
3. For each batch:
   - Forward target model → get logits + 3 captured hidden states
   - Concatenate hidden states → [bsz, seq, 3×4096]
   - Forward draft model → get draft logits
   - KL divergence loss between target and draft distributions
4. Save draft model in safetensors format

**Key Parameters**:
- Capture layers: [2, 16, 29] (default for 32-layer model)
- Trainable params: ~293M (FC: 50M, QKV: 50M, O: 17M, MLP: 176M)
- Training: AdamW, lr=1e-4, cosine schedule, 1000 steps
- Data: perf_public_set.jsonl (150 samples)

### Output Config (`config.json`):
```json
{
  "architectures": ["MiniCPMForCausalLMEagle3"],
  "num_hidden_layers": 1,
  "hidden_size": 4096,
  "intermediate_size": 14336,
  "eagle_config": {"use_aux_hidden_state": true}
}
```

## Weight Name Mapping

Training names → sglang weight loader:
| Training | Checkpoint Saved | sglang Target |
|----------|-----------------|---------------|
| `fc.weight` | `fc.weight` | `model.fc.weight` |
| `layer.self_attn.q_proj.weight` | `midlayer.self_attn.q_proj.weight` | → `model.midlayer.self_attn.qkv_proj.weight` (shard q) |
| `layer.self_attn.k_proj.weight` | `midlayer.self_attn.k_proj.weight` | → shard k |
| `layer.self_attn.v_proj.weight` | `midlayer.self_attn.v_proj.weight` | → shard v |
| `layer.mlp.gate_proj.weight` | `midlayer.mlp.gate_proj.weight` | → `gate_up_proj` shard 0 |
| `layer.mlp.up_proj.weight` | `midlayer.mlp.up_proj.weight` | → `gate_up_proj` shard 1 |

## Validation Commands

```bash
# Train draft model on fcloud
cd /root/sglang-minicpm/benchmark/soar/demo_sala
python3 train_eagle3_draft.py \
    --model-path /root/models/openbmb/MiniCPM-SALA-Copy \
    --data-path /root/data/perf_public_set.jsonl \
    --output-path /root/models/eagle3_draft_minicpm \
    --num-steps 1000 --lr 1e-4 --batch-size 1

# Test with speculative decoding (add to prepare_env.sh)
--speculative-algorithm EAGLE3 \
--speculative-draft-model-path /root/models/eagle3_draft_minicpm \
--speculative-num-steps 5 \
--speculative-num-draft-tokens 64 \
--speculative-eagle-topk 1

# Run accuracy + speed eval
python3 scripts/fcloud/fcloud_workflow.py accuracy
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

## Result Summary

| Metric | Baseline (no spec) | Expected (EAGLE3) |
|--------|-------------------|-------------------|
| S1 | 113.67s | ~40-60s |
| S8 | 41.07s | ~15-25s |
| Smax | 34.15s | ~15-20s |
| Accuracy | 80.64% | Same (exact) |
| C | 1.0 | 1.0 |

## Rollback

Remove EAGLE3 server args from `prepare_env.sh`. The base model operates normally without the draft model.

## Next Steps

1. **Run training on fcloud** — requires BF16 model + GPU
2. **Tune hyperparameters** — learning rate, steps, temperature
3. **Test end-to-end** — speculative decoding with trained draft
4. **Optimize acceptance rate** — try different capture layers, more training data
5. **Multi-step training** — train with draft model's own hidden states (Phase 3)
