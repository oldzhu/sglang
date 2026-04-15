#!/usr/bin/env python3
"""Train EAGLE3 draft head for MiniCPM-SALA.

Creates a lightweight draft model (FC + 1 decoder layer) that predicts the
target model's output distribution from auxiliary hidden states captured at
intermediate layers, enabling speculative decoding.

Draft architecture (matches sglang MiniCPMForCausalLMEagle3):
  FC(3*H → H) + 1 DecoderLayer(2H-input attention + MLP) + shared embed/lm_head

Usage:
    python3 train_eagle3_draft.py \
        --model-path /root/models/openbmb/MiniCPM-SALA-Copy \
        --data-path /root/data/perf_public_set.jsonl \
        --output-path /root/models/eagle3_draft_minicpm \
        --num-steps 1000 --lr 1e-4 --batch-size 1 --max-seq-len 2048
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Draft model components (plain PyTorch, matches sglang inference arch)
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x.to(self.weight.dtype))


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos + rotate_half(q) * sin), (k * cos + rotate_half(k) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position: int = 8192, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor):
        # positions: [batch, seq]
        freqs = torch.einsum("...i,j->...ij", positions.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


class DraftAttention(nn.Module):
    """Standard multi-head attention with 2*hidden_size input (embeds+hidden concat)."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, max_position: int = 8192, rope_theta: float = 10000.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # Input: concat(normed_embeds, normed_hidden) = 2 * hidden_size
        self.q_proj = nn.Linear(2 * hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(2 * hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(2 * hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.rotary_emb = RotaryEmbedding(head_dim, max_position, rope_theta)

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(positions)
        cos = cos.unsqueeze(1)  # [bsz, 1, seq, dim]
        sin = sin.unsqueeze(1)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # GQA: expand kv heads
        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output)


class DraftMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DraftDecoderLayer(nn.Module):
    """Single decoder layer for EAGLE3 draft model.

    Matches sglang's MiniCPMEagle3DecoderLayer:
      input_layernorm(embeds) || hidden_norm(hidden) → concat → attention → residual
      → post_attention_layernorm → MLP → residual
    """

    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int,
                 num_kv_heads: int, head_dim: int, rms_norm_eps: float,
                 max_position: int = 8192, rope_theta: float = 10000.0):
        super().__init__()
        self.self_attn = DraftAttention(
            hidden_size, num_heads, num_kv_heads, head_dim, max_position, rope_theta
        )
        self.mlp = DraftMLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, rms_norm_eps)
        self.hidden_norm = RMSNorm(hidden_size, rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, rms_norm_eps)

    def forward(self, embeds: torch.Tensor, hidden_states: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        residual = hidden_states

        # Normalize both inputs
        normed_embeds = self.input_layernorm(embeds)
        normed_hidden = self.hidden_norm(hidden_states)

        # Concat and attend
        combined = torch.cat([normed_embeds, normed_hidden], dim=-1)
        attn_out = self.self_attn(combined, positions)

        # Post-attention residual + norm + MLP
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Eagle3DraftModel(nn.Module):
    """Complete EAGLE3 draft model for training.

    Architecture: FC(3*H → H) + 1 DecoderLayer + norm
    Shared embed_tokens and lm_head from target model (frozen).
    """

    def __init__(self, config: dict):
        super().__init__()
        H = config["hidden_size"]

        self.fc = nn.Linear(3 * H, H, bias=False)

        self.layer = DraftDecoderLayer(
            hidden_size=H,
            intermediate_size=config["intermediate_size"],
            num_heads=config["num_attention_heads"],
            num_kv_heads=config["num_key_value_heads"],
            head_dim=H // config["num_attention_heads"],
            rms_norm_eps=config.get("rms_norm_eps", 1e-6),
            max_position=config.get("max_position_embeddings", 8192),
            rope_theta=config.get("rope_theta", 10000.0),
        )

        self.norm = RMSNorm(H, config.get("rms_norm_eps", 1e-6))

    def forward(self, embeds, aux_hidden_states, positions, lm_head):
        """
        Args:
            embeds: [bsz, seq, H] - token embeddings from target model
            aux_hidden_states: [bsz, seq, 3*H] - concat of 3 captured hidden states
            positions: [bsz, seq] - position ids
            lm_head: nn.Linear - shared lm_head from target model (frozen)
        Returns:
            logits: [bsz, seq, vocab_size]
        """
        # FC: combine 3 auxiliary hidden states → single hidden state
        hidden = self.fc(aux_hidden_states)

        # Decoder layer with embed+hidden concat attention
        hidden = self.layer(embeds, hidden, positions)

        # Final norm
        hidden = self.norm(hidden)

        # Shared lm_head to get logits (frozen weights)
        logits = F.linear(hidden, lm_head.weight)
        return logits


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


class TextDataset(Dataset):
    """Load training texts from JSONL. Supports various formats."""

    def __init__(self, data_path: str, tokenizer, max_seq_len: int = 2048):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.texts = []

        with open(data_path, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                text = self._extract_text(item)
                if text and len(text) > 10:
                    self.texts.append(text)

    def _extract_text(self, item: dict) -> Optional[str]:
        """Extract text from various JSONL formats."""
        # perf_public_set.jsonl format: {"question": "...", "gold": "..."}
        if "question" in item:
            text = item["question"]
            if "gold" in item:
                text += "\n" + str(item["gold"])
            return text
        # Standard formats
        if "text" in item:
            return item["text"]
        if "prompt" in item:
            text = item["prompt"]
            if "response" in item:
                text += " " + item["response"]
            return text
        if "conversations" in item:
            return " ".join(
                turn.get("content", turn.get("value", ""))
                for turn in item["conversations"]
            )
        # Fallback
        parts = [v for v in item.values() if isinstance(v, str)]
        return " ".join(parts) if parts else None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer(
            self.texts[idx],
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in tokens.items()}


def collate_fn(batch):
    """Left-pad to longest in batch. Use pad_token_id from first item."""
    max_len = max(item["input_ids"].shape[0] for item in batch)
    padded = {"input_ids": [], "attention_mask": []}
    for item in batch:
        pad_len = max_len - item["input_ids"].shape[0]
        padded["input_ids"].append(F.pad(item["input_ids"], (pad_len, 0), value=0))
        padded["attention_mask"].append(F.pad(item["attention_mask"], (pad_len, 0), value=0))
    return {k: torch.stack(v) for k, v in padded.items()}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def find_model_layers(target_model):
    """Find the transformer layers in the target model."""
    if hasattr(target_model, "model") and hasattr(target_model.model, "layers"):
        return target_model.model.layers
    if hasattr(target_model, "transformer") and hasattr(target_model.transformer, "h"):
        return target_model.transformer.h
    raise ValueError("Cannot find model layers. Unsupported model architecture.")


def find_embed_and_head(target_model):
    """Find embedding layer and lm_head."""
    embed = None
    head = None

    if hasattr(target_model, "model") and hasattr(target_model.model, "embed_tokens"):
        embed = target_model.model.embed_tokens
    elif hasattr(target_model, "get_input_embeddings"):
        embed = target_model.get_input_embeddings()

    if hasattr(target_model, "lm_head"):
        head = target_model.lm_head
    elif hasattr(target_model, "get_output_embeddings"):
        head = target_model.get_output_embeddings()

    if embed is None or head is None:
        raise ValueError("Cannot find embedding or lm_head layer.")
    return embed, head


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # ---- 1. Load target model ----
    print(f"[1/5] Loading target model from {args.model_path}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map={"": device},
    )
    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad_(False)

    config = target_model.config
    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers

    # Capture layers: default [2, num_layers//2, num_layers-3] as in sglang
    if args.capture_layers:
        capture_layers = [int(x) for x in args.capture_layers.split(",")]
    else:
        capture_layers = [2, num_layers // 2, num_layers - 3]

    print(f"  Hidden size: {hidden_size}, Layers: {num_layers}")
    print(f"  Capture layers (hook before): {capture_layers}")

    # ---- 2. Register hooks ----
    captured_hidden: Dict[int, torch.Tensor] = {}
    hooks = []

    def make_pre_hook(layer_idx):
        def hook_fn(module, args):
            # args[0] is hidden_states (input to this layer)
            captured_hidden[layer_idx] = args[0].detach()
        return hook_fn

    layers = find_model_layers(target_model)
    for idx in capture_layers:
        h = layers[idx].register_forward_pre_hook(make_pre_hook(idx))
        hooks.append(h)

    embed_layer, lm_head = find_embed_and_head(target_model)

    # Check for scale_emb (MiniCPM specific)
    scale_emb = getattr(config, "scale_emb", 1.0)
    print(f"  scale_emb: {scale_emb}")

    # ---- 3. Load data ----
    print(f"[2/5] Loading training data from {args.data_path}...")
    dataset = TextDataset(args.data_path, tokenizer, args.max_seq_len)
    print(f"  Loaded {len(dataset)} examples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=True,
    )

    # ---- 4. Create draft model ----
    print(f"[3/5] Creating draft model...")
    draft_config = {
        "hidden_size": hidden_size,
        "intermediate_size": getattr(config, "intermediate_size", 14336),
        "num_attention_heads": getattr(config, "num_attention_heads", 32),
        "num_key_value_heads": getattr(config, "num_key_value_heads", 8),
        "rms_norm_eps": getattr(config, "rms_norm_eps", 1e-6),
        "max_position_embeddings": getattr(config, "max_position_embeddings", 8192),
        "rope_theta": getattr(config, "rope_theta", 10000.0),
    }

    draft_model = Eagle3DraftModel(draft_config).to(device=device, dtype=dtype)

    num_params = sum(p.numel() for p in draft_model.parameters())
    print(f"  Draft model params: {num_params / 1e6:.1f}M ({num_params * 2 / 1e9:.2f}GB FP16)")

    optimizer = AdamW(draft_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.num_steps, eta_min=args.lr * 0.1
    )

    # ---- 5. Training loop ----
    print(f"[4/5] Training for {args.num_steps} steps...")
    draft_model.train()

    step = 0
    total_loss = 0.0
    best_loss = float("inf")

    while step < args.num_steps:
        for batch in dataloader:
            if step >= args.num_steps:
                break

            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            bsz, seq_len = input_ids.shape

            # Skip very short sequences
            if (attn_mask.sum(dim=1) < 10).any():
                continue

            # ---- Target forward (no grad) ----
            with torch.no_grad():
                captured_hidden.clear()
                target_output = target_model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                )
                target_logits = target_output.logits  # [bsz, seq, vocab]

                # Gather captured hidden states
                aux_list = [captured_hidden[idx] for idx in capture_layers]
                aux_hidden = torch.cat(aux_list, dim=-1)  # [bsz, seq, 3*H]

                # Get embeddings (with scale_emb if applicable)
                embeds = embed_layer(input_ids)
                if scale_emb != 1.0:
                    embeds = embeds * scale_emb

            # ---- Draft forward ----
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
            draft_logits = draft_model(embeds, aux_hidden, positions, lm_head)

            # ---- KL divergence loss ----
            # Match target distribution at each position
            # Only compute on non-padded positions
            mask = attn_mask.bool()  # [bsz, seq]

            # Flatten for loss computation
            target_flat = target_logits[mask]  # [N, vocab]
            draft_flat = draft_logits[mask]    # [N, vocab]

            # Forward KL: KL(target || draft)
            target_log_probs = F.log_softmax(target_flat.float() / args.temperature, dim=-1)
            draft_log_probs = F.log_softmax(draft_flat.float() / args.temperature, dim=-1)

            # KL divergence = sum p_target * (log p_target - log p_draft)
            kl = F.kl_div(draft_log_probs, target_log_probs, reduction="batchmean",
                          log_target=True)
            loss = kl * (args.temperature ** 2)

            # ---- Backprop ----
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                draft_model.parameters(), args.max_grad_norm
            )
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            step += 1

            if step % args.log_interval == 0:
                avg_loss = total_loss / args.log_interval
                lr = scheduler.get_last_lr()[0]
                print(f"  Step {step}/{args.num_steps} | loss={avg_loss:.4f} | "
                      f"grad_norm={grad_norm:.2f} | lr={lr:.2e}")
                if avg_loss < best_loss:
                    best_loss = avg_loss
                total_loss = 0.0

            # Save checkpoint periodically
            if args.save_interval > 0 and step % args.save_interval == 0:
                ckpt_path = os.path.join(args.output_path, f"checkpoint-{step}")
                save_draft_model(draft_model, draft_config, config, capture_layers,
                                 ckpt_path)
                print(f"  Checkpoint saved to {ckpt_path}")

    # Cleanup hooks
    for h in hooks:
        h.remove()

    # ---- 6. Save final model ----
    print(f"[5/5] Saving draft model to {args.output_path}...")
    save_draft_model(draft_model, draft_config, config, capture_layers, args.output_path)
    print(f"  Best loss: {best_loss:.4f}")
    print("Done!")


def save_draft_model(draft_model, draft_config, target_config, capture_layers,
                     output_path):
    """Save draft model compatible with sglang's EAGLE3 weight loader."""
    os.makedirs(output_path, exist_ok=True)

    # --- Save weights ---
    state_dict = {}
    for name, param in draft_model.named_parameters():
        # Map: "layer." → "midlayer." to match sglang naming
        sglang_name = name.replace("layer.", "midlayer.")
        state_dict[sglang_name] = param.data.cpu()

    try:
        from safetensors.torch import save_file
        save_file(state_dict, os.path.join(output_path, "model.safetensors"))
    except ImportError:
        torch.save(state_dict, os.path.join(output_path, "pytorch_model.bin"))
        print("  Warning: safetensors not available, saved as pytorch_model.bin")

    # --- Save config ---
    config_dict = {
        "architectures": ["MiniCPMForCausalLMEagle3"],
        "model_type": "minicpm3",
        "hidden_size": draft_config["hidden_size"],
        "intermediate_size": draft_config["intermediate_size"],
        "num_attention_heads": draft_config["num_attention_heads"],
        "num_key_value_heads": draft_config["num_key_value_heads"],
        "num_hidden_layers": 1,
        "vocab_size": getattr(target_config, "vocab_size", 150528),
        "hidden_act": "silu",
        "rms_norm_eps": draft_config["rms_norm_eps"],
        "max_position_embeddings": draft_config["max_position_embeddings"],
        "rope_theta": draft_config["rope_theta"],
        "tie_word_embeddings": False,
        "draft_vocab_size": None,
        "eagle_config": {
            "use_aux_hidden_state": True,
            # Omit layer_ids to use default [2, num_layers//2, num_layers-3]
        },
        "torch_dtype": "bfloat16",
    }

    # Copy rope_scaling if present
    rope_scaling = getattr(target_config, "rope_scaling", None)
    if rope_scaling:
        if isinstance(rope_scaling, dict):
            config_dict["rope_scaling"] = rope_scaling
        else:
            config_dict["rope_scaling"] = dict(rope_scaling)

    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"  Saved {len(state_dict)} weight tensors to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EAGLE3 draft head for MiniCPM-SALA")
    parser.add_argument("--model-path", required=True, help="Path to BF16 target model")
    parser.add_argument("--data-path", required=True, help="Training data JSONL")
    parser.add_argument("--output-path", required=True, help="Output directory for draft model")
    parser.add_argument("--num-steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0, help="KL distillation temperature")
    parser.add_argument("--capture-layers", type=str, default=None,
                        help="Comma-separated layer indices to capture (default: auto)")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=0,
                        help="Save checkpoint every N steps (0=disabled)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train(args)
