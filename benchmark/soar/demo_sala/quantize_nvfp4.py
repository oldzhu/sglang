#!/usr/bin/env python3
"""Quantize MiniCPM-SALA weights to NVFP4 (FP4 E2M1) format.

Performs offline Post-Training Quantization (PTQ) to convert BF16/FP16 weights
to NVFP4 format compatible with SGLang's ModelOptFp4LinearMethod.

Output checkpoint format:
  weight:        [out, in//2]          uint8   (2 packed FP4 E2M1 per byte)
  weight_scale:  [out, in//group_size] float8_e4m3fn (per-block scales)
  weight_scale_2: [1]                  float32 (per-tensor weight scale)
  input_scale:   [1]                   float32 (per-tensor input scale, calibrated)

Usage:
    python3 quantize_nvfp4.py \
        --input /root/models/openbmb/MiniCPM-SALA-Copy \
        --output /root/models/nvfp4_minicpm \
        --calib-data /root/data/perf_public_set.jsonl \
        --calib-samples 32 --group-size 128
"""

import argparse
import copy
import gc
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# FP4 E2M1 representable values (positive)
# Encoding: bit3=sign, bit2-1=exponent, bit0=mantissa
# Values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
FP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP4_MAX = 6.0


def quantize_to_fp4_e2m1(values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize float values to FP4 E2M1.

    Args:
        values: float tensor, expected range roughly [-6, 6] after scaling

    Returns:
        codes: uint8 tensor of 4-bit FP4 codes (0-15)
        quantized: dequantized float values (for error analysis)
    """
    fp4_pos = FP4_VALUES.to(device=values.device, dtype=values.dtype)

    sign = (values < 0).to(torch.uint8)
    abs_vals = values.abs()

    # Find nearest FP4 value via lookup
    # abs_vals: [...], fp4_pos: [8] → distances: [..., 8]
    distances = (abs_vals.unsqueeze(-1) - fp4_pos).abs()
    indices = distances.argmin(dim=-1).to(torch.uint8)  # 0-7

    # 4-bit code: sign(1) | magnitude_index(3)
    codes = (sign << 3) | indices

    # Dequantized values for error analysis
    quantized = fp4_pos[indices.long()] * (1 - 2 * sign.float())

    return codes, quantized


def pack_fp4_to_uint8(codes: torch.Tensor) -> torch.Tensor:
    """Pack pairs of FP4 4-bit codes into uint8.

    Args:
        codes: [..., N] uint8 tensor where each value is 0-15
               N must be even

    Returns:
        packed: [..., N//2] uint8 tensor where each byte = (high_nibble << 4) | low_nibble
    """
    assert codes.shape[-1] % 2 == 0, f"Last dim must be even, got {codes.shape[-1]}"
    even = codes[..., 0::2]  # low nibble (first element of pair)
    odd = codes[..., 1::2]   # high nibble (second element of pair)
    return ((odd << 4) | even).to(torch.uint8)


def quantize_weight(weight: torch.Tensor, group_size: int = 128
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a weight tensor to NVFP4 format.

    Args:
        weight: [out_features, in_features] float tensor
        group_size: elements per scale group (default 128)

    Returns:
        weight_packed: [out_features, in_features//2] uint8
        weight_scale: [out_features, in_features//group_size] float8_e4m3fn
        weight_scale_2: scalar float32 (per-tensor scale)
    """
    out_features, in_features = weight.shape

    # Pad input dimension to multiple of group_size if needed
    pad_amount = (group_size - in_features % group_size) % group_size
    if pad_amount > 0:
        weight = F.pad(weight, (0, pad_amount), value=0.0)
        in_features = weight.shape[1]

    assert in_features % group_size == 0
    num_groups = in_features // group_size

    # Reshape for groupwise quantization
    W = weight.reshape(out_features, num_groups, group_size)

    # Per-group max absolute values
    group_max = W.abs().amax(dim=-1)  # [out, num_groups]

    # Per-group scale: maps group values to [-6, 6] range
    # scale = max_val / 6.0, clamped to avoid division by zero
    group_scale = (group_max / FP4_MAX).clamp(min=1e-12)

    # Convert to float8_e4m3fn (clamp to FP8 representable range)
    # FP8 E4M3 max value is 448.0
    group_scale_fp8 = group_scale.to(torch.float8_e4m3fn)

    # Use the FP8-rounded scale for quantization (matches inference dequant path)
    group_scale_actual = group_scale_fp8.float()

    # Per-tensor weight scale (weight_scale_2)
    # This is an additional global scale factor
    weight_scale_2 = torch.tensor(1.0, dtype=torch.float32)

    # Quantize each group using the FP8-rounded scales
    W_scaled = W / group_scale_actual.unsqueeze(-1)  # [out, num_groups, group_size]
    W_flat = W_scaled.reshape(out_features, in_features)

    # Quantize to FP4 E2M1
    codes, _ = quantize_to_fp4_e2m1(W_flat)

    # Pack pairs into uint8
    weight_packed = pack_fp4_to_uint8(codes)

    return weight_packed, group_scale_fp8, weight_scale_2


def calibrate_input_scales(model, tokenizer, data_path: str, num_samples: int = 32,
                           max_seq_len: int = 2048, device: torch.device = None
                           ) -> Dict[str, float]:
    """Run calibration forward passes to compute per-layer input scales.

    Args:
        model: loaded BF16 model
        tokenizer: model tokenizer
        data_path: JSONL file with calibration data
        num_samples: number of calibration samples
        max_seq_len: max sequence length

    Returns:
        Dict mapping layer name → max absolute input value
    """
    import torch.nn.functional as F

    if device is None:
        device = next(model.parameters()).device

    # Collect input statistics per Linear layer
    input_max: Dict[str, float] = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, args, output):
            x = args[0]
            max_val = x.abs().max().item()
            if name in input_max:
                input_max[name] = max(input_max[name], max_val)
            else:
                input_max[name] = max_val
        return hook_fn

    # Register hooks on all Linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Load calibration data
    texts = []
    with open(data_path, "r") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            item = json.loads(line.strip())
            text = item.get("question", item.get("text", item.get("prompt", "")))
            if text:
                texts.append(text)

    print(f"  Running {len(texts)} calibration samples...")
    model.eval()
    with torch.no_grad():
        for i, text in enumerate(texts):
            tokens = tokenizer(
                text, max_length=max_seq_len, truncation=True, return_tensors="pt"
            ).to(device)
            model(**tokens)
            if (i + 1) % 10 == 0:
                print(f"    Calibrated {i+1}/{len(texts)} samples")

    # Remove hooks
    for h in hooks:
        h.remove()

    print(f"  Collected input scales for {len(input_max)} layers")
    return input_max


import torch.nn.functional as F


def quantize_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # ---- 1. Load model ----
    print(f"[1/4] Loading model from {args.input}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.input, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.input,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map={"": device},
    )
    model.eval()

    config = model.config
    print(f"  Model: {config.architectures}")
    print(f"  Hidden size: {config.hidden_size}, Layers: {config.num_hidden_layers}")

    # ---- 2. Calibrate input scales ----
    print(f"[2/4] Calibrating input scales ({args.calib_samples} samples)...")
    input_scales = calibrate_input_scales(
        model, tokenizer, args.calib_data,
        num_samples=args.calib_samples,
        max_seq_len=args.max_seq_len,
        device=device,
    )

    # ---- 3. Quantize weights ----
    print(f"[3/4] Quantizing weights to NVFP4 (group_size={args.group_size})...")

    # Build exclude set
    exclude_modules = set(args.exclude_modules.split(",")) if args.exclude_modules else set()
    exclude_modules.add("lm_head")  # Always keep lm_head in full precision

    quantized_state_dict = {}
    original_state_dict = model.state_dict()
    num_quantized = 0
    num_skipped = 0
    total_orig_bytes = 0
    total_quant_bytes = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        # Check exclusion
        short_name = name.split(".")[-1]
        if short_name in exclude_modules or any(ex in name for ex in exclude_modules):
            num_skipped += 1
            # Keep original weight
            quantized_state_dict[f"{name}.weight"] = module.weight.data.cpu()
            total_orig_bytes += module.weight.numel() * 2  # BF16
            total_quant_bytes += module.weight.numel() * 2
            continue

        # Check dimension compatibility
        out_features, in_features = module.weight.shape
        if in_features % 16 != 0:
            print(f"  Skip {name}: in_features={in_features} not multiple of 16")
            num_skipped += 1
            quantized_state_dict[f"{name}.weight"] = module.weight.data.cpu()
            continue

        # Quantize
        weight = module.weight.data.float()
        weight_packed, weight_scale, weight_scale_2 = quantize_weight(
            weight, group_size=args.group_size
        )

        # Get input scale from calibration
        input_scale = torch.tensor(
            input_scales.get(name, 1.0), dtype=torch.float32
        )

        # Store quantized tensors
        quantized_state_dict[f"{name}.weight"] = weight_packed.cpu()
        quantized_state_dict[f"{name}.weight_scale"] = weight_scale.cpu()
        quantized_state_dict[f"{name}.weight_scale_2"] = weight_scale_2.cpu().unsqueeze(0)
        quantized_state_dict[f"{name}.input_scale"] = input_scale.cpu().unsqueeze(0)

        num_quantized += 1
        total_orig_bytes += out_features * in_features * 2  # BF16
        total_quant_bytes += weight_packed.numel() + weight_scale.numel() + 8  # packed + scales

        if num_quantized % 20 == 0:
            print(f"    Quantized {num_quantized} layers...")

    # Copy non-Linear parameters (embeddings, norms, etc.)
    for name, param in model.named_parameters():
        if name not in quantized_state_dict and not any(
            name.startswith(f"{qn}.") for qn in [n for n, m in model.named_modules()
                                                   if isinstance(m, nn.Linear)]
        ):
            quantized_state_dict[name] = param.data.cpu()

    compression = total_orig_bytes / max(total_quant_bytes, 1)
    print(f"  Quantized: {num_quantized} layers, Skipped: {num_skipped}")
    print(f"  Size: {total_orig_bytes/1e9:.2f}GB → {total_quant_bytes/1e9:.2f}GB "
          f"({compression:.1f}x compression)")

    # ---- 4. Save ----
    print(f"[4/4] Saving to {args.output}...")
    os.makedirs(args.output, exist_ok=True)

    # Save weights
    try:
        from safetensors.torch import save_file
        save_file(quantized_state_dict, os.path.join(args.output, "model.safetensors"))
    except ImportError:
        torch.save(quantized_state_dict, os.path.join(args.output, "pytorch_model.bin"))
        print("  Warning: safetensors not available, saved as pytorch_model.bin")

    # Save quantization config
    quant_config = {
        "quant_algo": "NVFP4",
        "group_size": args.group_size,
        "kv_cache_quant_algo": args.kv_cache_quant,
        "exclude_modules": list(exclude_modules),
    }

    # Update model config
    orig_config = config.to_dict()
    orig_config["quantization_config"] = quant_config
    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(orig_config, f, indent=2)

    # Also save hf_quant_config.json (legacy format, some loaders prefer this)
    hf_quant = {"quantization": quant_config}
    with open(os.path.join(args.output, "hf_quant_config.json"), "w") as f:
        json.dump(hf_quant, f, indent=2)

    # Copy tokenizer files
    tokenizer.save_pretrained(args.output)

    print(f"  Saved {len(quantized_state_dict)} tensors")
    print("Done!")

    # Quick error analysis
    if args.verify:
        verify_quantization(model, quantized_state_dict, args.group_size)


def verify_quantization(model, quantized_state_dict, group_size):
    """Quick error analysis comparing original vs quantized weights."""
    print("\n--- Quantization Error Analysis ---")
    print(f"{'Layer':<50} {'RMSE':>10} {'MaxErr':>10} {'Scale':>10}")
    print("-" * 85)

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        weight_key = f"{name}.weight"
        scale_key = f"{name}.weight_scale"

        if scale_key not in quantized_state_dict:
            continue  # Not quantized

        # Dequantize for comparison
        packed = quantized_state_dict[weight_key]
        scales = quantized_state_dict[scale_key]
        weight_scale_2 = quantized_state_dict[f"{name}.weight_scale_2"]

        orig = module.weight.data.float().cpu()
        out_f, in_f = orig.shape

        # Unpack FP4
        low = (packed & 0x0F)
        high = (packed >> 4)
        codes = torch.zeros(out_f, packed.shape[1] * 2, dtype=torch.uint8)
        codes[:, 0::2] = low
        codes[:, 1::2] = high

        # Decode FP4
        sign = ((codes >> 3) & 1).float()
        magnitude_idx = (codes & 0x07).long()
        fp4_pos = FP4_VALUES
        magnitude = fp4_pos[magnitude_idx]
        dequant = magnitude * (1 - 2 * sign)

        # Apply scales
        num_groups = in_f // group_size
        dequant = dequant[:, :in_f].reshape(out_f, num_groups, group_size)
        scales_float = scales.float()[:, :num_groups].unsqueeze(-1)
        dequant = (dequant * scales_float * weight_scale_2.float()).reshape(out_f, in_f)

        rmse = (orig - dequant).pow(2).mean().sqrt().item()
        max_err = (orig - dequant).abs().max().item()
        w_scale = orig.abs().max().item()

        print(f"{name:<50} {rmse:>10.6f} {max_err:>10.6f} {w_scale:>10.4f}")

    print("-" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize model to NVFP4")
    parser.add_argument("--input", required=True, help="Input BF16 model path")
    parser.add_argument("--output", required=True, help="Output quantized model path")
    parser.add_argument("--calib-data", required=True, help="Calibration data JSONL")
    parser.add_argument("--calib-samples", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--exclude-modules", type=str, default="",
                        help="Comma-separated module names to exclude from quantization")
    parser.add_argument("--kv-cache-quant", type=str, default="FP8",
                        help="KV cache quantization algorithm (FP8, NVFP4, or none)")
    parser.add_argument("--verify", action="store_true",
                        help="Run quantization error analysis after quantizing")

    args = parser.parse_args()
    quantize_model(args)
