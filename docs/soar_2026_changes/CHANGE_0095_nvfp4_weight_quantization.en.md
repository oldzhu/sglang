# CHANGE_0095: NVFP4 Weight Quantization

## Background and Motivation

NVFP4 (FP4 E2M1) weight quantization leverages native Blackwell tensor core acceleration for both weight storage AND compute. Unlike GPTQ W4A16 (which dequantizes to FP16 for GEMM), NVFP4 performs FP4×FP4 GEMM natively on SM120 tensor cores, potentially achieving ~2x compute throughput at high batch sizes.

Champion team Slightwind (3x week champion, score 79.22) uses NVFP4 weight quantization. Our fcloud GPU (NVIDIA RTX 6000D, compute 12.0, Blackwell) has native hardware support.

SGLang already has complete NVFP4 inference support via `ModelOptFp4LinearMethod` and `--quantization modelopt_fp4`, but NO offline quantization generation pipeline. This script fills that gap.

## Rule Compliance

- **On-site quantization** ✓ (Python script runs on competition machine)
- **Model size ≤ 2GB** ✓ (FP4 model is same or smaller than GPTQ INT4)
- **Quantization time** — estimated 10-20 minutes for calibration + quantization
- **SM120 required** ✓ (confirmed Blackwell GPU on eval machine)

## Implementation Plan

### Quantization Script: `benchmark/soar/demo_sala/quantize_nvfp4.py`

**Process**:
1. Load BF16 model from `/root/models/openbmb/MiniCPM-SALA-Copy`
2. **Calibration**: Run 32 forward passes with perf_public_set.jsonl data, recording max absolute activation values per Linear layer
3. **Weight quantization**: For each Linear layer:
   - Group weights into blocks of 128 (group_size)
   - Compute per-block FP8 E4M3 scales: `scale = max(abs(block)) / 6.0`
   - Round each weight to nearest FP4 E2M1 value: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
   - Pack two 4-bit values into one uint8
4. **Save checkpoint** with quantization config in both `config.json` and `hf_quant_config.json`

**Output Format**:
```
weight:         [out, in//2]           uint8        (packed FP4 E2M1)
weight_scale:   [out, in//group_size]  float8_e4m3  (per-block scales)
weight_scale_2: [1]                    float32      (per-tensor weight scale)
input_scale:    [1]                    float32      (calibrated activation scale)
```

### Server Configuration Changes

```bash
# In prepare_env.sh:
# Replace:  --quantization gptq_marlin
# With:     --quantization modelopt_fp4
# Remove:   --force-dense-minicpm (test both with and without)
```

## Risk Assessment

- **Accuracy risk: MEDIUM** — FP4 has only 16 distinct values, and activations are also quantized to FP4 online (unlike GPTQ which keeps activations in FP16)
- **Mitigation**: Careful calibration data selection, can exclude sensitive layers
- **Fallback**: Keep GPTQ config unchanged if accuracy drops below 97%

## Validation Commands

```bash
# Quantize model
python3 quantize_nvfp4.py \
    --input /root/models/openbmb/MiniCPM-SALA-Copy \
    --output /root/models/nvfp4_minicpm \
    --calib-data /root/data/perf_public_set.jsonl \
    --calib-samples 32 --group-size 128 --verify

# Start server with NVFP4
python3 -m sglang.launch_server \
    --model-path /root/models/nvfp4_minicpm \
    --quantization modelopt_fp4 ...

# Test accuracy + speed
python3 scripts/fcloud/fcloud_workflow.py accuracy
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

## Result Summary

| Metric | Baseline (GPTQ W4A16) | Expected (NVFP4 W4A4) |
|--------|----------------------|----------------------|
| S1 | 113.67s | ~105-115s (marginal) |
| S8 | 41.07s | ~35-40s |
| Smax | 34.15s | ~25-30s |
| Accuracy | 80.64% | ≥79% target |
| C | 1.0 | ≥0.96 target |

## Rollback

Revert `prepare_env.sh` to use `--quantization gptq_marlin --force-dense-minicpm` and point model_path back to the GPTQ model.

## Next Steps

1. **Test on fcloud** — quantize then run accuracy eval
2. **Tune exclude_modules** — find accuracy-sensitive layers to keep in higher precision
3. **Compare with ModelOpt** — if `nvidia-modelopt` is available, compare quality
4. **Combine with EAGLE3** — NVFP4 target + EAGLE3 speculation for maximum throughput
