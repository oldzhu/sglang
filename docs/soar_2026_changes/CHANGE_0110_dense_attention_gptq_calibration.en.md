# CHANGE_0110: Dense-Attention GPTQ Calibration

## Background and Motivation

MiniCPM-SALA has 8 standard attention layers (indices 0,9,16,17,22,29,30,31) and 24 lightning attention layers. The standard attention layers support two modes:

- **Sparse (InfLLMv2)**: topk=96 block selection for sequences ≥ 8192 tokens
- **Dense**: full attention over all tokens

During GPTQ calibration, the HuggingFace model loads with `sparse_config` from `config.json`, causing `MiniCPMInfLLMv2Attention` to be instantiated. For calibration sequences ≥ 8192 tokens (many qa/cwe samples are 32K–128K), sparse attention is used.

However, at inference time, sglang always uses `--force-dense-minicpm` which replaces sparse attention with dense FlashInfer/FlashAttention backends. This creates a **calibration/inference attention mismatch** that may degrade quantization quality — the GPTQ weight quantization grid is optimized for activation patterns produced by sparse attention, but inference sees dense attention patterns.

Since GPTQ is W4A16 (weights-only quantization), the mismatch is second-order — it affects which weight columns see higher activation magnitudes during calibration, not quantized activations. However, eliminating this mismatch aligns quantization more closely with runtime behavior.

## Rule-Compliance Statement

- Only modifies quantization calibration procedure — no model architecture or weight format changes
- Fully compliant with competition rules (on-site quantization, ≤ 5h, ≤ 2GB)
- No forbidden tricks; simply aligns calibration attention with inference attention
- Controlled by `SOAR_GPTQ_FORCE_DENSE` env var (default=1, easily reversible with =0)

## Implementation Plan

### Approach: Null out `sparse_config` in calibration config

Setting `sparse_config = None` in the sanitized config before `GPTQModel.load()` causes the HF model to instantiate `MiniCPMFlashAttention2` (dense) instead of `MiniCPMInfLLMv2Attention` for all minicpm4 layers. This is the cleanest approach because:

1. It uses the existing `_prepare_gptq_load_source()` temp-config mechanism
2. No monkey-patching of model forward methods
3. Eliminates the `infllmv2` kernel dependency during calibration
4. Exactly matches the `--force-dense-minicpm` runtime behavior

### Files Changed

1. **`benchmark/soar/demo_sala/preprocess_model.py`** — `_sanitize_model_config_for_gptq()`
   - Added: check `SOAR_GPTQ_FORCE_DENSE` env var (default True)
   - When enabled and `sparse_config` is not None: set `sparse_config = None` and log the change

2. **`benchmark/soar/demo_sala/prepare_env.sh`**
   - Added: `export SOAR_GPTQ_FORCE_DENSE="${SOAR_GPTQ_FORCE_DENSE:-1}"`
   - Added: echo line for logging

## Actual Code Changes

### preprocess_model.py — `_sanitize_model_config_for_gptq()`

```python
# Added after rope_type removal block:
force_dense = _env_truthy("SOAR_GPTQ_FORCE_DENSE", default=True)
if force_dense and sanitized.get("sparse_config") is not None:
    sanitized["sparse_config"] = None
    changes.append(
        "set sparse_config=null to force dense attention during GPTQ calibration "
        "(matches --force-dense-minicpm inference mode)"
    )
```

### prepare_env.sh

```bash
export SOAR_GPTQ_FORCE_DENSE="${SOAR_GPTQ_FORCE_DENSE:-1}"
```

## Validation Commands

### Step 1: Recalibrate baseline (no M1 code)

```bash
# On fcloud — re-quantize with dense calibration
source /root/submission_sim/prepare_env.sh
python3 /root/submission_sim/prepare_model.sh \
  --input /root/models/openbmb/MiniCPM-SALA-Copy \
  --output /root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8

# Restart server and run accuracy + speed
python3 scripts/fcloud/fcloud_workflow.py full
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

### Step 2: Retest M1 on new weights

```bash
# Enable M1 branch, restart server, retest
python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py accuracy
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

## Result Summary

| Metric | Test 20 (sparse calib) | Test 24 (dense calib, TBD) | Test 25 (dense calib + M1, TBD) |
|--------|----------------------|---------------------------|--------------------------------|
| ori_accuracy | 80.64% | TBD | TBD |
| normalized | 100.80% | TBD | TBD |
| C | 1.0 | TBD | TBD |
| S1 (s) | 113.67 | TBD | TBD |
| S8 (s) | 41.07 | TBD | TBD |
| Smax (s) | 34.15 | TBD | TBD |

## Rollback Instructions

Set `SOAR_GPTQ_FORCE_DENSE=0` in `prepare_env.sh` and re-quantize to restore sparse-calibrated weights.

## Next Steps

- If dense calibration improves or maintains accuracy: adopt as new default, proceed to test M1 path
- If M1 + dense calibration achieves normalized accuracy >99% (C=1.0): adopt M1 for submission
- If accuracy still regresses with M1: move to next optimization priority (A1 or K4)
