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

### Test 24: Dense-Calibrated GPTQ Baseline (no M1)

| Metric | Test 20 (sparse calib) | Test 24 (dense calib) | Delta |
|--------|----------------------|---------------------------|-------|
| ori_accuracy | 80.64% | **77.64%** | **-3.00%** |
| normalized | 100.80% | 97.05% | -3.75% |
| C | 1.0 | **0.92** | **REGRESSION** |
| mcq | 63.33% | **50.00%** | **-13.33%** |
| qa | 63.33% | 60.00% | -3.33% |
| cwe | 77.67% | 79.33% | +1.66% |
| fwe | 98.89% | 98.89% | 0 |
| niah | 100% | 100% | 0 |
| S1 (s) | 113.67 | 110.59 | -2.7% (faster) |
| S8 (s) | 41.07 | 40.45 | -1.5% (faster) |
| Smax (s) | 34.15 | 33.64 | -1.5% (faster) |

### Conclusion: **FAILED**

Dense calibration made accuracy significantly **worse**, not better. The mcq accuracy crashed from 63.33% to 50.00% and overall C dropped from 1.0 to 0.92. Speed improved marginally (~1.5-2.7%) but cannot compensate for the accuracy catastrophe.

**Key insight**: The original sparse-calibration GPTQ weights are actually **better** for dense inference than dense-calibrated weights. This is counter-intuitive but the GPTQ quantization grid optimized under sparse attention patterns generalizes well to dense inference. The sparse attention during calibration may act as a form of regularization, producing weight quantization that is more robust across attention patterns.

Test 25 (M1 + dense calibration) was **skipped** since the baseline (Test 24) already showed unacceptable accuracy regression — adding M1 on top of bad weights would not produce meaningful results.

## Rollback Instructions

Set `SOAR_GPTQ_FORCE_DENSE=0` in `prepare_env.sh` and re-quantize to restore sparse-calibrated weights. The sparse-calibrated weights remain the production baseline.

## Next Steps

- **Phase B**: Test dense calibration with tuning levers (`SOAR_GPTQ_DAMP_PERCENT`, `SOAR_GPTQ_MSE`) to recover accuracy while keeping speed benefit
- Use `quick-accuracy --tasks mcq` (~3-5 min) for fast screening before full eval
- If Phase B fails, revert `SOAR_GPTQ_FORCE_DENSE=0` and move to other optimizations

## Appendix: Quick Accuracy Evaluation Methods Comparison

Two approaches for quickly evaluating quantization quality before full accuracy runs (~50-60 min):

### Method A: Task-Filter (Implemented)

Run a subset of the actual eval tasks (e.g., MCQ-only, 30 samples, ~3-5 min).

- **Pros**: Directly measures end-to-end accuracy; covers both prefill and decode; no setup needed
- **Cons**: High variance (30 MCQ samples, each worth 3.33%); cannot reliably distinguish configs within ~10% accuracy noise; still requires generation (~6K tokens per MCQ sample)
- **Usage**: `python3 scripts/fcloud/fcloud_workflow.py quick-accuracy --tasks mcq`

### Method B: Logprob Distribution Comparison (Not yet implemented)

Compare quantized model vs BF16 baseline via logprobs distribution (KL divergence, cosine similarity). Reference: 曹议 (SOAR 2026 Week 3 champion blog), inspired by "Accuracy is Not All You Need" paper.

- **Approach**: One-time BF16 baseline (greedy decode, extract top-256 logprobs of last 128 tokens). Per-config: send same prompts, compare distributions.
- **Pros**: Very fast (seconds-minutes, prefill only); stable continuous metric — can reliably rank configs; zero GPU for baseline data
- **Cons**: Prefill-only — no decode error propagation coverage; doesn't directly predict competition accuracy score; requires one-time BF16 baseline setup
- **When to implement**: If we need to compare >5 calibration configs, where MCQ noise makes task-filter unreliable for ranking

### Recommended Workflow

1. **Logprob** (if implemented) → screen and rank many configs quickly
2. **Task-filter** → validate top candidates pass accuracy threshold
3. **Full eval** → final verification before submission
