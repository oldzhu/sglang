# CHANGE_0029_gptq_drop_unsupported_internal_attn_kwarg

## 1) Background and Motivation
- After CHANGE_0028, GPTQ load failed with:
  - `TypeError: MiniCPMSALAForCausalLM.__init__() got an unexpected keyword argument '_attn_implementation'`
- The internal kwarg is incompatible with current model constructor signature.

## 2) Rule-Compliance Statement
- Preprocess compatibility fix only.
- No competition runtime behavior manipulation.

## 3) Planned Feature
- Keep `attn_implementation=flash_attention_2`.
- Remove unsupported `_attn_implementation` kwarg from `GPTQModel.load(...)`.

## 4) Actual Code Changes
- `benchmark/soar/demo_sala/preprocess_model.py`
  - Removed `_attn_implementation=attn_impl` in GPTQ load call.

## 5) Validation Commands
```bash
python3 -m py_compile benchmark/soar/demo_sala/preprocess_model.py

export SOAR_QUANT_MODE=gptq
export SOAR_TRUST_REMOTE_CODE=true
export SOAR_GPTQ_ATTN_IMPL=flash_attention_2
export SOAR_GPTQ_CALIBRATION_FILE=/root/data/perf_public_set.jsonl
bash benchmark/soar/demo_sala/prepare_model.sh \
  --input /root/models/openbmb/MiniCPM-SALA \
  --output /root/models/openbmb/MiniCPM-SALA-gptq
```

## 6) Risk
- Low. We only removed an unsupported argument while preserving required attention override.

## 7) Rollback
1. Revert GPTQ load call in `preprocess_model.py`.

## 8) Next-Step Suggestions
- If preprocess succeeds, proceed to quantized serving and eval/bench comparison.
