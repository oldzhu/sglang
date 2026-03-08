# CHANGE_0028_gptq_flash_attn2_load_fix

## 1) Background and Motivation
- GPTQ preprocess reached model construction but failed with:
  - `AssertionError: Only flash_attention_2 is supported for sparse attention`
- MiniCPM-SALA requires flash-attention-2 implementation flags during model initialization.

## 2) Rule-Compliance Statement
- Preprocess compatibility fix only.
- No runtime evaluation trick and no model replacement.

## 3) Planned Feature
- Force attention implementation at GPTQ load time.
- Add env override for implementation name.

## 4) Actual Code Changes
- `benchmark/soar/demo_sala/preprocess_model.py`
  - Added env option: `SOAR_GPTQ_ATTN_IMPL` (default `flash_attention_2`)
  - Passed both `attn_implementation` and `_attn_implementation` into `GPTQModel.load(...)`
  - Added log output with resolved attention implementation

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
- Low-to-medium: depends on GPTQModel argument compatibility.
- Mitigation: pass both common argument names for broader compatibility.

## 7) Rollback
1. Revert modified `GPTQModel.load(...)` kwargs and env handling in `preprocess_model.py`.

## 8) Next-Step Suggestions
- If this passes, proceed to quantized serving and eval/bench comparison.
