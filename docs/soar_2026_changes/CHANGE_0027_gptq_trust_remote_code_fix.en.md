# CHANGE_0027_gptq_trust_remote_code_fix

## 1) Background and Motivation
- GPTQ preprocess failed on MiniCPM-SALA with:
  - `ValueError: ... contains custom code ... pass trust_remote_code=True`
- This blocks quant-prep before any actual quantization runs.

## 2) Rule-Compliance Statement
- Preprocessing-stage reliability fix only.
- Keeps official `prepare_model.sh --input --output` flow.
- No runtime evaluation rule bypass.

## 3) Planned Feature
- Enable trusted custom code loading in GPTQModel load path.
- Add env override for explicit control.

## 4) Actual Code Changes
- `benchmark/soar/demo_sala/preprocess_model.py`
  - Added `trust_remote_code` resolution from env:
    - `SOAR_TRUST_REMOTE_CODE` (default `true`)
  - Passed `trust_remote_code` into `GPTQModel.load(...)`.
  - Added log output for resolved flag.

## 5) Validation Commands
```bash
python3 -m py_compile benchmark/soar/demo_sala/preprocess_model.py

export SOAR_QUANT_MODE=gptq
export SOAR_GPTQ_CALIBRATION_FILE=/root/data/perf_public_set.jsonl
bash benchmark/soar/demo_sala/prepare_model.sh \
  --input /root/models/openbmb/MiniCPM-SALA \
  --output /root/models/openbmb/MiniCPM-SALA-gptq
```

## 6) Risk
- `trust_remote_code=True` executes model repository custom code.
- For MiniCPM-SALA this is required to load config/model correctly.

## 7) Rollback
1. Revert `benchmark/soar/demo_sala/preprocess_model.py` to previous GPTQ load call.

## 8) Next-Step Suggestions
- After preprocess succeeds, run eval + 3x speed to measure quantized runtime trade-off.
