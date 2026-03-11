# CHANGE_0024_demo_sala_quant_prep_scaffold

## 1) Background and Motivation
- User requested submission-script customization based on SOAR toolkit `提交说明`.
- Current demo only copies model files and cannot express quant-prep workflow intent.

## 2) Rule-Compliance Statement
- This change aligns with official submission interface:
  - `prepare_env.sh`
  - `prepare_model.sh --input --output`
  - `preprocess_model.py` as preprocess entry
- No model runtime algorithm change in this repository step.

## 3) Planned Change (Before Edit)
- Add quant mode awareness in `prepare_env.sh`.
- Extend `preprocess_model.py` with `copy|gptq` modes.
- Keep default behavior as safe `copy` mode.

## 4) Actual Code Changes
- `benchmark/soar/demo_sala/prepare_env.sh`
  - enabled strict shell mode (`set -euo pipefail`)
  - reads `SOAR_QUANT_MODE` (default `copy`)
  - appends `--quantization gptq` to `SGLANG_SERVER_ARGS` when mode is `gptq`
  - prints mode/args for traceability
- `benchmark/soar/demo_sala/preprocess_model.py`
  - added `--mode copy|gptq` (fallback to `SOAR_QUANT_MODE`)
  - refactored copy logic into helper
  - added GPTQ preflight checks (input file and dependency checks)
  - keeps GPTQ path as explicit scaffold and raises clear message for missing concrete quantization implementation

## 5) Validation Commands
```bash
bash -n benchmark/soar/demo_sala/prepare_env.sh
python3 -m py_compile benchmark/soar/demo_sala/preprocess_model.py

# default copy mode
bash benchmark/soar/demo_sala/prepare_model.sh --input /path/to/raw --output /path/to/out

# gptq mode scaffold (expected to fail with explicit guidance until implementation is added)
SOAR_QUANT_MODE=gptq bash benchmark/soar/demo_sala/prepare_model.sh --input /path/to/raw --output /path/to/out
```

## 6) Risk
- Low. Default path remains file copy.
- GPTQ mode intentionally fails fast with explicit guidance until project-specific quantization logic is added.

## 7) Rollback
1. Revert `benchmark/soar/demo_sala/prepare_env.sh`.
2. Revert `benchmark/soar/demo_sala/preprocess_model.py`.

## 8) Next-Step Suggestions
- Implement concrete GPTQ quantization in `preprocess_model.py` using your selected toolchain and produce a model directory containing required quant config files.
