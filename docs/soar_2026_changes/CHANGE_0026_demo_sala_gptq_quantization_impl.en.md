# CHANGE_0026_demo_sala_gptq_quantization_impl

## 1) Background and Motivation
- Previous demo only provided GPTQ scaffold and preflight checks.
- Team needs a concrete quant-prep path in `preprocess_model.py` for submission workflow.

## 2) Rule-Compliance Statement
- Change is submission preprocessing logic only (`prepare_model.sh --input/--output` contract preserved).
- No forbidden runtime tricks; no base-model replacement in repository runtime path.

## 3) Planned Feature
- Implement real GPTQ mode in `preprocess_model.py` using GPTQModel.
- Keep `copy` mode as default fallback.
- Add configurable calibration settings via CLI/env.

## 4) Actual Code Changes
- `benchmark/soar/demo_sala/preprocess_model.py`
  - Added JSONL calibration loader.
  - Added GPTQ quantization execution function:
    - `GPTQModel.load(...)`
    - `model.quantize(calibration_texts, batch_size=...)`
    - `model.save(output_dir)`
  - Added required-output validation for `quantize_config.json`.
  - Added CLI/env parameters:
    - `--calibration-file` / `SOAR_GPTQ_CALIBRATION_FILE`
    - `--calibration-field` / `SOAR_GPTQ_CALIBRATION_FIELD`
    - `--calibration-samples` / `SOAR_GPTQ_CALIBRATION_SAMPLES`
    - `--gptq-bits` / `SOAR_GPTQ_BITS`
    - `--gptq-group-size` / `SOAR_GPTQ_GROUP_SIZE`
    - `--gptq-batch-size` / `SOAR_GPTQ_BATCH_SIZE`
- `benchmark/soar/demo_sala/README.md`
  - Added concrete `gptq` mode usage and environment configuration examples.

## 5) Validation Commands
```bash
python3 -m py_compile benchmark/soar/demo_sala/preprocess_model.py

# copy mode
bash benchmark/soar/demo_sala/prepare_model.sh --input /path/raw --output /path/out

# gptq mode
SOAR_QUANT_MODE=gptq \
SOAR_GPTQ_CALIBRATION_FILE=/path/calibration.jsonl \
bash benchmark/soar/demo_sala/prepare_model.sh --input /path/raw --output /path/out
```

## 6) Risks
- Medium. Quantization quality and runtime depend on calibration data quality and dependency availability.
- Mitigation: explicit dependency checks and strict output validation.

## 7) Rollback
1. Revert `benchmark/soar/demo_sala/preprocess_model.py`.
2. Revert related README section in `benchmark/soar/demo_sala/README.md`.

## 8) Next-Step Suggestions
- Add one focused follow-up feature for marlin-oriented compatibility validation and launch profile for quantized model benchmark.
