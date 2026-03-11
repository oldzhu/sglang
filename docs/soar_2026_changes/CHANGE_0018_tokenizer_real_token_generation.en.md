# CHANGE_0018_tokenizer_real_token_generation

## 1) Background & Motivation
- Phrase-count-based synthetic generation can drift from real token cost and distort benchmark time planning.
- Goal: generate benchmark samples based on real tokenizer token lengths.

## 2) SOAR Rule-Compliance Check
- Tooling-only benchmark dataset generation change.
- No model/inference kernel/runtime algorithm changes.

## 3) Plan Before Code Change
- Modify one file: `benchmark/soar/generate_speed_datasets.py`.
- Replace phrase-count stopping criterion with tokenizer token-length criterion.

## 4) Actual Code Change
- Added tokenizer loading via `AutoTokenizer.from_pretrained(..., trust_remote_code=True)`.
- Added `--model-path` argument (falls back to `OpenBMB/MiniCPM-SALA` when unset).
- Updated text generation loop to stop based on **real token count** (`tokenizer.encode(..., add_special_tokens=False)`).
- Kept existing profile controls and row/time estimation outputs.

## 5) Usage
```bash
python3 benchmark/soar/generate_speed_datasets.py \
  --profile quick10 \
  --model-path /root/models/openbmb/MiniCPM-SALA \
  --output-dir /root/soar_fast_data
```

## 6) Risks
- Low. Generation may be slightly slower due to tokenization in the loop.

## 7) Rollback
1. Revert `benchmark/soar/generate_speed_datasets.py`.

## 8) Next-Step Suggestions
- Optionally add a source corpus mode (reuse real prompts from local JSONL) in a future change.
