# CHANGE_0019_prompt_source_public_custom

## 1) Background & Motivation
- User requested more realistic benchmark prompts by loading from public eval JSONL or custom JSONL.
- Goal: reduce synthetic prompt repetition and improve local benchmark relevance.

## 2) SOAR Rule-Compliance Check
- Tooling-only benchmark data generation change.
- No model/kernel/runtime algorithm changes.

## 3) Plan Before Code Change
- Extend `benchmark/soar/generate_speed_datasets.py` with source modes and source field options.
- Keep tokenizer-based token-length control and existing profiles.

## 4) Actual Code Change
- Added CLI options:
  - `--prompt-source synthetic|public|custom`
  - `--source-jsonl`
  - `--source-prompt-field` (default `question`)
  - `--source-response-field` (optional)
- Added JSONL pool loader with validation.
- Prompt generation behavior:
  - if source mode is public/custom: sample base prompt from source pool and stretch/trim to target token length.
  - if synthetic: keep previous synthetic phrase mode.
- Optional response seed pool is supported via `--source-response-field`.

## 5) Usage
```bash
# Use public eval prompts
python3 benchmark/soar/generate_speed_datasets.py \
  --prompt-source public \
  --source-jsonl /root/data/perf_public_set.jsonl \
  --source-prompt-field question \
  --model-path /root/models/openbmb/MiniCPM-SALA \
  --profile quick10 \
  --output-dir /root/soar_fast_data

# Use custom jsonl prompts
python3 benchmark/soar/generate_speed_datasets.py \
  --prompt-source custom \
  --source-jsonl /root/data/my_prompts.jsonl \
  --source-prompt-field question \
  --model-path /root/models/openbmb/MiniCPM-SALA \
  --profile quick10 \
  --output-dir /root/soar_fast_data
```

## 6) Risks
- Low. Malformed/empty source files will now fail fast with clear errors.

## 7) Rollback
1. Revert `benchmark/soar/generate_speed_datasets.py`.

## 8) Next-Step Suggestions
- In a future change, add optional prompt dedup and stratified sampling by source length buckets.
