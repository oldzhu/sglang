# CHANGE_0004_context_safe_speed_data

## 1) Background & Motivation
- Problem statement: benchmark warmup can fail when generated prompt+response lengths exceed model context limits.
- Observed symptom: sequence length warning/error (e.g. total length larger than model max context), followed by benchmark failure.
- Goal: make generated speed datasets context-safe by construction.

## 2) SOAR Rule-Compliance Check
- This is local benchmark tooling only.
- No changes to model weights, kernels, or forbidden evaluation behavior.

## 3) Plan Before Code Change
- Modify only `benchmark/soar/generate_speed_datasets.py`.
- Add context-budget controls and clamp each sample to fit the budget.
- Print max token stats to quickly verify generated data safety.

## 4) Actual Code Change
- Added CLI args:
  - `--max-context-tokens` (default `262144`)
  - `--safety-margin` (default `4096`)
  - `--min-output-tokens` (default `64`)
  - `--max-output-cap` (default `20000`)
- Added per-sample budget enforcement:
  - guarantee `input_tokens + output_tokens <= max_context_tokens - safety_margin`
- Added generation summary stats per file:
  - max input tokens, max output tokens, max total tokens, budget.

## 5) Validation Commands
```bash
python3 benchmark/soar/generate_speed_datasets.py --output-dir /root/soar_test_data
python3 benchmark/soar/generate_speed_datasets.py --output-dir /root/soar_test_data --max-context-tokens 262144 --safety-margin 4096
```

Then run one tier sanity check:
```bash
python3 benchmark/soar/run_soar_suite.py \
  --api-base http://127.0.0.1:30000 \
  --model-path /root/models/openbmb/MiniCPM-SALA \
  --speed-data-s1 /root/soar_test_data/speed_s1.jsonl \
  --num-prompts 32
```

## 6) Results Summary
| Metric | Baseline | New | Delta |
|---|---:|---:|---:|
| Warmup/context-limit failures |  |  |  |
| S1 benchmark completion |  |  |  |
| S8 benchmark completion |  |  |  |
| S∞ benchmark completion |  |  |  |

## 7) Risks
- Very low. Distribution is still synthetic and may not fully match hidden official datasets.

## 8) Rollback
1. Revert `benchmark/soar/generate_speed_datasets.py`.
2. Regenerate datasets with previous behavior.

## 9) Next-Step Suggestions
- If needed, add profile presets (`quick`, `balanced`, `heavy`) in a separate change.
