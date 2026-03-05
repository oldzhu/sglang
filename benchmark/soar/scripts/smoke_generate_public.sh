#!/usr/bin/env bash
set -euo pipefail

python3 benchmark/soar/generate_speed_datasets.py \
  --prompt-source public \
  --source-jsonl /root/data/perf_public_set.jsonl \
  --source-prompt-field question \
  --model-path /root/models/openbmb/MiniCPM-SALA \
  --profile quick10 \
  --output-dir /root/soar_fast_data
