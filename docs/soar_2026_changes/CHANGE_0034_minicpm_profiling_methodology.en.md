# CHANGE_0034_minicpm_profiling_methodology

## 1) Background & Motivation
- Problem statement: recent MiniCPM high-concurrency work on this branch has relied too much on local reasoning and small smoke-test datasets, which is insufficient to distinguish CPU scheduling overhead, CUDA-graph replay overhead, sparse metadata overhead, and true kernel bottlenecks.
- Why this should improve speed work quality: a repeatable profiling workflow reduces blind trial-and-error, shortens the loop from symptom to root cause, and increases the chance that the next code change targets the dominant bottleneck instead of a secondary effect.
- Target stage(s): benchmark methodology / request metrics / CUDA timeline / CPU hot-path diagnosis

## 2) SOAR Rule-Compliance Check
- Allowed by rules because: this iteration adds profiling methodology documentation only; it does not modify model weights, inference outputs, benchmark concurrency, or submission interfaces.
- Latest rule/toolkit re-check (2026-03-11): the official competition page still requires fixed concurrency, disables prefix-cache advantages during benchmark, and emphasizes reproducible inference optimization; the toolkit page still supports self-testing with custom speed JSONL inputs and the official correctness script.
- Alignment with `技术路径指引`: profiling is not a separate scoring trick; it is a diagnostic workflow that supports the officially encouraged runtime optimization, quantization, and speculative-decoding directions.
- Expected impact on correctness coefficient C: neutral; this doc pair does not change runtime behavior.

## 3) Plan Before Code Change
- Files/functions to modify: add two documentation files only under `docs/soar_2026_changes/`; no runtime source files are modified in this feature iteration.
- Minimal diff strategy: document a single profiling workflow that reuses existing repo entrypoints instead of inventing new scripts.
- Measurement plan:
  - Generate one shared representative dataset with `benchmark/soar/generate_speed_datasets.py`.
  - Run all three speed tiers against the same shared file through `benchmark/soar/run_soar_suite.py`.
  - Launch MiniCPM with existing probe-style serving arguments from `benchmark/soar/scripts/launch_profile.sh`.
  - Collect request metrics, Nsight Systems traces, coarse GPU telemetry, and optional Linux `perf` stacks.
  - Use the traces to decide whether the next optimization should target host scheduling, CUDA-graph replay, sparse metadata setup, or kernels.
- Rollback plan: delete this doc pair if the methodology record is no longer wanted.

## 4) Actual Code Change (After Approval)
- Commit/patch summary: documentation-only addition of a repo-specific profiling methodology for MiniCPM-SALA high-concurrency diagnosis.
- Final modified files: `docs/soar_2026_changes/CHANGE_0034_minicpm_profiling_methodology.en.md`, `docs/soar_2026_changes/CHANGE_0034_minicpm_profiling_methodology.zh.md`.
- Key logic differences:
  - No runtime logic changed.
  - The branch now records a standard profiling sequence before future performance patches are chosen.
  - The methodology explicitly separates smoke tests from decision-grade profiling runs.

## 5) Validation Commands
### Shared Dataset Generation
```bash
python3 benchmark/soar/generate_speed_datasets.py \
  --profile balanced \
  --prompt-source synthetic \
  --output-dir benchmark/soar/data/balanced_shared_v1 \
  --model-path /root/models/openbmb/MiniCPM-SALA
```

### Server Launch For Profiling
```bash
python3 -m sglang.launch_server \
  --model-path /root/models/openbmb/MiniCPM-SALA \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code \
  --disable-radix-cache \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 32768 \
  --prefill-max-requests 1 \
  --max-running-requests 20 \
  --mem-fraction-static 0.84 \
  --schedule-conservativeness 1.0 \
  --skip-server-warmup \
  --enable-metrics \
  --export-metrics-to-file \
  --export-metrics-to-file-dir benchmark/soar/results/request_metrics
```

### Correctness
```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path /root/models/openbmb/MiniCPM-SALA \
  --data_path /path/to/perf_public_set.jsonl \
  --concurrency 32
```

### Speed
```bash
SHARED=benchmark/soar/data/balanced_shared_v1/speed_smax.jsonl

python3 benchmark/soar/run_soar_suite.py \
  --api-base http://127.0.0.1:30000 \
  --model-path /root/models/openbmb/MiniCPM-SALA \
  --speed-data-s1 "$SHARED" \
  --speed-data-s8 "$SHARED" \
  --speed-data-smax "$SHARED" \
  --disable-tqdm
```

### Coarse GPU Telemetry
```bash
nvidia-smi dmon -s pucvmet -d 1 -o TD -f benchmark/soar/results/nvidia_dmon.log
```

### Nsight Systems
```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=true \
  --cuda-graph-trace=node \
  --force-overwrite=true \
  --delay 20 \
  --duration 90 \
  -o benchmark/soar/results/nsys_minicpm_probe \
  python3 -m sglang.launch_server \
    --model-path /root/models/openbmb/MiniCPM-SALA \
    --host 0.0.0.0 \
    --port 30000 \
    --trust-remote-code \
    --disable-radix-cache \
    --attention-backend minicpm_flashinfer \
    --chunked-prefill-size 32768 \
    --max-prefill-tokens 32768 \
    --prefill-max-requests 1 \
    --max-running-requests 20 \
    --mem-fraction-static 0.84 \
    --schedule-conservativeness 1.0 \
    --skip-server-warmup
```

### Optional CPU Profiling
```bash
perf record -F 99 -g -p <SERVER_PID> -- sleep 60
perf report
```

## 6) Results Summary
| Item | Baseline | Profiled Finding | Next Action |
|---|---|---|---|
| Correctness / overall_accuracy | pending | pending | keep > 97% coefficient threshold |
| S1 benchmark_duration (s) | pending | pending | pending |
| S8 benchmark_duration (s) | pending | pending | pending |
| S∞ benchmark_duration (s) | pending | pending | pending |
| GPU timeline pattern | pending | pending | choose host-bound vs kernel-bound path |
| CPU hot stack | pending | pending | decide whether `perf` follow-up is needed |

## 7) Risk Assessment
- Accuracy risk: none from this documentation-only feature.
- Stability risk: low; the methodology uses existing repo scripts and optional external profilers only.
- Interpretation risk: medium; misleading conclusions are possible if traces are collected on smoke-test datasets or while startup noise is still present.
- Reproducibility risk: low if the same shared dataset, launch flags, and trace windows are reused between runs.

## 8) Rollback Instructions
1. Delete `docs/soar_2026_changes/CHANGE_0034_minicpm_profiling_methodology.en.md`.
2. Delete `docs/soar_2026_changes/CHANGE_0034_minicpm_profiling_methodology.zh.md`.

## 9) Next-Step Suggestions
- Collect one baseline Nsight Systems trace on the current branch before making another MiniCPM runtime patch.
- If the GPU timeline shows large idle gaps, prioritize host scheduling or replay setup instead of kernel changes.
- If one or two kernels dominate with dense GPU occupancy, move to kernel-level analysis and only then consider new code edits.