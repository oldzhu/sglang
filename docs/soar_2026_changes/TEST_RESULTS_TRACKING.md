# SOAR 2026 — Automated Test Results Tracking

This document records all accuracy and speed benchmark results from automated fcloud testing.
Every test run should be logged here with its configuration, commit, date, and results.

**Scoring formula**: `final_score = (S1×40% + S8×30% + Smax×30%) × C`
- C = 0 if normalized accuracy < 97%
- C = 0.8 if 97% ≤ normalized accuracy < 100%
- C = 1.0 if normalized accuracy ≥ 100%
- Baseline accuracy: ~80% → need ≥77.6% original accuracy for C=0.8

---

## Test Results Table

| Test # | Date | Commit | fcloud Instance | Config | Accuracy (orig) | Accuracy (norm) | C | mcq | cwe | fwe | niah | qa | Duration | TPS | Notes |
|--------|------|--------|-----------------|--------|-----------------|-----------------|---|-----|-----|-----|------|----|----------|-----|-------|
| 1 | 2026-04-07 | dba2815c1+0070 | 223.167.85.181 | Non-quant + FP8 KV + sparse | 53.18% | — | 0 | 66.67% | 43.67% | 92.22% | 36.67% | 26.67% | — | — | CHANGE_0070 + FP8 both hurt |
| 2 | 2026-04-07 | HEAD+0070+0071 | 223.167.85.181 | Non-quant + bf16 KV + sparse | 57.11% | — | 0 | 60% | 90.67% | 97.78% | 100% | 66.67% | — | — | CHANGE_0070 regression confirmed |
| 3 | 2026-04-08 | dba2815c1 | 223.167.85.181 | Non-quant + bf16 KV + sparse (pre-0070) | **83.02%** | — | — | 60% | 90.67% | 97.78% | 100% | 66.67% | — | — | Baseline (best non-quant sparse) |
| 4 | 2026-04-08 | HEAD−0070 | 223.167.85.181 | Non-quant + bf16 KV + sparse (bug fixes only) | **83.02%** | — | — | 60% | 90.67% | 97.78% | 100% | 66.67% | — | — | Bug fixes 6/8/3/71 safe |
| 5 | 2026-04-08 | HEAD−0070 | 223.167.85.181 | GPTQ + FP8 KV + sparse | 50.36% | — | 0 | 40% | 47.33% | 97.78% | 36.67% | 30% | — | — | GPTQ+FP8 breaks sparse badly |
| 6 | 2026-04-08 | HEAD−0070 | 223.167.85.181 | GPTQ + bf16 KV + sparse | 57.47% | — | 0 | 63.33% | 44% | 96.67% | 36.67% | 46.67% | — | — | GPTQ alone breaks sparse |
| 7 | 2026-04-08 | HEAD−0070 | 223.167.85.181 | GPTQ + FP8 KV + dense | 78.84% | — | 0.8 | 56.67% | 82% | 98.89% | 100% | 56.67% | — | — | Dense tolerates GPTQ+FP8 well |
| 8 | 2026-04-09 | 687ac4127 | 223.167.85.183 | GPTQ + FP8 KV + sparse + topk_scale=2 | 0% | 0% | 0 | — | — | — | — | — | — | — | OOM crash: page table 4.6 GiB (topk=160) |
| 8b | 2026-04-09 | 9d3ecd168 | 223.167.85.183 | GPTQ + FP8 KV + sparse (default topk=96) | **76.07%** | 95.08% | 0 | 60% | 60.33% | 96.67% | 96.67% | 66.67% | 2411s | 99.73 | Freshly prepared GPTQ model; huge improvement vs old Test 5 |
| 9 | 2026-04-09 | 79e49f39f | 223.167.85.183 | GPTQ + bf16 KV + sparse (Option D) | **79.67%** | **99.58%** | **0.8** | 63.33% | 81.67% | 100% | 96.67% | 56.67% | 3157s | 275.20 | **Best GPTQ+sparse config!** bf16 KV eliminates FP8 scoring error |
| 10 | 2026-04-09 | 430dd221c | 223.167.85.183 | GPTQ + bf16 KV + sparse + topk_scale=2 (Option C) | 0.20% | 0.25% | 0 | 0% | 1% | 0% | 0% | 0% | 9385s | 661.10 | **BROKEN**: topk_scale=2 causes garbage output (avg 41K output tokens) |

---

## Speed Benchmark Results

| Test # | Date | Commit | Config | S1 | S8 | Smax | Weighted Speed Score | Notes |
|--------|------|--------|--------|----|----|------|---------------------|-------|
| 9-spd | 2026-04-09 | 79e49f39f | GPTQ + bf16 KV + sparse (Option D) | 139.28s | 56.97s | 48.33s | — | First speed test on new fcloud |

---

## Key Configurations Reference

### Server Args (prepare_env.sh)
- **Base args**: `--trust-remote-code --disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 32768 --max-prefill-tokens 32768 --prefill-max-requests 1 --max-running-requests 20 --mem-fraction-static 0.84 --schedule-conservativeness 1.0 --dense-as-sparse --quantization gptq_marlin --enable-fused-qk-norm-rope`
- **FP8 KV**: add `--kv-cache-dtype fp8_e5m2`
- **Dense mode**: replace `--attention-backend minicpm_flashinfer` with `--force-dense-minicpm`
- **TopK scaling**: add `--sparse-topk-scale N` (default 1, effective topk = base_topk×N + local_blocks)

### Model Sparse Config (from config.json)
- `topk=64`, `block_size=64`, `window_size=2048`, `kernel_size=32`, `kernel_stride=16`
- `local_blocks = 2048/64 = 32`
- Effective sparse_topk = 64 + 32 = 96 (at scale=1)

### fcloud Instances
- **Old**: 223.167.85.181:12369 (unavailable since 2026-04-09)
- **New**: 223.167.85.183:20685 (active)

---

## Notes
- Tests 1-7 were on old fcloud instance with potentially different GPTQ model preparation
- Tests 8+ are on new fcloud instance with freshly prepared GPTQ model
- The large accuracy improvement from Test 5 (50%) to Test 8b (76%) is likely due to fresh GPTQ model preparation
- Accuracy eval uses `--concurrency 32` (changed from 8 starting from CHANGE_0073)
