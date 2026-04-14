# SOAR 2026 — Automated Test Results Tracking

This document records all accuracy and speed benchmark results from automated fcloud testing.
Every test run should be logged here with its configuration, commit, date, and results.

**Scoring formula**: `Final Score = Performance Score × C` (HIGHER = BETTER)
- `Performance Score = S1×40% + S8×30% + Smax×30%` where `S_N = (Duration_best / Duration_player) × 100`
- C = 0 if normalized accuracy ≤ 97% (eliminated)
- C = 0.92 if 97% < normalized accuracy ≤ 98%
- C = 0.96 if 98% < normalized accuracy ≤ 99%
- C = 1.0 if 99% < normalized accuracy ≤ 100%

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
| 7 | 2026-04-08 | HEAD−0070 | 223.167.85.181 | GPTQ + FP8 KV + dense | 78.84% | — | — | 56.67% | 82% | 98.89% | 100% | 56.67% | — | — | Dense tolerates GPTQ+FP8 well |
| 8 | 2026-04-09 | 687ac4127 | 223.167.85.183 | GPTQ + FP8 KV + sparse + topk_scale=2 | 0% | 0% | 0 | — | — | — | — | — | — | — | OOM crash: page table 4.6 GiB (topk=160) |
| 8b | 2026-04-09 | 9d3ecd168 | 223.167.85.183 | GPTQ + FP8 KV + sparse (default topk=96) | **76.07%** | 95.08% | 0 | 60% | 60.33% | 96.67% | 96.67% | 66.67% | 2411s | 99.73 | Freshly prepared GPTQ model; huge improvement vs old Test 5 |
| 9 | 2026-04-09 | 79e49f39f | 223.167.85.183 | GPTQ + bf16 KV + sparse (Option D) | **79.67%** | **99.58%** | **1.0** | 63.33% | 81.67% | 100% | 96.67% | 56.67% | 3157s | 275.20 | **Best GPTQ+sparse config!** bf16 KV eliminates FP8 scoring error |
| 10 | 2026-04-09 | 430dd221c | 223.167.85.183 | GPTQ + bf16 KV + sparse + topk_scale=2 (Option C) | 0.20% | 0.25% | 0 | 0% | 1% | 0% | 0% | 0% | 9385s | 661.10 | **BROKEN**: topk_scale=2 causes garbage output (avg 41K output tokens) |
| 11 | 2026-04-10 | 9e82efe43 | 223.167.85.181 | Non-quant + FP8 KV + sparse (retest w/o 0070 bug) | 55.82% | 69.78% | 0 | 66.67% | 44.67% | 97.78% | 36.67% | 33.33% | 7379s | — | FP8 KV severely hurts NIAH/qa on non-quant; concurrency=8, ~2h eval |
| 12 | 2026-04-12 | 9e82efe43 | 223.167.85.181 | GPTQ + FP8 KV + dense (freshly quant on old fcloud) | **79.29%** | **99.11%** | **1.0** | 63.33% | 72% | 97.78% | **100%** | 63.33% | 4244s | — | Dense mode + GPTQ + FP8; niah perfect; qa improved vs Test 9 |
| 13 | 2026-04-12 | 9e82efe43 | 223.167.85.181 | GPTQ + bf16 KV + dense (same quant model) | 76.67% | 95.83% | 0 | 50% | 80% | 100% | 96.67% | 56.67% | 4568s | — | bf16 KV dense; mcq dropped to 50%; below C=0.8 threshold |
| 14 | 2026-04-13 | c818ae261 | 223.167.85.181 | CHANGE_0075: bf16 RoPE + in-place residual | **52.64%** | **65.81%** | **0** | 53.33% | 47.67% | 98.89% | 36.67% | 26.67% | 4128s | — | **CATASTROPHIC**: bf16 RoPE destroys precision; qa=26.67%, niah=36.67% |
| 15 | 2026-04-13 | b8196b71e | 223.167.85.181 | CHANGE_0075 partial: in-place residual only (RoPE restored) | **51.91%** | **64.89%** | **0** | 46.67% | 44.0% | 98.89% | 40.0% | 30.0% | 4188s | — | **CATASTROPHIC**: in-place `*=` also destroys accuracy; WORSE than Test 14 |
| 16 | 2026-04-13 | caa93efe9 | 223.167.85.181 | Full baseline revert (no CHANGE_0075) | — | — | — | — | — | — | — | — | — | — | Running — verifying return to baseline |
| 17 | 2026-04-14 | 290e370e6 | 223.167.85.181 | CHANGE_0075 re-enabled (bf16 RoPE + in-place residual) + dense+FP8 | **79.98%** | **99.97%** | **1.0** | — | — | — | — | — | 3148s | 463.49 | **CHANGE_0075 VINDICATED**: Tests 14-16 were on wrong sparse config; on correct dense+FP8 config accuracy is excellent |

---

## Speed Benchmark Results

| Test # | Date | Commit | Config | S1 | S8 | Smax | Weighted Speed Score | Notes |
|--------|------|--------|--------|----|----|------|---------------------|-------|
| 9-spd | 2026-04-09 | 79e49f39f | GPTQ + bf16 KV + sparse (Option D) | 139.28s | 56.97s | 48.33s | — | First speed test on new fcloud |
| 3/4-spd | 2026-04-12 | 9e82efe43 | Non-quant + bf16 KV + sparse | 281.32s | 90.74s | 72.17s | — | ~2× slower than GPTQ; Smax crashed on 1st attempt, passed on retry |
| 12-spd | 2026-04-12 | 9e82efe43 | GPTQ + FP8 KV + dense | 121.71s | 44.09s | 35.86s | — | Fastest config tested! |
| 13-spd | 2026-04-12 | 9e82efe43 | GPTQ + bf16 KV + dense | 121.22s | 44.05s | 35.94s | — | Nearly identical speed to FP8 KV |
| 12-VarA | 2026-04-12 | 9e82efe43 | GPTQ+FP8+dense: chunk=65K, prefill=2, running=40, mem=0.87 | 121.68s | 44.11s | 35.91s | — | Zero improvement vs baseline |
| 12-VarB | 2026-04-12 | 9e82efe43 | GPTQ+FP8+dense: +mixed-chunk, conserv=0.7 | 121.63s | 43.70s | 35.71s | — | Marginal: S8 -1%, Smax -0.6% |
| 12-VarC | 2026-04-12 | 9e82efe43 | GPTQ+FP8+dense: +torch.compile(max-bs=32)+mixed-chunk | **113.06s** | **41.65s** | **33.86s** | — | **Best: S1 -7.1%, S8 -5.7%, Smax -5.7%**; server OOM during accuracy eval |
| 14-spd | 2026-04-13 | c818ae261 | CHANGE_0075: bf16 RoPE + in-place residual | 139.26s | 52.82s | **CRASH** | — | **SLOWER**: S1 +14.5%, S8 +19.6% vs baseline; Smax server crashed |
| 17-spd | 2026-04-14 | 290e370e6 | CHANGE_0075 re-enabled + dense+FP8 (correct config) | 122.01s | 44.14s | 35.94s | — | Essentially identical to baseline (all within ±0.3%); CHANGE_0075 does NOT hurt speed |

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
- **Old**: 223.167.85.181:12369 (restored 2026-04-10, non-quant model only)
- **New**: 223.167.85.183:20685 (active, has GPTQ model)

---

## Notes
- Tests 1-7 were on old fcloud instance with potentially different GPTQ model preparation
- Tests 8+ are on new fcloud instance with freshly prepared GPTQ model
- The large accuracy improvement from Test 5 (50%) to Test 8b (76%) is likely due to fresh GPTQ model preparation
- Accuracy eval uses `--concurrency 32` (changed from 8 starting from CHANGE_0073)
