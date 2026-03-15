# CHANGE_0037_minicpm_fp8_sm120_reference

## 1) Background & Motivation
- Problem statement: MiniCPM FlashInfer currently fails when `--kv-cache-dtype` is set to FP8, first with `fp8 tensor core is not supported in fa2 backend`, and then, after ad hoc backend patching, with a generated all-FP8 kernel build failure.
- Why this matters: the SOAR toolkit `技术路径指引` explicitly lists FP8 KV cache as an encouraged optimization direction for MiniCPM-SALA, especially for decode-heavy, long-context settings.
- Additional reference used in this iteration: a user-provided text extraction from an NVIDIA RTX PRO / SM120 optimization PDF. The extracted notes emphasize that SM120 provides native FP8 tensor-core capability, Hopper-like optimization characteristics, TMA support, and strong memory bandwidth, which supports the hypothesis that the current failure is a software-path mismatch rather than a hardware impossibility.
- Objective of this iteration: document the revised diagnosis and implement a narrow MiniCPM FlashInfer contract fix so FP8 KV-cache experiments can proceed on a better-aligned path.

## 2) SOAR Rule-Compliance Check
- Latest official references reviewed before this change: `https://soar.openbmb.cn/competition` and `https://soar.openbmb.cn/toolkit`.
- Allowed by rules because: the competition allows inference optimization, KV/memory optimization, kernel/backend tuning, and quantization-based acceleration; the toolkit `技术路径指引` explicitly mentions `FP8 KV Cache` as a supported optimization direction.
- Constraints respected: this change does not alter model weights, does not re-enable prefix cache, does not modify official concurrency logic, and keeps the optimization inside the runtime/backend implementation.
- Accuracy/stability risk: medium. FP8 KV cache still requires correctness verification because cache quantization can reduce answer quality if scaling or backend selection is wrong.

## 3) Hardware Reference Summary
The user-provided PDF extraction supports four practical conclusions:

1. SM120 is FP8-capable hardware, not an FP8-prohibited target.
2. Its optimization style is close to Hopper, so mature tensor-core and memory-path optimizations should be applicable in principle.
3. FP8 throughput is materially higher than BF16/FP16 throughput, so a decode-side KV-bandwidth win is plausible if the software stack uses the right kernels.
4. Nsight Systems and Nsight Compute are the correct tools for follow-up validation once the path is runnable.

## 4) Re-Thought Root Cause
The new working diagnosis is:

1. `fp8 tensor core is not supported in fa2 backend` is a backend-path limitation, not proof that SM120 cannot support FP8 KV cache.
2. The follow-up JIT failure after manually swapping `fa2 -> fa3` suggests the MiniCPM path was still generating the wrong dtype contract, likely an all-FP8 attention kernel rather than `BF16 query/output + FP8 KV cache`.
3. MiniCPM’s custom FlashInfer integration diverged from the generic SGLang FlashInfer backend in several important places.

Before this patch, the MiniCPM path had these likely blockers:

1. Prefill wrapper was hardcoded to `backend="fa2"`.
2. `q_data_type` was set equal to KV-cache dtype instead of model dtype.
3. Query-side tensors were explicitly cast to KV-cache dtype in MiniCPM backend logic.
4. FlashInfer wrapper `forward(...)` calls did not pass `k_scale` / `v_scale` through, unlike the generic backend.

## 5) Detailed Implementation Plan Before Change
This iteration was scoped to one enabling feature only: align MiniCPM FlashInfer’s KV-cache FP8 contract with the generic backend.

Planned code changes:

1. In `python/sglang/srt/layers/attention/minicpm_attention_kernels.py`
   - keep `q_data_type` in model dtype
   - add a configurable prefill backend selection with environment override
   - pass `k_scale` and `v_scale` into FlashInfer wrapper forward calls
2. In `python/sglang/srt/layers/attention/minicpm_backend.py`
   - stop casting `q`, `q_rope`, and `k_rope` to KV-cache dtype when FP8 KV cache is enabled
   - keep descale/scaling metadata preparation intact

Expected gain:

1. Eliminate the incorrect all-FP8 contract at the MiniCPM integration layer.
2. Make `auto` or a user-selected prefill backend testable without patching site-packages.
3. Improve the odds that FlashInfer will select or JIT the intended kernel family for `BF16 Q/O + FP8 KV`.

## 6) Actual Code Changes
Changed files:

1. `python/sglang/srt/layers/attention/minicpm_attention_kernels.py`
2. `python/sglang/srt/layers/attention/minicpm_backend.py`

Actual edits:

1. MiniCPM FlashInfer `q_data_type` now uses `model_runner.dtype` instead of `model_runner.kv_cache_dtype`.
2. MiniCPM prefill wrapper backend is no longer hardcoded to `fa2`; it now reads `SGLANG_MINICPM_FLASHINFER_PREFILL_BACKEND` and defaults to `auto`.
3. MiniCPM FlashInfer `wrapper.forward(...)` now passes `k_scale=layer.k_scale_float` and `v_scale=layer.v_scale_float` for both prefill and decode.
4. MiniCPM backend no longer casts query-side tensors to KV-cache dtype during FP8 KV-cache preparation.

## 7) Validation Commands
Syntax / import validation:

```bash
python3 -m compileall \
  python/sglang/srt/layers/attention/minicpm_attention_kernels.py \
  python/sglang/srt/layers/attention/minicpm_backend.py
```

Recommended first runtime smoke tests on fcloud:

```bash
export SGLANG_MINICPM_FLASHINFER_PREFILL_BACKEND=auto
python3 -m sglang.launch_server \
  --model-path <MODEL_PATH> \
  --attention-backend minicpm_flashinfer \
  --kv-cache-dtype fp8_e5m2 \
  --disable-cuda-graph
```

```bash
export SGLANG_MINICPM_FLASHINFER_PREFILL_BACKEND=fa3
python3 -m sglang.launch_server \
  --model-path <MODEL_PATH> \
  --attention-backend minicpm_flashinfer \
  --kv-cache-dtype fp8_e5m2 \
  --disable-cuda-graph
```

Correctness check after successful launch:

```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_PATH> \
  --data_path benchmark/soar/demo_sala/perf_public_set.jsonl \
  --concurrency 32
```

Proxy speed check after successful launch:

```bash
python3 benchmark/soar/run_soar_suite.py \
  --base-url http://127.0.0.1:30000 \
  --dataset-profile heavy
```

## 8) Result Summary Table
| Area | Baseline | New | Delta | Notes |
|---|---:|---:|---:|---|
| MiniCPM prefill backend selection | hardcoded `fa2` | env-configurable, default `auto` | n/a | enables `auto` / `fa3` testing |
| Query dtype contract | tied to KV dtype | tied to model dtype | n/a | avoids all-FP8 Q path |
| KV scale plumbing | missing in wrapper forward | passed in prefill/decode | n/a | aligns with generic backend |
| Local syntax validation | pending before patch | pending runtime | n/a | compile step required |
| fcloud FP8 launch | failing before patch | pending user run | pending | validate with `--disable-cuda-graph` first |

## 9) Rollback Instructions
If this iteration proves unstable:

1. Revert `python/sglang/srt/layers/attention/minicpm_attention_kernels.py` to restore the previous MiniCPM FlashInfer wrapper behavior.
2. Revert `python/sglang/srt/layers/attention/minicpm_backend.py` to restore the previous query-side casting behavior.
3. Unset `SGLANG_MINICPM_FLASHINFER_PREFILL_BACKEND` in the environment.

## 10) Next-Step Suggestions
1. Run the first smoke test with `--disable-cuda-graph` and `SGLANG_MINICPM_FLASHINFER_PREFILL_BACKEND=auto`.
2. If `auto` still lands on a bad backend, rerun with `fa3` to isolate backend selection from dtype-contract issues.
3. If the runtime still JIT-builds kernels for the wrong architecture or unsupported dtype tuple, the next iteration should target FlashInfer version/build configuration rather than MiniCPM logic.
4. After successful launch, use Nsight Systems first and Nsight Compute second to confirm whether decode-side bandwidth is the dominant bottleneck and whether FP8 KV cache is reducing it.