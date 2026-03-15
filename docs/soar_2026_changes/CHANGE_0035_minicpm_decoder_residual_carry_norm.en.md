# CHANGE_0035_minicpm_decoder_residual_carry_norm

## 1) Background & Motivation
- Problem statement: the MiniCPM decoder currently materializes explicit residual additions before each RMSNorm stage, even though the runtime already has fused add+RMSNorm kernels that can carry the pending residual state across layers.
- Why this should improve speed: carrying the scaled branch output as `hidden_states` and the accumulated skip path as `residual` removes two standalone residual-add materializations per decoder layer and lets both layernorm entry points use the existing fused kernel path.
- Target stage(s): decode / layer execution / pointwise fusion

## 2) SOAR Rule-Compliance Check
- Latest official references reviewed before this change: `https://soar.openbmb.cn/competition` and `https://soar.openbmb.cn/toolkit`.
- Allowed by rules because: the competition page explicitly allows inference-path optimizations such as operator fusion and decode-path tuning, and the toolkit `技术路径指引` encourages runtime optimization on top of the official MiniCPM-SALA base model.
- Not violating constraints: no base-model replacement, no prefix cache, no concurrency change, no submission-interface change, and no non-reproducible behavior is introduced.
- Expected impact on correctness coefficient C: neutral in intent; the patch preserves the original `scale_depth / sqrt(num_hidden_layers)` residual algebra and only changes when the addition is materialized.

## 3) Plan Before Code Change
- Files/functions to modify: `python/sglang/srt/models/minicpm.py`, specifically `MiniCPMDecoderLayer.forward()` and `MiniCPMModel.forward()`.
- Minimal diff strategy: reuse the existing `RMSNorm(x, residual)` fused path instead of adding new kernels; keep attention, MLP, and public APIs unchanged.
- Deferred scope: leave the q/k RoPE cast path unchanged in this feature so any performance change can be attributed to decoder residual fusion rather than mixed causes.
- Rollback plan: restore the previous eager residual additions and remove residual carry across layers.

## 4) Actual Code Change (After Approval)
- Commit/patch summary: `MiniCPMDecoderLayer` now keeps the accumulated skip connection in `residual`, feeds scaled branch outputs into fused `RMSNorm(..., residual)`, and returns the pending final residual-add pair to the next layer instead of materializing the sum immediately.
- Final modified files: `python/sglang/srt/models/minicpm.py`.
- Key logic differences:
  - The first decoder norm now uses the same residual-carry contract as other fused decoder implementations when a prior residual is available.
  - The post-attention norm fuses `residual + attn_output * residual_scale` directly through `RMSNorm`.
  - The model final norm consumes the last pending `(hidden_states, residual)` pair so the final output remains mathematically identical to the original eager-add path.

## 5) Validation Commands
### Correctness
```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_PATH> \
  --data_path benchmark/soar/demo_sala/perf_public_set.jsonl \
  --concurrency 32
```

### Speed
```bash
export SPEED_DATA_S1=<shared_representative_speed_set.jsonl>
export SPEED_DATA_S8=<shared_representative_speed_set.jsonl>
export SPEED_DATA_SMAX=<shared_representative_speed_set.jsonl>
python3 benchmark/soar/run_soar_suite.py \
  --base-url http://127.0.0.1:30000 \
  --dataset-profile heavy
```

### Local Syntax Gate
```bash
python3 -m compileall python/sglang/srt/models/minicpm.py
```

## 6) Results Summary
| Metric | Baseline | New | Delta |
|---|---:|---:|---:|
| Accuracy / overall_accuracy | pending | pending | pending |
| Shared-data S1 throughput | pending | pending | pending |
| Shared-data S8 throughput | pending | pending | pending |
| Shared-data S∞ throughput | pending | pending | pending |

## 7) Risk Assessment
- Accuracy risk: low-to-medium; the intended algebra is unchanged, but the residual is now carried across layers and consumed by fused norms, so functional verification is still required.
- Stability risk: low; the patch reuses existing RMSNorm fused behavior already exercised by other decoder models.
- Reproducibility risk: low; no randomization or environment-sensitive heuristics were added.

## 8) Rollback Instructions
1. Restore eager residual addition in `MiniCPMDecoderLayer.forward()` before each norm/branch transition.
2. Stop returning residual carry from decoder layers and restore `MiniCPMModel.forward()` to finalize with `self.norm(hidden_states)` only.

## 9) Next-Step Suggestions
- Run the shared-dataset S1/S8/Smax suite to measure whether the reduced pointwise traffic translates into a visible decode win.
- If the gain is still small, profile whether MiniCPM’s remaining overhead is dominated by attention-side dtype conversions or backend planning rather than decoder residual handling.