# CHANGE_0063 GPTQ Preserve K-Projection Precision

## Background and Motivation

The recent calibration-scope and calibration-sample experiments did not produce a stable recovery to the target correctness band. The weakest and most unstable categories remain concentrated in `qa`, `mcq`, and `cwe`, which suggests the main remaining problem is not only calibration coverage but also quantization sensitivity in attention routing.

Among the attention projections, `k_proj` is the most plausible low-risk rollback target because key precision has a disproportionate effect on attention score quality. Small K-side errors perturb token routing through the softmax and can amplify downstream mistakes, especially on long-context question-answering and extraction tasks.

This iteration therefore keeps the feature scope narrow: preserve `self_attn.k_proj` in higher precision by excluding it from GPTQ quantization, while leaving the rest of the current GPTQ W4A16 + Marlin path unchanged.

## Rule-Compliance Statement

This change remains compliant with the latest SOAR competition and toolkit pages checked on 2026-03-31.

- It only changes preprocessing quantization scope.
- It preserves the official `prepare_env.sh` and `prepare_model.sh --input/--output` workflow.
- It does not replace the MiniCPM-SALA base model.
- It does not alter concurrency rules, prefix-cache behavior, or the serving contract.
- It keeps the current GPTQ + Marlin + FP8 KV cache runtime path for the rest of the quantized model.

## Detailed Implementation Plan

Before change:

1. Keep the existing GPTQ flow and layer-aware selector.
2. Exclude `self_attn.k_proj` from the default GPTQ quantization scope.
3. Keep the same exclusion in the module-mismatch retry path so the feature remains consistent.
4. Avoid broader rollback such as preserving both Q and K in the same iteration.

## Actual Code Changes

Changed files:

- `benchmark/soar/demo_sala/prepare_env.sh`
- `benchmark/soar/demo_sala/preprocess_model.py`

What changed:

1. Updated the default `SOAR_GPTQ_INCLUDE_MODULES` list to remove `self_attn.k_proj`.
2. Updated the default `SOAR_GPTQ_EXCLUDE_MODULES` list to include `self_attn.k_proj`.
3. Updated the internal `preprocess_model.py` defaults so direct preprocess usage matches the same policy.
4. Updated the GPTQ retry fallback path so it also keeps `self_attn.k_proj` excluded.

## Design Notes

### Why preserve K first instead of Q or V

K-side precision affects attention routing more directly than V-side precision. If K is too noisy, the softmax can shift probability mass toward the wrong tokens, and that error propagates through the rest of the layer. V-side errors are typically less explosive because they affect the weighted sum after routing has already been decided.

### Why not preserve both Q and K immediately

That would change two important variables at once and increase the speed cost. Preserving only `k_proj` is the narrower experiment and gives a cleaner signal about whether K precision is the main missing ingredient.

### Why update the retry path too

If the retry fallback reintroduced `k_proj` into GPTQ quantization, the feature would silently disappear whenever the mismatch retry is triggered. Keeping the same exclusion in both the main and retry paths makes the experiment coherent.

## Validation Commands

Shell syntax:

```bash
bash -n benchmark/soar/demo_sala/prepare_env.sh
```

Python syntax:

```bash
python3 -m py_compile benchmark/soar/demo_sala/preprocess_model.py
```

Preprocess validation:

```bash
bash benchmark/soar/demo_sala/prepare_model.sh --input <RAW_MODEL_DIR> --output <OUTPUT_MODEL_DIR>
```

Correctness validation:

```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_DIR> \
  --data_path <DATA_DIR>/perf_public_set.jsonl \
  --concurrency 32
```

Serving benchmark:

```bash
bash SOAR/bench_serving.sh http://127.0.0.1:30000
```

Rollback to the previous quantization scope:

```bash
export SOAR_GPTQ_INCLUDE_MODULES=self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj
export SOAR_GPTQ_EXCLUDE_MODULES=self_attn.o_gate,self_attn.z_proj
```

## Result Summary Table

| Item | Before | After |
| --- | --- | --- |
| Default GPTQ attention projection scope | Q + K + V + O | Q + V + O |
| `k_proj` handling | quantized | preserved in higher precision |
| Retry-path behavior | could re-quantize K | preserves K rollback consistently |
| Speed expectation | current baseline | slight regression possible |
| Accuracy expectation | unstable `qa/mcq/cwe` | target improved routing stability |

## Rollback Instructions

If preserving `k_proj` does not improve correctness enough, or the speed cost is too high:

1. restore `self_attn.k_proj` to the GPTQ include list
2. remove it from the GPTQ exclude list
3. rerun the same preprocess, correctness, and serving checks

## Next-Step Suggestions

1. Measure whether `qa` and `cwe` improve first, because they are the most K-sensitive-looking buckets.
2. If this helps but is not enough, the next accuracy feature should consider preserving both Q and K as a separate experiment.
3. If it does not help, move away from calibration and projection rollback toward a different accuracy hypothesis rather than continuing to increase calibration samples.