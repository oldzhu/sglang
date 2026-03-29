# CHANGE_0053 Prepare-Model RoPE Scaling Sanitization

## Background and Motivation

An official submission run failed during `prepare_model.sh` before quantization completed. The failure occurred while GPTQModel was constructing the raw MiniCPM-SALA shell model from the input checkpoint and remote Hugging Face modeling code.

The key error was:

```text
ValueError: Unknown RoPE scaling type default
```

This indicates an environment-sensitive compatibility issue between the raw input `config.json` and the MiniCPM-SALA remote modeling code used during GPTQ preprocessing. The previous preprocess flow passed the raw model path directly into `GPTQModel.load(...)` without normalizing incompatible config fields first.

This iteration adds one focused submission-compatibility feature: sanitize unsupported RoPE-scaling metadata in a temporary GPTQ load source before shell-model construction, add targeted rope-debug traces for official-platform investigation, and patch GPTQModel's in-memory MiniCPM-SALA config normalization when that stack synthesizes unsupported default RoPE metadata after config-file parsing.

## Rule-Compliance Statement

This change remains compliant with the latest SOAR toolkit submission workflow.

- It preserves the `prepare_env.sh` + `prepare_model.sh --input/--output` contract.
- It does not replace the MiniCPM-SALA base model.
- It does not alter runtime concurrency, prefix-cache behavior, or evaluation logic.
- It only normalizes preprocessing input config for compatibility during official on-platform quantization.

## Detailed Implementation Plan

Before change:

1. Inspect the preprocess flow and confirm that GPTQ loading uses the raw source directory directly.
2. Add a temporary model-view builder used only during GPTQ loading.
3. Sanitize `config.json` in that temporary view when unsupported default-style RoPE markers are present.
4. Add concise traces that print raw and sanitized RoPE fields before `GPTQModel.load(...)`.
5. Instrument the in-memory GPTQModel config object immediately before shell-model construction.
6. Leave the original input directory untouched.

## Actual Code Changes

Changed files:

- `benchmark/soar/demo_sala/prepare_env.sh`
- `benchmark/soar/demo_sala/preprocess_model.py`

What changed:

1. Added a temporary GPTQ load-source preparation step.
2. Added config sanitization logic that removes `rope_scaling` when:
   - `rope_scaling` is a dict
  - `rope_scaling["type"] == "default"`, or
  - `rope_scaling["rope_type"] == "default"`
3. The temporary load source uses symlinks where possible and falls back to copy when symlinks are unavailable.
4. `GPTQModel.load(...)` now reads from the sanitized temporary model view instead of the raw source path.
5. Added preprocess logging describing the sanitization action and the temporary source path.
6. Restored GPTQ save-time source metadata back to the real `--input` model path after load succeeds.
7. Deferred temporary-directory cleanup until `model.save(...)` finishes, so the compatibility layer only affects load-time config normalization and not save-time bookkeeping.
8. Added rope-debug traces that log raw vs sanitized `rope_scaling` / `rope_type` fields and print the effective load source before GPTQ shell-model construction.
9. Added env-gated workaround `SOAR_GPTQ_FORCE_NULL_ROPE_SCALING=1`, which forces `"rope_scaling": null` in the temporary GPTQ config for troubleshooting on environments that appear to synthesize an unsupported default value when the field is omitted.
10. Added package-version traces for `gptqmodel` and `transformers` so local and official environments can be compared directly from preprocess logs.
11. Added a MiniCPM-SALA-specific in-memory compatibility patch that is version-adaptive: it wraps GPTQModel's `normalize_hf_config_compat(...)` when available, otherwise falls back to wrapping `build_shell_model(...)`. In both cases it logs the in-memory config state and clears synthesized default RoPE metadata from `rope_scaling`, `rope_parameters`, and top-level `rope_type` before shell-model construction.
12. Added a second, later patch point on `transformers.modeling_utils.PreTrainedModel.from_pretrained(...)` and `._from_config(...)` so the same MiniCPM-SALA in-memory cleanup is applied immediately before the real model-construction paths that appear in GPTQ turtle-model/direct-load failures.
13. Fixed the installer control flow so GPTQ-side hooks and later `transformers`-side hooks are both installed in the same run instead of the earlier hook branch returning too early.
14. Duplicated the dependency-version output immediately before those real `transformers` load entrypoints so `gptqmodel` / `transformers` versions remain visible even when the platform only exposes the last 50 log lines.
12. Exported `SOAR_GPTQ_DEBUG_IN_MEMORY_CONFIG=1` by default in `prepare_env.sh` and printed its value so official submissions enable the in-memory diagnostics consistently unless explicitly overridden.

## Design Notes

### Why remove `rope_scaling` instead of rewriting it

The failure is caused by an explicitly unsupported `type="default"` marker rather than by clear evidence that the model requires a different scaling algorithm. Removing that unsupported marker is safer than guessing a replacement such as `linear` or `dynamic`.

Different environments may surface that unsupported marker under either `type` or `rope_type`. The sanitization now handles both schema variants conservatively by dropping the full `rope_scaling` field when either one is set to `default`.

For troubleshooting only, `SOAR_GPTQ_FORCE_NULL_ROPE_SCALING=1` can be enabled to write an explicit `null` value instead of relying on field omission. This is intended as a controlled investigation aid when the official stack appears to synthesize a default rope-scaling state after config load.

### Why add rope-debug traces

The official platform still reproduced the same load failure even after the initial sanitization logic. That means we need to separate three possibilities: the raw input config already contains a bad RoPE marker, the temporary sanitized config is not what we expect, or a downstream config-construction step recreates the unsupported value. Compact rope-debug logs before `GPTQModel.load(...)` make that distinction observable from platform logs.

### Why patch the in-memory GPTQModel config object

Later official-platform traces showed that both the raw config and the temporary sanitized config already had `rope_scaling=None`, yet GPTQModel still reached MiniCPM-SALA remote code with `scaling_type="default"`. That means the unsupported value is likely being synthesized after config-file parsing inside GPTQModel or transformers compatibility normalization. Patching the in-memory config object immediately after GPTQModel normalization is the narrowest place to neutralize that default marker without changing the raw model files.

Because GPTQModel internals differ across releases, the patch must be version-adaptive. Some environments expose `normalize_hf_config_compat(...)`, while others only expose `build_shell_model(...)` as the stable interception point.

The latest traces also showed that some failures occur later in the direct `transformers` load path (`PreTrainedModel.from_pretrained(...)`) rather than in the earlier shell-config path. For those cases, the same cleanup must be applied again at the actual model-construction boundary.

This only works if both hook families are installed together. The installer now tracks each hook independently and emits a summary of which hooks were installed.

### Why use a temporary model view

The official submission platform re-runs preprocessing from raw input every time. Mutating the original input directory in place would be less safe and harder to reason about. A temporary sanitized load source keeps the fix isolated to preprocessing.

### Why restore the canonical source path before save

GPTQModel records the load source path internally and reuses it during `model.save(...)` for model-size bookkeeping. If the temporary sanitized directory remains recorded as the canonical source, downstream save logic can mis-handle it as a Hugging Face repo identifier. Resetting that metadata to the real `--input` path keeps the temporary view load-only.

## Validation Commands

Preprocess validation:

```bash
bash benchmark/soar/demo_sala/prepare_model.sh --input <RAW_MODEL_DIR> --output <OUTPUT_MODEL_DIR>
```

Inspect the raw config field:

```bash
python3 - <<'PY'
import json
from pathlib import Path
cfg = json.loads(Path("<RAW_MODEL_DIR>/config.json").read_text())
print(json.dumps(cfg.get("rope_scaling"), ensure_ascii=False, indent=2))
PY
```

Troubleshooting run with explicit null workaround:

```bash
SOAR_GPTQ_FORCE_NULL_ROPE_SCALING=1 \
bash benchmark/soar/demo_sala/prepare_model.sh --input <RAW_MODEL_DIR> --output <OUTPUT_MODEL_DIR>
```

Troubleshooting run with in-memory config tracing enabled:

```bash
SOAR_GPTQ_DEBUG_IN_MEMORY_CONFIG=1 \
bash benchmark/soar/demo_sala/prepare_model.sh --input <RAW_MODEL_DIR> --output <OUTPUT_MODEL_DIR>
```

Correctness validation after quantization:

```bash
python3 benchmark/soar/demo_sala/eval_model_001.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <OUTPUT_MODEL_DIR> \
  --data_path benchmark/soar/demo_sala/perf_public_set.jsonl \
  --concurrency 32
```

## Result Summary Table

| Item | Before | After |
| --- | --- | --- |
| GPTQ load source | raw model directory | sanitized temporary model view |
| Unsupported `rope_scaling.type=default` | unhandled | removed during GPTQ shell-model load |
| RoPE investigation visibility | none | raw/sanitized rope-debug traces before GPTQ load |
| GPTQ in-memory config visibility | none | version logs plus version-adaptive in-memory snapshots around GPTQModel and `transformers` load entrypoints |
| GPTQ save metadata source | temporary sanitized path could leak into save bookkeeping | restored to the real raw-model path before save |
| Original input mutation | not applicable | unchanged, still untouched |
| Official prepare-model compatibility | failed | pending validation |

## Rollback Instructions

If this compatibility fix is not needed or proves incorrect:

1. remove the temporary GPTQ load-source preparation helper
2. restore direct `GPTQModel.load(str(src), ...)`
3. rerun `prepare_model.sh`

No runtime rollback is required because this feature only affects preprocessing compatibility.

## Next-Step Suggestions

1. Re-run the official submission and compare the new `gptqmodel` / `transformers` version logs against the local environment.
2. Inspect the new `[preprocess][rope-debug] in_memory_config ...` lines to determine whether the unsupported `default` value is created inside GPTQModel normalization.
3. Keep this troubleshooting instrumentation until the root cause is identified; only then decide whether the final fix should remain as a scoped in-memory patch or move to a cleaner upstream compatibility adjustment.