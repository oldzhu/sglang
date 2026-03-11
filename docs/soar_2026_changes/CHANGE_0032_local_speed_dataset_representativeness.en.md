# CHANGE_0032 Local Speed Dataset Representativeness

## Background and Motivation

During submission analysis on `mixed_minicpm_cudagraph`, local speed tests showed the pattern:

- `Smax < S8 < S1`

while the official leaderboard run showed the reverse pattern:

- `Smax > S8 > S1`

This mismatch indicates that local speed data shape can materially affect the observed concurrency trend. A light or homogeneous local dataset can favor overlap and throughput at higher concurrency, while the official hidden speed set includes a heavy long-context and long-output tail that can amplify queueing, memory pressure, stragglers, and scheduler contention.

The purpose of this documentation-only iteration is to record the benchmarking interpretation and evaluation guidance before further CUDA graph optimization work.

## Rule-Compliance Statement

This iteration is documentation-only.

- It does not modify model behavior.
- It does not alter official evaluation logic.
- It does not enable forbidden features such as prefix cache.
- It does not change concurrency settings in official evaluation.

This is compliant with the latest SOAR competition and toolkit guidance, which require reproducible and explainable optimizations under the official environment.

## Detailed Implementation Plan

Before making any source changes for runtime optimization, record the following evaluation guidance:

1. Local `Smax < S8 < S1` does not imply the official result is incorrect.
2. The official hidden speed set is likely much heavier than the current lightweight local set.
3. For runtime optimization comparisons, use one shared representative dataset across `S1`, `S8`, and `Smax`, and vary only `--max-concurrency`.
4. Keep a second smaller quick-iteration dataset for debugging only.
5. Continue with CUDA graph optimization, but judge effect size using a heavier local dataset that better matches the official published token-length distribution.

## Actual Code Changes

No runtime or model code was changed in this iteration.

Added documentation files only:

- `docs/soar_2026_changes/CHANGE_0032_local_speed_dataset_representativeness.en.md`
- `docs/soar_2026_changes/CHANGE_0032_local_speed_dataset_representativeness.zh.md`

## Validation Commands

Docs validation only:

```bash
# Example editor/problem check
# verify the two markdown files are error-free in the workspace
```

Recommended follow-up local evaluation commands:

```bash
python3 benchmark/soar/generate_speed_datasets.py \
  --profile heavy \
  --output-dir benchmark/soar/data_heavy_shared

python3 benchmark/soar/run_soar_suite.py \
  --api-base http://127.0.0.1:30000 \
  --model-path <MODEL_PATH> \
  --speed-data-s1 benchmark/soar/data_heavy_shared/speed_s1.jsonl \
  --speed-data-s8 benchmark/soar/data_heavy_shared/speed_s1.jsonl \
  --speed-data-smax benchmark/soar/data_heavy_shared/speed_s1.jsonl
```

The command above intentionally uses the same dataset for all three tiers so that only concurrency changes between `S1`, `S8`, and `Smax`.

## Result Summary Table

| Item | Previous understanding | Updated guidance |
| --- | --- | --- |
| Interpretation of local `Smax < S8 < S1` | Possibly concerning | Can happen on light local datasets |
| Interpretation of official `Smax > S8 > S1` | Possibly inconsistent | Plausible under heavy hidden workload |
| Tier comparison method | Separate datasets may be mixed in practice | Use one shared representative dataset when comparing runtime changes |
| CUDA graph work start condition | Might require perfect local reproduction first | Safe to start now, but evaluate on heavier representative data |

## Rollback Instructions

If this documentation is later found to be misleading or incomplete:

1. Remove the `CHANGE_0032` EN/ZH files.
2. Replace them with an updated continuation or correction doc pair.
3. Do not use lightweight local benchmark conclusions as the sole basis for leaderboard-facing runtime decisions.

## Next-Step Suggestions

1. Prepare one heavier shared local speed dataset that better matches the official published input/output length distribution.
2. Use that shared dataset to compare CUDA graph changes on `mixed_minicpm_cudagraph` under `S1`, `S8`, and `Smax`.
3. Keep the existing quick local dataset only for fast regression checks.