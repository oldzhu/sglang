# CHANGE_0013_fast_iteration_time_control

## 1) Background & Motivation
- Existing generated benchmark datasets could be too heavy for fast iteration.
- User requested easier control so each tier can finish around ~10 minutes during rapid tuning loops.

## 2) SOAR Rule-Compliance Check
- Tooling-only data generation change.
- No model/kernel/runtime code path changes.

## 3) Plan Before Code Change
- Modify one file: `benchmark/soar/generate_speed_datasets.py`.
- Add profile presets and per-tier row overrides.
- Add rough duration estimate output for quick planning.

## 4) Actual Code Change
- Added `--profile` with presets:
  - `quick10` (default, lightweight for fast iteration)
  - `balanced`
  - `heavy`
- Added per-tier row controls:
  - `--rows-s1`, `--rows-s8`, `--rows-smax`
- Added rough time estimation:
  - `--estimate-input-tps` (default `8000`)
  - `--estimate-output-tps` (default `250`)
  - prints estimated minutes per tier and total.

## 5) Usage Examples
```bash
# Fast iteration defaults
python3 benchmark/soar/generate_speed_datasets.py --profile quick10 --output-dir /root/soar_fast_data

# Explicitly target even faster loop by lowering rows
python3 benchmark/soar/generate_speed_datasets.py \
  --profile quick10 \
  --rows-s1 12 --rows-s8 16 --rows-smax 20 \
  --output-dir /root/soar_fast_data

# Heavier run
python3 benchmark/soar/generate_speed_datasets.py --profile balanced --output-dir /root/soar_balanced_data
```

## 6) Risks
- Low. Duration estimate is approximate and depends on actual runtime throughput.

## 7) Rollback
1. Revert `benchmark/soar/generate_speed_datasets.py`.

## 8) Next-Step Suggestions
- If needed, add auto-calibration from a short pilot benchmark in a future change.
