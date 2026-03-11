# CHANGE_0031: First Submission Package for Quantized MiniCPM-SALA

## Background and Motivation
After validating a working MiniCPM-SALA 4-bit GPTQ path on fcloud, the next priority is preparing a first submission package that reproduces the verified preprocessing and serving behavior under the official SOAR execution model.

The official toolkit requires a `prepare_env.sh` plus optional `prepare_model.sh` workflow, and explicitly disallows direct submission of quantized model weights. Therefore, the first submission must package the quantization toolchain and run GPTQ on-platform.

## Rule-Compliance Statement (SOAR)
This iteration is submission-preparation work only.

- it does not submit quantized model weights directly
- it keeps GPTQ as an on-platform preprocessing step through `prepare_model.sh --input --output`
- it uses `uv pip install` as required by the latest toolkit
- it aligns with the official technical path of GPTQ W4A16 plus Marlin
- it does not modify fixed competition concurrency behavior or re-enable forbidden prefix cache behavior

## Detailed Implementation Plan
Planned and implemented in this iteration:

1. Align `prepare_env.sh` with the exact validated quantized runtime path.
2. Install the extra dependencies using the same commands that resolve successfully in the clean fcloud submission simulation.
3. Use an organizer-provided precompiled `flash-attn` wheel placed beside `prepare_env.sh` so submission setup avoids long build time and build-time OOM risk.
4. Export the validated SGLang runtime arguments for GPTQ Marlin serving.
5. Lower the default GPTQ calibration sample count from 128 to 32 to match the validated fcloud memory limit.
6. Default the calibration file path to `./perf_public_set.jsonl` so the submission tarball can carry its own calibration JSONL.
7. Make `gptq` the default submission preprocess mode so organizer-style execution does not silently fall back to `copy`.
8. Document the expected tarball contents and packaging method for the first submission.

## Actual Code Changes
Updated file:

- `benchmark/soar/demo_sala/prepare_env.sh`

Changes:

1. Install editable vendored SGLang:
- `uv pip install --no-deps -e ./sglang/python`

2. Install GPTQ dependencies using the validated submission-simulation commands:
- `gptqmodel`

3. Install `flash-attn` from local precompiled wheel:
- `flash_attn-2.8.3+cu128sm120-cp310-cp310-linux_x86_64.whl`

4. Fail fast if the local wheel is missing from the submission directory.

5. Export validated GPTQ preprocessing defaults:
- `SOAR_QUANT_MODE` defaults to `gptq`
- calibration file defaults to `./perf_public_set.jsonl`
- calibration samples default to `32`
- calibration batch size default to `1`
- bits default to `4`
- group size default to `128`
- attention impl defaults to `flash_attention_2`

6. Export validated serving args for quantized runtime:
- `--trust-remote-code`
- `--disable-radix-cache`
- `--attention-backend minicpm_flashinfer`
- `--chunked-prefill-size 32768`
- `--max-prefill-tokens 32768`
- `--prefill-max-requests 1`
- `--max-running-requests 20`
- `--mem-fraction-static 0.84`
- `--schedule-conservativeness 1.0`
- `--skip-server-warmup`
- `--dense-as-sparse`
- `--quantization gptq_marlin`

Updated file:

- `benchmark/soar/demo_sala/preprocess_model.py`

Changes:

7. Default calibration file fallback changed from empty to:
- `Path(__file__).resolve().parent / "perf_public_set.jsonl"`

8. Default calibration samples changed from `128` to `32`

Updated file:

- `benchmark/soar/demo_sala/README.md`

Changes:

9. Documented the local flash-attn wheel requirement, runtime args, calibration-file expectation, safe defaults, and tarball packaging method.

## Why These Choices Were Used
The submission-prep package follows the exact validated fcloud path because it is already known to produce a working GPTQ artifact and a successful SGLang launch.

Important practical findings captured here:

- GPTQ calibration samples above `32` can OOM on the validated fcloud setup
- the successful runtime path used `gptq_marlin`, not plain `gptq`
- the calibration JSONL should travel with the submission package so preprocessing remains self-contained
- organizer-style execution can silently run in `copy` mode unless `SOAR_QUANT_MODE` is explicitly exported
- an organizer-provided precompiled `flash-attn` wheel removes the need for slow and OOM-prone source builds during submission setup

## Risk to Accuracy/Stability
Advantages:

- uses the dependency install path that actually resolves in the clean fcloud simulation
- aligns the package with the exact working runtime flags
- makes on-platform GPTQ preprocessing more reproducible

Remaining risks:

- the official evaluation environment may still differ slightly from fcloud in memory behavior
- the local wheel filename and placement must match the submission script expectation exactly
- correctness and speed should be rechecked once on the final packaged submission flow

## Validation Commands
1. Shell syntax checks
```bash
bash -n benchmark/soar/demo_sala/prepare_env.sh
bash -n benchmark/soar/demo_sala/prepare_model.sh
```

2. Python syntax checks
```bash
python3 -m py_compile benchmark/soar/demo_sala/preprocess_model.py
python3 -m py_compile benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py
```

3. Submission-style preprocess smoke
```bash
cd benchmark/soar/demo_sala
SOAR_QUANT_MODE=gptq \
bash prepare_model.sh --input /path/to/raw_model --output /path/to/quant_output
```

4. Quantized runtime launch
```bash
python3 -m sglang.launch_server \
  --model-path /path/to/quant_output \
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
  --dense-as-sparse \
  --quantization gptq_marlin
```

5. Packaging command on fcloud
```bash
tar --exclude='__pycache__' --exclude='*.pyc' -czf minicpm_sala_submit_v1.tar.gz .
```

## Result Summary Table
Status: submission package aligned with validated quantized workflow and adjusted to the dependency resolution behavior observed in the clean fcloud simulation; final packaged end-to-end check pending.

| Item | Before CHANGE_0031 | After CHANGE_0031 |
|---|---:|---:|
| GPTQ dependency install path | unresolved | validated |
| FlashAttention installation mode | source build | local precompiled wheel |
| Default GPTQ calibration samples | 128 | 32 |
| Default packaged calibration JSONL path | No | Yes |
| Quantized runtime mode | ambiguous | `gptq_marlin` |
| Submission tarball guidance | partial | explicit |

## Rollback Instructions
1. Revert CHANGE_0031.
2. Restore `prepare_env.sh` to the simpler editable-install-only version.
3. Restore `preprocess_model.py` GPTQ defaults to the prior values.
4. If needed, fall back to `SOAR_QUANT_MODE=copy` for a non-quantized submission path.

## Next-Step Suggestions
1. Build the submission directory on fcloud with `perf_public_set.jsonl` beside the scripts.
2. Run one final submission-style preprocess plus serve smoke.
3. Package only the submission directory contents into `minicpm_sala_submit_v1.tar.gz`.