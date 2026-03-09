# CHANGE_0030_006: MiniCPM-SALA Custom GPTQModel Definition

## Background and Motivation
The previous CHANGE_0030 iterations proved that GPTQModel 5.7.0's official `QuantizeConfig.dynamic` path can exclude modules like `self_attn.o_gate`, but the new SGLang failure showed that exclusion alone is not sufficient for MiniCPM-SALA.

New checkpoint evidence from fcloud:

- `model.layers.1.self_attn.z_proj.weight` exists as a native tensor
- `model.layers.1.self_attn.o_proj.qweight` exists as a quantized tensor
- the same lightning layer therefore mixes native and quantized attention submodules correctly in principle
- but GPTQModel's generic AutoCompat path still does not provide a stable architecture description for the whole mixed-layer model

This indicates the root issue is not simply module exclusion. The real problem is that GPTQModel AutoCompat derives a module tree from a representative layer, while MiniCPM-SALA contains heterogeneous decoder layer families.

The week-1 SOAR champion note also strengthens the priority of this path: 4-bit quantization reduced model size from about 18 GB to 5.4 GB and improved single-concurrency latency by about 40%, showing that a correct 4-bit path is worth substantial engineering effort.

## Rule-Compliance Statement (SOAR)
This change remains compliant because it:

- only changes offline preprocessing behavior
- does not alter evaluation-time concurrency or hidden runtime behavior
- preserves the official `prepare_model.sh --input/--output` contract
- keeps correctness validation mandatory before treating the optimization as usable

## Detailed Implementation Plan
Planned and implemented in this iteration:

1. Add a local runtime GPTQModel definition for `model_type = "minicpm_sala"`
2. Register that definition into GPTQModel's model registry before `GPTQModel.load(...)`
3. Replace reliance on layer-0 AutoCompat with an explicit MiniCPM-SALA module tree
4. Keep optional mixed-family modules in the tree with `:!` so structure is visible to GPTQ, while those modules remain native weights
5. Keep `self_attn.o_gate` and `self_attn.z_proj` excluded as defensive dynamic rules during preprocessing

## Actual Code Changes
Added file:

- `benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py`

Key implementation details:

1. Added `MiniCPMSALAGPTQ(BaseQModel)`
- `require_trust_remote_code = True`
- `layer_modules_strict = False`
- `pre_lm_head_norm_module = "model.norm"`
- explicit `module_tree` for MiniCPM-SALA

2. Encoded mixed-layer helper modules as non-quantized structural nodes
- `self_attn.q_norm`
- `self_attn.k_norm`
- `self_attn.o_norm`
- `self_attn.z_proj`
- `self_attn.o_gate`

3. Kept quantized attention/MLP targets explicit
- `self_attn.q_proj`
- `self_attn.k_proj`
- `self_attn.v_proj`
- `self_attn.o_proj`
- `mlp.gate_proj`
- `mlp.up_proj`
- `mlp.down_proj`

4. Added runtime registration helper
- `register_minicpm_sala_gptq_model()` patches GPTQModel's `MODEL_MAP`
- it also updates `SUPPORTED_MODELS` when needed

Updated file:

- `benchmark/soar/demo_sala/preprocess_model.py`

5. Registered the custom MiniCPM-SALA GPTQ definition before `GPTQModel.load(...)`
6. Expanded defensive exclude defaults from only `self_attn.o_gate` to:
- `self_attn.o_gate`
- `self_attn.z_proj`

## Risk to Accuracy/Stability
Advantages:

- fixes the root abstraction mismatch instead of adding another exclusion workaround
- keeps native lightning gate weights and sparse gate weights out of the quantized set
- removes dependence on layer-0 AutoCompat for an architecture that is not homogeneous by layer

Remaining risks:

- raw HF MiniCPM-SALA module names on fcloud may still differ slightly from the local SGLang-aligned expectation
- 4-bit correctness may still fall below an acceptable SOAR threshold even if checkpoint loading is fixed

## Validation Commands
1. Local syntax checks
```bash
python3 -m py_compile benchmark/soar/demo_sala/preprocess_model.py
python3 -m py_compile benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py
```

2. fcloud smoke quantization
```bash
cd benchmark/soar/demo_sala
SOAR_QUANT_MODE=gptq \
SOAR_TRUST_REMOTE_CODE=true \
SOAR_GPTQ_ATTN_IMPL=flash_attention_2 \
SOAR_GPTQ_CALIBRATION_FILE=/path/to/calib.jsonl \
SOAR_GPTQ_CALIBRATION_SAMPLES=16 \
SOAR_GPTQ_BATCH_SIZE=1 \
./prepare_model.sh --input /path/to/raw_model --output /path/to/quant_smoke 2>&1 | tee /path/to/quant_smoke.log
```

3. Checkpoint structure verification
```bash
python3 - <<'PY'
import json
from pathlib import Path

index_path = Path('/path/to/quant_smoke/model.safetensors.index.json')
payload = json.loads(index_path.read_text())
keys = payload['weight_map']
for name in [
    'model.layers.1.self_attn.z_proj.weight',
    'model.layers.1.self_attn.o_proj.qweight',
    'model.layers.0.self_attn.o_gate.weight',
]:
    print(name, name in keys)
PY
```

4. SGLang load smoke
```bash
python3 -m sglang.launch_server \
  --model-path /path/to/quant_smoke \
  --trust-remote-code \
  --quantization gptq_marlin
```

## Result Summary Table
Status: implemented locally; fcloud validation pending.

| Metric | Before CHANGE_0030_006 | After CHANGE_0030_006 |
|---|---:|---:|
| GPTQ definition source | AutoCompat from layer 0 | Explicit `minicpm_sala` definition |
| `o_gate` handling | dynamic exclusion only | native structural node + dynamic defense |
| `z_proj` handling | implicit / unstable | native structural node + dynamic defense |
| Quantized checkpoint load in SGLang | fails on `z_proj` path | TBD |
| Correctness score | TBD | TBD |
| Speed S1 | TBD | TBD |
| Speed S8 | TBD | TBD |
| Speed Smax | TBD | TBD |

## Rollback Instructions
1. Revert the CHANGE_0030_006 commit.
2. Remove the registration call from `preprocess_model.py`.
3. Delete `benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py`.
4. Fall back to `SOAR_QUANT_MODE=copy` or the previous dynamic-only path if needed.

## Next-Step Suggestions
1. Run the 16-sample fcloud smoke and capture the new `simple_layer_modules` log.
2. Inspect the checkpoint index for both a lightning layer and a sparse layer.
3. If SGLang still fails, compare the raw HF model module names against the registered tree before changing any loader code.