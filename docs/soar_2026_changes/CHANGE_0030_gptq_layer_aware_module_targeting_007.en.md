# CHANGE_0030_007: TorchAO Environment Drift on fcloud and Resolution

## Background and Motivation
After CHANGE_0030_006, the quantized MiniCPM-SALA artifact progressed further in SGLang startup, but fcloud serving still failed with a TorchAO import error:

```text
ImportError: cannot import name 'float8_dynamic_activation_float8_weight' from 'torchao.quantization'
```

This error first looked like a possible side effect of the new GPTQModel support, but further inspection showed that the real issue was environment drift between the official docker image and the fcloud instance.

The key practical question was why SGLang could start successfully when loading the original model, but fail on the quantized validation path on fcloud. The answer is that the failure was not caused by the original model itself versus the quantized model itself. Instead, it depended on whether the active runtime environment exposed the TorchAO API expected by the currently imported SGLang code path.

## Rule-Compliance Statement (SOAR)
This iteration is documentation only. It does not change preprocessing logic, serving logic, model weights, or benchmark behavior. It records an environment-alignment action taken to keep the fcloud instance consistent with the official testing environment.

## Detailed Implementation Plan
Planned and completed in this iteration:

1. Compare PyTorch and TorchAO versions between the official docker image and the fcloud instance.
2. Confirm whether the missing TorchAO symbol exists in each environment.
3. Decide whether to patch code or align the fcloud environment.
4. Prefer environment alignment because the official docker image is the authoritative competition reference.
5. Record the exact commands, diagnosis, and reasoning in the CHANGE_0030 documentation chain.

## Actual Findings and Actions
Official docker environment check:

```text
torch.__version__ = 2.9.1+cu128
torch.version.cuda = 12.8
torchao.__version__ = 0.9.0
has float8_dynamic_activation_float8_weight = True
```

fcloud environment before fix:

```text
Skipping import of cpp extensions due to incompatible torch version 2.9.1+cu128 for torchao version 0.16.0
torch.__version__ = 2.9.1+cu128
torchao.__version__ = 0.16.0
has float8_dynamic_activation_float8_weight = False
```

Interpretation:

- PyTorch itself was already aligned: both environments used `torch==2.9.1+cu128`.
- The real drift was TorchAO: official docker used `torchao==0.9.0`, while fcloud used `torchao==0.16.0`.
- The fcloud TorchAO build explicitly reported incompatibility with the installed PyTorch version.
- Because the active SGLang TorchAO helper imported TorchAO symbols eagerly, the missing symbol caused startup failure on fcloud.
- The same eager-import bug was masked in official docker because `torchao==0.9.0` still exported the expected symbol.

Environment-alignment fix applied on fcloud:

```bash
uv pip uninstall -y torchao
uv pip install torchao==0.9.0
```

Observed result after the reinstall:

- SGLang started successfully.

## Why the Environment Reinstall Was Chosen
The decision was to align fcloud with the official docker image instead of immediately introducing another code fix.

Reason 1:
The user prefers the fcloud instance environment to match the official testing environment as closely as possible.

Reason 2:
The user does not want to add unrelated bug fixes or new features unless they are required for performance improvement or they directly block competition progress.

This made TorchAO reinstallation the more appropriate immediate action than expanding the scope of the current iteration with a new SGLang robustness patch.

## Prior Diagnostic Explanation Recorded
The earlier diagnosis can be summarized as follows:

- SGLang can start successfully with the original model when the runtime environment provides a compatible TorchAO package or when the exact code path does not hit the incompatible import surface.
- The quantized validation failure on fcloud did not prove a new GPTQ checkpoint-structure bug by itself.
- The mismatch was environmental: official docker exposed the expected TorchAO symbol, while fcloud did not.
- Therefore, environment alignment was a valid and lower-scope resolution for continuing competition work.

## Validation Commands
Official docker inspection:

```bash
python - <<'PY'
import torch
import torchao
import torchao.quantization as q

print("torch.__version__ =", getattr(torch, "__version__", "unknown"))
print("torch.version.cuda =", getattr(torch.version, "cuda", "unknown"))
print("torchao.__version__ =", getattr(torchao, "__version__", "unknown"))
print("has float8_dynamic_activation_float8_weight =", hasattr(q, "float8_dynamic_activation_float8_weight"))
PY
```

fcloud inspection after reinstall:

```bash
python - <<'PY'
import torch
import torchao
import torchao.quantization as q

print("torch.__version__ =", getattr(torch, "__version__", "unknown"))
print("torch.version.cuda =", getattr(torch.version, "cuda", "unknown"))
print("torchao.__version__ =", getattr(torchao, "__version__", "unknown"))
print("has float8_dynamic_activation_float8_weight =", hasattr(q, "float8_dynamic_activation_float8_weight"))
PY
```

fcloud environment-alignment command:

```bash
uv pip uninstall -y torchao
uv pip install torchao==0.9.0
```

## Result Summary Table
| Item | Official docker | fcloud before fix | fcloud after fix |
|---|---:|---:|---:|
| `torch` | `2.9.1+cu128` | `2.9.1+cu128` | `2.9.1+cu128` |
| `torchao` | `0.9.0` | `0.16.0` | `0.9.0` |
| FP8 TorchAO symbol present | Yes | No | Yes |
| SGLang quantized startup | Works | Fails | Works |

## Rollback Instructions
If the environment alignment must be reverted on fcloud:

```bash
uv pip uninstall -y torchao
uv pip install torchao==0.16.0
```

However, for competition consistency this rollback is not recommended unless a later dependency explicitly requires it.

## Next-Step Suggestions
1. Continue quantized validation on the now-aligned fcloud environment and capture the next blocking issue, if any.
2. Record any remaining differences between fcloud and the official docker image before further optimization iterations.
3. Avoid adding non-performance code changes unless a blocker cannot be resolved by keeping the environment aligned with the official reference.