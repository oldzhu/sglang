# CHANGE_0030_007：fcloud 上 TorchAO 环境漂移问题与解决

## 背景与动机
在 CHANGE_0030_006 之后，量化后的 MiniCPM-SALA 已经能在 SGLang 启动流程中走得更远，但 fcloud 上的服务仍然因为 TorchAO 导入错误而失败：

```text
ImportError: cannot import name 'float8_dynamic_activation_float8_weight' from 'torchao.quantization'
```

这个错误一开始看起来像是新的 GPTQModel 支持引入的副作用，但进一步排查后确认，真正的问题是 fcloud 实例和官方 docker 镜像之间的环境漂移。

这里的核心问题是：为什么加载原始模型时 SGLang 可以正常启动，而在 fcloud 上跑量化验证时却失败。答案并不是“原始模型没问题、量化模型有问题”，而是当前运行环境是否暴露了当前 SGLang 代码路径所期望的 TorchAO API。

## 规则合规说明（SOAR）
本轮仅为文档记录，不修改预处理逻辑、服务逻辑、模型权重或 benchmark 行为。其目的是记录一次为了让 fcloud 环境与官方测试环境保持一致而采取的环境对齐动作。

## 详细实施计划
本轮计划并已完成：

1. 对比官方 docker 镜像与 fcloud 实例中的 PyTorch 和 TorchAO 版本。
2. 确认缺失的 TorchAO 符号在两个环境中是否存在。
3. 判断应该修代码还是先对齐 fcloud 环境。
4. 因官方 docker 镜像才是比赛参考环境，所以优先选择环境对齐。
5. 将排查过程、命令、结论与决策理由写入 CHANGE_0030 文档链。

## 实际发现与处理动作
官方 docker 环境检查结果：

```text
torch.__version__ = 2.9.1+cu128
torch.version.cuda = 12.8
torchao.__version__ = 0.9.0
has float8_dynamic_activation_float8_weight = True
```

fcloud 修复前环境检查结果：

```text
Skipping import of cpp extensions due to incompatible torch version 2.9.1+cu128 for torchao version 0.16.0
torch.__version__ = 2.9.1+cu128
torchao.__version__ = 0.16.0
has float8_dynamic_activation_float8_weight = False
```

结论解释：

- PyTorch 本身并没有漂移，两个环境都是 `torch==2.9.1+cu128`。
- 真正漂移的是 TorchAO：官方 docker 使用 `torchao==0.9.0`，而 fcloud 使用 `torchao==0.16.0`。
- fcloud 上的 TorchAO 还明确报告它与当前 PyTorch 版本不兼容。
- 由于当前 SGLang TorchAO helper 会提前导入 TorchAO 符号，缺失符号会直接导致 fcloud 启动失败。
- 在官方 docker 中之所以没有暴露这个问题，是因为 `torchao==0.9.0` 仍然提供了预期符号，从而把这个问题“掩盖”了。

在 fcloud 上采取的环境对齐修复：

```bash
uv pip uninstall -y torchao
uv pip install torchao==0.9.0
```

修复后的观察结果：

- SGLang 可以成功启动。

## 为什么选择重装 TorchAO 作为解决方案
这次选择先把 fcloud 与官方 docker 对齐，而不是立刻继续引入新的代码修复。

原因 1：
用户希望 fcloud 实例环境尽量与官方测试环境保持一致。

原因 2：
用户不希望增加额外的 bug fix 或新功能，除非它是性能提升所必需，或者它已经直接阻塞比赛推进。

因此，相比把当前迭代范围扩大到新的 SGLang 健壮性修复，先重装 TorchAO 进行环境对齐，是更符合当前比赛策略的选择。

## 对此前诊断结论的记录
此前回复中的核心判断可以归纳为：

- 当运行环境提供兼容的 TorchAO 包，或者实际代码路径没有命中不兼容的导入面时，SGLang 可以成功启动并加载原始模型。
- fcloud 上量化验证失败，并不自动等价于新的 GPTQ checkpoint 结构问题。
- 这次差异的根因是环境问题：官方 docker 暴露了预期的 TorchAO 符号，而 fcloud 没有。
- 因此，为了继续比赛工作，优先做环境对齐是一种有效且范围更小的解决办法。

## 验证命令
官方 docker 环境检查：

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

fcloud 重装后检查：

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

fcloud 环境对齐命令：

```bash
uv pip uninstall -y torchao
uv pip install torchao==0.9.0
```

## 结果汇总表
| 项目 | 官方 docker | fcloud 修复前 | fcloud 修复后 |
|---|---:|---:|---:|
| `torch` | `2.9.1+cu128` | `2.9.1+cu128` | `2.9.1+cu128` |
| `torchao` | `0.9.0` | `0.16.0` | `0.9.0` |
| FP8 TorchAO 符号是否存在 | 是 | 否 | 是 |
| SGLang 量化启动 | 成功 | 失败 | 成功 |

## 回滚说明
如果后续必须在 fcloud 上回退此次环境对齐，可执行：

```bash
uv pip uninstall -y torchao
uv pip install torchao==0.16.0
```

但从比赛一致性角度看，不建议这样做，除非后续依赖明确要求。

## 后续建议
1. 在已经对齐的 fcloud 环境上继续跑量化验证，捕获下一个真正的阻塞点（如果还有）。
2. 在后续优化迭代前，继续记录 fcloud 与官方 docker 镜像之间是否仍有其他环境差异。
3. 除非无法通过保持环境与官方参考一致来解决问题，否则不要额外引入非性能类代码修改。