# CHANGE_0056 Prepare-Env 本地 Wheel 安装

## 背景与动机

在把提交环境冻结到已知的 fcloud 依赖集合之后，下一个明显瓶颈变成了依赖安装过程的耗时和不稳定性。通过远端索引安装 `gptqmodel==5.7.0`、`transformers==4.57.1`、`torchao==0.9.0`，在网络较慢时会比较脆弱，而且 `gptqmodel==5.7.0` 也没有可直接下载的目标环境预编译 wheel。

因此，本次迭代把这组已固定版本的依赖改为本地 wheel 安装。用户已经准备好：

- 本地构建的 `gptqmodel-5.7.0-*.whl`
- 已下载的 `transformers-4.57.1-*.whl`
- 已下载的 `torchao-0.9.0-*.whl`

目标是让提交环境安装过程更可控，减少对远程网络与索引可用性的依赖。

## 规则合规说明

本改动符合最新 SOAR toolkit 的提交流程要求。

- 保持 `prepare_env.sh` + `prepare_model.sh --input/--output` 契约不变。
- 仅修改 `prepare_env.sh` 中的依赖安装方式。
- 不替换 MiniCPM-SALA 基座模型，也不改变评测行为。
- 不修改并发、prefix cache 或运行时打分逻辑。

## 详细实施计划

改动前计划：

1. 停止从远端索引安装已固定版本的 GPTQ 依赖集合。
2. 在 demo 提交目录中查找所需的本地 wheel。
3. 如果任意 wheel 缺失或匹配到多个版本，则尽早失败并打印清晰错误。
4. 通过 `uv pip install --force-reinstall --no-deps ...` 安装这些本地 wheel。
5. 保留安装后版本探针，确保官方日志可以证明实际生效版本。

## 实际代码改动

修改文件：

- `benchmark/soar/demo_sala/prepare_env.sh`

具体变更：

1. 新增以下本地 wheel 的 glob 查找：
   - `gptqmodel-5.7.0-*.whl`
   - `transformers-4.57.1-*.whl`
   - `torchao-0.9.0-*.whl`
2. 新增严格检查，要求每个包都必须且只能匹配到一个 wheel。
3. 将以下包的远端安装替换为本地 wheel 安装：
   - `gptqmodel`
   - `transformers`
   - `torchao`
4. 保留现有的 flash-attn 与 sgl-kernel 本地 wheel 安装流程。
5. 保留原有的安装后 Python 探针，继续打印 `torch`、`gptqmodel`、`transformers`、`torchao` 的实际版本和导入路径。

## 设计说明

### 为什么要改成本地 wheel，而不是只保留远端 pin

精确 pin 可以解决版本漂移，但无法解决慢网络和不稳定远程下载的问题。本地 wheel 可以同时消除这两类不确定性：既固定版本，也避免网络波动。

### 为什么只有 `gptqmodel` 需要本地构建

`transformers` 是纯 Python wheel，`torchao` 也已经存在可用的预编译 wheel。只有 `gptqmodel==5.7.0` 没有直接可下载的目标环境 wheel，因此用户在 fcloud 上从 source archive 构建了本地 wheel。

### 为什么不重建 `transformers` 或 `torchao`

重建 `transformers` 不会带来 6000D 特定的性能收益，而这里使用 `torchao` 的目的主要是固定兼容版本，而不是生成面向某个 GPU 架构专门调优的自定义 kernel 包。这与 `sgl-kernel` wheel 不同，后者的目标硬件定向原生代码生成确实直接影响性能。

## 验证命令

先把三个 wheel 放到 demo 目录下，然后执行：

```bash
bash benchmark/soar/demo_sala/prepare_env.sh
```

检查实际版本：

```bash
python3 - <<'PY'
import importlib
import json
for name in ["torch", "gptqmodel", "transformers", "torchao"]:
    module = importlib.import_module(name)
    print(json.dumps({
        "module": name,
        "version": getattr(module, "__version__", "unknown"),
        "file": getattr(module, "__file__", None),
    }, ensure_ascii=False))
PY
```

恢复 preprocess 验证：

```bash
bash benchmark/soar/demo_sala/prepare_model.sh --input <RAW_MODEL_DIR> --output <OUTPUT_MODEL_DIR>
```

## 结果汇总表

| 项目 | 修改前 | 修改后 |
| --- | --- | --- |
| `gptqmodel` 安装 | 远端 pin 安装 | 本地 wheel 安装 |
| `transformers` 安装 | 远端 pin 安装 | 本地 wheel 安装 |
| `torchao` 安装 | 远端 pin 安装 | 本地 wheel 安装 |
| 环境构建对网络的依赖 | 需要 | 对这组已固定 GPTQ 依赖不再需要 |
| 依赖可复现性 | 已 pin 但仍依赖远端获取 | 已 pin 且绑定到本地制品 |

## 回滚说明

如果本地 wheel 打包流程过于麻烦，或者某个 wheel 暂时不可用：

1. 恢复 `prepare_env.sh` 中此前的远端精确 pin 安装逻辑
2. 删除本地 wheel 存在性检查
3. 重新执行 `prepare_env.sh`，并通过安装后探针确认实际版本

## 下一步建议

1. 在打包提交前，把三个 wheel 文件复制到 `benchmark/soar/demo_sala/` 目录。
2. 重新提交官方任务，并确认日志中显示的是本地 wheel 安装以及预期的固定版本。
3. 如果在这一步之后 preprocess 仍然失败，再在“没有依赖漂移、没有网络变量”的前提下继续排障。