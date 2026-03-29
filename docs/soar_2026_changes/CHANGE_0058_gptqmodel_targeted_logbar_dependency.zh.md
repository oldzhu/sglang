# CHANGE_0058 GPTQModel 定向补齐轻量依赖

## 背景与动机

让本地 `gptqmodel` wheel 自行解析依赖，虽然可以解决第一个已观察到的缺失模块 `logbar`，但也带来了大量远程下载以及对重量级依赖的重新解析，包括 `torch`、`triton` 和多种 NVIDIA CUDA wheels。这会重新引入安装耗时与兼容性漂移风险。

截至目前，官方日志已经明确报出的轻量缺失依赖有 `logbar`、`accelerate`、`threadpoolctl`、`tokenicer`、由 `pypcre` 提供的 `pcre` 模块，以及由 `device-smi` 提供的 `device_smi` 模块。之后又出现了 `huggingface-hub==1.7.2` 与 `transformers==4.57.1` 不兼容的问题，因为后者要求 `huggingface-hub>=0.34.0,<1.0`。因此，本次迭代继续采用更窄的修复方式：显式安装这些轻量依赖，并把 `huggingface-hub` 固定回兼容版本，然后把本地 `gptqmodel` wheel 安装方式恢复为 `--no-deps`。

## 规则合规说明

本改动符合最新 SOAR toolkit 的提交流程要求。

- 保持 `prepare_env.sh` + `prepare_model.sh --input/--output` 契约不变。
- 仅修改 `prepare_env.sh` 中的依赖安装行为。
- 不改变 MiniCPM-SALA 基座模型、运行时打分逻辑、并发或 prefix-cache 行为。

## 详细实施计划

改动前计划：

1. 保持 fail-fast shell 行为开启。
2. 保留 `transformers` 与 `torchao` 的 pinned local wheel 安装。
3. 新增对轻量依赖 `logbar`、`accelerate`、`threadpoolctl`、`tokenicer`、`pypcre` 与 `device-smi` 的显式安装。
4. 在安装 `accelerate` 后，把 `huggingface-hub` 固定回与 `transformers==4.57.1` 兼容的版本。
4. 将本地 `gptqmodel` wheel 安装方式恢复为 `--no-deps`。
5. 保留安装后导入探针，以便下一次如果还有其他轻量依赖缺失，也能立即发现。

## 实际代码改动

修改文件：

- `benchmark/soar/demo_sala/prepare_env.sh`

具体变更：

1. 新增显式安装以下轻量依赖：
    - `logbar`
    - `accelerate`
    - `threadpoolctl`
    - `tokenicer`
    - `pypcre`
    - `device-smi`
2. 将 `accelerate` 安装改为使用 `--no-deps`
3. 新增显式固定 `huggingface-hub==0.34.4`
4. 将本地 `gptqmodel` wheel 安装恢复为 `--no-deps`
5. 保留以下本地 wheel 安装：
   - `transformers==4.57.1`
   - `torchao==0.9.0`
6. 保留 fail-fast shell 模式与安装后导入/版本探针

## 设计说明

### 为什么显式安装 `logbar`、`accelerate`、`threadpoolctl`、`tokenicer`、`pypcre` 和 `device-smi`

`logbar`、`accelerate`、`threadpoolctl`、`tokenicer`、`pypcre` 和 `device-smi` 是目前官方日志已经确认缺失的轻量依赖。只补齐这些已知缺失的小包，比放开 `gptqmodel` 的整个依赖图从远端重新解析要安全得多。

### 为什么要固定 `huggingface-hub`

官方日志显示过 `huggingface-hub==1.7.2`，但 `transformers==4.57.1` 要求 `huggingface-hub>=0.34.0,<1.0`。如果不加约束地安装 `accelerate`，可能把 hub 升级到不兼容的 `1.x` 版本。因此脚本现在会显式恢复 `huggingface-hub==0.34.4`。

### 为什么把 `gptqmodel` 恢复成 `--no-deps`

前一版允许依赖解析后，触发了大量重量级包下载，并可能改变原本已经 pin 住的运行时栈。恢复到 `--no-deps` 后，大包依赖重新具备确定性，而轻量依赖则通过定向补齐的方式处理。

### 为什么未来继续采用增量补齐方式

如果下一次官方运行又报出另一个小型 Python 依赖缺失，可以用同样方式继续显式添加。虽然这种方法比完整 resolver 更慢，但对于必须保证提交环境可复现的场景更安全。

## 验证命令

环境安装验证：

```bash
bash benchmark/soar/demo_sala/prepare_env.sh
```

显式检查导入：

```bash
python3 - <<'PY'
import importlib
import json
for name in ["torch", "gptqmodel", "transformers", "torchao", "logbar", "accelerate", "threadpoolctl", "tokenicer", "pcre", "device_smi", "huggingface_hub"]:
    module = importlib.import_module(name)
    print(json.dumps({
        "module": name,
        "version": getattr(module, "__version__", "unknown"),
        "file": getattr(module, "__file__", None),
    }, ensure_ascii=False))
PY
```

然后重试 preprocess：

```bash
bash benchmark/soar/demo_sala/prepare_model.sh --input <RAW_MODEL_DIR> --output <OUTPUT_MODEL_DIR>
```

## 结果汇总表

| 项目 | 修改前 | 修改后 |
| --- | --- | --- |
| `gptqmodel` 安装策略 | 本地 wheel + 自动解析依赖 | 本地 wheel + `--no-deps` |
| `gptqmodel` 轻量依赖 | 运行时缺失 | 显式安装 `logbar`、`accelerate`、`threadpoolctl`、`tokenicer`、`pypcre` 与 `device-smi` |
| `huggingface-hub` 兼容性 | 可能漂移到不兼容的 `1.x` | 固定为 `0.34.4` 以匹配 `transformers==4.57.1` |
| 重量级包重新解析风险 | 高 | 尽量压低 |
| 后续小依赖补齐方式 | 隐式 resolver | 显式增量 vendoring |

## 回滚说明

如果显式安装这些轻量依赖仍然不够，且手工补齐的小依赖越来越多：

1. 查看下一次官方日志中缺失的模块
2. 决定是继续增加一个轻量显式依赖，还是把这组轻量依赖整体做成 local wheel bundle
3. 在没有充分必要之前，继续避免放开重量级依赖的自由解析

## 下一步建议

1. 重新提交官方任务，确认 `logbar`、`accelerate`、`threadpoolctl`、`tokenicer`、`pcre` 与 `device_smi` 不再缺失。
2. 如果又有其他轻量依赖缺失，继续以显式补齐方式处理，而不是重新放开完整依赖解析。
3. 确认 `huggingface-hub` 仍保持 `<1.0`，保证 `transformers==4.57.1` 能正常导入。
4. 如果缺失的小包数量开始变多，再考虑把这批轻量依赖整体打成本地 wheel 一起提交。