# CHANGE_0057 GPTQModel 本地 Wheel 依赖解析

## 背景与动机

在把已固定的 GPTQ 依赖集合改成本地 wheel 安装之后，官方提交仍然在量化开始前失败。失败发生在导入 `gptqmodel` 时，错误表明本地 wheel 以 `--no-deps` 安装后缺失了运行所需的轻量级 Python 依赖：

```text
ModuleNotFoundError: No module named 'logbar'
```

根因是本地 `gptqmodel` wheel 使用 `--no-deps` 安装，阻止了其传递依赖被自动解析。同时，`prepare_env.sh` 的 fail-fast 模式此前是关闭的，因此安装后的导入探针失败没有在环境准备阶段就终止流程。

本次迭代只修复这一处打包问题，同时继续保持 `transformers` 与 `torchao` 通过本地 wheel 固定安装。

## 规则合规说明

本改动符合最新 SOAR toolkit 的提交流程要求。

- 保持 `prepare_env.sh` + `prepare_model.sh --input/--output` 契约不变。
- 仅修改 `prepare_env.sh` 内部的依赖安装行为。
- 不改变 MiniCPM-SALA 基座模型、运行时打分逻辑、并发或 prefix-cache 行为。

## 详细实施计划

改动前计划：

1. 恢复 `prepare_env.sh` 的 fail-fast shell 行为。
2. 继续保留 `transformers` 和 `torchao` 的本地 wheel 固定安装。
3. 将本地 `gptqmodel` wheel 改为允许解析依赖。
4. 把 `gptqmodel` 放到 pinned local wheels 之后安装，尽量避免它的依赖解析成为主要版本漂移来源。
5. 保留安装后导入探针，确保缺失依赖在 `prepare_env.sh` 阶段就被发现。

## 实际代码改动

修改文件：

- `benchmark/soar/demo_sala/prepare_env.sh`

具体变更：

1. 恢复 `set -euo pipefail`，让环境准备阶段在探针或安装失败时立即停止。
2. 调整安装顺序：
   - 先安装本地 `transformers` wheel
   - 再安装本地 `torchao` wheel
   - 最后安装本地 `gptqmodel` wheel
3. 去掉 `gptqmodel` 本地 wheel 安装中的 `--no-deps`，允许其解析依赖。
4. 在重装 `gptqmodel` 前增加显式卸载步骤，与现有 `torchao` 处理方式一致。
5. 保留原有安装后 Python 探针，继续检查 `torch`、`gptqmodel`、`transformers`、`torchao` 的导入与版本。

## 设计说明

### 为什么只让 `gptqmodel` 解析依赖

`transformers` 与 `torchao` 已经通过本地 wheel 明确 pin 住，因此应继续保持确定性和无网络漂移。`gptqmodel` 则依赖一些轻量级 Python 包，例如 `logbar`。对 `gptqmodel` 使用 `--no-deps`，会让本地 wheel 在运行时缺少这些必要组件。

### 为什么要把 `gptqmodel` 放在 pinned wheels 之后安装

先装好 pinned local wheels，可以减少 `gptqmodel` 依赖解析时改写 `transformers` 或 `torchao` 版本的风险。希望最终效果是：`gptqmodel` 只补齐缺失的轻量级依赖，而不破坏主版本集合。

### 为什么要重新开启 fail-fast

之前脚本在导入探针失败后仍继续执行，导致真正的错误延迟到 `prepare_model.sh` 才暴露。现在在 `prepare_env.sh` 内就失败，能让官方日志更直接反映打包依赖问题。

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
for name in ["torch", "gptqmodel", "transformers", "torchao", "logbar"]:
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
| `gptqmodel` 本地 wheel 安装 | 使用 `--no-deps` | 启用依赖解析 |
| `logbar` 这类轻量依赖缺失 | 可能发生 | 期望在安装期被补齐 |
| `prepare_env.sh` 错误处理 | 未开启 fail-fast | 开启 fail-fast |
| 打包错误暴露时机 | 延后到 `prepare_model.sh` | 提前到 `prepare_env.sh` |

## 回滚说明

如果允许 `gptqmodel` 解析依赖后导致 pinned package 被意外升级：

1. 回退到之前的本地 wheel 安装逻辑
2. 将缺失的轻量依赖单独做成 local wheel 或精确 pin
3. 保留 fail-fast shell 模式，确保依赖问题仍能尽早暴露

## 下一步建议

1. 重新提交官方任务，确认 `logbar` 不再是导入错误。
2. 检查 `gptqmodel` 依赖解析后，`transformers==4.57.1` 和 `torchao==0.9.0` 是否仍然保持不变。
3. 如果又暴露出其他轻量级依赖缺失，再把那批小依赖显式 vendor 或 pin，而不是放弃本地 wheel 方案。