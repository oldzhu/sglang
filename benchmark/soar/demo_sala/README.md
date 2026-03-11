# MiniCPM-SALA 提交 Demo

本目录是一个最小可运行的提交示例，演示如何按照平台要求组织 `prepare_env.sh` + `prepare_model.sh` 提交包。

当前第一版提交准备基于已验证的 MiniCPM-SALA 4-bit GPTQ 路径，采用平台现场量化，不直接提交量化后模型文件。

默认提交模式为 `gptq`。如果需要手动回退到非量化路径，可显式设置 `SOAR_QUANT_MODE=copy`。

## 目录结构

```
.
├── prepare_env.sh          # 必须 — 环境构建脚本
├── prepare_model.sh        # 可选 — 模型预处理入口
├── preprocess_model.py     # prepare_model.sh 调用的 Python 脚本
├── gptqmodel_minicpm_sala.py  # MiniCPM-SALA 自定义 GPTQModel 定义
├── perf_public_set.jsonl   # 提交包内校准样本（需与脚本同目录）
├── flash_attn-2.8.3+cu128sm120-cp310-cp310-linux_x86_64.whl  # 官方提供的预编译 flash-attn wheel
└── sglang/python/          # 自定义 sglang 源码（editable install）
```

## 各文件说明

### `prepare_env.sh`（必须）

平台在基础环境启动后自动执行此脚本。本 demo 中做了两件事：

1. 用 `uv pip install --no-deps -e ./sglang/python` 将自定义 sglang 以 editable 模式安装，替换镜像内置版本
2. 安装 GPTQ 预处理所需依赖：`gptqmodel` 与本地 `flash-attn` 预编译 wheel
3. 通过 `export SGLANG_SERVER_ARGS` 追加已验证的量化推理启动参数

```bash
uv pip install --no-deps -e ./sglang/python
uv pip install gptqmodel --no-build-isolation -v
uv pip install ./flash_attn-2.8.3+cu128sm120-cp310-cp310-linux_x86_64.whl --no-build-isolation -v
export SGLANG_SERVER_ARGS="${SGLANG_SERVER_ARGS:-} --trust-remote-code --disable-radix-cache --attention-backend minicpm_flashinfer --chunked-prefill-size 32768 --max-prefill-tokens 32768 --prefill-max-requests 1 --max-running-requests 20 --mem-fraction-static 0.84 --schedule-conservativeness 1.0 --skip-server-warmup --dense-as-sparse --quantization gptq_marlin --log-level info"
```

> **注意**：`prepare_env.sh` 会被 `source` 进入平台主脚本，因此 `export` 的环境变量可以直接生效。
>
> **注意**：当前提交包不再现场编译 `flash-attn`，而是直接安装官方提供的预编译 wheel。请确保该 `.whl` 文件与 `prepare_env.sh` 处于同一目录。
>
> **注意**：在新的 fcloud 模拟环境里，带本地构建标签的 `gptqmodel==5.7.0+cu128torch2.9` 版本字符串无法直接通过 `uv` 解析，因此提交包改为使用未固定版本的 `gptqmodel` 安装命令，以匹配已验证可工作的环境构建路径。

### `prepare_model.sh`（可选）

平台在环境就绪后调用此脚本，接口固定为：

```bash
bash prepare_model.sh --input <原始模型路径> --output <处理后模型路径>
```

两个路径均由平台提供，选手无需关心容器内的具体挂载位置。

本 demo 目前支持两种预处理模式：

1. `copy`（默认）：仅复制模型文件
2. `gptq`：使用 GPTQModel 执行离线 GPTQ 量化

当前第一版提交已将默认模式调整为 `gptq`，`copy` 仅作为手动回退路径保留。

可通过环境变量控制 `gptq` 模式参数：

```bash
export SOAR_QUANT_MODE=gptq
export SOAR_GPTQ_CALIBRATION_FILE=/path/to/calibration.jsonl
export SOAR_GPTQ_CALIBRATION_FIELD=question
export SOAR_GPTQ_CALIBRATION_SAMPLES=32
export SOAR_GPTQ_BITS=4
export SOAR_GPTQ_GROUP_SIZE=128
export SOAR_GPTQ_BATCH_SIZE=1
```

执行：

```bash
bash prepare_model.sh --input <原始模型路径> --output <处理后模型路径>
```

说明：
- `gptq` 模式要求可用 `gptqmodel` 依赖（可在 `prepare_env.sh` 中安装）。
- 量化输出目录应包含 `quantize_config.json`，供 SGLang 加载 `--quantization gptq` 使用。
- 当前提交路径默认会尝试读取与脚本同目录下的 `perf_public_set.jsonl` 作为校准文件。
- 当前 fcloud 验证下，`SOAR_GPTQ_CALIBRATION_SAMPLES` 安全默认值为 `32`；更大的值可能在量化阶段触发 OOM。
- 当前量化推理启动路径使用 `--quantization gptq_marlin`。

### `sglang/python/`

自定义的 sglang 源码目录。通过 editable install，平台会使用此目录下的代码替代镜像内置 sglang，选手可以在此修改推理引擎的实现。

## 扩展示例

| 场景 | 修改点 |
|---|---|
| 安装额外 pip 包 | `prepare_env.sh` 中添加 `uv pip install xxx` |
| 自定义推理参数 | `prepare_env.sh` 中修改 `SGLANG_SERVER_ARGS` |
| GPTQ 量化 | `preprocess_model.py` 使用 GPTQModel 离线量化，`prepare_env.sh` 中追加 `--quantization gptq` |
| 模型剪枝/蒸馏 | `preprocess_model.py` 中实现，输出到 `--output` 目录 |

## 第一版提交打包建议

在 fcloud 上打包时，建议只打包本目录内容，不要把整个仓库一起打进提交包。

目录内应至少包含：

- `prepare_env.sh`
- `prepare_model.sh`
- `preprocess_model.py`
- `gptqmodel_minicpm_sala.py`
- `perf_public_set.jsonl`
- `flash_attn-2.8.3+cu128sm120-cp310-cp310-linux_x86_64.whl`
- `sglang/`

建议在提交目录内执行：

```bash
tar --exclude='__pycache__' --exclude='*.pyc' -czf minicpm_sala_submit_v1.tar.gz .
```
