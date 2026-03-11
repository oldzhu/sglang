# CHANGE_0031：量化 MiniCPM-SALA 的首次提交包准备

## 背景与动机
在 fcloud 上已经验证出一条可工作的 MiniCPM-SALA 4-bit GPTQ 路径后，下一步重点就是准备一个能够在官方 SOAR 执行模型下复现该行为的首次提交包。

官方 toolkit 要求使用 `prepare_env.sh` 加可选 `prepare_model.sh` 的提交方式，并明确禁止直接提交量化后的模型权重。因此，第一版提交必须打包量化工具链，并在评测机上现场执行 GPTQ。

## 规则合规说明（SOAR）
本轮仅做提交准备工作。

- 不直接提交量化后模型权重
- 保持 GPTQ 作为 `prepare_model.sh --input --output` 中的现场预处理步骤
- 按最新 toolkit 要求使用 `uv pip install`
- 对齐官方建议的 GPTQ W4A16 + Marlin 技术路径
- 不修改固定评测并发逻辑，也不重新启用被禁止的 prefix cache

## 详细实施计划
本轮计划并已实施：

1. 让 `prepare_env.sh` 与已验证的量化运行路径完全对齐。
2. 使用在干净 fcloud 提交模拟环境中实际可以成功解析的依赖安装命令。
3. 使用 organizer 提供的预编译 `flash-attn` wheel，并将其放在 `prepare_env.sh` 同目录下，从而避免提交时长时间编译和编译期 OOM 风险。
4. 导出已验证的 GPTQ Marlin 推理启动参数。
5. 将默认 GPTQ 校准样本数从 128 下调到 32，以匹配已验证的 fcloud 显存上限。
6. 将默认校准文件路径设为 `./perf_public_set.jsonl`，让提交包可自带校准 JSONL。
7. 将 `gptq` 设为默认提交预处理模式，避免 organizer 风格执行时静默回退到 `copy`。
8. 文档化首次提交包所需的目录内容与打包方式。

## 实际代码改动
更新文件：

- `benchmark/soar/demo_sala/prepare_env.sh`

改动内容：

1. 安装 editable vendored SGLang：
- `uv pip install --no-deps -e ./sglang/python`

2. 使用已验证可工作的提交模拟安装命令安装 GPTQ 相关依赖：
- `gptqmodel`

3. 从本地预编译 wheel 安装 `flash-attn`：
- `flash_attn-2.8.3+cu128sm120-cp310-cp310-linux_x86_64.whl`

4. 如果提交目录中缺少该 wheel，则立即失败。

5. 导出已验证的 GPTQ 预处理默认参数：
- `SOAR_QUANT_MODE` 默认值设为 `gptq`
- 校准文件默认值为 `./perf_public_set.jsonl`
- 校准样本数默认值为 `32`
- 校准 batch size 默认值为 `1`
- bits 默认值为 `4`
- group size 默认值为 `128`
- attention impl 默认值为 `flash_attention_2`

6. 导出量化推理时已验证的运行参数：
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

更新文件：

- `benchmark/soar/demo_sala/preprocess_model.py`

改动内容：

7. 默认校准文件回退路径从空字符串改为：
- `Path(__file__).resolve().parent / "perf_public_set.jsonl"`

8. 默认校准样本数从 `128` 改为 `32`

更新文件：

- `benchmark/soar/demo_sala/README.md`

改动内容：

9. 记录了本地 `flash-attn` 预编译 wheel 要求、运行参数、校准文件放置要求、安全默认值，以及首次提交的打包方式。

## 为什么采用这些设置
本提交准备包直接复用已在 fcloud 上验证成功的路径，因为这条路径已经证明可以产出可加载的 GPTQ 模型并成功启动 SGLang。

本轮记录下来的关键工程事实：

- GPTQ 校准样本数大于 `32` 时，在已验证的 fcloud 环境里可能 OOM
- 已成功的运行路径使用的是 `gptq_marlin`，不是普通 `gptq`
- 校准 JSONL 应该随提交包一起走，保证预处理自包含
- 如果 `SOAR_QUANT_MODE` 没有显式 export，organizer 风格执行可能会静默走到 `copy` 模式
- organizer 提供的预编译 `flash-attn` wheel 可以避免提交时长时间编译与编译期 OOM 风险

## 准确性/稳定性风险
优势：

- 使用了在干净 fcloud 模拟环境里实际可以成功解析的依赖安装路径
- 让提交包与已验证运行参数完全对齐
- 让评测机上的 GPTQ 预处理更可复现

剩余风险：

- 官方评测环境在显存行为上仍可能与 fcloud 有轻微差异
- 本地 wheel 的文件名与放置位置必须与脚本预期完全一致
- 正确率与速度仍建议在最终提交包流程上再做一次检查

## 验证命令
1. Shell 语法检查
```bash
bash -n benchmark/soar/demo_sala/prepare_env.sh
bash -n benchmark/soar/demo_sala/prepare_model.sh
```

2. Python 语法检查
```bash
python3 -m py_compile benchmark/soar/demo_sala/preprocess_model.py
python3 -m py_compile benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py
```

3. 提交风格预处理冒烟
```bash
cd benchmark/soar/demo_sala
SOAR_QUANT_MODE=gptq \
bash prepare_model.sh --input /path/to/raw_model --output /path/to/quant_output
```

4. 量化推理启动
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

5. 在 fcloud 上打包
```bash
tar --exclude='__pycache__' --exclude='*.pyc' -czf minicpm_sala_submit_v1.tar.gz .
```

## 结果汇总表
状态：提交包已对齐到已验证的量化工作流，并根据干净 fcloud 模拟环境中观察到的依赖解析行为完成调整；最终打包后的端到端检查仍待完成。

| 项目 | CHANGE_0031 前 | CHANGE_0031 后 |
|---|---:|---:|
| GPTQ 依赖安装路径 | 不稳定 | 已验证 |
| FlashAttention 安装方式 | 源码编译 | 本地预编译 wheel |
| 默认 GPTQ 校准样本数 | 128 | 32 |
| 默认打包校准 JSONL 路径 | 否 | 是 |
| 量化运行模式 | 不明确 | `gptq_marlin` |
| 提交包打包说明 | 部分 | 明确 |

## 回滚说明
1. 回滚 CHANGE_0031。
2. 将 `prepare_env.sh` 恢复为仅做 editable install 的简单版本。
3. 将 `preprocess_model.py` 的 GPTQ 默认值恢复到旧版本。
4. 如果需要，可退回 `SOAR_QUANT_MODE=copy` 的非量化提交路径。

## 后续建议
1. 在 fcloud 上把 `perf_public_set.jsonl` 放到脚本同目录后，构建最终提交目录。
2. 按提交方式再跑一次预处理加启动冒烟。
3. 仅打包提交目录内容，生成 `minicpm_sala_submit_v1.tar.gz`。