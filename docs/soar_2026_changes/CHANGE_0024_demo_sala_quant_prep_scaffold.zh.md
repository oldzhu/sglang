# CHANGE_0024_demo_sala_quant_prep_scaffold

## 1) 背景与动机
- 用户要求基于 SOAR 工具页 `提交说明` 定制提交脚本。
- 当前 demo 仅做模型拷贝，无法表达量化预处理流程。

## 2) 规则合规说明
- 本次改动遵循官方提交接口：
  - `prepare_env.sh`
  - `prepare_model.sh --input --output`
  - `preprocess_model.py` 作为预处理入口
- 不涉及仓库内推理算法改动。

## 3) 变更前计划
- 在 `prepare_env.sh` 增加量化模式感知。
- 在 `preprocess_model.py` 增加 `copy|gptq` 两种模式。
- 默认仍保持安全的 `copy` 行为。

## 4) 实际代码改动
- `benchmark/soar/demo_sala/prepare_env.sh`
  - 启用严格模式（`set -euo pipefail`）
  - 读取 `SOAR_QUANT_MODE`（默认 `copy`）
  - 当 mode 为 `gptq` 时，向 `SGLANG_SERVER_ARGS` 追加 `--quantization gptq`
  - 打印 mode/args 方便追踪
- `benchmark/soar/demo_sala/preprocess_model.py`
  - 新增 `--mode copy|gptq`（未传时回退到 `SOAR_QUANT_MODE`）
  - 抽离拷贝逻辑为 helper
  - 增加 GPTQ 预检（输入文件与依赖检查）
  - GPTQ 路径保留为脚手架，并在缺少具体实现时给出明确报错

## 5) 验证命令
```bash
bash -n benchmark/soar/demo_sala/prepare_env.sh
python3 -m py_compile benchmark/soar/demo_sala/preprocess_model.py

# 默认 copy 模式
bash benchmark/soar/demo_sala/prepare_model.sh --input /path/to/raw --output /path/to/out

# gptq 脚手架模式（在补充具体实现前，预期给出明确失败提示）
SOAR_QUANT_MODE=gptq bash benchmark/soar/demo_sala/prepare_model.sh --input /path/to/raw --output /path/to/out
```

## 6) 风险
- 低。默认路径仍是文件拷贝。
- GPTQ 模式在未补充具体实现前会快速失败并给出明确提示。

## 7) 回滚方案
1. 回退 `benchmark/soar/demo_sala/prepare_env.sh`。
2. 回退 `benchmark/soar/demo_sala/preprocess_model.py`。

## 8) 下一步建议
- 在 `preprocess_model.py` 中接入具体 GPTQ 量化流程，输出包含量化配置文件的模型目录。
