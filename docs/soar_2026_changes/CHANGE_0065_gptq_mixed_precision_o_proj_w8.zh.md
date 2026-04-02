# CHANGE_0065 GPTQ 混合精度：将 Attention Output Projection 提升到 W8

## 背景与动机

最新一次 calibration-only 实验已经把 `qa`、`mcq`、`cwe` 三个任务的 90 条公开样本全部用于校准，但正确性仍然停留在相近区间。这一结果明显削弱了“剩余精度损失主要来自校准采样不足”的假设。

因此，本轮迭代不再继续扩大 calibration，而是转向一个更窄、更安全的混合精度实验，前提是必须与当前 GPTQ + Marlin 的服务路径兼容。

本次选择的目标模块是 `self_attn.o_proj`：仅把它提升到 8-bit，其余已量化的注意力与 MLP 投影继续保持现有 GPTQ W4A16 baseline。之所以选它，是因为：

- 它位于之前出现兼容性风险的 merged `qkv_proj` 边界之外
- 它处在注意力路由完成之后，可能同时影响短上下文和长上下文行为
- 它仍然处于当前 `gptq_marlin` 支持的格式范围内；这条代码路径当前支持 4-bit 和 8-bit 对称格式

## 规则合规说明

本改动符合 2026-04-02 重新检查的最新 SOAR 比赛页与 toolkit 页面要求，包括 toolkit 中的 `技术路径指引` 与 `提交说明`。

- 仅修改预处理阶段的量化配置。
- 保持官方 `prepare_env.sh` 与 `prepare_model.sh --input/--output` 工作流不变。
- 不替换 MiniCPM-SALA 基座模型。
- 不修改固定并发设置、prefix-cache 规则或服务契约。
- 运行时仍保持在现有 `gptq_marlin` + FP8 KV-cache 路径上。

## 详细实施计划

改动前计划：

1. 保持当前 GPTQ include/exclude 模块范围不变。
2. 扩展 preprocess 的 dynamic-rule 构建逻辑，使其不仅支持负向 skip 规则，也支持正向的 per-module override。
3. 只增加一个支持的混合精度 preset：`o_proj_w8`，把 `self_attn.o_proj` 提升为 8-bit，group size 仍为 128。
4. 保持该 preset 由环境变量控制，便于无代码回滚地做 A/B 对比。

## 实际代码改动

修改文件：

- `benchmark/soar/demo_sala/preprocess_model.py`
- `benchmark/soar/demo_sala/prepare_env.sh`

具体变更：

1. `preprocess_model.py` 现在除了 exclusion rules 之外，也能生成正向 GPTQ dynamic override。
2. 新增 `SOAR_GPTQ_MIXED_PRECISION_PRESET=o_proj_w8` 支持。
3. 新增 `SOAR_GPTQ_O_PROJ_BITS` 和 `SOAR_GPTQ_O_PROJ_GROUP_SIZE` 作为 env 控制项。
4. 在 `prepare_env.sh` 中默认启用 `o_proj_w8` preset。

## 设计说明

### 为什么先选 `o_proj`

在本仓库里，`o_proj` 比 Q、K、V 更适合作为第一轮混合精度目标。MiniCPM 运行时会通过合并后的 `qkv_proj` 边界来服务 Q、K、V，而之前的 selective rollback 正是在这个 fused loader 路径上暴露出结构风险。`o_proj` 不受这个边界影响。

### 为什么不先做 3-bit MLP

当前代码库里的 `gptq_marlin` 运行时路径支持的是 4-bit 和 8-bit 对称格式，而不是 3-bit Marlin 服务格式。若在这里直接引入 3-bit，会牵涉不同运行时路径或更深层的 kernel/runtime 改造，不适合本轮 feature。

### 为什么保持 preset 可开关

这仍然是实验功能。通过环境变量控制开关，可以在不继续修改源码的前提下，直接比较 baseline W4A16 与 W8 `o_proj` 方案的差异。

## 验证命令

Shell 语法检查：

```bash
bash -n benchmark/soar/demo_sala/prepare_env.sh
```

Python 语法检查：

```bash
python3 -m py_compile benchmark/soar/demo_sala/preprocess_model.py
```

预处理验证：

```bash
bash benchmark/soar/demo_sala/prepare_model.sh --input <RAW_MODEL_DIR> --output <OUTPUT_MODEL_DIR>
```

正确性验证：

```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_DIR> \
  --data_path <DATA_DIR>/perf_public_set.jsonl \
  --concurrency 32
```

服务速度验证：

```bash
bash SOAR/bench_serving.sh http://127.0.0.1:30000
```

关闭该 preset，回到 baseline 对比：

```bash
export SOAR_GPTQ_MIXED_PRECISION_PRESET=off
```

## 结果汇总表

| 项目 | 修改前 | 修改后 |
| --- | --- | --- |
| GPTQ dynamic override 能力 | 仅 exclude | exclude + 正向 override |
| 混合精度 preset | 无 | `o_proj_w8` |
| Attention QKV 格式 | W4 baseline | 不变 |
| `self_attn.o_proj` 格式 | W4 baseline | W8 |
| 预期风险 | 低 | 速度小幅回退到中等风险 |
| 预期收益 | 当前 78 左右平台期 | 目标是提升精度稳定性 |

## 回滚说明

如果 W8 `o_proj` 实验不能明显改善正确性，或者速度代价过大：

1. 设置 `SOAR_GPTQ_MIXED_PRECISION_PRESET=off`
2. 重新执行 preprocess、correctness 和 serving 检查

如果后续需要彻底从源码中移除该功能，再删除 `prepare_env.sh` 中的 preset env 变量，以及 `preprocess_model.py` 中对应的正向 override 分支即可。

## 下一步建议

1. 先重点观察 `mcq` 是否改善，因为最新结果表明剩余精度损失不只是长上下文路由噪声。
2. 如果 `o_proj_w8` 有帮助但还不够，下一轮 mixed precision 实验仍应保持 fused-group-safe，优先考虑更窄的 `down_proj` 或 layer-subset 扩展。
3. 如果无效，就应把注意力从 calibration 和 mixed precision 转回运行时或更架构相关的精度假设。