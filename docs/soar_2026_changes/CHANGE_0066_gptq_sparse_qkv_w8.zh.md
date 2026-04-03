# CHANGE_0066 GPTQ 混合精度：Sparse 层 QKV 提升到 W8

## 背景与动机

上一轮混合精度实验只把 `self_attn.o_proj` 提升到 8-bit，其余量化路径保持 GPTQ W4A16 baseline。该实验对 `mcq` 有一定帮助，但不足以把 `qa` 拉回目标区间，因此整体分数仍未恢复到预期带宽。

这个结果说明，剩余精度损失更可能与注意力形成过程有关，而不只是后置的输出投影问题。同时，如果把 QKV 全局提升到 W8，又会带来更大的运行时成本。

MiniCPM-SALA 已经通过模型配置中的 `mixer_types` 区分 sparse/full-attention 层与 lightning-attention 层，这给出了一个比手工猜测 layer ID 更合理的子集。因此，本轮迭代只在 sparse/full-attention 层上，把 `q_proj`、`k_proj`、`v_proj` 一起提升到 8-bit，而 lightning 层和其余量化路径保持不变。

## 规则合规说明

本改动符合 2026-04-03 重新检查的最新 SOAR 比赛页与 toolkit 页面要求，包括 toolkit 中的 `技术路径指引` 与 `提交说明`。

- 仅修改预处理阶段的量化配置。
- 保持官方 `prepare_env.sh` 与 `prepare_model.sh --input/--output` 工作流不变。
- 不替换 MiniCPM-SALA 基座模型。
- 不修改固定并发设置、prefix-cache 规则或服务契约。
- 运行时仍保持在现有 `gptq_marlin` + FP8 KV-cache 路径上。

## 详细实施计划

改动前计划：

1. 保持当前 GPTQ include/exclude 模块范围不变。
2. 扩展 mixed-precision preset 逻辑，使其不仅能匹配全局模块名，也能生成带 layer 限定的模块 regex。
3. 增加一个支持的 preset：`sparse_qkv_w8`，仅在 sparse/full-attention 层上同时把 `self_attn.q_proj`、`self_attn.k_proj`、`self_attn.v_proj` 提升到 8-bit。
4. 默认从 `config.mixer_types` 自动解析 sparse layer IDs，同时保留环境变量覆盖，方便后续缩小子集。

## 实际代码改动

修改文件：

- `benchmark/soar/demo_sala/preprocess_model.py`
- `benchmark/soar/demo_sala/prepare_env.sh`

具体变更：

1. 新增对环境变量整数 layer-id 列表的解析支持。
2. 新增从 `config.mixer_types` 自动解析 sparse layer IDs 的逻辑。
3. 扩展 GPTQ dynamic-rule 生成逻辑，支持新的 `sparse_qkv_w8` preset。
4. 新增 `SOAR_GPTQ_SPARSE_QKV_BITS`、`SOAR_GPTQ_SPARSE_QKV_GROUP_SIZE` 与可选的 `SOAR_GPTQ_SPARSE_LAYER_IDS`。
5. 在 `prepare_env.sh` 中把默认 mixed-precision preset 切换为 `sparse_qkv_w8`。

## 设计说明

### 为什么先用全部 sparse 层，而不是手工指定 layer IDs

sparse/full-attention 层更有可能承载 `qa` 类型所依赖的路由质量恢复，而 lightning 层又与当前重点优化的 recurrent fast path 绑定更紧。先使用架构定义好的 sparse 子集，比在没有更多证据之前随意指定 layer IDs 更合理。

### 为什么要一起提升 Q、K、V

MiniCPM 运行时通过融合的 QKV 边界来服务这三个投影。一起提升整个 QKV group，可以避免早前 selective rollback 曾暴露过的结构不一致风险。

### 为什么保留显式 layer-id 覆盖能力

如果“全部 sparse 层”能提升精度但速度代价偏大，下一轮就可能只保留其中一部分 sparse 层，而不需要再改代码。这个 env 覆盖就是为那一步准备的。

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

Baseline 对比：

```bash
export SOAR_GPTQ_MIXED_PRECISION_PRESET=off
```

手工缩小子集示例：

```bash
export SOAR_GPTQ_SPARSE_LAYER_IDS=0,4,8,12
```

## 结果汇总表

| 项目 | 修改前 | 修改后 |
| --- | --- | --- |
| Mixed-precision preset | `o_proj_w8` | `sparse_qkv_w8` |
| W8 注意力目标 | 仅 `o_proj` | sparse 层上的 fused Q + K + V |
| 层选择方式 | 全局模块匹配 | config 驱动的 sparse 层子集 |
| Loader 安全性 | 安全 | 只要 QKV 一起提升则安全 |
| 预期风险 | 速度低到中等回退 | 中等速度回退 |
| 预期收益 | 更偏向改善 `mcq` | 目标是更好恢复 `qa` |

## 回滚说明

如果 sparse-layer QKV W8 不能明显改善正确性，或者速度代价过大：

1. 设置 `SOAR_GPTQ_MIXED_PRECISION_PRESET=off`
2. 重新执行 preprocess、correctness 和 serving 检查

如果精度提升但速度代价偏大，可以保留该 preset，并通过 `SOAR_GPTQ_SPARSE_LAYER_IDS` 缩小受影响的 sparse 层范围。

## 下一步建议

1. 先重点比较 `qa`，尤其是 `len_4k_32k` 与 `len_32k_128k` 两个 bucket，因为上一轮 W8 `o_proj` 在这些位置仍然偏弱。
2. 如果 sparse-layer QKV W8 带来精度收益但速度回退过大，下一轮 feature 应该缩小 sparse 子集，而不是立即回退整个方向。
3. 如果仍然不能恢复准确率，下一步就应把重点从 calibration 和权重量化精度转向 generation-behavior 控制或其他更架构相关的假设。