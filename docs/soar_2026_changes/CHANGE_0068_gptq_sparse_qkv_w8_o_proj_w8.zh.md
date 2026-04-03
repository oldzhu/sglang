# CHANGE_0068 GPTQ 混合精度：Sparse-QKV W8 与 O-Proj W8 组合预设

## 背景与动机

之前两条 mixed-precision 分支单独测试后，都只表现出不完整的收益。

- `o_proj_w8` 在 `mcq` 上有一些正向信号，但不足以把 `qa` 和总分拉回目标区间。
- `sparse_qkv_w8` 经过 `CHANGE_0067` 的 loader 对齐修正后已经可以作为有效实验方向，但单独看也没有给出足够强的精度恢复结果。

因此，本轮迭代把这两条已经验证过结构可行性的假设组合为一个受控 preset：在 sparse/full-attention 层上把 QKV 提升到 8-bit，同时把全局 `self_attn.o_proj` 提升到 8-bit。目标是验证剩余精度损失是否同时分布在注意力形成阶段和后置输出投影阶段，而不是只集中在其中之一。

## 规则合规说明

本改动符合 2026-04-03 重新检查的最新 SOAR 比赛页与 toolkit 页面要求，包括 toolkit 中的 `技术路径指引` 与 `提交说明`。

- 仅修改预处理阶段的量化配置。
- 保持官方 `prepare_env.sh` 与 `prepare_model.sh --input/--output` 工作流不变。
- 不替换 MiniCPM-SALA 基座模型。
- 不修改固定并发设置、prefix-cache 规则或服务契约。
- 运行时仍保持在现有 `gptq_marlin` + FP8 KV-cache 路径上。

## 详细实施计划

改动前计划：

1. 保持现有 `o_proj_w8` 与 `sparse_qkv_w8` 行为不变。
2. 增加一个新的组合 preset：`sparse_qkv_w8_o_proj_w8`。
3. 在同一个 dynamic-rule builder 中组合已有的 sparse-layer QKV W8 override 和全局 `self_attn.o_proj` W8 override。
4. 将组合 preset 设为默认值，便于下一轮 fcloud 直接测试，无需额外改 env。

## 实际代码改动

修改文件：

- `benchmark/soar/demo_sala/preprocess_model.py`
- `benchmark/soar/demo_sala/prepare_env.sh`

具体变更：

1. 重构 mixed-precision preset 的判定逻辑，使一个 preset 可以启用一组或两组 override。
2. 新增 `SOAR_GPTQ_MIXED_PRECISION_PRESET=sparse_qkv_w8_o_proj_w8` 支持。
3. 保留原有 `o_proj_w8`、`sparse_qkv_w8` 和 `off` 的有效性。
4. 在 `prepare_env.sh` 中把默认 mixed-precision preset 切换为新的组合 preset。

## 设计说明

### 为什么不直接支持逗号分隔 preset 名称

当前 preset parser 设计上是一次接受一个受控 feature 名称。相比自由组合的逗号分隔字符串，单独定义一个组合 preset 更容易验证、记录和对比。

### 为什么在转向其他方向前先做这次组合测试

这基本上是权重量化 mixed-precision 分支里的最后一次干净组合实验。如果这个组合 preset 仍然不能恢复足够的精度，那么就有更强证据说明这条分支已经接近耗尽。

### 为什么保留旧 preset

保留单独的 preset，可以继续做干净的 A/B 对比，而不用再进行源码回滚。

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

Preset 对比：

```bash
export SOAR_GPTQ_MIXED_PRECISION_PRESET=off
export SOAR_GPTQ_MIXED_PRECISION_PRESET=o_proj_w8
export SOAR_GPTQ_MIXED_PRECISION_PRESET=sparse_qkv_w8
export SOAR_GPTQ_MIXED_PRECISION_PRESET=sparse_qkv_w8_o_proj_w8
```

## 结果汇总表

| 项目 | 修改前 | 修改后 |
| --- | --- | --- |
| Mixed-precision preset | 仅单分支 | 单分支或组合分支 |
| W8 注意力目标 | sparse QKV 或 `o_proj` | sparse QKV + `o_proj` |
| 默认 preset | `sparse_qkv_w8` | `sparse_qkv_w8_o_proj_w8` |
| 预期风险 | 中等 | 中到偏高的速度回退 |
| 预期收益 | 局部或不明确恢复 | mixed-precision 分支的最后一次组合精度恢复测试 |

## 回滚说明

如果组合 preset 不能明显改善正确性，或者速度代价过大：

1. 设置 `SOAR_GPTQ_MIXED_PRECISION_PRESET=off`
2. 重新执行 preprocess、correctness 和 serving 检查

如果最后只有其中一个单分支更有希望，可以直接切回 `o_proj_w8` 或 `sparse_qkv_w8`，无需改代码。

## 下一步建议

1. 比较 aggregate score，但重点仍然看 `qa` 和 `mcq`，因为这次组合 preset 的目的就是叠加之前两个分支各自最强的信号。
2. 如果这次仍然失败，就停止继续扩展 mixed-precision 分支，转向新的假设类别。
3. 如果精度改善但速度回退过大，优先缩小 sparse 层范围，而不是继续引入新的 W8 目标。