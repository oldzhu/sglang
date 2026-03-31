# CHANGE_0064 GPTQ 保留完整 QKV 精度

## 背景与动机

CHANGE_0063 试图只保留 `self_attn.k_proj` 的高精度，同时继续量化 `q_proj` 和 `v_proj`。这个假设出发点是为了改善精度，但在本仓库的 MiniCPM 运行时加载结构下，它与模型加载逻辑并不兼容。

MiniCPM 在加载时会把 `q_proj`、`k_proj`、`v_proj` 重写并合并为单个 `qkv_proj` 参数。只保留 K 会形成一种混合 checkpoint 结构：量化流程可以跑完，但服务加载时会因为缺少兼容的 `qkv_proj.weight` 路径而失败。因此，上一个 feature 不是“效果一般”，而是结构上不可加载。

本次纠偏迭代用一个与 loader 兼容的策略替代它：保留完整 QKV 投影的高精度，把 GPTQ 量化范围收缩到其余注意力输出与 MLP 投影。

## 规则合规说明

本改动符合 2026-03-31 重新检查的最新 SOAR 比赛页与 toolkit 页面要求，包括 toolkit 中的 `技术路径指引` 和提交工作流约束。

- 仅修改预处理阶段的量化范围。
- 保持官方 `prepare_env.sh` 与 `prepare_model.sh --input/--output` 工作流不变。
- 不改变服务契约、并发规则或 prefix-cache 行为。
- 对模型其余部分仍保持当前 SGLang 运行时、GPTQ/Marlin 后端和 FP8 KV-cache 配置。
- 这是去除已知不兼容量化布局的修正，不是引入任何违规运行时技巧。

## 详细实施计划

改动前计划：

1. 从默认 GPTQ include/exclude 列表中移除“只保留 K”的策略。
2. 改成保留完整 QKV，以保证三个投影在加载时遵循同一种格式。
3. 在模块不匹配的 retry 路径里也应用同样策略，避免 retry 时悄悄回到不兼容布局。
4. 其余量化范围保持不变，把改动严格限制在 QKV 兼容性问题上。

## 实际代码改动

修改文件：

- `benchmark/soar/demo_sala/prepare_env.sh`
- `benchmark/soar/demo_sala/preprocess_model.py`

具体变更：

1. 更新默认 `SOAR_GPTQ_INCLUDE_MODULES`，将 `self_attn.q_proj`、`self_attn.k_proj`、`self_attn.v_proj` 从 GPTQ 量化范围中移除。
2. 更新默认 `SOAR_GPTQ_EXCLUDE_MODULES`，显式排除完整 QKV 三个投影。
3. 更新 `preprocess_model.py` 内部默认值，使直接执行 preprocess 时也采用相同的完整 QKV 保留策略。
4. 更新 GPTQ module-mismatch retry 路径，保持与主路径一致的 include/exclude 行为。

## 设计说明

### 为什么要保留完整 QKV，而不是只保留 K

在这个 MiniCPM 实现里，Q、K、V 在服务加载时并不是彼此独立的终点，而是会堆叠成 `qkv_proj`。在这个合并边界内部做局部回退，会得到不一致的 checkpoint 表示。保留完整 QKV，是恢复结构一致性的最小改动。

### 为什么这仍然是有意义的精度实验

原始精度假设是：注意力投影精度是 `qa`、`mcq`、`cwe` 持续不稳定的来源之一。保留完整 QKV 可以继续验证这个更宽的假设，同时避免与运行时结构冲突。

### 为什么不连 `o_proj` 一起保留

那样会进一步扩大回退范围，也更难判断精度或速度变化究竟来自哪里。本次纠偏只修复已经确认出问题的 QKV 边界。

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

回滚到 0064 之前的量化范围：

```bash
export SOAR_GPTQ_INCLUDE_MODULES=self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj
export SOAR_GPTQ_EXCLUDE_MODULES=self_attn.o_gate,self_attn.z_proj
```

## 结果汇总表

| 项目 | CHANGE_0063 | CHANGE_0064 |
| --- | --- | --- |
| 保留的注意力投影 | 仅 K | 完整 Q + K + V |
| Loader 兼容性 | 与合并 `qkv_proj` 冲突 | 与合并 `qkv_proj` loader 对齐 |
| 预处理结果 | 量化可能完成 | 量化后模型应保持可加载 |
| 速度预期 | 可能略有回退 | 回退可能大于 K-only，但仍可控 |
| 精度预期 | 只针对 K 敏感性 | 更广义的注意力精度恢复 |

## 回滚说明

如果保留完整 QKV 仍然不能明显改善正确性，或者速度代价过大：

1. 把 `self_attn.q_proj`、`self_attn.k_proj`、`self_attn.v_proj` 恢复到 GPTQ include 列表
2. 从 GPTQ exclude 列表中移除这三个模块
3. 重新执行 preprocess、模型加载、correctness 和 serving 检查

不要回滚到 CHANGE_0063 的“仅保留 K”布局，因为该布局已经确认与当前 MiniCPM merged-QKV loader 不兼容。

## 下一步建议

1. 先确认量化后的模型现在能否成功被 SGLang 加载；这是这个修正是否成立的第一道门槛。
2. 如果加载成功，再对比 `qa`、`mcq`、`cwe` 相对当前 baseline 是否改善，然后再决定是否继续调整 calibration。
3. 如果速度回退过大，下一轮 accuracy 实验应换一个边界，而不是回到已经确认不可用的 K-only 分裂方案。

## 采用前的状态更新

这个 feature 目前也已临时回滚，不作为当前代码路径中的有效方案。

原因如下：

1. 在 CHANGE_0063 之后的排查中，已经确认服务失败的直接报错是 `KeyError: 'model.layers.0.self_attn.qkv_proj.weight'`。
2. 虽然“保留完整 QKV”在 checkpoint 结构上比“只保留 K”更合理，但它仍然没有在当前 SGLang MiniCPM 运行时的 quantization-selection 路径上完成充分验证。
3. 在继续投入更多迭代到 projection-preservation 之前，当前计划先回到最后一个已知可加载的 GPTQ scope，做一次更干净的 calibration-only 实验。

临时回滚的原因：

- 需要更多时间复核 merged-QKV loader 边界和运行时量化映射细节。
- 当前马上要做的实验风险更低：保持 `qa,mcq,cwe` 任务范围不变，把 calibration samples 提高到 90，使这三个任务的全部公开样本都参与校准。
- 保留本文件，是把它作为后续可能重启的设计记录，而不是已经采用的优化 feature。