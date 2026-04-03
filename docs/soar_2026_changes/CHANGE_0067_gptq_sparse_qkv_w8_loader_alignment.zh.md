# CHANGE_0067 GPTQ Sparse-QKV W8 Loader 对齐修正

## 背景与动机

CHANGE_0066 引入了一个 mixed-precision preset，在 sparse/full-attention 层上把 `q_proj`、`k_proj`、`v_proj` 提升到 8-bit。这个思路本身是合理的，但第一次端到端运行在 SGLang 加载阶段失败，并出现了 fused-QKV shape mismatch：

```text
AssertionError: param_data.shape=torch.Size([512, 256]), loaded_weight.shape=torch.Size([1024, 256])
```

这个报错说明：sparse-layer QKV W8 规则已经作用到了 GPTQ preprocess 阶段的 unfused shard 名称，但没有正确作用到 SGLang runtime 阶段的 fused `qkv_proj` 名称。结果就是 checkpoint shard 按 W8 packing 生成，而运行时目标参数仍按 W4 packing 创建。

本次纠偏迭代保留 sparse-layer QKV W8 方向，但通过同时对 unfused shard 名称和 fused `qkv_proj` 运行时名称发出 dynamic override，修复 preprocess/runtime 的命名不一致问题。

## 规则合规说明

本改动符合 2026-04-03 重新检查的最新 SOAR 比赛页与 toolkit 页面要求，包括 toolkit 中的 `技术路径指引` 与 `提交说明`。

- 仅修改预处理阶段的量化配置。
- 保持官方 `prepare_env.sh` 与 `prepare_model.sh --input/--output` 工作流不变。
- 不替换 MiniCPM-SALA 基座模型。
- 不修改固定并发设置、prefix-cache 规则或服务契约。
- 运行时仍保持在现有 `gptq_marlin` + FP8 KV-cache 路径上。

## 详细实施计划

改动前计划：

1. 保持 sparse-layer mixed-precision preset 名称和环境变量接口不变。
2. 保留 GPTQ preprocess 侧对 `q_proj`、`k_proj`、`v_proj` 的 shard-level override。
3. 为相同 sparse layers 增加 `qkv_proj` 的 fused-name override。
4. 其余量化范围保持不变，把修正严格限制在 load-shape mismatch 问题上。

## 实际代码改动

修改文件：

- `benchmark/soar/demo_sala/preprocess_model.py`

具体变更：

1. `sparse_qkv_w8` preset 仍然会为 sparse-layer 的 `self_attn.q_proj`、`self_attn.k_proj`、`self_attn.v_proj` 生成正向 override。
2. 现在会额外为相同 sparse-layer 的 `self_attn.qkv_proj` 生成匹配的正向 override。
3. bits 和 group size 继续由现有 sparse-QKV 环境变量控制。

## 设计说明

### 为什么要同时发出 unfused 和 fused 两套名称

GPTQ preprocess 处理的是 unfused checkpoint module tree，而 SGLang runtime 会把同一组逻辑权重加载到 fused QKV 参数里。因此，同一个 mixed-precision 决策必须在这两套命名下都可见。

### 为什么这个 shape mismatch 能说明 bit-width 不一致

报错里的目标形状 `[512, 256]` 与来源形状 `[1024, 256]`，与 4-bit 和 8-bit GPTQ Marlin 权重的 packing factor 不一致相吻合。这说明运行时参数和 checkpoint shard 并不是按同一种量化格式创建的。

### 为什么现在还不应放弃 sparse-QKV W8

这次失败发生在加载阶段，精度根本还没有机会被评估。它是 loader 对齐问题，不是 sparse-layer QKV W8 假设已经无效的证据。

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

## 结果汇总表

| 项目 | CHANGE_0066 | CHANGE_0067 |
| --- | --- | --- |
| Sparse-layer W8 目标 | Q + K + V shards | Q + K + V shards + fused `qkv_proj` |
| Preprocess/runtime 命名对齐 | 不完整 | 对齐 |
| Loader 结果 | fused-QKV shape mismatch | 目标是可加载的 sparse-QKV W8 |
| 精度信号 | 尚无法测量 | 加载成功后才可测量 |

## 回滚说明

如果修正后的 sparse-QKV W8 路径仍然无法加载，或者速度/精度代价过大：

1. 设置 `SOAR_GPTQ_MIXED_PRECISION_PRESET=off`
2. 重新执行 preprocess、correctness 和 serving 检查

如果加载成功但速度回退偏大，应优先通过 `SOAR_GPTQ_SPARSE_LAYER_IDS` 缩小 sparse 层范围，而不是直接移除这个修正。

## 下一步建议

1. 首先确认模型现在能够成功加载，这是这次纠偏是否成立的第一道门槛。
2. 如果加载成功，再优先比较 `qa`，而不是只看 aggregate score。
3. 如果精度改善但速度回退过大，下一步先缩小 sparse 层子集，再考虑更换目标模块。