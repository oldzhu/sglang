# CHANGE_0052 Force-Dense Lightning 运行时优化

## 背景与动机

当前提交风格运行配置保持了 `--force-dense-minicpm`，这会关闭 MiniCPM 稀疏路径，使 lightning 层成为当前最主要的模型特定运行时优化目标。在这个配置下，首个高价值 feature 不应优先放在 sparse-topk，而应放在仍覆盖大多数 decoder 层的 SimpleGLA lightning backend 上。

因此，本次迭代只实现一个聚焦的 force-dense runtime feature：

- 针对 6000D 风格部署优化 lightning backend 的 dispatch 策略
- 降低 SimpleGLA 路径中 recurrent state gather/scatter 的开销

本次改动不改变模型数学逻辑，也不改输出语义。

## 规则合规说明

本次实现前已重新检查最新 SOAR competition / toolkit 页面。

- 属于官方允许的运行时优化范围：kernel/runtime 调度、内存与 KV 读写优化、推理 backend 调优。
- 不替换 MiniCPM-SALA 官方基座模型。
- 不依赖被禁止的 prefix cache 行为。
- 不修改官方固定并发评测模型。
- 保持 `prepare_env.sh` 与 `prepare_model.sh` 的提交契约不变。

## 详细实施计划

改动前计划：

1. 重新确认当前 force-dense serving 确实关闭了 sparse MiniCPM 层。
2. 追踪 lightning 路径，从 `MiniCPMLightningMixer` 到 `SimpleGLAAttnBackend`。
3. 将优化范围限定在本仓库可控的 SimpleGLA 边界上。
4. 使用环境变量做调优，而不是大范围引入新的 server args，确保回滚简单。

## 实际代码改动

修改文件：

- `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`
- `benchmark/soar/demo_sala/prepare_env.sh`

`hybrid_linear_attn_backend.py` 中的运行时改动：

1. 增加 force-dense 感知的 recurrent threshold 调优：
   - 新环境变量：`SGLANG_MINICPM_LIGHTNING_RECURRENT_THRESHOLD`
   - 在 force-dense MiniCPM 下默认值改为 `128`，不再写死为 `64`
2. 增加 fast state-IO 路径：
   - 新环境变量：`SGLANG_MINICPM_LIGHTNING_FAST_STATE_IO`
   - gather 使用 `torch.index_select`
   - writeback 使用 `index_copy_`
3. 在 backend 内缓存 lightning layer 到 cache slot 的映射，减少热路径上的重复字典查找。
4. 将 recurrent/chunk 模式选择收敛到单独 helper 中，使策略显式可调。
5. 删除 forward 热路径中的重复状态索引解析和重复 layer-cache 查找。

`prepare_env.sh` 中的环境默认值：

1. `SGLANG_MINICPM_LIGHTNING_FAST_STATE_IO=1`
2. `SGLANG_MINICPM_LIGHTNING_RECURRENT_THRESHOLD=128`
3. 在日志中打印这两个变量，便于 fcloud 运行时确认实际调优值。

## 设计说明

### 为什么优先针对 force-dense runtime

在 `--force-dense-minicpm` 下，当前并不是 sparse MiniCPM kernel 在主导优化价值。剩下最值得优先处理的模型特定运行时路径，就是覆盖多数层的 lightning backend。

### 为什么要改 recurrent threshold

旧 lightning 路径用的是固定的 `seq_len < 64` 分界来决定 recurrent 还是 chunk。这个阈值更像通用经验值，而不是针对当前 6000D 风格配置调出来的。此次迭代将 force-dense 下的默认阈值提高，用于更积极地把中等长度 extend 保留在 recurrent 快路径中，而更长输入仍然走 chunk。

### 为什么把 state IO 也纳入 feature

每一层 lightning 在 kernel 调用前都要 gather temporal state，调用后还要写回。这个过程会在大多数层、每个 decode step 中重复出现。对于当前 force-dense 运行时，先降低这部分 gather/scatter 开销，比直接去动 dense flashinfer attention 更适合作为第一 feature。

## 验证命令

语法检查：

```bash
python3 -m py_compile python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py
```

正确性验证：

```bash
python3 benchmark/soar/demo_sala/eval_model_001.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_DIR> \
  --data_path benchmark/soar/demo_sala/perf_public_set.jsonl \
  --concurrency 32
```

服务速度验证：

```bash
bash SOAR/bench_serving.sh http://127.0.0.1:30000
```

回滚对照：

```bash
export SGLANG_MINICPM_LIGHTNING_FAST_STATE_IO=0
export SGLANG_MINICPM_LIGHTNING_RECURRENT_THRESHOLD=64
```

## 结果汇总表

| 项目 | 修改前 | 修改后 |
| --- | --- | --- |
| Lightning recurrent threshold | 写死 `64` | 环境变量可调，force-dense 默认 `128` |
| State gather 路径 | advanced indexing + contiguous copy | `index_select` 快路径 |
| State writeback 路径 | advanced indexing assignment | `index_copy_` 快路径 |
| 回滚方式 | 需要改代码 | 仅需环境变量回滚 |
| 速度结果 | 待验证 | 待验证 |

## 回滚说明

如果新的 lightning 调优不稳定或者没有收益：

1. 设置 `SGLANG_MINICPM_LIGHTNING_FAST_STATE_IO=0`
2. 设置 `SGLANG_MINICPM_LIGHTNING_RECURRENT_THRESHOLD=64`
3. 重新执行相同的正确性和速度验证命令

回滚不需要重新量化模型，也不需要修改提交包结构。

## 下一步建议

1. 先看 S1 和 S8，因为这个 feature 预计会更直接改善 decode 主导路径。
2. 如果收益成立且正确性稳定，下一步可以继续向 external SimpleGLA kernel 边界推进。
3. 稀疏路径优化应当等当前 sparse crash 修复后再继续。