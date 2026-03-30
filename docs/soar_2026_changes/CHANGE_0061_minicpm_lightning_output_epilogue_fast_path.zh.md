# CHANGE_0061 MiniCPM Lightning 输出尾段快路径

## 背景与动机

在 lightning-only `qk_norm_rope` 融合路径上线之后，当前本地 baseline 已经有了明显提升，但 MiniCPM lightning mixer 仍然保留了一段重复执行的 post-attention epilogue：

1. 可选 output RMSNorm
2. `z_proj(hidden_states)`
3. `sigmoid(z)`
4. 与 attention output 做逐元素 gating multiply
5. `o_proj`

在当前 `--force-dense-minicpm` 运行模式下，lightning 层是主要活跃路径，因此即便只是减少这段 epilogue 的小量临时张量开销，也值得先在本地基线上验证，再决定是否进入更高风险的 kernel 级优化。

本次迭代保持数学逻辑完全不变，只加入一个低风险快路径，用于降低 lightning 输出 gating epilogue 的临时张量开销。

## 规则合规说明

本改动符合 2026-03-30 重新检查的最新 SOAR 比赛页与 toolkit 页面要求。

- 属于纯运行时路径优化。
- 不替换 MiniCPM-SALA 基座模型。
- 不修改提交并发规则或 prefix-cache 行为。
- 保持现有 GPTQ + Marlin + FP8 KV cache 路径不变。
- 通过单个环境变量即可回滚。

## 详细实施计划

改动前计划：

1. 保持 lightning 输出数学逻辑不变。
2. 把 post-attention epilogue 收敛到一个 helper 中。
3. 仅为推理风格执行增加一个受控快路径。
4. 用环境变量控制 feature，避免回滚时必须改代码。

## 实际代码改动

修改文件：

- `python/sglang/srt/models/minicpm.py`
- `benchmark/soar/demo_sala/prepare_env.sh`

具体变更：

1. 新增本地运行时开关 `SGLANG_MINICPM_LIGHTNING_FAST_OUTPUT_GATE`。
2. 为 `MiniCPMLightningMixer` 新增 `_apply_output_epilogue()`。
3. 把 lightning post-attention 尾段逻辑迁移到该 helper 中。
4. 为输出 gating 序列新增受控快路径，触发条件为：
   - `SGLANG_MINICPM_LIGHTNING_FAST_OUTPUT_GATE=1`
   - autograd 已关闭
   - `z_proj` 输出 dtype 与 attention output dtype 一致
5. 在快路径中：
   - 使用 `z.sigmoid_()` 原地计算 sigmoid
   - 使用 `attn_output.mul_(z)` 原地应用 gate
6. 如果任一守卫不满足，则自动回退到原先的非原地实现。
7. 在 `prepare_env.sh` 中打印该环境变量。

## 设计说明

### 为什么说它是低风险

数学顺序没有改变：

1. `o_norm`
2. `z_proj`
3. `sigmoid`
4. gate multiply
5. `o_proj`

本次优化只是在张量 contract 安全时，把其中一部分改成原地操作。

### 为什么这仍可能带来速度收益

这个 feature 并没有去掉 epilogue 里的两个线性层，因此不应期待大幅收益。它针对的是更小但会重复出现的开销：

- 少分配一个 sigmoid 输出临时张量
- 少分配一个 out-of-place gating multiply 结果张量
- 降低在 hot path 中重复出现的内存读写开销

### 为什么要加开关

收益可能较小，也可能依赖硬件。单独的环境变量可以让当前本地 baseline 的 A/B 对比更直接。

## 验证命令

Python 语法检查：

```bash
python3 -m py_compile python/sglang/srt/models/minicpm.py
```

Shell 语法检查：

```bash
bash -n benchmark/soar/demo_sala/prepare_env.sh
```

正确性验证：

```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_DIR> \
  --data_path <DATA_DIR>/perf_public_set.jsonl \
  --concurrency 32
```

速度验证：

```bash
bash SOAR/bench_serving.sh http://127.0.0.1:30000
```

回滚开关：

```bash
export SGLANG_MINICPM_LIGHTNING_FAST_OUTPUT_GATE=0
```

## 结果汇总表

| 项目 | 修改前 | 修改后 |
| --- | --- | --- |
| Lightning 输出 gate 路径 | 通用的非原地 sigmoid + multiply | 带 fallback 的受控原地快路径 |
| 数学顺序 | 不变 | 不变 |
| 回滚方式 | 需要改代码 | 环境变量即可 |
| 预期收益 | baseline | 降低重复 epilogue 小开销 |

## 回滚说明

如果这个 fast epilogue 路径不稳定，或者没有收益：

1. 设置 `SGLANG_MINICPM_LIGHTNING_FAST_OUTPUT_GATE=0`
2. 重新执行正确性与速度验证
3. 如果后续仍想保留 helper 结构，也可以只关闭快路径，继续作为以后 A/B 的基础代码

## 下一步建议

1. 先用当前新的本地 baseline 做 A/B，尤其重点看 S1。
2. 如果收益太小，下一步再转向更深的 Marlin epilogue 融合或 output-projection layout 调优。
3. 如果收益可测且正确性稳定，就把它并入新的本地 baseline，再继续后续 kernel 级优化。