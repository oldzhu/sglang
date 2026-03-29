# CHANGE_0060 MiniCPM Lightning 融合 QK-Norm-RoPE

## 背景与动机

当前提交仍保持 `--force-dense-minicpm`，原因是 MiniCPM 稀疏路径仍有稳定性问题。在这个运行模式下，真正活跃的模型特定热点不再是稀疏 `minicpm4` 路径，而是 MiniCPM-SALA 的 lightning attention 路径。

MiniCPM-SALA 官方配置中，`qk_norm=true`、`lightning_use_rope=true`，同时整体层结构是约 75% lightning、25% sparse。由于当前 force-dense 规避了 sparse 路径，因此把算子融合先落在 lightning 路径，是一个更低风险、更聚焦的下一步。

本仓库里已经存在一个现成的 SRT `fused_qk_norm_rope` kernel，并且已经被其他模型族使用。本次迭代优先复用这条现有 kernel 路径到 MiniCPM lightning mixer，而不是一开始就新写 MiniCPM 专用 kernel。

## 规则合规说明

本改动符合 2026-03-29 重新检查的最新 SOAR 比赛页与 toolkit 页面要求。

- 属于允许的运行时优化范围：算子融合与推理路径优化。
- 不替换 MiniCPM-SALA 基座模型。
- 保持 `prepare_env.sh` 与 `prepare_model.sh` 的提交契约不变。
- 不修改 benchmark 固定并发规则，也不触碰被禁止的 prefix-cache 行为。
- 保持现有 GPTQ W4A16 + Marlin + FP8 KV cache 路径不变。

## 详细实施计划

改动前计划：

1. 优先复用现有的 fused `qk_norm_rope` kernel，而不是先写新的 MiniCPM 专用 kernel。
2. 把 feature 范围严格限制在 force-dense 模式下的 lightning mixer 路径。
3. 保留原始 MiniCPM lightning 实现作为保守 fallback。
4. 通过现有 server arg 开启 feature，同时允许用单个环境变量快速回滚。

## 实际代码改动

修改文件：

- `python/sglang/srt/models/minicpm.py`
- `benchmark/soar/demo_sala/prepare_env.sh`

具体变更：

1. 在 `minicpm.py` 中新增 MiniCPM 本地 `_compute_yarn_parameters()` helper，用于给 fused kernel 提供与现有 SRT 融合路径一致的 RoPE scaling 参数。
2. 仅在 CUDA 上导入现有 `sgl_kernel.fused_qk_norm_rope`。
3. 在 `MiniCPMLightningMixer` 中加入兼容性守卫：
   - 仅 CUDA
   - `qk_norm` 已开启
   - RoPE 已开启
   - `head_dim` 只接受 `64`、`128`、`256`
   - 排除 `MRotaryEmbedding`
4. 新增 `MiniCPMLightningMixer._apply_qk_norm_rope()`。
5. 用 fused-or-fallback helper 替换原来的 lightning 显式流程：
   - split q/k/v
   - `q_norm`
   - `k_norm`
   - RoPE
6. 在 `prepare_env.sh` 中新增默认开关：
   - `SOAR_ENABLE_FUSED_QK_NORM_ROPE=1`
   - 自动追加 `--enable-fused-qk-norm-rope`
7. 在 `prepare_env.sh` 日志中打印这个新开关。

## 设计说明

### 为什么只做 lightning 路径

MiniCPM-SALA 官方配置中，稀疏 `minicpm4` 层使用 `attn_use_rope=false`，而 lightning 层使用 `lightning_use_rope=true` 且 `qk_norm=true`。因此 fused `qk_norm_rope` 与 lightning 路径匹配，而不是 sparse 路径。

### 为什么先复用现有 fused op

仓库里已经有一个可工作的融合 kernel，并且已有 SRT 模型在用。优先复用可以把 feature 范围控制得更小，也减少后续 kernel 维护成本。只有在这条复用路径被证明不兼容或太脆弱时，才值得继续写 MiniCPM 专用 kernel。

### 为什么保留严格 fallback

当前 MiniCPM lightning 的原始实现仍然是正确性基线。对于不满足 tensor contract 的情况，自动回退到原实现，比强制所有请求都走 fused path 更安全。

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
export SOAR_ENABLE_FUSED_QK_NORM_ROPE=0
```

## 结果汇总表

| 项目 | 修改前 | 修改后 |
| --- | --- | --- |
| MiniCPM lightning Q/K norm + RoPE | 分离执行 | 融合快路径 + 保守 fallback |
| Kernel 来源 | MiniCPM 原始 Python 路径 | 复用现有 SRT fused kernel |
| Force-dense 兼容性 | 无专门优化 | 显式针对当前模式 |
| 回滚方式 | 需要改代码 | `prepare_env.sh` 中环境变量可控 |

## 回滚说明

如果 fused lightning 路径不稳定，或速度收益不明显：

1. 设置 `SOAR_ENABLE_FUSED_QK_NORM_ROPE=0`
2. 重新执行同样的正确性与速度验证
3. 如有需要，可以保留代码但在提交时关闭该 feature，再评估是否值得进入 MiniCPM 专用 fused op 路线

## 下一步建议

1. 先验证正确性，因为当前 public correctness gate 仍是主约束。
2. 如果正确性稳定且速度有收益，就把它作为新的 force-dense lightning 基线。
3. 如果速度收益有限，再通过 profiling 判断剩余瓶颈更偏向 SimpleGLA backend 还是 Marlin 路径，再决定继续推进 `CHANGE_0052` 还是回到 `sgl-kernel` 调优。