# CHANGE_0100: 残差缩放因子折叠

## 背景与动机

MiniCPM-SALA 在每次前向传播时使用三个运行时缩放操作：

1. **`residual_scale`**（`scale_depth / √num_hidden_layers` = 0.247487）：每个解码层执行两次 — 自注意力 o_proj 输出后和 MLP down_proj 输出后。56 层意味着每次前向传播 **112 次标量乘法内核启动**。

2. **`scale_emb`**（12）：在进入解码器栈之前应用于嵌入查找输出。每次前向传播 **1 次乘法**。

3. **`1/scale_width`**（`1/16.0`）：在计算 logits 之前应用于最终隐藏状态。每次前向传播 **1 次除法**。

每个操作都是简单的标量-张量运算，但每个都会启动一个独立的 CUDA 内核。总计：每次前向传播 **114 次不必要的内核启动**。

由于这些都是模型加载时已知的固定标量，可以在加载时折叠（预乘）到相关权重矩阵中，完全消除运行时开销。

## 规则合规声明

- **数学精确**：标量乘法与所有线性运算（GEMM、Marlin 反量化、排列）可交换。不引入任何近似。
- **无精度影响**：计算完全相同；只是标量应用的时机改变。
- **无提交约束影响**：无额外文件、无大小增加、无额外量化时间。
- **符合 SOAR 规则**：无禁止技巧；纯代数优化。

## 实施方案

### 折叠策略

| 缩放因子 | 折叠位置 | 目标属性 |
|---|---|---|
| `residual_scale` (0.247487) | o_proj Marlin 反量化缩放 | `layer.self_attn.o_proj.scales` × 56 层 |
| `residual_scale` (0.247487) | down_proj Marlin 反量化缩放 | `layer.mlp.down_proj.scales` × 56 层 |
| `scale_emb` (12) | embed_tokens 权重 | `model.embed_tokens.weight` |
| `1/scale_width` (1/16) | lm_head 权重 | `lm_head.weight` |

### GPTQ Marlin 兼容性

Marlin 内核计算：`output = X @ (scales * Q_int)`。将 `scales` 乘以标量 `s` 得到 `output' = X @ (s * scales * Q_int) = s * output`。这正是运行时 `hidden_states *= residual_scale` 所做的。标量乘法也与 `marlin_permute_scales()` 可交换，因为那是纯排列操作。

### 时序

折叠在 `post_load_weights()` 中执行，由模型加载器在所有量化层的 `process_weights_after_loading()` 运行**之后**调用。这意味着 Marlin 缩放已完全物化和排列，然后我们再乘入标量。

## 实际代码更改

### 文件：`python/sglang/srt/models/minicpm.py`

1. **添加 `logging` 导入和 logger**。

2. **`MiniCPMDecoderLayer.__init__`**：添加 `self.scaling_folded = False` 标志。

3. **`MiniCPMDecoderLayer.forward()`**：将两个 `hidden_states *= self.residual_scale` 行用 `if not self.scaling_folded:` 保护。

4. **`MiniCPMModel.__init__`**：添加 `self._scaling_folded = False` 标志。

5. **`MiniCPMModel.forward()`**：使 `* self.config.scale_emb` 在 `not self._scaling_folded` 条件下执行。

6. **`MiniCPMForCausalLM.__init__`**：添加 `self._scaling_folded = False` 标志。

7. **`MiniCPMForCausalLM.forward()`**：使 `/ self.scale_width` 在 `not self._scaling_folded` 条件下执行。

8. **`MiniCPMForCausalLM._fold_scaling_factors()`**：新方法，执行：
   - 将所有 56 层的 `o_proj.scales` 和 `down_proj.scales` 乘以 `residual_scale`
   - 将 `embed_tokens.weight` 乘以 `scale_emb`
   - 将 `lm_head.weight` 除以 `scale_width`
   - 设置所有 `scaling_folded` 标志为 `True`

9. **`MiniCPMForCausalLM.post_load_weights()`**：新方法（在权重处理后由 sglang 模型加载器调用），调用 `_fold_scaling_factors()`。

### 向后兼容性

- `if not self.scaling_folded:` 保护确保代码在未调用 `post_load_weights()` 时仍然正确（例如不同的模型加载器）。此时使用原始运行时缩放。
- `MiniCPMForCausalLM.forward()` 中的外部 `input_embeds` 路径仍在运行时应用 `* self.config.scale_emb`，因为外部嵌入不经过 `embed_tokens`。

## 验证命令

### 精度测试
```bash
python3 scripts/fcloud/fcloud_workflow.py accuracy
```

### 速度测试
```bash
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

### 预期行为
- 服务器日志应显示：`Folded scaling factors: residual_scale=0.247487, scale_emb=12, scale_width=16.0`
- 精度应与基线相同（在浮点噪声范围内）
- 速度提升：1-3%（消除每次前向传播 114 次内核启动）

## 结果汇总

| 指标 | 基线 (Test 20) | M1 后 | 变化 |
|---|---|---|---|
| S1 | 113.67s | 112.55s | -0.99% |
| S8 | 41.07s | 41.04s | -0.07% |
| Smax | 34.15s | 34.58s | +1.26%（变慢） |
| 精度 | 80.64% | 78.64% | -2.00 点 |
| 归一化 | 100.80% | 98.30% | -2.50 点 |
| C | 1.0 | 0.96 | 下降一个档位 |

### Test 23 详情（2026-04-17，commit `f373fbade`）

- 精度测试总时长：3234.19s
- 输出 TPS：492.87 tokens/s
- 分任务精度：cwe=81.00%，fwe=98.89%，mcq=56.67%，niah=100.00%，qa=56.67%

### 结论

在当前基线组合上，M1 **没有带来稳定净收益**。速度收益接近噪声（S1 小幅提升、S8 基本持平、Smax 略慢），但归一化精度降至 98.30%，导致 C 从 1.0 降到 0.96，不满足提交安全要求。

建议：暂时保留 CHANGE_0100 代码用于后续排查，但在恢复到 normalized >99%（C=1.0）前，不纳入最终提交路径。

## 回滚说明

回滚实现此更改的单个提交：
```bash
git revert <commit-hash>
```

或手动：移除 `post_load_weights`、`_fold_scaling_factors` 方法，移除所有 `scaling_folded` 标志和保护，恢复无条件的 `*= residual_scale`、`* scale_emb`、`/ scale_width` 行。

## 后续建议

- **K4**：将 RMSNorm + residual_scale 融合为单个 CUDA 内核
- **A1**：SimpleGLA 状态连续性保证（5-8% 解码提升）
- **A3**：将状态 I/O 融合到 FLA 内核（10-15% 解码提升）
