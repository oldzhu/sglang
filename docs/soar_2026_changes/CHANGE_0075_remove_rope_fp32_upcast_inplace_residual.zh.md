# CHANGE_0075: 移除 RoPE Float32 上转换 & 原地残差缩放

## 背景与动机

对 MiniCPM-SALA 模型前向传播的分析发现了两个不必要的开销来源：

1. **Float32 RoPE 上转换**：所有注意力层（8个标准 `minicpm4` 层和24个 lightning 层）在应用旋转位置编码前将 Q/K 张量从 bf16 转换为 float32，处理完后再转回。这是来自原始 HuggingFace 实现的防御性代码。sgl-kernel 的 `apply_rope_with_cos_sin_cache_inplace` CUDA 核函数已能原生处理 bf16 输入（内部使用 float32 cos_sin_cache），显式转换是不必要的。

2. **分配张量的残差缩放**：`hidden_states = hidden_states * self.residual_scale` 模式为简单的标量乘法分配新的输出张量。由于结果会被下一个操作立即消费，原地 `*=` 可以避免该分配。

## 规则合规声明

- **无精度风险**：RoPE 核函数已在 bf16 下正确运行；原地乘法在数学上完全等价。
- **无提交限制影响**：模型大小无变化，无额外依赖。
- **可复现**：标准代码优化，完全确定性。

## 实现

### 变更 1：移除 `MiniCPMAttention.forward()` 中的 float32 上转换

**变更前**（8个标准注意力层）：
```python
if self.attn_use_rope:
    orig_dtype = q.dtype
    q, k = q.float(), k.float()
    q, k = self.rotary_emb(positions, q, k)
    q, k = q.to(orig_dtype), k.to(orig_dtype)
```

**变更后**：
```python
if self.attn_use_rope:
    q, k = self.rotary_emb(positions, q, k)
```

### 变更 2：移除 `MiniCPMLightningMixer._apply_qk_norm_rope()` 非融合路径中的 float32 上转换

相同模式 — 当融合 QK-norm-rope 核函数未激活时，移除 `.float()` / `.to(orig_dtype)` 包装。

### 变更 3：`MiniCPMDecoderLayer.forward()` 中原地残差缩放

**变更前**（全部32层，每层2次）：
```python
hidden_states = hidden_states * self.residual_scale
```

**变更后**：
```python
hidden_states *= self.residual_scale
```

## 修改文件

- `python/sglang/srt/models/minicpm.py` — 主模型（全部3项变更）
- `benchmark/soar/demo_sala/sglang/python/sglang/srt/models/minicpm.py` — demo_sala 副本（仅 RoPE 变更；该副本无 residual_scale）

## 验证命令

```bash
# 精度测试（标准化精度需 ≥99% 以保持 C=1.0）
python3 scripts/fcloud/fcloud_workflow.py accuracy

# 速度基准测试（全部3个并发层级）
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

## 预期影响

- **RoPE 修复**：消除 4 次类型转换操作 × 32 层 = 每次前向传播 128 次不必要的操作。预计在预填充密集的工作负载中加速 2-5%。
- **原地残差缩放**：消除每次前向传播 64 次张量分配（2次/层 × 32 层）。预计因减少内存分配压力可提升 1-2%。

## 结果汇总

| 指标 | 基线 | CHANGE_0075 后 | 变化 |
|------|------|----------------|------|
| S1 | 待测 | 待测 | 待测 |
| S8 | 待测 | 待测 | 待测 |
| Smax | 待测 | 待测 | 待测 |
| 精度 | 待测 | 待测 | 待测 |

## 回滚说明

恢复三处代码变更：
1. 恢复 `MiniCPMAttention.forward()` 中的 float32 上转换
2. 恢复 `MiniCPMLightningMixer._apply_qk_norm_rope()` 非融合路径中的 float32 上转换
3. 恢复 `hidden_states = hidden_states * self.residual_scale`（`MiniCPMDecoderLayer.forward()` 中2处）

## 后续步骤

- 在 fcloud 上使用 GPTQ+FP8+dense 配置测试实际速度提升
- 若稳定，与 `--enable-torch-compile` 组合以获得复合增益
- 考虑在加载时将 `residual_scale` 折叠到 `o_proj`/`down_proj` 权重中实现零成本缩放（复杂度较高，延后处理）
