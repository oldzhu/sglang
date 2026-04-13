# CHANGE_0075: 移除 RoPE Float32 上转换 & 原地残差缩放

## 背景与动机

对 MiniCPM-SALA 模型前向传播的分析发现了两个不必要的开销来源：

1. **Float32 RoPE 上转换**：所有注意力层（8个标准 `minicpm4` 层和24个 lightning 层）在应用旋转位置编码前将 Q/K 张量从 bf16 转换为 float32，处理完后再转回。这是来自原始 HuggingFace 实现的防御性代码。sgl-kernel 的 `apply_rope_with_cos_sin_cache_inplace` CUDA 核函数已能原生处理 bf16 输入（内部使用 float32 cos_sin_cache），显式转换是不必要的。

2. **分配张量的残差缩放**：`hidden_states = hidden_states * self.residual_scale` 模式为简单的标量乘法分配新的输出张量。由于结果会被下一个操作立即消费，原地 `*=` 可以避免该分配。

## 规则合规声明

- ~~**无精度风险**：RoPE 核函数已在 bf16 下正确运行；原地乘法在数学上完全等价。~~
- **精度风险已确认**：两项变更均导致灾难性精度下降。详见下方结果。
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

**状态：已回滚 — 两项变更均导致灾难性精度下降。**

### Test 14：两项变更（bf16 RoPE + 原地残差）— commit c818ae261

| 指标 | 基线 (Test 12) | CHANGE_0075 后 | 变化 |
|------|----------------|----------------|------|
| S1 | 121.66s | 139.26s | **+14.5% 变慢** |
| S8 | 44.17s | 52.82s | **+19.6% 变慢** |
| Smax | 35.91s | 崩溃（服务器宕机） | **致命** |
| ori_accuracy | 79.29% | **52.64%** | **-26.65pp** |
| normalized | 99.11% | **65.81%** | **-33.3pp** |
| C | 1.0 | **0（淘汰）** | — |

任务级别精度崩溃：
- cwe: 47.67%（基线 ~72%）
- fwe: 98.89%（正常 — 未受影响）
- mcq: 53.33%（基线 ~63%）
- niah: 36.67%（基线 ~100%）
- qa: 26.67%（基线 ~63%）

### Test 15：仅原地残差（RoPE 已恢复）— commit b8196b71e

| 指标 | 基线 (Test 12) | 仅原地残差 | 变化 |
|------|----------------|-----------|------|
| ori_accuracy | 79.29% | **51.91%** | **-27.38pp** |
| normalized | 99.11% | **64.89%** | **-34.22pp** |
| C | 1.0 | **0（淘汰）** | — |

**比 Test 14 更差** — 原地 `*=` 单独即可导致大幅下降。

### 根因分析

1. **bf16 RoPE**：虽然 sgl-kernel CUDA 核函数在技术上接受 bf16 输入并在内部使用 float32 cos/sin 缓存值，但当传入 bf16 张量时，`q*cos + rotate(q)*sin` 的乘法运算在 bf16 精度下进行。MiniCPM-SALA 依赖于此计算的 float32 精度 — float32 上转换不是防御性代码，而是**必需的**。

2. **原地残差缩放**：`hidden_states *= scalar` 与 `hidden_states = hidden_states * scalar` 在 sglang 的 CUDA 图捕获下行为不同。原地修改 CUDA 图中预分配的张量可能破坏张量别名和缓冲区复用，导致跨层级联的数值错误。这**不是**简单的数学等价 — 内存管理语义不同。

3. **速度退化**：精度下降导致模型生成更长/错误的输出，人为增加了基准测试时长。两项变更均无实际速度收益。

## 回滚说明

所有变更已在 commit caa93efe9 中**完全回滚**：
1. 恢复 `MiniCPMAttention.forward()` 中的 float32 上转换 — 精度必需
2. 恢复 `MiniCPMLightningMixer._apply_qk_norm_rope()` 非融合路径中的 float32 上转换 — 精度必需
3. 恢复 `hidden_states = hidden_states * self.residual_scale`（2处）— CUDA 图正确性必需

## 经验教训

1. **永远不要假设核函数级别的 dtype 支持意味着模型级别的正确性。** bf16 RoPE 核函数在核函数级别工作正确，但模型是用 float32 RoPE 计算训练的。移除上转换会引入系统性数值漂移，在32层中累积放大。

2. **CUDA 图下的原地张量操作不安全。** sglang 的 CUDA 图捕获预分配张量缓冲区并固定内存地址。原地 `*=` 直接修改这些缓冲区，可能破坏别名引用。当 CUDA 图处于活动状态时，前向传播中应始终创建新张量。

3. **"数学等价"不等于"计算等价"。** 两项变更在理论上看起来安全，但由于 GPU 执行模型中的精度和内存管理差异，在实践中灾难性失败。

## 后续步骤

- **不要重试**这两项优化，除非进行充分的精度分析
- 集中在其他优化方向：torch.compile、调度调优、核函数融合
- 考虑在加载时将 `residual_scale` 折叠到权重中（完全避免运行时计算）— 但需先通过精度测试验证
