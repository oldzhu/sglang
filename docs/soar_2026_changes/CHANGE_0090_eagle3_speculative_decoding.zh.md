# CHANGE_0090: MiniCPM-SALA EAGLE3 投机解码

## 背景与动机

MiniCPM-SALA 使用 24 层 SimpleGLA (lightning attention) 作为循环主干。
当前瓶颈是解码速度——每个 token 需要通过全部 32 层（8 层 MLA + 24 层 SimpleGLA）完整前向传播。
EAGLE3 投机解码可以用轻量级草稿模型生成多个候选 token 并行验证，有望实现 2-4 倍 S1 加速。

团队距离前 5 差距约 40%（team-beta 得分 52.94 vs 第 5 名 ≥79.41）。
配置调优无法弥合此差距。EAGLE3 是通向前 5 的唯一可行战略路径。

## 规则合规性说明

- 工具包明确允许 EAGLE3（"投机头允许，计入 2GB"）
- 草稿模型权重（~300M 参数 FP16 = ~600MB）在 2GB 限制内
- 现场训练在 5 小时内可行
- 不预期精度下降（验证步骤确保与目标模型精确匹配）

## 实施计划

### 阶段 1：基础设施（已完成 - 提交 4059c2420）

#### 目标模型修改 (minicpm3.py)
- MiniCPM3Model 添加 `layers_to_capture` 列表用于中间隐藏状态捕获
- 修改前向循环在可配置层索引处收集 `aux_hidden_states`
- MiniCPM3ForCausalLM 添加 `capture_aux_hidden_states` 标志
- 添加 `get_embed_and_head()`、`set_embed_and_head()`、`set_embed()` 用于权重共享
- 添加 `set_eagle3_layers_to_capture(layer_ids)` — 默认 [2, num_layers//2, num_layers-3]

#### 草稿模型 (minicpm_eagle3.py — 新文件)
- `MiniCPMEagle3Attention`：标准 QKV 注意力（非 MLA），输入 2×hidden_size
- `MiniCPMEagle3MLP`：标准 SiLU+Mul MLP
- `MiniCPMEagle3DecoderLayer`：embeds 上 input_layernorm，hidden_states 上 hidden_norm
- `MiniCPMEagle3Model`：embed_tokens、FC(3×hidden_size → hidden_size)、1 层解码器、norm
- `MiniCPMForCausalLMEagle3`：完整草稿模型

#### GDR 内核 SKIP_DELTA_RULE (fused_recurrent.py)
- 添加 `SKIP_DELTA_RULE: tl.constexpr` 到更新内核
- True 时：跳过 delta rule 减法和 beta 缩放
- 将 GDR 递归简化为 SimpleGLA：`h = exp(g)·h + k⊗v`

#### SimpleGLA 验证路径 (hybrid_linear_attn_backend.py)
- `SimpleGLAAttnBackend.forward()`：添加 `is_target_verify` 分支
- `update_mamba_state_after_mtp_verify()`：为无 conv 模型添加 `has_conv` 检查
- `_compute_retrieve_parent_token()`：独立计算父 token 索引

### 阶段 2：草稿模型训练（待定）
### 阶段 3：端到端测试（待定）

## 验证命令

同英文文档。

## 预期结果

| 指标 | 基线（无投机解码）| 预期（EAGLE3）|
|------|-----------------|--------------|
| S1   | ~113s           | ~45-60s (2-2.5x) |
| S8   | ~42s            | ~25-35s (1.5-2x)  |
| Smax | ~36s            | ~25-30s (1.2-1.4x)|
| 精度 | 79.29% (C=1.0)  | 79.29% (C=1.0)    |

## 回滚说明

EAGLE3 修改在独立分支 `eagle3-spec-decode`。回滚：
```bash
git checkout mixed_minicpm_cudagraph
```

## 后续步骤

1. 创建草稿模型训练脚本
2. 在 fcloud A800 上训练草稿模型
3. EAGLE3 端到端测试
4. 调优 num_steps、num_draft_tokens、topk
5. 验证模式 CUDA graph 支持
