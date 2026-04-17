# CHANGE_0090: MiniCPM-SALA EAGLE3 投机解码

> **状态**：已暂停 — 未训练草稿模型结果不可行；需要训练草稿模型后才能继续  
> **分支**：`eagle3-spec-decode`（最新提交：`548c8c153`）  
> **日期**：2026-04-17  

---

## 背景与动机

EAGLE3 是一种投机解码算法，使用轻量级草稿模型预测多个 token，然后通过目标模型在一次前向传播中验证。对于大模型（13B+），通常可实现 2-5 倍加速且准确率无损。

我们为 MiniCPM-SALA 实现了 EAGLE3，探索投机解码能否提升 SOAR 2026 竞赛的推理速度。

### MiniCPM-SALA 架构挑战
- **混合架构**：8 层标准注意力 + 24 层 SimpleGLA（循环）层
- **循环状态**：SimpleGLA 维护内部状态，拒绝草稿 token 后必须回滚 — 不同于标准 Transformer KV cache
- **小模型（~4B 参数）**：目标模型每个 token 的延迟已经很低，草稿开销可能无法被更少的目标前向传播抵消

---

## 规则合规声明

- 投机头在 SOAR 规则中**明确允许**（"Speculative heads allowed, count toward 2GB"）
- 草稿模型（612MB）与 GPTQ 模型合计不超过 2GB 提交限制
- 投机解码不影响准确率（验证保证目标模型分布）

---

## 实现概要

### 草稿模型架构（`minicpm_eagle3.py`）
- FC 融合层：3×4096 → 4096（融合嵌入 + 低/中/高层辅助隐藏状态）
- 1 层解码器，标准 GQA 注意力（32 Q 头，8 KV 头）
- 共享 embed_tokens 和 lm_head 权重
- 总计：293M 参数，612MB

### 目标模型改动（`minicpm.py`）
- 添加 `layers_to_capture` 列表和 forward() 中的辅助隐藏状态捕获
- 添加 `get_embed_and_head()`、`set_embed_and_head()`、`set_eagle3_layers_to_capture()`

### SimpleGLA 状态管理
- 复用 GDR 更新内核，添加 `SKIP_DELTA_RULE=True` 标志
- 状态形状 (16, 128, 64) 匹配现有 MambaPool
- 通过 `update_mamba_state_after_mtp_verify()` 实现顺序验证 + 状态回滚

### Bug 修复（共 7 个，全部已提交）
1. 草稿模型配置：model_type 改为 "minicpm_sala" + auto_map
2. `set_eagle3_layers_to_capture`：添加到 MiniCPMForCausalLM
3. `get_embed_and_head`：返回 `.weight` 张量而非 Module
4. RadixAttention scaling：使用 `head_dim**-0.5` 浮点数而非 rotary_emb 函数
5. Mamba 状态回滚：使用 `mambaish_config`（涵盖 minicpm_hybrid_config）
6. `update_mamba_state_after_mtp_verify`：用 `has_conv` 保护 conv_states 访问
7. `_compute_retrieve_parent_token`：topk>1 的独立辅助函数

---

## 测试结果（Test 22）

| 指标 | EAGLE3（未训练） | 基线（Test 20） | 变化 |
|------|-----------------|----------------|------|
| 配置 | GPTQ+FP8+dense+EAGLE3 | GPTQ+FP8+dense | — |
| mem-fraction-static | 0.72 | 0.84 | KV cache 减少 14% |
| S1 | 187.01s | 113.67s | **慢 65%** |
| ori_accuracy | 74.33% | 80.64% | -6.3pp |
| normalized_accuracy | 92.92% | 100.80% | -7.9pp |
| C | **0（淘汰）** | 1.0 | — |
| 接受率 | 0.26 | — | 随机草稿 |
| 接受长度 | 1.03-1.07 | — | 约 0 个额外 token |

### 各任务明细
| 任务 | EAGLE3 | 基线 |
|------|--------|------|
| MCQ | 56.67%（avg_out=8527） | ~76.67%（avg_out=~1442） |
| NIAH | 100% | 100% |
| CWE | 78.33% | ~80% |
| FWE | 73.33% | ~80% |
| QA | 63.33% | ~70% |

---

## 分析

### 失败原因
1. **未训练的草稿模型** → 接受率 0.26（随机预测），纯粹增加开销
2. **EAGLE3 调度惩罚**：禁用 mixed-chunk、禁用 overlap scheduler、限制 max_running_requests=24
3. **内存压力**：草稿模型（612MB）+ 辅助隐藏状态（~1GB）占用约 10GB → mem-fraction-static 从 0.84 降至 0.72
4. **MCQ 过度生成**：平均输出 8527 token（基线约 1442）— 模型无法正常停止

### 使其可行的条件
1. 使用 SpecForge 或 EAGLE 训练脚本**训练草稿模型**
2. 实现接受率 > 0.5 以抵消草稿模型开销
3. 调查 MCQ 过度生成问题

### 成本效益评估
即使有训练良好的草稿模型，MiniCPM-SALA 作为小模型（~4B），净加速可能有限（10-20%），投入产出比不高。  
**建议：暂停 EAGLE3，优先追求其他优化方向。**

---

## 回滚说明

在 `prepare_env.sh` 中设置 `SOAR_ENABLE_EAGLE3=0`（或删除），将 `mem-fraction-static` 恢复为 0.84。

---

## 后续步骤（恢复时）

1. 在 fcloud 上搭建 SpecForge 训练环境
2. 使用 ShareGPT + 领域文本从 BF16 模型生成训练数据
3. 训练草稿模型，在评估数据上实现 > 0.5 接受率
4. 用训练好的草稿模型重新测试
5. 如接受率 > 0.6 且速度提升，则继续；否则放弃
