# CHANGE_0100: EAGLE3 草稿模型训练脚本

## 背景与动机

EAGLE3 推测性解码是 SOAR 2026 竞赛中影响最大的优化路径。排名前两位的团队（#1 FlashSALA 94.29分，#2 智算一队 90.22分）很可能使用了推测性解码。我们的第一阶段（推理基础设施）已在 `eagle3-spec-decode` 分支完成——本次变更实现第二阶段：训练草稿头。

草稿模型是一个轻量级单层 Transformer，它通过中间层捕获的辅助隐藏状态预测目标模型的输出分布，可实现 2-4 倍的解码加速，且输出在数学上完全准确（拒绝采样）。

## 规则合规性

- **推测头（speculative heads）明确被竞赛规则允许**（计入 2GB 限制）
- 草稿模型约 586MB FP16（约 293M 参数）——远在 2GB 总限制内
- 无违规技巧——标准知识蒸馏训练
- Apache 2.0 兼容
- 推测性解码精确保持输出分布（C = 1.0）

## 实现方案

### 训练脚本：`benchmark/soar/demo_sala/train_eagle3_draft.py`

**架构**（与 sglang 推理端 `MiniCPMForCausalLMEagle3` 一致）：
- `FC(3×4096 → 4096)`：融合 3 个辅助隐藏状态
- `DraftDecoderLayer`：标准 QKV 注意力（2×H 输入）+ SiLU 门控 MLP
- `RMSNorm`：最终归一化
- 共享目标模型的 `embed_tokens` 和 `lm_head`（冻结参数）

**训练流程**：
1. 使用 HuggingFace transformers 加载目标模型（MiniCPM-SALA BF16）
2. 在第 [2, 16, 29] 层注册前向钩子以捕获隐藏状态
3. 每个 batch：
   - 目标模型前向 → 获取 logits + 3 个捕获的隐藏状态
   - 拼接隐藏状态 → [bsz, seq, 3×4096]
   - 草稿模型前向 → 获取草稿 logits
   - 目标与草稿分布的 KL 散度损失
4. 以 safetensors 格式保存草稿模型

**关键参数**：
- 捕获层：[2, 16, 29]（32 层模型的默认值）
- 可训练参数：~293M（FC: 50M, QKV: 50M, O: 17M, MLP: 176M）
- 训练配置：AdamW, lr=1e-4, 余弦退火, 1000 步
- 数据：perf_public_set.jsonl（150 个样本）

## 验证命令

```bash
# 在 fcloud 上训练草稿模型
cd /root/sglang-minicpm/benchmark/soar/demo_sala
python3 train_eagle3_draft.py \
    --model-path /root/models/openbmb/MiniCPM-SALA-Copy \
    --data-path /root/data/perf_public_set.jsonl \
    --output-path /root/models/eagle3_draft_minicpm \
    --num-steps 1000 --lr 1e-4 --batch-size 1

# 推测性解码测试（添加到 prepare_env.sh）
--speculative-algorithm EAGLE3 \
--speculative-draft-model-path /root/models/eagle3_draft_minicpm \
--speculative-num-steps 5 \
--speculative-num-draft-tokens 64 \
--speculative-eagle-topk 1
```

## 结果摘要

| 指标 | 基线（无推测） | 预期（EAGLE3） |
|------|---------------|----------------|
| S1 | 113.67s | ~40-60s |
| S8 | 41.07s | ~15-25s |
| Smax | 34.15s | ~15-20s |
| 准确率 | 80.64% | 相同（精确） |
| C | 1.0 | 1.0 |

## 回滚方案

从 `prepare_env.sh` 中移除 EAGLE3 相关服务器参数即可。基础模型在没有草稿模型的情况下正常运行。

## 后续步骤

1. **在 fcloud 上运行训练** — 需要 BF16 模型 + GPU
2. **调优超参数** — 学习率、步数、温度
3. **端到端测试** — 使用训练好的草稿模型进行推测性解码
4. **优化接受率** — 尝试不同的捕获层、更多训练数据
5. **多步训练** — 使用草稿模型自身的隐藏状态训练（第三阶段）
