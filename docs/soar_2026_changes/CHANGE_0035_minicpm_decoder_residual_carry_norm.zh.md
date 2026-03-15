# CHANGE_0035_minicpm_decoder_residual_carry_norm

## 1）背景与动机
- 问题描述：当前 MiniCPM decoder 会在每个 RMSNorm 之前显式物化 residual add，而运行时其实已经具备 fused add+RMSNorm kernel，可以把待合并的残差状态跨层传递下去。
- 为什么该改动预期会提速：把缩放后的分支输出保留在 `hidden_states`，把累计 skip 路径保留在 `residual`，可以减少每层两次独立 residual add 的物化，并让两个 norm 入口都走现有 fused kernel 路径。
- 目标阶段：decode / layer execution / pointwise fusion

## 2）SOAR 规则合规性检查
- 改动前已重新查阅最新官方页面：`https://soar.openbmb.cn/competition` 与 `https://soar.openbmb.cn/toolkit`。
- 允许原因：比赛页面明确允许算子融合、decode 路径调优等推理优化；toolkit 的 `技术路径指引` 也鼓励围绕官方 MiniCPM-SALA 基座进行运行时优化。
- 不触碰限制：不替换基座模型、不启用 prefix cache、不修改并发配置、不改变提交接口，也不引入不可复现行为。
- 对正确性系数 C 的预期影响：理论上中性；补丁保持原有 `scale_depth / sqrt(num_hidden_layers)` 残差代数，只改变加法的物化时机。

## 3）改动前实施计划
- 计划修改的文件/函数：`python/sglang/srt/models/minicpm.py`，重点是 `MiniCPMDecoderLayer.forward()` 与 `MiniCPMModel.forward()`。
- 最小化 diff 策略：直接复用现有 `RMSNorm(x, residual)` fused 路径，不新增 kernel；attention、MLP 和外部接口保持不变。
- 延后范围：本特性先不改 q/k 的 RoPE cast 路径，避免多个因素混在一起影响归因。
- 回滚方案：恢复原先 eager residual add 逻辑，并取消跨层 residual carry。

## 4）实际代码改动（获批后填写）
- 补丁摘要：`MiniCPMDecoderLayer` 现在把累计 skip connection 保存在 `residual` 中，把缩放后的分支输出传给 fused `RMSNorm(..., residual)`，并把最终待合并的 residual 对传给下一层，而不是立刻物化求和。
- 最终修改文件：`python/sglang/srt/models/minicpm.py`。
- 核心逻辑变化：
  - 当上一层已经有 residual carry 时，decoder 第一处 norm 改为与其他 fused decoder 实现一致的 residual-carry 入口。
  - post-attention norm 直接通过 `RMSNorm` 融合 `residual + attn_output * residual_scale`。
  - 模型最后一层 norm 会消费最后一组待合并的 `(hidden_states, residual)`，从而保持与原 eager-add 路径数学等价。

## 5）验证命令
### 正确性
```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_PATH> \
  --data_path benchmark/soar/demo_sala/perf_public_set.jsonl \
  --concurrency 32
```

### 速度
```bash
export SPEED_DATA_S1=<shared_representative_speed_set.jsonl>
export SPEED_DATA_S8=<shared_representative_speed_set.jsonl>
export SPEED_DATA_SMAX=<shared_representative_speed_set.jsonl>
python3 benchmark/soar/run_soar_suite.py \
  --base-url http://127.0.0.1:30000 \
  --dataset-profile heavy
```

### 本地语法检查
```bash
python3 -m compileall python/sglang/srt/models/minicpm.py
```

## 6）结果汇总
| 指标 | 基线 | 新方案 | 变化 |
|---|---:|---:|---:|
| Accuracy / overall_accuracy | pending | pending | pending |
| 共享数据 S1 吞吐 | pending | pending | pending |
| 共享数据 S8 吞吐 | pending | pending | pending |
| 共享数据 S∞ 吞吐 | pending | pending | pending |

## 7）风险评估
- 准确率风险：中低；目标代数不变，但 residual 现在跨层传递并由 fused norm 消费，仍需做功能验证。
- 稳定性风险：低；该补丁复用了其他 decoder 模型已在使用的 RMSNorm fused 行为。
- 可复现风险：低；没有新增随机逻辑，也没有新增依赖环境差异的启发式路径。

## 8）回滚说明
1. 恢复 `MiniCPMDecoderLayer.forward()` 中在每个 norm/分支切换前的 eager residual add。
2. 停止从 decoder layer 返回 residual carry，并把 `MiniCPMModel.forward()` 恢复为仅执行 `self.norm(hidden_states)` 收尾。

## 9）下一步建议
- 在共享数据集上重新跑 S1/S8/Smax，确认减少 pointwise traffic 后是否能带来可见的 decode 收益。
- 如果收益仍然较小，再 profiling 判断 MiniCPM 剩余瓶颈是否主要来自 attention 侧 dtype 转换或 backend planning，而不是 decoder residual 处理。