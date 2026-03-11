# CHANGE_0002_sparse_seqlen_nonzero_reduction

## 1）背景与动机
- 问题描述：在 MiniCPM 稀疏注意力的 extend 路径中，长上下文+并发评测触发了显存 OOM。
- 触发位置：`python/sglang/srt/layers/attention/minicpm_backend.py` 中 sparse cache seqlen 统计逻辑附近。
- 预期收益：降低该统计步骤的临时张量显存占用。

## 2）SOAR 规则合规性检查
- 合规原因：属于推理引擎运行时内存优化。
- 不触碰限制：不涉及 prefix cache 违规手段，不改官方并发评测协议。
- 可复现性：等价语义替换，行为可追踪。

## 3）改动前计划
- 仅做一个聚焦改动：
  - 将 `(metadata.sparse_page_table != 0).sum(dim=1)`
  - 替换为 `torch.count_nonzero(metadata.sparse_page_table, dim=1)`
- 保持后续 dtype/device 转换不变。
- 回滚方式：恢复该单行表达式。

## 4）速度影响评估（记录用户问答）
- 预期推理速度影响：基本无明显变慢。
- 原因：`count_nonzero` 可避免先显式构造大布尔临时张量再做求和。
- 实际预期：
  - Decode 吞吐：基本不变。
  - Prefill 长上下文：通常不变或略有改善。
  - 主要收益：峰值显存更低，OOM 中断概率下降。
- 风险说明：仍可能存在很小的随机波动，需要通过 A/B 多次重复确认。

## 5）实际代码改动（获批后）
- 修改文件：
  - `python/sglang/srt/layers/attention/minicpm_backend.py`
- 逻辑变化：
  - 旧：布尔临时张量 + `sum`。
  - 新：直接 `count_nonzero` 统计。

## 6）验证命令（fcloud）
### 正确性并发爬坡
```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path /root/models/openbmb/MiniCPM-SALA \
  --data_path <DATA_DIR>/perf_public_set.jsonl \
  --concurrency 8

python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path /root/models/openbmb/MiniCPM-SALA \
  --data_path <DATA_DIR>/perf_public_set.jsonl \
  --concurrency 16

python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path /root/models/openbmb/MiniCPM-SALA \
  --data_path <DATA_DIR>/perf_public_set.jsonl \
  --concurrency 32
```

### A/B 速度回归检查
```bash
python3 benchmark/soar/run_soar_suite.py \
  --api-base http://127.0.0.1:30000 \
  --model-path /root/models/openbmb/MiniCPM-SALA \
  --speed-data-s1 <S1_JSONL> \
  --speed-data-s8 <S8_JSONL> \
  --speed-data-smax <SMAX_JSONL> \
  --num-prompts 64
```

## 7）结果汇总
| 指标 | 基线 | 新方案 | 变化 |
|---|---:|---:|---:|
| Eval 阶段 OOM 次数 |  |  |  |
| Accuracy / overall_accuracy |  |  |  |
| S1 benchmark_duration (s) |  |  |  |
| S8 benchmark_duration (s) |  |  |  |
| S∞ benchmark_duration (s) |  |  |  |

## 8）风险评估
- 准确率风险：很低（计数语义等价）。
- 稳定性风险：低。
- 性能风险：低；若有变化，通常应在噪声范围内，除非原先已被显存瓶颈限制。

## 9）回滚说明
1. 回退 `python/sglang/srt/layers/attention/minicpm_backend.py` 的该单行改动。
2. 重新执行 8/16/32 并发正确性爬坡验证回退效果。

## 10）下一步建议
- 若仍出现 OOM，优先做调度参数保护（`--prefill-max-requests`、`--max-running-requests`），再考虑更深层内核改动。
