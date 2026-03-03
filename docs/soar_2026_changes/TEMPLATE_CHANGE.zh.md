# CHANGE_XXXX_<short_title>

## 1）背景与动机
- 问题描述：
- 为什么该改动预期会提速：
- 目标阶段：（prefill / decode / memory / kernel launch / scheduling）

## 2）SOAR 规则合规性检查
- 允许原因：
- 不触碰限制（prefix cache/固定并发/可复现性）：
- 对正确性系数 C 的预期影响：

## 3）改动前实施计划
- 计划修改的文件/函数：
- 最小化 diff 策略：
- 回滚方案：

## 4）实际代码改动（获批后填写）
- 补丁摘要：
- 最终修改文件：
- 核心逻辑变化：

## 5）验证命令
### 正确性
```bash
# 示例
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_PATH> \
  --data_path <DATA_PATH>/perf_public_set.jsonl \
  --concurrency 32
```

### 速度
```bash
# 示例
export SPEED_DATA_S1=<path_to_s1.jsonl>
export SPEED_DATA_S8=<path_to_s8.jsonl>
export SPEED_DATA_SMAX=<path_to_smax.jsonl>
bash bench_serving.sh http://127.0.0.1:30000
```

## 6）结果汇总
| 指标 | 基线 | 新方案 | 变化 |
|---|---:|---:|---:|
| Accuracy / overall_accuracy |  |  |  |
| S1 benchmark_duration (s) |  |  |  |
| S8 benchmark_duration (s) |  |  |  |
| S∞ benchmark_duration (s) |  |  |  |

## 7）风险评估
- 准确率风险：
- 稳定性风险：
- 可复现风险：

## 8）回滚说明
1.
2.

## 9）下一步建议
-
