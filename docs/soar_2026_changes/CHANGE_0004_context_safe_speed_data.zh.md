# CHANGE_0004_context_safe_speed_data

## 1）背景与动机
- 问题描述：生成的速度测试样本在 warmup 阶段可能超过模型上下文上限，导致 benchmark 失败。
- 现象：出现序列长度超限告警/错误，然后 bench 中断。
- 目标：让生成的数据在构造时就满足上下文安全约束。

## 2）SOAR 规则合规性检查
- 本改动仅针对本地 benchmark 工具链。
- 不涉及模型权重、算子内核或违规评测行为。

## 3）改动前计划
- 仅修改 `benchmark/soar/generate_speed_datasets.py`。
- 增加上下文预算参数并对每条样本做长度约束。
- 输出每个数据文件的长度统计便于快速校验。

## 4）实际代码改动
- 新增参数：
  - `--max-context-tokens`（默认 `262144`）
  - `--safety-margin`（默认 `4096`）
  - `--min-output-tokens`（默认 `64`）
  - `--max-output-cap`（默认 `20000`）
- 对每条样本执行预算约束：
  - 保证 `input_tokens + output_tokens <= max_context_tokens - safety_margin`
- 为每个输出文件打印统计：
  - 最大输入长度、最大输出长度、最大总长度、预算值。

## 5）验证命令
```bash
python3 benchmark/soar/generate_speed_datasets.py --output-dir /root/soar_test_data
python3 benchmark/soar/generate_speed_datasets.py --output-dir /root/soar_test_data --max-context-tokens 262144 --safety-margin 4096
```

然后先做单档位验证：
```bash
python3 benchmark/soar/run_soar_suite.py \
  --api-base http://127.0.0.1:30000 \
  --model-path /root/models/openbmb/MiniCPM-SALA \
  --speed-data-s1 /root/soar_test_data/speed_s1.jsonl \
  --num-prompts 32
```

## 6）结果汇总
| 指标 | 基线 | 新方案 | 变化 |
|---|---:|---:|---:|
| Warmup/上下文超限失败次数 |  |  |  |
| S1 benchmark 完成情况 |  |  |  |
| S8 benchmark 完成情况 |  |  |  |
| S∞ benchmark 完成情况 |  |  |  |

## 7）风险评估
- 风险很低。需要注意生成数据依旧是合成分布，不等价于官方隐藏数据集。

## 8）回滚说明
1. 回退 `benchmark/soar/generate_speed_datasets.py`。
2. 按旧逻辑重新生成数据。

## 9）下一步建议
- 如有需要，可在下一次单独变更中加入 `quick / balanced / heavy` 预设档位。
