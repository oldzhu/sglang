# CHANGE_0001_local_eval_benchmark_harness

## 1）背景与动机
- 问题描述：当前正确性与速度评测依赖手工执行与人工整理日志，迭代效率低且易出错。
- 为什么该改动预期会提速：通过统一脚本自动化执行与汇总，缩短每轮实验的反馈周期，减少无效提交。
- 目标阶段：调度与评测工作流（不涉及运行时内核/推理引擎逻辑）。

## 2）SOAR 规则合规性检查
- 允许原因：该改动仅用于本地评测流程自动化，不改变模型或服务推理行为。
- 不触碰限制：不启用违规 prefix cache，不修改官方固定并发规则，保留可复现日志与摘要。
- 对正确性系数 C 的预期影响：无直接影响（仅提升观测与回归控制能力）。

## 3）改动前实施计划
- 计划修改文件：
  - `benchmark/soar/run_soar_suite.py`（新增）
  - 本次中英文变更文档
- 最小化 diff 策略：只新增独立脚本和文档，不修改推理引擎源码。
- 回滚方案：删除新增脚本和文档对。

## 4）实际代码改动（获批后）
- 补丁摘要：
  - 新增 `benchmark/soar/run_soar_suite.py`。
  - 提供一键本地评测流程：
    - 可选调用 toolkit `eval_model.py` 做正确性评测。
    - 使用 `python -m sglang.bench_serving` 跑 S1/S8/S∞ 速度。
    - 自动将 SOAR 速度数据格式（`question` + `model_response`）转换为 bench custom 格式。
    - 输出结构化 `summary.json` 与各档日志。
- 最终修改文件：
  - `benchmark/soar/run_soar_suite.py`
  - `docs/soar_2026_changes/CHANGE_0001_local_eval_benchmark_harness.en.md`
  - `docs/soar_2026_changes/CHANGE_0001_local_eval_benchmark_harness.zh.md`
- 核心逻辑变化：仅统一命令与结果采集，不改变模型推理路径。

## 5）验证命令
### 正确性 + 速度（单命令）
```bash
python3 benchmark/soar/run_soar_suite.py \
  --api-base http://127.0.0.1:30000 \
  --model-path /root/models/openbmb/MiniCPM-SALA \
  --eval-script /path/to/SOAR-Toolkit/eval_model.py \
  --public-data /path/to/perf_public_set.jsonl \
  --speed-data-s1 /path/to/s1.jsonl \
  --speed-data-s8 /path/to/s8.jsonl \
  --speed-data-smax /path/to/smax.jsonl
```

### 仅速度
```bash
python3 benchmark/soar/run_soar_suite.py \
  --api-base http://127.0.0.1:30000 \
  --model-path /root/models/openbmb/MiniCPM-SALA \
  --speed-data-s1 /path/to/s1.jsonl \
  --speed-data-s8 /path/to/s8.jsonl \
  --speed-data-smax /path/to/smax.jsonl
```

## 6）结果汇总
| 指标 | 基线 | 新方案 | 变化 |
|---|---:|---:|---:|
| Accuracy / overall_accuracy | N/A | N/A | N/A |
| S1 benchmark_duration (s) | N/A | N/A | N/A |
| S8 benchmark_duration (s) | N/A | N/A | N/A |
| S∞ benchmark_duration (s) | N/A | N/A | N/A |

说明：
- 本变更仅为流程工具，不宣称直接带来推理加速。
- 待你在 fcloud 执行后回填 baseline/new 数据。

## 7）风险评估
- 准确率风险：无（未改推理逻辑）。
- 稳定性风险：低；若输入数据格式不匹配会报错。
- 可复现风险：低；脚本统一记录日志与 summary，反而提升可复现性。

## 8）回滚说明
1. 删除 `benchmark/soar/run_soar_suite.py`。
2. 删除本次中英文文档对。

## 9）下一步建议
- 先用该脚本固化 baseline（S1/S8/S∞ + 正确性），再进入首个引擎级优化（注意力后端微优化）并做严格 A/B 对比。
