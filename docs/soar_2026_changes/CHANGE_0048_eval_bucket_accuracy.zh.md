# CHANGE_0048 评测分桶准确率诊断

## 背景与动机

在当前 GPTQ 排查阶段，仅看总的 `ori_accuracy` 已经不够细。我们需要知道量化后的退化究竟集中在哪些任务类型、哪些输入长度区间，或者两者的交叉区域。这个改动新增一个诊断版评测脚本，在保持原有兼容输出不变的前提下，补充分桶准确率统计。

## 规则合规说明

本改动仅用于诊断分析，符合当前 SOAR 2026 规则。

- 不修改模型本身，不改变线上推理路径，也不替换官方评测逻辑。
- 保留原有 `ori_accuracy` 与 `overall_accuracy` 输出兼容性。
- 仅用于本地/公开集评测分析，辅助校准集设计。

## 详细实施计划

改动前计划：

- 将 `benchmark/soar/demo_sala/eval_model.py` 复制为新的变体文件。
- 保持总准确率相关 stdout 与 summary 输出兼容。
- 新增按任务、按长度桶、按任务与长度交叉桶的聚合统计。
- 将这些诊断结果同时输出到控制台和 `summary.json`。

## 实际代码改动

- 新增 `benchmark/soar/demo_sala/eval_model_001.py`。
- 保留兼容输出：
  - stdout 中的 `Average Score`
  - `summary.json` 中的 `ori_accuracy`
  - `summary.json` 中的 `overall_accuracy`
- 新增三类统计：
  - `task`
  - `length_bucket`
  - `task_length_bucket`
- 在逐样本输出中新增字段：
  - `length_bucket`
  - `task_length_bucket`
- 在 `summary.json` 中新增机器可读的 `bucket_accuracy`，并在控制台打印单行 JSON，方便后续抓日志做对比。

长度桶边界与校准分析保持一致：

- `len_0_4k`
- `len_4k_32k`
- `len_32k_128k`
- `len_128k_plus`

## 验证命令

```bash
python3 -m py_compile benchmark/soar/demo_sala/eval_model_001.py
python3 benchmark/soar/demo_sala/eval_model_001.py --help
```

示例运行：

```bash
python3 benchmark/soar/demo_sala/eval_model_001.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_DIR> \
  --data_path benchmark/soar/demo_sala/perf_public_set.jsonl \
  --concurrency 32
```

## 结果汇总表

| 版本 | 总准确率兼容输出 | 分桶诊断 | 状态 |
|---|---|---|---|
| 原始 `eval_model.py` | 是 | 否 | 已有 |
| 新增 `eval_model_001.py` | 是 | 是 | 已新增 |

## 回滚说明

- 继续使用原始的 `benchmark/soar/demo_sala/eval_model.py` 即可。
- 如果不再需要诊断功能，可忽略或删除 `benchmark/soar/demo_sala/eval_model_001.py`。
- 本改动不涉及运行时或预处理回退，因为它只是独立的本地评测辅助工具。

## 下一步建议

1. 用当前 sequential calibration 集和新的 semantic-priority 集分别跑同一套评测。
2. 比较哪些任务桶、长度桶先恢复准确率。
3. 根据这些桶的变化，决定下一轮 calibration 应该补更多 `qa`、更多长上下文样本，还是调整整体配比。