# CHANGE_0051 快速校准筛选评测

## 背景与动机

现在每个 calibration 候选做一次完整公开集评测大约需要 1 小时，手工试错的成本已经过高。前面的 bucket 分析也已经说明，当前最主要的不稳定点集中在 `mcq|len_0_4k`、`qa|len_4k_32k` 和 `qa|len_32k_128k`。因此更有价值的下一步，是构建一个更小但固定的公开集 micro-eval，在尽量保留这些弱桶信号的同时，显著缩短试验周转时间。

## 规则合规说明

本改动符合当前 SOAR 2026 规则。

- 仅读取公开集 `perf_public_set.jsonl` 中的样本。
- 不修改模型权重、运行时内核、并发规则或官方评测逻辑。
- 本质上是一个离线诊断与排序工具，用来决定哪些 calibration 候选值得再做完整评测。

## 详细实施计划

改动前计划：

- 定义一个固定的 20 条公开集子集，聚焦当前弱桶。
- 不复制第二份大 JSONL 数据，而是通过索引配置文件保证子集可复现。
- 复用现有 `eval_model_001.py` 的生成和打分逻辑，保证 quick-screen 与完整评测尽量同向。
- 同时输出整体子集准确率和弱桶聚焦指标。

## 实际代码改动

- 新增 `benchmark/soar/demo_sala/quick_screen_public_subset.json`。
- 新增 `benchmark/soar/demo_sala/quick_calibration_screen.py`。

子集设计：

- 总计 20 条公开样本
- 固定索引：`1, 2, 4, 8, 11, 17, 23, 25, 61, 63, 65, 66, 68, 70, 71, 76, 80, 81, 85, 90`
- 目标 focus buckets：
  - `task=mcq|len_0_4k`
  - `task=qa|len_4k_32k`
  - `task=qa|len_32k_128k`

quick-screen 输出包括：

- `quick_screen_accuracy`：20 条子集上的原始准确率
- `focus_bucket_average`：三个目标 task-length bucket 的平均准确率
- `focus_bucket_min`：最弱目标 bucket 的准确率
- 按 task、length bucket、task-length bucket 的统计表

脚本还支持 `--check`，可以在不调用模型的情况下校验固定子集配置。

## 验证命令

```bash
python3 -m py_compile benchmark/soar/demo_sala/quick_calibration_screen.py
python3 benchmark/soar/demo_sala/quick_calibration_screen.py --check
python3 benchmark/soar/demo_sala/quick_calibration_screen.py \
  --model_path openbmb/MiniCPM-SALA \
  --api_base http://127.0.0.1:30000
```

## 结果汇总表

| 产物 | 用途 | 状态 |
|---|---|---|
| `quick_screen_public_subset.json` | 固定可复现的 20 条 quick-screen 定义 | 新增 |
| `quick_calibration_screen.py` | 快速的 bucket-aware 代理评测器 | 新增 |

## 回滚说明

- 停止使用 `quick_calibration_screen.py`，继续只用 `eval_model_001.py`。
- 如果更偏好手工或完整公开集评测，可以直接忽略该 subset 配置。
- 该改动只新增离线工具，不涉及模型或运行时，因此无需做运行时回滚。

## 下一步建议

1. 先对 2 到 4 组最强的 calibration 候选跑 quick screen。
2. 只把 quick-screen 表现最好的候选提升到 1 小时完整评测。
3. 如果 quick-screen 排名和完整评测相关性不足，下一步应该升级到 BF16-vs-GPTQ 输出漂移筛选，而不是继续手工猜子集。