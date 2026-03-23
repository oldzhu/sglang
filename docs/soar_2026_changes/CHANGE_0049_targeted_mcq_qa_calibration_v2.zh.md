# CHANGE_0049 定向修复 MCQ-QA 校准集 V2

## 背景与动机

分桶评测结果表明，当前主要准确率短板已经不是 `niah` 或 `fwe`，而是这两个关键区域：

- `mcq|len_0_4k`
- `qa|len_4k_32k`

上一版语义优先 8 条子集虽然对部分长上下文桶有帮助，但在这两个关键区域上引入了明显波动。因此，这次改动新增一个第二版 8 条校准子集，明确针对这两个弱桶做修复。

## 规则合规说明

本改动符合 SOAR 2026 当前要求。

- 仅从公开集 `perf_public_set.jsonl` 中选择样本。
- 不直接修改模型权重，不改运行时内核，不改评测逻辑。
- 仍属于官方允许的离线 GPTQ calibration data 优化。

## 详细实施计划

改动前计划：

- 保留现有 semantic-priority 8 与 16 文件不变。
- 在现有生成脚本中增加一个新的命名输出。
- 生成一个新的 `8_v2` 文件，重点覆盖短 `mcq` 与中长度 `qa`，同时只保留一个长 `qa` anchor。

## 实际代码改动

- 更新 `benchmark/soar/demo_sala/build_semantic_priority_calibration_sets.py`。
- 新增 `benchmark/soar/demo_sala/calib_semantic_priority_8_v2.jsonl`。

新子集索引：

- `1, 2, 4, 8, 61, 63, 66, 76`

设计理由：

- `1, 2, 4, 8`：尽量保留原始 sequential 8 中较稳定的短 `mcq` anchor 结构。
- `61, 63, 66`：明确增加 `qa|len_4k_32k` 覆盖。
- `76`：保留一个 `qa|len_32k_128k` anchor，避免长 `qa` 完全失守。
- 这轮定向修复不再包含 `niah`、`fwe`、`cwe`。

## 验证命令

```bash
python3 -m py_compile benchmark/soar/demo_sala/build_semantic_priority_calibration_sets.py
python3 benchmark/soar/demo_sala/build_semantic_priority_calibration_sets.py
python3 benchmark/soar/demo_sala/build_semantic_priority_calibration_sets.py --check
wc -l benchmark/soar/demo_sala/calib_semantic_priority_8_v2.jsonl
```

示例运行：

```bash
export SOAR_GPTQ_CALIBRATION_FILE=benchmark/soar/demo_sala/calib_semantic_priority_8_v2.jsonl
bash benchmark/soar/demo_sala/prepare_env.sh
```

## 结果汇总表

| 版本 | 目标 | 状态 |
|---|---|---|
| `semantic_priority_8` | 通用语义优先子集 | 已有 |
| `semantic_priority_16` | 更大语义子集 | 已有 |
| `semantic_priority_8_v2` | 修复 `mcq|len_0_4k` 与 `qa|len_4k_32k` | 新增 |

## 回滚说明

- 继续使用 `calib_semantic_priority_8.jsonl` 或原始 calibration 文件即可。
- 如有需要，可通过生成脚本重新生成所有文件。
- 这次改动不涉及运行时代码回滚，本质上只是切换 calibration 文件。

## 下一步建议

1. 用 `semantic_priority_8_v2` 直接对比原始 sequential 8。
2. 结合 `eval_model_001.py` 检查 `mcq|len_0_4k` 与 `qa|len_4k_32k` 是否同时回升。
3. 只有在这两个弱桶一起改善且总准确率不跌破安全线时，才考虑继续做 v3。