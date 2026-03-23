# CHANGE_0050 校准候选集生成器

## 背景与动机

手工一次次猜 calibration 子集已经接近边际收益递减。最近几轮实验表明，不同的 8 条子集主要是在 `mcq|len_0_4k`、`qa|len_4k_32k`、`qa|len_32k_128k` 以及部分 `niah` 桶之间搬移误差，而不是带来稳定的全局提升。下一步更合理的做法，是把候选集生成和结果记录都工具化、可复现化。

## 规则合规说明

本改动符合当前 SOAR 2026 规则。

- 仅使用公开集 `perf_public_set.jsonl` 中的样本。
- 不修改运行时推理逻辑、内核或官方评分逻辑。
- 本质上是离线 GPTQ calibration 数据实验的辅助工具层，仍在现有预处理流程内。

## 详细实施计划

改动前计划：

- 将候选集定义从硬编码 Python 字面量迁移到 JSON 配置文件。
- 扩展现有生成脚本，使其从配置文件读取候选集。
- 增加多组 8 条候选，用于受控 A/B 测试。
- 增加统一的结果记录模板，用相同决策标准比较重复实验。

## 实际代码改动

- 更新 `benchmark/soar/demo_sala/build_semantic_priority_calibration_sets.py`。
- 新增 `benchmark/soar/demo_sala/calibration_candidates.json`。
- 新增 `benchmark/soar/demo_sala/calibration_candidate_results_template.md`。

支持的工作流：

- `--list`：只打印候选集元信息，不生成文件
- 默认执行：生成所有候选 JSONL 文件
- `--check`：校验已生成文件是否与源数据和配置索引一致

初始候选集包括：

- 历史子集：
  - `semantic_priority_8`
  - `semantic_priority_8_v2`
  - `semantic_priority_16`
- 新实验候选：
  - `candidate_baseline_like_8`
  - `candidate_mcq_qa_mid_8`
  - `candidate_mcq_heavy_8`
  - `candidate_qa_balanced_8`

## 验证命令

```bash
python3 -m py_compile benchmark/soar/demo_sala/build_semantic_priority_calibration_sets.py
python3 benchmark/soar/demo_sala/build_semantic_priority_calibration_sets.py --list
python3 benchmark/soar/demo_sala/build_semantic_priority_calibration_sets.py
python3 benchmark/soar/demo_sala/build_semantic_priority_calibration_sets.py --check
```

## 结果汇总表

| 产物 | 用途 | 状态 |
|---|---|---|
| `calibration_candidates.json` | 候选集定义的唯一事实来源 | 新增 |
| `build_semantic_priority_calibration_sets.py` | 候选集生成与校验工具 | 已更新 |
| `calibration_candidate_results_template.md` | 标准化实验记录模板 | 新增 |

## 回滚说明

- 继续只使用之前已经生成的 calibration 文件即可。
- 如果更偏好手工实验，可以忽略新的候选配置和模板。
- 这次改动不影响运行时，只影响离线实验工具层，因此无需运行时回滚。

## 下一步建议

1. 先统一生成所有候选 JSONL 文件。
2. 不要一次性全测，先挑 2 到 4 组最有价值的候选做实验。
3. 每组都用同一模板记录结果，便于直接比较稳定性。