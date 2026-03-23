# CHANGE_0047 语义优先校准子集

## 背景与动机

最近的本地结果表明，当前 MiniCPM-SALA 的主要瓶颈已经不是运行时内核，而是 GPTQ W4A16 量化质量本身。单纯增加 calibration sample 数量并没有带来更稳定的正确性，说明“样本组成”比“样本个数”更值得优先验证。

当前假设是：如果校准集中过多包含低语义密度、长上下文的抽取类样本，会把 GPTQ 的尺度拟合往不利方向拉偏。这个改动新增两份手工筛选的子集文件，优先覆盖 `mcq` 和 `qa`，只保留极少量 `niah`。

## 规则合规说明

本改动符合当前 SOAR 2026 官方规则与工具链要求。

- 仅使用官方公开数据文件 `perf_public_set.jsonl` 中的样本子集。
- 不修改基座模型，不改评测逻辑，不改在线推理路径。
- 符合官方 `prepare_env.sh` + `prepare_model.sh` 的提交执行模型。
- 与官方技术路径指引中的 GPTQ W4A16 预处理路线一致。

## 详细实施计划

改动前计划：

- 在 `benchmark/soar/demo_sala/` 下增加一个可复现的小脚本。
- 在脚本中固化两组索引，分别对应 8 条和 16 条“语义优先”样本。
- 在 `perf_public_set.jsonl` 同目录生成可直接使用的 JSONL 文件。
- 增加 `--check` 校验模式，方便拷贝、打包、上传后再次确认内容未漂移。

## 实际代码改动

- 新增 `benchmark/soar/demo_sala/build_semantic_priority_calibration_sets.py`。
- 新增 `benchmark/soar/demo_sala/calib_semantic_priority_8.jsonl`。
- 新增 `benchmark/soar/demo_sala/calib_semantic_priority_16.jsonl`。

子集设计：

- 8 条子集：`2, 11, 17, 23, 25, 61, 76, 90`
- 16 条子集：`2, 5, 8, 11, 17, 23, 25, 30, 31, 60, 61, 63, 66, 76, 81, 90`

任务分布策略：

- 以 `mcq` 和 `qa` 为主，优先保留更高语义密度样本。
- 仅在 16 条集合里保留少量 `niah` 覆盖。
- 第一轮人工子集不包含 `cwe` 和 `fwe`。

## 验证命令

```bash
python3 -m py_compile benchmark/soar/demo_sala/build_semantic_priority_calibration_sets.py
python3 benchmark/soar/demo_sala/build_semantic_priority_calibration_sets.py
python3 benchmark/soar/demo_sala/build_semantic_priority_calibration_sets.py --check
wc -l benchmark/soar/demo_sala/calib_semantic_priority_8.jsonl
wc -l benchmark/soar/demo_sala/calib_semantic_priority_16.jsonl
```

示例量化命令：

```bash
export SOAR_GPTQ_CALIBRATION_FILE=benchmark/soar/demo_sala/calib_semantic_priority_8.jsonl
bash benchmark/soar/demo_sala/prepare_env.sh
```

```bash
export SOAR_GPTQ_CALIBRATION_FILE=benchmark/soar/demo_sala/calib_semantic_priority_16.jsonl
bash benchmark/soar/demo_sala/prepare_env.sh
```

## 结果汇总表

| 方案 | 校准集 | 公开正确性 | 本地时长 | 状态 |
|---|---|---:|---:|---|
| 现有量化基线 | Sequential 8 | 79.31 / 75.96 | 3535.36 / 3054.16 | 已有参考 |
| 现有量化基线 | Sequential 16 | 77.58 / 74.18 | 3755.73 / 3707.54 | 已有参考 |
| 新候选 | Semantic-priority 8 | TBD | TBD | 待验证 |
| 新候选 | Semantic-priority 16 | TBD | TBD | 待验证 |

## 回滚说明

- 将 `SOAR_GPTQ_CALIBRATION_FILE` 切回之前使用的文件即可停止使用本改动。
- 如果担心文件在传输中被改动，可重新执行生成脚本恢复标准内容。
- 本改动不涉及运行时代码回退，实质上只是切换 calibration 文件。

## 下一步建议

1. 先把 semantic-priority 8 与现有 sequential 8 做直接 A/B。
2. 如果 semantic-priority 8 有精度收益且速度无明显恶化，再看 semantic-priority 16 是否能带来更稳定的重复性。
3. 如果两组都没有改善，就不要继续手工猜样本，转向轻量级的量化前快速精度筛选工具。