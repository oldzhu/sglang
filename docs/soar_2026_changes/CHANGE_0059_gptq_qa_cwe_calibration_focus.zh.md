# CHANGE_0059 GPTQ 聚焦 QA+CWE 校准集

## 背景与动机

最近两次官方提交已经能稳定跑通，但量化后正确性仍低于 SOAR 要求的 97% 门槛，因此最终分数仍为 0。根据本地 non-quant 与 quant 对比，当前最大的精度损失集中在 `qa` 和 `cwe` 两类任务上，尤其是 `len_4k_32k` 与 `len_32k_128k` 两个长度桶。

2026-03-28 重新检查的 SOAR toolkit `技术路径指引` 仍然明确鼓励使用 W4A16 GPTQ + Marlin 路线，`提交说明` 也仍然允许通过 `prepare_model.sh --input --output` 在评测机上执行预处理。因此，调整校准集组成仍然是合规且低风险的精度恢复手段。

本次迭代先把 GPTQ 校准样本聚焦到当前掉点最严重的两类任务：`qa` 和 `cwe`。目标是在固定 32 条校准预算内，优先覆盖最弱任务，再决定是否进入更高风险的 selective de-quantization 等下一步。

## 规则合规说明

本改动符合 2026-03-28 检查的最新 SOAR 比赛页与 toolkit 页面要求。

- 保持官方要求的 `prepare_env.sh` 与 `prepare_model.sh --input/--output` 契约不变。
- 仍然沿用官方鼓励的 GPTQ W4A16 + Marlin 量化路径。
- 不修改 MiniCPM-SALA 基座模型，不改 benchmark 并发逻辑，也不触碰被禁止的 prefix cache 行为。
- 仅在允许的预处理阶段调整校准样本筛选逻辑。

## 详细实施计划

改动前计划：

1. 保持现有 32 条 GPTQ 校准流程不变。
2. 保留 stratified 采样，继续覆盖不同 prompt 长度桶。
3. 增加一个由环境变量控制的任务白名单过滤器。
4. 本次实验默认只使用 `qa,cwe`。
5. 在日志中打印当前任务过滤配置，保证官方运行可审计。

## 实际代码改动

修改文件：

- `benchmark/soar/demo_sala/preprocess_model.py`
- `benchmark/soar/demo_sala/prepare_env.sh`

具体变更：

1. 在 `preprocess_model.py` 中新增 `_filter_calibration_records_by_task()`。
2. 新增任务名标准化逻辑，使 `SOAR_GPTQ_CALIBRATION_TASK_INCLUDE` 以大小写不敏感方式匹配。
3. 在现有 sequential、shuffled、stratified 采样逻辑之前先做任务过滤。
4. 扩展校准摘要输出，新增以下字段：
   - 是否启用了任务过滤
   - 实际包含的任务列表
   - 过滤前记录数
   - 过滤后记录数
5. 在 `prepare_env.sh` 中把默认值设置为：
   - `SOAR_GPTQ_CALIBRATION_TASK_INCLUDE=qa,cwe`
6. 在 `prepare_env.sh` 日志中输出当前任务过滤配置。

## 设计说明

### 为什么先过滤任务，再做 stratified 采样

当前代码已经支持在 `stratified` 模式下按任务与 prompt 长度桶做分层。先过滤任务，可以把固定的 32 条预算集中到 `qa` 和 `cwe`，同时仍保留这两类任务内部的长度覆盖。

### 为什么继续保留 32 条样本

目前主问题是精度，不是预处理耗时。因此这一步不再同时改动样本数，避免一次引入两个变量。

### 为什么用环境变量控制

用环境变量可以在不继续改代码的情况下快速回退或扩展任务集合。后续如果要恢复更多任务，只需覆盖 `SOAR_GPTQ_CALIBRATION_TASK_INCLUDE`。

## 验证命令

Shell 语法检查：

```bash
bash -n benchmark/soar/demo_sala/prepare_env.sh
```

Python 语法检查：

```bash
python3 -m py_compile benchmark/soar/demo_sala/preprocess_model.py
```

快速校准选择探针：

```bash
SOAR_GPTQ_CALIBRATION_TASK_INCLUDE=qa,cwe \
python3 benchmark/soar/demo_sala/preprocess_model.py \
  --input <RAW_MODEL_DIR> \
  --output <OUTPUT_MODEL_DIR> \
  --mode copy
```

完整 preprocess：

```bash
bash benchmark/soar/demo_sala/prepare_model.sh --input <RAW_MODEL_DIR> --output <OUTPUT_MODEL_DIR>
```

量化后正确性检查：

```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <OUTPUT_MODEL_DIR> \
  --data_path <DATA_DIR>/perf_public_set.jsonl \
  --concurrency 32
```

## 结果汇总表

| 项目 | 修改前 | 修改后 |
| --- | --- | --- |
| 校准任务范围 | 使用校准文件中的全部任务 | 默认只使用 `qa` 与 `cwe` |
| 样本预算 | 32 | 32 |
| 采样模式 | stratified | stratified |
| 长度桶覆盖 | 已启用 | 在过滤后的任务集合内继续保留 |
| 量化功能范围 | 不变 | 不变 |

## 回滚说明

如果聚焦 `qa,cwe` 后正确性提升仍然不够：

1. 可以把 `SOAR_GPTQ_CALIBRATION_TASK_INCLUDE` 覆盖成更宽的任务集合，例如 `mcq,qa,cwe`
2. 或者在 `prepare_env.sh` 中移除该默认过滤
3. 保留当前代码路径，进入下一步低风险实验，例如在保留长度分层的前提下把 `mcq` 加回校准集

## 下一步建议

1. 先做一次本地量化加正确性验证，优先比较之前已知最弱的四个 bucket 是否改善。
2. 如果提升有限，下一步先尝试 `qa,cwe,mcq`，再决定是否进入 selective de-quantization。
3. 如果 `qa` 明显恢复但 `cwe` 仍然偏弱，可以继续做下一轮，只在过滤后的任务集合内进一步提高 `len_4k_32k` 与 `len_32k_128k` 的覆盖权重。