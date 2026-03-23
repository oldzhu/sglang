# CHANGE_0046：GPTQ 公开集校准采样策略

## 背景与动机

当前 GPTQ 预处理路径使用 `perf_public_set.jsonl` 作为校准来源。根据最新 SOAR toolkit 页面，这样做是允许的，因为官方已经明确公开了正确性评测集供选手自查。

但旧实现存在一个明显问题：校准样本只是按文件顺序读取，然后直接截取前 `N` 条。这会带来两个后果：

- 即使把 `SOAR_GPTQ_CALIBRATION_SAMPLES` 从 `32` 增加到 `64`，准确率也未必提升，因为样本组成偏置并没有改变
- 如果 `perf_public_set.jsonl` 按任务或长度分块排列，前 `N` 条会过度代表文件开头那一段分布

对于 MiniCPM-SALA，这个问题尤其明显，因为公开集同时包含较短的 MCQ 题和更长的信息提取题。GPTQ 的效果依赖激活分布覆盖，因此“选哪些样本”至少和“选多少样本”同样重要。

因此，本次迭代不改变量化方法本身，只实现一个低风险的单一优化 feature：把公开集校准采样改为可配置、可复现，并提供更能覆盖任务与长度分布的分层采样模式。

## 规则合规说明

本次修改与最新官方 competition / toolkit 页面保持一致：

- 仍然位于官方鼓励的 `路径一：量化加速` 范围内
- 不替换 MiniCPM-SALA 官方基座模型
- 仍然通过 `prepare_model.sh` 在评测机现场执行量化
- 仅使用公开发布的 `perf_public_set.jsonl` 做公开集校准，不使用私有集或隐藏数据
- 不修改服务并发规则、不启用 prefix cache，也不改官方正确性脚本

## 详细实现计划

修改前计划：

1. 重新检查最新 `competition` 与 `toolkit` 页面，确认公开集自查和 GPTQ 路径一优化仍然允许。
2. 确认当前 preprocess 代码中的校准样本选择确实是顺序截断，没有随机化。
3. 在文本提取之前增加一个确定性的样本选择层，使采样策略变化不影响后续量化逻辑。
4. 保留可回滚的 `sequential` 模式，确保旧行为随时可恢复。

## 实际代码变更

修改文件：

- `benchmark/soar/demo_sala/preprocess_model.py`
- `benchmark/soar/demo_sala/prepare_env.sh`

`preprocess_model.py` 中的变更：

1. 新增三种确定性的校准采样模式：
   - `sequential`
   - `shuffled`
   - `stratified`
2. 为 `shuffled` 与 `stratified` 增加固定随机种子，确保可复现。
3. 新增长度与任务分层辅助逻辑，分层依据包括：
   - task 类型
   - 若存在则使用 `prompt_tokens` 的长度桶
4. 新增 largest-remainder 分配逻辑，将样本预算分配到多个 bucket，而不是退化为“文件前缀样本”。
5. 扩展 GPTQ 启动日志，打印实际生效的校准采样摘要，包括：
   - mode
   - seed
   - available rows
   - selected rows
   - 各 bucket 的选样数量

`prepare_env.sh` 中的变更：

1. 新增默认环境变量：
   - `SOAR_GPTQ_CALIBRATION_SAMPLING=stratified`
   - `SOAR_GPTQ_CALIBRATION_SEED=20260320`
   - `SOAR_GPTQ_CALIBRATION_TASK_BALANCE=1`
   - `SOAR_GPTQ_CALIBRATION_USE_PROMPT_TOKENS=1`
2. 在启动日志中打印这些变量，方便在远端日志中确认实际策略。

## 设计说明

### 为什么不再只增加校准样本数

此前 `32 -> 64` 的实验没有带来正确率提升。根因并不一定是样本数不足，而是旧逻辑只会取文件前 `N` 条。如果数据文件本身按任务或长度分块排列，那么增大 `N` 可能只是扩大相同偏置，并不会改善校准覆盖度。

### 为什么默认使用 stratified

公开集里同时有短 MCQ 和长上下文检索任务。相比文件前缀截取，覆盖任务类型与多档 `prompt_tokens` 长度的校准切片，更有可能在 W4A16 量化后保住整体正确率。

### 为什么仍然保留回滚安全性

旧行为仍然可以通过以下设置完整恢复：

- `SOAR_GPTQ_CALIBRATION_SAMPLING=sequential`

如果新的采样策略在公开集或私有集上表现不好，回滚成本很低。

## 验证命令

使用新的默认策略执行量化：

```bash
bash prepare_model.sh --input <raw_model_dir> --output <processed_model_dir>
```

强制回滚为顺序采样做 A/B：

```bash
export SOAR_GPTQ_CALIBRATION_SAMPLING=sequential
bash prepare_model.sh --input <raw_model_dir> --output <processed_model_dir>
```

正确性验证：

```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_DIR> \
  --data_path ./perf_public_set.jsonl \
  --concurrency 32
```

本地汇总脚本验证：

```bash
python3 benchmark/soar/run_soar_suite.py \
  --api-base http://127.0.0.1:30000 \
  --model-path <MODEL_DIR> \
  --eval-script benchmark/soar/demo_sala/eval_model.py \
  --public-data benchmark/soar/demo_sala/perf_public_set.jsonl
```

## 结果摘要表

| 项目 | 修改前 | 修改后 |
| --- | --- | --- |
| 校准样本选择 | 仅取前 `N` 条 | 可配置采样模式 |
| 可复现性 | 依赖文件顺序 | 显式固定 seed |
| 任务覆盖 | 取决于文件前缀 | 按 task 分层 |
| 长度覆盖 | 取决于文件前缀 | 按 `prompt_tokens` 长度桶分层 |
| 回滚路径 | 不可配置 | `sequential` 模式 |
| 正确率结果 | 待验证 | 待验证 |

## 回滚说明

如果新的采样策略没有带来正确率提升：

1. 设置 `SOAR_GPTQ_CALIBRATION_SAMPLING=sequential`
2. 如有需要，关闭 task 与 prompt-length 分层
3. 重新执行 `prepare_model.sh` 生成 GPTQ 产物

回滚示例：

```bash
export SOAR_GPTQ_CALIBRATION_SAMPLING=sequential
export SOAR_GPTQ_CALIBRATION_TASK_BALANCE=0
export SOAR_GPTQ_CALIBRATION_USE_PROMPT_TOKENS=0
bash prepare_model.sh --input <raw_model_dir> --output <processed_model_dir>
```

## 后续建议

1. 在相同 `64` 样本数下，先对比 `sequential` 与 `stratified` 的 A/B 结果。
2. 如果 `stratified` 能稳定提升正确率，再将 `SOAR_GPTQ_GROUP_SIZE=64` 作为下一次独立 feature 测试。
3. 如果 `stratified` 没有帮助，再去分析 MCQ 与长上下文任务的分任务差异，不要立刻改量化模块覆盖率。