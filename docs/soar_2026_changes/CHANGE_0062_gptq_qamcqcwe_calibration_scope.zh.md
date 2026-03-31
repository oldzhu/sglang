# CHANGE_0062 GPTQ 扩大到 QA+MCQ+CWE 校准范围

## 背景与动机

最近的本地与官方结果表明，当前 speed 优化包在 benchmark duration 上已经有明显竞争力，但正确性仍然是总分的主要限制项。当前最不稳定或最容易掉点的任务类别已经比较明确地集中在：

1. `qa`
2. `mcq`
3. `cwe`

上一版默认校准 focus 是 `qa,cwe`。这意味着尽管近期本地评测已经显示 `mcq` 也是三个最弱类别之一，但它并没有被纳入默认 GPTQ 校准范围。

本次迭代只改变一个变量：在保持 `32` 条校准预算不变的前提下，把默认校准任务范围从 `qa,cwe` 扩大到 `qa,mcq,cwe`。

## 规则合规说明

本改动符合 2026-03-31 重新检查的最新 SOAR 比赛页与 toolkit 页面要求。

- 仅修改允许的预处理阶段中的校准样本选择逻辑。
- 保持现有 GPTQ W4A16 + Marlin 路径不变。
- 不修改运行时并发规则或 prefix-cache 行为。
- 不替换 MiniCPM-SALA 基座模型。
- 通过环境变量即可回退到旧的任务范围。

## 详细实施计划

改动前计划：

1. 保持现有校准筛选实现不变。
2. 保持总校准样本数仍为 `32`。
3. 保持 stratified 采样开启。
4. 只把默认任务白名单从 `qa,cwe` 改为 `qa,mcq,cwe`。
5. 把 `48` 样本实验留到这个更窄的 A/B 完成之后再做。

## 实际代码改动

修改文件：

- `benchmark/soar/demo_sala/prepare_env.sh`

具体变更：

1. 更新 `SOAR_GPTQ_CALIBRATION_TASK_INCLUDE` 的默认值。
2. 旧默认值：
   - `qa,cwe`
3. 新默认值：
   - `qa,mcq,cwe`
4. 样本数保持不变：
   - `SOAR_GPTQ_CALIBRATION_SAMPLES=32`
5. 采样模式保持不变：
   - `SOAR_GPTQ_CALIBRATION_SAMPLING=stratified`

## 设计说明

### 为什么不直接上 48 条样本

如果直接改成 `48` 条样本，就会一次性改变两个变量：

1. 新增 `mcq`
2. 增大样本预算

这样会使结果更难解释，而且近期测试已经证明 `64` 样本会稳定 OOM。先在现有 `32` 条预算下把 `mcq` 加入默认任务范围，是更干净的一步一 feature 实验。

### 为什么现在加入 mcq

近期本地结果已经显示 `mcq` 与 `qa`、`cwe` 一样，属于主要弱项之一。既然当前 selector 已经支持按任务过滤，最低风险的下一步就是确保这三个弱项都参与默认校准。

### 为什么继续保留 stratified

现有 stratified 流程仍然可以在选定任务内部保留长度桶多样性。考虑到当前不稳定性并不只出现在任务维度，还和长度桶有关，因此继续保留这点是合理的。

## 验证命令

Shell 语法检查：

```bash
bash -n benchmark/soar/demo_sala/prepare_env.sh
```

预处理验证：

```bash
bash benchmark/soar/demo_sala/prepare_model.sh --input <RAW_MODEL_DIR> --output <OUTPUT_MODEL_DIR>
```

正确性验证：

```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_DIR> \
  --data_path <DATA_DIR>/perf_public_set.jsonl \
  --concurrency 32
```

回滚到上一版范围：

```bash
export SOAR_GPTQ_CALIBRATION_TASK_INCLUDE=qa,cwe
```

如果这一步还不够，再进行下一步实验：

```bash
export SOAR_GPTQ_CALIBRATION_SAMPLES=48
export SOAR_GPTQ_CALIBRATION_TASK_INCLUDE=qa,mcq,cwe
```

## 结果汇总表

| 项目 | 修改前 | 修改后 |
| --- | --- | --- |
| 默认校准任务范围 | `qa,cwe` | `qa,mcq,cwe` |
| 样本数 | `32` | `32` |
| 采样模式 | `stratified` | `stratified` |
| 主要变化变量 | 无 | 把 `mcq` 纳入默认校准 focus |

## 回滚说明

如果加入 `mcq` 之后仍然不能明显改善正确性：

1. 设置 `SOAR_GPTQ_CALIBRATION_TASK_INCLUDE=qa,cwe`
2. 重新执行同样的 preprocess 与正确性验证
3. 再进入下一 feature：`48` 样本的 `qa,mcq,cwe`，或者如果出现 OOM，则改做更稳妥的校准内存优化

## 下一步建议

1. 先验证加入 `mcq` 是否能改善正确性，同时不拖累 `qa` 与 `cwe`。
2. 如果准确率仍不稳定，再把 `48` 样本同任务范围作为下一单独 feature 测试。
3. 如果 `48` 触发 OOM，下一轮应优先解决校准阶段的内存稳定性，而不是简单继续加样本。