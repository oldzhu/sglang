# CHANGE_0110: Dense注意力GPTQ校准

## 背景与动机

MiniCPM-SALA有8个标准注意力层（索引0,9,16,17,22,29,30,31）和24个lightning注意力层。标准注意力层支持两种模式：

- **稀疏（InfLLMv2）**：对长度≥8192的序列使用topk=96块选择
- **全连接（Dense）**：对所有token进行完整注意力计算

在GPTQ校准期间，HuggingFace模型加载时使用`config.json`中的`sparse_config`，导致实例化`MiniCPMInfLLMv2Attention`。对于≥8192 token的校准序列（qa/cwe中包含大量32K–128K的样本），使用的是稀疏注意力。

然而在推理时，sglang始终使用`--force-dense-minicpm`，将稀疏注意力替换为全连接的FlashInfer/FlashAttention后端。这造成了**校准/推理注意力模式不匹配**，可能降低量化质量——GPTQ权重量化网格是针对稀疏注意力产生的激活模式优化的，但推理时看到的是全连接注意力模式。

由于GPTQ是W4A16（仅权重量化），这种不匹配是二阶效应——它影响的是校准期间哪些权重列看到更高的激活幅度，而不是量化激活本身。但消除这种不匹配可以使量化与运行时行为更加一致。

## 规则合规声明

- 仅修改量化校准过程——无模型架构或权重格式变更
- 完全符合比赛规则（现场量化、≤5小时、≤2GB）
- 无违规操作；仅对齐校准注意力与推理注意力
- 通过`SOAR_GPTQ_FORCE_DENSE`环境变量控制（默认=1，设为0即可还原）

## 实现方案

### 方法：在校准配置中将`sparse_config`置为null

在`GPTQModel.load()`之前将sanitized config中的`sparse_config`设为`None`，使HF模型为所有minicpm4层实例化`MiniCPMFlashAttention2`（全连接注意力）而非`MiniCPMInfLLMv2Attention`。该方法最优因为：

1. 利用现有的`_prepare_gptq_load_source()`临时配置机制
2. 无需对模型forward方法打猴子补丁
3. 消除校准期间对`infllmv2`内核的依赖
4. 与`--force-dense-minicpm`运行时行为完全匹配

### 修改文件

1. **`benchmark/soar/demo_sala/preprocess_model.py`** — `_sanitize_model_config_for_gptq()`
   - 新增：检查`SOAR_GPTQ_FORCE_DENSE`环境变量（默认为True）
   - 启用时且`sparse_config`不为None时：将`sparse_config`设为`None`并记录更改

2. **`benchmark/soar/demo_sala/prepare_env.sh`**
   - 新增：`export SOAR_GPTQ_FORCE_DENSE="${SOAR_GPTQ_FORCE_DENSE:-1}"`
   - 新增：echo日志行

## 实际代码变更

### preprocess_model.py — `_sanitize_model_config_for_gptq()`

```python
# 在rope_type移除块之后添加：
force_dense = _env_truthy("SOAR_GPTQ_FORCE_DENSE", default=True)
if force_dense and sanitized.get("sparse_config") is not None:
    sanitized["sparse_config"] = None
    changes.append(
        "set sparse_config=null to force dense attention during GPTQ calibration "
        "(matches --force-dense-minicpm inference mode)"
    )
```

### prepare_env.sh

```bash
export SOAR_GPTQ_FORCE_DENSE="${SOAR_GPTQ_FORCE_DENSE:-1}"
```

## 验证命令

### 步骤1：重新校准基线（无M1代码）

```bash
# 在fcloud上——使用dense校准重新量化
source /root/submission_sim/prepare_env.sh
python3 /root/submission_sim/prepare_model.sh \
  --input /root/models/openbmb/MiniCPM-SALA-Copy \
  --output /root/models/openbmb/MiniCPM-SALA-90-qa-cwe-mcq-sparse_qkv_w8

# 重启服务器并运行精度+速度测试
python3 scripts/fcloud/fcloud_workflow.py full
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

### 步骤2：在新权重上重新测试M1

```bash
# 启用M1分支，重启服务器，重新测试
python3 scripts/fcloud/fcloud_workflow.py restart-server
python3 scripts/fcloud/fcloud_workflow.py accuracy
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

## 结果汇总

### Test 24: Dense校准GPTQ基线（无M1）

| 指标 | Test 20（稀疏校准） | Test 24（dense校准） | 变化 |
|------|-------------------|--------------------------|-------|
| ori_accuracy | 80.64% | **77.64%** | **-3.00%** |
| normalized | 100.80% | 97.05% | -3.75% |
| C | 1.0 | **0.92** | **回归** |
| mcq | 63.33% | **50.00%** | **-13.33%** |
| qa | 63.33% | 60.00% | -3.33% |
| cwe | 77.67% | 79.33% | +1.66% |
| fwe | 98.89% | 98.89% | 0 |
| niah | 100% | 100% | 0 |
| S1 (s) | 113.67 | 110.59 | -2.7%（更快） |
| S8 (s) | 41.07 | 40.45 | -1.5%（更快） |
| Smax (s) | 34.15 | 33.64 | -1.5%（更快） |

### 结论：**失败**

Dense校准使精度显著**恶化**，而非改善。mcq精度从63.33%暴跌至50.00%，整体C从1.0降至0.92。速度有微小提升（~1.5-2.7%），但无法弥补精度灾难。

**关键发现**：原始稀疏校准的GPTQ权重对于dense推理实际上**更好**。这是违反直觉的，但在稀疏注意力模式下优化的GPTQ量化网格对dense推理具有良好的泛化能力。校准期间的稀疏注意力可能起到一种正则化的作用，产生的权重量化在不同注意力模式下更加鲁棒。

Test 25（M1 + dense校准）已**跳过**，因为基线（Test 24）已显示不可接受的精度回归——在差的权重上叠加M1不会产生有意义的结果。

## 回滚说明

在`prepare_env.sh`中设置`SOAR_GPTQ_FORCE_DENSE=0`并重新量化，即可恢复稀疏校准权重。稀疏校准权重仍为生产基线。

## 后续步骤

- **Phase B**: 使用调优参数（`SOAR_GPTQ_DAMP_PERCENT`、`SOAR_GPTQ_MSE`）测试 dense 校准，在保持速度优势的同时恢复精度
- 使用 `quick-accuracy --tasks mcq`（~3-5 分钟）进行快速筛选，然后再进行完整评测
- 如果 Phase B 失败，恢复 `SOAR_GPTQ_FORCE_DENSE=0` 并转向其他优化方向

## 附录：快速精度评估方法对比

在完整精度测试（~50-60 分钟）之前，有两种快速评估量化质量的方法：

### 方法 A：任务过滤法（已实现）

运行实际评测任务的子集（如仅 MCQ，30 个样本，~3-5 分钟）。

- **优点**：直接测量端到端精度；覆盖 prefill 和 decode 阶段；无需额外设置
- **缺点**：方差大（30 个 MCQ 样本，每个占 3.33%）；无法可靠区分精度差距在 ~10% 以内的配置；仍需生成（每个 MCQ 样本约 6K token）
- **用法**：`python3 scripts/fcloud/fcloud_workflow.py quick-accuracy --tasks mcq`

### 方法 B：Logprob 分布对比法（尚未实现）

通过 logprobs 分布（KL 散度、余弦相似度）对比量化模型与 BF16 基线。参考：曹议（SOAR 2026 第三周冠军博客），灵感来自 "Accuracy is Not All You Need" 论文。

- **方法**：一次性 BF16 基线（贪婪解码，提取最后 128 个 token 的 top-256 logprobs）。每个配置：发送相同 prompt，对比分布。
- **优点**：极快（秒到分钟级，仅 prefill）；稳定的连续指标——可可靠地对配置进行排序；基线数据零 GPU 占用
- **缺点**：仅覆盖 prefill 阶段——不捕获 decode 阶段的误差传播；不能直接预测比赛精度分数；需要一次性 BF16 基线设置
- **实现时机**：当需要比较 >5 个校准配置时，MCQ 噪声使任务过滤法无法可靠排序

### 建议工作流

1. **Logprob**（如已实现）→ 快速筛选和排序大量配置
2. **任务过滤** → 验证最优候选是否通过精度阈值
3. **完整评测** → 提交前最终验证
