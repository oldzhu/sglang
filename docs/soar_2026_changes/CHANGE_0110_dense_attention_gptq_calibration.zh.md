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

| 指标 | Test 20（稀疏校准） | Test 24（dense校准，待定） | Test 25（dense校准+M1，待定） |
|------|-------------------|--------------------------|------------------------------|
| ori_accuracy | 80.64% | 待定 | 待定 |
| normalized | 100.80% | 待定 | 待定 |
| C | 1.0 | 待定 | 待定 |
| S1 (s) | 113.67 | 待定 | 待定 |
| S8 (s) | 41.07 | 待定 | 待定 |
| Smax (s) | 34.15 | 待定 | 待定 |

## 回滚说明

在`prepare_env.sh`中设置`SOAR_GPTQ_FORCE_DENSE=0`并重新量化，即可恢复稀疏校准权重。

## 后续步骤

- 如果dense校准改善或维持精度：采用为新默认值，继续测试M1路径
- 如果M1+dense校准达到归一化精度>99%（C=1.0）：采用M1用于提交
- 如果M1精度仍然回归：转向下一优化优先级（A1或K4）
