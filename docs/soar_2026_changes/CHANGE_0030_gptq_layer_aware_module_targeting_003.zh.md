# CHANGE_0030_003：通过模块树净化实现 GPTQ 保守解阻

## 背景与动机
在 CHANGE_0030_002 之后，fcloud 日志已经表明失败根因不在前面尝试写入的 `QuantizeConfig` 字段，而在“已加载 GPTQ 模型实例”内部。

观察到的证据：

- GPTQ AutoCompat 基于 `layer0` 推导出了共享 `module_tree`
- 该树中包含 `self_attn.o_gate`
- 后续 lightning 风格层并不总是存在 `o_gate`
- 因此量化阶段报错：
  - `ValueError: layer module item self_attn.o_gate not found in model`

本续篇先采取“保守解阻”方案。

## 评审结论记录
针对前一轮的两个问题，结论如下：

1. `o_gate` 只对 lightning 层移除，还是全局移除？
- 第一阶段保守方案：从 GPTQ 量化目标树中**全局移除** `o_gate`。
- 这**不会**从模型结构中删除 `o_gate`。
- 仅表示 GPTQ 不再尝试对该模块做量化/校准。

2. 这会影响 calibration 吗？
- 会，但仅影响 `o_gate`。
- `o_gate` 不再参与 GPTQ 的校准统计。
- 其他目标模块（`q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`、`down_proj`）仍然正常参与。

之所以接受该方案：

- 这是当前风险最低的解阻方式
- 可以保留跨层稳定模块的量化
- 在获得端到端成功路径前，不必过早深入对抗 GPTQ 更复杂的内部逻辑

## 规则合规说明（SOAR）
该改动仍然合规，因为它：

- 仅修改离线预处理路径
- 不改变官方评测时行为
- 保持 `prepare_model.sh --input/--output` 提交契约
- 仍需在 quant-prep 成功后继续验证正确率与速度

## 实际代码改动
更新文件：

- `benchmark/soar/demo_sala/preprocess_model.py`

已实现逻辑：

1. 新增 `_sanitize_module_tree_node(...)`
- 递归处理 list、tuple、dict
- 删除被禁用的叶子模块名，例如 `o_gate`
- 删除后若容器为空则一并裁剪

2. 新增 `_sanitize_model_module_tree(...)`
- 处理 `model.module_tree`
- 若存在，也处理 `model.module_tree_overrides`
- 修改时打印 before/after 日志

3. 在 `GPTQModel.load(...)` 后立即净化
- 首次 `model.quantize(...)` 前即对已加载模型实例做净化
- 重试路径中重新加载的模型也会再次净化

## 准确性/稳定性风险
当前权衡：

- 在该保守路径下，`o_gate` 将被全局跳过量化
- 对于确实存在 `o_gate` 的稀疏层，可能损失一部分压缩/速度空间
- 但相比之下，这比沿用一个会直接失败的混合层计划更稳妥

预期实际影响：

- 量化覆盖略有下降
- 预处理成功率显著提高的概率更大

## 验证命令
1. 本地语法检查
```bash
python3 -m py_compile benchmark/soar/demo_sala/preprocess_model.py
```

2. fcloud 小样本冒烟
```bash
cd benchmark/soar/demo_sala
SOAR_QUANT_MODE=gptq \
SOAR_TRUST_REMOTE_CODE=true \
SOAR_GPTQ_ATTN_IMPL=flash_attention_2 \
SOAR_GPTQ_LAYER_AWARE=1 \
SOAR_GPTQ_INCLUDE_MODULES=self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj \
SOAR_GPTQ_EXCLUDE_MODULES=self_attn.o_gate \
SOAR_GPTQ_CALIBRATION_FILE=/path/to/calib.jsonl \
SOAR_GPTQ_CALIBRATION_SAMPLES=16 \
SOAR_GPTQ_BATCH_SIZE=1 \
./prepare_model.sh --input /path/to/raw_model --output /path/to/quant_smoke 2>&1 | tee /path/to/quant_smoke.log
```

3. fcloud 大样本质量轮
```bash
cd benchmark/soar/demo_sala
SOAR_QUANT_MODE=gptq \
SOAR_GPTQ_CALIBRATION_FILE=/path/to/calib.jsonl \
SOAR_GPTQ_CALIBRATION_SAMPLES=128 \
SOAR_GPTQ_BATCH_SIZE=2 \
./prepare_model.sh --input /path/to/raw_model --output /path/to/quant_full 2>&1 | tee /path/to/quant_full.log
```

## 结果汇总表
状态：本地实现已完成，待 fcloud 验证。

| 指标 | CHANGE_0030_003 前 | CHANGE_0030_003 后 |
|---|---:|---:|
| GPTQ 预处理完成率 | 因 `self_attn.o_gate` 不匹配失败 | TBD |
| 正确率分数 | TBD | TBD |
| 速度 S1 | TBD | TBD |
| 速度 S8 | TBD | TBD |
| 速度 Smax | TBD | TBD |

## 回滚说明
1. 回滚 CHANGE_0030_003 提交。
2. 或临时退回 `SOAR_QUANT_MODE=copy`。
3. 如有需要，可移除模块树净化逻辑，同时保留前面已有的兼容性辅助函数。

## 后续建议
1. 先跑 16 样本 fcloud 冒烟并回传完整日志。
2. 确认日志中 `module_tree` 已不再包含 `o_gate`。
3. 若成功，再继续 128 样本质量轮以及正确率/速度验证。
