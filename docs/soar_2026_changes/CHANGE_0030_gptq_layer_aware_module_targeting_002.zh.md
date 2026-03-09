# CHANGE_0030_002：GPTQ 分层模块目标选择（实施记录）

## 背景与动机
本续篇记录 CHANGE_0030 在批准后的实际代码落地。此前 fcloud API 探测已经确认 `gptqmodel==5.7.0` 的量化入口是实例级 `BaseQModel.quantize(...)`。

本次实施目标保持不变：

- 降低混合层模块不匹配导致的失败
- 保持提交接口兼容
- 通过显式日志与可控重试提升可调试性

## 规则合规说明（SOAR）
该特性仍然合规，因为它：

- 仅修改离线预处理行为
- 不引入评测侧捷径
- 保持官方 `prepare_model.sh --input/--output` 接口契约
- 仍要求后续用正确率验证后再考虑提交

## 实际代码改动
更新文件：

- `benchmark/soar/demo_sala/preprocess_model.py`

已实现内容：

1. 新增兼容辅助函数
- `_env_truthy(...)`
- `_parse_csv_env(...)`
- `_call_with_supported_kwargs(...)`
- `_apply_quant_module_controls(...)`
- `_is_module_mismatch_error(...)`

2. 新增分层 GPTQ 默认策略
- `SOAR_GPTQ_LAYER_AWARE=1` 默认开启
- 默认 include：
  - `self_attn.q_proj`
  - `self_attn.k_proj`
  - `self_attn.v_proj`
  - `self_attn.o_proj`
  - `mlp.gate_proj`
  - `mlp.up_proj`
  - `mlp.down_proj`
- 默认 exclude：
  - `self_attn.o_gate`

3. 新增版本兼容的 GPTQ 调用
- `GPTQModel.load(...)` 在需要时可自动去掉不被支持的可选参数，例如 `attn_implementation`
- `model.quantize(...)` 在需要时可自动去掉不被支持的可选参数，例如 `batch_size`

4. 新增混合层重试路径
- 若量化因模块不匹配类错误失败，脚本会重建 `QuantizeConfig`、重新应用保守公共目标、重新加载模型并重试一次
- 保守重试仍默认排除 `self_attn.o_gate`

5. 新增诊断日志
- 打印当前 include/exclude 模块列表
- 打印成功写入的 quant-config 属性名
- 打印重试原因与重试控制项

## 准确性/稳定性风险
当前已知权衡：

- 如果实际 `gptqmodel` 版本忽略本次尝试写入的 quant-config 模块控制属性，则该改动未必能在所有版本中彻底消除根因。
- 保守目标集合可能降低量化覆盖率。
- 真实行为仍取决于 fcloud 上的 `gptqmodel` 内部实现，因此必须以 fcloud 结果为准。

## 验证命令
1. 本地语法检查
```bash
python3 -m py_compile benchmark/soar/demo_sala/preprocess_model.py
```

2. fcloud 小样本冒烟
```bash
cd benchmark/soar/demo_sala
SOAR_QUANT_MODE=gptq SOAR_GPTQ_CALIBRATION_FILE=/path/to/calib.jsonl SOAR_GPTQ_CALIBRATION_SAMPLES=16 SOAR_GPTQ_BATCH_SIZE=1 ./prepare_model.sh --input /path/raw_model --output /path/quant_model
```

3. fcloud 大样本质量轮
```bash
cd benchmark/soar/demo_sala
SOAR_QUANT_MODE=gptq SOAR_GPTQ_CALIBRATION_FILE=/path/to/calib.jsonl SOAR_GPTQ_CALIBRATION_SAMPLES=128 SOAR_GPTQ_BATCH_SIZE=2 ./prepare_model.sh --input /path/raw_model --output /path/quant_model
```

4. 可选显式模块覆盖测试
```bash
SOAR_GPTQ_LAYER_AWARE=1 SOAR_GPTQ_INCLUDE_MODULES=self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj SOAR_GPTQ_EXCLUDE_MODULES=self_attn.o_gate ./prepare_model.sh --input /path/raw_model --output /path/quant_model
```

## 结果汇总表
状态：本地代码已落地，运行指标待 fcloud 回填。

| 指标 | CHANGE_0030_002 前 | CHANGE_0030_002 后 |
|---|---:|---:|
| GPTQ 预处理完成率 | 混合层模块不匹配失败 | TBD |
| 正确率分数 | TBD | TBD |
| 速度 S1 | TBD | TBD |
| 速度 S8 | TBD | TBD |
| 速度 Smax | TBD | TBD |

## 回滚说明
1. 回滚 CHANGE_0030 的实现提交。
2. 或临时切回 `SOAR_QUANT_MODE=copy`。
3. 若需要，可设置 `SOAR_GPTQ_LAYER_AWARE=0`，跳过新目标策略，同时保留其他兼容性修复。

## 后续建议
1. 先跑 fcloud 小样本冒烟并回传完整日志。
2. 若仍出现 mismatch，请重点保留日志中打印的 applied include/exclude attrs，方便确认 `QuantizeConfig` 真正接受了哪些控制项。
3. 一旦 quant-prep 成功，再跑正确率和 3 次速度评测，并把结果回填到 CHANGE_0030 文档。
