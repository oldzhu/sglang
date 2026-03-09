# CHANGE_0030_004：GPTQ 官方 Dynamic 控制路径调研与提案

## 背景与动机
在多次 CHANGE_0030 尝试后，本轮暂停直接改代码，转而先阅读 GPTQModel 5.7.0 的源码与文档，避免继续在未确认的内部字段上“猜来猜去”。

本轮调研目标：

- 找到报错堆栈对应的 Python 源文件
- 确认 GPTQModel 如何构建 calibration/quantization 所使用的模块列表
- 判断是否存在官方支持的模块 include/exclude 控制方式

本续篇记录调研结论。

## 规则合规说明（SOAR）
这是一次仅做调研和提案的迭代。

- 本轮未修改模型行为
- 未改变评测时行为
- 提交接口契约保持不变

## fcloud 上 GPTQModel 5.7.0 的问题栈源码位置
与当前失败直接相关的 Python 源码路径：

- `/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/gptqmodel/quantization/config.py`
- `/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/gptqmodel/models/base.py`
- `/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/gptqmodel/looper/module_looper.py`
- `/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/gptqmodel/looper/stage_layer.py`
- `/root/sglang-minicpm/sglang_minicpm_sala_env/lib/python3.10/site-packages/gptqmodel/utils/model.py`

traceback 对应的调用路径：

- `BaseQModel.quantize(...)`
- `ModuleLooper.loop(...)`
- `ModuleLooper._loop_impl(...)`
- `run_layer_stage(...)`
- `create_named_modules(...)`
- 最终抛出 `ValueError: layer module item self_attn.o_gate not found in model`

## 调研结论
### 1. 官方确实提供了逐模块控制机制
从 GPTQModel README 与源码可以确认，官方提供的逐模块控制入口是：

- `QuantizeConfig.dynamic`

文档语义：

- 正向 regex 匹配：对特定模块覆盖 bits/group_size 等配置
- 负向 regex 匹配：将匹配模块跳过量化

### 2. 官方过滤路径位于 BaseQModel
`gptqmodel/models/base.py` 中的关键逻辑：

- `simple_layer_modules(...)`
- `filter_not_quantize_module(...)`

其中 `filter_not_quantize_module(...)` 的核心行为是：

- 当 `dynamic_get(quantize_config.dynamic, module_name=m) is False` 时，直接把该模块从 `layer_modules` 中移除

这说明 `dynamic` 就是 GPTQModel 官方预期的模块排除方式。

### 3. 二次保障逻辑也存在于 quant module 创建阶段
`gptqmodel/utils/model.py` 中的：

- `create_quant_module(...)`

这里同样会读取 `dynamic`，如果是负向匹配，则跳过该模块。

### 4. fcloud 运行结果进一步证实了正确控制路径
你回传的结果：

- `module_tree = ['model', 'layers', '#', {'self_attn': ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'o_gate'), 'mlp': ('gate_proj', 'up_proj', 'down_proj')}]`
- `simple_layer_modules = [['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj'], ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']]`
- `full_layer_modules = [['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'self_attn.o_gate'], ['mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']]`

解释：

- `module_tree` 是结构级描述，仍然包含 `o_gate`
- `full_layer_modules` 是结构展开结果，也仍然包含 `o_gate`
- `simple_layer_modules` 才是“用于量化的过滤后模块列表”，它已经不包含 `o_gate`

这基本可以确认：GPTQModel 期望通过官方过滤路径做排除，而不是通过手工修改 `module_tree`。

## 提案：切换到官方 Dynamic 方案
### 目标与预期收益
用文档支持的 `QuantizeConfig.dynamic` 负向匹配，替代当前的启发式内部树修改方式，让 `self_attn.o_gate` 通过 GPTQModel 的官方路径被排除。

预期收益：

- 不再依赖未确认的内部字段
- 与 GPTQModel 5.7.0 官方逐模块排除流程对齐
- 降低继续试错的时间成本

### 需要改动的文件
- `benchmark/soar/demo_sala/preprocess_model.py`

### 计划中的代码修改
通过 `dynamic` 构造 `QuantizeConfig`，例如：

```python
dynamic = {
    r"-:.*self_attn\.o_gate.*": {},
}
quant_config = QuantizeConfig(bits=bits, group_size=group_size, dynamic=dynamic)
```

若需要更宽松匹配，可退回：

```python
dynamic = {
    r"-:.*o_gate.*": {},
}
```

同时计划清理：

- 移除之前的启发式 `module_tree` 修改逻辑
- 保留 `GPTQModel.load(...)` / `model.quantize(...)` 的 API 兼容性封装
- 增加对最终 `dynamic` 规则的日志输出，方便复现

## 准确性/稳定性风险
风险：

- 在保守路径下，`o_gate` 仍将被全局跳过量化
- 如果 GPTQModel 对模块名匹配规则与预期不同，regex 可能需要微调一次

缓解：

- 使用官方支持的 `dynamic` 机制，而不是私有内部字段
- 通过 `simple_layer_modules(...)` 验证过滤是否生效
- 先跑 16 样本冒烟，再决定是否扩大规模

## 验证命令
1. 在 fcloud 上直接验证官方过滤路径
```bash
python3 - <<'PY'
from gptqmodel import GPTQModel, QuantizeConfig

qcfg = QuantizeConfig(
    bits=4,
    group_size=128,
    dynamic={r"-:.*self_attn\\.o_gate.*": {}},
)

model = GPTQModel.load(
    "/path/to/raw_model",
    qcfg,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)

print("module_tree =", getattr(model, "module_tree", None))
print("simple_layer_modules =", model.simple_layer_modules(model.model.config, model.quantize_config))
print("full_layer_modules =", model.full_layer_modules(model.model.config))
PY
```

2. 实施后做 fcloud 小样本冒烟
```bash
cd benchmark/soar/demo_sala
SOAR_QUANT_MODE=gptq \
SOAR_GPTQ_CALIBRATION_FILE=/path/to/calib.jsonl \
SOAR_GPTQ_CALIBRATION_SAMPLES=16 \
SOAR_GPTQ_BATCH_SIZE=1 \
./prepare_model.sh --input /path/to/raw_model --output /path/to/quant_smoke
```

## 结果汇总表
状态：调研完成；本续篇尚未实施代码改动。

| 指标 | CHANGE_0030_004 前 | CHANGE_0030_004 后 |
|---|---:|---:|
| 是否找到官方模块排除路径 | 否 | 是 |
| GPTQ 预处理完成率 | 因 `o_gate` 不匹配失败 | TBD |
| 正确率分数 | TBD | TBD |
| 速度 S1 | TBD | TBD |
| 速度 S8 | TBD | TBD |
| 速度 Smax | TBD | TBD |

## 回滚说明
本续篇仅记录调研与提案，不涉及代码改动，因此无需回滚。

## 后续建议
1. 批准一次聚焦实现：用 `QuantizeConfig.dynamic` 取代启发式树修改。
2. 实施后第一时间在 fcloud 上用 `simple_layer_modules(...)` 验证过滤结果。
3. 只有在官方 `dynamic` 仍无效时，才回头考虑更深层的内部 hook。