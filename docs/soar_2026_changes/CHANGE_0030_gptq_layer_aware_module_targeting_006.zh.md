# CHANGE_0030_006：MiniCPM-SALA 自定义 GPTQModel 定义

## 背景与动机
此前 CHANGE_0030 的多轮迭代已经证明，GPTQModel 5.7.0 官方 `QuantizeConfig.dynamic` 路径可以排除 `self_attn.o_gate` 这类模块，但最新的 SGLang 报错说明，仅靠排除模块还不足以解决 MiniCPM-SALA 的问题。

新的 fcloud checkpoint 证据显示：

- `model.layers.1.self_attn.z_proj.weight` 以原生权重形式存在
- `model.layers.1.self_attn.o_proj.qweight` 以量化权重形式存在
- 这说明同一个 lightning 层本身就允许“部分原生、部分量化”的 attention 子模块布局
- 但 GPTQModel 的通用 AutoCompat 路径仍然没有为整个混合层结构提供稳定、完整的架构描述

因此根因已经很清楚：问题不只是模块排除，而是 GPTQModel AutoCompat 会从一个代表层推断整棵模块树，而 MiniCPM-SALA 的 decoder layer 家族并不统一。

另外，SOAR 第一周冠军笔记也进一步证明这条路线值得投入：4-bit 量化将模型体积从约 18 GB 压缩到 5.4 GB，并把单并发耗时降低约 40%。因此，只要能把 4-bit 路径做正确，它就仍然是高优先级优化方向。

## 规则合规说明（SOAR）
该改动仍然合规，因为它：

- 只影响离线预处理行为
- 不改变评测时并发或隐藏运行时行为
- 保持官方 `prepare_model.sh --input/--output` 契约不变
- 仍要求在使用前完成正确率验证

## 详细实施计划
本轮计划并已实施：

1. 为 `model_type = "minicpm_sala"` 增加本地运行时 GPTQModel 定义
2. 在 `GPTQModel.load(...)` 之前把该定义注册进 GPTQModel 模型映射
3. 用显式 MiniCPM-SALA `module_tree` 取代对 layer-0 AutoCompat 的依赖
4. 把混合层中的可选辅助模块保留在结构树中，但用 `:!` 标记为非量化
5. 同时把 `self_attn.o_gate` 和 `self_attn.z_proj` 继续保留为防御性 dynamic 排除项

## 实际代码改动
新增文件：

- `benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py`

关键实现点：

1. 新增 `MiniCPMSALAGPTQ(BaseQModel)`
- `require_trust_remote_code = True`
- `layer_modules_strict = False`
- `pre_lm_head_norm_module = "model.norm"`
- 为 MiniCPM-SALA 写明了显式 `module_tree`

2. 将混合层辅助模块编码为非量化结构节点
- `self_attn.q_norm`
- `self_attn.k_norm`
- `self_attn.o_norm`
- `self_attn.z_proj`
- `self_attn.o_gate`

3. 保留明确的量化目标模块
- `self_attn.q_proj`
- `self_attn.k_proj`
- `self_attn.v_proj`
- `self_attn.o_proj`
- `mlp.gate_proj`
- `mlp.up_proj`
- `mlp.down_proj`

4. 新增运行时注册函数
- `register_minicpm_sala_gptq_model()` 会补丁 GPTQModel 的 `MODEL_MAP`
- 必要时也会同步更新 `SUPPORTED_MODELS`

更新文件：

- `benchmark/soar/demo_sala/preprocess_model.py`

5. 在 `GPTQModel.load(...)` 前注册 MiniCPM-SALA 自定义定义
6. 将防御性排除默认值从仅 `self_attn.o_gate` 扩展为：
- `self_attn.o_gate`
- `self_attn.z_proj`

## 准确性/稳定性风险
优势：

- 修复的是根本抽象不匹配，而不是再叠加一层排除规则
- 保持 lightning 门控权重与 sparse 门控权重为原生模块
- 不再依赖一个异构模型中并不可靠的 layer-0 AutoCompat

剩余风险：

- fcloud 上原始 HF MiniCPM-SALA 模块名仍可能与本地按 SGLang 对齐的预期有细微差异
- 即使 checkpoint 加载问题被修复，4-bit 正确率也仍可能低于 SOAR 可接受阈值

## 验证命令
1. 本地语法检查
```bash
python3 -m py_compile benchmark/soar/demo_sala/preprocess_model.py
python3 -m py_compile benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py
```

2. fcloud 小样本量化冒烟
```bash
cd benchmark/soar/demo_sala
SOAR_QUANT_MODE=gptq \
SOAR_TRUST_REMOTE_CODE=true \
SOAR_GPTQ_ATTN_IMPL=flash_attention_2 \
SOAR_GPTQ_CALIBRATION_FILE=/path/to/calib.jsonl \
SOAR_GPTQ_CALIBRATION_SAMPLES=16 \
SOAR_GPTQ_BATCH_SIZE=1 \
./prepare_model.sh --input /path/to/raw_model --output /path/to/quant_smoke 2>&1 | tee /path/to/quant_smoke.log
```

3. 检查 checkpoint 结构
```bash
python3 - <<'PY'
import json
from pathlib import Path

index_path = Path('/path/to/quant_smoke/model.safetensors.index.json')
payload = json.loads(index_path.read_text())
keys = payload['weight_map']
for name in [
    'model.layers.1.self_attn.z_proj.weight',
    'model.layers.1.self_attn.o_proj.qweight',
    'model.layers.0.self_attn.o_gate.weight',
]:
    print(name, name in keys)
PY
```

4. SGLang 加载冒烟
```bash
python3 -m sglang.launch_server \
  --model-path /path/to/quant_smoke \
  --trust-remote-code \
  --quantization gptq_marlin
```

## 结果汇总表
状态：已在本地实现，待 fcloud 验证。

| 指标 | CHANGE_0030_006 前 | CHANGE_0030_006 后 |
|---|---:|---:|
| GPTQ 定义来源 | layer 0 AutoCompat | 显式 `minicpm_sala` 定义 |
| `o_gate` 处理 | 仅 dynamic 排除 | 原生结构节点 + dynamic 防御 |
| `z_proj` 处理 | 隐式/不稳定 | 原生结构节点 + dynamic 防御 |
| SGLang 加载量化 checkpoint | 在 `z_proj` 路径失败 | TBD |
| 正确率分数 | TBD | TBD |
| 速度 S1 | TBD | TBD |
| 速度 S8 | TBD | TBD |
| 速度 Smax | TBD | TBD |

## 回滚说明
1. 回滚 CHANGE_0030_006 提交。
2. 删除 `preprocess_model.py` 中的注册调用。
3. 删除 `benchmark/soar/demo_sala/gptqmodel_minicpm_sala.py`。
4. 必要时退回 `SOAR_QUANT_MODE=copy` 或此前的 dynamic-only 路径。

## 后续建议
1. 先跑 16 样本 fcloud 冒烟，并保留新的 `simple_layer_modules` 日志。
2. 同时检查 lightning 层和 sparse 层的 checkpoint 索引结构。
3. 如果 SGLang 仍失败，先比对原始 HF 模块名与注册树是否一致，再决定是否改 loader 代码。