# CHANGE_0030_005：GPTQ 官方 Dynamic 排除机制实施记录

## 背景与动机
CHANGE_0030_004 的调研已确认，GPTQModel 5.7.0 官方提供的逐模块排除机制是 `QuantizeConfig.dynamic`。

fcloud 证据显示：

- `module_tree` 仍包含 `self_attn.o_gate`
- `full_layer_modules` 仍包含 `self_attn.o_gate`
- `simple_layer_modules` 已排除 `self_attn.o_gate`

这说明正确修复路径应当是走 GPTQ 的官方过滤逻辑，而不是继续修改内部树结构。

## 规则合规说明（SOAR）
该改动仍然合规，因为它：

- 仅修改离线预处理行为
- 不改变评测时行为
- 保持官方 `prepare_model.sh --input/--output` 契约
- 仍要求后续做正确率与速度验证

## 实际代码改动
更新文件：

- `benchmark/soar/demo_sala/preprocess_model.py`

已实现内容：

1. 用官方 dynamic 控制替换启发式模块树修补
- 当开启 layer-aware 模式时，`QuantizeConfig` 通过 `dynamic` 负向匹配规则构造
- 例如排除规则：
  - `-:.*self_attn\.o_gate.*`

2. 新增 `_build_dynamic_rules(...)`
- 根据 `exclude_modules` 生成负向 regex 规则

3. 保留 GPTQ API 兼容性封装
- `GPTQModel.load(...)` 仍兼容可选参数差异
- `model.quantize(...)` 仍兼容可选参数差异

4. 增加模块解析调试日志
- 打印 `dynamic_rules`
- 打印已加载 GPTQ 模型的 `simple_layer_modules(...)`，用于确认官方过滤是否生效
- 重试路径也会打印同类信息

5. 不再把内部模块树修改作为主要修复路径
- 在 GPTQModel 5.7.0 中，树修改并不是权威控制点

## 准确性/稳定性风险
当前权衡：

- 在保守路径下，`self_attn.o_gate` 仍然是全局排除
- 若 GPTQModel 内部模块命名与预期略有差异，regex 可能还需要一次小幅微调

为什么这比之前更稳妥：

- 它使用了 GPTQModel 文档支持的公开机制
- 它与 fcloud 上观察到的 `simple_layer_modules(...)` 行为一致

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

3. 成功标志
- 启动日志中打印 `dynamic_rules`
- 启动日志中打印的 `simple_layer_modules=` 不再包含 `self_attn.o_gate`
- 量化能越过此前 `o_gate` 不匹配的失败点

## 结果汇总表
状态：本地代码已落地，待 fcloud 验证。

| 指标 | CHANGE_0030_005 前 | CHANGE_0030_005 后 |
|---|---:|---:|
| 是否使用官方模块排除路径 | 否 | 是 |
| GPTQ 预处理完成率 | 因 `o_gate` 不匹配失败 | TBD |
| 正确率分数 | TBD | TBD |
| 速度 S1 | TBD | TBD |
| 速度 S8 | TBD | TBD |
| 速度 Smax | TBD | TBD |

## 回滚说明
1. 回滚 CHANGE_0030_005 提交。
2. 或临时退回 `SOAR_QUANT_MODE=copy`。
3. 如有需要，可保留 GPTQ API 兼容性封装，仅回退 `dynamic` 规则构造。

## 后续建议
1. 先跑 16 样本 fcloud 冒烟。
2. 确认新日志中的 `simple_layer_modules` 已排除 `o_gate`。
3. 若冒烟通过，再继续 128 样本质量轮以及正确率/速度验证。
