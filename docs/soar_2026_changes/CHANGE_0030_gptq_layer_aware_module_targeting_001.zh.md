# CHANGE_0030_001：GPTQ 分层模块目标选择（基于 API 探测的续篇）

## 目的
本续篇用于承接 CHANGE_0030 中较长的 API 探测结论，避免主文档过长，同时保留完整决策链路，便于后续复盘与迭代。

## 来自 fcloud 的新证据（API 探测）
根据你提供的 fcloud 输出：

- `gptqmodel` 版本：`5.7.0`
- 加载后实例类型：`gptqmodel.models.base.BaseQModel`
- 实例方法中包含：
  - `quantize(...)`
  - `save(...)`
  - `save_quantized(...)`
- 之前的报错（`GPTQModel has no attribute quantize`）来自“类级别探测”，并不代表“实例级别不可用”。

日志中的附加线索：

- Module Tree AutoCompat 以 `layer0` 推断目标模块并包含 `self_attn.o_gate`。
- 在混合层结构中，这仍有风险，因为部分层可能没有 `o_gate`。

## 结论解读
已确认的兼容性事实：

1. 量化调用应以“已加载实例 API”为准，不应依赖类静态假设。
2. 当前 `model.quantize(calibration_texts, batch_size=...)` 在该版本可用。
3. 主要失败根因仍是“混合层模块目标不一致”（AutoCompat 基于非代表性层推断）。

## 可直接实施的提案（已批准方向）
### 目标
让 `benchmark/soar/demo_sala/preprocess_model.py` 在 `gptqmodel==5.7.0` 与异构层结构下更稳健。

### 预期收益
- 规避 `self_attn.o_gate not found in model` 这类硬失败。
- 通过显式目标日志提升可复现性。

### 合规性
- 仅离线预处理。
- 不使用评测时禁用技巧。
- 不改变提交接口契约。

### 风险与缓解
风险：
- 目标过保守会降低量化覆盖。
- 目标过激进仍可能触发缺失模块错误。

缓解：
- 通过环境变量开关控制分层策略。
- 量化前打印目标集合。
- 提供一次可控保守重试路径。

## 需要修改的文件/函数
主文件：

- `benchmark/soar/demo_sala/preprocess_model.py`

计划在 `run_gptq_quantization(...)` 与辅助函数中实现：

1. 新增环境变量控制：
- `SOAR_GPTQ_LAYER_AWARE`（默认 `1`）
- `SOAR_GPTQ_INCLUDE_MODULES`（CSV 覆盖）
- `SOAR_GPTQ_EXCLUDE_MODULES`（CSV）

2. 新增模块列表解析工具：
- 规范化 CSV
- 过滤空 token

3. 安全配置量化目标：
- 通过 include/exclude + 安全默认值生成目标集合
- 保守回退集合默认不含 `o_gate`

4. 增加一次模块不匹配重试：
- 首次按配置目标运行
- 若触发“模块不存在”类错误，则回退到保守公共集合：
  - `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`

5. 增强诊断信息：
- 打印生效目标集合与重试原因
- 若两轮均失败，给出可执行的错误提示

## 验证命令
1. 小样本冒烟：
```bash
SOAR_QUANT_MODE=gptq \
SOAR_GPTQ_CALIBRATION_FILE=/path/to/calib.jsonl \
SOAR_GPTQ_CALIBRATION_SAMPLES=16 \
SOAR_GPTQ_BATCH_SIZE=1 \
./prepare_model.sh --input /path/raw_model --output /path/quant_model
```

2. 大样本质量轮：
```bash
SOAR_QUANT_MODE=gptq \
SOAR_GPTQ_CALIBRATION_FILE=/path/to/calib.jsonl \
SOAR_GPTQ_CALIBRATION_SAMPLES=128 \
SOAR_GPTQ_BATCH_SIZE=2 \
./prepare_model.sh --input /path/raw_model --output /path/quant_model
```

3. 然后做正确率与速度评测：
```bash
python benchmark/soar/eval_model.py --help
python benchmark/soar/run_soar_suite.py --help
```

## 结果汇总表（待回填）
| 指标 | 续篇改动前 | 续篇改动后 |
|---|---:|---:|
| GPTQ 预处理完成率 | 混合模块不匹配失败 | TBD |
| 正确率 | TBD | TBD |
| 速度 S1 | TBD | TBD |
| 速度 S8 | TBD | TBD |
| 速度 Smax | TBD | TBD |

## 回滚方案
1. 禁用分层策略环境开关（若已实现）后重试。
2. 临时使用 `SOAR_QUANT_MODE=copy`。
3. 必要时回滚续篇提交，恢复旧行为。

## 下一步
1. 将本续篇方案落实为 CHANGE_0030 的代码补丁。
2. 在 fcloud 先做冒烟并回传日志。
3. 依据失败日志继续调整 include/exclude 默认值。
