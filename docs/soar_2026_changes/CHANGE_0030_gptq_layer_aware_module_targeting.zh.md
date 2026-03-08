# CHANGE_0030：GPTQ 分层模块目标选择（提案）

## 背景与动机
当前 `preprocess_model.py` 的 GPTQ 流程已越过早期阻塞点（`trust_remote_code`、不兼容的内部 attention 参数），但出现新错误：

- `ValueError: layer module item self_attn.o_gate not found in model`

你提供的模型结构证据表明 MiniCPM-SALA 不同层并不一致：

- 部分稀疏层包含 `self_attn.o_gate`
- 部分 lightning/其他层不含 `o_gate`，而是 `z_proj` 与相关 norm

这说明“全局统一模块列表”在该混合架构上不安全。

## 目标与预期收益
在 `benchmark/soar/demo_sala/preprocess_model.py` 实现“按层感知”的 GPTQ 模块选择策略，使量化过程：

- 不因缺失模块直接失败
- 在异构层结构下尽量保持有效量化覆盖
- 通过可追踪日志提升可复现性与调优效率

预期收益：

- 打通 fcloud 端到端 quant-prep
- 降低因模块不匹配导致的反复重跑成本
- 为后续质量/速度调优提供稳定底座

## 规则合规说明（SOAR）
该改动符合 SOAR 约束，因为它：

- 仅影响离线预处理路径
- 不涉及在线评测禁用技巧
- 以正确率验证为优先
- 保持提交脚本接口契约（`prepare_model.sh --input/--output`）

规则新鲜度说明：

- 最终提交前仍需按最新官方页面核对：
  - `https://soar.openbmb.cn/competition`
  - `https://soar.openbmb.cn/toolkit`（重点 `技术路径指引`、`提交说明`）

## 准确性/稳定性风险
潜在风险：

- 目标集合过保守会降低量化覆盖和速度收益。
- 目标集合过激进可能导致校准不稳或质量下降。
- 分层差异可能导致不同模型版本表现不一致。

缓解措施：

- 使用“显式可配策略 + 按实际存在性过滤”
- 量化前输出目标计划与分组统计
- 分阶段验证（小样本冒烟 -> 大样本质量）

## 详细实施计划（改动前）
计划修改 `benchmark/soar/demo_sala/preprocess_model.py`：

1. 新增目标策略环境变量
- `SOAR_GPTQ_LAYER_AWARE=1`（默认开启）
- `SOAR_GPTQ_INCLUDE_MODULES`（逗号分隔白名单，可选）
- `SOAR_GPTQ_EXCLUDE_MODULES`（逗号分隔黑名单，可选）
- `SOAR_GPTQ_STRICT_TARGETS=0/1`（若无可用目标是否严格失败）

2. 构建分层模块清单
- 通过模型结构遍历收集每层真实可见模块路径
- 区分“全层共通模块”与“仅部分层存在模块”

3. 生成有效目标集合
- 从安全默认值开始（q/k/v/o + 常见 MLP 投影）
- 稀疏专属、lightning 专属模块仅在“真实存在”时纳入
- 最后应用排除列表

4. 量化守护与重试
- 若量化器拒绝部分目标，在可控条件下记录并收缩重试
- 若最终目标为空，给出明确错误并停止

5. 增加可观测日志
- 输出：选中目标、跳过目标、按层存在性统计
- 可选落盘 summary JSON 到输出目录便于复盘

## 上一轮回复中的核心策略（纳入文档便于评审）
采用两层目标策略：

- A层（优先共通）：在多数层都存在且收益高的 GPTQ 模块
- B层（条件纳入）：仅在特定层族存在时才纳入（如稀疏层 `o_gate`、lightning 层特有投影）

执行规则：

- 未在当前模型清单中观测到的模块，不允许传给 GPTQ

## 验证命令（正确率 + 速度）
实现后在 fcloud 执行：

1. 小样本 quant-prep 冒烟
```bash
SOAR_QUANT_MODE=gptq \
SOAR_GPTQ_CALIBRATION_FILE=/path/to/calib.jsonl \
SOAR_GPTQ_CALIBRATION_SAMPLES=16 \
SOAR_GPTQ_BATCH_SIZE=1 \
./prepare_model.sh --input /path/raw_model --output /path/quant_model
```

2. 大样本 quant-prep 质量轮
```bash
SOAR_QUANT_MODE=gptq \
SOAR_GPTQ_CALIBRATION_FILE=/path/to/calib.jsonl \
SOAR_GPTQ_CALIBRATION_SAMPLES=128 \
SOAR_GPTQ_BATCH_SIZE=2 \
./prepare_model.sh --input /path/raw_model --output /path/quant_model
```

3. 启服 + 正确率验证
```bash
# 使用量化模型启动后，执行官方/本地正确率评测
python benchmark/soar/eval_model.py --help
```

4. 3 次速度评测（S1/S8/Smax）
```bash
python benchmark/soar/run_soar_suite.py --help
```

## 结果汇总表（基线 vs 新方案）
状态：待实现并回填 fcloud 指标。

| 指标 | 基线（CHANGE_0030 前） | 新方案（CHANGE_0030 后） |
|---|---:|---:|
| GPTQ 预处理完成率 | 混合模块不匹配导致失败 | TBD |
| 正确率分数 | TBD | TBD |
| 速度 S1 | TBD | TBD |
| 速度 S8 | TBD | TBD |
| 速度 Smax | TBD | TBD |

## 回滚说明
若出现回归：

1. 通过环境开关禁用分层策略（若已实现）或回滚 CHANGE_0030 提交。
2. 临时切回 `SOAR_QUANT_MODE=copy` 保证提交流程可用。
3. 重新跑正确率与速度基线确认回滚稳定。

## 后续建议
1. 按本提案把分层目标策略作为一个完整特性落地到 `preprocess_model.py`。
2. 先跑小样本冒烟并回传日志。
3. 根据模块清单输出，微调 include/exclude 默认值。
4. 再做完整正确率 + 3 次速度评测并回填本文档。
