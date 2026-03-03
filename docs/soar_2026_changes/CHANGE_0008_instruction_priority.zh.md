# CHANGE_0008_instruction_priority

## 1）背景与动机
- 在长周期优化对话中，可能同时出现多来源指令，需要明确优先级以避免歧义。
- 目标：固定指令优先级，保证后续行为一致。

## 2）SOAR 规则合规性检查
- 本次仅为文档/流程更新。
- 不影响模型推理、正确性或 benchmark 逻辑。

## 3）改动前计划
- 在 `.github/copilot-instructions.md` 中新增“指令优先级”章节。
- 按要求补齐中英文变更文档。

## 4）实际代码改动
- 在 `.github/copilot-instructions.md` 新增：
  - 指令优先级顺序（system/developer -> 项目级 instructions -> 当前用户请求 -> 其他仓库文档）
  - 冲突处理原则（遵循高优先级并简要说明）

## 5）验证
- 人工阅读更新后的指令文件确认条目存在且顺序正确。

## 6）结果汇总
| 项目 | 状态 |
|---|---|
| 已新增优先级策略 | 完成 |
| 已新增冲突处理说明 | 完成 |

## 7）风险评估
- 无（仅流程澄清）。

## 8）回滚说明
1. 从 `.github/copilot-instructions.md` 中移除 `Instruction priority order` 章节。

## 9）下一步建议
- 后续可单独补充“冲突示例”小节，便于团队快速理解。
