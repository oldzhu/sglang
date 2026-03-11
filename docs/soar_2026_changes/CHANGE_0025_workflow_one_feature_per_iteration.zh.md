# CHANGE_0025_workflow_one_feature_per_iteration

## 1) 背景与动机
- 当前“每次一个很小改动”的节奏过慢，影响迭代效率。
- 团队希望在保证可追踪性的前提下，改为按“功能特性”推进。

## 2) 规则合规说明
- 本次为项目流程策略更新。
- 不涉及模型/推理行为或竞赛评测逻辑改动。

## 3) 变更前计划
- 在 `.github/copilot-instructions.md` 中将流程表述从：
  - 每次一个小改动
  - 调整为每次一个“改进特性”
- 双语文档要求保持不变，但改为“一组文档对应一个特性迭代”。

## 4) 实际代码改动
- 更新 `.github/copilot-instructions.md`：
  - `One change at a time` 改为 `One improving feature at a time`
  - 明确一个特性可包含多个文件的协同改动
  - 明确文档映射关系为“一次特性迭代一组 EN/ZH 文档”

## 5) 验证
```bash
git diff .github/copilot-instructions.md
```

## 6) 风险
- 低。单次改动范围增大可能增加评审负担。
- 缓解：保持特性内聚，避免混入无关改动。

## 7) 回滚方案
1. 回退 `.github/copilot-instructions.md` 中对应流程段落。

## 8) 下一步建议
- 后续量化预处理实现按“单次一个完整特性”推进。
