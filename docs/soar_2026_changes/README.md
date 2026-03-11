# SOAR 2026 Change Log / 变更记录

This folder stores one bilingual document pair per optimization change.

该目录用于存放每次优化变更的一对中英文件（一次变更对应一对文档）。

## Naming / 命名规范

- English: `CHANGE_XXXX_<short_title>.en.md`
- 中文: `CHANGE_XXXX_<short_title>.zh.md`

Example / 示例:

- `CHANGE_0001_sparse_prefill_guard.en.md`
- `CHANGE_0001_sparse_prefill_guard.zh.md`

## Workflow / 工作流

1. Draft proposal docs before code changes / 改代码前先写提案文档。
2. Wait for user approval / 等待用户批准。
3. Apply one focused change / 实施一次聚焦变更。
4. Update both EN/ZH docs with implementation and results / 用实施细节与结果更新中英文文档。
5. Record test commands and benchmark numbers / 记录测试命令与性能数据。
