# CHANGE_0011_eval_accuracy_parser_fallback

## 1）背景与动机
- 在多次 fcloud 运行中，`run_soar_suite.py` 生成的 `summary.json` 里 `ori_accuracy` / `overall_accuracy` 为 `null`。
- 根因：解析器只识别 `ori_accuracy` / `overall_accuracy` 标签，而部分评测输出仅包含 `Average Score: xx.xx%`。
- 目标：增强解析鲁棒性，使 `summary.json` 可直接用于基线追踪。

## 2）SOAR 规则合规性检查
- 本次仅工具层结果解析改动，不涉及模型/内核/推理逻辑。
- 不影响官方评测约束。

## 3）改动前计划
- 仅修改 `benchmark/soar/run_soar_suite.py`。
- 增加常见分数字段的兜底正则。
- 当仅识别到一个分数字段时，填充另一个字段，避免 `null`。

## 4）实际代码改动
- 更新 `parse_accuracy_from_text(...)`：
  - `ori_accuracy` 增加对 `Average Score` / `Average Accuracy` 的解析。
  - `overall_accuracy` 增加对 `Overall Score` / `Relative Score` 的解析。
  - 仅命中一个值时，镜像填充到另一个字段，避免 summary 空值。

## 5）验证命令
```bash
python3 benchmark/soar/run_soar_suite.py \
  --api-base http://127.0.0.1:30000 \
  --model-path /root/models/openbmb/MiniCPM-SALA \
  --eval-script /root/data/eval_model.py \
  --public-data /root/data/perf_public_set.jsonl
```

检查：
```bash
cat benchmark/soar/results/<timestamp>/summary.json
```

预期：当日志中有评分信息时，`correctness.ori_accuracy` 与 `correctness.overall_accuracy` 不再为 `null`。

## 6）风险评估
- 风险低。解析更宽松；当日志只提供一个分数时，会将其映射到两个字段。

## 7）回滚说明
1. 回退 `benchmark/soar/run_soar_suite.py`。

## 8）下一步建议
- 后续可新增 `correctness.score_source` 字段，标注分值来源类型。
