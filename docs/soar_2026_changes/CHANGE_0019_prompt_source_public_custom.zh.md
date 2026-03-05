# CHANGE_0019_prompt_source_public_custom

## 1) 背景与动机
- 用户希望基准数据支持从公开评测 JSONL 或自定义 JSONL 读取 prompt，提升数据真实性。
- 目标：减少纯合成文本的重复性，让本地优化结果更贴近真实负载。

## 2) SOAR 规则合规性
- 本次为工具层数据生成改造。
- 不涉及模型/算子/推理算法改动。

## 3) 变更前实施计划
- 扩展 `benchmark/soar/generate_speed_datasets.py` 增加来源模式与字段参数。
- 保留现有 tokenizer 控长逻辑和 profile 机制。

## 4) 实际代码改动
- 新增参数：
  - `--prompt-source synthetic|public|custom`
  - `--source-jsonl`
  - `--source-prompt-field`（默认 `question`）
  - `--source-response-field`（可选）
- 新增 JSONL 文本池加载与校验。
- 生成逻辑：
  - public/custom 模式：从文本池采样基础 prompt，再按目标 token 数拉伸/裁剪。
  - synthetic 模式：保持原有合成短语模式。
- 可选支持从 `--source-response-field` 构造响应种子池。

## 5) 使用示例
```bash
# 使用公开评测集 prompt
python3 benchmark/soar/generate_speed_datasets.py \
  --prompt-source public \
  --source-jsonl /root/data/perf_public_set.jsonl \
  --source-prompt-field question \
  --model-path /root/models/openbmb/MiniCPM-SALA \
  --profile quick10 \
  --output-dir /root/soar_fast_data

# 使用自定义 jsonl prompt
python3 benchmark/soar/generate_speed_datasets.py \
  --prompt-source custom \
  --source-jsonl /root/data/my_prompts.jsonl \
  --source-prompt-field question \
  --model-path /root/models/openbmb/MiniCPM-SALA \
  --profile quick10 \
  --output-dir /root/soar_fast_data
```

## 6) 风险
- 低。对于空文件或字段缺失会快速失败并给出明确报错。

## 7) 回滚方法
1. 回退 `benchmark/soar/generate_speed_datasets.py`。

## 8) 下一步建议
- 后续可增加可选的去重与按长度分桶采样。
