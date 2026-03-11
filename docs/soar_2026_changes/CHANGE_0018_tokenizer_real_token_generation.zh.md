# CHANGE_0018_tokenizer_real_token_generation

## 1）背景与动机
- 仅按短语数量拼接文本会偏离真实 token 开销，导致 benchmark 时长与负载判断失真。
- 目标：按真实 tokenizer token 长度生成测试样本。

## 2）SOAR 规则合规性检查
- 本次仅 benchmark 数据生成工具改动。
- 不涉及模型/推理内核/运行时算法改动。

## 3）改动前计划
- 仅修改 `benchmark/soar/generate_speed_datasets.py`。
- 将生成终止条件从“短语计数”改为“tokenizer 实际 token 数”。

## 4）实际代码改动
- 新增 tokenizer 加载：`AutoTokenizer.from_pretrained(..., trust_remote_code=True)`。
- 新增参数 `--model-path`（未设置时回退到 `OpenBMB/MiniCPM-SALA`）。
- 文本生成循环改为按真实 token 数终止（`tokenizer.encode(..., add_special_tokens=False)`）。
- 保留现有 profile 和行数/耗时估算控制。

## 5）使用方式
```bash
python3 benchmark/soar/generate_speed_datasets.py \
  --profile quick10 \
  --model-path /root/models/openbmb/MiniCPM-SALA \
  --output-dir /root/soar_fast_data
```

## 6）风险评估
- 风险低。由于循环中引入 tokenization，生成速度可能略慢。

## 7）回滚说明
1. 回退 `benchmark/soar/generate_speed_datasets.py`。

## 8）下一步建议
- 后续可增加“从本地真实 JSONL prompt 语料采样”的模式，进一步提升负载真实性。
