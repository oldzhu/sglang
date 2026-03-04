# CHANGE_0013_fast_iteration_time_control

## 1）背景与动机
- 现有生成的 benchmark 数据集可能过重，不利于快速迭代。
- 用户希望更容易控制时长，使每个档位尽量在约 10 分钟内完成（用于调参闭环）。

## 2）SOAR 规则合规性检查
- 本次仅数据生成工具改动。
- 不涉及模型/内核/推理路径代码。

## 3）改动前计划
- 仅修改 `benchmark/soar/generate_speed_datasets.py`。
- 增加档位预设与每档样本数覆盖参数。
- 增加粗略耗时估算输出，便于快速规划。

## 4）实际代码改动
- 新增 `--profile`：
  - `quick10`（默认，轻量快速迭代）
  - `balanced`
  - `heavy`
- 新增每档样本数参数：
  - `--rows-s1`、`--rows-s8`、`--rows-smax`
- 新增粗略耗时估算参数与输出：
  - `--estimate-input-tps`（默认 `8000`）
  - `--estimate-output-tps`（默认 `250`）
  - 输出每档及总计预计分钟数。

## 5）使用示例
```bash
# 快速迭代默认配置
python3 benchmark/soar/generate_speed_datasets.py --profile quick10 --output-dir /root/soar_fast_data

# 更快回环：进一步降低样本数
python3 benchmark/soar/generate_speed_datasets.py \
  --profile quick10 \
  --rows-s1 12 --rows-s8 16 --rows-smax 20 \
  --output-dir /root/soar_fast_data

# 更重压测
python3 benchmark/soar/generate_speed_datasets.py --profile balanced --output-dir /root/soar_balanced_data
```

## 6）风险评估
- 风险低。耗时估算为近似值，实际时长仍取决于真实吞吐。

## 7）回滚说明
1. 回退 `benchmark/soar/generate_speed_datasets.py`。

## 8）下一步建议
- 后续可增加“短跑自动校准”功能，用真实 pilot 结果自动反推样本规模。
