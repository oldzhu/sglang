# CHANGE_0017_sparse_empty_max_guard

## 1）背景与动机
- 在 MiniCPM 后端元数据初始化阶段出现早期崩溃：
  - `RuntimeError: max(): Expected reduction dim to be specified for input.numel() == 0`
- 根因：当稀疏序列张量为空时仍调用了 `seqlen_q_sparse_tensor.max()`。

## 2）SOAR 规则合规性检查
- 仅鲁棒性修复。
- 不涉及模型/内核算法改动，仅处理空张量边界情况。

## 3）改动前计划
- 仅修改 `python/sglang/srt/layers/attention/minicpm_backend.py` 一处。
- 在 `max()` 前增加空张量判断。

## 4）实际代码改动
- 新增条件分支：
  - 若 `seqlen_q_sparse_tensor.numel() == 0`，设置 `metadata.max_seqlen_q_adjusted = 0`
  - 否则保留原逻辑 `max().item() * heads_per_group`

## 5）验证命令
```bash
# 重启后复现此前路径
python3 benchmark/soar/run_soar_suite.py \
  --api-base http://127.0.0.1:30000 \
  --model-path /root/models/openbmb/MiniCPM-SALA \
  --speed-data-s8 /root/soar_fast_data/speed_s8.jsonl
```

## 6）风险评估
- 风险极低。仅影响“稀疏张量为空”的边界分支。

## 7）回滚说明
1. 回退 `minicpm_backend.py` 中本次 guard 分支。

## 8）下一步建议
- 如需进一步诊断，可在后续变更中增加 empty sparse-batch 计数日志。
