# CHANGE_0016_sparse_page_table_copy_bound

## 1）背景与动机
- 在 S8 benchmark 中，MiniCPM 后端出现运行时错误：
  - 将 dense page table 拷贝到 sparse page table 时发生张量尺寸不匹配。
- 日志表明该分支中 `kv_len` 可能超过源/目标表的实际宽度。

## 2）SOAR 规则合规性检查
- 属于推理路径鲁棒性修复。
- 不涉及违规评测手段。
- 不涉及模型替换。

## 3）改动前计划
- 仅修改 `python/sglang/srt/layers/attention/minicpm_backend.py` 一处逻辑。
- 将拷贝长度限制在源/目标共同容量范围内。

## 4）实际代码改动
- 将原先 `:kv_len` 的直接赋值改为有界 `copy_len`：
  - `copy_len = min(kv_len, sparse_page_table.shape[1], page_table.shape[1])`
- 两个 dense head-group 赋值均使用 `copy_len`。

## 5）修复原因说明
- 当 `kv_len` 大于源或目标的表宽时，直接赋值会触发 shape mismatch。
- 有界拷贝可避免越界尺寸赋值，同时在正常范围内行为不变。

## 6）验证命令
```bash
# 重启后执行 S8 档位
python3 benchmark/soar/run_soar_suite.py \
  --api-base http://127.0.0.1:30000 \
  --model-path /root/models/openbmb/MiniCPM-SALA \
  --speed-data-s8 /root/soar_fast_data/speed_s8.jsonl
```

## 7）风险评估
- 风险低。仅将拷贝限制在张量容量范围内。

## 8）回滚说明
1. 回退 `minicpm_backend.py` 中本次修改。

## 9）下一步建议
- 若需要，可在后续单独变更中增加“发生拷贝截断次数”的调试计数。
