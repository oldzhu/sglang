## 背景与动机

MiniCPM FP8 decode 现在已经能够进入 CUDA graph replay 阶段，当前故障表现为 replay 期间的 illegal memory access，而不是更早的 Python 侧 sparse scorer 类型错误。排查 decode 元数据路径时，发现 `python/sglang/srt/mem_cache/common.py` 中 decode 阶段的 sparse table 写入使用了 `(start, end)` 元组索引，而不是预期的区间写入 `slice(start, end)`。

这个差异在 PyTorch 中有实质性语义区别：

- `table[row, slice(3, 4)] = [999]` 只会写入索引 `3`
- `table[row, (3, 4)] = [999]` 会选择两个显式位置 `3` 和 `4`，并把数值广播写入

对于 decode，`token_per_req=1`，因此很多 sparse decode 更新本意只是追加一个新的 sparse 项。若使用元组索引，就可能额外覆盖下一个槽位，悄悄破坏 `req_to_sparse_k1_token` / `req_to_sparse_k2_token`。当这些被污染的表再次被 decode replay 内核消费时，最终就可能在 graph replay 中表现为 CUDA illegal access。

## 规则合规说明

这次变更是 sparse decode 元数据处理的运行时正确性修复，不修改模型权重、不改变评测并发、不涉及违规 cache 行为，也不改变提交接口，符合 SOAR 约束。

本轮优化线程此前已复核过相关规则与 toolkit 页面，本变更本身不依赖任何规则敏感的绕过方式。

## 详细实现计划

1. 将 decode 阶段 sparse table 写入从元组索引修正为 `slice(start, end)`。
2. 为 sparse k1/k2 的 decode 写入增加临时边界检查，使越界尽早在 Python 侧报错，而不是晚些时候以异步 CUDA illegal access 的形式出现。
3. 将这些边界检查作为当前 FP8 decode 调试与稳定化阶段的临时保护。
4. 如果 crash 完全修复且性能稳定，后续可再评估移除这些显式边界检查，以避免热路径中的额外 Python 开销。

## 实际代码修改

修改文件：

- `python/sglang/srt/mem_cache/common.py`

行为变化：

- 将 decode sparse k1 写入从元组索引改为 `slice(k1_len, k1_end)`。
- 将 decode sparse k2 写入从元组索引改为 `slice(k2_len, k2_end)`。
- 增加针对 `req_to_sparse_k1_token.shape[1]` 与 `req_to_sparse_k2_token.shape[1]` 的临时显式边界检查。
- 错误信息中包含 request index、计算出的 sparse 偏移、追加 token 数量以及 table 宽度，便于定位。

## 验证命令

在服务端进程中打开阻塞式 CUDA 栈定位：

```bash
CUDA_LAUNCH_BLOCKING=1 python3 -m sglang.launch_server ...
```

随后单独运行 correctness 客户端：

```bash
python3 eval_model.py --api_base http://127.0.0.1:30000 --model_path <MODEL_PATH> --data_path benchmark/soar/demo_sala/perf_public_set.jsonl --concurrency 32
```

工作区静态验证：

```bash
python3 -m compileall python/sglang/srt/mem_cache/common.py
```

正确性稳定后再做 serving sanity：

```bash
python3 benchmark/soar/demo_sala/run_serving_bench.py ...
```

## 结果汇总表

| 项目 | CHANGE_0040 前基线 | CHANGE_0040 后 |
| --- | --- | --- |
| Decode sparse 写入语义 | 元组索引，可能多写相邻槽位 | 精确 slice 写入目标区间 |
| Sparse decode 越界处理 | 可能延后表现为 CUDA illegal access | 提前在 Python 侧确定性报错 |
| Correctness 评测稳定性 | 当前在 decode CUDA graph replay 崩溃 | 待用户验证 |
| Serving 性能 | FP8 路径已有较好结果 | 待修复后复测 |

## 回滚说明

如果后续证据表明该变更有害：

1. 回滚 `python/sglang/srt/mem_cache/common.py` 中的 `slice(...)` 和临时边界检查改动。
2. 使用相同的 `CUDA_LAUNCH_BLOCKING=1` correctness 命令再次复现，对比行为差异。
3. 只有在证据显示该修复本身引入回归时，才保留回滚。

默认建议：不要因为 crash 仍然存在就自动回滚。如果修复逻辑正确且无害，应保留它，同时继续缩小剩余 decode CUDA graph 故障范围。

## 后续建议

1. 在 sglang 服务端继续保持 `CUDA_LAUNCH_BLOCKING=1`，带着本修复重新复现 FP8 correctness。
2. 如果 crash 仍在，对比 CHANGE_0040 前后的 blocking stack，判断剩余问题更偏向元数据还是 replay 内核本身。
3. 当 crash 完全修复后，再评估是否移除这些临时边界检查以做最终比赛性能调优。