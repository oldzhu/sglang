# CHANGE_0033_minicpm_cudagraph_sparse_length_sync

## 状态更新
- 当前状态：已在运行时代码分支中回滚。
- 回滚原因：共享数据集 smoke bench 没有显示出有意义的速度收益，因此代码路径已恢复为基线行为，本文档仅保留为后续可能重新评估时的记录。

## 1）背景与动机
- 问题描述：MiniCPM 的 CUDA-graph decode 路径虽然在 capture 时固定了 FlashInfer sparse wrapper 的存储，但 replay 仍然按捕获时的“满 top-k”长度做规划，而不是按当前 batch 的真实 sparse 累积长度做规划。
- 为什么该改动预期会提速：如果补齐行或较短请求被当成满 top-k 行来规划，FlashInfer 会把实际不存在的工作量也算进去，这类额外开销通常会在 S8 和 Smax 下更明显地侵蚀 decode 快路径。
- 目标阶段：decode / kernel launch / scheduling

## 2）SOAR 规则合规性检查
- 允许原因：这是现有 CUDA-graph decode 路径中的运行时元数据同步修正，不改变提交接口、不改变官方评测并发，也不有意改变模型输出。
- 不触碰限制（prefix cache/固定并发/可复现性）：没有引入 prefix cache，没有修改官方并发设置，replay 元数据完全由当前 batch 确定性生成。
- 对正确性系数 C 的预期影响：理论上中性；改动只是让 FlashInfer 的规划元数据与已有 sparse 长度保持一致，并把空的补齐行标记为空。

## 3）改动前实施计划
- 计划修改的文件/函数：`python/sglang/srt/layers/attention/minicpm_backend.py` 中的 CUDA-graph replay 元数据准备逻辑，以及 `python/sglang/srt/layers/attention/minicpm_sparse_kernels.py` 中 sparse page table 到 FlashInfer 格式的转换逻辑。
- 最小化 diff 策略：只修 replay 时 FlashInfer 规划张量的同步和空行序列化行为，不改模型逻辑、不改调度策略、不改外部接口。
- 回滚方案：回退本次补丁，恢复之前的占位 `kv_indptr` replay 行为，以及无条件 `kv_last_page_len=1` 的转换逻辑。

## 4）实际代码改动（获批后填写）
- 补丁摘要：replay 现在会在 `begin_forward()` 之前把 `forward_batch.sparse_cu_seqlens_k_cpu` 复制到 FlashInfer 规划用的 `kv_indptr` 缓冲区，并且把尾部补齐为最后一个真实 offset，而不是 capture 容量；同时让补齐空行保持 `kv_last_page_len=0`。
- 最终修改文件：`python/sglang/srt/layers/attention/minicpm_backend.py`、`python/sglang/srt/layers/attention/minicpm_sparse_kernels.py`。
- 核心逻辑变化：
  - FlashInfer 的 CUDA-graph 规划不再默认每个 sparse 行都占满 `num_sparse_topk_tokens`。
  - 空的补齐 sparse 行在 sparse-to-FlashInfer 转换时会被序列化为空行。
  - CUDA-graph 相关注释已更新为与当前 MiniCPM 图执行流程一致。

## 5）验证命令
### 正确性
```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_PATH> \
  --data_path benchmark/soar/demo_sala/perf_public_set.jsonl \
  --concurrency 32
```

### 速度
```bash
export SPEED_DATA_S1=<shared_representative_speed_set.jsonl>
export SPEED_DATA_S8=<shared_representative_speed_set.jsonl>
export SPEED_DATA_SMAX=<shared_representative_speed_set.jsonl>
python3 benchmark/soar/run_soar_suite.py \
  --base-url http://127.0.0.1:30000 \
  --dataset-profile heavy
```

## 6）结果汇总
| 指标 | 基线 | 新方案 | 变化 |
|---|---:|---:|---:|
| Accuracy / overall_accuracy | 未重跑 | 未重跑 | n/a |
| 共享数据 S1 吞吐 | 143.04 | 145.29 | +1.6%（仅 smoke） |
| 共享数据 S8 吞吐 | 31.44 | 31.50 | +0.2%（仅 smoke） |
| 共享数据 S∞ 吞吐 | 25.90 | 25.97 | +0.3%（仅 smoke） |

- 结论：这些变化幅度过小，不足以支撑保留该运行时改动，因此代码已回滚，文档作为实验记录保留。

## 7）风险评估
- 准确率风险：低；改动只是在 replay 时同步已有 sparse 长度对应的元数据。
- 稳定性风险：中低；修改点位于 CUDA-graph replay 路径，验证时应重点覆盖带 padding 的 decode batch，以及长短请求混合场景。
- 可复现风险：低；没有新增随机逻辑，也没有新增依赖环境差异的分支。

## 8）回滚说明
1. 回退 `python/sglang/srt/layers/attention/minicpm_backend.py` 中把实时 sparse 累积长度复制到 FlashInfer replay 缓冲区的改动。
2. 回退 `python/sglang/srt/layers/attention/minicpm_sparse_kernels.py` 中把空行序列化为 `kv_last_page_len=0` 的改动。
3. 保留本文档作为历史记录；如果后续 profiling 再次显示 wrapper planning 与真实 sparse 工作量不匹配，可重新评估该方向。

## 9）下一步建议
- 在共享代表性数据集上重新跑 S1/S8/Smax，对比当前分支基线的 graph 命中情况和 benchmark_duration。
- 如果 S8/Smax 仍明显偏离预期，再继续检查 MiniCPM decode 是否还因为 graph batch-size 选择或其他 replay 形状守卫而回退。