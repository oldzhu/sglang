# CHANGE_0039_minicpm_fp8_sparse_compression_scratch_path

## 1）背景与动机
- 问题描述：在启用 MiniCPM FP8 KV cache 并修正 FlashInfer / sparse-scoring dtype 问题之后，正确性评测仍会在 decode 准备阶段报异步 `CUDA illegal memory access`。
- 诊断结论：当前最强可疑点是 MiniCPM 的 compressed-K 维护 kernel。它会通过共享 KV-cache 指针进行读写；而在 FP8 KV 模式下，这个共享指针对应的是 FP8-backed 的 GPU 全局显存，但该 compression kernel 仍按 BF16 风格来处理。
- 本轮目标：在 FP8 KV 模式下，不再通过共享 FP8 KV-cache 路径维护 sparse compressed-K，而是直接把 compressed K1/K2 重算到 BF16 scratch buffer 中。

## 2）SOAR 规则合规性检查
- 本次修改前已重新核对官方最新页面：`https://soar.openbmb.cn/competition` 与 `https://soar.openbmb.cn/toolkit`。
- 合规原因：这是运行时/kernel 稳定性优化，属于允许的推理路径与 KV-cache 优化范围。
- 未触碰限制：不改模型权重、不重开 prefix cache、不改并发逻辑、不改提交接口。
- 精度/稳定性风险：中低。Scratch path 改变的是 sparse routing 特征的物化方式，因此仍需重新做正确性验证。

## 3）设计决策记录
本轮前比较了两个方向：

1. **BF16 scratch compressed-K 路径**
   - 真实 KV cache 继续保持 FP8。
   - compressed K1/K2 重算到 BF16 scratch buffer。
   - sparse top-k routing 从这些 BF16 scratch buffer 读取。
   - 优点：实现风险最小，最容易先稳定下来。

2. **真正的 FP8-aware compressed-K 路径**
   - 当 KV cache 是 FP8 时，compressed K1/K2 也保持 FP8。
   - 需要完整的 scale-aware quantize/store/dequantize 语义。
   - 需要 compressed-K 的所有 reader / writer 都支持 FP8。
   - 暂缓到后续设计/实现，因为范围更大、风险更高。

本轮决策：

1. 先实现 BF16 scratch compressed-K 路径。
2. 把真正的 FP8-aware compressed-K 作为后续设计选项保留。

## 4）Compressed-K 是什么，用来做什么
MiniCPM 的 sparse attention 并不是每次都对全历史做全量 dense attention。它会先构造 compressed K1/K2 摘要来做粗粒度路由：

1. compressed K1/K2 是对历史 key 的两档池化表示
2. sparse top-k scoring 利用它们来选择更可能相关的 block
3. 后续 sparse attention 再只在被选中的 block 上做更细的计算

因此，compressed K 的角色是路由 / block 选择辅助，而不是主注意力 kernel 最终消费的 KV-cache 存储格式。

## 5）为什么 Dtype 会影响准确率
即使主 KV cache 已经是 FP8，把 compressed K 从 BF16 改成 FP8 仍然可能影响准确率：

1. compressed K 参与的是 block 选择，不只是最终 value lookup
2. routing 对分数排序非常敏感，尤其是在 top-k 边界附近
3. compressed K 额外量化噪声可能导致选择到不同的 sparse block
4. 被选中的 sparse block 不同，最终注意力上下文和答案质量也可能变化

这也是为什么未来若要做真正的 FP8 compressed-K 设计，必须同时评估正确性，而不能只看速度。

## 6）修改前的详细实施计划
计划修改文件/函数：

1. `python/sglang/srt/layers/attention/minicpm_sparse_kernels.py`
   - 新增 scratch-only compression kernel，只写 `full_compressed_k`
2. `python/sglang/srt/layers/attention/minicpm_sparse_utils.py`
   - 给 compression helper 增加 `scratch_only` 开关
3. `python/sglang/srt/layers/attention/minicpm_backend.py`
   - 在 FP8 KV 模式下自动启用 scratch-only compression

预期收益：

1. 去掉最可疑的共享 FP8 KV 变异写入路径
2. 提高 FP8 KV cache 下的正确性稳定性
3. 尽量保留当前已经出现的较强 serving 性能

## 7）实际代码修改
修改文件：

1. `python/sglang/srt/layers/attention/minicpm_sparse_kernels.py`
2. `python/sglang/srt/layers/attention/minicpm_sparse_utils.py`
3. `python/sglang/srt/layers/attention/minicpm_backend.py`

实际改动：

1. 新增 scratch-only Triton kernel，直接从 base token table 重算 compressed K，并写入 `full_compressed_k`。
2. 为 `get_compress_k_v2`、`get_compress_k_v2_padded`、`allocate_and_compress_keys` 增加 `scratch_only` 支持。
3. 当 MiniCPM 运行在 FP8 KV 模式下时，自动切换到 scratch-only compression 路径。

## 8）验证命令
语法 / 导入验证：

```bash
python3 -m compileall \
  python/sglang/srt/layers/attention/minicpm_sparse_kernels.py \
  python/sglang/srt/layers/attention/minicpm_sparse_utils.py \
  python/sglang/srt/layers/attention/minicpm_backend.py
```

建议运行验证：

```bash
CUDA_LAUNCH_BLOCKING=1 python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_PATH> \
  --data_path benchmark/soar/demo_sala/perf_public_set.jsonl \
  --concurrency 32
```

```bash
python3 benchmark/soar/run_soar_suite.py \
  --base-url http://127.0.0.1:30000 \
  --dataset-profile heavy
```

## 9）结果汇总表
| 项目 | 基线 | 新方案 | 变化 | 备注 |
|---|---:|---:|---:|---|
| Sparse compression 是否会改写共享 KV | FP8 模式下存在 | FP8 模式下移除 | n/a | 本轮核心设计变化 |
| Compressed-K 物化方式 | 共享 KV 支撑 | FP8 模式下改为 BF16 scratch | n/a | 以稳定性优先 |
| 正确性评测崩溃 | 修改前存在 | 待验证 | pending | 本轮目标 |
| 未来 FP8 compressed-K 设计 | 概念阶段 | 已记录为设计备选 | n/a | 暂缓实现 |

## 10）回滚说明
1. 回退 scratch-only compression kernel 与 helper 开关。
2. 回退 MiniCPM backend 中 FP8 模式自动启用 scratch-only compression 的逻辑。
3. 重新运行 FP8 路径，确认旧行为能够复现。

## 11）下一步建议
1. 先用 `CUDA_LAUNCH_BLOCKING=1` 重新跑正确性评测，确认 illegal access 是否消失。
2. 对比本轮修改前后的 serving benchmark，量化 scratch path 的开销。
3. 若稳定且速度损失可接受，则保留 BF16 scratch 路径。
4. 若速度损失过大，再回到暂缓的真正 FP8-aware compressed-K 设计。