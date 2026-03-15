# CHANGE_0038_minicpm_fp8_sparse_topk_bf16_bridge

## 1）背景与动机
- 问题描述：在修正 MiniCPM FlashInfer 的 FP8 dtype/backend 合同之后，第一次真实请求仍然在 sparse top-k scoring 阶段失败，报错为 `RuntimeError: FlashAttention only support fp16 and bf16 data type`。
- 为什么重要：这说明当前活跃阻塞点已经不再是 FlashInfer 启动或 wrapper planning，而是 MiniCPM 独立的 sparse-routing scorer。
- 本轮目标：保留真实 KV cache 和注意力主路径的 FP8，同时让 sparse top-k scorer 临时走 BF16，从而绕过其不支持 FP8 的 kernel 限制。

## 2）SOAR 规则合规性检查
- 本次修改前已重新核对官方最新页面：`https://soar.openbmb.cn/competition` 与 `https://soar.openbmb.cn/toolkit`。
- 合规原因：这是推理栈内部的 runtime/kernel 兼容性优化，属于官方鼓励的 KV-cache 优化方向之内。
- 未触碰限制：不改模型权重、不重开 prefix cache、不改并发逻辑、不改提交接口。
- 精度/稳定性风险：低到中等。该桥接只作用于 sparse scoring 的 scratch 路径，但仍需重新验证正确性。

## 3）修改前的详细实施计划
本轮严格限制为一个特性：

1. 在 MiniCPM sparse scoring 中检测 FP8 KV-cache 模式。
2. 仅对 sparse top-k scorer 的输入做 BF16 cast。
3. 真实 KV cache 与 FlashInfer 注意力主路径继续保持 FP8。

计划修改的文件/函数：

1. `python/sglang/srt/layers/attention/minicpm_backend.py`
   - `sparse_get_topk_impl(...)`

预期收益：

1. 解除 FP8 KV cache 下 prefill/decode sparse routing 的阻塞。
2. 保留 FP8 KV-cache 的核心显存/decode 收益。
3. 把 BF16 fallback 严格限定在当前不支持 FP8 的 kernel 路径上。

## 4）实际代码修改
修改文件：

1. `python/sglang/srt/layers/attention/minicpm_backend.py`

实际改动：

1. 在 `sparse_get_topk_impl(...)` 中，当 `kv_cache_dtype` 以 `fp8` 开头时，把以下张量在进入 sparse top-k scorer 之前转换为 `torch.bfloat16`：
   - `query_layer`
   - `compressed_k`
   - `compressed_k2`

保持不变的部分：

1. 真实 KV-cache 存储仍然是 FP8。
2. FlashInfer prefill/decode 注意力主路径仍然保持 FP8 感知。
3. sparse scorer 的输出仍然是相同语义的 `topk_idx` 路由索引。

## 5）为什么不需要再 Cast 回 FP8
这个 BF16 bridge 只用于 sparse scoring 的临时输入。

1. scorer 读取 query/compressed-key 张量。
2. 它输出的是 `topk_idx` block 选择结果。
3. 它不会回写 KV cache。

因此数据流是：

1. KV cache 真实存储为 FP8
2. sparse scoring 临时使用 BF16 张量
3. 产生路由索引
4. 实际注意力继续走 FP8 KV-cache 主路径

这不是对真实 cache 内容做 `FP8 -> BF16 -> FP8` 的往返转换。

## 6）验证命令
语法 / 导入验证：

```bash
python3 -m compileall \
  python/sglang/srt/layers/attention/minicpm_backend.py
```

建议的首次运行验证：

```bash
export SGLANG_MINICPM_FLASHINFER_PREFILL_BACKEND=auto
python3 -m sglang.launch_server \
  --model-path <MODEL_PATH> \
  --attention-backend minicpm_flashinfer \
  --kv-cache-dtype fp8_e5m2 \
  --disable-cuda-graph
```

成功启动后的正确性验证：

```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_PATH> \
  --data_path benchmark/soar/demo_sala/perf_public_set.jsonl \
  --concurrency 32
```

成功启动后的速度代理验证：

```bash
python3 benchmark/soar/run_soar_suite.py \
  --base-url http://127.0.0.1:30000 \
  --dataset-profile heavy
```

## 7）结果汇总表
| 项目 | 基线 | 新方案 | 变化 | 备注 |
|---|---:|---:|---:|---|
| FlashInfer + FP8 KV 启动 | 已能到达首个请求 | 不变 | n/a | 上一轮已解决 |
| Sparse top-k scorer dtype | 可能收到 FP8 张量 | 在 FP8 KV 模式下强制 BF16 | n/a | 本地兼容性桥接 |
| 真实 KV-cache 存储 | FP8 | FP8 | 无变化 | 保持不变 |
| 首请求 sparse scorer 崩溃 | 修改前存在 | 待运行验证 | pending | 本轮目标 |

## 8）回滚说明
若本轮改动无效或引入回退：

1. 回退 `python/sglang/srt/layers/attention/minicpm_backend.py` 中的 BF16 bridge 逻辑。
2. 重新运行 FP8 启动，确认旧的 sparse scorer 失败能够复现。

## 9）下一步建议
1. 用之前同一个会触发失败的首请求重新测试。
2. 如果 scorer 仍失败，继续看下一个报错位置，确认是否还有其他 compression 子路径仍收到 FP8。
3. 如果通过，立刻转入正确性和 decode-heavy 速度验证，判断 FP8 KV cache 是否仍然值得保留。