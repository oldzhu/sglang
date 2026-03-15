# CHANGE_0036_minicpm_fp8_kv_cache_benchmark_matrix

## 1）背景与动机
- 问题描述：之前对 `--kv-cache-dtype` 是否只影响 MiniCPM prefill、是否同样影响 decode，以及 FP8 KV cache 应该如何做有效评估存在混淆。
- 为什么这件事重要：FP8 KV cache 本质上主要是内存带宽与 KV 容量优化。如果工作负载拆分不对，很容易把结果解读错，或者把瓶颈归因到错误阶段。
- 目标阶段：KV-cache 写入路径、KV-cache 读取路径、prefill/decode 分解、基准方法论

## 2）SOAR 规则合规性检查
- 编写本文档前已重新查阅最新官方页面：`https://soar.openbmb.cn/competition` 与 `https://soar.openbmb.cn/toolkit`。
- 允许原因：比赛页面明确允许推理路径优化、内存/KV 优化以及量化加速；toolkit 的 `技术路径指引` 也明确将 FP8 KV cache 列为可行优化方向。
- 不触碰限制：本文档不修改模型权重、不修改并发配置、不启用 prefix cache，也不改变提交接口；仅用于澄清运行时行为和本地评测方法。
- 对正确性系数 C 的预期影响：文档本身没有影响；但任何 FP8 KV cache 实验都必须重新跑公开正确性集，因为 cache 量化可能影响模型精度。

## 3）技术解释
- `--kv-cache-dtype` 不是只作用于 prefill 的开关。它改变的是 KV cache 本身的存储格式，因此会同时影响：
  - prefill/extend：把新产生的 K/V 张量写入 cache
  - decode：在注意力计算中读取已存储的 K/V 张量
- 在当前 MiniCPM FlashInfer 路径中，prefill 和 decode 使用的是不同 wrapper：
  - decode 使用 `BatchDecodeWithPagedKVCacheWrapper`
  - prefill 使用 `BatchPrefillWithPagedKVCacheWrapper`
- 这两个 wrapper 都会接收 `kv_data_type=self.data_type`，其中 `self.data_type` 来自 `model_runner.kv_cache_dtype`。
- 因此，KV-cache dtype 在概念上是“整份 cache 的格式选择”，而不是某个单独阶段的局部参数。

## 4）为什么当前失败看起来像是 Prefill 问题
- 当前 MiniCPM FlashInfer 实现中，prefill wrapper 是用 `backend="fa2"` 创建的。
- FlashInfer 对该 backend 组合下的 FP8 tensor-core 路径不支持，因此运行时会在 prefill planning 或第一次 prefill 执行时失败。
- 这 **并不意味着** decode 不受 KV-cache dtype 影响，只是当前不兼容的 wrapper 在进入 decode 收益评估之前就先触发了失败。

## 5）按资源与阶段划分的预期影响
### 显存容量
- 通常是最直接、最明显的收益。
- 相比 BF16/FP16，FP8 KV cache 可以将 KV 存储字节数压缩到大约一半。
- 但总进程显存下降不会达到 2 倍，因为权重、激活、workspace 和碎片都不会同比例变化。

### Prefill / extend
- 收益通常较小，甚至可能为 0。
- 原因：prefill 经常主要受 prompt-side GEMM 和 prompt attention 计算支配，而 KV dtype 主要改变的是 cache 写入成本和存储占用。

### Decode
- 通常是 FP8 KV cache 最可能真正带来收益的阶段。
- 原因：decode 会反复读取历史 KV cache，减少每个 cached element 的字节数可以直接降低内存带宽压力。
- 长上下文和更高并发下通常更容易看到收益。

### 精度与稳定性
- FP8 KV cache 存在准确率风险，因为 cached activation 的量化强度高于 BF16/FP16。
- 任何速度或显存收益都必须经过 `perf_public_set.jsonl` 验证后，才能视为 SOAR 可用路径。

## 6）具体 Benchmark Matrix
### Matrix A：启动与兼容性
目标：先判断某个 `attention-backend + kv-cache-dtype` 组合到底能不能跑通。

测试组合：
1. `attention-backend=flashinfer`，`kv-cache-dtype=bfloat16`
2. `attention-backend=flashinfer`，`kv-cache-dtype=fp8_e5m2`
3. `attention-backend=flashinfer`，`kv-cache-dtype=fp8_e4m3`
4. 如果 MiniCPM 支持，再补测非 FlashInfer attention backend 下相同的 KV dtype 组合

记录：
1. 服务是否成功启动
2. 如果失败，是发生在启动阶段、第一次 prefill 请求，还是第一次 decode 请求
3. 具体 assertion/error 文本

解释方式：
1. 如果只有 FP8 + FlashInfer 失败，这是兼容性问题，不是性能结论。
2. 如果其他 backend 可以带 FP8 跑通，那么阻塞点大概率是 MiniCPM FlashInfer 集成，而不是 FP8 KV cache 这个方向本身。

### Matrix B：显存影响
目标：量化 KV-cache dtype 对实际显存占用的影响。

工作负载：
1. 一个固定长上下文请求，或一小批固定长上下文请求
2. 保持模型、prompt、输出长度和并发一致

比较项：
1. `bfloat16`
2. `fp8_e4m3`
3. `fp8_e5m2`，如果能启动

记录：
1. 峰值 GPU memory
2. Prefill 后稳定阶段 GPU memory
3. 若可观测，记录 KV allocator 大小或 cache 占用统计

预期：
1. FP8 下 KV 存储占用应明显下降
2. 总进程显存下降会小于 KV-only 理论比例

### Matrix C：Prefill-heavy 敏感性
目标：单独评估 FP8 KV cache 对 prompt ingestion 的影响。

工作负载：
1. 长输入
2. 很短输出，例如 1 到 16 个生成 token

记录：
1. TTFT
2. 端到端延迟
3. 峰值 GPU memory

预期：
1. 小幅收益、无收益、甚至轻微回退都可能出现
2. 因为 prefill 通常并不是明显的 KV-bandwidth-bound 场景

### Matrix D：Decode-heavy 敏感性
目标：单独评估 FP8 KV cache 最可能发挥作用的阶段。

工作负载：
1. 中长输入，保证 cache 不小
2. 长输出，保证 decode 会反复读取历史 KV

记录：
1. TPOT 或输出 tokens/sec
2. 延迟尾部
3. 峰值 GPU memory
4. 如有条件，补充 profiler 对带宽压力的证据

预期：
1. 这是最可能看到 FP8 KV cache 可测收益的位置
2. 上下文越长、并发越高，越容易体现收益

### Matrix E：端到端 SOAR 代理评测
目标：确认显存和 decode 阶段收益能否传导到实际得分代理。

工作负载：
1. 共享数据集 S1
2. 共享数据集 S8
3. 共享数据集 Smax

记录：
1. `benchmark_duration`
2. `acc_ori` / `overall_accuracy`
3. 峰值 GPU memory
4. 并发下是否出现不稳定

解释方式：
1. 如果 decode 带宽是真瓶颈，收益通常会在 S8 和 Smax 比 S1 更明显。
2. 如果显存下降了但 Smax 仍基本不变，说明更可能被调度、wrapper planning 或权重侧计算所主导。

## 7）建议命令
### 兼容性
```bash
python3 -m sglang.launch_server \
  --model-path <MODEL_PATH> \
  --attention-backend flashinfer \
  --kv-cache-dtype bfloat16
```

```bash
python3 -m sglang.launch_server \
  --model-path <MODEL_PATH> \
  --attention-backend flashinfer \
  --kv-cache-dtype fp8_e5m2
```

```bash
python3 -m sglang.launch_server \
  --model-path <MODEL_PATH> \
  --attention-backend flashinfer \
  --kv-cache-dtype fp8_e4m3
```

### 正确性
```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_PATH> \
  --data_path benchmark/soar/demo_sala/perf_public_set.jsonl \
  --concurrency 32
```

### 端到端速度代理
```bash
export SPEED_DATA_S1=<shared_representative_speed_set.jsonl>
export SPEED_DATA_S8=<shared_representative_speed_set.jsonl>
export SPEED_DATA_SMAX=<shared_representative_speed_set.jsonl>
python3 benchmark/soar/run_soar_suite.py \
  --base-url http://127.0.0.1:30000 \
  --dataset-profile heavy
```

### 显存观测
```bash
nvidia-smi dmon -s mu -d 1
```

## 8）结果汇总模板
| 项目 | BF16 基线 | FP8 候选 | 变化 | 备注 |
|---|---:|---:|---:|---|
| 启动是否成功 | pending | pending | n/a | startup / first prefill / first decode |
| 峰值 GPU memory | pending | pending | pending | |
| Prefill-heavy TTFT | pending | pending | pending | |
| Decode-heavy TPOT | pending | pending | pending | |
| S1 benchmark_duration | pending | pending | pending | |
| S8 benchmark_duration | pending | pending | pending | |
| S∞ benchmark_duration | pending | pending | pending | |
| Accuracy / overall_accuracy | pending | pending | pending | |

## 9）回滚 / 决策指导
1. 如果 FP8 在进入 decode 前就启动失败，应将问题定义为 backend 兼容性，而不是性能不佳。
2. 如果显存下降但速度没有改善，只有在额外容量本身有战略价值时才值得保留。
3. 如果 decode 提速但正确率下降，在精度问题查清之前不应作为 SOAR 提交路径保留。
4. 如果 S8/Smax 稳定改善且正确率安全，FP8 KV cache 就成为值得继续调优的方向。

## 10）下一步建议
- 在修改 MiniCPM FP8 逻辑之前，先用本矩阵把兼容性、显存影响和速度影响拆开评估，避免多因素混淆。
- 如果当前 FlashInfer 路径仍在 prefill 失败，下一个特性应优先补齐 MiniCPM 的显式兼容性 guardrail 或受支持 fallback，而不是继续让运行时在 FlashInfer 深处崩溃。