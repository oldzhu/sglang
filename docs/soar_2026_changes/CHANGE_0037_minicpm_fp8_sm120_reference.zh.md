# CHANGE_0037_minicpm_fp8_sm120_reference

## 1）背景与动机
- 问题描述：当前 MiniCPM FlashInfer 路径在 `--kv-cache-dtype` 设置为 FP8 时会失败，最初报错为 `fp8 tensor core is not supported in fa2 backend`；在临时改 backend 后，又进一步暴露出生成 all-FP8 kernel 的编译失败。
- 为什么重要：SOAR toolkit 的 `技术路径指引` 已明确将 FP8 KV cache 列为 MiniCPM-SALA 的推荐优化方向之一，尤其适用于 decode-heavy、长上下文场景。
- 本次额外参考：用户提供了一份 NVIDIA RTX PRO / SM120 优化 PDF 的文字提取。该提取说明了 SM120 具备原生 FP8 tensor core 能力、优化风格接近 Hopper、支持 TMA，且显存带宽很强。这强化了当前失败更像是“软件路径不匹配”，而不是“硬件不支持”。
- 本轮目标：把新的诊断整理成文档，并实现一个范围收敛的 MiniCPM FlashInfer 合同修正，使 FP8 KV-cache 的后续实验建立在更合理的路径上。

## 2）SOAR 规则合规性检查
- 在本次修改前已重新核对官方最新页面：`https://soar.openbmb.cn/competition` 与 `https://soar.openbmb.cn/toolkit`。
- 合规原因：比赛允许推理优化、KV/内存优化、kernel/backend 调优以及量化加速；toolkit `技术路径指引` 也明确给出 `FP8 KV Cache` 作为允许的优化方向。
- 未触碰限制：本次修改不改模型权重、不重开 prefix cache、不改变官方并发逻辑，改动仅限于运行时 backend 集成层。
- 精度/稳定性风险：中等。FP8 KV cache 仍需重新做正确性验证，因为若缩放或 backend 选择错误，可能影响回答质量。

## 3）硬件参考结论
用户提供的 PDF 提取可以归纳为四个实践结论：

1. SM120 是支持 FP8 的硬件目标，并不是硬件层面禁止 FP8。
2. 其优化风格接近 Hopper，因此成熟的 tensor-core / memory-path 优化理论上可以迁移。
3. FP8 吞吐明显高于 BF16/FP16，因此如果软件栈走对 kernel 路径，decode 侧出现 KV 带宽收益是合理预期。
4. 一旦链路跑通，后续性能验证应优先使用 Nsight Systems，再用 Nsight Compute 深挖具体 kernel。

## 4）重新思考后的根因判断
目前更合理的工作假设是：

1. `fp8 tensor core is not supported in fa2 backend` 是 backend 路径限制，不是 SM120 硬件不支持 FP8 KV cache。
2. 之后把 `fa2 -> fa3` 强行替换后出现的 JIT 失败，说明 MiniCPM 路径依然在生成错误的 dtype 合同，很可能是在请求 all-FP8 attention kernel，而不是 `BF16 query/output + FP8 KV cache`。
3. MiniCPM 自己的 FlashInfer 集成和通用 SGLang FlashInfer backend 在几个关键点上发生了偏离。

在本补丁之前，MiniCPM 路径的主要疑点有：

1. prefill wrapper 被硬编码为 `backend="fa2"`
2. `q_data_type` 被设置成 KV-cache dtype，而不是模型 dtype
3. 当启用 FP8 KV cache 时，MiniCPM backend 会主动把 query 侧张量 cast 到 KV dtype
4. FlashInfer wrapper `forward(...)` 调用没有像通用 backend 一样传入 `k_scale` / `v_scale`

## 5）修改前的详细实施计划
本轮严格保持一个特性目标：让 MiniCPM FlashInfer 的 KV-cache FP8 合同与通用 backend 对齐。

计划修改：

1. 在 `python/sglang/srt/layers/attention/minicpm_attention_kernels.py`
   - 让 `q_data_type` 保持模型 dtype
   - 增加可配置的 prefill backend 选择入口
   - 在 FlashInfer wrapper forward 调用中补齐 `k_scale` 与 `v_scale`
2. 在 `python/sglang/srt/layers/attention/minicpm_backend.py`
   - 停止在启用 FP8 KV cache 时把 `q`、`q_rope`、`k_rope` cast 到 KV-cache dtype
   - 保留 descale / scaling 元数据准备逻辑

预期收益：

1. 去掉 MiniCPM 集成层错误的 all-FP8 合同。
2. 无需修改 site-packages，就能测试 `auto` 或用户指定 backend。
3. 提高 FlashInfer 选择或 JIT 到正确 `BF16 Q/O + FP8 KV` kernel 家族的概率。

## 6）实际代码修改
修改文件：

1. `python/sglang/srt/layers/attention/minicpm_attention_kernels.py`
2. `python/sglang/srt/layers/attention/minicpm_backend.py`

实际改动：

1. MiniCPM FlashInfer 中的 `q_data_type` 从 `model_runner.kv_cache_dtype` 改为 `model_runner.dtype`。
2. MiniCPM prefill wrapper 不再硬编码 `fa2`，改为读取环境变量 `SGLANG_MINICPM_FLASHINFER_PREFILL_BACKEND`，默认值为 `auto`。
3. MiniCPM FlashInfer 的 `wrapper.forward(...)` 现在在 prefill 和 decode 两侧都传入 `k_scale=layer.k_scale_float` 与 `v_scale=layer.v_scale_float`。
4. MiniCPM backend 不再在 FP8 KV-cache 准备阶段把 query 侧张量 cast 到 KV-cache dtype。

## 7）验证命令
语法 / 导入验证：

```bash
python3 -m compileall \
  python/sglang/srt/layers/attention/minicpm_attention_kernels.py \
  python/sglang/srt/layers/attention/minicpm_backend.py
```

建议在 fcloud 上优先做的首次 smoke test：

```bash
export SGLANG_MINICPM_FLASHINFER_PREFILL_BACKEND=auto
python3 -m sglang.launch_server \
  --model-path <MODEL_PATH> \
  --attention-backend minicpm_flashinfer \
  --kv-cache-dtype fp8_e5m2 \
  --disable-cuda-graph
```

```bash
export SGLANG_MINICPM_FLASHINFER_PREFILL_BACKEND=fa3
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

## 8）结果汇总表
| 项目 | 基线 | 新方案 | 变化 | 备注 |
|---|---:|---:|---:|---|
| MiniCPM prefill backend 选择 | 硬编码 `fa2` | 环境变量可配置，默认 `auto` | n/a | 支持 `auto` / `fa3` 测试 |
| Query dtype 合同 | 绑定 KV dtype | 绑定模型 dtype | n/a | 避免 all-FP8 Q 路径 |
| KV scale 传递 | wrapper forward 缺失 | prefill/decode 都传入 | n/a | 与通用 backend 对齐 |
| 本地语法验证 | 修改前待验证 | 运行时待验证 | n/a | 需先跑 compile |
| fcloud FP8 启动 | 修改前失败 | 待用户实测 | pending | 先用 `--disable-cuda-graph` |

## 9）回滚说明
若本轮改动不稳定：

1. 回退 `python/sglang/srt/layers/attention/minicpm_attention_kernels.py`，恢复旧的 MiniCPM FlashInfer wrapper 行为。
2. 回退 `python/sglang/srt/layers/attention/minicpm_backend.py`，恢复旧的 query 侧 cast 行为。
3. 在环境中取消 `SGLANG_MINICPM_FLASHINFER_PREFILL_BACKEND`。

## 10）下一步建议
1. 先使用 `--disable-cuda-graph` 和 `SGLANG_MINICPM_FLASHINFER_PREFILL_BACKEND=auto` 做第一轮 smoke test。
2. 如果 `auto` 仍落到错误 backend，再用 `fa3` 重测，把 backend 选择问题和 dtype 合同问题拆开。
3. 如果运行时仍然为错误架构或不支持的 dtype 组合做 JIT 编译，下一轮应优先处理 FlashInfer 版本或编译配置，而不是继续改 MiniCPM 逻辑。
4. 一旦能成功启动，就应先用 Nsight Systems，再用 Nsight Compute，确认 decode 侧是否真的是主要带宽瓶颈，以及 FP8 KV cache 是否确实降低了该瓶颈。