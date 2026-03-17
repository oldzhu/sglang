# CHANGE_0043：SM120 Marlin Kernel 调优与提交 Wheel 安装

## 背景与动机

当前稳定的 MiniCPM-SALA 提交路径仍然是 `gptq_marlin + fp8_e5m2 + force-dense-minicpm`。官方 toolkit 在路径一中明确建议，可以针对 6000D / SM120 GPU 做 Marlin 的 tile / warp 调优。

现有 Marlin launcher 仍在使用较小的通用执行配置表，并没有针对 SM120 做分支，这说明在最热的量化线性层路径上仍然存在一个较低风险的 kernel 优化机会。

上一轮服务侧实验（`CHANGE_0042`）将正确性降到了 `acc_ori 79.40`，因此本次迭代一方面恢复 `CHANGE_0041` 的已知稳定服务基线，另一方面把优化重点转到 Marlin kernel 路径。

## 规则合规说明

本次修改符合最新 SOAR 2026 官方规则与技术路径说明：

- 遵循 toolkit 路径一：GPTQ W4A16 + Marlin + FP8 KV cache。
- 针对官方 RTX PRO / SM120 环境进行 kernel 调度优化。
- 不替换官方基座模型。
- 不启用任何被禁止的 prefix cache 行为。
- 不改变官方固定并发评测逻辑。
- 通过 `prepare_env.sh` 安装本地 wheel 工件，这与当前提交机制完全兼容。

## 修改前的详细实现计划

1. 保持 `CHANGE_0041` 的稳定服务配置，作为精度安全基线。
2. 修改 repo root 下的 `sgl-kernel`，而不是 submission mirror 中的源码。
3. 为 Marlin auto-config 增加 SM120 专用候选配置表。
4. 不改动现有 kernel 数学逻辑和 reduction 精度行为。
5. 在 repo root 中构建 patched `sgl-kernel` wheel。
6. 手动将该 wheel 拷贝到 `benchmark/soar/demo_sala/`。
7. 修改 `prepare_env.sh`，让 submission 在启动时用 `--force-reinstall --no-deps` 安装唯一的本地 `sgl-kernel` wheel。
8. 在 Marlin launcher 中打印一次日志，便于确认测试时确实进入了 SM120 auto-config 路径。

## 实际代码修改

### 1. SM120 专用 Marlin auto-config

修改 [sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu](/home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu)：

- 增加 SM120 专用 small-batch 和 large-batch 候选配置表
- 在 Marlin launcher 中检测 SM120 及更高架构 GPU
- 在 SM120 上使用专用候选表参与 auto-config
- 第一次命中时向 stderr 打印一条一次性日志，输出选中的配置

新的 SM120 候选顺序优先尝试更宽的 `thread_n` 与 `256` 线程配置，再回退到旧的通用配置。

### 2. 让 patched wheel 易于识别

修改 [sgl-kernel/pyproject.toml](/home/oldzhu/sglang/sgl-kernel/pyproject.toml)，将版本从 `0.3.20` 提升为 `0.3.20.post1`，这样生成的 patched wheel 更容易识别、拷贝和安装。

### 3. 提交时安装本地 patched kernel wheel

修改 [benchmark/soar/demo_sala/prepare_env.sh](/home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh)：

- 要求 submission 目录中恰好存在一个本地 `sgl-kernel` wheel
- 使用 `uv pip install --force-reinstall --no-deps` 安装该 wheel
- 同时恢复到 `CHANGE_0041` 的已知稳定服务配置

这样做避免了把整个 `sgl-kernel` 源码树打包进提交目录，同时又能保证官方运行时真正使用的是 patched kernel。

## 如何验证测试时确实使用了 SM120 配置

当 patched wheel 安装成功、服务启动后，查看 server log 中是否出现类似下面的一次性日志：

```text
[sgl-kernel] SM120 Marlin auto-config enabled: M=... N=... K=... thread_m_blocks=... thread_n=... thread_k=... num_threads=...
```

如果在 correctness 或 `S1/S8/Smax` 测试时出现这条日志，就说明 patched SM120 配置路径已经被进入。

同时也可以查看 `prepare_env.sh` 的日志，确认其实际安装的是你拷贝进去的本地 wheel。

## Wheel 构建与拷贝命令

在 repo root 下构建 patched wheel：

```bash
cd sgl-kernel
./build.sh 3.10 12.8 x86_64
```

将构建出的 wheel 拷贝到 submission 目录：

```bash
cp dist/*/sgl_kernel-0.3.20.post1-*.whl ../benchmark/soar/demo_sala/
```

如果你的构建输出目录结构不同，也可以手动把生成的 `sgl_kernel-0.3.20.post1-*.whl` 拷贝到 `benchmark/soar/demo_sala/`。

## 验证命令

Shell 语法验证：

```bash
bash -n benchmark/soar/demo_sala/prepare_env.sh
```

正确性验证：

```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_DIR> \
  --data_path <DATA_DIR>/perf_public_set.jsonl \
  --concurrency 32
```

速度验证：

```bash
export SPEED_DATA_S1=<PATH_TO_S1_JSONL>
export SPEED_DATA_S8=<PATH_TO_S8_JSONL>
export SPEED_DATA_SMAX=<PATH_TO_SMAX_JSONL>

bash SOAR/bench_serving.sh http://127.0.0.1:30000
```

## 结果汇总表

| 指标 | 基线（`CHANGE_0041`） | 新方案（`CHANGE_0043`） |
| --- | --- | --- |
| `acc_ori` | 第二次官方分数 80.07 | 待验证 |
| `S1` | 第二次官方分数 458.78 | 待验证 |
| `S8` | 第二次官方分数 634.35 | 待验证 |
| `Smax` | 第二次官方分数 1140.66 | 待验证 |
| 本地 wheel 安装 | 否 | 是 |
| SM120 Marlin 日志 | 否 | 待验证 |

## 回滚说明

1. 回退 [sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu](/home/oldzhu/sglang/sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu) 到原始通用配置表。
2. 回退 [sgl-kernel/pyproject.toml](/home/oldzhu/sglang/sgl-kernel/pyproject.toml) 的版本号修改。
3. 从 [benchmark/soar/demo_sala/prepare_env.sh](/home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh) 中删除本地 `sgl-kernel` wheel 安装逻辑。
4. 恢复使用官方 pip 提供的 `sgl-kernel`。

## 下一步建议

1. 先构建 wheel 并做 correctness，确认 kernel-only 变更不影响精度。
2. `S1/S8/Smax` 对比请以 `CHANGE_0041` 为基线，而不是已经失效的 `CHANGE_0042`。
3. 如果日志确认命中了 SM120 auto-config，但收益仍然有限，再继续做一轮 SM120 配置表顺序调整，然后再考虑 sparse FP8 恢复。