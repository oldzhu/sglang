# CHANGE_0034_minicpm_profiling_methodology

## 1）背景与动机
- 问题描述：当前分支上针对 MiniCPM 高并发路径的分析，仍有较多“基于现象推测再试补丁”的成分；如果没有更重的代表性数据和系统级 profiling，就很难区分 CPU 调度开销、CUDA-graph replay 开销、sparse 元数据开销以及真正的 kernel 瓶颈。
- 为什么这会提升后续提速工作的质量：建立可重复的 profiling 流程后，可以更快把现象收敛到根因，减少低价值试错，并提高下一次代码改动命中主瓶颈的概率。
- 目标阶段：benchmark 方法学 / request metrics / CUDA 时间线 / CPU 热路径诊断

## 2）SOAR 规则合规性检查
- 允许原因：本次迭代只新增 profiling 方法学文档，不修改模型权重、不改变模型输出、不改变 benchmark 并发设置，也不修改提交接口。
- 最新规则/toolkit 复核（2026-03-11）：官方 competition 页面仍要求固定并发、禁止利用 prefix cache 获取速度优势，并强调优化方案需可复现、可稳定运行；toolkit 页面仍支持通过自定义 speed JSONL 做自测，并提供官方 correctness 脚本。
- 与 `技术路径指引` 的关系：profiling 不是单独的“刷分技巧”，而是为官方鼓励的运行时优化、量化和投机采样路线提供诊断依据。
- 对正确性系数 C 的预期影响：中性；本次文档新增不会改变任何运行时行为。

## 3）改动前实施计划
- 计划修改的文件/函数：仅新增 `docs/soar_2026_changes/` 下的两份文档；本次 feature 不修改任何运行时代码。
- 最小化 diff 策略：复用仓库中已有入口脚本与参数，不额外发明新的 benchmark/profiling 脚本。
- 测量方案：
  - 用 `benchmark/soar/generate_speed_datasets.py` 生成一个共享代表性数据集。
  - 用 `benchmark/soar/run_soar_suite.py` 让 S1/S8/Smax 三档都跑同一个共享文件。
  - 服务端启动参数复用 `benchmark/soar/scripts/launch_profile.sh` 中已有的 probe 风格配置。
  - 采集 request metrics、Nsight Systems trace、粗粒度 GPU 遥测，以及可选的 Linux `perf` 调用栈。
  - 根据 trace 判断下一步应该优先优化 host 调度、CUDA-graph replay、sparse 元数据准备，还是 kernel。
- 回滚方案：如果不再需要保留这套方法学记录，删除本次新增的中英文文档即可。

## 4）实际代码改动（获批后填写）
- 补丁摘要：仅新增一对 repo-specific 的 MiniCPM-SALA profiling 方法学文档，用于指导后续高并发瓶颈定位。
- 最终修改文件：`docs/soar_2026_changes/CHANGE_0034_minicpm_profiling_methodology.en.md`、`docs/soar_2026_changes/CHANGE_0034_minicpm_profiling_methodology.zh.md`。
- 核心变化：
  - 没有任何运行时代码改动。
  - 当前分支新增了一套标准化 profiling 顺序，可在未来性能 patch 前先执行。
  - 该方法学明确区分“烟雾测试”和“可用于决策的重负载 profiling 运行”。

## 5）验证命令
### 共享数据集生成
```bash
python3 benchmark/soar/generate_speed_datasets.py \
  --profile balanced \
  --prompt-source synthetic \
  --output-dir benchmark/soar/data/balanced_shared_v1 \
  --model-path /root/models/openbmb/MiniCPM-SALA
```

### 用于 Profiling 的服务启动
```bash
python3 -m sglang.launch_server \
  --model-path /root/models/openbmb/MiniCPM-SALA \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code \
  --disable-radix-cache \
  --attention-backend minicpm_flashinfer \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 32768 \
  --prefill-max-requests 1 \
  --max-running-requests 20 \
  --mem-fraction-static 0.84 \
  --schedule-conservativeness 1.0 \
  --skip-server-warmup \
  --enable-metrics \
  --export-metrics-to-file \
  --export-metrics-to-file-dir benchmark/soar/results/request_metrics
```

### 正确性
```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path /root/models/openbmb/MiniCPM-SALA \
  --data_path /path/to/perf_public_set.jsonl \
  --concurrency 32
```

### 速度
```bash
SHARED=benchmark/soar/data/balanced_shared_v1/speed_smax.jsonl

python3 benchmark/soar/run_soar_suite.py \
  --api-base http://127.0.0.1:30000 \
  --model-path /root/models/openbmb/MiniCPM-SALA \
  --speed-data-s1 "$SHARED" \
  --speed-data-s8 "$SHARED" \
  --speed-data-smax "$SHARED" \
  --disable-tqdm
```

### 粗粒度 GPU 遥测
```bash
nvidia-smi dmon -s pucvmet -d 1 -o TD -f benchmark/soar/results/nvidia_dmon.log
```

### Nsight Systems
```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=true \
  --cuda-graph-trace=node \
  --force-overwrite=true \
  --delay 20 \
  --duration 90 \
  -o benchmark/soar/results/nsys_minicpm_probe \
  python3 -m sglang.launch_server \
    --model-path /root/models/openbmb/MiniCPM-SALA \
    --host 0.0.0.0 \
    --port 30000 \
    --trust-remote-code \
    --disable-radix-cache \
    --attention-backend minicpm_flashinfer \
    --chunked-prefill-size 32768 \
    --max-prefill-tokens 32768 \
    --prefill-max-requests 1 \
    --max-running-requests 20 \
    --mem-fraction-static 0.84 \
    --schedule-conservativeness 1.0 \
    --skip-server-warmup
```

### 可选 CPU Profiling
```bash
perf record -F 99 -g -p <SERVER_PID> -- sleep 60
perf report
```

## 6）结果汇总
| 项目 | 基线 | Profiling 发现 | 下一步动作 |
|---|---|---|---|
| Correctness / overall_accuracy | pending | pending | 保持在 > 97% 系数阈值之上 |
| S1 benchmark_duration (s) | pending | pending | pending |
| S8 benchmark_duration (s) | pending | pending | pending |
| S∞ benchmark_duration (s) | pending | pending | pending |
| GPU 时间线模式 | pending | pending | 判断是 host-bound 还是 kernel-bound |
| CPU 热栈 | pending | pending | 决定是否继续做 `perf` 深挖 |

## 7）风险评估
- 准确率风险：无；这是纯文档 feature。
- 稳定性风险：低；方法学只复用仓库已有脚本与外部 profiler。
- 解释风险：中等；如果 trace 采集时仍使用过轻数据集，或者把启动噪声也算进去，就可能得出误导性结论。
- 可复现风险：低；只要共享数据集、启动参数和 trace 窗口保持一致，就能稳定复现。

## 8）回滚说明
1. 删除 `docs/soar_2026_changes/CHANGE_0034_minicpm_profiling_methodology.en.md`。
2. 删除 `docs/soar_2026_changes/CHANGE_0034_minicpm_profiling_methodology.zh.md`。

## 9）下一步建议
- 在当前分支真正继续修改 MiniCPM 运行时代码之前，先采一份基线 Nsight Systems trace。
- 如果 GPU 时间线上存在大段 idle gap，优先排查 host 调度或 replay setup，而不是继续改 kernel。
- 如果 GPU 占用已经很密集且只有一两个 kernel 主导耗时，再进入 kernel 级分析并决定是否做新代码改动。