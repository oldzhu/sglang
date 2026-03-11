# CHANGE_0032 本地速度数据代表性说明

## 背景与动机

在 `mixed_minicpm_cudagraph` 分支的提交分析中，本地速度测试呈现：

- `Smax < S8 < S1`

而官方排行榜结果呈现相反模式：

- `Smax > S8 > S1`

这种差异说明，本地速度测试数据的分布会显著影响并发趋势判断。较轻、较均匀、样本较少的本地数据集，可能让更高并发更容易体现重叠与吞吐优势；而官方隐藏速度集包含更重的长输入与长输出尾部分布，更容易放大排队、显存压力、慢请求拖尾和调度竞争。

本次迭代仅记录这一 benchmarking 解释和评估建议，为后续 CUDA graph 优化提供更可靠的本地评估依据。

## 规则合规说明

本次迭代仅涉及文档更新。

- 不修改模型行为。
- 不修改官方评测逻辑。
- 不启用 prefix cache 等违规特性。
- 不改变官方评测的并发配置。

因此本次迭代符合最新 SOAR 比赛与 toolkit 中关于可复现、可解释优化的要求。

## 详细实施计划

在继续 runtime 优化之前，先明确如下评估原则：

1. 本地出现 `Smax < S8 < S1`，并不代表官方结果一定异常。
2. 官方隐藏速度集很可能比当前本地轻量数据集更重。
3. 做 runtime 优化对比时，应尽量在 `S1`、`S8`、`Smax` 三档使用同一份具有代表性的数据集，仅改变 `--max-concurrency`。
4. 仍可保留一份较小的快速调试数据集，但不应用它单独决定 leaderboard 方向。
5. CUDA graph 优化可以现在就开始，但效果判断应更多依赖更重、更接近官方分布的本地数据集。

## 实际代码改动

本次迭代没有修改任何 runtime 或模型代码。

仅新增文档：

- `docs/soar_2026_changes/CHANGE_0032_local_speed_dataset_representativeness.en.md`
- `docs/soar_2026_changes/CHANGE_0032_local_speed_dataset_representativeness.zh.md`

## 验证命令

本次仅需文档校验：

```bash
# 示例：在工作区中检查两个 markdown 文件没有报错
```

建议的后续本地评估命令：

```bash
python3 benchmark/soar/generate_speed_datasets.py \
  --profile heavy \
  --output-dir benchmark/soar/data_heavy_shared

python3 benchmark/soar/run_soar_suite.py \
  --api-base http://127.0.0.1:30000 \
  --model-path <MODEL_PATH> \
  --speed-data-s1 benchmark/soar/data_heavy_shared/speed_s1.jsonl \
  --speed-data-s8 benchmark/soar/data_heavy_shared/speed_s1.jsonl \
  --speed-data-smax benchmark/soar/data_heavy_shared/speed_s1.jsonl
```

上述命令刻意让三档使用同一份数据，从而保证 `S1`、`S8`、`Smax` 的差异主要来自并发，而不是样本分布差异。

## 结果总结表

| 项目 | 之前的理解 | 更新后的建议 |
| --- | --- | --- |
| 对本地 `Smax < S8 < S1` 的解释 | 可能有问题 | 在轻量本地数据上是可能出现的 |
| 对官方 `Smax > S8 > S1` 的解释 | 可能不一致 | 在更重的隐藏 workload 下完全可能 |
| 三档速度对比方法 | 实际上可能混入不同数据集差异 | 做 runtime 对比时优先使用同一份代表性数据 |
| CUDA graph 工作起点 | 似乎需要先完全复现官方模式 | 可以立即开始，但需用更重数据集评估 |

## 回滚说明

如果后续发现本说明不准确或不完整：

1. 删除本次 `CHANGE_0032` 的 EN/ZH 文档。
2. 以新的 continuation 或修正文档对替代。
3. 不要仅依据轻量本地 benchmark 的结论做 leaderboard 方向判断。

## 后续建议

1. 先准备一份更重、并更接近官方公开 token 长度分布的共享本地速度数据集。
2. 在 `mixed_minicpm_cudagraph` 上用这份共享数据集评估 CUDA graph 改动在 `S1`、`S8`、`Smax` 三档的影响。
3. 现有快速本地数据集继续保留，但只用于快速回归检查。