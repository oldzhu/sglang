# CHANGE_0042：Dense 配置的调度与预热调优

## 背景与动机

当前周提交配置采用稳定的 `fp8 + gpu graph + force-dense-minicpm` 路径，但服务侧配置对吞吐较为保守，尤其是 `--prefill-max-requests 1`、`--max-running-requests 20` 和 `--skip-server-warmup`。

这种配置有利于稳定性，但在官方 `S8` 与 `Smax` 档位下可能没有把吞吐能力用满。本次迭代的目标是在不改变当前稳定 dense MiniCPM 路径的前提下，改善请求打包效率，并去掉首个请求的启动开销。

## 规则合规说明

本次修改符合 SOAR 2026 允许的优化范围：

- 仅调整推理运行时行为与提交启动行为。
- 不修改基座模型结构，也不替换官方模型。
- 不启用任何被禁止的 prefix cache 行为。
- 不改变官方固定并发评测逻辑。
- 保持现有稳定量化路径不变：权重量化仍为 `gptq_marlin`，KV Cache 仍为 `fp8_e5m2`。

## 修改前的详细实现计划

1. 保持 `--force-dense-minicpm`，继续禁用 MiniCPM sparse attention。
2. 保持 `--kv-cache-dtype fp8_e5m2` 与 `--quantization gptq_marlin` 不变。
3. 将 `--prefill-max-requests` 从 `1` 提升到 `2`，允许适度 prefill 合批。
4. 将 `--max-running-requests` 从 `20` 提升到 `28`，改善队列利用率与重叠能力。
5. 将 `--schedule-conservativeness` 从 `1.0` 调低到 `0.85`，让调度器在 token 准入上不那么保守。
6. 去掉 `--skip-server-warmup`，让服务在正式评测流量到来前执行一次预热请求。

## 实际代码修改

更新了 [benchmark/soar/demo_sala/prepare_env.sh](/home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh)：

- 将 `--prefill-max-requests 1` 改为 `--prefill-max-requests 2`
- 将 `--max-running-requests 20` 改为 `--max-running-requests 28`
- 将 `--schedule-conservativeness 1.0` 改为 `--schedule-conservativeness 0.85`
- 删除 `--skip-server-warmup`

本次迭代没有修改 MiniCPM dense attention、Lightning attention、sparse attention 或 Marlin kernel 的代码路径。

## 验证命令

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

Shell 语法验证：

```bash
bash -n benchmark/soar/demo_sala/prepare_env.sh
```

## 结果汇总表

| 指标 | 基线（`CHANGE_0041`） | 新方案（`CHANGE_0042`） |
| --- | --- | --- |
| `acc_ori` | 第二次官方分数 80.07 | 待验证 |
| `S1` | 第二次官方分数 458.78 | 待验证 |
| `S8` | 第二次官方分数 634.35 | 待验证 |
| `Smax` | 第二次官方分数 1140.66 | 待验证 |
| 稳定性 | 已稳定 | 待验证 |

## 回滚说明

如果本次调优导致吞吐或稳定性退化，可在 [benchmark/soar/demo_sala/prepare_env.sh](/home/oldzhu/sglang/benchmark/soar/demo_sala/prepare_env.sh) 中回滚以下四项：

- 恢复 `--prefill-max-requests 1`
- 恢复 `--max-running-requests 20`
- 恢复 `--schedule-conservativeness 1.0`
- 重新加回 `--skip-server-warmup`

## 下一步建议

1. 先在 fcloud 上验证正确性与速度，确认收益是否主要体现在 `S8` 与 `Smax`。
2. 如果本配置有提升但仍不够，再做一轮受控调度参数扫描，重点看 `prefill-max-requests 2/4` 与 `max-running-requests 28/36`。
3. 如果服务侧调优收益接近饱和，再进入 SM120 定向 Marlin kernel 调优，而不是立刻切到 sparse FP8 或 EAGLE3。