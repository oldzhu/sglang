# CHANGE_0021_probe_prefill_32768_tuning

## 1) 背景与动机
- 3x 基线结果显示 S1 是加权时长的主要瓶颈。
- 首先采用低风险的启动参数优化，目标是提升长上下文 prefill 吞吐。

## 2) 规则合规说明
- 本次仅为运行时启动参数优化。
- 不涉及更换基座模型，不涉及违规开启 prefix cache，不绕过固定并发评测规则。

## 3) 变更前计划
- 仅调整 `probe` 档位：
  - `--chunked-prefill-size`: `8192 -> 32768`
  - `--max-prefill-tokens`: `16384 -> 32768`
- 其余参数保持不变，便于归因。

## 4) 实际代码改动
- 更新 `benchmark/soar/scripts/launch_perf_probe.sh` 中上述两个 prefill 参数。
- 同步更新 `benchmark/soar/scripts/launch_profile.sh` 中 probe 的 `resolved_cmd` 展示字符串，保证 status/dry-run 一致。

## 5) 验证命令
```bash
bash -n benchmark/soar/scripts/launch_perf_probe.sh
bash -n benchmark/soar/scripts/launch_profile.sh

bash benchmark/soar/scripts/launch_profile.sh probe
python3 benchmark/soar/run_soar_suite.py ...
```

## 6) 预期收益与风险
- 预期：长输入负载下 S1 与加权时长改善。
- 风险：高队列压力下显存压力上升，可能带来偶发不稳定。

## 7) 结果表模板
| 指标 | Baseline | New | Delta |
|---|---:|---:|---:|
| S1 时长 (s) | TBD | TBD | TBD |
| S8 时长 (s) | TBD | TBD | TBD |
| Smax 时长 (s) | TBD | TBD | TBD |
| 加权代理时长 (s) | TBD | TBD | TBD |
| Correctness overall_accuracy | TBD | TBD | TBD |

## 8) 回滚方案
1. 将 `benchmark/soar/scripts/launch_perf_probe.sh` 的 prefill 参数回退到原值。
2. 同步回退 `benchmark/soar/scripts/launch_profile.sh` 中 probe 展示命令。

## 9) 下一步建议
- 若本次改动稳定且加权时长下降，再在下一迭代执行一项新的低风险参数优化。
