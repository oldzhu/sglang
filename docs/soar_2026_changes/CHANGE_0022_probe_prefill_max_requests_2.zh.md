# CHANGE_0022_probe_prefill_max_requests_2

## 1) 背景与动机
- 在 CHANGE_0021 后，加权时长有小幅改善且整体稳定。
- 下一步采用低风险方式提升 prefill 批处理利用率。

## 2) 规则合规说明
- 仅涉及运行时启动参数调优。
- 不涉及更换基座模型，不涉及违规策略。

## 3) 变更前计划
- 仅在 probe 档位调整：
  - `--prefill-max-requests`: `1 -> 2`
- 其余参数保持不变，确保单因子归因。

## 4) 实际代码改动
- 更新 `benchmark/soar/scripts/launch_perf_probe.sh`。
- 同步更新 `benchmark/soar/scripts/launch_profile.sh` 中 probe 的 `resolved_cmd`。

## 5) 验证命令
```bash
bash -n benchmark/soar/scripts/launch_perf_probe.sh
bash -n benchmark/soar/scripts/launch_profile.sh

bash benchmark/soar/scripts/launch_profile.sh probe
python3 benchmark/soar/run_soar_suite.py ...
```

## 6) 预期收益与风险
- 预期：通过 prefill 重叠提升 S8/Smax 的时长表现。
- 风险：长上下文高峰时显存压力略增，可能带来偶发不稳定。

## 7) 结果表模板
| 指标 | Baseline | New | Delta |
|---|---:|---:|---:|
| S1 时长 (s) | TBD | TBD | TBD |
| S8 时长 (s) | TBD | TBD | TBD |
| Smax 时长 (s) | TBD | TBD | TBD |
| 加权代理时长 (s) | TBD | TBD | TBD |
| Correctness overall_accuracy | TBD | TBD | TBD |

## 8) 回滚方案
1. 将 `benchmark/soar/scripts/launch_perf_probe.sh` 的 `--prefill-max-requests` 回退到 `1`。
2. 同步回退 `benchmark/soar/scripts/launch_profile.sh` 中 probe 的展示命令。

## 9) 下一步建议
- 若收益稳定则继续一次调度/内存参数优化；若不稳定则回滚并转入量化路线。
