# CHANGE_0023_rollback_prefill_max_requests_2

## 1) 背景与动机
- CHANGE_0022 将 probe 的 `--prefill-max-requests` 从 1 调到 2。
- 新测试仅有极小速度收益，但正确性明显下降（`ori_accuracy` 降至 79.18），因此需要回滚。

## 2) 规则合规说明
- 本次是运行时启动参数回滚。
- 不涉及模型替换或任何违规策略。

## 3) 变更前计划
- 回退 probe 参数：
  - `--prefill-max-requests`: `2 -> 1`
- 其余参数不变。

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

## 6) 结果结论
- 决策：否决 CHANGE_0022 并执行回滚。
- 原因：正确性风险大于微小速度收益。

## 7) 回滚执行
- 已完成：probe prefill 请求上限恢复为 1。

## 8) 下一步建议
- 继续尝试其他单一低风险参数，或在运行时收益趋缓后转入量化路径规划。
