# CHANGE_0005_launch_profiles

## 1）背景与动机
- 需要可复现的启动档位，分别用于：
  - Eval-safe（正确性测试优先稳定，降低 OOM 风险）
  - Perf-probe（探索更高吞吐）
  - Toolkit-default（对齐 SOAR toolkit 文档默认示例）

## 2）SOAR 规则合规性检查
- 本次仅新增启动脚本。
- 不改模型/内核推理逻辑。
- 不绕过官方评测限制。

## 3）改动前计划
- 在 `benchmark/soar/scripts` 下新增 3 个脚本。
- 保持脚本逻辑直观、参数显式。

## 4）实际代码改动
- 新增文件：
  - `benchmark/soar/scripts/launch_eval_safe.sh`
  - `benchmark/soar/scripts/launch_perf_probe.sh`
  - `benchmark/soar/scripts/launch_toolkit_default.sh`

## 5）关于 Toolkit 默认参数说明
- 根据 SOAR toolkit 文档示例，默认参数风格包含：
  - `--disable-radix-cache --attention-backend flashinfer --chunked-prefill-size 32768`
- 线上最终以官方评测流程为准，但该脚本用于对齐文档默认示例。

## 6）验证命令
```bash
bash benchmark/soar/scripts/launch_eval_safe.sh
bash benchmark/soar/scripts/launch_perf_probe.sh
bash benchmark/soar/scripts/launch_toolkit_default.sh
```

## 7）风险评估
- 风险很低。仅为运维启动便利性改动。

## 8）回滚说明
1. 删除上述 3 个脚本。

## 9）下一步建议
- 可在后续单独变更中增加统一入口脚本（`safe|probe|default`）。
