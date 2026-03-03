# CHANGE_0006_unified_profile_launcher

## 1）背景与动机
- 当前已有三套启动脚本（`eval-safe`、`perf-probe`、`toolkit-default`）。
- 频繁切换时手动记脚本名容易出错。
- 目标：提供统一入口，通过参数选择档位。

## 2）SOAR 规则合规性检查
- 仅脚本层运维便利性改动。
- 不涉及模型/内核/推理算法改动。
- 不影响官方评测约束。

## 3）改动前计划
- 新增一个脚本：
  - `benchmark/soar/scripts/launch_profile.sh`
- 根据档位名称分发到已有脚本。

## 4）实际代码改动
- 新增 `launch_profile.sh`，支持：
  - `safe` -> `launch_eval_safe.sh`
  - `probe` -> `launch_perf_probe.sh`
  - `default` -> `launch_toolkit_default.sh`
- 增加了用法提示与非法参数处理。

## 5）使用方式
```bash
bash benchmark/soar/scripts/launch_profile.sh safe
bash benchmark/soar/scripts/launch_profile.sh probe
bash benchmark/soar/scripts/launch_profile.sh default
```

## 6）风险评估
- 风险极低；仅复用已验证脚本。

## 7）回滚说明
1. 删除 `benchmark/soar/scripts/launch_profile.sh`。

## 8）下一步建议
- 后续可单独增加按档位加载环境变量文件（env file）功能。
