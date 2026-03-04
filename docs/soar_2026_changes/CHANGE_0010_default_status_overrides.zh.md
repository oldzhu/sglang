# CHANGE_0010_default_status_overrides

## 1）背景与动机
- `CHANGE_0009` 已为 toolkit-default 启动脚本增加 `CHUNKED_PREFILL_SIZE` 与 `MEM_FRACTION_STATIC` 覆盖参数。
- 但 `launch_profile.sh status` 未显示这些实际值，排障不够直观。

## 2）SOAR 规则合规性检查
- 仅脚本状态输出增强。
- 不影响模型/推理行为。

## 3）改动前计划
- 仅修改 `benchmark/soar/scripts/launch_profile.sh`。
- 在 default 档位 status 输出中展示覆盖参数。

## 4）实际代码改动
- 在 `default` 档位 status 输出新增：
  - `CHUNKED_PREFILL_SIZE`（默认 `32768`）
  - `MEM_FRACTION_STATIC`（未设置显示 `<unset>`）
- 同时更新“解析后的命令”文本，体现 chunked-prefill 来自环境变量、mem-fraction 为可选。

## 5）验证命令
```bash
bash benchmark/soar/scripts/launch_profile.sh default status
CHUNKED_PREFILL_SIZE=8192 MEM_FRACTION_STATIC=0.80 bash benchmark/soar/scripts/launch_profile.sh default --dry-run
```

## 6）风险评估
- 风险极低（仅状态输出）。

## 7）回滚说明
1. 回退 `benchmark/soar/scripts/launch_profile.sh`。

## 8）下一步建议
- 下一次变更可考虑增强 `run_soar_suite.py` 的准确率解析兜底，避免 `ori_accuracy/overall_accuracy` 为空。
