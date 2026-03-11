# CHANGE_0007_launch_profile_status_mode

## 1）背景与动机
- 已有统一启动脚本，但在实际启动前仍需要查看“最终会执行什么命令/参数”。
- 目标：增加 `status` / `--dry-run` 只读模式，不真正拉起服务。

## 2）SOAR 规则合规性检查
- 仅脚本运维增强。
- 不涉及推理算法或内核逻辑。

## 3）改动前计划
- 仅修改 `benchmark/soar/scripts/launch_profile.sh`。
- 新增 mode 参数，默认保持 `run`。

## 4）实际代码改动
- 新用法：
  - `bash benchmark/soar/scripts/launch_profile.sh <safe|probe|default> [run|status|--dry-run]`
- 新增 `status`/`--dry-run` 功能：
  - 打印档位
  - 打印目标脚本路径
  - 打印解析后的环境变量默认值（`MODEL_PATH`、`HOST`、`PORT`，以及适用时的 `PYTORCH_CUDA_ALLOC_CONF`）
  - 打印解析后的启动命令
  - 仅输出，不启动服务
- 默认 `run` 行为保持不变。

## 5）验证命令
```bash
bash benchmark/soar/scripts/launch_profile.sh safe status
bash benchmark/soar/scripts/launch_profile.sh probe --dry-run
bash benchmark/soar/scripts/launch_profile.sh default status
```

## 6）风险评估
- 风险很低，仅 shell 控制流改动。

## 7）回滚说明
1. 回退 `benchmark/soar/scripts/launch_profile.sh`。

## 8）下一步建议
- 可在后续独立变更中增加 status 模式下的端口占用检测输出。
