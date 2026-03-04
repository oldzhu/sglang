# CHANGE_0015_run_with_gpu_watch_wrapper

## 1）背景与动机
- 已有独立 GPU 监控脚本，但每次 benchmark 前后手动启动/停止较繁琐。
- 目标：提供一个封装脚本，自动完成“启动监控 -> 执行命令 -> 停止监控”。

## 2）SOAR 规则合规性检查
- 仅工具链编排增强。
- 不涉及模型/推理算法改动。

## 3）改动前计划
- 在 `benchmark/soar/scripts` 新增一个脚本。
- 通过命令字符串参数保持通用性。

## 4）实际代码改动
- 新增 `benchmark/soar/scripts/run_with_gpu_watch.sh`。
- 脚本行为：
  - 后台启动 `watch_gpu_mem.sh`；
  - 执行目标命令；
  - 通过 EXIT/INT/TERM trap 自动停止 watcher；
  - 返回目标命令退出码。

## 5）使用方式
```bash
# 封装一次 S1 运行
bash benchmark/soar/scripts/run_with_gpu_watch.sh \
  "python3 benchmark/soar/run_soar_suite.py --api-base http://127.0.0.1:30000 --model-path /root/models/openbmb/MiniCPM-SALA --speed-data-s1 /root/soar_fast_data/speed_s1.jsonl" \
  /root/soar_logs 1 0
```

## 6）风险评估
- 风险很低，仅 shell 封装。

## 7）回滚说明
1. 删除 `benchmark/soar/scripts/run_with_gpu_watch.sh`。

## 8）下一步建议
- 后续可考虑增加“并行 tail server 日志”的可选模式。
