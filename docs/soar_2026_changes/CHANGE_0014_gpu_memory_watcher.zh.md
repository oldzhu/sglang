# CHANGE_0014_gpu_memory_watcher

## 1）背景与动机
- OOM 问题常具有偶发性，仅看最终 traceback 不足以定位峰值发生时刻。
- 需要在 eval/bench 过程中持续记录 GPU 显存与利用率时间序列。

## 2）SOAR 规则合规性检查
- 本次仅为诊断工具增强。
- 不涉及模型、内核和推理路径改动。

## 3）改动前计划
- 新增一个轻量脚本到 `benchmark/soar/scripts`。
- 仅依赖 `nvidia-smi`，按固定间隔输出 CSV。

## 4）实际代码改动
- 新增 `benchmark/soar/scripts/watch_gpu_mem.sh`。
- 脚本功能：
  - 记录时间戳、显存已用/总量、GPU/显存利用率、温度、功耗。
  - 支持输出目录、采样间隔、GPU ID 参数。

## 5）使用方式
```bash
# 默认（输出目录、2秒、GPU 0）
bash benchmark/soar/scripts/watch_gpu_mem.sh

# 自定义输出目录 + 1秒采样 + GPU 0
bash benchmark/soar/scripts/watch_gpu_mem.sh /root/soar_logs 1 0
```

## 6）风险评估
- 风险很低，仅监控读取。

## 7）回滚说明
1. 删除 `benchmark/soar/scripts/watch_gpu_mem.sh`。

## 8）下一步建议
- 后续可单独增加“随 bench 自动启动/停止 watcher”的封装脚本。
