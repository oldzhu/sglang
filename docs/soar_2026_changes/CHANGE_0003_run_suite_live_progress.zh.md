# CHANGE_0003_run_suite_live_progress

## 1）背景与动机
- 问题描述：通过 `.sh` 启动 `run_soar_suite.py` 时，在任务结束前几乎看不到进度输出。
- 根因：
  1）脚本使用 `subprocess.run(..., stdout=PIPE)`，输出被缓冲后一次性返回。
  2）bench 子命令固定加了 `--disable-tqdm`。
- 目标：终端实时显示进度，同时保留日志文件。

## 2）SOAR 规则合规性检查
- 该改动仅提升工具可观测性，不改模型算法、内核或推理路径。
- 不影响官方规则中的限制项。

## 3）改动前计划
- 仅修改一个脚本：`benchmark/soar/run_soar_suite.py`。
- 将“缓冲采集”改为“实时流式采集”。
- 将 bench 的 tqdm 关闭改为可选参数（默认显示进度）。

## 4）实际代码改动（获批后）
- `run_and_capture(...)` 改为 `subprocess.Popen`，按行实时输出到终端并同步写日志。
- 增加阶段标签输出（如 `correctness`、`speed/s1`）。
- 新增 `run_soar_suite.py` 参数：`--disable-tqdm`。
- 删除 bench 调用中的固定 `--disable-tqdm`。

## 5）预期行为
- 通过 `.sh` 运行时可实时看到各阶段输出。
- 日志仍保存在 run 目录。
- 如需安静模式，可显式传 `--disable-tqdm`。

## 6）验证命令
```bash
python3 benchmark/soar/run_soar_suite.py \
  --api-base http://127.0.0.1:30000 \
  --model-path /root/models/openbmb/MiniCPM-SALA \
  --eval-script /root/data/eval_model.py \
  --public-data /root/data/perf_public_set.jsonl \
  --speed-data-s1 /root/data/speed_s1.jsonl
```

可选静默 bench 进度：
```bash
python3 benchmark/soar/run_soar_suite.py \
  --api-base http://127.0.0.1:30000 \
  --model-path /root/models/openbmb/MiniCPM-SALA \
  --speed-data-s1 /root/data/speed_s1.jsonl \
  --disable-tqdm
```

## 7）风险评估
- 风险很低。仅涉及子进程输出处理和 bench 参数透传。

## 8）回滚说明
1. 回退 `benchmark/soar/run_soar_suite.py`。
2. 重新执行小规模 correctness + S1 验证行为恢复。

## 9）下一步建议
- 如有需要，可在后续单独变更中添加“分阶段超时/重试”能力。
