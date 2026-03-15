# CHANGE_0041_minicpm_fp8_force_dense_submission_profile

## 1) 背景与动机
- 问题描述：MiniCPM 稀疏解码路径在 `fp8 + gpu graph` 下仍然不稳定，持续在 CUDA graph replay 阶段崩溃；而 `fp8 + gpu graph + --force-dense-minicpm` 已经可以通过正确性评测。
- 重要性：这为本周提交提供了一个可直接使用的稳定运行配置，同时保留 FP8 KV cache 与 CUDA graph 的主要加速收益。
- 本次迭代目标：将提交脚本中的启动配置切换为 `--kv-cache-dtype fp8_e5m2 --force-dense-minicpm`，让本周提交使用当前已验证且速度更优的稳定方案。

## 2) SOAR 规则合规检查
- 本次修改前已重新检查最新官方页面：`https://soar.openbmb.cn/competition` 与 `https://soar.openbmb.cn/toolkit`。
- 合规原因：这属于官方 `prepare_env.sh` 提交流程中的运行时启动参数调整，并且符合官方建议的 KV-cache 优化方向。
- 已遵守约束：不替换基座模型、不重新开启 prefix cache、不修改并发评测设置、不修改提交接口。
- 精度/稳定性风险：本次迭代风险较低，因为所选配置已经在用户最新的 fcloud 正确性测试中通过。

## 3) 修改前的详细实施计划
本次迭代范围：

1. 在 `prepare_env.sh` 中更新提交启动参数，启用 FP8 KV cache。
2. 强制 MiniCPM 走 dense 模式，绕开当前不稳定的稀疏解码 graph 路径。
3. 不再在提交脚本中额外追加 `--log-level info`。

需要修改的文件/函数：

1. `benchmark/soar/demo_sala/prepare_env.sh`

预期收益：

1. 为本周提交保留 `fp8 + gpu graph` 的核心收益。
2. 通过关闭 MiniCPM 稀疏注意力，规避当前 replay 崩溃。
3. 保持提交脚本符合官方 `prepare_env.sh` 环境变量注入方式。

## 4) 实际代码修改
修改文件：

1. `benchmark/soar/demo_sala/prepare_env.sh`

实际修改内容：

1. 在 GPTQ 服务配置导出的 `SGLANG_SERVER_ARGS` 中加入 `--kv-cache-dtype fp8_e5m2`。
2. 在同一启动配置中加入 `--force-dense-minicpm`。
3. 注释掉额外追加 `--log-level info` 的语句。

保持不变的部分：

1. 本周提交配置仍然保留 GPTQ Marlin。
2. CUDA graph 仍然开启。
3. 现有服务限制与显存相关参数保持不变。

## 5) 验证命令
脚本语法验证：

```bash
bash -n benchmark/soar/demo_sala/prepare_env.sh
```

推荐在 fcloud 上进行正确性验证：

```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_PATH> \
  --data_path benchmark/soar/demo_sala/perf_public_set.jsonl \
  --concurrency 32
```

推荐在 fcloud 上进行速度验证：

```bash
python3 benchmark/soar/run_soar_suite.py \
  --base-url http://127.0.0.1:30000 \
  --dataset-profile heavy
```

## 6) 结果总结表
| 项目 | 之前的稳定候选 | 新提交配置 | 变化 | 说明 |
|---|---:|---:|---:|---|
| KV-cache 类型 | 非 FP8 或 FP8 稀疏路径不稳定 | FP8 E5M2 | 改善 | 保留 FP8 decode 优化 |
| MiniCPM 稀疏路径 | 开启 | 通过 `--force-dense-minicpm` 关闭 | 取舍 | 避开当前 replay 崩溃 |
| 正确性评测 | FP8 稀疏 graph 下崩溃 | `acc_ori: 80.84%` | 改善 | 用户 fcloud 实测 |
| 速度代理 S1 | `142.00` | `136.32` | 更好 | 用户实测 |
| 速度代理 S8 | `31.21` | `28.65` | 更好 | 用户实测 |
| 速度代理 Smax | `25.68` | `23.61` | 更好 | 用户实测 |

## 7) 回滚说明
如果本次迭代收益不足或引入回归：

1. 从 `benchmark/soar/demo_sala/prepare_env.sh` 中移除 `--kv-cache-dtype fp8_e5m2` 与 `--force-dense-minicpm`。
2. 如有需要，恢复显式追加 `--log-level info` 的语句以方便调试。
3. 使用之前的启动配置重新跑正确性与速度测试。

## 8) 后续建议
1. 如果最终 fcloud 复测继续稳定，可用该配置提交本周成绩。
2. 继续定位 `fp8 + gpu graph + sparse` 的 replay 崩溃根因，后续争取重新启用稀疏 MiniCPM。
3. 在修复稀疏 replay 问题后，再比较 hidden speed set 上 force-dense 与 sparse 的最终优劣。