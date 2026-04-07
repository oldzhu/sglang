# CHANGE_0071: 修复 MiniCPM 稀疏 Triton int64 分支类型不匹配

## 1）背景与动机

- **问题**：Test 2（`MiniCPM-SALA-Copy` + 稀疏注意力 + bf16 KV cache）在 CUDA graph capture 阶段失败，错误来自 `compress_k_complete_kernel_new` 的 Triton 编译。
- **观测到的报错**：`AssertionError('Mismatched type for token_k_indices between then block (int64) and else block (int32)')`。
- **影响**：非 FP8 的诊断路径在正确性测试前就被阻塞，无法继续验证 sparse 本身的问题。
- **附带的运维问题**：手工 Test 1 表明，如果启动 server 前没有执行 `source /root/submission_sim/prepare_env.sh`，容易触发 CUDA OOM，导致得到误导性的低准确率结果。

## 2）SOAR 规则合规性检查

- 已重新核对官方页面：`https://soar.openbmb.cn/competition` 与 `https://soar.openbmb.cn/toolkit`。
- 本次属于正确性修复和测试流程补全，不修改模型权重、不改评测并发、不使用受限技巧。
- 修改内容仅包括 Triton 分支类型一致化，以及补充 fcloud 测试必备说明。
- 风险：低。该 kernel 本来就希望在此路径使用 64 位索引，本次只是移除编译期类型不一致。

## 3）修改前的详细实施计划

### 修复 A：统一 Triton 分支 dtype
1. 在 `compress_k_complete_kernel_new` 和 `compress_k_complete_kernel_new_padded` 中，替换越界分支里的 `token_k_indices = 0`。
2. 改为 `tl.zeros([], dtype=tl.int64)`，使其与 `tl.load(...).to(tl.int64)` 分支保持同一类型。

### 修复 B：补充 fcloud 测试必备说明
1. 在 `.github/copilot-instructions.md` 中明确记录：新 terminal 启动 server 前必须先执行 `/root/submission_sim/prepare_env.sh`。
2. 记录 `eval_model_001.py` 的评测数据路径应为 `/root/data/perf_public_set.jsonl`。

## 4）实际代码修改

### 文件：`python/sglang/srt/layers/attention/minicpm_sparse_kernels.py`

- 将四处 `token_k_indices = 0` 的 fallback 改为：

```python
token_k_indices = tl.zeros([], dtype=tl.int64)
```

- 应用于以下两个 kernel：
  - `compress_k_complete_kernel_new`
  - `compress_k_complete_kernel_new_padded`

### 文件：`.github/copilot-instructions.md`

- 新增 fcloud 测试说明：
  - 在新 terminal 启动 `sglang` 前，必须先运行 `source /root/submission_sim/prepare_env.sh`
  - `eval_model_001.py` 需要使用 `--data_path /root/data/perf_public_set.jsonl`

## 5）修复正确性说明

- Triton 要求同一变量在控制流的各个分支上保持相同 dtype。
- 正常分支已经把 `token_k_indices` 转成了 `tl.int64`，因为后续地址计算可能超出 32 位范围。
- 越界分支原来写成普通 `0`，导致该变量在该分支上变成 `int32`，从而在编译阶段直接失败。
- 改成 `tl.zeros([], dtype=tl.int64)` 后，语义不变，但满足 Triton 的类型要求。

## 6）验证命令

```bash
# 同步最新提交到 fcloud
python3 scripts/fcloud/fcloud_workflow.py sync

# 启动 Test 2 server
source /root/submission_sim/prepare_env.sh
python3 -m sglang.launch_server \
  --model-path /root/models/openbmb/MiniCPM-SALA-Copy \
  --trust-remote-code --disable-radix-cache \
  --attention-backend minicpm_flashinfer \
  --dense-as-sparse \
  --chunked-prefill-size 32768 --max-prefill-tokens 32768 \
  --prefill-max-requests 1 --max-running-requests 20 \
  --mem-fraction-static 0.84 --schedule-conservativeness 1.0 \
  --port 30000

# 准确率评测
cd /root/data
python3 eval_model_001.py \
  --api_base http://127.0.0.1:30000 \
  --model_path openbmb/MiniCPM-SALA \
  --data_path /root/data/perf_public_set.jsonl \
  --concurrency 8
```

## 7）结果汇总表

| 项目 | 修改前 | 修改后 | 变化 | 备注 |
|---|---|---|---|---|
| Test 2 启动 | Triton 编译失败 | 可以继续启动测试 | 修复 | 仍需用户在 fcloud 验证 |
| `token_k_indices` fallback dtype | int32 | int64 | 修复 | 与正常分支保持一致 |
| fcloud 启动流程 | 依赖记忆 | 已文档化 | 安全性提升 | 避免遗漏 env 导致 CUDA OOM |
| eval 数据路径 | 容易写错 | 已文档化 | 安全性提升 | 使用 `/root/data/perf_public_set.jsonl` |

## 8）回滚说明

```bash
git revert <commit>
```

或者手动将 `minicpm_sparse_kernels.py` 中四处 fallback 赋值改回 `0`，并删除 `.github/copilot-instructions.md` 中新增的 fcloud 说明。

## 9）下一步建议

1. 重新运行 Test 2，确认 Triton 编译错误已消失。
2. 如果 Test 2 仍然低准确率，则继续分析 sparse prefill/decode 逻辑，而不是量化问题。
3. 如果 Test 2 跑通，再比较 bf16 KV 与 FP8 KV 的准确率差异，定位剩余误差来源。