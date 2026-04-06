# CHANGE_0070: 修复稀疏注意力 CUDA Graph kv_indptr 计划不匹配问题

## 1）背景与动机

- **问题描述**：稀疏注意力在 CUDA graph replay 阶段崩溃，原因是 FlashInfer 的 workspace plan（通过 `begin_forward()` 在图外计算）使用了静态 `kv_indptr = [0, K, 2K, ...]`，与实际的稀疏累积序列长度不匹配。当捕获的图执行 `convert_sparse_page_table_to_flashinfer()` 时，会用正确的值覆写 kv_indptr，但计划已核错 → workspace 越界 → 崩溃。
- **现有的规避措施**：`--force-dense-minicpm` 标志完全绕过稀疏注意力，将 `minicpm_flashinfer` 切换为标准 `flashinfer` 后端。这禁用了模型原生的稀疏路由，强制每层做全上下文密集注意力。
- **影响**：密集注意力在长上下文输入（32K-160K tokens = SOAR 速度评测分布的 50%）上显著更慢。启用稀疏注意力预计可获得 20-40% 的速度提升。
- **根因**：稀疏注意力 CUDA graph 路径中的 Bug 4 —— replay 函数 `init_forward_metadata_replay_cuda_graph()` 向 `begin_forward()` 传递了静态 kv_indptr，而非从当前批次的 `sparse_cache_seqlens` 计算正确值。

## 2）SOAR 规则合规性检查

- 已重新核对官方最新页面：`https://soar.openbmb.cn/competition` 与 `https://soar.openbmb.cn/toolkit`。
- 本次是正确性修复，非新优化技术。稀疏注意力是模型原生架构。
- 未触碰限制：不改模型权重、不重开 prefix cache、不改并发逻辑。
- 风险：低。恢复模型预期执行路径。密集注意力仍可通过 `--force-dense-minicpm` 回滚。

## 3）修改前的详细实施计划

### 修复 A：修正 replay 中的 kv_indptr（`minicpm_backend.py`）
1. 在 `init_forward_metadata_replay_cuda_graph()` 中，用 `forward_batch.sparse_cu_seqlens_k_cpu`（已有的 `pad(cumsum(sparse_cache_seqlens), (1,0))`）替换静态 kv_indptr。
2. 对尾部条目（`real_bs < bs` 时的填充部分）用最后一个有效累积和值填充，使 FlashInfer 看到零长度序列。
3. 将过期的 `sparse_cache_seqlens_int32` 尾部条目清零，确保图内 `convert_sparse_page_table_to_flashinfer()` 计算出匹配的 kv_indptr。

### 修复 B：启用稀疏注意力（`prepare_env.sh`）
1. 从 `SGLANG_SERVER_ARGS` 中移除 `--force-dense-minicpm`。

## 4）实际代码修改

### 文件：`python/sglang/srt/layers/attention/minicpm_backend.py`

**改动 1** —— 清零过期的 sparse_cache_seqlens 尾部：
```python
metadata.sparse_cache_seqlens_int32[: 2 * real_bs].copy_(
    forward_batch.sparse_cache_seqlens_int32_cpu
)
# 清零过期尾部条目，确保图内 convert_sparse_page_table_to_flashinfer
# 计算的 kv_indptr 与 replay 传给 begin_forward 的一致。
metadata.sparse_cache_seqlens_int32[2 * real_bs :].fill_(0)
```

**改动 2** —— 在 replay 中写入正确的 kv_indptr（替换静态模式）：
```python
# Bug 4 修复：从实际稀疏缓存序列长度写入正确的 kv_indptr
# 而非使用静态 [0, K, 2K, ...] 模式。
kv_indptr_view[: sparse_real_bs + 1].copy_(
    forward_batch.sparse_cu_seqlens_k_cpu[: sparse_real_bs + 1]
)
# 填充剩余条目使 FlashInfer 看到零长度序列
if sparse_real_bs < sparse_bs:
    kv_indptr_view[sparse_real_bs + 1 :].fill_(
        kv_indptr_view[sparse_real_bs].item()
    )
```

### 文件：`benchmark/soar/demo_sala/prepare_env.sh`

从 `SGLANG_SERVER_ARGS` 中移除了 `--force-dense-minicpm`。

## 5）修复正确性论证

FlashInfer CUDA graph 稀疏注意力的数据流：

1. **Replay**（图外）：从 `sparse_cu_seqlens_k_cpu` 写入正确的 `kv_indptr` → `begin_forward()` 计算匹配实际批次的 workspace plan
2. **图执行**：`convert_sparse_page_table_to_flashinfer()` 用相同的值覆写同一缓冲区（来源相同的 `sparse_cache_seqlens_int32`）—— 冗余但无害
3. **图执行**：`wrapper.forward()` 使用正确的 plan 并读取正确的缓冲区 → 无越界

静态模式 `[0, K, 2K, ...]` 假设每个批次条目始终恰好有 `num_sparse_topk_tokens` 个条目。实际上 `sparse_cache_seqlens` 随序列长度和 block 对齐而变化：
```python
sparse_cache_seqlens = where(
    seq_lens <= K, seq_lens,
    (topk-1)*block_size + seq_lens % block_size
)
```

## 6）验证命令

```bash
# 测试无 --force-dense-minicpm（启用稀疏注意力）
# 正确性验证
python benchmark/soar/demo_sala/eval/eval.py --preset sparse_qkv_w8

# 速度验证
python benchmark/soar/demo_sala/eval/speed_eval.py
```

## 7）结果汇总表

| 项目 | 基线（密集） | 新方案（稀疏） | 变化 | 备注 |
|---|---|---|---|---|
| --force-dense-minicpm | 有 | 已移除 | — | 启用稀疏注意力 |
| kv_indptr 来源 | 静态 [0,K,2K,...] | 来自 sparse_cu_seqlens_k_cpu | 修复 | 与图内计算匹配 |
| 过期 seqlens 清零 | 未做 | 已添加 | 安全措施 | 确保 plan/graph 一致性 |
| 预期加速 | — | 长输入 20-40% | — | 待用户验证 |

## 8）回滚说明

在 `benchmark/soar/demo_sala/prepare_env.sh` 的 `SGLANG_SERVER_ARGS` 中恢复 `--force-dense-minicpm`：
```bash
--kv-cache-dtype fp8_e5m2 --force-dense-minicpm${FUSED_QK_NORM_ROPE_ARG}
```

通过 `git revert` 回滚 `minicpm_backend.py` 的改动。

## 9）下一步建议

1. 测试正确性 —— 准确率须保持 ≥99% 以获得 C=1.0
2. 测试速度 —— 测量 s1、s8、smax 改善
3. 若稀疏注意力以不同错误崩溃，进一步排查（Bug 3：scratch kernel 的 FP8 反量化可能需要处理）
4. 若稳定，继续推进 Path C.1（融合 scaled-residual+RMSNorm）以获得额外 3-8% 加速
