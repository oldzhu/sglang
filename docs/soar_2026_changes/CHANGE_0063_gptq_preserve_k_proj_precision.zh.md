# CHANGE_0063 GPTQ 保留 K-Projection 精度

## 背景与动机

最近几轮校准任务范围和校准样本数实验都没有把正确性稳定拉回目标区间。当前最弱、最不稳定的类别仍然集中在 `qa`、`mcq`、`cwe`，这说明问题可能已经不只是校准覆盖度，而是注意力路径本身对量化误差比较敏感。

在注意力几个投影里，`k_proj` 是最合理、最低风险的回退目标，因为 Key 侧精度对注意力分数质量的影响通常更大。K 侧的小误差会直接扰动 softmax 前的路由分数，尤其在长上下文问答和信息提取任务里，容易放大成最终答案错误。

因此，本次迭代保持 feature 范围很窄：把 `self_attn.k_proj` 从 GPTQ 量化范围中排除，使其保留更高精度，而其余 GPTQ W4A16 + Marlin 路径保持不变。

## 规则合规说明

本改动符合 2026-03-31 重新检查的最新 SOAR 比赛页与 toolkit 页面要求。

- 仅修改预处理阶段的量化范围。
- 保持官方 `prepare_env.sh` 与 `prepare_model.sh --input/--output` 工作流不变。
- 不替换 MiniCPM-SALA 基座模型。
- 不修改并发规则、prefix-cache 行为或服务启动契约。
- 对模型其余部分仍保持当前 GPTQ + Marlin + FP8 KV cache 运行路径。

## 详细实施计划

改动前计划：

1. 保持现有 GPTQ 流程和 layer-aware selector 不变。
2. 在默认 GPTQ 量化范围中排除 `self_attn.k_proj`。
3. 在模块不匹配的 retry 路径里也保持同样的排除策略，避免 feature 失效。
4. 本轮不同时回退 Q 和 K，避免一次改变两个重要变量。

## 实际代码改动

修改文件：

- `benchmark/soar/demo_sala/prepare_env.sh`
- `benchmark/soar/demo_sala/preprocess_model.py`

具体变更：

1. 更新默认 `SOAR_GPTQ_INCLUDE_MODULES`，移除 `self_attn.k_proj`。
2. 更新默认 `SOAR_GPTQ_EXCLUDE_MODULES`，新增 `self_attn.k_proj`。
3. 更新 `preprocess_model.py` 内部默认值，使直接运行 preprocess 时也采用同样策略。
4. 更新 GPTQ retry fallback 路径，使其同样保持 `self_attn.k_proj` 不参与量化。

## 设计说明

### 为什么优先保留 K，而不是 Q 或 V

K 侧精度对注意力路由影响更直接。如果 K 太噪，softmax 可能把注意力权重分配到错误 token，上游错误会继续向后传递。相比之下，V 侧误差通常发生在路由已经决定之后，因此破坏性没有那么强。

### 为什么不直接同时保留 Q 和 K

那样会一次性改变两个重要变量，也会提高速度回退风险。先只保留 `k_proj`，可以更干净地判断 K 精度是否是当前最关键的缺失因素。

### 为什么连 retry 路径也要一起改

如果 mismatch retry 回退路径重新把 `k_proj` 放回量化范围，那么一旦触发 retry，这个 feature 就会被悄悄绕过。让主路径和 retry 路径保持同一策略，实验才是自洽的。

## 验证命令

Shell 语法检查：

```bash
bash -n benchmark/soar/demo_sala/prepare_env.sh
```

Python 语法检查：

```bash
python3 -m py_compile benchmark/soar/demo_sala/preprocess_model.py
```

预处理验证：

```bash
bash benchmark/soar/demo_sala/prepare_model.sh --input <RAW_MODEL_DIR> --output <OUTPUT_MODEL_DIR>
```

正确性验证：

```bash
python3 eval_model.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <MODEL_DIR> \
  --data_path <DATA_DIR>/perf_public_set.jsonl \
  --concurrency 32
```

服务速度验证：

```bash
bash SOAR/bench_serving.sh http://127.0.0.1:30000
```

回滚到上一版量化范围：

```bash
export SOAR_GPTQ_INCLUDE_MODULES=self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj,mlp.down_proj
export SOAR_GPTQ_EXCLUDE_MODULES=self_attn.o_gate,self_attn.z_proj
```

## 结果汇总表

| 项目 | 修改前 | 修改后 |
| --- | --- | --- |
| 默认 GPTQ 注意力投影范围 | Q + K + V + O | Q + V + O |
| `k_proj` 处理方式 | 量化 | 保留更高精度 |
| Retry 路径行为 | 可能重新量化 K | 一致保留 K 回退 |
| 速度预期 | 当前 baseline | 可能略有回退 |
| 精度预期 | `qa/mcq/cwe` 不稳定 | 目标是改善路由稳定性 |

## 回滚说明

如果保留 `k_proj` 不能明显改善正确性，或者速度代价太大：

1. 把 `self_attn.k_proj` 恢复到 GPTQ include 列表
2. 从 GPTQ exclude 列表中移除它
3. 重新执行相同的 preprocess、correctness 和 serving 检查

## 下一步建议

1. 先重点看 `qa` 和 `cwe` 是否改善，因为这两个 bucket 看起来最像是 K 精度敏感型问题。
2. 如果有效但还不够，下一轮 accuracy feature 再考虑同时保留 Q 和 K。
3. 如果无效，就应从 calibration / projection rollback 这条线跳出，转向其他 accuracy 假设，而不是继续增加校准样本。