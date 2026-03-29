# CHANGE_0053 Prepare-Model RoPE Scaling 配置清洗

## 背景与动机

一次官方提交在 `prepare_model.sh` 阶段失败，且失败发生在量化完成之前。报错出现在 GPTQModel 使用原始 checkpoint 和 Hugging Face 远程模型代码构造 MiniCPM-SALA shell model 的过程中。

关键错误为：

```text
ValueError: Unknown RoPE scaling type default
```

这说明原始输入 `config.json` 与 GPTQ 预处理阶段使用的 MiniCPM-SALA 远程建模代码之间存在环境敏感的兼容性问题。此前的 preprocess 流程会把原始模型目录直接传给 `GPTQModel.load(...)`，在 shell-model 构造前没有对不兼容配置字段做任何规范化处理。

因此，本次迭代只加入一个聚焦的提交兼容性 feature：在 GPTQ shell-model 构造前，对临时 GPTQ 加载源中的不兼容 RoPE scaling 配置做清洗，并补充面向官方平台排障的 rope-debug 日志，同时对 GPTQModel 内部的 MiniCPM-SALA config 规范化过程做定向 in-memory 补丁。

## 规则合规说明

本改动符合最新 SOAR toolkit 提交流程要求。

- 保持 `prepare_env.sh` + `prepare_model.sh --input/--output` 契约不变。
- 不替换 MiniCPM-SALA 官方基座模型。
- 不修改运行时并发、prefix cache 行为或评测逻辑。
- 仅在官方平台现场量化时，对预处理输入配置做兼容性规范化。

## 详细实施计划

改动前计划：

1. 检查 preprocess 流程，确认 GPTQ 加载直接使用原始源目录。
2. 增加一个仅用于 GPTQ 加载的临时模型视图构建步骤。
3. 当存在不兼容的默认 RoPE 标记时，在该临时视图里清洗 `config.json`。
4. 在 `GPTQModel.load(...)` 前输出精简的原始/清洗后 RoPE 字段日志。
5. 对 shell-model 构造前的 GPTQModel in-memory config 做定向观测与修正。
6. 保证原始输入目录不被原地修改。

## 实际代码改动

修改文件：

- `benchmark/soar/demo_sala/prepare_env.sh`
- `benchmark/soar/demo_sala/preprocess_model.py`

具体变更：

1. 新增临时 GPTQ load-source 准备步骤。
2. 新增配置清洗逻辑，当满足以下条件时移除 `rope_scaling`：
   - `rope_scaling` 是 dict
  - `rope_scaling["type"] == "default"`，或
  - `rope_scaling["rope_type"] == "default"`
3. 临时加载源优先使用符号链接，若不可用则回退为复制。
4. `GPTQModel.load(...)` 现在读取清洗后的临时模型视图，而不再直接读取原始源目录。
5. 在 preprocess 日志中输出清洗动作与临时源路径，便于官方日志排查。
6. 在 GPTQ load 成功后，把模型内部记录的源路径元数据恢复为真实的 `--input` 模型目录。
7. 将临时目录清理延后到 `model.save(...)` 完成之后，确保临时视图只用于 load 阶段的配置兼容，而不会污染 save 阶段的元数据统计。
8. 新增 rope-debug 日志，在 GPTQ shell-model 构造前输出原始与清洗后的 `rope_scaling` / `rope_type` 字段，以及实际使用的 load source。
9. 新增环境变量开关 `SOAR_GPTQ_FORCE_NULL_ROPE_SCALING=1`，用于在排障时强制把临时 GPTQ config 中的 `"rope_scaling"` 写成 `null`，应对“字段缺失却在下游被重新补成默认值”的怀疑场景。
10. 新增 `gptqmodel` 与 `transformers` 版本日志，便于对比本地与官方环境的依赖差异。
11. 新增 MiniCPM-SALA 定向的 in-memory 兼容补丁，并做成版本自适应：优先包装 GPTQModel 的 `normalize_hf_config_compat(...)`，若该版本不存在，则回退为包装 `build_shell_model(...)`。两种情况下都会打印内存态 config 快照，并在 shell-model 构造前清除 `rope_scaling`、`rope_parameters`、顶层 `rope_type` 中被下游补出的 `default` 标记。
12. 新增更晚的 `transformers.modeling_utils.PreTrainedModel.from_pretrained(...)` 与 `._from_config(...)` 补丁点，在 GPTQ turtle-model / direct-load 真正构造模型之前再次执行同样的 MiniCPM-SALA in-memory 清理。
13. 修复补丁安装器的控制流，确保 GPTQ 侧 hooks 与更晚的 `transformers` 侧 hooks 会在同一次运行里同时安装，而不是前一个分支过早 `return`。
14. 在这些真正的 `transformers` 加载入口前再次打印依赖版本，确保即便平台只保留最后 50 行日志，也仍然能看到 `gptqmodel` / `transformers` 版本用于比较。
12. 在 `prepare_env.sh` 中默认导出 `SOAR_GPTQ_DEBUG_IN_MEMORY_CONFIG=1` 并打印其值，确保官方提交默认开启 in-memory 诊断，除非显式覆盖。

## 设计说明

### 为什么删除 `rope_scaling` 而不是改写成别的类型

当前错误的根因是显式写出的 `type="default"` 不被接受，而不是已经明确证明该模型必须使用另一种 scaling 算法。因此，删除这个不兼容标记比猜测 `linear` 或 `dynamic` 更稳妥。

不同环境里，这个不兼容标记可能出现在 `type` 或 `rope_type` 字段上。现在的清洗逻辑会在任一字段等于 `default` 时保守地移除整个 `rope_scaling`，避免继续把不兼容 schema 传给 GPTQ shell-model load。

仅用于排障时，也可以打开 `SOAR_GPTQ_FORCE_NULL_ROPE_SCALING=1`，把临时配置中的 `rope_scaling` 显式写成 `null`，从而验证官方栈是否会在字段缺失时自行补成不兼容的默认值。

### 为什么要加 rope-debug 日志

官方平台在初版清洗逻辑上线后仍然复现了同样的加载失败。这意味着需要区分三种情况：原始输入配置本身就有问题、临时清洗配置并非预期值、或者下游 config 构造过程又重新生成了不兼容值。在 `GPTQModel.load(...)` 前输出紧凑的 rope-debug 日志，才能用平台日志把这三类情况分开。

### 为什么要补 in-memory GPTQModel config 补丁

后续官方日志已经证明：原始 config 和临时清洗 config 在进入 GPTQModel.load 之前都已经是 `rope_scaling=None`，但 MiniCPM-SALA 远程代码仍然收到了 `scaling_type="default"`。这说明问题很可能出在 GPTQModel / transformers 的内存态 config 规范化过程中，而不是原始文件本身。因此，最小且针对性的补丁点，就是在 GPTQModel 完成规范化之后、`loader.from_config(...)` 真正构造 shell model 之前，把被自动补出的默认 RoPE 标记清掉。

由于 GPTQModel 不同版本暴露的内部 hook 不完全一致，这个补丁必须做成版本自适应。有些环境里存在 `normalize_hf_config_compat(...)`，而另一些环境只能在 `build_shell_model(...)` 入口处做稳定拦截。

最新日志还表明：有些失败并不是发生在前面的 shell-config 路径，而是更晚地发生在 `transformers` 的直接加载路径 (`PreTrainedModel.from_pretrained(...)`)。因此，真正稳定的 workaround 还需要在这个实际构造模型的边界再补一次。

这要求两类 hook 必须一起安装。现在安装器会分别跟踪每个 hook，并输出最终安装了哪些 hook 的摘要，便于确认控制流是否正确执行。

### 为什么使用临时模型视图

官方提交平台每次都会从原始输入重新执行预处理。直接修改原始输入目录既不安全，也不利于排查。临时清洗视图可以把修复严格限定在预处理阶段。

### 为什么要在保存前恢复真实源路径

GPTQModel 会在内部记录加载源路径，并在 `model.save(...)` 阶段复用该路径做模型大小统计。如果继续保留临时清洗目录作为 canonical source，下游保存逻辑可能会把它误判成 Hugging Face repo id。把该元数据恢复为真实的 `--input` 路径，可以确保临时视图只参与 load，不影响 save。

## 验证命令

预处理验证：

```bash
bash benchmark/soar/demo_sala/prepare_model.sh --input <RAW_MODEL_DIR> --output <OUTPUT_MODEL_DIR>
```

检查原始配置字段：

```bash
python3 - <<'PY'
import json
from pathlib import Path
cfg = json.loads(Path("<RAW_MODEL_DIR>/config.json").read_text())
print(json.dumps(cfg.get("rope_scaling"), ensure_ascii=False, indent=2))
PY
```

使用显式 null workaround 的排障命令：

```bash
SOAR_GPTQ_FORCE_NULL_ROPE_SCALING=1 \
bash benchmark/soar/demo_sala/prepare_model.sh --input <RAW_MODEL_DIR> --output <OUTPUT_MODEL_DIR>
```

开启 in-memory config 跟踪的排障命令：

```bash
SOAR_GPTQ_DEBUG_IN_MEMORY_CONFIG=1 \
bash benchmark/soar/demo_sala/prepare_model.sh --input <RAW_MODEL_DIR> --output <OUTPUT_MODEL_DIR>
```

量化后正确性验证：

```bash
python3 benchmark/soar/demo_sala/eval_model_001.py \
  --api_base http://127.0.0.1:30000 \
  --model_path <OUTPUT_MODEL_DIR> \
  --data_path benchmark/soar/demo_sala/perf_public_set.jsonl \
  --concurrency 32
```

## 结果汇总表

| 项目 | 修改前 | 修改后 |
| --- | --- | --- |
| GPTQ load source | 原始模型目录 | 清洗后的临时模型视图 |
| 不兼容 `rope_scaling.type=default` | 无处理 | 在 GPTQ shell-model load 前移除 |
| RoPE 排障可见性 | 无 | GPTQ load 前输出原始/清洗后 rope-debug 日志 |
| GPTQ 内存态 config 可见性 | 无 | 输出依赖版本及版本自适应的 GPTQModel 与 `transformers` 加载入口内存态 config 快照 |
| GPTQ save 元数据源路径 | 临时清洗路径可能泄漏到保存阶段统计逻辑 | 保存前恢复为真实原始模型路径 |
| 原始输入是否被修改 | 不涉及 | 保持不变 |
| 官方 prepare-model 兼容性 | 失败 | 待验证 |

## 回滚说明

如果这次兼容性修复不需要或者效果不正确：

1. 删除临时 GPTQ load-source 准备 helper
2. 恢复直接 `GPTQModel.load(str(src), ...)`
3. 重新执行 `prepare_model.sh`

本 feature 只影响预处理兼容性，因此不涉及运行时回滚。

## 下一步建议

1. 重新提交官方任务，并先对比新的 `gptqmodel` / `transformers` 版本日志，确认本地与官方环境差异。
2. 重点查看新的 `[preprocess][rope-debug] in_memory_config ...` 日志，判断 `default` 是否是在 GPTQModel 规范化过程中被补出来的。
3. 在根因查清之前保留这套排障日志；根因清楚后，再决定最终方案是保留这个定向 in-memory 补丁，还是回收为更干净的上游兼容修复。