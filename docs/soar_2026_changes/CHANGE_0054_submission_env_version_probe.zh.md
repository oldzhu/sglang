# CHANGE_0054 提交环境版本探针

## 背景与动机

最近官方提交开始出现新的失败现象，表现上很像官方评测环境与用户自己的 fcloud 实例在关键预处理依赖上存在差异。当前重点怀疑对象是 `gptqmodel`、`transformers` 以及相关运行时包，因为本地行为与官方行为已经出现明显分叉。

在继续新增兼容性代码之前，本次迭代先加入一个聚焦的诊断型提交 feature：提供一个独立的 preprocess 探针脚本，在官方平台上尽早打印实际生效的 Python 与依赖版本，然后主动报错退出，以便这些版本信息稳定出现在平台最后 50 行错误日志中。

## 规则合规说明

本改动符合最新 SOAR competition 与 toolkit 的要求。

- 保持 `prepare_env.sh` + `prepare_model.sh --input/--output` 提交契约不变。
- 不替换 MiniCPM-SALA 官方基座模型，也不修改评测逻辑。
- 不改变运行时并发、prefix cache 或服务启动参数。
- 这是一个仅用于排障的提交期诊断 feature，用来观察官方预处理环境。

## 详细实施计划

改动前计划：

1. 保持现有 GPTQ preprocess 主逻辑不动。
2. 新增一个独立的诊断 preprocess 入口，而不是把主动失败逻辑混进当前量化流程。
3. 尽可能早地打印 `torch`、`gptqmodel`、`transformers` 的版本和导入位置。
4. 在打印完成后立刻主动抛错，确保官方平台最后 50 行日志里能看到这些信息。
5. 仅在本次探针提交中，把 `prepare_model.sh` 临时指向这个诊断入口。

## 实际代码改动

修改文件：

- `benchmark/soar/demo_sala/preprocess_model_001.py`
- `benchmark/soar/demo_sala/prepare_model.sh`

具体变更：

1. 新增独立脚本 `preprocess_model_001.py`。
2. 新脚本仍然解析 `--input` 与 `--output`，保持与官方提交接口一致。
3. 它会打印：
   - 调用上下文（`argv`、cwd、输入输出路径）
   - Python 版本与解释器路径
   - `torch` 版本与导入文件路径
   - `gptqmodel` 版本与导入文件路径
   - `transformers` 版本与导入文件路径
4. 每个依赖探针都具备容错性：若导入失败，会记录异常类型与异常信息，而不是直接吞掉上下文。
5. 打印完成后立即抛出一个故意的 `RuntimeError`，让提交尽早失败，从而把环境信息暴露到官方错误日志里。
6. `prepare_model.sh` 在本次诊断提交中临时改为调用 `preprocess_model_001.py`。

## 设计说明

### 为什么要用独立脚本

当前 `preprocess_model.py` 已经承载了正在推进的 GPTQ 兼容性逻辑。单独增加一个探针脚本，可以把这次排障提交与正常 preprocess 路径彻底隔离，避免把“主动失败”的逻辑混入工作流，也让回滚更直接。

### 为什么要主动失败

这次迭代的目标不是生成处理后的模型，而是逼官方平台把关键版本信息打印出来。如果继续走完整 preprocess 流程，日志里会混入大量无关内容，最后 50 行也更容易看不到版本信息。打印后立刻失败，能把日志焦点收敛在环境差异本身。

### 为什么除了版本号还要打印导入路径

仅看版本字符串，有时不足以判断真实问题，尤其是在 editable install、模块遮蔽、多个 site-packages 并存时。把模块文件路径一起打印，可以更明确地说明官方环境究竟加载的是哪一份包。

## 验证命令

本地通过 wrapper 执行：

```bash
bash benchmark/soar/demo_sala/prepare_model.sh --input <RAW_MODEL_DIR> --output <OUTPUT_MODEL_DIR>
```

直接执行 Python 探针：

```bash
python3 benchmark/soar/demo_sala/preprocess_model_001.py \
  --input <RAW_MODEL_DIR> \
  --output <OUTPUT_MODEL_DIR>
```

预期结果：

- 会打印 Python、`torch`、`gptqmodel`、`transformers` 的版本探针日志
- 随后脚本以故意的 `RuntimeError` 退出

## 结果汇总表

| 项目 | 修改前 | 修改后 |
| --- | --- | --- |
| 官方 preprocess 可见性 | 只能看到当前 GPTQ 路径的失败栈 | 失败前先输出明确的环境探针日志 |
| 依赖对比方式 | 只能间接推断 | 直接打印 `torch` / `gptqmodel` / `transformers` 版本 |
| 导入来源证据 | 无 | 输出模块文件路径 |
| 现有 GPTQ preprocess 主逻辑 | 会继续执行并在后面失败 | 保持不变，但本次诊断提交暂时绕过 |

## 回滚说明

在拿到官方环境信息后：

1. 把 `benchmark/soar/demo_sala/prepare_model.sh` 恢复为调用 `preprocess_model.py`
2. 根据后续是否还需要环境探针，决定保留还是删除 `preprocess_model_001.py`
3. 对比完官方版本与 fcloud 版本之后，再继续推进真正的 GPTQ 修复

## 下一步建议

1. 提交这份探针包，拿到官方最后 50 行日志。
2. 将日志里的 `torch`、`gptqmodel`、`transformers` 版本与 fcloud 实例逐项对比。
3. 如果版本确实不同，就带着这个精确差异去继续询问官方平台 owner。
4. 如果版本一致，就不要再把问题归因于环境差异，而应回到现有 GPTQ 兼容性链路继续排查。