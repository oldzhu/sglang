# CHANGE_0055 Prepare-Env 依赖冻结

## 背景与动机

官方诊断型提交已经证明，官方评测环境解析出的依赖版本比用户长期复用的 fcloud 实例更新：

- 官方：`gptqmodel==5.8.0`、`transformers==5.3.0`、`torchao==0.16.0`
- fcloud：`gptqmodel==5.7.0`、`transformers==4.57.1`、`torchao==0.9.0`

这种差异会让 GPTQ 排障结果变得不可靠，因为失败既可能来自提交代码，也可能只是来自依赖漂移。在继续为新版依赖补兼容逻辑之前，本次迭代先把提交环境冻结到 fcloud 已实际使用过的那组版本。

## 规则合规说明

本改动符合最新 SOAR toolkit 的提交流程要求。

- 保持 `prepare_env.sh` + `prepare_model.sh --input/--output` 契约不变。
- 仅在 `prepare_env.sh` 里固定 Python 包版本，这属于官方允许的自定义环境构建范围。
- 不替换 MiniCPM-SALA 基座模型，也不修改评测逻辑。
- 不改变 prefix cache、并发配置或任何隐藏评测行为。

## 详细实施计划

改动前计划：

1. 把浮动的 `gptqmodel` 安装改成精确版本安装。
2. 显式固定 `transformers`，不再依赖下游自动解析。
3. 重新安装 `torchao==0.9.0`，确保与 fcloud 已验证环境一致。
4. 把 `prepare_model.sh` 从诊断探针恢复到真实 preprocess 入口。
5. 在 `prepare_env.sh` 中打印实际生效的版本，确保官方日志能证明 pin 已经成功。

## 实际代码改动

修改文件：

- `benchmark/soar/demo_sala/prepare_env.sh`
- `benchmark/soar/demo_sala/prepare_model.sh`

具体变更：

1. 将原先浮动的 `uv pip install gptqmodel ...` 改为精确且强制重装的依赖集合：
   - `gptqmodel==5.7.0`
   - `transformers==4.57.1`
2. 将 `torchao` 安装改为强制重装并固定到 `0.9.0`。
3. 新增一个简短的安装后 Python 探针，打印以下模块的实际版本与导入路径：
   - `torch`
   - `gptqmodel`
   - `transformers`
   - `torchao`
4. 将 `prepare_model.sh` 恢复为调用 `preprocess_model.py`，不再指向临时诊断探针。

## 设计说明

### 为什么要固定整套已验证依赖，而不只固定 `gptqmodel`

本次观察到的环境差异并不只是 `gptqmodel`。`transformers` 和 `torchao` 也发生了变化，而这两个包都可能直接影响模型加载路径与量化支持行为。只固定一个包，无法真正消除依赖漂移带来的噪声。

### 为什么优先回到旧版已验证栈

fcloud 环境已经在前一轮 GPTQ 排障中真实跑过这一组版本。先冻结到这组已知环境，比同时升级到多个新版本再排查新行为要低风险得多。

### 以后新版依赖会不会提升量化精度

新版 `gptqmodel` / `transformers` 可能会带来 loader 修复、API 变化、kernel 集成变化或量化相关 bugfix，但这并不等于一定会提升量化精度。量化后的精度通常更受量化方法、校准数据、group size、模型特定兼容处理等因素影响。若想判断 `gptqmodel==5.8.0` 与 `transformers==5.3.0` 是否真的更好，必须在同一模型、同一校准集、同一评测集上做严格 A/B 对比。

## 验证命令

环境安装验证：

```bash
bash benchmark/soar/demo_sala/prepare_env.sh
```

显式检查固定版本：

```bash
python3 - <<'PY'
import importlib
import json
for name in ["torch", "gptqmodel", "transformers", "torchao"]:
    module = importlib.import_module(name)
    print(json.dumps({
        "module": name,
        "version": getattr(module, "__version__", "unknown"),
        "file": getattr(module, "__file__", None),
    }, ensure_ascii=False))
PY
```

恢复 preprocess 验证：

```bash
bash benchmark/soar/demo_sala/prepare_model.sh --input <RAW_MODEL_DIR> --output <OUTPUT_MODEL_DIR>
```

## 结果汇总表

| 项目 | 修改前 | 修改后 |
| --- | --- | --- |
| `gptqmodel` 安装 | 浮动解析到最新版本 | 固定为 `5.7.0` |
| `transformers` 安装 | 由依赖解析决定 | 固定为 `4.57.1` |
| `torchao` 安装 | 单独恢复，但未与整套依赖一起严格冻结 | 固定为 `0.9.0` |
| 官方/fcloud 可复现性 | 依赖集合漂移 | 对齐到已知 fcloud 栈 |
| `prepare_model.sh` 入口 | 诊断探针 | 真实 preprocess 入口 |

## 回滚说明

如果旧版依赖集合在当前官方镜像上无法正常安装，或者证明不兼容：

1. 恢复为浮动安装，或更新为另一组经过验证的新 pin
2. 重新运行依赖探针，确认实际生效版本
3. 如有需要，再进入下一轮迭代，把 fcloud 与官方都统一升级到 `gptqmodel==5.8.0` / `transformers==5.3.0` 的新栈

## 下一步建议

1. 用这组固定依赖重新提交官方任务，并确认日志中出现预期的版本号。
2. 如果 preprocess 仍然失败，再在没有“依赖漂移”这个变量的前提下继续排查。
3. 后续如果时间允许，再对旧栈与新栈做严格 A/B，判断新版依赖是否真的对加载路径或量化精度有帮助。