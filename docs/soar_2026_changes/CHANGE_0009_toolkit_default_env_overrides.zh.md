# CHANGE_0009_toolkit_default_env_overrides

## 1）背景与动机
- 在部分环境中，toolkit-default 启动档位会在服务初始化阶段报 `Not enough memory`。
- 需要快速参数化验证启动 OOM 是否由以下因素引起：
  - `chunked-prefill-size` 过大；
  - 自动推断的静态显存比例过低。

## 2）SOAR 规则合规性检查
- 本次仅为脚本层运维参数增强。
- 不修改模型/内核/推理逻辑。
- 不与官方评测约束冲突。

## 3）改动前计划
- 仅修改一个脚本：`benchmark/soar/scripts/launch_toolkit_default.sh`。
- 当环境变量未设置时，保持默认行为不变。

## 4）实际代码改动
- 新增可选环境变量：
  - `CHUNKED_PREFILL_SIZE`（默认 `32768`）
  - `MEM_FRACTION_STATIC`（可选，设置后才传入）
- 启动命令改为：
  - `--chunked-prefill-size "$CHUNKED_PREFILL_SIZE"`
  - 若设置则追加 `--mem-fraction-static "$MEM_FRACTION_STATIC"`

## 5）使用示例
```bash
# 保持 toolkit 默认行为
bash benchmark/soar/scripts/launch_toolkit_default.sh

# 测试降低 chunked prefill
CHUNKED_PREFILL_SIZE=8192 bash benchmark/soar/scripts/launch_toolkit_default.sh

# 测试显存静态比例
MEM_FRACTION_STATIC=0.80 bash benchmark/soar/scripts/launch_toolkit_default.sh

# 同时测试两者
CHUNKED_PREFILL_SIZE=8192 MEM_FRACTION_STATIC=0.80 bash benchmark/soar/scripts/launch_toolkit_default.sh
```

## 6）风险评估
- 风险很低，仅启动脚本参数化。

## 7）回滚说明
1. 回退 `benchmark/soar/scripts/launch_toolkit_default.sh`。

## 8）下一步建议
- 若验证有效，可在后续单独变更中同步到 profile `status` 输出以显示实际覆盖参数。
