# CHANGE_0028_gptq_flash_attn2_load_fix

## 1) 背景与动机
- GPTQ 预处理已进入模型构造阶段，但报错：
  - `AssertionError: Only flash_attention_2 is supported for sparse attention`
- MiniCPM-SALA 在初始化时要求使用 flash-attention-2 实现标记。

## 2) 规则合规说明
- 本次仅为预处理兼容性修复。
- 不涉及评测规避手段，不替换基座模型。

## 3) 特性计划
- 在 GPTQ 加载时强制指定注意力实现。
- 增加环境变量可配置。

## 4) 实际代码改动
- `benchmark/soar/demo_sala/preprocess_model.py`
  - 新增环境变量：`SOAR_GPTQ_ATTN_IMPL`（默认 `flash_attention_2`）
  - 在 `GPTQModel.load(...)` 中同时传入 `attn_implementation` 与 `_attn_implementation`
  - 日志增加最终注意力实现值

## 5) 验证命令
```bash
python3 -m py_compile benchmark/soar/demo_sala/preprocess_model.py

export SOAR_QUANT_MODE=gptq
export SOAR_TRUST_REMOTE_CODE=true
export SOAR_GPTQ_ATTN_IMPL=flash_attention_2
export SOAR_GPTQ_CALIBRATION_FILE=/root/data/perf_public_set.jsonl
bash benchmark/soar/demo_sala/prepare_model.sh \
  --input /root/models/openbmb/MiniCPM-SALA \
  --output /root/models/openbmb/MiniCPM-SALA-gptq
```

## 6) 风险
- 低到中：依赖 GPTQModel 对参数名的兼容性。
- 缓解：同时传两种常见参数名，提升兼容性。

## 7) 回滚方案
1. 回退 `preprocess_model.py` 中 `GPTQModel.load(...)` 新增参数与环境变量处理。

## 8) 下一步建议
- 若预处理通过，继续做量化模型启动与 eval/bench 对比测试。
