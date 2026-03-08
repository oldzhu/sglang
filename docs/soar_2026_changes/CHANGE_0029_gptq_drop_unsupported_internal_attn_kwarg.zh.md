# CHANGE_0029_gptq_drop_unsupported_internal_attn_kwarg

## 1) 背景与动机
- 在 CHANGE_0028 后，GPTQ 加载报错：
  - `TypeError: MiniCPMSALAForCausalLM.__init__() got an unexpected keyword argument '_attn_implementation'`
- 说明该内部参数与当前模型构造函数签名不兼容。

## 2) 规则合规说明
- 本次仅修复预处理兼容性。
- 不涉及竞赛运行时策略改动。

## 3) 特性计划
- 保留 `attn_implementation=flash_attention_2`。
- 从 `GPTQModel.load(...)` 中移除不兼容的 `_attn_implementation` 参数。

## 4) 实际代码改动
- `benchmark/soar/demo_sala/preprocess_model.py`
  - 删除 GPTQ 加载调用中的 `_attn_implementation=attn_impl`。

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
- 低。仅移除不被支持的参数，同时保留必须的注意力实现覆盖。

## 7) 回滚方案
1. 回退 `preprocess_model.py` 中 GPTQ 加载调用。

## 8) 下一步建议
- 若预处理成功，继续做量化模型启动与 eval/bench 对比。
