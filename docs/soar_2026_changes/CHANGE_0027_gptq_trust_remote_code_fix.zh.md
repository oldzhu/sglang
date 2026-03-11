# CHANGE_0027_gptq_trust_remote_code_fix

## 1) 背景与动机
- MiniCPM-SALA 的 GPTQ 预处理报错：
  - `ValueError: ... contains custom code ... pass trust_remote_code=True`
- 该问题会在量化开始前直接中断流程。

## 2) 规则合规说明
- 本次仅修复预处理可用性。
- 保持官方 `prepare_model.sh --input --output` 接口。
- 不涉及运行时评测规则绕过。

## 3) 特性计划
- 在 GPTQModel 加载路径开启 `trust_remote_code`。
- 增加环境变量开关以便显式控制。

## 4) 实际代码改动
- `benchmark/soar/demo_sala/preprocess_model.py`
  - 新增 `trust_remote_code` 解析：
    - `SOAR_TRUST_REMOTE_CODE`（默认 `true`）
  - 在 `GPTQModel.load(...)` 中传入 `trust_remote_code`。
  - 打印日志显示最终值。

## 5) 验证命令
```bash
python3 -m py_compile benchmark/soar/demo_sala/preprocess_model.py

export SOAR_QUANT_MODE=gptq
export SOAR_GPTQ_CALIBRATION_FILE=/root/data/perf_public_set.jsonl
bash benchmark/soar/demo_sala/prepare_model.sh \
  --input /root/models/openbmb/MiniCPM-SALA \
  --output /root/models/openbmb/MiniCPM-SALA-gptq
```

## 6) 风险
- `trust_remote_code=True` 会执行模型仓库自定义代码。
- 对 MiniCPM-SALA 来说这是正确加载模型所必需。

## 7) 回滚方案
1. 回退 `benchmark/soar/demo_sala/preprocess_model.py` 中 GPTQ 加载改动。

## 8) 下一步建议
- 预处理成功后，继续进行 eval + 3x 速度测试，评估量化收益与精度影响。
