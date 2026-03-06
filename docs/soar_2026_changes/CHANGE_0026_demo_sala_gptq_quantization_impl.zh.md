# CHANGE_0026_demo_sala_gptq_quantization_impl

## 1) 背景与动机
- 上一版本 demo 仅提供 GPTQ 脚手架与预检，未包含实际量化流程。
- 团队需要在 `preprocess_model.py` 中实现可执行的量化预处理路径用于提交。

## 2) 规则合规说明
- 本次仅修改提交预处理逻辑，保持 `prepare_model.sh --input/--output` 官方接口。
- 不涉及运行时违规策略，不替换基座模型。

## 3) 特性计划
- 在 `preprocess_model.py` 中实现 GPTQModel 真实量化流程。
- 默认仍保留 `copy` 模式。
- 通过 CLI/环境变量配置校准数据与量化参数。

## 4) 实际代码改动
- `benchmark/soar/demo_sala/preprocess_model.py`
  - 新增 JSONL 校准数据加载。
  - 新增 GPTQ 量化执行函数：
    - `GPTQModel.load(...)`
    - `model.quantize(calibration_texts, batch_size=...)`
    - `model.save(output_dir)`
  - 新增 `quantize_config.json` 输出校验。
  - 新增参数与环境变量：
    - `--calibration-file` / `SOAR_GPTQ_CALIBRATION_FILE`
    - `--calibration-field` / `SOAR_GPTQ_CALIBRATION_FIELD`
    - `--calibration-samples` / `SOAR_GPTQ_CALIBRATION_SAMPLES`
    - `--gptq-bits` / `SOAR_GPTQ_BITS`
    - `--gptq-group-size` / `SOAR_GPTQ_GROUP_SIZE`
    - `--gptq-batch-size` / `SOAR_GPTQ_BATCH_SIZE`
- `benchmark/soar/demo_sala/README.md`
  - 补充 `gptq` 模式的实际用法与环境变量示例。

## 5) 验证命令
```bash
python3 -m py_compile benchmark/soar/demo_sala/preprocess_model.py

# copy 模式
bash benchmark/soar/demo_sala/prepare_model.sh --input /path/raw --output /path/out

# gptq 模式
SOAR_QUANT_MODE=gptq \
SOAR_GPTQ_CALIBRATION_FILE=/path/calibration.jsonl \
bash benchmark/soar/demo_sala/prepare_model.sh --input /path/raw --output /path/out
```

## 6) 风险
- 中等。量化精度与速度收益依赖校准数据质量与依赖环境。
- 缓解：依赖预检与输出文件校验。

## 7) 回滚方案
1. 回退 `benchmark/soar/demo_sala/preprocess_model.py`。
2. 回退 `benchmark/soar/demo_sala/README.md` 对应新增内容。

## 8) 下一步建议
- 后续增加一个单独特性，用于 marlin 兼容性验证与量化模型专用启动/评测配置。
