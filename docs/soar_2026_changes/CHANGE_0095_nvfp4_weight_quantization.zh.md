# CHANGE_0095: NVFP4 权重量化

## 背景与动机

NVFP4（FP4 E2M1）权重量化利用 Blackwell 张量核心的原生加速能力，同时实现权重存储和计算的优化。不同于 GPTQ W4A16（将权重反量化为 FP16 进行 GEMM），NVFP4 在 SM120 张量核心上原生执行 FP4×FP4 GEMM，在高批次大小下可能实现约 2 倍的计算吞吐量提升。

冠军队伍 Slightwind（3 次周冠军，79.22 分）使用 NVFP4 权重量化。我们的 fcloud GPU（NVIDIA RTX 6000D，计算能力 12.0，Blackwell）具有原生硬件支持。

SGLang 已通过 `ModelOptFp4LinearMethod` 和 `--quantization modelopt_fp4` 提供完整的 NVFP4 推理支持，但缺少离线量化生成流程。本脚本填补了这一空白。

## 规则合规性

- **现场量化** ✓（Python 脚本在比赛机器上运行）
- **模型大小 ≤ 2GB** ✓（FP4 模型与 GPTQ INT4 大小相当或更小）
- **量化时间** — 预计校准+量化约 10-20 分钟
- **SM120 要求** ✓（已确认评测机器为 Blackwell GPU）

## 实现方案

### 量化脚本：`benchmark/soar/demo_sala/quantize_nvfp4.py`

**流程**：
1. 从 `/root/models/openbmb/MiniCPM-SALA-Copy` 加载 BF16 模型
2. **校准**：使用 perf_public_set.jsonl 数据运行 32 次前向传播，记录每个 Linear 层的最大绝对激活值
3. **权重量化**：对每个 Linear 层：
   - 将权重分组为 128 个元素的块（group_size）
   - 计算每块 FP8 E4M3 缩放因子：`scale = max(abs(block)) / 6.0`
   - 将每个权重值四舍五入到最近的 FP4 E2M1 值：{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
   - 将两个 4 位值打包为一个 uint8
4. **保存检查点**，在 `config.json` 和 `hf_quant_config.json` 中包含量化配置

**输出格式**：
```
weight:         [out, in//2]           uint8        (打包的 FP4 E2M1)
weight_scale:   [out, in//group_size]  float8_e4m3  (逐块缩放因子)
weight_scale_2: [1]                    float32      (逐张量权重缩放因子)
input_scale:    [1]                    float32      (校准的激活缩放因子)
```

## 验证命令

```bash
# 量化模型
python3 quantize_nvfp4.py \
    --input /root/models/openbmb/MiniCPM-SALA-Copy \
    --output /root/models/nvfp4_minicpm \
    --calib-data /root/data/perf_public_set.jsonl \
    --calib-samples 32 --group-size 128 --verify

# 使用 NVFP4 启动服务器
python3 -m sglang.launch_server \
    --model-path /root/models/nvfp4_minicpm \
    --quantization modelopt_fp4 ...

# 测试准确率和速度
python3 scripts/fcloud/fcloud_workflow.py accuracy
python3 scripts/fcloud/fcloud_workflow.py speed --variant all
```

## 结果摘要

| 指标 | 基线 (GPTQ W4A16) | 预期 (NVFP4 W4A4) |
|------|-------------------|-------------------|
| S1 | 113.67s | ~105-115s（边际改善）|
| S8 | 41.07s | ~35-40s |
| Smax | 34.15s | ~25-30s |
| 准确率 | 80.64% | ≥79% 目标 |
| C | 1.0 | ≥0.96 目标 |

## 回滚方案

将 `prepare_env.sh` 恢复为使用 `--quantization gptq_marlin --force-dense-minicpm`，并将 model_path 指向 GPTQ 模型。

## 后续步骤

1. **在 fcloud 上测试** — 量化后运行准确率评估
2. **调优 exclude_modules** — 找出对精度敏感的层，保持更高精度
3. **与 ModelOpt 对比** — 如果有 `nvidia-modelopt`，对比质量
4. **与 EAGLE3 结合** — NVFP4 目标模型 + EAGLE3 推测，实现最大吞吐量
