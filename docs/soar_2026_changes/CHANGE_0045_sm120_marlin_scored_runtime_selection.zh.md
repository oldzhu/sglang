# CHANGE_0045：SM120 Marlin 运行时打分选择

## 背景与动机

`CHANGE_0043` 已经证明服务过程中确实启用了 SM120 专用的 Marlin 路径，日志中可以看到：

```text
[sgl-kernel] SM120 Marlin auto-config enabled: ...
```

但在重新构建补丁 `sgl-kernel` wheel 并进行端到端验证后，仅仅启用 SM120 配置表并没有带来明显提速：

- `acc_ori`: `79.31%`
- `S1`: `124.07`
- `S8`: `26.44`
- `Smax`: `21.86`

继续检查 launcher 后发现，上一版改动的提升空间有限，原因在于：

- Marlin launcher 仍然只是从一个较小的预编译内核集合中选择**第一个合法配置**
- 它并不会把当前 GEMM 形状下的所有合法候选都拿出来比较
- 也不会把已找到的更优形状选择缓存起来复用

因此，本次迭代不去新增内核形状，而是选择更安全的一步：对当前形状下所有合法的 SM120 候选进行运行时打分，选出最佳配置，并按形状缓存该选择。

## 规则合规说明

本次修改仍然严格位于最新 toolkit 官方鼓励的 `路径一：量化加速` 范围内：

- 仅调整 Marlin W4A16 推理内核的 launch 选择逻辑
- 不替换官方提供的 MiniCPM-SALA 基座模型
- 不修改比赛并发规则，也不重新启用 prefix cache
- 继续兼容当前 `prepare_env.sh` + 本地 `sgl-kernel` wheel 的提交工作流

## 详细实现计划

修改前计划：

1. 再次确认最新 toolkit / competition 页面仍然允许路径一的 Marlin 调优，并继续要求通过 `prepare_env.sh` 组织提交环境。
2. 确认当前 SM120 launcher 的选择逻辑依旧是“第一个合法配置返回”，而不是对候选进行比较。
3. 增加按形状缓存的选择逻辑，使打分代价只在新形状首次出现时支付一次，而不是每次 GEMM 都重复支付。
4. 将新的打分逻辑仅限制在 SM120 路径，保持非 SM120 的行为完全不变。

## 实际代码变更

修改文件：

- `sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu`

具体变更：

1. 为 SM120 增加了基于打分的运行时选择逻辑，仍然只在现有预编译 Marlin 内核集合内选择。
2. 增加了按形状缓存的配置选择，重复形状可以直接复用已选出的配置。
3. 非 SM120 路径保持原有行为不变，仍然沿用“第一个合法配置”策略。
4. 扩展了单次 SM120 日志输出，增加“当前配置来自打分还是缓存复用”、估计 occupancy 和分数字段，便于后续核对。

本次对所有合法 SM120 候选使用的打分输入包括：

- M 方向 tile 覆盖率
- 通过总 tile 数体现的 N 方向覆盖与并行度
- 预估并行能力
- shared memory fit
- 来自 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 的可选 occupancy 信号

本次设计讨论中保留的关键结论：

- **选择发生在运行时，并且是按 GEMM 形状进行的**
- 它**不是**一次性的全局初始化选择，也不是编译时决定
- 新增的打分开销通过形状缓存进行摊销，目标是在热点形状上只支付一次，而不是每次 launch 都重复付费

## 讨论要点备忘

这一节专门记录本次实现讨论中的关键信息，便于后续回顾。

### 1. 这里的“auto-scored”具体是什么意思

这里的意思是：launcher 不再按候选表顺序返回第一个合法配置，而是把当前形状下所有合法的已编译候选取出来，进行比较后再选择最优项。

参与决策的运行时形状特征包括：

- `M`、`N`、`K`
- tile 覆盖情况
- 预期并行度
- shared memory 压力
- occupancy 估计

### 2. 选择在什么时候发生

选择发生在**运行时**，不是编译时。

- 编译时决定 wheel 中有哪些 Marlin 内核变体存在
- 运行时根据当前 GEMM 形状，决定本次 launch 使用哪个已经编译好的变体

这意味着 warmup、CUDA graph capture，以及后续未被图捕获到的新形状，都可能各自触发一次形状级选择；后续重复形状则应命中缓存。

### 3. 为什么不继续使用单一静态配置

单一的静态 Marlin 配置很难在全部服务形状上都保持最优，因为工作负载至少会随着以下因素发生变化：

- 批大小变化带来的 `M` 变化
- prefill / decode 路径差异
- 图捕获形状与非图捕获形状的混合
- tile 数量与 occupancy 行为差异

因此本次实现保留的核心判断是：

- 很难存在一个静态配置，能够在所有热点形状上都同时给出最佳性能

### 4. 打分开销值不值得付

相较纯静态配置，的确会增加一些 host 侧开销。

但目标设计是：

- 只在某个新形状第一次出现时，对合法候选做一次打分
- 一旦选出结果，后续相同形状直接命中缓存复用

因此本次方案追求的是**带缓存的运行时选择**，而不是对每次 launch 都做重复动态搜索。

## 验证命令

重新构建补丁 wheel：

```bash
cd /root/sglang-minicpm/sgl-kernel
rm -rf build dist
python -m pip install -U uv scikit-build-core ninja
make build MAX_JOBS=2 CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1"
```

正确性验证：

```bash
python3 eval_model.py --api_base http://127.0.0.1:30000 --model_path <MODEL_DIR> --data_path <DATA_DIR>/perf_public_set.jsonl --concurrency 32
```

速度验证：

```bash
bash SOAR/bench_serving.sh http://127.0.0.1:30000
```

从服务日志确认新的 SM120 选择日志：

```bash
grep -F "[sgl-kernel] SM120 Marlin auto-config enabled:" <server_log_file>
```

## 结果摘要表

| 项目 | 修改前 | 修改后 |
| --- | --- | --- |
| SM120 配置选择 | 第一个合法候选 | 对合法候选打分后选择最优 |
| 选择时机 | 运行时 | 运行时 |
| 重复形状开销 | 重复顺序扫描 | 走缓存复用 |
| 非 SM120 行为 | 第一个合法候选 | 不变 |
| 速度结果 | 待验证 | 待验证 |

## 回滚说明

如果打分式运行时选择带来不稳定性，或者不能提升服务速度：

1. 回退 `sgl-kernel/csrc/gemm/marlin/gptq_marlin.cu`
2. 重新构建 `sgl-kernel` wheel
3. 将新 wheel 重新拷入提交目录

回滚命令：

```bash
git revert <commit>
```

## 后续建议

1. 在 fcloud 重新构建补丁 `sgl-kernel` wheel，并确认新的 SM120 日志行出现。
2. 先跑正确性，因为当前最近一次 `acc_ori` 仍低于安全提交水位。
3. 只有当正确性稳定后，再将 `S1`、`S8`、`Smax` 与 `CHANGE_0041` 基线以及上一版 `CHANGE_0043` wheel 做比较。