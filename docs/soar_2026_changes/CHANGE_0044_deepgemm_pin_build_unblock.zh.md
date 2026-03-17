# CHANGE_0044：DeepGEMM 固定版本失效导致的构建解阻

## 背景与动机

`CHANGE_0043` 已经加入了面向 SM120 的 Marlin 调优路径，以及提交包侧通过本地 `sgl-kernel` wheel 安装补丁内核的工作流。下一步必须在用户的 fcloud 环境中无 Docker 构建这个补丁 wheel。

仓库中已经存在可用的无 Docker 构建路径（`make build` / `uv build --wheel --no-build-isolation`），但当前构建会在 CMake 拉取三方依赖阶段失败，尚未进入 Marlin 代码编译。根因是 `sgl-kernel/CMakeLists.txt` 里 `repo-deepgemm` 的 git 固定版本已经失效：

- 旧固定版本：`54f99a8af537b3c6eb4819b69907ccbe2b600792`
- 实际报错：`fatal: reference is not a tree`

本次迭代的目标是仅通过替换为同一 fork 上仍然存在的不可变 commit，解除 wheel 构建阻塞。

## 规则合规说明

本次修改仍然严格位于最新 toolkit 页面推荐的 `路径一：量化加速` 范围内：

- 不替换官方提供的 MiniCPM-SALA 基座模型
- 不修改评测并发规则，也不重新启用被禁止的 prefix cache
- 继续遵循最新 `提交说明` 中的 `prepare_env.sh` 自定义环境构建方式
- 使用固定 commit 而不是浮动分支，提升提交物可复现性

## 详细实现计划

修改前计划：

1. 再次确认最新官方 toolkit / competition 页面仍允许通过 `prepare_env.sh` 自定义环境，并继续推荐路径一的 Marlin 优化方向。
2. 确认 `sgl-kernel/CMakeLists.txt` 中的 DeepGEMM 固定版本就是当前构建失败的直接原因。
3. 将失效的 DeepGEMM commit 替换为 `sgl-project/DeepGEMM` 中仍可访问的不可变 commit，优先选择 fork 上的 `sgl-release` 分支头，而不是直接跟随 `main`。
4. 在 fcloud 上重新执行无 Docker wheel 构建。

## 实际代码变更

修改文件：

- `sgl-kernel/CMakeLists.txt`

具体变更：

- 将 `repo-deepgemm` 的 `GIT_TAG` 从已失效的 `54f99a8af537b3c6eb4819b69907ccbe2b600792`
- 更新为新的固定 commit：`ffe2b6b97420a9f8c58268ca55755168e6e2f360`

原因：

- 旧 pin 在上游已经不存在，会直接阻塞所有本地源码构建
- 新 pin 是 fork 的 `sgl-release` 对应的具体 commit，更适合提交物复现
- 该变更只影响第三方源码拉取配置，不修改服务运行逻辑，也不改变 SM120 Marlin 内核实现本身

## 验证命令

验证依赖 pin 是否存在：

```bash
cd /root/sglang-minicpm/sgl-kernel
git ls-remote https://github.com/sgl-project/DeepGEMM | grep ffe2b6b97420a9f8c58268ca55755168e6e2f360
```

无 Docker 构建 wheel：

```bash
cd /root/sglang-minicpm/sgl-kernel
python -m pip install -U uv scikit-build-core ninja
make build MAX_JOBS=2 CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1"
```

将构建出的 wheel 拷入 demo 提交目录后，检查提交脚本语法：

```bash
cd /root/sglang-minicpm
bash -n benchmark/soar/demo_sala/prepare_env.sh
```

wheel 安装后执行正确性与速度验证：

```bash
python3 eval_model.py --api_base http://127.0.0.1:30000 --model_path <MODEL_DIR> --data_path <DATA_DIR>/perf_public_set.jsonl --concurrency 32
```

```bash
bash SOAR/bench_serving.sh http://127.0.0.1:30000
```

验证 SM120 Marlin 路径是否生效：

```bash
grep -F "[sgl-kernel] SM120 Marlin auto-config enabled:" <server_log_file>
```

## 结果摘要表

| 项目 | 修改前 | 修改后 |
| --- | --- | --- |
| DeepGEMM 源码拉取 | 在失效 commit 处失败 | 改为有效固定 commit |
| 无 Docker 的 `sgl-kernel` wheel 构建 | 在 CMake Populate 阶段被阻塞 | 至少解除依赖拉取阶段阻塞 |
| 服务配置 | `CHANGE_0041` 基线 + `CHANGE_0043` wheel 工作流 | 不变 |
| SM120 Marlin 运行时逻辑 | 已存在 | 不变 |

## 回滚说明

如果新的 DeepGEMM pin 引入兼容性问题：

1. 回退 `sgl-kernel/CMakeLists.txt` 中的 `repo-deepgemm` `GIT_TAG`
2. 删除失败构建产生的 `sgl-kernel/build` 目录
3. 在确认上游 ref 真实存在后，再尝试替换为另一个不可变 commit

回滚命令：

```bash
git revert <commit>
```

## 后续建议

1. 先在 fcloud 用上面的低并发命令构建补丁 `sgl-kernel` wheel。
2. 将生成的 wheel 拷贝到 `benchmark/soar/demo_sala/`，先跑正确性。
3. 如果正确性稳定，再跑 `S1`、`S8`、`Smax`，并只与 `CHANGE_0041` 基线比较。