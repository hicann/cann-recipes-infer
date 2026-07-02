# USAGE — DeepSeek-V4-Flash 单卡 Ascend NPU + K920 CPU MoE

从**一台干净 container** 开始，到从开源仓库拉代码、设置 third_party、打补丁、编译、转权重、拉起服务、
连贯性验收的完整步骤。方案原理见 [dsv4_flash_single_card_design.md](dsv4_flash_single_card_design.md)。

> 交付物边界：**patch 只含三仓的代码改动**；转权重 / 拉起 / 校验脚本是**独立文件**（本目录 `scripts/`），
> 不在 patch 内。

---

## 0. 三仓 pristine 基线（必须钉死这三个 SHA）

| 仓 | 公开来源 | pristine SHA | patch 目录 |
|---|---|---|---|
| **ktransformers-AK**（父仓） | `github.com/kvcache-ai/ktransformers`（`0.6.2.post1`） | **`d7b5b49`** | `main_repo/` |
| **sglang** | `github.com/iforgetmyname/sglang`（`dsv4_release`，Ascend DSv4 基线） | **`298193eb3`** | `sglang/` |
| **llama.cpp** | `github.com/ggerganov/llama.cpp`（tag `b3173`） | **`a94e6ff`** | `llama_cpp/` |

> ⚠️ sglang 这部分**目前不是正式版本**（以 patch 形式打在 DSv4 公开基线上）；待 sglang 主干正式支持该路径后会改为基于主干。

`third_party/pybind11`、`third_party/custom_flashinfer` 用父仓自带的上游子模块，**无额外 patch**。

---

## 1. 准备：镜像、权重、容器、系统依赖

> 路径约定：除「本工程内」用相对路径（`$REPO`）外，权重/缓存目录都用环境变量，按你的环境配置即可。
> 下文用到三个：`$W8A8_DIR`（NPU 侧权重）、`$MXFP4_SRC`（CPU 转换源权重）、`$GGUF_CACHE`（MXFP4 GGUF 输出，需 ≥150 GiB 可写空间）。

### 1.1 拉镜像
```bash
docker pull lmsysorg/sglang:deepseek-v4-npu-910b     # Atlas 910B
docker pull lmsysorg/sglang:deepseek-v4-npu-a3       # A3
```

### 1.2 下载两份权重（缺一不可）
```text
NPU 侧 W8A8（int8）   : https://modelscope.cn/models/sgl-npu/DeepSeek-V4-Flash-W8A8
CPU 转换源 原生 MXFP4 : https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
```
下到宿主机同一权重根目录（下一步会整体挂进容器的 `/workspace/models`）。

### 1.3 起容器（参考脚本 `scripts/launch_dsv4_singleCard_cann8.5.0_910b.sh`）
```bash
WORKSPACE=<宿主机代码目录> MODEL_DIR=<宿主机权重根目录> \
  bash scripts/launch_dsv4_singleCard_cann8.5.0_910b.sh
# 自动挂 NPU 驱动/设备 + 代码(/workspace/code) + 权重(/workspace/models) + 映射服务端口。
# 可配置：IMAGE / NAME / SERVICE_PORT(默认 8020) / SHM_SIZE / NPU_VISIBLE_DEVICES(auto 或 "0,3")。
```

### 1.4 容器内系统依赖
```bash
apt-get update && apt-get install -y git build-essential cmake libhwloc-dev libhwloc15
# libhwloc 是 kt-kernel 的硬依赖（运行期 import + cmake 编译都需），容器重启后通常要重装。
# 下文命令直接用 python（容器内通常即 python3.11，本方案验证版本）；torch/torch_npu/CANN 由镜像提供。
```

### 1.5 设置路径变量（按你挂载后的实际位置）
```bash
export W8A8_DIR=/workspace/models/DeepSeek-V4-Flash-W8A8   # 1.2 下载的 W8A8 目录（容器内路径）
export MXFP4_SRC=/workspace/models/DeepSeek-V4-Flash       # 1.2 下载的原生 MXFP4 目录
export GGUF_CACHE=/workspace/models/cache                  # MXFP4 GGUF 输出目录（任意可写、≥150 GiB）
```

---

## 2. 从开源仓库拉代码 + 设置 third_party 指针

> 关键：pristine `d7b5b49` 的 `.gitmodules` 把 sglang 指向 `kvcache-ai/sglang@main`、llama.cpp 指向
> `ggerganov/llama.cpp`，**和本方案需要的 DSv4 基线不一致**。所以 sglang / llama.cpp 这两个 `third_party`
> 子目录**不能直接 `git submodule update`**，要手动 clone 到指定 repo@SHA（即“改 third_party 指针”）。

> 下面统一用 `checkout -b <本地分支> <SHA>` **建一个本地分支**再切过去，而不是 `checkout <SHA>`——
> 后者会进入 detached HEAD（无头）状态，apply patch 后想 commit/出新 patch 都不方便。分支名随意。

```bash
# (1) 父仓：clone 后建分支钉到 pristine 基线
git clone https://github.com/kvcache-ai/ktransformers.git ktransformers-AK
cd ktransformers-AK
git checkout -b dsv4-npu-release d7b5b49      # 建本地分支，避免 detached HEAD
REPO=$(pwd)                                    # 记下父仓根目录，后文都用 $REPO

# (2) 上游子模块（按原样用，无 patch）
git submodule update --init third_party/pybind11 third_party/custom_flashinfer

# (3) sglang：把 third_party/sglang 换成 DSv4 公开基线 @ 298193eb3
rm -rf third_party/sglang
git clone https://github.com/iforgetmyname/sglang.git third_party/sglang
git -C third_party/sglang checkout -b dsv4-release-base 298193eb3

# (4) llama.cpp：把 third_party/llama.cpp 换成 tag b3173 (a94e6ff)，头文件在根目录
rm -rf third_party/llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git third_party/llama.cpp
git -C third_party/llama.cpp checkout -b b3173-base a94e6ff
```

> 这样 `third_party/sglang`、`third_party/llama.cpp` 是独立 git 仓库（不再跟 pristine 的子模块指针走），
> 各自在自己的本地分支上。父仓会把它们显示为 “modified submodule”，**不要紧**——我们走 patch + 本地构建，
> 不提交父仓。建分支后，apply 完 patch 可以直接 `git commit` 固化，也方便后续再出 patch（见 §10）。
> 校验三仓 SHA：
> ```bash
> git -C $REPO rev-parse --short HEAD                       # d7b5b49
> git -C $REPO/third_party/sglang rev-parse --short HEAD    # 298193eb3
> git -C $REPO/third_party/llama.cpp rev-parse --short HEAD # a94e6ff
> ```

---

## 3. 打补丁（仅代码）

补丁清单（只动三仓源码）：
```
main_repo/   (apply 到 ktransformers-AK @ d7b5b49) —— 只动 kt-kernel 引擎代码
  0001-kt-kernel-ascend-npu-backend.patch     NPU 后端 + ACL callback worker + 构建系统
  0002-kt-kernel-cpu-moe-mxfp4-kernel.patch    CPU MoE 原生 MXFP4 GEMV kernel + GGUF loader
sglang/      (apply 到 sglang @ 298193eb3)
  0001-sglang-npu-kv-triton-fallback.patch     triton×ascend KV/MoE 自动回退 torch 等价
  0002-sglang-kt-ep-cpu-moe-offload.patch      KT EP wrapper（CPU MoE offload）+ 热专家 mask
  0003-sglang-packaging.patch                  Ascend/NPU 打包配置
llama_cpp/   (apply 到 llama.cpp @ a94e6ff)
  0001-fix-gguf-NumPy-2-GGUFReader.patch        gguf-py NumPy 2.0 兼容
  0002-add-ggml-type-mxfp4.patch                注册 GGML_TYPE_MXFP4=39 + NEON kernel（CPU MXFP4 硬依赖）
```

一键（`<release_dir>` = 本交付目录的绝对路径）：
```bash
bash <release_dir>/apply_all.sh $REPO
```
`apply_all.sh` 逐仓 `git apply --check` 后再 `apply`，任一 `--check` 失败即中止（多半是 SHA 没钉对）。

手工等价：
```bash
cd $REPO                        && for p in <release_dir>/main_repo/*.patch;  do git apply "$p"; done
cd $REPO/third_party/sglang     && for p in <release_dir>/sglang/*.patch;     do git apply "$p"; done
cd $REPO/third_party/llama.cpp  && for p in <release_dir>/llama_cpp/*.patch;  do git apply -p1 "$p"; done
```

把独立脚本放回父仓 `tools/`（**这些不在 patch 内，单独分发**）：
```bash
cp -r <release_dir>/scripts/tools/* $REPO/tools/
```

---

## 4. 编译 kt-kernel（带 Ascend NPU 后端）

> 下文命令里的 `python` = 目标 Python（本方案验证于 `python3.11`；容器内 `python` 通常已是它，
> 否则把命令里的 `python` 换成你的 `python3.11`）。

```bash
cd $REPO/kt-kernel
CPUINFER_USE_ASCEND_NPU=1 python setup.py build_ext --inplace
```
- 配置期日志应出现 `LLAMA_ARM_DOTPROD=ON` 且 `SVE=OFF / BF16=OFF / I8MM=OFF`（红线 R1）。
- `Found Ascend CL library … libascendcl.so` = NPU 后端就绪。
- ggml `GGML_TYPE_MXFP4 not handled in switch` 警告**良性**（非 MoE 路径 op 不需要 mxfp4 分支）。

体检：
```bash
find $REPO/kt-kernel -name "kt_kernel_ext*.so"
python -c "import ctypes,glob; ctypes.CDLL(glob.glob('$REPO/kt-kernel/python/kt_kernel_ext*.so')[0]); print('dlopen OK')"
```

### 4b. 让 `import kt_kernel` 可用（关键，别漏）

`build_ext --inplace` 只产出 `.so`，**不注册包名**。setup.py 把包名 `kt_kernel` 映射到 `python/` 目录，
所以直接把 `kt-kernel` 加进 PYTHONPATH 只能 `import python`，不能 `import kt_kernel`。两种方式二选一：

```bash
# 方式 A（推荐，无 site-packages 污染）：建符号链接 kt_kernel -> python，PYTHONPATH 指 kt-kernel 父目录
ln -sfn python $REPO/kt-kernel/kt_kernel
export PYTHONPATH="$REPO/third_party/sglang/python:$REPO/kt-kernel${PYTHONPATH:+:$PYTHONPATH}"
# 项目自带 helper 就是干这个的：source tools/ensure_kt_kernel.sh && ensure_kt_kernel $REPO

# 方式 B：editable 安装（写 site-packages，之后任意 cwd 可 import）
cd $REPO/kt-kernel && python -m pip install -e .
```
验证：`python -c "import kt_kernel; print('import kt_kernel OK')"`
> 拉服务脚本 `launch_ds4flash_npu.sh` 内部已 source 上述 helper，§7 不会遇到此错；
> 但所有**单独跑**的脚本（§6 离线对账、§5 转换若以 `import kt_kernel` 的方式调用等）都要先做 4b。

---

## 5. 原生 MXFP4 → 43 层 GGUF（现行主路径）

```bash
mkdir -p "$GGUF_CACHE"
nohup python tools/batch_convert_mxfp4_layers_mp.py \
  --input "$MXFP4_SRC" --output-dir "$GGUF_CACHE" \
  --layer-start 0 --layer-end 42 --jobs 16 --verify-sample 3 \
  > /tmp/kt_mxfp4_convert.log 2>&1 &
# 产出 dsv4_layer{0..42}_mxfp4.gguf，每层 ~3.42 GiB，合计 ~138 GiB
```
收尾**全集校验**（尺寸 + sha256 + 抽样逐元素 bit-exact）：
```bash
python tools/verify_mxfp4_gguf_set.py --dir "$GGUF_CACHE" --sha256-manifest tools/mxfp4_gguf_sha256.txt
```
> ⚠️ 并发转换曾把某层写截断，**收尾务必跑全集校验**。
> 单层快验：`convert_mxfp4_layer_to_gguf.py --input "$MXFP4_SRC" --layer-idx 16 --output /tmp/l16.gguf`
> → `verify_mxfp4_layer.py --gguf /tmp/l16.gguf --layer-idx 16`（bit-exact）。

> 💾 **转完并通过校验后，原生 MXFP4 源 `$MXFP4_SRC` 可删，回收磁盘**：拉服务（§7）只用
> `$GGUF_CACHE`（CPU 专家）+ `$W8A8_DIR`（NPU 侧），**不再碰 MXFP4 源**。注意：
> - 删前先过校验——单层 bit-exact、§5 全集校验的 `--deep`/L3 深抽、§6 cosine 对账都要读源；
> - 全集 **sha256 + 尺寸**校验是自包含的（只需 GGUF + `mxfp4_gguf_sha256.txt`），删源后仍可复验 GGUF 完整性；
> - **别删 `$W8A8_DIR`**（NPU 侧一直要用），它和 MXFP4 源是两份不同权重。

---

## 6.（推荐）拉服务前离线对账 kernel（cosine ≥ 0.9999）

```bash
# 先确保 import kt_kernel 可用（见 §4b）；helper 会建符号链接并设好 PYTHONPATH
source tools/ensure_kt_kernel.sh && ensure_kt_kernel "$REPO"

python tools/cpu_moe_reference_check_mxfp4.py --model-dir "$MXFP4_SRC" \
  --gguf "$GGUF_CACHE/dsv4_layer16_mxfp4.gguf" --layer-idx 16
# 脚本内置 KT_FORCE_SYNC_SUBMIT=1（孤立单层调用须同步提交，否则 cand 全零）
# 若没用 helper：先 ln -sfn python $REPO/kt-kernel/kt_kernel，再带
#   PYTHONPATH="$REPO/third_party/sglang/python:$REPO/kt-kernel" 运行
```

---

## 7. 拉起服务（MXFP4，graph-on）

（可选）拉起前预检 GGUF 层文件与 `kt_kernel` 是否就位：
```bash
GGUF_DIR="$GGUF_CACHE" GGUF_SUFFIX=_mxfp4 bash tools/e2e_preflight.sh   # 退出码 0=通过
```

先 `npu-smi info` 选空闲卡（避开别容器/别 session）。**长跑服务在自己终端前台拉**
（远程/后台拉的父进程上下文会被回收 → `main process disappeared`）。

```bash
cd $REPO
NPU_DEVICE_ID=<空闲卡> PORT=8020 \
  KT_GGUF_TEMPLATE="$GGUF_CACHE/dsv4_layer{layer_idx}_mxfp4.gguf" \
  KT_CPUINFER=128 MODEL_PATH="$W8A8_DIR" KT_DECODE_TIMING=1 \
  bash tools/launch_ds4flash_npu.sh 2>&1 | tee /tmp/kt_serve.log
```
- `KT_GGUF_TEMPLATE` **必填**（脚本无内置默认，不设会报错退出）：指向你的 GGUF 目录 + `_mxfp4` 模板，
  如上；`{layer_idx}` 是脚本的层号占位符（**不是 shell 变量**，保持字面量；`$GGUF_CACHE` 会正常展开）。
- `MODEL_PATH` 指 **W8A8**（NPU 侧），与 GGUF（CPU 侧）缺一不可。
- graph-on 是默认（**勿**传 `--disable-cuda-graph`）；`KT_CPUINFER` 默认 128（甜点，勿设 192）。

### 环境变量（仅计时类）
| env | 作用 |
|---|---|
| `KT_DECODE_TIMING=1` | 每 token 打印 CPU MoE submit→sync wall（`cpu_moe_wall`）|
| `KT_MOE_PHASE_TIMING=1` | kernel 相位细分（gateup / down / merge / quant µs）|
| `KT_FORCE_SYNC_SUBMIT=1` | eager 兜底（配 `EXTRA_FLAGS="--disable-cuda-graph"`）+ 离线对账依赖 |

> CPU MoE 行内预取 + 优化 GEMV 恒生效；triton×ascend 错配自动回退 torch 等价——均无需任何开关。

---

## 8. 验收（加载约 2–3.5 min 热 cache）

```bash
until curl -sf http://127.0.0.1:8020/health >/dev/null; do sleep 5; done    # 就绪
PORT=8020 bash tools/curl_f2_prompts.sh                                  # 四-prompt 连贯性验收（核心验收）
curl -sS -X POST http://127.0.0.1:8020/generate -H 'Content-Type: application/json' \
  -d '{"text":"中国的首都是","sampling_params":{"max_new_tokens":64,"temperature":0}}'   # 期望"北京…"
```
性能观测：
```bash
grep KT_DECODE_TIMING /tmp/kt_serve.log    # cpu_moe_wall（MXFP4 生产 min ~17ms / median ~22–27ms）
grep "gen throughput" /tmp/kt_serve.log    # decode tok/s
```

**预期吞吐**（单发、短上下文、graph-on、`kt-cpuinfer 128`）：
- 清净独占（NPU 空卡 + CPU 无邻居争抢）：**~16 tok/s**
- 中等争抢（共享机有邻居吃 DRAM 带宽）：~13–14 tok/s
- 低于此区间多半是：**邻居抢带宽 / NPU 卡被占 / `KT_CPUINFER` 设错 / 并发多发 / 上下文太长**。
  原理（带宽 vs 算力 roofline、与理想的 ~2.5–3.5× gap、各下降因素）见 dsv4_flash_single_card_design.md「当前状态」一节。

> ⚠️ **`--max-running-requests 1`，别并发多发**（并发撞争抢窗口会触发 NPU runtime 失稳崩）。
> 收服务：跑服务的终端 `Ctrl-C`（SIGTERM 优雅释放 HBM）；**绝不 `pkill -f sglang.launch_server`**。

---

## 9. 常见坑速查

| 现象 | 根因 | 修复 |
|---|---|---|
| `git submodule update` 后 sglang/llama.cpp 不是预期版本 | pristine `.gitmodules` 指向别的 repo/branch | 按 §2 手动 clone 到 repo@SHA |
| CMake 找不到 hwloc | 系统未装 | `apt-get install -y libhwloc-dev libhwloc15`（每容器）|
| `No module named 'kt_kernel'`（单独跑脚本时） | 只 `build_ext` 没注册包名 | 见 §4b：建 `kt_kernel->python` 符号链接 或 `pip install -e .` |
| `import kt_kernel` 失败 / `not installed`（已建链接后） | `libhwloc.so.15` 缺失 或 `.so` 没编出来 | 装 libhwloc（§1）；确认 `kt-kernel/python/kt_kernel_ext*.so` 存在 |
| 启动崩 quant 类型不匹配 | sglang 没钉对 `298193eb3` | 重新 checkout 基线 SHA |
| 输出乱码 + MXFP4 对账偏 | nibble 序未重排 | 用本仓转换器（逐 32-group 重排）；`verify_mxfp4_layer.py` 把关 |
| `--chunked-prefill-size -1` → malloc 越界 | `max_len=-1` 按 1 分配 | `llamafile.py` 已对 ≤0 回落 2048 |
| 跑一会儿 `main process disappeared` | 远程/后台拉服务被回收 | 在自己终端前台拉 |
| eager 出 token 但乱码 | CPU MoE async submit 未 flush | eager 下加 `KT_FORCE_SYNC_SUBMIT=1` |

---

## 10. 在此基础上再出 patch（你改完代码后如何重新生成）

本交付里的 `gen_main_repo_patches.sh` / `gen_sglang_patches.sh` 就是 patch 的生成器，它们**对 pristine 基线
做 `git diff`**——所以你 apply 完 patch、再继续改代码后，**重跑同一个生成器就能产出更新后的全量 patch**
（覆盖我们的版本：包含「我们的改动 + 你的改动」相对 pristine 的完整 diff）。前提是 §2 建分支的方式让
pristine 基线 SHA 仍在历史里可达。

底层就是一行 `git diff`，按需直接用：

```bash
# 父仓 kt-kernel 全量（相对 pristine d7b5b49）——含工作区未提交改动
git -C $REPO diff d7b5b49 -- kt-kernel/operators/llamafile/moe.hpp \
  kt-kernel/python/experts_base.py kt-kernel/python/utils/llamafile.py \
  kt-kernel/python/utils/loader.py > /tmp/my_kt_kernel.patch

# sglang 全量（相对 pristine 298193eb3）
git -C $REPO/third_party/sglang diff 298193eb3 -- python/sglang/... > /tmp/my_sglang.patch
```

- **全量 vs pristine**（推荐，交付用）：`git diff <pristine_SHA> -- <paths>`，和我们出 patch 完全同法。
- **只要你相对“已 apply 我们 patch 那一刻”的增量**：apply 完先 `git add -A && git commit`（或打 tag）固化成基线 `B`，
  之后 `git diff B -- <paths>` 即只含你后续的改动。
- 生成器脚本里的 `OUT=`/`BASE=`/pathspec 可按你的目录与改动范围改；它不是黑盒，就是 `git diff` 的封装。
