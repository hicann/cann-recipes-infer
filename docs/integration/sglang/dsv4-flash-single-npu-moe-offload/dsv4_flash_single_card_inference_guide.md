# DeepSeek-V4-Flash 单卡推理部署指南（Ascend NPU 910 + Kunpeng CPU MoE）

正文（§0–§8）是一条按顺序执行的部署路径：环境 → 取码打补丁 → 构建 → 转权重 → 启动 → 验收。
环境细节、手工等价操作、调优与排障都在附录，正常部署不需要看。方案原理见
[dsv4_flash_single_card_design.md](dsv4_flash_single_card_design.md)。

补丁仅包含三个上游仓库的代码改动；转换、启动、校验脚本以独立文件分发（`scripts/`），不在补丁内。

| 附录 | 内容 |
|---|---|
| [A](#附录-a-a3-环境准备细节) | A3 环境前提、第三方算子仓库、分阶段执行 |
| [B](#附录-b-补丁清单与手工操作) | 补丁清单与逐仓手工 apply |
| [C](#附录-c-构建的预期输出与手工编译) | kt-kernel 构建的预期日志、AscendC 算子手工编译 |
| [D](#附录-d-数值对账可选) | 数值对账（可选） |
| [E](#附录-e-cpu-线程池调优) | CPU 线程池调优（多 NUMA 主机） |
| [F](#附录-f-计时与排障开关) | 计时与排障开关 |
| [G](#附录-g-常见问题) | 常见问题、长 prefill 的 HBM 预算 |
| [H](#附录-h-重新生成补丁) | 重新生成补丁 |

---

## 0. 上游基线

三个仓库必须固定在下列 SHA：

| 仓库 | 来源 | SHA | 补丁目录 |
|---|---|---|---|
| ktransformers | `github.com/kvcache-ai/ktransformers`（`0.6.2.post1`） | `d7b5b49a3` | `main_repo/` |
| sglang | `github.com/iforgetmyname/sglang`（`dsv4_release`） | `298193eb3` | `sglang/` |
| llama.cpp | `github.com/ggerganov/llama.cpp`（tag `b3173`） | `a94e6ff87` | `llama_cpp/` |

`third_party/pybind11`、`third_party/custom_flashinfer` 使用父仓自带子模块，无需打补丁。

> sglang 部分为过渡形态：当前以补丁形式打在 DSv4 公开基线上。待 sglang 主干支持该路径后改为基于主干。

---

## 1. 环境准备

在仓库根目录执行：

```bash
cd integration/sglang/dsv4-flash-single-npu-moe-offload
export RELEASE_DIR="$(pwd)"
```

按硬件选择一条路径；§2 起的流程两者共通。

| 硬件 | 环境准备 | CANN | 自定义算子 | 参见 |
|---|---|---|---|---|
| **A3** | CANNLab 的 lite-and-infer 镜像 + 源码构建 | 9.0.0–9.2.0 | 从源码构建 | **§1A** |
| Atlas 910B + K920 裸机 | 已集成依赖的镜像 | 8.5.0（镜像自带） | 镜像自带 | §1B |

两套环境均已验证可运行，功能一致；本文的性能数据测自 A3。

### 1.1 下载权重

两份权重都必需：

| 权重 | 来源 |
|---|---|
| NPU 侧 W8A8（int8） | https://modelscope.cn/models/sgl-npu/DeepSeek-V4-Flash-W8A8 |
| CPU 转换源（原生 MXFP4） | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash |

> **必须使用官方发布的 W8A8。** CPU 侧 GGUF 与 NPU 侧 W8A8 需为同一量化基底（quarot 旋转）。
> 若自行用 modelslim 重新量化 W8A8，而 GGUF 来自官方原生 MXFP4，两者基底不一致会导致输出乱码且不报错。

设置路径变量：

```bash
export W8A8_DIR=/workspace/models/DeepSeek-V4-Flash-W8A8   # NPU 侧权重 （根据需求自定义路径）
export MXFP4_SRC=/workspace/models/DeepSeek-V4-Flash       # CPU 转换源（根据需求自定义路径）
export GGUF_CACHE=/workspace/models/cache                  # GGUF 输出目录，需 ≥150 GiB（根据需求自定义路径）
```

### 1A. A3 —— CANNLab 的 lite-and-infer 镜像（CANN 9.0.0–9.2.0）

支持 Python 3.11 / 3.12，可直接使用系统 Python。
脚本支持自动选择 CANN 和 Python，也可显式指定安装路径、解释器和编译器：

```bash
export CANN_HOME=$HOME/Ascend/cann-9.0.0   # 按实际安装路径修改
export PYTHON_BIN="$(command -v python3.11 || command -v python3.12)"
export CC_BIN=/usr/bin/gcc-13 CXX_BIN=/usr/bin/g++-13

bash "$RELEASE_DIR/scripts/tools/setup_dsv4_env_from_clean_cann.sh" all
```

`all` 安装依赖并执行 `verify`；随后按 §2–§4 获取代码、打补丁并构建 kt-kernel。

环境前提、所装的第三方算子仓库、以及分阶段执行（排障用）见[附录 A](#附录-a-a3-环境准备细节)。

### 1B. Atlas 910B —— 已集成依赖的镜像（CANN 8.5.0）

```bash
docker pull lmsysorg/sglang:deepseek-v4-npu-910b

WORKSPACE=<宿主机代码目录> MODEL_DIR=<宿主机权重目录> \
  bash scripts/launch_dsv4_singleCard_cann8.5.0_910b.sh
```

脚本挂载 NPU 驱动与设备、代码（`/workspace/code`）、权重（`/workspace/models`），并映射服务端口。
可配置 `IMAGE` / `NAME` / `SERVICE_PORT`（默认 8020）/ `SHM_SIZE` / `NPU_VISIBLE_DEVICES`。

容器内设置交付目录并安装系统依赖：

```bash
export RELEASE_DIR=/workspace/code/integration/sglang/dsv4-flash-single-npu-moe-offload
apt-get update && apt-get install -y git build-essential cmake libhwloc-dev libhwloc15
```

libhwloc 为 kt-kernel 的运行期与编译期依赖，容器重启后需重新安装。torch、torch_npu、CANN 及自定义算子由镜像提供。

环境就绪，转 §2。

---

## 2. 获取代码并设置 third_party

pristine `d7b5b49a3` 的 `.gitmodules` 指向的 sglang / llama.cpp 与本方案所需基线不一致，
因此这两个子目录需手动 clone 到指定 SHA，不能用 `git submodule update`。

```bash
# 选一个工作区放三仓（示例，按需改；有写权限即可）
export WORKSPACE=$HOME/dsv4-workspace
mkdir -p "$WORKSPACE" && cd "$WORKSPACE"

# 父仓
git clone https://github.com/kvcache-ai/ktransformers.git ktransformers-AK
cd ktransformers-AK
git checkout -b dsv4-npu-release d7b5b49a3
export REPO=$(pwd)   # export：§3 apply_all.sh、§4 setup kt_kernel 同一 shell 直接用

# 上游子模块（无需补丁）
git submodule update --init third_party/pybind11 third_party/custom_flashinfer

# sglang
rm -rf third_party/sglang
git clone https://github.com/iforgetmyname/sglang.git third_party/sglang
git -C third_party/sglang checkout -b dsv4-release-base 298193eb3

# llama.cpp
rm -rf third_party/llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git third_party/llama.cpp
git -C third_party/llama.cpp checkout -b b3173-base a94e6ff87
```

`third_party/sglang`、`third_party/llama.cpp` 此时为独立仓库，父仓会将其显示为 modified submodule，
不影响后续构建。校验：

```bash
git -C $REPO rev-parse --short=9 HEAD                       # d7b5b49a3
git -C $REPO/third_party/sglang rev-parse --short=9 HEAD    # 298193eb3
git -C $REPO/third_party/llama.cpp rev-parse --short=9 HEAD # a94e6ff87
```

---

## 3. 打补丁

```bash
bash "$RELEASE_DIR/apply_all.sh" "$REPO"
```

`apply_all.sh` 逐仓先 `git apply --check` 再 apply，任一检查失败即中止（通常是基线 SHA 不匹配）。

补丁清单与等价的手动操作见[附录 B](#附录-b-补丁清单与手工操作)。

安装独立脚本：

```bash
mkdir -p "$REPO/tools"
cp -r "$RELEASE_DIR/scripts/tools/." "$REPO/tools/"
```

---

## 4. 构建 kt-kernel

加载 CANN 和 Python 环境：

```bash
source "$RELEASE_DIR/scripts/tools/setup_dsv4_env_from_clean_cann.sh" env
```

A3 环境执行 `kt_kernel` 阶段：

```bash
REPO="$REPO" bash "$RELEASE_DIR/scripts/tools/setup_dsv4_env_from_clean_cann.sh" kt_kernel
```

等价的手工编译命令：

```bash
cd $REPO/kt-kernel
CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 CPUINFER_USE_ASCEND_NPU=1 \
  CPUINFER_ARM_SVE=OFF CPUINFER_ARM_BF16=OFF CPUINFER_ARM_I8MM=OFF \
  "$PYTHON_BIN" setup.py build_ext --inplace --force
```

校验：

```bash
find $REPO/kt-kernel -name "kt_kernel_ext*.so"
"$PYTHON_BIN" -c "import ctypes,glob; ctypes.CDLL(glob.glob('$REPO/kt-kernel/python/kt_kernel_ext*.so')[0]); print('dlopen OK')"
```

### 4.1 使 `import kt_kernel` 生效

注册包路径：

```bash
# 符号链接（不写 site-packages）
ln -sfn python $REPO/kt-kernel/kt_kernel
export PYTHONPATH="$REPO/third_party/sglang/python:$REPO/kt-kernel${PYTHONPATH:+:$PYTHONPATH}"
# 等价 helper：source tools/ensure_kt_kernel.sh && ensure_kt_kernel $REPO
```

验证：`"$PYTHON_BIN" -c "import kt_kernel"`。

启动脚本已内置该 helper；单独运行其他脚本（§5、[附录 D](#附录-d-数值对账可选)）前需先完成本步。

---

## 5. 转换 MXFP4 GGUF（43 层）

```bash
mkdir -p "$GGUF_CACHE"
nohup "$PYTHON_BIN" tools/batch_convert_mxfp4_layers_mp.py \
  --input "$MXFP4_SRC" --output-dir "$GGUF_CACHE" \
  --layer-start 0 --layer-end 42 --jobs 16 --verify-sample 3 \
  > /tmp/kt_mxfp4_convert.log 2>&1 &
```

产出 `dsv4_layer{0..42}_mxfp4.gguf`，每层约 3.42 GiB，合计约 138 GiB。

转换完成后执行全集校验（文件齐全 + 尺寸 + sha256 + 抽样 bit-exact）：

```bash
"$PYTHON_BIN" tools/verify_mxfp4_gguf_set.py --dir "$GGUF_CACHE" --sha256-manifest tools/mxfp4_gguf_sha256.txt
```

> 并发转换可能产生截断文件，必须执行全集校验。

单层快速校验：

```bash
"$PYTHON_BIN" tools/convert_mxfp4_layer_to_gguf.py --input "$MXFP4_SRC" --layer-idx 16 --output /tmp/l16.gguf
"$PYTHON_BIN" tools/verify_mxfp4_layer.py --gguf /tmp/l16.gguf --model-dir "$MXFP4_SRC" --layer-idx 16
```

校验通过后，`$MXFP4_SRC` 可删除以回收磁盘：启动服务只使用 `$GGUF_CACHE` 与 `$W8A8_DIR`。
删除前需完成上述校验（单层 bit-exact、全集深度抽查、[附录 D](#附录-d-数值对账可选) 数值对账均需读取该目录）；
sha256 与尺寸校验不依赖源目录，删除后仍可复验 GGUF 完整性。`$W8A8_DIR` 不可删除。

---

## 6. AscendC MXFP4 算子

`KT_MXFP4_DEPOOL` 默认为 1（depool：不常驻整份 W8A8 池），该路径依赖一个 AscendC device kernel
（MXFP4 dequant + ND→NZ）。未准备该算子时，首次长 prefill 会因找不到 `libmxfp4fused.so` 而失败。

源码位于 `scripts/tools/ascendc_mxfp4/`（`mxfp4_fused_kernel.cpp` 与 host 封装 `mxfp4_fused_op.py`）。
`.so` 不入库，首次运行需现场编译。

首次调用时自动用 bisheng 编译并缓存；源码较 `.so` 新时自动重编。
运行时从 **`$REPO/tools/ascendc_mxfp4/`** 加载，即 §3「安装独立脚本」那步
`cp -r "$RELEASE_DIR/scripts/tools/." "$REPO/tools/"` 的目标位置——**更新算子源码后需重新拷贝一次**，
否则服务用的仍是旧副本。要指向别处（例如直接用交付目录）时设 `KT_MXFP4_OP_DIR`。

想在部署前先验证编译链，可按[附录 C](#附录-c-构建的预期输出与手工编译)手工编一次。

设置 `KT_MXFP4_DEPOOL=0` 可不使用该算子，代价是常驻整份 W8A8 池，内存占用显著增加。

---

## 7. 启动服务

启动前预检（可选）：

```bash
GGUF_DIR="$GGUF_CACHE" GGUF_SUFFIX=_mxfp4 bash tools/e2e_preflight.sh   # 退出码 0 表示通过
```

预检与启动脚本共用同一套解释器解析逻辑（`tools/python_env.sh`）和同一份模块清单，因此两者
必然解析到同一解释器；`PYTHON_BIN` 对两者同时生效。

先用 `npu-smi info` 选择空闲卡。服务需在前台运行（后台启动的父进程上下文可能被回收，
表现为 `main process disappeared`）。

```bash
cd $REPO
NPU_DEVICE_ID=<空闲卡> PORT=8020 \
  KT_GGUF_TEMPLATE="$GGUF_CACHE/dsv4_layer{layer_idx}_mxfp4.gguf" \
  MODEL_PATH="$W8A8_DIR" \
  CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-32768}" \
  bash tools/launch_ds4flash_npu.sh 2>&1 | tee /tmp/kt_serve.log
```

- ⚠️ **`CHUNKED_PREFILL_SIZE` 当前必须不小于最长 prompt 的 token 数**：NSA compressor 当前不支持跨 chunk 的
  prefill，prompt 超过该值时会在跨 chunk 处失败。脚本默认 8192，上例在未设置该变量时使用 32768，以覆盖 §8.1.1 吞吐测试
  的最长档（32k）；若要处理更长的 prompt，需同步调高（`context-length` 上限 65536，prompt + 输出合计）。
  该值须为 page-size(128) 的整数倍。调高不是免费的：单块 prefill 的激活占用随之增长（64k 单块峰值
  约 20 GB），因此脚本默认值保持在 8192，按实际 prompt 长度按需调高即可。
- `MEM_FRACTION` 默认 `0.81`，为长 prompt 的 prefill 留出激活空间；取值与下限见 [附录 G.1](#g1-长-prefill-的-hbm-预算)。
- `KT_GGUF_TEMPLATE`、`MODEL_PATH` 均为必填，未设置时脚本报错退出。
  `{layer_idx}` 为脚本的层号占位符，需保持字面量。
- `PYTHON_BIN` 指定解释器，须与 §4 构建 kt-kernel 的一致；未设置时按
  `python3` → `python3.11` → `python3.12` → 已知安装路径探测，取第一个可 import
  `numpy`/`torch`/`torch_npu`/`sglang` 的（`tools/python_env.sh`，预检脚本共用）。
- graph 模式为默认，不要传 `--disable-cuda-graph`。

### 7.1 功能开关

默认不开启 MTP；需要时按 §7.2 配置。

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `CHUNKED_PREFILL_SIZE` | 8192 | prefill 分块大小，须为 128 的整数倍。**当前必须 ≥ 最长 prompt 的 token 数**，见上 |
| `KT_MXFP4_DEPOOL` | 1 | depool，依赖 §6 的 AscendC 算子 |
| `KT_MXFP4_GGUF_DEDUP` | 1 | 复用 CPU 已 mmap 的 GGUF，减少一份常驻内存 |
| `KT_DYNAMIC_RESIDENT` | 1 | 动态热专家常驻 |
| `KT_PREFILL_STREAM` | 1 | 长 prefill 流式加载专家到 NPU |
| `KT_PREFILL_STREAM_THRESHOLD` | 512 | 触发流式的 token 数阈值 |
| `KT_SIDE_STREAM` / `KT_SHARED_EXPERTS_STREAM` | 1 | CPU MoE 与 NPU 计算重叠 |
| `SGLANG_ENABLE_SPEC_V2` | False | 选择 SGLang 的 V2 投机执行路径；本身不启用 MTP，还需 §7.2 的 `--speculative-*` 参数 |
| `KT_NSA_COMPRESSOR_MODE` | 由 CANN 版本派生 | `single`（CANN 9.0.0+，公开 18 参算子）/ `split`（CANN 8.5.0，私有 19 参算子）。启动脚本按已安装的 CANN 版本自动设置；显式设置则以显式值为准。设错会导致 NSA compressor 调用失败 |

### 7.2 可选：MTP 投机解码

MTP 使用同一份 W8A8 检查点中的 `mtp.0.*` 权重（`num_nextn_predict_layers=1`），无需另下模型。
target 的 43 层继续使用 KT 混合计算，MTP 层在 NPU 上计算。按以下方式启动：
新增路径已在 A3、CANN 9.1、Torch/Torch-NPU 2.8、单请求图模式下验证。

```bash
export MODEL_PATH="$W8A8_DIR"
export CHUNKED_PREFILL_SIZE=8192
export SGLANG_ENABLE_SPEC_V2=True
export EXTRA_FLAGS="--context-length 16384 \
--max-prefill-tokens 16383 --max-total-tokens 32768 \
--max-running-requests 1 --cuda-graph-bs 1 \
--speculative-algorithm EAGLE \
--speculative-draft-model-path ${MODEL_PATH} \
--speculative-num-steps 3 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4"

export KT_NUM_GPU_EXPERTS=16
# 随后执行 §7 的启动命令
```

MTP 增加 NPU 占用，上例使用已验证的 16 常驻配置；同配置改为 32 常驻时，MTP 权重 NZ 布局转换的临时分配触发 OOM。
上例一次验证 4 行；较短配置将 steps / draft-tokens 改为 1 / 2，一次验证 2 行。
验证行数不再加 1；专家 **K=6** 与投机分支宽度 `eagle-topk=1` 不同。
移除五个 `--speculative-*` 参数并取消 `SGLANG_ENABLE_SPEC_V2` 后，重启即可关闭 MTP。

## 8. 验收

模型加载约需 2–3.5 分钟（page cache 命中时）。等服务就绪后，通过对话接口使用 DeepSeek-V4 对话模板验证：

```bash
until curl -sf http://127.0.0.1:8020/health >/dev/null; do sleep 5; done

curl -sS -X POST http://127.0.0.1:8020/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "中国的首都是什么？请用一句话回答。"}],
    "chat_template_kwargs": {"thinking": true},
    "temperature": 0,
    "max_tokens": 64
  }'
```

响应的 `choices[0].message.content` 应是连贯的中文（如“中国的首都是北京。”），正常结束时 `choices[0].finish_reason` 为 `stop`。
若出现乱码、全为重复字符或 NaN，说明链路有问题——常见原因见 [附录 G](#附录-g-常见问题)。

性能观测：

```bash
grep "gen throughput" /tmp/kt_serve.log    # decode 吞吐
```

需要观察 CPU MoE 每 token 耗时时，在启动命令中加 `KT_DECODE_TIMING=1`（默认关闭，见 [附录 F](#附录-f-计时与排障开关)），
之后即可从日志中读取：

```bash
grep KT_DECODE_TIMING /tmp/kt_serve.log    # cpu_moe_wall，A3 生产配置、预热后约 18.3 ms/token
```

### 8.1 性能复现

#### 8.1.1 无 MTP 默认配置

```bash
PY="$PYTHON_BIN" PORT=8020 bash tools/decode_throughput_test.sh
# 默认档位 130/1k/8k/16k/32k：每档先预热、再循环多次，报稳态 decode（均值/中位/min/max）与 prefill
# 可配置：TARGET_TOKENS_LIST / MAX_NEW / REPEAT / WARMUP / PORT
```

**前置条件**：服务须以 `CHUNKED_PREFILL_SIZE=32768` 启动（见 §7）。默认的 8192 小于 16k / 32k 两档的
prompt 长度，这两档会在跨 chunk 处失败。只跑短档时可相应降低，但仍须不小于该档的 prompt token 数。

**必须预热后再测**：动态热专家常驻需要若干次请求才收敛，冷启动的前几次请求结果明显偏低（脚本已内置预热）。

**实测参考（A3，graph 模式，单请求）**

配置：`KT_CPUINFER=32`、`KT_THREADPOOL_COUNT=1`、`KT_NUM_GPU_EXPERTS=32`，depool + 流式 prefill +
动态常驻全开——以上均为默认值；另需显式设 `CHUNKED_PREFILL_SIZE=32768`（默认 8192，见上）。

decode（inter-token 中位）：

| prompt | decode | 开启 `KT_HOT_TAIL_TOKENS=64` |
|---|---|---|
| 118 tok | 21.6 tok/s | 20.7 |
| 801 tok | 22.6 tok/s | 22.2 |
| 7.8k | 19.8 tok/s | 20.4 |
| 15.6k | 19.3 tok/s | 20.8 |
| 31.5k | 19.2 tok/s | 20.5 |

区间约 **19–22.5 tok/s**（中位），单 token 快端可达 23–25，随上下文变长小幅下降（KV 增大）。
`KT_HOT_TAIL_TOKENS=64` 对长上下文有增益（+3%~8%）、对短 prompt 略有损失，因此默认关闭。

prefill（预热后）：130/1k/8k/16k/32k = 15.4 / 15.6 / 16.5 / 17.6 / 20.3 s。
流式 prefill 有约 15 s 的固定开销（每次请求都需将全套专家权重从 DDR 搬到 HBM），几乎不随 prompt 长度增长。
页缓存未热时 32k 可达约 63 s，预热后回到约 21 s。

CPU MoE wall/token 约 18.3 ms（A3 生产配置、resident 命中率 H≈26%、长 prompt、预热后）。

低于上述区间的常见原因：未预热、DRAM 带宽被争抢、NPU 卡被占用、线程池配置与主机 NUMA 拓扑不匹配、并发请求。
原理分析见 [dsv4_flash_single_card_design.md](dsv4_flash_single_card_design.md)。

> 使用 `--max-running-requests 1`，不要并发发送请求。
> 停止服务用 `Ctrl-C`（SIGTERM 会正常释放 HBM），不要使用 `pkill -f sglang.launch_server`。

#### 8.1.2 MTP 性能参考

A3、CANN 9.1、图模式、TP=1；CPU 128 线程 / 8 池（参考环境 NUMA 映射 `0,0,0,1,1,2,3,3`），每层 16 个 NPU 常驻专家，专家 **K=6**。
相同 128 条 GSM8K 输入，并发 1，预热 8 条，固定输出 256 token，`temperature=0`、`ignore_eos=true`。
输入 51–166 token（平均 84.867）；上下文 16384、`max-total-tokens=32768`、
`chunked-prefill-size=8192`，随机种子 215102194。按 §7.2 切换投机参数，其余条件保持一致。

| MTP 配置 | 平均 TPOT，ms | 平均 TTFT，ms | 全程输出吞吐，tok/s | 平均接受长度 |
|---|---:|---:|---:|---:|
| 关闭 | 41.113 | 736.350 | 22.802 | — |
| B：1 step / 2 行 | 22.687 | 740.859 | 39.226 | 1.774 |
| D：3 step / 4 行 | 34.544 | 773.056 | 26.705 | 2.221 |

TPOT 为每请求首末文本增长间隔 /（输出 token 数 − 1）的均值；TTFT 为请求至首次文本增长时间的均值。
全程吞吐包含 prefill；接受长度为每请求 `completion_tokens / spec_verify_ct` 的均值。

数据来自 2026-10-27 的参考实现（KT `e2cc4dc`、llama.cpp `5a8ebd6`、SGLang `d5c201d`），
已包含 CPU 优化，未启用专家裁剪。当前补丁另在 NUMA `0..7`、相同 16 条输入的 MTP 4 行配对中完成图重放及端到端验证：同分组阈值 2 下，内核优化前后 TPOT 为 35.291 / 31.544 ms，TTFT 为 771.960 / 642.939 ms，全程输出吞吐为 26.199 / 29.469 tok/s；每条输出及接受计数一致。
该小样本对照不替代上表 128 条参考数据，也未覆盖旧默认分组阈值 10；与 §8.1.1 的环境不同，GSM8K 仅用作性能负载。
固定问题测试脚本不能精确复现本表；需使用相同输入及计时口径。TTFT 变化尚未做独立归因。

### 8.2 精度评测（GPQA-Diamond）

**安装 evalscope**

```bash
pip install evalscope
```

**运行**（服务需已起；脚本会等 `/health` 返回 200）

```bash
MODEL_PATH="$W8A8_DIR" PORT=8020 bash tools/gpqa_accuracy_repeat.sh
# 默认重复 10 轮，逐轮打分并输出 mean / min / max / SD
# 可配置：REPEATS / PORT / HOST / OUT_DIR / OUT_PREFIX / EVALSCOPE
```

脚本对每一轮调用：

```bash
evalscope eval --model "$MODEL_PATH" --api-url http://127.0.0.1:8020/v1 --api-key EMPTY \
  --eval-type openai_api --datasets gpqa_diamond \
  --generation-config '{"temperature":1,"top_p":1,"max_tokens":32768,
                        "extra_body":{"chat_template_kwargs":{"thinking":false,"high_effort":false}}}' \
  --eval-batch-size 1 --repeats 1 --work-dir <每轮输出目录>
```

**必须重复多轮，单次结果不能下结论。** GPQA-Diamond 只有 198 题，`temperature=1` 下单次的二项标准误约
 **±3.3pp**；单次运行的高位/低位样本都不代表真实水平。

**实测参考（910B 实测，3 轮重复，thinking 关闭）**

| 轮次 | 分数 |
|---|---|
| R1 | 69.19% |
| R2 | 72.73% |
| R3 | 73.23% |
| **mean** | **71.72%** |
| min / max | 69.19% / 73.23% |
| SD | 1.80pp |

对比其它实现时应比较多轮均值，而非单次结果。

---

## 附录 A. A3 环境准备细节

§1A 的 `setup_dsv4_env_from_clean_cann.sh` 会处理下列全部内容，正常流程无需手工介入；
本附录供核对环境与排障使用。

**环境前提**

| 项 | 要求 |
|---|---|
| 硬件 | Ascend A3 |
| CANN | 9.0.0–9.2.0（自动选择已安装版本，可通过 `CANN_HOME` 指定） |
| Python | 3.11 / 3.12（可使用系统 Python，无需 conda） |
| 编译器 | gcc-13 / g++-13（gcc-9 不支持 `+bf16/+i8mm` 与 `-std=gnu++20`） |
| 系统库 | `pkg-config`、`libhwloc-dev`（kt-kernel 的 CMake 对 hwloc 是 REQUIRED）、`libhwloc15`（`kt_kernel_ext` 运行期依赖） |
| torch | 2.8.0 + torch_npu 2.8.0.post4（不可升级到 2.10，`kt_kernel_ext.so` 为 torch-2.8 ABI） |

`system_deps` 阶段安装系统库，`prereq` 阶段校验环境。`PYTHON_BIN`、`GITCODE` 可用环境变量指定。

**第三方算子仓库（均固定版本）**

| 用途 | 仓库 | 版本 |
|---|---|---|
| NSA/DSA 算子 → `custom_transformer` vendor | `gitcode.com/cann/ops-transformer` | `dd9f31f34` |
| 融合算子 → `customize` vendor + `custom_ops` 绑定 | `gitcode.com/cann/cann-recipes-infer` | `c5cc95e` |
| NPU 融合算子（sglang 依赖） | `github.com/sgl-project/sgl-kernel-npu` | tag `2026.6.2` |

三者构成：`customize` vendor（aclnn 融合算子）、`custom_transformer` vendor（NSA 算子）、
`custom_ops` 绑定（将上述 aclnn 暴露为 `torch.ops.custom.*`）。缺任一项，forward 会在
`aclnnXxx not in libopapi.so` 或 `torch.ops.custom.xxx 不存在` 处失败。

**分阶段执行**（`setup_dsv4_env_from_clean_cann.sh <phase>`，用于排障）

| phase | 内容 | 注意 |
|---|---|---|
| `system_deps` | 安装 `pkg-config`、`libhwloc-dev`、`libhwloc15` | 非 root 用户使用 sudo |
| `prereq` | 工具链与版本检查，`umask 0022` | umask 为 0002 时 CANN `msopgen` 会因安全检查中止构建 |
| `torch` | 校验 torch 2.8 / torch_npu 2.8.0.post4 | — |
| `triton` | 安装 triton-ascend | 版本不匹配时 import 即失败（CANN 9.0.0 缺符号） |
| `sglang_deps` | `pip install -r dsv4_sglang_base_reqs.txt -c dsv4_torch_lock.txt` | torch-lock 锁定 torch 版本 |
| `vendor_customize` | 构建并安装 `customize` vendor | 构建前需 `chmod -R go-w` |
| `custom_ops` | 构建并安装 `custom_ops` torch 绑定 | — |
| `vendor_transformer` | 构建并安装 `custom_transformer` vendor | vendor 名传 `--vendor_name=custom` |
| `sgl_kernel_npu` | 源码构建 sgl_kernel_npu | 需补 `-ldl` |
| `verify` | **环境** import 检查（torch/triton/sgl_kernel_npu/custom_ops + `torch.ops.custom.*`） | 不含 kt-kernel；`all` 末尾会执行一次 |

（`kt_kernel` 阶段属 §4，不在 `all` 内。）

---

## 附录 B. 补丁清单与手工操作

```
main_repo/   → ktransformers @ d7b5b49a3
  0001-kt-kernel-ascend-npu-backend.patch          NPU 后端 + ACL callback worker + 构建系统
  0002-kt-kernel-cpu-moe-mxfp4-kernel.patch        CPU MoE MXFP4 kernel + GGUF loader + 多 token 计算优化
sglang/      → sglang @ 298193eb3
  0001-sglang-npu-kv-triton-fallback.patch         triton×ascend 回退 + MTP 位置元数据 + KT 层级开关
  0002-sglang-kt-ep-cpu-moe-offload.patch          KT EP wrapper + 专家放置 + DSv4 target/MTP 隔离
  0003-sglang-streaming-prefill-depool.patch       流式 prefill + depool + 动态常驻 + GGUF dedup
  0004-sglang-nsa-compressor-and-mem-pools.patch   NSA compressor 模式 + NPU 内存池
  0005-sglang-packaging.patch                      Ascend/NPU 打包配置
llama_cpp/   → llama.cpp @ a94e6ff87
  0001-fix-gguf-NumPy-2-GGUFReader.patch           gguf-py NumPy 2.0 兼容
  0002-add-ggml-type-mxfp4.patch                   注册 GGML_TYPE_MXFP4=39 + NEON kernel + 多激活点积
```

CPU 优化让最多 4 条同专家激活共享 MXFP4 权重块的加载和解码，单条沿用原内核。
仅当扩展编译为 ARM/no-SVE、三个投影均为 MXFP4 且复用开启时，默认分组阈值从 10 降为 2；
x86 等其他构建保持原默认值 10。这影响 CPU 输入为 2–9 行的调用，包括 MTP、多请求合批和短 prefill，
不是专家 Top-K 或内核的 4 行 tile 上限。目前没有 x86 机器，未做 x86 实机性能验证。

`KT_GROUP_MIN_LEN` 显式设置始终优先，包括复用关闭时。
完整回退需设置 `KT_MOE_MXFP4_TOKEN_REUSE=0` 并取消 `KT_GROUP_MIN_LEN`，重启后使用原内核及默认阈值 10。

等价的手动操作（在 §0 的干净基线上按编号应用，不对已有补丁重复 apply）：

```bash
cd "$REPO"                       && for p in "$RELEASE_DIR"/main_repo/*.patch; do git apply "$p"; done
cd "$REPO/third_party/sglang"     && for p in "$RELEASE_DIR"/sglang/*.patch;    do git apply "$p"; done
cd "$REPO/third_party/llama.cpp"  && for p in "$RELEASE_DIR"/llama_cpp/*.patch; do git apply -p1 "$p"; done
```

应用后按 §4 重建 kt-kernel 并重启服务。CPU 优化默认生效，MTP 按 §7.2 开启。

---

## 附录 C. 构建的预期输出与手工编译

**kt-kernel 构建的预期输出**（§4）

预期：

- 配置期日志出现 `LLAMA_ARM_DOTPROD=ON`，且 `SVE=OFF / BF16=OFF / I8MM=OFF`。
  MXFP4 kernel 只用 dotprod（SDOT），不使用 SVE/BF16/I8MM；**若 CPU 带这些扩展（如 A3 主机），
  `setup.py` 会据 `/proc/cpuinfo` 自动开启，必须关掉**——SVE 打开后 MXFP4 MoE 会报 `llamafile not supported`。
- 出现 `Found Ascend CL library … libascendcl.so`，表示 NPU 后端就绪。
- ggml 的 `GGML_TYPE_MXFP4 not handled in switch` 为良性警告（非 MoE 路径的算子不需要该分支）。

**手工编译 AscendC 算子**（§6，用于在部署前验证编译链）

**方式 B（预先验证编译链）**：

```bash
cd scripts/tools/ascendc_mxfp4

# CANN 根目录。source 过 CANN 的 set_env.sh 后 ASCEND_TOOLKIT_HOME 已由 CANN 导出；
# 若在新 shell 中执行本步骤，显式指定即可，例如 CANN=$HOME/Ascend/cann-9.0.0。
CANN="${ASCEND_TOOLKIT_HOME:-${CANN_HOME:-}}"
[ -f "$CANN/aarch64-linux/tikcpp/tikcfw/kernel_operator.h" ] \
  || { echo "请先 source CANN 的 set_env.sh，或设置 CANN=<CANN 根目录>"; exit 1; }

TK=$CANN/aarch64-linux/tikcpp
bisheng -x asc --cce-aicore-arch=dav-c220 -O2 -std=c++17 -fPIC -shared \
  -I$TK/tikcfw -I$TK/tikcfw/impl -I$TK/tikcfw/interface -I$TK/tikcfw/lib \
  -I$CANN/aarch64-linux/include \
  mxfp4_fused_kernel.cpp -o libmxfp4fused.so \
  -L$CANN/aarch64-linux/lib64 -lruntime -lascendcl
```

---

## 附录 D. 数值对账（可选）

```bash
source tools/ensure_kt_kernel.sh && ensure_kt_kernel "$REPO"

"$PYTHON_BIN" tools/cpu_moe_reference_check_mxfp4.py --model-dir "$MXFP4_SRC" \
  --gguf "$GGUF_CACHE/dsv4_layer16_mxfp4.gguf" --layer-idx 16
```

预期 cosine ≥ 0.9999。脚本内置 `KT_FORCE_SYNC_SUBMIT=1`（单层孤立调用需同步提交，否则输出为零）。

---

---

## 附录 E. CPU 线程池调优

默认按单 NUMA 的 A3 主机配置：`KT_THREADPOOL_COUNT=1`、`KT_CPUINFER=32`。
该配置可移植，在多 NUMA 主机上同样能运行，但不会用满其带宽。

kt-kernel 为每个 threadpool 建一个子池并绑定到对应 NUMA 节点，因此
**`KT_THREADPOOL_COUNT` 不得超过主机的 NUMA 节点数**，否则启动时报 `NUMA node N not found`。
每子池线程数 = `KT_CPUINFER / KT_THREADPOOL_COUNT`。

多 NUMA 主机（如 192 核 / 8 NUMA 的 Kunpeng-920）应调高两者：

```bash
KT_THREADPOOL_COUNT=8 KT_CPUINFER=128    # 每 NUMA 16 线程
```

在该机型上实测：有效带宽在 128 线程处出现拐点（96→88，112→96，128→114，160→110 GB/s）；
96→128 使 CPU MoE 从 67.7 降至 55.1 ms/token（decode +24%）。不要使用全部核心（192 = 每 NUMA 24 线程）：
NUMA 任务分发线程、NPU host callback 与 python/OS 需要留有余量，否则线程池会 thrash。

---

## 附录 F. 计时与排障开关

下列开关均默认关闭，需要时在启动命令中加上即可（如
`MODEL_PATH=... KT_DECODE_TIMING=1 bash tools/launch_ds4flash_npu.sh`）。
关闭状态下仅有一次 getenv 判断，对性能无影响；开启后会按 token 输出计时日志。

| 环境变量 | 说明 |
|---|---|
| `KT_DECODE_TIMING=1` | 每 token 打印 CPU MoE submit→sync 耗时（`cpu_moe_wall`） |
| `KT_PREFILL_TIMING=1` | prefill 侧同类计时 |
| `KT_MOE_PHASE_TIMING=1` | 计算分段耗时（输入量化 / gate+up / down / merge） |
| `KT_FORCE_SYNC_SUBMIT=1` | 强制同步提交，配合 `EXTRA_FLAGS="--disable-cuda-graph"` 用于排障 |

CPU MoE 的行内预取与优化 GEMV 恒定生效；triton 与 ascend 版本不匹配时自动回退 torch 等价路径，均无需配置。

---

---

## 附录 G. 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| `git submodule update` 后 sglang / llama.cpp 版本不符 | pristine `.gitmodules` 指向其他仓库 | 按 §2 手动 clone 到指定 SHA |
| CMake 找不到 hwloc | 未安装 | `apt-get install -y pkg-config libhwloc-dev libhwloc15` |
| `No module named 'kt_kernel'` | `build_ext` 未注册包名 | 见 §4.1 |
| `import kt_kernel` 失败（已建链接） | 缺 `libhwloc.so.15` 或 `.so` 未生成 | 安装 libhwloc；确认 `kt_kernel_ext*.so` 存在 |
| 启动报量化类型不匹配 | sglang 未固定在 `298193eb3` | 重新 checkout 基线 SHA |
| 输出乱码，MXFP4 对账偏差 | nibble 序未重排 | 使用本交付的转换器；以 `verify_mxfp4_layer.py` 校验 |
| 输出乱码，但无任何报错 | W8A8 与 GGUF 量化基底不一致 | 使用官方发布的 W8A8，见 §1.1 |
| 长 prefill 报找不到 `libmxfp4fused.so` | AscendC 算子未编译 | 见 §6，或设 `KT_MXFP4_DEPOOL=0` |
| MXFP4 MoE 报 `llamafile not supported` | 构建 kt-kernel 时 SVE=ON | 关闭 ARM 扩展重新构建 |
| CANN `msopgen` 安全检查中止 | umask 为 0002 | `umask 0022` 后重新构建 |
| 长 prompt 请求失败，短 prompt 正常 | prompt 超过 `CHUNKED_PREFILL_SIZE`，NSA compressor 当前不支持跨 chunk prefill | 调高 `CHUNKED_PREFILL_SIZE` 至不小于最长 prompt，见 §7 |
| `[KT_STREAM] streaming failed … -> hybrid fallback` | 长 prefill 的激活超出启动后余量 | 确认 `MEM_FRACTION` 为默认的 0.81（见 [附录 G.1](#g1-长-prefill-的-hbm-预算)）、`CHUNKED_PREFILL_SIZE` 已生效 |
| `--chunked-prefill-size -1` 触发 malloc 越界 | `max_len=-1` 按 1 分配 | 已在 `llamafile.py` 对 ≤0 回落为 2048 |
| 服务运行中出现 `main process disappeared` | 后台启动，父进程被回收 | 前台运行服务 |
| eager 模式输出乱码 | CPU MoE 异步提交未 flush | 设 `KT_FORCE_SYNC_SUBMIT=1` |

### G.1 长 prefill 的 HBM 预算

单卡 HBM 的实测分配（A3，`KT_NUM_GPU_EXPERTS=32`，总容量 61.27 GiB）：

| 组成 | 大小 | 对应日志 |
|---|---|---|
| 加载期合计 | 48.32 GB | `Load weight end … mem usage=48.32 GB` |
| ├ 权重（含每层 32 个常驻专家） | ≈ 41.9 GB | — |
| └ depool 流式 slot | 6.44 GB | `[KT_STREAM][depool] reserved ND streaming slot (256,4096,4096)+(256,2048,4096) (6.44GB) at model-load time` |
| KV 池 | 3.41 GB | `SWAC4C128KVPool mem usage: 3.41 GB` |
| graph | 0.27 GB | `Capture npu graph end … mem usage=0.27 GB` |
| **启动后余量** | **≈ 8.75 GB** | `available_gpu_mem=8.75 GB` |

流式 slot 在 **KV 池 sizing 之前**预留，所以上面的余量是扣除它之后的净值——不要把 slot 再从余量里减一次。
启动日志里出现 `reserved ND streaming slot … at model-load time` 即表示预留成功。

这 8.7 GB 是运行期 prefill 激活与转换临时量的全部预算。转换按专家分块进行，单块的 fp16 转置临时量
约 1 GB（上表那条 OOM 里 `Tried to allocate 1.00 GiB` 即为此），加上该 chunk 的 prefill 激活；
`CHUNKED_PREFILL_SIZE` 越大，激活越多，余量越紧。

两个可用的调节项，作用对象不同：

| 调节项 | 腾出的来源 | 代价 |
|---|---|---|
| `KT_NUM_GPU_EXPERTS`（默认 32） | 常驻专家权重，每个约 1.0 GB | 命中率下降，decode 变慢 |
| `MEM_FRACTION`（默认 0.81） | KV 池 | 见下，**长 prefill OOM 时优先用这个** |

**`MEM_FRACTION` 是首选**：KV 池取的是扣除 `1 - MEM_FRACTION` 预留后的全部剩余 HBM，取值越低
池子越小、留给 prefill 激活的越多。池子容量远超 `context-length`(65536) 的部分不会被用到，缩小
它不损失能力；下限是 `max_total_num_tokens` 不得低于 `context-length`：

| `MEM_FRACTION` | KV 池 | `max_total_num_tokens` | 启动后余量 | 32k prompt 实测 |
|---|---|---|---|---|
| 0.85 | 3.28 GB | 519981 | 8.88 GB | 32k prefill OOM，退回 hybrid |
| 0.83 | 2.05 GB | 325687 | 10.11 GB | 不再 OOM，但余量吃紧、分配器反复回收：32k prefill 125 s |
| 0.82 | 1.44 GB | 228541 | 10.72 GB | — |
| **0.81（默认）** | 0.83 GB | 131394 | 11.33 GB | **32k prefill 恢复正常** |
| 0.80 | 0.22 GB | 34248 | — | `max_total_num_tokens` 低于 context-length，不可用 |

`KT_NUM_GPU_EXPERTS` 会改变热专家命中率、影响 decode 与性能口径，仅在 `MEM_FRACTION` 已到下限
仍不够时才动。也可降低 `CHUNKED_PREFILL_SIZE`，但它不得小于 prompt 的 token 数（见 §7）。

---

---

## 附录 H. 重新生成补丁

`gen_main_repo_patches.sh` 与 `gen_sglang_patches.sh` 对 pristine 基线执行 `git diff`。
在本交付基础上继续修改代码后，重新执行生成器即可产出更新后的全量补丁（前提是基线 SHA 在历史中可达）。

底层等价命令：

```bash
# 相对 pristine 的全量 diff（交付用）
git -C $REPO diff d7b5b49a3 -- kt-kernel/... > /tmp/my_kt_kernel.patch
git -C $REPO/third_party/sglang diff 298193eb3 -- python/sglang/... > /tmp/my_sglang.patch
```

若只需相对「已应用本交付补丁」的增量，先将该状态提交或打 tag 作为基线 `B`，再 `git diff B -- <paths>`。
生成器中的 `BASE`、`OUT`、pathspec 均可按需修改。

---
