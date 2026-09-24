# GLM-5.3-Flash 单卡推理部署指南（昇腾 910C + 鲲鹏 CPU 混合专家）

本文介绍如何把 GLM-5.3-Flash 部署到一张 Ascend 910C die 上。注意力、3 个稠密层和
32 个常驻专家用 INT8 W8A8 放在 die 上，其余路由专家用 MXFP4 常驻 Kunpeng 主机内存，
由 kt-kernel 计算。

架构、数据流、容量模型和关键实现约束见
[技术设计文档](../../../docs/integration/sglang/glm53-flash-single-npu-moe-offload/glm53_flash_single_card_design.md)。

## 部署环境

当前已完成实测的部署环境为 CANNLab 一站式开发平台，环境基线如下：

| 项目 | 要求 |
|---|---|
| 开发平台 | CANNLab 一站式开发平台，通过 CANNLab Dev Space 进入环境 |
| 环境规格 | A3 单卡，1× Ascend 910C die（`ascend910_93`）+ Kunpeng CPU |
| 参考镜像 | `py3.12-a3-arm-20260829` |
| CANN | 包元数据 9.2.0、编译器 9.1.0；自定义算子按编译器版本构建 |
| Python | 3.12.9 |
| 主机内存 | ≥ 200 GiB；验证环境为 40 核、1 NUMA、229 GiB |
| 持久磁盘 | 服务产物至少约 630 GB；从 FP8 开始转换时还需额外约 306 GB |

以下八步均在该 CANNLab 环境的**登录 shell** 中执行。完整要求见附录 A，上游代码基线见附录 B。

```bash
git clone https://gitcode.com/cann/cann-recipes-infer.git
cd cann-recipes-infer/integration/sglang/glm53-flash-single-npu-moe-offload/guide-scripts
```

---

## 1. 配置文件

`01-config.sh` 不依赖固定用户名、挂载点或 Python 路径。默认把工作目录设为
`$HOME/glm53`、权重目录设为 `<工作目录>/models`，并使用 PATH 中的 `python3`。
需要调整时使用脚本选项，不新增路径类环境变量：

```bash
bash -l 01-config.sh \
  --root /data/glm53 \
  --model-root /models \
  --python-bin /opt/python/bin/python3
```

其余选项通过 `bash -l 01-config.sh --help` 查看。

**预期** 末行 `=== 配置 完成 ===`。

每开一个新 shell 执行一次。

```bash
source ./glm53.env
```

---

## 2. 取码并打补丁

浅克隆 ktransformers 与 sglang，打 11 份补丁，复制部署脚本，做 7 项校验。

```bash
bash -l 02-fetch-patch.sh
```

**预期** 末尾 9 行

```
[取码打补丁] PASS  ktransformers HEAD = 6d460cc10780f2e0ad541b5d58ad28086dd32ef0
[取码打补丁] PASS  sglang HEAD = 5aab054ec8ce6b6100fbfb7aafe67d632a7df3aa
[取码打补丁] PASS  sglang 树 = f34d08b7f3a447034f2f83cdc69a236b605142b9
[取码打补丁] PASS  补丁改动量 = 3 files changed, 211 insertions(+), 70 deletions(-)
[取码打补丁] PASS  部署脚本与 cann-recipes-infer 里的原件逐字节一致
[取码打补丁] PASS  脚本可执行位
[取码打补丁] PASS  third_party/sglang 是符号链接
[取码打补丁] PASS  符号链接可解析
=== 取码打补丁 完成 ===
```

---

## 3. Python 环境

项目虚拟环境的直接依赖由交付目录 `requirements.txt` 统一锁定，其中 torch、torch_npu
和 scipy 分别为 2.9.1、2.9.1 和 1.13.1；随后拷入 cp312 的 triton_ascend。

```bash
bash -l 03-python-env.sh --triton-src <含 triton 的 site-packages 目录>
```

**预期** 末尾包含

```
[Python 环境] PASS triton_ascend 3.2.2
[Python 环境] PASS triton.__version__ 3.2.0
[Python 环境] PASS torch_npu 要求并使用 torch 2.9.1
[Python 环境] PASS triton_ascend 要求并使用 scipy 1.13.1
[Python 环境] 锁定依赖 <交付目录>/requirements.txt
=== Python 环境 完成 ===
```

---

## 4. 权重

从厂商 FP8 转出 INT8 W8A8 与 MXFP4。已经有这两份就跳过本节。
改 `04-weights.sh` 顶部的 `# ==== 改这里 ====`，然后执行。

```bash
bash 04-weights.sh
```

**预期** 末行 `=== 权重 完成 ===`，`$GLM53_MODEL_ROOT` 下多出 `GLM-5.3-Flash-W8A8`
与 `GLM-5.3-Flash-MXFP4`。

四个阶段可以分开跑。

```bash
bash 04-weights.sh space      # 只查盘
bash 04-weights.sh download   # FP8 原件，62 分片约 306 GiB
bash 04-weights.sh int8       # 转 INT8 W8A8，约 307 GiB
bash 04-weights.sh mxfp4      # 转 MXFP4，约 170 GiB
```

### 4.1 空间

三份并存要 783 GiB。

```bash
bash 04-weights.sh space
```

**预期**

```
[权重] FP8   <GLM53_MODEL_ROOT>/GLM-5.3-Flash-FP8 已占 0 GiB，满量 306 GiB
[权重] W8A8  <GLM53_MODEL_ROOT>/GLM-5.3-Flash-W8A8 已占 0 GiB，满量 307 GiB
[权重] MXFP4 <GLM53_MODEL_ROOT>/GLM-5.3-Flash-MXFP4 已占 0 GiB，满量 170 GiB
[权重] 三份并存要 783 GiB
=== 权重 完成 ===
```

### 4.2 下载 FP8

62 个分片共 305.8 GiB，断点续传。

```bash
bash 04-weights.sh download
```

**预期**

```
[权重] PASS 下载 FP8 空间够，需要 346 GiB
[权重] PASS FP8 原始权重就绪 <GLM53_MODEL_ROOT>/GLM-5.3-Flash-FP8
=== 权重 完成 ===
```

### 4.3 转出 W8A8 与 MXFP4

```bash
bash 04-weights.sh int8 mxfp4
```

**预期**

```
[权重] PASS 转 W8A8 空间够，需要 347 GiB
[权重] PASS FP8 软链视图就绪 <工作目录>/weights-conv/fp8-view
weight_block_size=[128, 128]  tensors_to_quantize=37338
62 shards, 62 to convert
[权重] PASS W8A8 就绪 <GLM53_MODEL_ROOT>/GLM-5.3-Flash-W8A8
[权重] PASS MXFP4 就绪 <GLM53_MODEL_ROOT>/GLM-5.3-Flash-MXFP4
=== 权重 完成 ===
```

### 4.4 删除 FP8

两份派生权重都在之后，FP8 原件可以删，能腾出 306 GiB。

```bash
rm -rf "$GLM53_MODEL_ROOT/GLM-5.3-Flash-FP8"
```

---

## 5. 构建

五步构建，补 `deep_ep` 的 `.pth`，跑启动前自检。约 1.5 到 2.5 小时。

```bash
bash 05-build.sh
```

**预期**

```
[构建] 2026-09-09 10:02:11 PASS deps 完成，用时 4 分 12 秒
[构建] 2026-09-09 10:47:36 PASS sgl-kernel 完成，用时 45 分 25 秒
[构建] 2026-09-09 11:39:02 PASS cann-ops 完成，用时 51 分 26 秒
[构建] 2026-09-09 11:43:55 PASS kt-kernel 完成，用时 4 分 53 秒
[构建] 2026-09-09 11:58:20 PASS gguf 完成，用时 14 分 25 秒
[构建] 2026-09-09 11:58:31 PASS 补完 .pth 后 deep_ep ok，用时 0 分 11 秒
[构建] 2026-09-09 11:59:04 PASS PREFLIGHT OK，用时 0 分 33 秒
=== 构建 完成 ===
```

只重跑其中一步，把步骤名当参数传，可用 `precheck` `deps` `sgl-kernel` `cann-ops`
`kt-kernel` `gguf` `deep-ep` `check`。

```bash
bash 05-build.sh sgl-kernel
```

---

## 6. 拉起单卡推理

`06-serve.sh` 是面向用户的启动入口，接收一个可选动作：

| 参数 | 选项 | 默认值 | 含义 |
|---|---|---|---|
| `ACTION` | `start` / `stop` | `start` | 不传或传 `start` 时拉起并等待服务就绪；`stop` 按配置端口停止服务 |

### 6.1 配置并启动

启动参数来自第 1 步生成的 `glm53.env`。需要覆盖默认值时，必须在 `source ./glm53.env`
之后、执行 `06-serve.sh` 之前 export；`KT_*` 参数大多在模块 import 时读取，服务启动后修改
不会生效。

```bash
source ./glm53.env
export GLM53_NPU_DEVICE_ID=0
export GLM53_PORT=30013
export GLM53_PREFILL_STREAM=1
export GLM53_NUM_GPU_EXPERTS=32
bash 06-serve.sh start
```

不需要修改参数时可直接执行：

```bash
bash 06-serve.sh
```

**预期**

```
[06-serve] PASS 服务已就绪，日志里出现了 fired up and ready to roll
=== 06-serve 完成 ===
```

### 6.2 功能开关

| 参数 | 选项或格式 | 标准值 | 含义 |
|---|---|---|---|
| `GLM53_PREFILL_STREAM` | `0` / `1` | `1` | `1` 对长 prefill 使用整层专家流式加载；`0` 全部走 CPU/NPU hybrid |
| `KT_DYNAMIC_RESIDENT` | `0` / `1` | 流式路径下为 `1` | 按 prompt 路由统计更新每层常驻专家；仅流式路径生效 |
| `KT_SIDE_STREAM` | `0` / `1` | `1` | 把 CPU MoE host callback 放到第二条流，与 NPU resident expert 计算重叠 |
| `GLM53_EAGER` | `0` / `1` | `0` | `0` 使用 decode graph；`1` 关闭图捕获，并失去图提交路径的 side-stream overlap |
| `KT_MXFP4_GGUF_DEDUP` | `0` / `1` | `1` | `1` 让 streaming 与 CPU MoE 复用同一份 GGUF mmap |
| `KT_MXFP4_PREFETCH` | `0` / `1` | `1` | 是否在主机侧预取下一层 MXFP4 数据 |
| `KT_STREAM_WARMUP` | `0` / `1` | `1` | 首次流式请求前先预热 CPU MoE 线程池和内存页 |
| `KT_STREAM_STRICT` | `0` / `1` | `0` | `0` 在流式异常时回退 hybrid；`1` 直接抛错，适合调试 |

### 6.3 运行时配置

| 参数 | 选项或格式 | 标准值 | 含义 |
|---|---|---|---|
| `GLM53_NPU_DEVICE_ID` | 非负整数 | `0` | CANNLab 容器内的逻辑 die 序号 |
| `GLM53_PORT` | `1`～`65535` | `30013` | HTTP 服务端口，也是停止服务时的进程匹配条件 |
| `GLM53_NUM_GPU_EXPERTS` | 正整数 | `32` | 每层常驻 NPU 的路由专家数；增大时 HBM 按约 `0.9925 GiB × N` 增长 |
| `GLM53_MEM_FRACTION` | `0`～`1` 的小数 | 流式 `0.95`；hybrid `0.85` | SGLang 静态显存占比 |
| `GLM53_MAX_TOTAL_TOKENS` | 正整数或留空 | 流式 `40960`；hybrid 留空 | KV pool token 上限；与显存占比、并发和 context 一起配置 |
| `GLM53_CONTEXT_LENGTH` | 正整数 | `32768` | 最大上下文长度 |
| `GLM53_CHUNKED_PREFILL_SIZE` | 64 的正整数倍 | 流式 `6144`；hybrid `8192` | 每个 prefill chunk 的 token 数；增大会增加 KDA workspace |
| `GLM53_MAX_RUNNING_REQUESTS` | 正整数 | `1` | 最大并发请求数；当前只验证单请求 |
| `KT_PREFILL_STREAM_THRESHOLD` | 正整数 | `512` | 当前 prefill chunk 达到该 token 数才进入流式路径 |
| `KT_HOT_TAIL_TOKENS` | 正整数 | `512` | 用 prompt 尾部多少个 token 统计 decode 热专家 |
| `KT_MXFP4_NZ_CHUNK` | 正整数 | `16` | 单次搬运并在线转换为 W8A8-NZ 的专家数；越大瞬时 HBM 越高 |
| `GLM53_THREADPOOL_COUNT` | 正整数 | 自动取 NUMA 节点数 | kt-kernel CPU MoE 线程池数，每个线程池对应一个 NUMA 子池 |
| `GLM53_CPUINFER` | 正整数 | `线程池数 × 16`，不超过核数的 3/4 | CPU MoE 工作线程总数 |
| `GLM53_KT_NUMA_NODES` | 逗号分隔节点号 | 留空 | 指定 CPU MoE 子池绑定的 NUMA 节点，例如 `0,1` |
| `GLM53_PIN_CORES` | `taskset` 核号格式 | 留空 | 从进程启动开始绑核，例如 `0-39`；应与 NUMA 节点一致 |

`GLM53_PREFILL_STREAM` 会联动 `GLM53_MEM_FRACTION`、`GLM53_MAX_TOTAL_TOKENS` 和
`GLM53_CHUNKED_PREFILL_SIZE`。除非重新完成容量与长 prompt 验收，否则不要只修改其中一项。
详细容量关系见[技术设计文档 §8](../../../docs/integration/sglang/glm53-flash-single-npu-moe-offload/glm53_flash_single_card_design.md#8-参数耦合)。

### 6.4 检查解析结果和最终命令

修改参数后，可以先查看解析结果和最终启动命令。以下命令不会正式拉起服务：

| 底层 `serve.sh` 选项 | 含义 |
|---|---|
| `--foreground` | 在当前终端前台运行，不使用 `nohup` 后台启动 |
| `-h` / `--help` | 显示底层启动脚本用法 |
| `--dry-run` | 完成前置检查并打印最终 `sglang.launch_server` 命令，但不占用 NPU |

```bash
source ./glm53.env
(cd "$KT_REPO/kt-kernel/tools/ascend_glm53" && bash glm53_env.sh --show)
(cd "$KT_REPO/kt-kernel/tools/ascend_glm53" && ./serve.sh --dry-run)
```

### 6.5 停服务

第 7 节与第 8 节都要服务在跑，两节做完再停。

```bash
bash 06-serve.sh stop
```

**预期**

```
[06-serve] PASS 服务已停
=== 06-serve 完成 ===
```

---

## 7. 验收

约 7 分钟。

```bash
bash 07-verify.sh
```

**预期** 最后两行

```
[验收] 自查 31 项，PASS 31，FAIL 0
=== 验收 完成 ===
```

---

## 8. 精度评测

判定标准是困惑度。三步可以一次跑完 `bash 08-accuracy.sh`，也可以分开跑。

### 8.1 语料

困惑度用 wikitext-2-raw-v1 的 test split，一个含 `text` 列的 parquet，
落到 `$GLM53_EVAL_DIR/wikitext/test.parquet`，`GLM53_EVAL_DIR` 默认是 `$GLM53_ENV_ROOT/eval`。

```bash
bash 08-accuracy.sh corpus
```

脚本按这个顺序找，前一条命中就不做后面的。

| 顺序 | 来源 |
|---|---|
| 1 | 目标路径上已有的那份 |
| 2 | 脚本顶部的 `SRC_PARQUET`，填了就从那里拷 |
| 3 | 全盘搜 `*wikitext*/test.parquet`，最多 5 分钟 |
| 4 | 从 HuggingFace 下 `load_dataset("wikitext", "wikitext-2-raw-v1", split="test")` |

**预期**

```
  列    ['text']  期望 ['text']
  行数  4358  期望 4358
  字符  1285622  期望 1285622
[精度评测] PASS 语料就位 <你的 GLM53_ENV_ROOT>/eval/wikitext/test.parquet
=== 精度评测 完成 ===
```

三项必须全中。换一份语料就是换基准，跑出来的困惑度不能和 3.588836580023618 比。

连不上 HuggingFace 时，把手上那份 parquet 的路径填进 `08-accuracy.sh` 顶部的 `SRC_PARQUET`，重跑本节。

语料要放在别的目录，通过 `GLM53_EVAL_DIR` 指定路径并重跑第 1 步。
只在当前 shell `export` 是不够的，换个 shell 就丢。

### 8.2 困惑度

32 窗，约 10 分钟。

```bash
bash 08-accuracy.sh ppl
```

**预期**

```
  perplexity 3.588836580023618  期望 3.588836580023618
  mean_nll   1.2778280775605546  期望 1.2778280775605546
[精度评测] PASS 32 窗困惑度与参考值逐位相同
=== 精度评测 完成 ===
```

### 8.3 GSM8K 与 GPQA

这两个基准的语料由 evalscope 从 ModelScope 拉，缓存在 `$GLM53_ENV_ROOT/ms_cache`，
换目录改脚本顶部的 `MS_CACHE`。evalscope 装在独立虚拟环境 `$GLM53_ENV_ROOT/.venv-eval` 里，
装它要能连 pypi。默认每个基准跑前 5 题。

```bash
bash 08-accuracy.sh bench
bash 08-accuracy.sh bench --limit 0    # 全量
```

**预期**

```
[精度评测] PASS gsm8k score=<分数> scored=5 wall=<秒>s
  每题约 <秒> 秒。GSM8K 全量 1319 题按这个速度约 <小时> 小时
[精度评测] PASS gpqa score=<分数> scored=5 wall=<秒>s
=== 精度评测 完成 ===
```

当前参考环境已完成 GPQA-Diamond 全量 198 题，`reasoning_effort=max`，结果为
`accuracy 0.8485`（168/198）。GSM8K、MATH500 和 MMLU-Pro 尚未形成该配置下的参考结果。

---

## 附录 A 环境要求

| 部件 | 要求 |
|---|---|
| NPU | 1× 昇腾 910C（A3，`ascend910_93`），一张 die 的 HBM 要空着，约 61.3 GiB |
| CPU | aarch64 鲲鹏，验证机为 40 核 / 1 NUMA |
| 内存 | ≥ 200 GiB |
| 磁盘 | W8A8 307 GB + MXFP4 170 GB + GGUF 151 GiB，日志与构建树另需约 10 GiB。从 FP8 转起还要额外 306 GB |
| CANN | 包元数据 9.2.0、编译器 9.1.0；自定义算子按编译器版本构建并安装到项目环境的 `opp_custom` |
| 编译器 | gcc ≥ 11 |
| hwloc | 已装开发包，含 `.pc` 文件 |
| Python | 3.12.9 / torch 2.9.1 / torch_npu 2.9.1 / scipy 1.13.1 / triton_ascend 3.2.2 |

CANN 装在 `$HOME/Ascend`、`/home/developer/Ascend`、`/usr/local/Ascend`、`/opt/Ascend` 四者之一。

`triton_ascend` 的 cp312 版本不在任何公开镜像源上，要从已有的昇腾镜像里取。

## 附录 B 上游基线

| 仓库 | 来源 | 基线提交号 |
|---|---|---|
| ktransformers | `github.com/kvcache-ai/ktransformers`，主线 `main` | `6d460cc10780f2e0ad541b5d58ad28086dd32ef0` |
| sglang | `github.com/sgl-project/sglang`，主线 `main` | `5aab054ec8ce6b6100fbfb7aafe67d632a7df3aa` |
| sgl-kernel-npu | `github.com/sgl-project/sgl-kernel-npu` | tag `20260826`，commit `146153e581eb243818511dd3f0669679a59398c4` |

补丁对 ktransformers 的改动量是 `3 files changed, 211 insertions(+), 70 deletions(-)`，
不含随后复制进去的 `ascend_glm53`。sglang 打完 10 份补丁的树是
`f34d08b7f3a447034f2f83cdc69a236b605142b9`。

`weight-conversion/tools/` 下两个脚本取自 sglang 提交 `97c69783` 的
`docs/docs/glm53_npu_support/tools/`，该路径在基线 `5aab054e` 上已被上游移除。
