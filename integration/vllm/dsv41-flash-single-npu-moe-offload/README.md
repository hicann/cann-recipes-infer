# DeepSeek-V4.1-Flash 单卡推理部署指南（Ascend 950PR + x86 CPU MoE）

把 DeepSeek-V4.1-Flash 部署到一张 Ascend 950PR 上。注意力、稠密层和 100 个常驻专家在卡上，
其余路由专家以 MXFP4 常驻主机内存由 kt-kernel 计算。layers.1 与 layers.14 的 Engram 表默认
只读映射 `/dev/shm` 中的 RAM 文件，也可选择 [SSD offload](#42-ssd-offload)。
完成基础部署后，可选启用 [DSpark K2/K5 推测解码](#51-可选dspark-k2k5)。本文从容器已创建、已进入容器 shell 开始。

方案的容量划分、两类权重的卸载机制、CPU–NPU 副流重叠与热专家标定置换见
[技术设计文档](../../../docs/models/deepseek_v4_1/dsv41-flash-950-single-card.md)。

## 部署环境

| 项目 | 要求 |
|---|---|
| NPU | 1× Ascend 950PR，`soc_version` 260。单卡 HBM 131072 MiB，启动时需 ≥ 116000 MiB 空闲 |
| CPU | x86_64，AVX512-BF16；验证机 2× AMD EPYC 9575F，256 线程、2 NUMA |
| 主机内存（DRAM） | RAM 模式 ≥ 560 GiB，且每个 NUMA node ≥ 240 GiB；SSD 模式省去约 188.83 GiB 的 Engram 常驻表，CPU 专家仍约需 370 GiB，另加运行和文件缓存余量 |
| `/dev/shm` | RAM 模式需 ≥ 200 GiB 可用；SSD 模式不暂存全量 Engram 表 |
| 容器 | `--privileged=true`、`--ipc=host`、`--network host`，挂载 `/dev/davinci*`、`/dev/davinci_manager`、`/dev/hisi_hdc` |
| CANN | 9.2.0-beta.2，位于 `/usr/local/Ascend/cann-9.2.0-beta.2` |
| Python | 3.12.13 / torch 2.10.0+cpu / torch_npu 2.10.0.post4.dev20260715 / triton 3.2.0 |
| 构建工具链 | C++20 编译器（验证机 g++ 13.3.0）、cmake ≥ 3.24、OpenMP、`libhwloc-dev`、`libnuma-dev`、`pkg-config` |
| 权重 | DeepSeek-V4.1-Flash，MXFP4，磁盘 475 GiB（48 个 shard 共 510296708312 B）|
| `$DSV41_ROOT` 磁盘 | ≥ 30 GiB |
| SSD（可选） | 除 checkpoint 外，另需约 189 GiB；使用支持 hard link 的文件系统，例如 XFS |

DSpark 权重额外占用约 7.39 GiB HBM。

下列三项由 Ascend vendor 提供，须预置于容器内：CANN 9.2.0-beta.2、ds41 算子包
（`/usr/local/Ascend/ds41_site`）、custom_transformer 算子包（`/usr/local/Ascend/dsv41_prebuilt`）。
vllm、vllm-ascend、kt-kernel 三棵源码树由第 2 步从公开仓库拉取。

容器镜像 `dsv41-singlecard:2026-09-26-clean` 可从
<https://cann-ai.obs.cn-north-4.myhuaweicloud.com:443/vllm-cann/DeepSeekV4.1-Flash-SingleCard/dsv41-singlecard-2026-09-26-clean.tar.gz?AccessKeyId=HPUAZLVQNN603FBK9RLL&Expires=1821795413&Signature=HycDv8AjGBOJOvfTA%2BwIWXdswWM%3D>
下载（6951073109 B，sha256 `e18753276c116891f42d24c74d710a44009430d4792b5c0b41868a3f40f406f0`），
`docker load` 导入后即含上述三项。

如果用 `sleep` 启动空容器，进入容器前可在宿主机执行 `docker top <容器名> -eo pid,comm`，确认进程列表只有一条 `sleep` 记录。

```bash
git clone https://gitcode.com/cann/cann-recipes-infer.git
cd cann-recipes-infer/integration/vllm/dsv41-flash-single-npu-moe-offload/guide-scripts
```

---

## 1. 配置

生成 `dsv41.env`，后续步骤只从该文件读取信息。每开一个新 shell 执行一次 `source ./dsv41.env`。

```bash
bash -l 01-config.sh --root /root/dsv41 --model /workspace/models/DeepSeek-V4.1-Flash
```

**或**（换卡换端口。与上一条二者选一，不要都执行）：

```bash
bash -l 01-config.sh --root /root/dsv41 --model /workspace/models/DeepSeek-V4.1-Flash --card 3 --port 8200
```

选定一条执行后，再 source 一次：

```bash
source ./dsv41.env
```

| 选项 | 默认值 | 含义 |
|---|---|---|
| `--root` | **必填** | 工作目录，绝对路径，放源码树、编译产物与日志 |
| `--model` | **必填** | 权重目录，绝对路径，需含 `model.safetensors.index.json` |
| `--cann` | `/usr/local/Ascend/cann-9.2.0-beta.2` | CANN 根目录，需含 `set_env.sh` |
| `--ds41-site` | `/usr/local/Ascend/ds41_site` | ds41 算子包 |
| `--prebuilt` | `/usr/local/Ascend/dsv41_prebuilt` | custom_transformer 算子包 |
| `--kt-site` | `<--root>/kt_site` | kt-kernel 安装目录，第 3 步产出 |
| `--kt-sha` | `105359b4bf70073af9fab992ad29920666acccc9` | ktransformers 主仓基线提交号 |
| `--kt-patch-dir` | `../ktransformers-patches` | 10 个 patch 所在目录 |
| `--card` | `0` | NPU 逻辑序号 |
| `--port` | `8100` | HTTP 端口 |

**预期**

```
[配置] PASS 工作目录 /root/dsv41
[配置] PASS 权重 /workspace/models/DeepSeek-V4.1-Flash
[配置] PASS CANN /usr/local/Ascend/cann-9.2.0-beta.2
[配置] PASS ktransformers patch 10 个 @ <仓库>/ktransformers-patches
[配置] PASS 配置写入 <guide-scripts 目录>/dsv41.env
=== 配置 完成 ===
```

---

## 2. 取码

取三棵源码树，初始化 kt 的两个子模块，按序打 10 个 patch，铺 custom_transformer 算子包。

更换源码版本时，用新的 `--root` 重新执行 `01-config.sh`，`--model` 可沿用原权重目录。
同一版本重跑取码会重置源码，有本地改动时也用新的 `--root`。

```bash
bash -l 02-fetch.sh
```

**或**（放宽取码重试。与上一条二者选一，不要都执行）：

```bash
DSV41_FETCH_TRIES=16 DSV41_FETCH_TIMEOUT=1800 bash -l 02-fetch.sh
```

| 参数 | 标准值 | 含义 |
|---|---|---|
| `DSV41_PROBE_TRIES` | `90` | 建连探测的最大次数，每次间隔 5 秒 |
| `DSV41_FETCH_TRIES` | `8` | 单棵树的取码尝试次数 |
| `DSV41_FETCH_TIMEOUT` | `900` | 单次取码的秒数上限 |
| `GIT_HTTP_LOW_SPEED_LIMIT` | `1000` | 低于该字节率视为停滞 |
| `GIT_HTTP_LOW_SPEED_TIME` | `60` | 持续停滞该秒数即中止本次尝试 |

**预期**

```
[取码] PASS vllm        a9686bc6e6809a49cb3e3c592a337d0721c3619f
[取码] PASS vllm-ascend 3609547566ef4aa753b105b5a071244777a1bdca
[取码] PASS ktransformers 105359b4bf70073af9fab992ad29920666acccc9
[取码] PASS 子模块 llama.cpp a94e6ff / pybind11 bb05e08
[取码] PASS patch 已应用 10 个
[取码] PASS write-tree = 393d8d5d99afef42449e4e804467c668d71a40c4
[取码] PASS diffstat = 16 files changed, 2080 insertions(+), 161 deletions(-)
[取码] PASS custom_transformer 文件数 = 786
[取码] PASS custom_opp_compiler_version=9.2.0-beta.2
[取码] PASS _build_info.py 23 B
[取码] PASS _version.py 80 B
=== 取码 完成 ===
```

---

## 3. 编译

编译 `vllm_ascend_C.so` 与 kt-kernel wheel，并把 kt-kernel 安装到 `$KT_SITE`。约 3～4 分钟。

```bash
bash -l 03-build.sh
```

**或**（只编 kt-kernel 并调编译并行度。与上一条二者选一，不要都执行）：

```bash
CPUINFER_PARALLEL=32 bash -l 03-build.sh kt
```

| 动作 | 含义 |
|---|---|
| `all`（默认） | 先编 vllm-ascend，再编 kt-kernel |
| `ascend` | 只编 `vllm_ascend_C.so` |
| `kt` | 只编并安装 kt-kernel |

| 参数 | 标准值 | 含义 |
|---|---|---|
| `CPUINFER_CPU_INSTRUCT` | `NATIVE` | 目标指令集，跨机分发用 `AVX512` |
| `CPUINFER_ENABLE_AMX` | `OFF` | Intel AMX，AMD EPYC 无此指令 |
| `CPUINFER_ENABLE_AVX512_VNNI` | `ON` | AVX512-VNNI |
| `CPUINFER_ENABLE_AVX512_BF16` | `ON` | AVX512-BF16 |
| `CPUINFER_ENABLE_AVX512_VBMI` | `ON` | AVX512-VBMI |
| `CPUINFER_PARALLEL` | `16` | C++ 编译并行度 |
| `TMPDIR` | `$DSV41_ROOT/tmp` | pip 构建临时目录 |

`CPUINFER_ENABLE_KML` / `CPUINFER_ENABLE_BLIS` / `CPUINFER_ENABLE_CPPTRACE` / `CPUINFER_BUILD_TYPE` /
`CPUINFER_VERBOSE` 写死在 `03-build.sh` 里，无对应环境变量。

**预期**

```
[编译] PASS vllm_ascend_C.so 1112760 B，用时 48 秒
[编译] PASS 日志 /root/dsv41/build-ascend.log
[编译] PASS pkg-config 1.8.1 / hwloc 2.10.0
[编译] PASS kt_kernel-0.7.0.post4-cp312-cp312-linux_x86_64.whl 2979211 B，用时 134 秒
[编译] PASS 定制开关 KT_ZEROCOPY_WEIGHTS / KT_ZEROCOPY_SCOPE / KT_FULLSET_LOAD 均在产物内
[编译] PASS kt_kernel 0.7.0.post4，__file__ 在 $KT_SITE 下
[编译] INFO __cpu_variant__ = avx512_bf16
[编译] PASS 日志 /root/dsv41/build-kt.log
=== 编译 完成 ===
```

---

## 4. 选择 Engram 存储方式

RAM 与 SSD 选一种，完成后进入第 5 步。切换前先执行 `bash -l 05-serve.sh stop`。

### 4.1 RAM（默认）

使用 RAM 时清除 SSD 设置：

```bash
unset VLLM_ASCEND_DSV41_ENGRAM_BACKEND VLLM_ASCEND_DSV41_ENGRAM_STAGE_DIR
```

把 `layers.1` 与 `layers.14` 的 embed weight/scale 从权重逐字节拷进 `/dev/shm/engram_stage`，
再校验大小与内容指纹。三轮实测 131 / 336 / 344 秒，取决于权重文件的 page cache 状态。

```bash
bash -l 04-engram.sh
```

**或**（只做校验，不重新暂存。与上一条二者选一，不要都执行）：

```bash
bash -l 04-engram.sh verify
```

| 动作 | 含义 |
|---|---|
| `all`（默认） | 暂存后校验 |
| `stage` | 只暂存 |
| `verify` | 只校验，读满 189 GiB，约 21 秒 |
| `release` | 释放这 188.83 GiB，必须先停服务；仍有进程映射时脚本报 FAIL 拒绝执行 |

RAM 暂存目录默认为 `DSV41_STAGE_DIR=/dev/shm/engram_stage`。
排布好后不会自动释放：停服务、删容器都不释放，只有 `release` 或宿主重启才归还。

**预期**

```
[engram] layers.1.engram.embed.weight           98305579008 B  162.1s    606 MB/s
[engram] layers.1.engram.embed.scale             3072049344 B    4.7s    656 MB/s
[engram] layers.14.engram.embed.weight          98308270592 B  164.5s    598 MB/s
[engram] layers.14.engram.embed.scale            3072133456 B    5.2s    595 MB/s
[engram] 暂存合计 336s，/dev/shm 增长 188.8 GiB
[engram] PASS verify_staged_tables，合计 202758032400 B
[engram] PASS verify_staged_content
[engram] /dev/shm 205G 已用 / 174G 可用
=== engram 完成 ===
```

### 4.2 SSD offload

把 Engram 表存到 SSD。权重目录中的索引文件和 Engram 所在的 shard 必须是普通文件，不能是软链接；使用 Hugging Face 缓存中的 snapshot 时先检查这一限制。

```bash
export SSD_STAGE=/mnt/nvme/dsv41-engram
python3 "$ASCEND_TREE/examples/kt_moe_offload/prepare_engram_ssd.py" \
  --checkpoint "$MODEL" --output "$SSD_STAGE"

# 校验已准备的表。
python3 "$ASCEND_TREE/examples/kt_moe_offload/prepare_engram_ssd.py" \
  --checkpoint "$MODEL" --output "$SSD_STAGE" --verify
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--checkpoint` | 必填 | 权重目录 |
| `--output` | 必填 | SSD 表的最终目录，首次准备时使用空目录 |
| `--verify` | 不启用 | 校验已有表 |
| `--chunk-mib` | `8` | 每次读写的缓冲大小，单位 MiB |

将目录设为只读，并选择 SSD 后端：

```bash
(
set -eu
mount --bind "$SSD_STAGE" "$SSD_STAGE"
mount -o remount,bind,ro "$SSD_STAGE" "$SSD_STAGE"
findmnt -rn -M "$SSD_STAGE" -o TARGET,OPTIONS
findmnt -rn -M "$SSD_STAGE" -o OPTIONS | tr ',' '\n' | grep -qx ro
)
ssd_mount_status=$?
if [ "$ssd_mount_status" -eq 0 ]; then
  export VLLM_ASCEND_DSV41_ENGRAM_BACKEND=ssd
  export VLLM_ASCEND_DSV41_ENGRAM_STAGE_DIR="$SSD_STAGE"
else
  echo "SSD 只读挂载失败，请检查后重试"
  false
fi
```

确认准备、校验和只读挂载均成功后，若已暂存 RAM 表，再执行 `bash -l 04-engram.sh release`。

**预期**

准备和校验的末行分别含 `"status": "prepared"`、`"status": "verified"`，
`total_bytes` 为 `202758032400`。`findmnt` 打印 SSD 目录，挂载选项含 `ro`。

---

## 5. 拉起服务

低时延档三轮实测就绪 511 / 571 / 691 秒，吞吐档两轮 361 / 371 秒。服务日志在 `$DSV41_ROOT/serve.log`。

```bash
bash -l 05-serve.sh
```

**或**（切吞吐档。与上一条二者选一，不要都执行）：

```bash
PROFILE=throughput MAXSEQS=8 bash -l 05-serve.sh start
```

| 动作 | 含义 |
|---|---|
| `start`（默认） | 后台拉起并等待 `/v1/models` 可用；端口已有服务应答时报 FAIL |
| `stop` | 按 `--port $PORT` 定位本部署的 vllm 会话，对会话内进程发 SIGTERM，最多等 200 秒；仍存活报 FAIL |

| 参数 | 格式 | 标准值 | 含义 |
|---|---|---|---|
| `PROFILE` | `lowlatency` / `throughput` | `lowlatency` | 服务档，对应 `serve_<PROFILE>.sh` |
| `CARD` | 非负整数 | `0` | 使用的 NPU 逻辑序号 |
| `PORT` | `1`～`65535` | `8100` | HTTP 服务端口 |
| `RESIDENT` | 正整数 | `100` | 每层常驻 NPU 的路由专家数 |
| `GPUUTIL` | `0`～`1` 小数 | `0.9` | 静态显存占比 |
| `MAXLEN` | 正整数 | `8192`（吞吐档 `32768`）| 最大上下文长度 |
| `MAXSEQS` | 正整数 | `1`（吞吐档 `8`）| 最大并发请求数 |
| `MAXBATCH` | 正整数 | `4096` | 单 batch 最大 token 数 |
| `VLLM_LOG_STATS_INTERVAL` | 正整数 | `5` | 吞吐统计打印间隔秒数 |
| `VLLM_ASCEND_DSV41_SWA_BLOCK_SIZE` | 后端支持的块大小 | `128` | SWA 块大小 |
| `KT_CONFIG` | 整段 JSON | 见 `serve_<PROFILE>.sh` | 整段替换 `--additional-config` 的 `kt_offload_config` |
| `EXTRA_ARGS` | `vllm serve` 参数 | 空 | 追加到 `vllm serve` 命令末尾 |

`kt_offload_config` 的细项、`COMMON_ARGS` 里的 serve 参数、`KT_FULLSET_LOAD` 没有对应环境变量，
要改得编辑 `$ASCEND_TREE/examples/kt_moe_offload/` 下的 `serve_<PROFILE>.sh` 与 `_common.sh`。

**预期**

```
[拉起] PASS 服务已就绪，用时 511 秒，会话 3792
[拉起]   ok: cann_ops_transformer -> /usr/local/Ascend/ds41_site
[拉起] INFO GPU KV cache size: 172,639 tokens, Maximum concurrency for 8,192 tokens per request: 21.07x
=== 拉起 完成 ===
```

### 5.1 可选：DSpark K2/K5

开启 DSpark K5，使用低时延档、并发 1：

```bash
bash -l 05-serve.sh stop
export PROFILE=lowlatency MAXSEQS=1 MAXLEN=8192 MAXBATCH=4096
unset KT_CONFIG  # 使用低时延档默认值，stream_prefill_min_tokens=512、dynamic_resident=true
export K=5
export EXTRA_ARGS="--language-model-only --seed 0 --compilation-config {\"cudagraph_capture_sizes\":[1,2,3,6],\"max_cudagraph_capture_size\":6} --speculative-config {\"method\":\"dspark\",\"num_speculative_tokens\":$K,\"enforce_eager\":true}"
bash -l 05-serve.sh start
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `K` | `5` | 每轮草稿 token 数，可选 `2` 或 `5`；修改后重新执行本节启动命令 |
| `KT_CONFIG` | 低时延档默认配置 | 自定义时使用完整 JSON，设置 `stream_prefill=true`、`stream_prefill_min_tokens=512`、`dynamic_resident=true` |

检查启动日志：

```bash
grep -F 'keeps all 128 draft experts in HBM' "$DSV41_ROOT/serve.log"
```

**预期**（日志节选）

```
kt_offload: model.layers.40.ffn.experts keeps all 128 draft experts in HBM
kt_offload: model.layers.41.ffn.experts keeps all 128 draft experts in HBM
kt_offload: model.layers.42.ffn.experts keeps all 128 draft experts in HBM
```

关闭 DSpark：

```bash
bash -l 05-serve.sh stop
unset EXTRA_ARGS
bash -l 05-serve.sh start
```

---

## 6. 验收

RAM 模式使用 `06-verify.sh`；SSD 模式使用第 6.1 节。

```bash
bash -l 06-verify.sh
```

**预期**

```
[验收] PASS model id = dsv41
[验收] PASS max_model_len = 8192
[验收] PASS 17*23 含 391 = yes
[验收] INFO EngineCore PID 4092（会话 3792，端口 8100）
[验收] PASS cann-9.1.0 映射 = 0
[验收] PASS 镜像内包映射 = 589
[验收] PASS engram 只读映射 = 4
[验收] PASS engram 映射模式 = r--s
[验收] PASS kt_kernel 解析 = yes
[验收] PASS 算子解析 = 1
[验收] PASS engram 未分配 = 2
[验收] INFO GPU KV cache size: 172,639 tokens, Maximum concurrency for 8,192 tokens per request: 21.07x
[验收] INFO numa FINAL_of_all_resident={0: '31.37%', 1: '68.63%'} on_target=100.00%
[验收] 自查 10 项，PASS 10，FAIL 0
=== 验收 完成 ===
```

### 6.1 SSD 后端验收

检查两个 Engram 层的 SSD 路径，并发送一个请求：

```bash
grep -F "Engram layer 1: backend=ssd path=$VLLM_ASCEND_DSV41_ENGRAM_STAGE_DIR bytes=" "$DSV41_ROOT/serve.log"
grep -F "Engram layer 14: backend=ssd path=$VLLM_ASCEND_DSV41_ENGRAM_STAGE_DIR bytes=" "$DSV41_ROOT/serve.log"
curl --noproxy '*' -fsS "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv41","messages":[{"role":"user","content":"What is 17 * 23?"}],"max_tokens":64,"temperature":0,"chat_template_kwargs":{"thinking":false}}'
```

**预期**

两条日志均含 `backend=ssd` 和设置的 SSD 目录，回答包含 `391`。

---

## 7. 吞吐

调用 `bench_latency.sh`，打印 TTFT、TPOT 和请求总耗时，结果写入 `OUTPUT_DIR`。

```bash
bash -l 07-throughput.sh
```

**或**（自定义负载。与上一条二者选一，不要都执行）：

```bash
CONCURRENCY=4 REQUESTS=20 PROMPT_TOKENS=1024 GEN_TOKENS=256 bash -l 07-throughput.sh
```

| 参数 | 格式 | 标准值 | 含义 |
|---|---|---|---|
| `CONCURRENCY` | 正整数 | `1` | 并发请求数，不要超过服务端的 `max_num_seqs` |
| `REQUESTS` | 正整数 | `5` | 总请求数 |
| `PROMPT_TOKENS` | 正整数 | `4096` | 每请求 prompt token 数 |
| `GEN_TOKENS` | 正整数 | `128` | 每请求生成 token 数 |
| `MODEL_NAME` | 字符串 | `dsv41` | 请求里的 model 字段，与 `--served-model-name` 一致 |
| `PROMPT_FILE` | 文本文件路径 | 内置英文语料 | 用文件内容校准输入长度 |
| `OUTPUT_DIR` | 新目录路径 | `$DSV41_ROOT/throughput/<UTC时间戳>` | 本次测量结果目录 |

**预期**（2026-11-14 实测：DSpark OFF、SSD Engram；4096 输入 / 128 输出，并发 1、5 次请求）

```
prompt: 4096 tokens of English prose (276 distinct source words)
{
  "status": "pass",
  "wall_s": 121.35536438300187,
  "requests": 5,
  "ttft_ids_s": {
    "mean": 20.882050534399603,
    "min": 20.525561056001607,
    "p50": 21.0033001160009,
    "max": 21.18487472599736,
    "sample_sd": 0.32815274986156673
  },
  "e2e_done_s": {
    "mean": 24.269457305399555,
    "min": 23.832427784000174,
    "p50": 24.32105287200102,
    "max": 24.774687264998647,
    "sample_sd": 0.4180651633020037
  },
  "tpot_delivery_ms": {
    "mean": 26.671676228344705,
    "min": 25.955000834646782,
    "p50": 26.262183322829717,
    "max": 28.45740940945574,
    "sample_sd": 1.0226653440021436
  },
  "aggregate_e2e_output_tokens_per_s": 5.273767692544164
}
evidence: <OUTPUT_DIR>
=== 吞吐 完成 ===
```

| 输出字段 | 含义 |
|---|---|
| `ttft_ids_s` | 等到首批输出 token 的时间，单位秒 |
| `tpot_delivery_ms` | 首末输出批次的时间差 /（输出 token 数 − 1），单位毫秒 |
| `e2e_done_s` | 请求到完成的总耗时，单位秒 |

**K5 性能参考**：上述硬件，SSD Engram、每层 100 个 hot expert、thinking OFF、prefill 门槛 512。
HumanEval 前 10 题拼成一条 1331-token 输入，输出 128 tokens，并发 1，1 次预热＋3 次正式测量，外部 1 Hz 资源采样。

| 模式 | Decode TPOT 均值 | 样本标准差 | 草稿接受率 |
|---|---:|---:|---:|
| DSpark K5 | **16.70 ms/token** | 0.60 ms | 64.52% |

此表 TPOT = `request_decode_time_seconds_sum` 增量 / 127。

---

## 8. 精度评测

关闭 DSpark，清除低时延档设置：

```bash
bash -l 05-serve.sh stop
unset EXTRA_ARGS MAXSEQS MAXLEN MAXBATCH
```

使用吞吐档启动服务，两个数据集依次各跑一次。`setup` 需要访问 github、PyPI 与语料所在的
对象存储（地址见附录 F），实测 673 秒，装出 299 个包、6.5 GiB 的 venv。

```bash
PROFILE=throughput bash -l 05-serve.sh
```

再执行 setup：

```bash
bash -l 08-accuracy.sh setup
```

**或**（链路差时换就近镜像。与上一条二者选一，不要都执行）：

```bash
PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple bash -l 08-accuracy.sh setup
```

**预期**

```
[拉起] PASS 服务已就绪，用时 361 秒，会话 11917
[拉起]   ok: cann_ops_transformer -> /usr/local/Ascend/ds41_site
[拉起] INFO GPU KV cache size: 615,399 tokens, Maximum concurrency for 32,768 tokens per request: 18.78x
=== 拉起 完成 ===
  profile=throughput  card=1 port=8100 max_model_len=32768 max_num_seqs=8 swa_block=128

[精度] PASS AISBench 52c2ae90629127ecca30b96b446526fc8378f091
[精度] PASS ais_bench 可执行
[精度] PASS gsm8k 语料
[精度] PASS gpqa_diamond 语料
=== 精度 完成 ===
```

### 8.1 GSM8K

1319 题 1283 正确，两轮实测 51 分 34 秒与 49 分 11 秒。

```bash
bash -l 08-accuracy.sh gsm8k
```

**预期**

```
[精度] PASS max_model_len 32768 >= 5120
dataset    version    metric    mode      dsv41
---------  ---------  --------  ------  -------
gsm8k      50861a     accuracy  gen       97.27
=== 精度 完成 ===
```

### 8.2 GPQA-Diamond

198 题 151 正确，两轮实测 33 分 2 秒与 31 分 24 秒。

```bash
bash -l 08-accuracy.sh gpqa
```

**预期**

```
[精度] PASS max_model_len 32768 >= 9216
dataset       version    metric    mode      dsv41
------------  ---------  --------  ------  -------
GPQA_diamond  b1ed2c     accuracy  gen       76.26
=== 精度 完成 ===
```

### 8.3 可选参数

下面两条是两个数据集各自的例子，不是二者选一：

```bash
NUM_SAMPLES=2 bash -l 08-accuracy.sh gsm8k
CONCURRENCY=4 THINKING=1 REASONING_EFFORT=high bash -l 08-accuracy.sh gpqa
```

| 参数 | 格式 | 标准值 | 含义 |
|---|---|---|---|
| `CONCURRENCY` | 正整数 | `8` | 并发请求数，不超过服务端 `max_num_seqs` |
| `THINKING` | `0` / `1` | `0` | `1` 开思考模式 |
| `REASONING_EFFORT` | `low` / `high` / `xhigh` / `max` / `1`-`100` | `high` | `THINKING=1` 时生效 |
| `TEMPERATURE` | 浮点 | `0.0` | 采样温度 |
| `TOP_P` | 浮点 | `1.0` | 采样 top-p |
| `NUM_SAMPLES` | 正整数 | 空 | 只跑每个数据集前 n 题 |
| `MODE` | `all` / `infer` / `eval` | `all` | 只推理或只判分 |
| `AISBENCH_REQUEST_TIMEOUT` | 秒 | 空 | 单请求客户端超时，空为不限 |
| `DSV41_ACC_DIR` | 路径 | `$DSV41_ROOT/accuracy` | AISBench 与产物目录 |
| `AIS_SHA` | 40 位 SHA | `52c2ae90629127ecca30b96b446526fc8378f091` | AISBench 基线提交号 |
| `AIS_REPO` | URL | `https://github.com/AISBench/benchmark.git` | AISBench 仓库 |
| `AIS_DATASET_URL` | URL | `http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data` | 语料下载地址 |
| `PIP_INDEX_URL` | URL | 空 | `setup` 装 AISBench 依赖时的 PyPI 源 |
| `PIP_TIMEOUT` | 秒 | `30` | `setup` 里 pip 的 socket 超时 |
| `PIP_RETRIES` | 正整数 | `10` | `setup` 里 pip 的重试次数 |
| `AIS_FETCH_TRIES` | 正整数 | `8` | `setup` 取 AISBench 的尝试次数 |
| `AIS_FETCH_TIMEOUT` | 秒 | `900` | 单次取 AISBench 的秒数上限 |
| `AIS_PROBE_TRIES` | 正整数 | `60` | 取 AISBench 前探测 `github.com:443` 的最大次数，每次间隔 5 秒 |

产物在 `$DSV41_ACC_DIR/outputs/<数据集>/<时间戳>/`：`summary/` 是上表，
`results/` 是判分明细，`predictions/` 是逐条预测。

---

## 附录 A 上游基线

| 组件 | 来源 | 基线提交号 | 形态 |
|---|---|---|---|
| vllm | `github.com/wenxuewuhd/vllm`，分支 `dsv41-moe-offload` | `a9686bc6e6809a49cb3e3c592a337d0721c3619f` | 第 2 步 `git fetch --depth 1 <SHA>` |
| vllm-ascend | `github.com/Pan-Boyi/vllm-ascend`，分支 `codex/dsv41-token-id-benchmark-20261119` | `3609547566ef4aa753b105b5a071244777a1bdca` | 第 2 步 `git fetch --depth 1 <SHA>` |
| kt-kernel | `github.com/kvcache-ai/ktransformers` 主仓 | `105359b4bf70073af9fab992ad29920666acccc9` | 第 2 步取基线 + `ktransformers-patches/` 10 个 patch，第 3 步编译 |
| `llama.cpp`（kt 子模块） | `github.com/ggerganov/llama.cpp` | `a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f` | 由 kt 基线的 gitlink 固定 |
| `pybind11`（kt 子模块） | `github.com/pybind/pybind11` | `bb05e0810b87e74709d9f4c4545f1f57a1b386f5` | 由 kt 基线的 gitlink 固定 |
| CANN | Ascend vendor | 9.2.0-beta.2 | 容器内预置 |
| ds41 算子包 | Ascend vendor | `cann_ops_transformer` 1.0.0 / `cannbotdsl` 0.5.0 / `cannbot_arena_net_ops` 0.1.0 | 容器内预置 |
| custom_transformer 算子包 | Ascend vendor | `custom_opp_compiler_version=9.2.0-beta.2`，786 个文件 | 容器内预置 |

patch 逐个说明见 [`ktransformers-patches/README.md`](ktransformers-patches/README.md)。

## 附录 B 环境变量

`dsv41.env` 导出下列变量，第 1 步生成，其余步骤只读。

| 变量 | 含义 |
|---|---|
| `DSV41_ROOT` | 工作目录 |
| `DSV41_SCRIPTS` | guide-scripts 目录 |
| `DSV41_PREBUILT` | custom_transformer 算子包 |
| `DSV41_STAGE_DIR` | RAM 表目录，默认 `/dev/shm/engram_stage` |
| `DSV41_VLLM_REPO` / `DSV41_VLLM_BRANCH` / `DSV41_VLLM_SHA` | vllm 来源与基线 |
| `DSV41_ASCEND_REPO` / `DSV41_ASCEND_BRANCH` / `DSV41_ASCEND_SHA` | vllm-ascend 来源与基线 |
| `DSV41_KT_REPO` / `DSV41_KT_SHA` | ktransformers 主仓与基线 |
| `DSV41_KT_TREE_HASH` | 打完 10 个 patch 后的期望 `write-tree` |
| `DSV41_KT_DIFFSTAT` | 打完 10 个 patch 后的期望 `diff --stat` 末行 |
| `DSV41_KT_SUBMODULE_LLAMA` / `DSV41_KT_SUBMODULE_PYBIND` | 两个子模块的期望 gitlink |
| `DSV41_KT_PATCH_DIR` | 10 个 patch 所在目录 |
| `DSV41_KT_WHEEL_DIR` | kt-kernel wheel 输出目录 |
| `MODEL` | 权重目录 |
| `CANN` | CANN 根目录 |
| `DS41_SITE` | ds41 算子包 |
| `KT_SITE` | kt-kernel 安装目录 |
| `KT_TREE` | 第 2 步拉取的 ktransformers 源码树 |
| `VLLM_TREE` / `ASCEND_TREE` | 第 2 步拉取的 vllm 与 vllm-ascend 源码树 |
| `CARD` / `PORT` | NPU 逻辑序号与 HTTP 端口 |

以下变量需在每个新 shell 中重新导出：

| 变量 | 默认值 | 含义 |
|---|---|---|
| `VLLM_ASCEND_DSV41_ENGRAM_BACKEND` | `ram` | 选择 `ram` 或 `ssd` |
| `VLLM_ASCEND_DSV41_ENGRAM_STAGE_DIR` | `/dev/shm/engram_stage` | SSD 模式设为已准备表的绝对路径 |

## 附录 C 量化

权重为 MXFP4，量化方式由权重目录 `config.json` 的 `quantization_config` 决定，不通过脚本参数控制。

## 附录 D License

本目录脚本与 patch 以 Apache 2.0 发布，见 [LICENSE.txt](LICENSE.txt)。
vllm、vllm-ascend、ktransformers 为上游代码，各自许可证见其仓库。

## 附录 E kt-kernel 数值校验

纯 CPU 运行，不打开 NPU。校验项与可调参数见
[`ktransformers-tests/README.md`](ktransformers-tests/README.md)。

```bash
env -u LD_PRELOAD PYTHONPATH="$KT_SITE" python3 ../ktransformers-tests/dsv41_mxfp4_offload_check.py
```

末行 `VERDICT: PASS`，实测墙钟 27 秒；加 `--roundtrip-only` 只做字节回读与权重落点审计，5 秒。
`--cpu-ids` / `--n-cpu` 至少给 2 个专家，全量模式至少 4 个。

## 附录 F 精度评测的数据集来源

两个数据集都不在本仓内，由 `08-accuracy.sh setup` 从开源站点下载。

| 内容 | 来源 | 落地位置 |
|---|---|---|
| 评测框架 AISBench | `github.com/AISBench/benchmark`，提交号 `52c2ae90629127ecca30b96b446526fc8378f091` | `$DSV41_ACC_DIR/benchmark` |
| GSM8K 语料 | OpenCompass 公开对象存储 `http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip` | `<benchmark>/ais_bench/datasets/gsm8k/`，判据文件 `test.jsonl`，1319 题 |
| GPQA-Diamond 语料 | 同一个对象存储 `.../datasets/data/gpqa.zip` | `<benchmark>/ais_bench/datasets/gpqa/`，判据文件 `gpqa_diamond.csv`，198 题 |

地址前缀由 `AIS_DATASET_URL` 决定，按 `<前缀>/<数据集>.zip` 拼，内网可指向自建镜像。
语料不经 HuggingFace，评测时 `HF_DATASETS_OFFLINE=1` / `HF_HUB_OFFLINE=1`。
本仓 `accuracy/` 只有评测口径（数据集 config、服务模型 config、GSM8K 判分），
题目本身不在仓里。`setup` 可重复执行，语料与 venv 已存在就跳过。
