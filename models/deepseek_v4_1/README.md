# DeepSeek-V4.1 Inference on NPU
## 概述
DeepSeek团队发布了最新的模型DeepSeek-V4.1-Flash，本实践基于DeepSeek开源代码进行迁移，并在CANN平台上完成性能优化，支持在昇腾`Ascend 950`平台部署。

- 本实践的优化特性参见[DeepSeek-V4.1-Flash CANN优化实践](../../docs/models/deepseek_v4_1/deepseek_v4.1_flash_cann_tech_report.md)。

---

## 硬件要求
产品型号：Ascend 950 系列

操作系统：Linux ARM（Ascend 950）

## 环境准备
### Atlas A5 Docker 部署

A5 使用预置 ARM 镜像，镜像中已包含本实践所需的 PyTorch、torch_npu、torchair 和 CANN 运行环境。从 [A5 ARM 镜像地址](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-quantization/DeepSeek/cann9.2.0.pt2.13.0_dsv4.1_aarch_a5_image_custom_20260930.tar) 下载 docker 镜像，上传到 A5 服务器的每个节点，并在每个节点执行：

```text
docker load -i cann9.2.0.pt2.13.0_dsv4.1_aarch_a5_image_custom_20260930.tar
```

### DeepSeek 高性能算子库

DeepSeek 官方开源了 [TileKernels](https://github.com/deepseek-ai/TileKernels) 和 [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) 高性能算子库，本实践已提供相应的使用样例支持。TileKernels 依赖的 TileLang 及相关算子已编译并集成在本节提供的预置镜像中，无需额外安装；DeepGEMM 未集成在镜像中，如需使用，请按照 [DeepGEMM 官方 README](https://github.com/deepseek-ai/DeepGEMM/blob/main/README.md) 自行编译安装。具体算子实现请参考对应的官方开源仓库。

### 拉起 docker 容器

在各个节点上通过如下脚本拉起容器，默认容器名为 `cann_recipes_infer`。请将权重路径和源码路径挂载到容器中。

```text
docker run -u root -itd --name cann_recipes_infer --ulimit nproc=65535:65535 --ipc=host \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci6 \
    --device=/dev/davinci7 \
    --device=/dev/davinci_manager --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    --device=/dev/ummu --device=/dev/uburma \
    -v /usr/local/sbin/urma_perftest:/usr/local/sbin/urma_perftest \
    -v /usr/bin/urma_admin:/usr/bin/urma_admin \
    -v /usr/lib64:/usr/lib64 -v /usr/lib:/usr/lib \
    -v /home/:/home -v /data:/data \
    -v /etc/localtime:/etc/localtime \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /etc/ascend_install.info:/etc/ascend_install.info -v /var/log/npu/:/usr/slog \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
    -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/sbin:/usr/local/sbin \
    -v /etc/hccn.conf:/etc/hccn.conf -v /root/.pip:/root/.pip \
    -v /etc/hosts:/etc/hosts -v /usr/bin/hostname:/usr/bin/hostname \
    -v /etc/hixlep:/etc/hixlep -v /lib/route.conf:/lib/route.conf \
    -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json \
    --net=host --shm-size=128g --privileged \
    cann9.2.0.pt2.13.0_dsv4.1_aarch_a5_image_custom_20260930:latest /bin/bash
```

进入容器后，将源码挂载或放置在 `/home/code/cann-recipes-infer`，并按“修改配置”和“拉起多卡推理”章节执行。容器名和镜像 tag 必须与上述命令保持一致。

### 配置样例运行环境

   修改 `executor/scripts/set_env.sh` 中的节点 IP 和 CANN 安装路径。其中 `cann_path` 需设为 `/usr/local/Ascend/cann`。

---

## 快速启动

### 下载InfiniteBench数据集
  从[链接](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/blob/main/longbook_qa_eng.jsonl)中下载长序列输入数据集longbook_qa_eng，并上传到各个节点上新建的路径`dataset/InfiniteBench`下。
  ```shell
  mkdir -p dataset/InfiniteBench
  ```
### 下载MMMU数据集
  从[链接](https://huggingface.co/datasets/MMMU/MMMU)中下载多模态理解数据集MMMU，并上传到各个节点上新建的路径`dataset/MMMU`下。

  ```shell
  mkdir -p dataset/MMMU
  ```
### 下载权重

  下载[DeepSeek-V4-1-Flash原始Hybrid FP8-MXFP4权重](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)，并上传到各节点的某个固定的路径下，比如`/data/models/deepseek_v4_1_hybrid_fp8_mxfp4`。

### 转换权重

DeepSeek-v4.1的原始权重中使用了32x32-block的量化，通过broadcast scale可以将其转化为1x32-group的标准MXFP8量化。在各个节点上进入 `models/deepseek_v4_1` 目录，使用`utils/convert_model.py` 脚本完成 Hybrid FP8-MXFP4 到 Hybrid MXFP8-MXFP4 的权重转换。

  >入参介绍：`input_fp8_hf_path`：原始权重路径；`output_hf_path`：转换后输出的权重路径；

  如果权重转换的运行环境为NPU，需要先执行：

  ```shell
  cann_path=/usr/local/Ascend/cann  # cann包安装路径
  source ${cann_path}/bin/setenv.bash
  ```

权重转换拉起示例：
```shell
python utils/convert_model.py --input_fp8_hf_path /data/models/deepseek_v4_1_hybrid_fp8_mxfp4  --output_hf_path /data/models/deepseek_v4_1_hybrid_mxfp8_mxfp4
```

### 修改配置
- 在各个节点上修改公共环境脚本 `cann-recipes-infer/executor/scripts/set_env.sh` 中的如下字段:
   - `IPs`：离线模式配置所有节点的IP，按照rank id排序，多个节点的ip通过空格分开，例如：`('xxx.xxx.xxx.xxx' 'xxx.xxx.xxx.xxx')`。
   - `cann_path`: CANN软件包安装路径，例如`/usr/local/Ascend/cann`。
- `executor/scripts/infer.sh` 会先加载公共环境脚本 `executor/scripts/set_env.sh`；然后再继续加载模型私有环境脚本 `models/deepseek_v4_1/set_env.sh`。

### DSV4.1 纯 TP 配置

DSV4.1 的基础多卡 TP 路径需要在 `parallel_config` 中同时设置以下参数：

在 `model_config.custom_params` 中开启 `low_latency_tp: true`。该开关要求 `attn_tp_size == world_size`，并自动将 MoE、O 投影、Embedding、LM Head 等 TP 度统一为 `world_size`。

| 参数 | 说明 |
| --- | --- |
| `attn_tp_size` | Attention TP 卡数 |
| `moe_tp_size` | MoE TP 卡数 |
| `o_proj_tp_size` | Attention 输出投影 TP 卡数 |

当前基础 TP 路径要求这三个值都等于 `world_size`，框架会根据 `world_size / moe_tp_size` 自动推导 `moe_ep_size=1`，只适用于纯 TP。8 卡示例：

```yaml
model_config:
  custom_params:
    low_latency_tp: true

parallel_config:
  world_size: 8
  attn_tp_size: 8
  moe_tp_size: 8
  o_proj_tp_size: 8
```

### DSV4.1 长序列 CP 配置

长序列 prefill 可以开启 Context Parallel，把一条请求切分到各卡并行计算。在 `parallel_config` 中设置 `cp_size`，框架要求 `cp_size == world_size` 且 `attn_tp_size == 1`；`scheduler_config.cp_mini_batch` 为单次 prefill 调度的请求数上限，开启 CP 时必须为正整数，取值越大需同步放大 `max_prefill_tokens`。8 卡示例：

```yaml
parallel_config:
  world_size: 8
  attn_tp_size: 1
  cp_size: 8

scheduler_config:
  cp_mini_batch: 1
```

### 模型扩展配置

除框架统一配置之外，DeepSeek-V4.1 还支持以下模型扩展参数。这些参数放置在 YAML 文件的 `model_config.custom_params` 下：

| 参数名 | 类型 | 默认值 | 含义与约束 |
| --- | --- | --- | --- |
| `low_latency_tp` | bool | `false` | 开启低时延 TP 路径。要求 `world_size` 为偶数、`attn_tp_size == world_size` 且 `attn_tp_size > 1`，仅支持 `eager` 或 `npugraph_ex`。开启后会将相关 TP 配置统一为 `world_size`。 |
| `enable_multi_streams` | bool | `false` | 开启模型内多流并行，用于 attention metadata、MoE shared expert 和 Engram 等计算的并行调度。 |
| `enable_engram_multi_stream` | bool | 跟随 `enable_multi_streams` | 单独控制 Engram 预计算是否使用独立流；未配置时继承 `enable_multi_streams` 的值。 |
| `enable_mega_moe` | bool | `false` | 开启 MegaMoE 融合路径。当前要求 `moe_ep_size > 1`、`moe_tp_size == 1`，并使用 W4A8 MXFP4 的 GMM 量化模式。 |
| `enable_superkernel` | bool | `false` | 开启 SuperKernel。不支持 `eager` 模式，需与图模式配合使用。 |
| `moe_chunk_max_len` | int | `65536` | MoE token 分发的最大 chunk 长度，必须为正整数；可用于长序列 prefill 场景降低峰值显存。 |
| `enable_engram_offload` | bool | `false` | 将 Engram Table offload 到 Host。单机多卡时需设置 `engram_tp_size == world_size`；多机时 `engram_tp_size` 至少为单机的 NPU 数。 |
| `engram_tp_size` | int | `1` | Engram Table 的 TP 并行度，必须为正数、不大于 `world_size`，且能整除 `world_size`。 |
| `unsafe_skip_npugraph_capture_validation` | bool | `false` | 跳过 `npugraph_ex` 抓图前的校验，仅支持 `exe_mode: npugraph_ex`。该选项可能隐藏抓图问题，仅用于明确了解风险的调试场景。 |
| `kernel_config` | dict | `{}` | 按算子类型选择 kernel 实现，具体取值见下表。 |

`kernel_config` 支持的配置项如下：

| 算子类型 | 可选实现 | Atlas A5 默认值 | 说明 |
| --- | --- | --- | --- |
| `hc_pre` | `ascendc` | `ascendc` | mHC 前处理。 |
| `hc_post` | `ascendc` / `tilelang` | `ascendc` | mHC 后处理。 |
| `gate_topk` | `ascendc` / `tilelang` | `ascendc` | MoE gate TopK。 |
| `indexer_prolog_qw` | `native` / `ascendc` | `ascendc` | Lightning Indexer QW 前处理。 |
| `megamoe` | `ascendc` / `deepgemm` | `ascendc` | 仅在 `enable_mega_moe: true` 时生效。 |
| `engram` | `native` / `tilelang` | `native` | TileLang 实现仅在 prefill 阶段生效，decode 仍走 native 路径。 |

### 拉起多卡推理
以下命令在仓库根目录执行。统一入口脚本位于 `executor/scripts/infer.sh`，通过以下参数控制启动：

| 参数 | 含义 | 取值示例 |
| --- | --- | --- |
| `--model` | 模型目录名，对应 `models/` 下的子目录 | `deepseek_v4_1` |
| `--mode` | 推理模式 | `offline` / `online` |
| `--yaml` | 离线模式：yaml 文件名，路径相对 `models/deepseek_v4_1/config/` | `deepseek_v4_1_flash_rank_8_8ep.yaml` |

> 在线模式 IP 等更多配置可参考 [executor 设计文档 §5.1 启动方式](../../docs/design/executor_design.md#51-启动方式)。

**使用方式一：命令行传参**
```shell
# offline 模式，Ascend 950
bash executor/scripts/infer.sh --model deepseek_v4_1 --yaml deepseek_v4_1_flash_rank_8_8ep.yaml
```

如需查看参数说明，可执行 `bash executor/scripts/infer.sh --help`。

**使用方式二：直接修改脚本默认值后执行**
编辑 `executor/scripts/infer.sh`，按需修改 `MODEL` / `MODE` / `YAML_FILE` 等参数的默认值，例如：
```shell
MODEL=deepseek_v4_1
MODE=offline
YAML_FILE=deepseek_v4_1_flash_rank_8_8ep.yaml
```
保存后直接执行：
```shell
bash executor/scripts/infer.sh
```

> 如果是多机环境，需要在每个节点上同步执行拉起命令。

> **Note：** 不同平台最小部署单元要求如下

| 平台  | 模型型号             |  推荐量化策略  | 最小部署单元（chips）|
|-------|---------------------|--------------|--------------|
| Ascend 950  | DeepSeek-V4.1 Flash    | Hybrid MXFP8-MXFP4 |8          |

## vLLM 框架 PD 分离部署

本节在两台 Atlas A5（Ascend 950）节点上部署 DeepSeek-V4.1-Flash 在线推理服务。Prefill（P）和 Decode（D）各占一台节点，Proxy 提供统一的 OpenAI 兼容接口。此部署使用 vLLM，与上文的离线推理流程分别配置。

默认通过 `MooncakeHybridConnector` 在 P/D 间传输 KV Cache，并设置 `use_ascend_direct: true`；默认不启用 Mooncake Store 池化，无需启动 Mooncake Master 或配置 `mooncake.json`。

Proxy 可以运行在 P、D 或其他能够访问两侧服务的节点上。下文的容器命令在 P/D 宿主机运行，服务命令在相应容器内运行。

### 环境准备

#### 下载并导入镜像

下载适用于 A5 的 DeepSeek-v4.1-Flash 容器镜像包，并上传到 P/D 两节点。

镜像下载链接：[vllm_cann_ds41.tar](https://cann-ai.obs.cn-north-4.myhuaweicloud.com:443/vllm-cann/DeepSeek/vllm_cann_ds41.tar?AccessKeyId=HPUAZLVQNN603FBK9RLL&Expires=1821798734&Signature=rvJOISCHtg5BA9tuHwRgDJbwEj4%3D)

下载完成并上传到两节点后，在两节点宿主机上分别执行以下命令导入镜像：

```bash
docker load -i vllm_cann_ds41.tar
```

导入后的镜像名称为 `vllm_cann:ds41`。

#### 创建容器

在 **P/D 两节点的宿主机上分别执行**以下脚本创建容器。将 `CONTAINER_NAME` 和 `IMAGE_NAME` 替换为实际容器名及支持 A5、DeepSeek-v4.1-Flash 的镜像名称，两节点使用同一版本镜像。

```bash
#!/usr/bin/env bash
set -euo pipefail

export CONTAINER_NAME="deepseek-v41-p"   # D 节点可使用 deepseek-v41-d
export IMAGE_NAME="vllm_cann:ds41"

mkdir -p /home/deploy_scripts

docker run --runtime=runc -u root -it -d --name "${CONTAINER_NAME}" --net=host --privileged=true --shm-size=2g \
    --device=/dev/davinci_manager --device=/dev/hisi_hdc --device=/dev/ummu --device=/dev/uburma \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci6 \
    --device=/dev/davinci7 \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /root/host:/root/host \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /var/log/npu/:/usr/slog \
    -v /mnt:/mnt \
    -v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json \
    -v /usr/lib64:/usr/lib64 \
    -v /usr/bin:/usr/bin \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /home/:/home/ \
    -v /etc/hixlep:/etc/hixlep \
    "${IMAGE_NAME}" \
    bash

docker exec -it "${CONTAINER_NAME}" bash
```

脚本和权重目录通过 `/home:/home` 挂载到容器中。若使用其他目录，请相应增加挂载；设备和工具路径按宿主机实际环境调整。

#### 下载权重

下载 [DeepSeek-V4.1-Flash 原始 Hybrid FP8-MXFP4 权重](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)，将完整模型文件放到两节点固定路径下，例如：

```text
/home/models/DeepSeek-V4.1-Flash
```

两节点应使用相同版本的权重、`config.json` 和 tokenizer 等配套文件。`prefill.sh` 和 `decode.sh` 已使用：

```bash
vllm serve /home/models/DeepSeek-V4.1-Flash \
  --quantization deepseek_v4_fp8 \
  ...
```

修改权重存储路径时，同时修改 P/D 模板中 `vllm serve` 后的路径。

### 脚本准备

在 P、D 节点分别将下列对应脚本保存到 `/home/deploy_scripts`；Proxy 所在节点也使用该目录。文件采用 UTF-8 编码和 Linux 换行符（LF）。两节点的模型路径、通信 IP、网卡和 LocalCommRes 配置按本机实际情况修改。公共启动器和启动脚本见下一节。

#### P 节点

将以下内容保存为 `/home/deploy_scripts/prefill.sh`。将 `NIC_NAME` 改成 P 节点通信网卡；若 `hostname -I` 的第一个地址不是 P 节点通信 IP，则直接设置 `local_ip`。

```bash
#!/usr/bin/env bash
set -euo pipefail

NIC_NAME="xxxx" # change to your own nic name
# 若命令不可用或首个 IP 不是通信网卡 IP，可直接改为 local_ip="本机通信 IP"。
local_ip=$(hostname -I | awk '{print $1}')

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export ASCEND_RT_VISIBLE_DEVICES=$1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export VLLM_ASCEND_ENABLE_PRIVATE_CIRCLE_POOL=1
# 只装可选参数，初始为空数组
ARGS=()
# 默认启用 DSpark 和 Engram，关闭 Mooncake Store。
ENABLE_DSPARK="1"
ENABLE_ENGRAM="1"
ENABLE_MOONCAKE="${ENABLE_MOONCAKE:-0}"

if [[ "$ENABLE_DSPARK" == "1" ]]; then
  ARGS+=(--speculative-config '{"num_speculative_tokens": 5,"method": "dspark","enforce_eager": true,"draft_sample_method": "probabilistic"}')
fi

if [[ "$ENABLE_ENGRAM" == "1" ]]; then
  ENABLE_ENGRAM="true"
  ENABLE_ENGRAM_OFFLOAD="true"
else
  ENABLE_ENGRAM="false"
  ENABLE_ENGRAM_OFFLOAD="false"
fi

ADDITIONAL_CONFIG=$(cat <<EOF
{
  "enable_engram": ${ENABLE_ENGRAM},
  "engram_storage": "fp8",
  "enable_engram_offload": ${ENABLE_ENGRAM_OFFLOAD},
  "engram_tp_size": 8,
  "weight_nz_mode":2,
  "enable_fused_mc2": 1,
  "enable_cpu_binding": true,
  "ascend_compilation_config": {
    "enable_npugraph_ex": false,
    "enable_static_kernel": false
  }
}
EOF
)

if [[ "$ENABLE_MOONCAKE" == "1" ]]; then
    #mooncake
    export PYTHONHASHSEED=0
    export MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_PATH:-$(pwd)/mooncake.json}"
    export ACL_OP_INIT_MODE=2
    export HCCL_RDMA_TIMEOUT=17
    export ASCEND_CONNECT_TIMEOUT=10000
    export ASCEND_TRANSFER_TIMEOUT=10000
    KV_TRANSFER_CONFIG='{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_producer",
        "kv_load_failure_policy": "recompute",
        "engine_id": "0",
        "kv_connector_extra_config": {
            "ascend_local_comm_res_path": "/etc/hixlep",
            "connectors": [
                {
                    "kv_connector": "MooncakeHybridConnector",
                    "kv_role": "kv_producer",
                    "kv_port": "30000",
                    "kv_connector_extra_config": {
                        "prefill": {
                            "dp_size": 8,
                            "tp_size": 1
                        },
                        "decode": {
                            "dp_size": 8,
                            "tp_size": 1
                        }
                    }
                },
                {
                    "kv_connector": "AscendStoreConnector",
                    "kv_role": "kv_producer",
                    "kv_connector_extra_config": {
                        "lookup_rpc_port": "10001",
                        "backend": "mooncake",
                        "load_async": true
                    }
                }
            ]
        }
    }'
else
    KV_TRANSFER_CONFIG='{
    "kv_connector": "MooncakeHybridConnector",
    "kv_role": "kv_producer",
    "kv_port": "30000",
    "engine_id": "0",
    "kv_connector_extra_config": {
                "use_ascend_direct": true,
                "prefill": {
                        "dp_size": 8,
                        "tp_size": 1
                },
                "decode": {
                        "dp_size": 8,
                        "tp_size": 1
                },
                "ascend_local_comm_res_path": "/etc/hixlep"
        }
    }'
fi

export TASK_QUEUE_ENABLE=2

if [[ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]]; then
  export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
fi

vllm serve /home/models/DeepSeek-V4.1-Flash \
  --host 0.0.0.0 \
  --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --enable-expert-parallel \
  --served-model-name deepseek-v41 \
  --max-model-len 1048576 \
  --max-num-batched-tokens 24576 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.82 \
  --block-size 128 \
  --tokenizer-mode deepseek_v41 \
  --reasoning-parser deepseek_v41 \
  --tool-call-parser deepseek_v41 \
  --enable-auto-tool-choice \
  --no-async-scheduling \
  --trust-remote-code \
  --quantization deepseek_v4_fp8 \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
  --safetensors-load-strategy lazy \
  --additional-config "$ADDITIONAL_CONFIG" \
  --kv-transfer-config "$KV_TRANSFER_CONFIG" \
  --enforce-eager \
  --profiler-config \
    '{"profiler": "torch",
    "torch_profiler_dir": "./vllm_profile",
    "torch_profiler_with_stack": false}' \
  "${ARGS[@]}"
```

P 节点默认使用 `MooncakeHybridConnector` 的 `kv_producer` 分支，传输端口为 `30000`。脚本中的 `/etc/hixlep` 必须指向本机有效的 LocalCommRes 配置。

#### D 节点

将以下内容保存为 `/home/deploy_scripts/decode.sh`。将 `NIC_NAME` 和 `local_ip` 按 D 节点实际通信网卡与 IP 修改；D 节点使用自己的 LocalCommRes 配置。

```bash
#!/usr/bin/env bash
set -euo pipefail

NIC_NAME="xxxx" # change to your own nic name
# 若命令不可用或首个 IP 不是通信网卡 IP，可直接改为 local_ip="本机通信 IP"。
local_ip=$(hostname -I | awk '{print $1}')
export HCCL_BUFFSIZE=400
export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME="$NIC_NAME"
export TP_SOCKET_IFNAME="$NIC_NAME"
export HCCL_SOCKET_IFNAME="$NIC_NAME"
export ASCEND_RT_VISIBLE_DEVICES=$1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export VLLM_ASCEND_ENABLE_PRIVATE_CIRCLE_POOL=1
# 只装可选参数，初始为空数组
ARGS=()
# 默认启用 DSpark 和 Engram，关闭 Mooncake Store。
ENABLE_DSPARK="1"
ENABLE_ENGRAM="1"
ENABLE_MOONCAKE="${ENABLE_MOONCAKE:-0}"

if [[ "$ENABLE_DSPARK" == "1" ]]; then
  ARGS+=(--speculative-config '{"num_speculative_tokens": 5,"method": "dspark"}')
fi

if [[ "$ENABLE_ENGRAM" == "1" ]]; then
  ENABLE_ENGRAM="true"
  ENABLE_ENGRAM_OFFLOAD="true"
else
  ENABLE_ENGRAM="false"
  ENABLE_ENGRAM_OFFLOAD="false"
fi

ADDITIONAL_CONFIG=$(cat <<EOF
{
  "enable_engram": ${ENABLE_ENGRAM},
  "engram_storage": "fp8",
  "enable_engram_offload": ${ENABLE_ENGRAM_OFFLOAD},
  "engram_tp_size": 8,
  "weight_nz_mode":2,
  "multistream_overlap_shared_expert":true,
  "enable_cpu_binding": true,
  "ascend_compilation_config": {
    "enable_npugraph_ex": true,
    "enable_static_kernel": true
  }
}
EOF
)

if [[ "$ENABLE_MOONCAKE" == "1" ]]; then
    #mooncake
    export PYTHONHASHSEED=0
    export MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_PATH:-$(pwd)/mooncake.json}"
    export ACL_OP_INIT_MODE=2
    export HCCL_RDMA_TIMEOUT=17
    export ASCEND_CONNECT_TIMEOUT=10000
    export ASCEND_TRANSFER_TIMEOUT=10000
    KV_TRANSFER_CONFIG='{
        "kv_connector": "MultiConnector",
        "kv_role": "kv_consumer",
        "kv_load_failure_policy": "recompute",
        "engine_id": "1",
        "kv_connector_extra_config": {
            "ascend_local_comm_res_path": "/etc/hixlep",
            "connectors": [
                {
                    "kv_connector": "MooncakeHybridConnector",
                    "kv_role": "kv_consumer",
                    "kv_port": "30100",
                    "kv_connector_extra_config": {
                        "prefill": {
                            "dp_size": 8,
                            "tp_size": 1
                        },
                        "decode": {
                            "dp_size": 8,
                            "tp_size": 1
                        }
                    }
                },
                {
                    "kv_connector": "AscendStoreConnector",
                    "kv_role": "kv_consumer",
                    "kv_connector_extra_config": {
                        "lookup_rpc_port": "10002",
                        "backend": "mooncake",
                        "load_async": true
                    }
                }
            ]
        }
    }'
else
    KV_TRANSFER_CONFIG='{
        "kv_connector": "MooncakeHybridConnector",
        "kv_role": "kv_consumer",
        "kv_port": "30100",
        "engine_id": "1",
        "kv_connector_extra_config": {
                "use_ascend_direct": true,
                "prefill": {
                        "dp_size": 8,
                        "tp_size": 1
                },
                "decode": {
                        "dp_size": 8,
                        "tp_size": 1
                },
                "ascend_local_comm_res_path": "/etc/hixlep"
        }
    }'
fi

export LOCAL_WORLD_SIZE=8
export TASK_QUEUE_ENABLE=1

if [[ -f /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ]]; then
  export LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2${LD_PRELOAD:+:$LD_PRELOAD}"
fi

vllm serve /home/models/DeepSeek-V4.1-Flash \
  --host 0.0.0.0 \
  --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --enable-expert-parallel \
  --served-model-name deepseek-v41 \
  --max-model-len 1048576 \
  --max-num-batched-tokens 255 \
  --max-num-seqs 12 \
  --async-scheduling \
  --gpu-memory-utilization 0.85 \
  --block-size 128 \
  --tokenizer-mode deepseek_v41 \
  --reasoning-parser deepseek_v41 \
  --tool-call-parser deepseek_v41 \
  --enable-auto-tool-choice \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --quantization deepseek_v4_fp8 \
  --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":128}' \
  --safetensors-load-strategy lazy \
  --additional-config "$ADDITIONAL_CONFIG" \
  --kv-transfer-config "$KV_TRANSFER_CONFIG" \
  --profiler-config \
    '{"profiler": "torch",
    "torch_profiler_dir": "./vllm_profile",
    "torch_profiler_with_stack": true}' \
  "${ARGS[@]}"
```

D 节点默认使用 `MooncakeHybridConnector` 的 `kv_consumer` 分支，传输端口为 `30100`。P/D 模板中的 `prefill.dp_size`、`decode.dp_size` 均为 8，`tp_size` 均为 1；调整并行策略时应同步修改两侧配置和 Proxy 后端数量。

#### /etc/hixlep配置

按照 [A5 LocalCommRes 配置指南](https://gitcode.com/cann/hixl/wiki/A5%20LocalCommRes%E9%85%8D%E7%BD%AE%E6%8C%87%E5%8D%97.md) 在各节点准备与本机设备和网络拓扑匹配的配置。容器命令已挂载 `/etc/hixlep`，两侧模板的默认分支和池化分支都使用此路径。不要直接复制另一节点的设备资源文件。P/D 所用 DP RPC、KV 传输和 HIXL 端口均需互通；`30000/30100` 并非全部通信端口。

#### Proxy 节点

Proxy 可以与 P 或 D 共用容器；独立部署时需要可访问 P/D 后端的 Python 环境，包含 `fastapi`、`httpx`、`uvicorn`。从 [load_balance_proxy_server_example.py](https://github.com/vllm-project/vllm-ascend/blob/main/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py) 获取脚本，保存为 `/home/deploy_scripts/load_balance_proxy_server_example.py`。

Proxy 的 P/D 后端地址由下一节的 `start_proxy.sh` 读取 `P_IP`、`D_IP`。八个 P 后端使用 `7100–7107`，八个 D 后端使用 `7200–7207`，请确保 Proxy 可以访问这些 HTTP 端口。独立 Proxy 节点也需将上述 Python 文件和启动脚本放在同一目录。

### 服务启动与验证

P/D 各自构成一个 DP 组，两侧 rank 均从 `0` 开始。P 侧 RPC 端口为 `12321`，HTTP 端口为 `7100–7107`；D 侧 RPC 端口为 `12322`，HTTP 端口为 `7200–7207`。启动前请检查 P/D 启动脚本中的 `local_ip`，以及模板中的模型路径和网卡。

#### 公共启动器

在 P、D 节点各保存一份 `/home/deploy_scripts/launch_online_dp.py`。它根据 `--role` 调用当前目录的 `prefill.sh` 或 `decode.sh`，为每个本地 DP rank 分配设备、HTTP 端口和 rank。

```python
import argparse
import multiprocessing
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dp-size",
        type=int,
        required=True,
        help="Data parallel size."
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size."
    )
    parser.add_argument(
        "--dp-size-local",
        type=int,
        default=-1,
        help="Local data parallel size."
    )
    parser.add_argument(
        "--dp-rank-start",
        type=int,
        default=0,
        help="Starting rank for data parallel."
    )
    parser.add_argument(
        "--dp-address",
        type=str,
        required=True,
        help="IP address for data parallel master node."
    )
    parser.add_argument(
        "--dp-rpc-port",
        type=str,
        default="12345",
        help="Port for data parallel master node."
    )
    parser.add_argument(
        "--vllm-start-port",
        type=int,
        default=9000,
        help="Starting port for the engine."
    )
    # 新增 --role 参数，默认为 prefill
    parser.add_argument(
        "--role",
        type=str,
        default="prefill",
        choices=["prefill", "decode"],
        help="Role type: 'prefill' uses prefill.sh, 'decode' uses decode.sh."
    )
    parser.add_argument("--device-start", type=int, default=0)
    return parser.parse_args()


args = parse_args()
dp_size = args.dp_size
tp_size = args.tp_size
dp_size_local = args.dp_size_local
if dp_size_local == -1:
    dp_size_local = dp_size
dp_rank_start = args.dp_rank_start
dp_address = args.dp_address
dp_rpc_port = args.dp_rpc_port
vllm_start_port = args.vllm_start_port
role = args.role
device_start = args.device_start

# 根据 role 动态选择模板脚本
if role == "prefill":
    template_path = "./prefill.sh"
else:
    template_path = "./decode.sh"


def run_command(visible_devices, dp_rank, vllm_engine_port):
    command = [
        "bash",
        template_path,
        visible_devices,
        str(vllm_engine_port),
        str(dp_size),
        str(dp_rank),
        dp_address,
        dp_rpc_port,
        str(tp_size),
    ]
    print(f"command is {command}")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    if not os.path.exists(template_path):
        print(f"Template file {template_path} does not exist.")
        sys.exit(1)

    processes = []
    num_cards = dp_size_local * tp_size
    for i in range(dp_size_local):
        dp_rank = dp_rank_start + i
        vllm_engine_port = vllm_start_port + i
        visible_devices = ",".join(str(x) for x in range(device_start + i * tp_size, device_start + (i + 1) * tp_size))
        process = multiprocessing.Process(
            target=run_command,
            args=(visible_devices, dp_rank, vllm_engine_port)
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
```

#### 1. 启动 P 节点

将以下脚本保存为 P 节点的 `/home/deploy_scripts/start_p.sh`。

```bash
mkdir -p ./log
# 若命令不可用或首个 IP 不是通信网卡 IP，可直接改为 local_ip="本机通信 IP"。
local_ip=$(hostname -I | awk '{print $1}')
nohup python launch_online_dp.py \
      --dp-size 8 \
      --tp-size 1 \
      --dp-size-local 8 \
      --dp-rank-start 0 \
      --dp-address $local_ip \
      --dp-rpc-port 12321 \
      --vllm-start-port 7100 > ./log/prefill.log 2>&1 < /dev/null &
```

在 P 节点容器内执行：

```bash
cd /home/deploy_scripts
bash start_p.sh
tail -f log/prefill.log
```

#### 2. 启动 D 节点

将以下脚本保存为 D 节点的 `/home/deploy_scripts/start_d.sh`。

```bash
mkdir -p ./log
# 若命令不可用或首个 IP 不是通信网卡 IP，可直接改为 local_ip="本机通信 IP"。
local_ip=$(hostname -I | awk '{print $1}')
nohup python launch_online_dp.py \
       --dp-size 8 \
       --tp-size 1 \
       --dp-size-local 8 \
       --dp-rank-start 0 \
       --dp-address $local_ip \
       --dp-rpc-port 12322 \
       --vllm-start-port 7200 \
       --device-start 0 \
       --role decode > ./log/decode.log 2>&1 < /dev/null &
```

在 D 节点容器内执行：

```bash
cd /home/deploy_scripts
bash start_d.sh
tail -f log/decode.log
```

P/D 可以并行启动。待模型加载及 D 侧图编译完成，并确认日志无启动异常后，再启动 Proxy。启动脚本返回只表示已提交后台进程，不代表服务就绪。

#### 3. 启动 Proxy

将以下脚本保存为 Proxy 所在节点的 `/home/deploy_scripts/start_proxy.sh`。

```bash
mkdir -p ./log
unset no_proxy https_proxy HTTPS_PROXY HTTP_PROXY  http_proxy
node_p0_ip="${P_IP:?请设置 P_IP 为 Prefill 节点 IP}"
node_d0_ip="${D_IP:?请设置 D_IP 为 Decode 节点 IP}"
proxy_host="${PROXY_HOST:-0.0.0.0}"
nohup python load_balance_proxy_server_example.py \
  --port 1999 \
  --host "$proxy_host" \
  --prefiller-hosts \
    $node_p0_ip \
    $node_p0_ip \
    $node_p0_ip \
    $node_p0_ip \
    $node_p0_ip \
    $node_p0_ip \
    $node_p0_ip \
    $node_p0_ip \
  --prefiller-ports  \
    7100 7101 7102 7103 7104 7105 7106 7107 \
  --decoder-hosts \
    $node_d0_ip \
    $node_d0_ip \
    $node_d0_ip \
    $node_d0_ip \
    $node_d0_ip \
    $node_d0_ip \
    $node_d0_ip \
    $node_d0_ip \
  --decoder-ports  \
    7200 7201 7202 7203 7204 7205 7206 7207 > ./log/proxy.log 2>&1 < /dev/null &
```

在 Proxy 所在节点容器内执行，将两个地址替换为实际 IP。`PROXY_HOST` 默认监听 `0.0.0.0:1999`；客户端应使用 Proxy 节点的实际 IP。

```bash
cd /home/deploy_scripts
export P_IP="<P 节点 IP>"
export D_IP="<D 节点 IP>"
bash start_proxy.sh
tail -f log/proxy.log
```

#### 验证服务

在能够访问 Proxy 的终端中执行：

```bash
export PROXY_IP="<Proxy 所在节点 IP>"

curl --noproxy '*' --fail --show-error "http://${PROXY_IP}:1999/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v41",
    "messages": [{"role": "user", "content": "请简要介绍一下你自己。"}],
    "max_tokens": 256,
    "temperature": 0,
    "stream": false
  }'
```

预期请求返回 HTTP 200，JSON 中包含 `choices`。同时检查 `log/prefill.log`、`log/decode.log` 和 `log/proxy.log`，确认请求完成 P/D 协作。

### 开启 Mooncake Store 池化

#### 能力介绍

Mooncake Store 将 KV Cache 存入共享缓存池，供后续请求复用，以减少重复 Prefill 计算。开启后使用 `MultiConnector` 组合两个连接器：`MooncakeHybridConnector` 负责 P/D 间 KV 传输，`AscendStoreConnector` 使用 Mooncake 作为池化后端。Mooncake Master 负责 Store 管理，Proxy 仍负责推理请求分发。

能力及配置说明参见 [vLLM Ascend Mooncake KV 池化指南](https://docs.vllm.ai/projects/ascend/zh-cn/v0.26.0rc1/user_guide/feature_guide/kv_pool.html)。

#### 配置 Mooncake Store

在默认部署配置完成后，再进行以下设置。若已有不启用 Store 的服务在运行，需先停止已有 P/D 服务和 Proxy，再按本节顺序重新启动，避免设备和端口占用。

1. **在 P/D 各节点新增 `mooncake.json`。** 在各节点的 `/home/deploy_scripts` 目录下创建文件，内容如下：

   ```json
   {
     "local_hostname": "<当前节点 IP>",
     "metadata_server": "P2PHANDSHAKE",
     "protocol": "ascend",
     "use_ascend_direct": true,
     "device_name": "",
     "master_server_address": "<Master IP>:50088",
     "global_segment_size": "64GB",
     "enable_ssd_offload": false
   }
   ```

   `local_hostname` 建议填写当前节点的实际 IP：P 节点填写 P 节点 IP，D 节点填写 D 节点 IP。将 `<Master IP>` 替换为 Mooncake Master 所在节点的实际 IP。Master 可以部署在 P 或 D 节点，只需启动一个；两节点的 `master_server_address` 必须指向同一个 Master，端口与 `start_mooncake_master.sh` 中的 `50088` 一致。

   `global_segment_size` 是每个贡献内存的 worker 注册的池容量，需要按 worker 数量估算宿主机内存用量；示例 `64GB` 不是整机总容量，实际值应根据可用内存调整并按 1GB 对齐。`enable_ssd_offload` 设置为 `false`，表示不开启 SSD offload。

2. **开启 P/D 模板的池化分支。** 在两节点启动服务的容器终端中分别设置：

   ```bash
   cd /home/deploy_scripts
   export ENABLE_MOONCAKE=1
   export MOONCAKE_CONFIG_PATH="/home/deploy_scripts/mooncake.json"
   ```

   `prefill.sh`、`decode.sh` 会继承该开关；未指定配置路径时，默认读取启动目录下的 `mooncake.json`。JSON 中的 `<当前节点 IP>` 和 `<Master IP>` 需直接改为实际值，不会自动展开 Shell 变量。

池化分支的配置关系如下：

| 配置项 | Prefill | Decode |
| --- | --- | --- |
| 顶层 `kv_connector` | `MultiConnector` | `MultiConnector` |
| 顶层及子连接器 `kv_role` | `kv_producer` | `kv_consumer` |
| 顶层 `engine_id` | `0` | `1` |
| 传输子连接器 | `MooncakeHybridConnector`，`kv_port=30000` | `MooncakeHybridConnector`，`kv_port=30100` |
| 池化子连接器 | `AscendStoreConnector` | `AscendStoreConnector` |
| 池化 `backend` / `load_async` | `mooncake` / `true` | `mooncake` / `true` |
| 池化 `lookup_rpc_port` | `10001` | `10002` |

池化分支将 `ascend_local_comm_res_path` 放在顶层 `kv_connector_extra_config` 中，该路径仍需指向本节点有效的 LocalCommRes 配置。

两节点池化分支均设置 `PYTHONHASHSEED=0`，以保持缓存键哈希一致；同时设置 `ACL_OP_INIT_MODE=2`、`HCCL_RDMA_TIMEOUT=17`，以及 `ASCEND_CONNECT_TIMEOUT=10000`、`ASCEND_TRANSFER_TIMEOUT=10000`。脚本中的 `lookup_rpc_port` 用于构造本地 IPC 标识，并结合 DP rank 区分实例，不是对外的 HTTP 端口。

池化分支配置了 `kv_load_failure_policy: "recompute"`。

#### Mooncake Master 启动脚本

在选定的 Master 节点，将以下内容保存为 `/home/deploy_scripts/start_mooncake_master.sh`。P 或 D 中只需选择一台作为 Master；两侧 `mooncake.json` 的 `master_server_address` 必须指向同一台的 `50088` 端口。

```bash
mkdir -p ./log
nohup mooncake_master --port 50088 \
  --eviction_high_watermark_ratio 0.9 \
  --eviction_ratio 0.2 \
  --default_kv_lease_ttl 11000 \
  --client_ttl=120 \
  --enable_offload=false > ./log/mooncake_master.log 2>&1 < /dev/null &
```

#### 启动顺序

**开启 Store 时的顺序为：启动前整理内存 → 启动 Mooncake Master → 启动 P/D → 启动 Proxy。**

1. 建议在 **P/D 两节点宿主机上、服务启动前** 使用 root 执行内存整理，以释放页缓存并整理内存碎片：

   ```bash
   echo 3 > /proc/sys/vm/drop_caches
   echo 1 > /proc/sys/vm/compact_memory
   ```

   这些命令作用于宿主机，不是容器私有内存。应在模型和 Store 分配内存前执行；它们不能代替足够的可用物理内存。

2. 在选定的 Master 节点容器内执行一次：

   ```bash
   cd /home/deploy_scripts
   bash start_mooncake_master.sh
   tail -f log/mooncake_master.log
   ```

   脚本监听 `50088`，配置驱逐高水位 `0.9`、驱逐比例 `0.2`、默认 KV 租约 `11000 ms`、客户端 TTL `120 s`，并关闭 SSD offload。确认 Master 启动成功，且 P/D 能访问其配置地址后继续。

3. P 节点容器内执行：

   ```bash
   cd /home/deploy_scripts
   export ENABLE_MOONCAKE=1
   export MOONCAKE_CONFIG_PATH="/home/deploy_scripts/mooncake.json"
   bash start_p.sh
   ```

4. D 节点容器内执行：

   ```bash
   cd /home/deploy_scripts
   export ENABLE_MOONCAKE=1
   export MOONCAKE_CONFIG_PATH="/home/deploy_scripts/mooncake.json"
   bash start_d.sh
   ```

5. 待 P/D 全部服务就绪，在任意能够访问 P/D 后端的节点容器内启动 Proxy：

   ```bash
   cd /home/deploy_scripts
   export P_IP="<P 节点 IP>"
   export D_IP="<D 节点 IP>"
   bash start_proxy.sh
   ```

按“验证服务”章节发送推理请求。进一步验证池化时，可连续发送具有相同长前缀的请求，并结合 Store 初始化、KV 保存及加载相关日志或当前版本提供的缓存统计确认实际复用情况；仅返回正常文本不能证明缓存命中。

#### 恢复默认模式

停止当前 P/D 服务与 Proxy，在 P/D 启动终端中分别设置 `export ENABLE_MOONCAKE=0`，然后按默认模式重新启动 P/D 和 Proxy 即可。默认模式不依赖 Mooncake Master。
