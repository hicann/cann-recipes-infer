# DeepSeek-V4 AFD 部署与启动指南

本文面向在昇腾 NPU 上运行 DeepSeek-V4 Attention-FFN（AF）分离服务的部署人员，给出交付配置下的同步和异步启动方式。同步方案使用 `P2pHcclAFDConnector`，异步方案使用 `WindowAFDConnector`。dsv4 afd-plugin PR 地址：https://github.com/vllm-project/afd-plugin/pull/408

两种方案的设计见[《DeepSeek-V4 Attention-FFN 分离方案》](../../../docs/models/deepseek_v4/deepseek_v4_afd_guide.md)。本指南只关注环境准备、服务启动参数和最小请求验证；性能数据、内存布局和算子细节不在本指南中展开。

## 1. 配置概览

| 方案 | 拓扑  | 执行配置 |
| --- | ---  | --- |
| 同步 | 单机 A5，4A2F；Attention 使用 NPU 0–3，FFN 使用 NPU 4–5  | U2、`FULL_DECODE_ONLY`、DSpark |
| 异步 | 双机 A5，2A14F；Attention 所在节点 2 卡，FFN 使用 6+8 卡  | U2、`FULL_DECODE_ONLY`、DSpark |

## 2. 版本、镜像和代码

| 组件 | 同步基线 | 异步基线 |
| --- | --- | --- |
| vLLM | `0.23.0` | `0.23.0` |
| vLLM-Ascend | `rfc/vllm_cann`，`11ee45653b1` | `rfc/vllm_cann`，`11ee45653b1` |
| afd-plugin | `a3fa2c7` | ``a3fa2c7`` |

### 2.1 镜像

[镜像下载链接](https://cann-ai.obs.cn-north-4.myhuaweicloud.com:443/afd/vllm_ascend_dsv4_afd_950_cann9.3.0_py3.12_aarch_image_20260929.tar.gz?AccessKeyId=HPUAZLVQNN603FBK9RLL&Expires=1821854646&Signature=BIKj/vXcRVte5pZLO%2B6eqF7j6Bw%3D)

```bash
docker load -i vllm_ascend_dsv4_afd_950_cann9.3.0_py3.12_aarch_image_20260929.tar.gz
```

### 2.2 启动容器

宿主机路径和设备节点应按实际环境调整。

```bash
docker run --runtime=runc -u root -it -d --name vllm-ascend-afd --net=host --privileged=true --shm-size=2g \
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
-v /root/host:/root/host  \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /usr/local/sbin:/usr/local/sbin \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /var/log/npu/:/usr/slog \
-v /mnt:/mnt \
-v /etc/hccl_rootinfo.json:/etc/hccl_rootinfo.json  \
-v /usr/lib64:/usr/lib64   \
-v /usr/bin:/usr/bin \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
-v /home/:/home/ \
-v /etc/hixlep:/etc/hixlep \
vllm-ascend-afd:v1.1 bash

docker exec -it vllm-ascend-afd bash
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash
```

### 2.3 编译安装环境
容器中已经安装了对应cann包，以及vllm和vllm-ascend等相关依赖包，需要手动编译安装afd-plugin插件。

```bash
git clone https://github.com/vllm-project/afd-plugin
cd afd-plugin
git fetch origin pull/408/head:pr-408
git checkout pr-408
AFD_BUILD_ASCEND_OPS=0 SOC_VERSION=ascend950dt_9582 \
  python -m pip install -v --no-build-isolation --no-deps -e .
```

## 3. 同步方案服务拉起
同步方案的 Attention 和 FFN 按相同的 layer/microbatch 顺序推进。客户端只访问 Attention 的 `8910` 端口，FFN 进程只由 Connector 驱动。

下面的 A4F2 命令按 `P2pHcclAFDConnector` recipe 脚本和当前 DeepSeek-V4 功能校验整理，组合了 U2、`FULL_DECODE_ONLY` 和 DSpark；如果使用的插件提交早于该组合支持，请先切换到版本表中的同步基线。

### 3.1 公共环境变量

```bash
export MODEL_PATH=/home/models/DeepSeek-V4-Flash-DSpark
export AFD_HOST=127.0.0.1
export AFD_PORT=29761
export MAX_MODEL_LEN=4096
export MAX_NUM_SEQS=8
export VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,afd
```

同步方案为单机部署，AFD 控制面使用 `127.0.0.1`。

### 3.2 Attention 启动命令

```bash
HCCL_BUFFSIZE=1024 \
VLLM_PLUGINS="$VLLM_PLUGINS" \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
HCCL_IF_BASE_PORT=51000 \
python -m vllm.entrypoints.openai.api_server \
--model "$MODEL_PATH" --port 8910 --served-model-name dsv4-afd \
--max-model-len "$MAX_MODEL_LEN" --max-num-batched-tokens 4096 \
--max-num-seqs "$MAX_NUM_SEQS" --data-parallel-size 4 \
--tensor-parallel-size 1 --enable-expert-parallel \
--tokenizer-mode deepseek_v4 --safetensors-load-strategy lazy \
--block-size 128 --no-enable-prefix-caching \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32]}' \
--enable-dbo --dbo-decode-token-threshold 2 --dbo-prefill-token-threshold 12 \
--speculative-config '{"num_speculative_tokens":5,"method":"mtp","enforce_eager":false}' \
--additional-config "{\"afd\":{\"role\":\"attention\",\"connector\":\"P2pHcclAFDConnector\",\"host\":\"${AFD_HOST}\",\"port\":${AFD_PORT},\"num_attention_ranks\":4,\"num_ffn_ranks\":2}}"
```

### 3.3 FFN 启动命令

```bash
HCCL_BUFFSIZE=2048 \
VLLM_PLUGINS="$VLLM_PLUGINS" \
ASCEND_RT_VISIBLE_DEVICES=4,5 \
HCCL_IF_BASE_PORT=52000 \
python -m vllm.entrypoints.openai.api_server \
--model "$MODEL_PATH" --port 8911 --served-model-name dsv4-afd-ffn \
--max-model-len "$MAX_MODEL_LEN" --max-num-batched-tokens 8192 \
--max-num-seqs "$MAX_NUM_SEQS" --data-parallel-size 2 \
--tensor-parallel-size 1 --enable-expert-parallel \
--tokenizer-mode deepseek_v4 --safetensors-load-strategy lazy \
--block-size 128 --no-enable-prefix-caching \
--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32]}' \
--enable-dbo --dbo-decode-token-threshold 2 --dbo-prefill-token-threshold 12 \
--additional-config "{\"afd\":{\"role\":\"ffn\",\"connector\":\"P2pHcclAFDConnector\",\"host\":\"${AFD_HOST}\",\"port\":${AFD_PORT},\"num_attention_ranks\":4,\"num_ffn_ranks\":2}}"
```

先启动 FFN，再启动 Attention。两侧都使用 `--enable-dbo`，因此运行时采用两个 ubatch；`FULL_DECODE_ONLY` 表示只对 decode 形状进行 ACLGraph capture/replay。DSpark 的 draft/proposer 只在 Attention 服务中启用，FFN 命令不能添加 `--speculative-config`。

## 4. 异步方案服务拉起

下面是双机 A5 的最终 2A14F 配置。Attention 节点使用 2 张卡；FFN 的 14 个 rank 分布在第一节点 6 张卡和第二节点 8 张卡。FFN 第二节点使用 `--headless`，不提供用户请求入口。

### 4.1 公共环境变量

在每个启动终端中先设置以下变量，两台节点使用相同的对应关系：`NODE1_IP` 为第一节点（2A+6F）的可达 IP，`NODE2_IP` 为第二节点（8F）的可达 IP。下方命令中的 `VLLM_HOST_IP` 和 `HCCL_IF_IP` 使用各自节点的 IP；`afd.host` 和 FFN 的 `--data-parallel-address` 均指向 `NODE1_IP`。命令中的网卡名 `enp34s0f1` 按当前节点的实际网卡修改。

```bash
export NODE1_IP=<节点1的ip地址>
export NODE2_IP=<节点2的ip地址>
```

以下 JSON 中的 `'"$NODE1_IP"'` 用于让 Bash 展开变量，同时保留 JSON 字符串所需的双引号。

### 4.2 第一节点：Attention（2A）

```bash
# 2A
HCCL_BUFFSIZE=1024 VLLM_HOST_IP="$NODE1_IP" HCCL_IF_IP="$NODE1_IP" GLOO_SOCKET_IFNAME="enp34s0f1" HCCL_SOCKET_IFNAME="enp34s0f1" TP_SOCKET_IFNAME="enp34s0f1" VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,afd ASCEND_RT_VISIBLE_DEVICES=0,1 HCCL_IF_BASE_PORT=51000 python -m vllm.entrypoints.openai.api_server \
  --model /home/models/DeepSeek-V4-Flash-DSpark \
  --port 8900 \
  --served-model-name dsv4-afd \
  --max-model-len 2048 \
  --max-num-batched-tokens 192 \
  --max-num-seqs 32 \
  --data-parallel-size 2 \
  --enable-expert-parallel \
  --tokenizer-mode deepseek_v4 \
  --safetensors-load-strategy lazy \
  --block-size 128 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32]}' \
  --enable-dbo \
  --speculative-config '{"num_speculative_tokens": 5, "method": "mtp","enforce_eager": false}' \
  --additional-config '{
    "afd": {
      "role": "attention",
      "connector": "WindowAFDConnector",
      "host": "'"$NODE1_IP"'",
      "port": 29761,
      "num_attention_ranks": 2,
      "num_ffn_ranks": 14,
      "compute_gate_on_attention": true,
      "async": true,
      "connector_extra_config": {
        "micro_batch_num": 2,
        "quant_mode": 0
      }
    }
  }'
```

### 4.3 第一节点：FFN（6F）

```bash
# 6F
HCCL_BUFFSIZE=1024 VLLM_HOST_IP="$NODE1_IP" HCCL_IF_IP="$NODE1_IP" GLOO_SOCKET_IFNAME="enp34s0f1" HCCL_SOCKET_IFNAME="enp34s0f1" TP_SOCKET_IFNAME="enp34s0f1" VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,afd ASCEND_RT_VISIBLE_DEVICES=2,3,4,5,6,7 HCCL_IF_BASE_PORT=52000 python -m vllm.entrypoints.openai.api_server \
  --model /home/models/DeepSeek-V4-Flash-DSpark \
  --port 8911 \
  --served-model-name dsv4-afd-ffn \
  --max-model-len 2048 \
  --max-num-batched-tokens 192 \
  --max-num-seqs 32 \
  --data-parallel-size 14 \
  --data-parallel-size-local 6 \
  --data-parallel-address "$NODE1_IP" \
  --data-parallel-rpc-port 13345 \
  --enable-expert-parallel \
  --tokenizer-mode deepseek_v4 \
  --safetensors-load-strategy lazy \
  --block-size 128 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32]}' \
  --enable-dbo \
  --additional-config '{
    "afd": {
      "role": "ffn",
      "connector": "WindowAFDConnector",
      "host": "'"$NODE1_IP"'",
      "port": 29761,
      "num_attention_ranks": 2,
      "num_ffn_ranks": 14,
      "compute_gate_on_attention": true,
      "async": true,
      "connector_extra_config": {
        "micro_batch_num": 2,
        "quant_mode": 0
      }
    }
  }'
```

### 4.4 第二节点：FFN（8F）

```bash
# 8F
HCCL_BUFFSIZE=1024 VLLM_HOST_IP="$NODE2_IP" HCCL_IF_IP="$NODE2_IP" GLOO_SOCKET_IFNAME="enp34s0f1" HCCL_SOCKET_IFNAME="enp34s0f1" TP_SOCKET_IFNAME="enp34s0f1" VLLM_PLUGINS=ascend,ascend_model,ascend_model_loader,ascend_kv_connector,afd ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 HCCL_IF_BASE_PORT=52000 vllm serve /home/models/DeepSeek-V4-Flash-DSpark \
  --served-model-name dsv4-afd-ffn \
  --max-model-len 2048 \
  --max-num-batched-tokens 192 \
  --max-num-seqs 32 \
  --data-parallel-size 14 \
  --data-parallel-size-local 8 \
  --data-parallel-start-rank 6 \
  --data-parallel-address "$NODE1_IP" \
  --data-parallel-rpc-port 13345 \
  --headless \
  --enable-expert-parallel \
  --tokenizer-mode deepseek_v4 \
  --safetensors-load-strategy lazy \
  --block-size 128 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32]}' \
  --enable-dbo \
  --additional-config '{
    "afd": {
      "role": "ffn",
      "connector": "WindowAFDConnector",
      "host": "'"$NODE1_IP"'",
      "port": 29761,
      "num_attention_ranks": 2,
      "num_ffn_ranks": 14,
      "compute_gate_on_attention": true,
      "async": true,
      "connector_extra_config": {
        "micro_batch_num": 2,
        "quant_mode": 0
      }
    }
  }'
```

异步方案先启动两个 FFN 服务，再启动 Attention 服务。三个进程中的 `num_attention_ranks`、`num_ffn_ranks`、Window 端口、`micro_batch_num` 和 Graph 模式必须一致；请求只发送到 Attention 的 `8900` 端口。

## 5. 启动参数说明

| 参数 | 同步方案 | 异步方案 | 作用 |
| --- | --- | --- | --- |
| `--model` | DSpark 权重路径 | DSpark 权重路径 | 指定两侧加载的同一套 DeepSeek-V4 权重；权重目录必须能被对应角色读取。 |
| `--port` | Attention=`8910`，FFN=`8911` | Attention=`8900`，第一台 FFN=`8911`；第二台 FFN 使用 `--headless` | API 服务端口；用户请求只发送到 Attention。 |
| `--served-model-name` | `dsv4-afd` / `dsv4-afd-ffn` | `dsv4-afd` / `dsv4-afd-ffn` | 注册服务名；请求中的 `model` 使用 Attention 服务名。 |
| `--max-model-len` | `4096` | `2048` | 服务允许的最大上下文长度；需要同时满足模型 KV Cache 和显存容量。 |
| `ASCEND_RT_VISIBLE_DEVICES` | `0,1,2,3` / `4,5` | `0,1` / `2,3,4,5,6,7` / 第二节点 `0..7` | 指定当前启动命令可见的 NPU；本指南 TP=1，可见卡数对应该角色在当前节点的本地 DP rank 数，跨节点 FFN 分别为 6 和 8，而非 DP 总数 14。 |
| `HCCL_IF_BASE_PORT` | Attention=`51000`，FFN=`52000` | Attention=`51000`，FFN=`52000` | 为不同角色分开分配 HCCL 端口范围，避免通信组冲突。 |
| `NODE1_IP`、`NODE2_IP` | 不使用 | 第一节点（2A+6F）、第二节点（8F）的 IP | 启动命令引用的 Shell 变量，两台节点填写相同的对应值。 |
| `VLLM_HOST_IP`、`HCCL_IF_IP` | 不显式设置 | 第一节点使用 `$NODE1_IP`，第二节点使用 `$NODE2_IP` | 分别指定 vLLM 和 HCCL 使用的本机通信 IP。 |
| `GLOO_SOCKET_IFNAME`、`HCCL_SOCKET_IFNAME`、`TP_SOCKET_IFNAME` | 不显式设置 | 示例为 `enp34s0f1` | 指定控制面、HCCL 和 TP 初始化使用的网卡，按各节点实际网卡名修改。 |
| `VLLM_PLUGINS` | 包含 `afd` | 包含 `afd` | 加载 vLLM-Ascend 和 afd-plugin 的插件入口。 |
| `--data-parallel-size` | Attention=4，FFN=2 | Attention=2，FFN 总数=14 | 当前服务角色的 DP rank 数；异步 FFN 通过 `--data-parallel-size-local` 和 `--data-parallel-start-rank` 分布到两台机器。 |
| `--data-parallel-size-local` | 不使用 | 6F=`6`，8F=`8` | 当前节点承载的 FFN DP rank 数。 |
| `--data-parallel-start-rank` | 不使用 | 8F=`6` | 第二台 FFN 节点的全局 DP 起始 rank。 |
| `--data-parallel-address`、`--data-parallel-rpc-port` | 不使用 | `${NODE1_IP}:13345` | FFN 多节点 DP worker 发现和控制通信地址，统一指向第一节点（2A+6F）。 |
| `--tensor-parallel-size` | 1 | 1（默认） | 单个 DP rank 内的 TP 规模；本指南拓扑不使用 TP 切分。 |
| `--enable-expert-parallel` | 启用 | 启用 | 将 MoE 专家按 FFN DP rank 分布。 |
| `--max-num-batched-tokens` | Attention=4096，FFN=8192 | 两侧=192 | 本地 scheduler/Window 的 token 容量；异步两侧必须一致。 |
| `--max-num-seqs` | 8 | 32 | 单次调度允许的最大序列数。 |
| `--block-size` | 128 | 128 | KV Cache block 大小。 |
| `--tokenizer-mode deepseek_v4` | 启用 | 启用 | 使用 DeepSeek-V4 tokenizer 和模型适配。 |
| `--safetensors-load-strategy lazy` | 启用 | 启用 | 以 lazy 方式加载 safetensors，降低启动阶段峰值内存。 |
| `--no-enable-prefix-caching` | 启用 | 不使用 | 同步基线关闭 prefix caching；异步最终命令按验证版本保留默认配置。 |
| `--compilation-config` | `FULL_DECODE_ONLY`，capture `[1,2,4,8,16,32]` | 同左 | 只对 decode 形状进行 ACLGraph capture/replay。 |
| `--enable-dbo` | 启用 | 启用 | 打开 vLLM DBO/ubatching；与同步的 U2 调度或异步的 `micro_batch_num=2` 配套。 |
| `--dbo-decode-token-threshold`、`--dbo-prefill-token-threshold` | `2`、`12` | 使用版本默认值 | 同步 recipe 用这两个阈值触发两个 decode/prefill ubatch；异步最终命令保留参考 README 的默认配置。 |
| `--speculative-config` | 仅 Attention，DSpark 的 `num_speculative_tokens=5` | 仅 Attention，同左 | 启用 DSpark draft/proposer；FFN 不配置该参数。 |
| `connector` | `P2pHcclAFDConnector` | `WindowAFDConnector` | 选择通信和调度协议。 |
| `afd.role` | `attention` / `ffn` | `attention` / `ffn` | 指定当前进程属于 Attention 还是 FFN。 |
| `afd.host`、`afd.port` | `127.0.0.1:29761` | `${NODE1_IP}:29761` | AFD 控制面 rendezvous 地址；异步所有角色必须使用第一节点（2A+6F）的可达 IP。 |
| `afd.num_attention_ranks`、`afd.num_ffn_ranks` | `4`、`2` | `2`、`14` | 告知 Connector 全局 A/F rank 数，用于建立配对或 Window 全局 rank 空间。 |
| `compute_gate_on_attention` | 默认关闭 | 两侧均为 `true` | Window 路径要求 Attention 生成专家路由信息；P2P 路径由 FFN 保留 Gate。 |
| `async` | 不配置 | `true` | 打开 Window 的异步执行协议。 |
| `connector_extra_config.micro_batch_num` | 不配置，U2 由 DBO 驱动 | `2` | Window 的物理 microbatch 数；必须和运行时 ubatch 数一致。 |
| `connector_extra_config.quant_mode` | 不配置 | `0` | Window 量化数据格式选择，必须和已验证权重及算子包匹配。 |
| `HCCL_BUFFSIZE` | `1024`（FFN=`2048`） | `1024` | HCCL 通信缓冲区大小，需与目标机器和已验证运行栈匹配。 |
| `--headless` | 不使用 | 仅第二台 FFN 使用 | 让远端 FFN 作为 DP worker 加入已有服务，不创建客户端 API。 |

## 6. 请求验证

服务就绪后，只向 Attention 发送请求。同步入口为 `8910`，异步入口为 `8900`：

```bash
curl -fsS http://127.0.0.1:8910/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv4-afd","messages":[{"role":"user","content":"请用三句话介绍杭州。"}],"temperature":0,"max_tokens":64}'

curl -fsS http://127.0.0.1:8900/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv4-afd","prompt":"介绍 Attention-FFN 分离。","temperature":0,"max_tokens":32}'
```

首次请求成功后，至少再发送一次相同请求和一次不同长度的请求，确认 Graph replay、U2 两个 stage 和 Window/P2P 的状态可以复用。

## 7. 相关链接
- [DeepSeek-V4 Attention-FFN 分离方案](../../../docs/models/deepseek_v4/deepseek_v4_afd_guide.md)
- [afd-plugin 仓库](https://github.com/vllm-project/afd-plugin)
- afd-plugin PR：https://github.com/vllm-project/afd-plugin/pull/408
- [vLLM](https://github.com/vllm-project/vllm)
- [vLLM-Ascend](https://github.com/vllm-project/vllm-ascend)
