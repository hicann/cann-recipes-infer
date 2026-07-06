# Hy3模型在NPU实现高性能推理

## 概述

腾讯混元 Hy3 正式发布，总参数量 295B，激活参数量约 21B/token，是一款大规模 MoE 语言模型。本样例基于 Hy3 开源实现完成 NPU 推理适配，并接入 cann-recipes-infer 统一推理框架，可在 Atlas A3 和 Ascend 950PR/DT 平台上运行。

模型主要结构如下：

- Decoder-only MoE：80 层，首层 Dense FFN，后续 79 层 MoE。
- Attention：GQA，64 个 Q heads / 8 个 KV heads，head_dim 为 128，支持 QK Norm 和 RoPE。
- MoE：192 个 routed experts，top-8 sigmoid routing，包含 1 个 shared expert。
- 词表大小：120832；最大上下文长度：262144。

本样例的优化方案与性能数据可参考 [Hy3 推理优化实践](../../docs/models/hy3/hy3_optimization.md)。

---

## 硬件要求

| 平台 | 产品型号 | 配置 |
| --- | --- | --- |
| Atlas A3 | Atlas A3 系列产品 | `ci_a3/hy3_rank16_bf16.yaml` |
| Atlas A5 | Ascend 950PR/DT 系列产品 | `ci_950/hy3_rank4_mxfp48.yaml` |

基础软件版本：

- A3 手动部署：CANN 9.1.0.beta.1、PyTorch 2.8.0、torch_npu v26.0.0。
- A5 Docker 部署：支持 x86 操作系统，使用 cann9.1.0.pt2.9.0_hy3_x86_image_v1.tar 镜像。

> 说明：执行前可通过 `npu-smi info` 检查 Ascend NPU 固件和驱动是否正确安装。

---

## 快速启动

### 下载源码

在各个节点上执行如下命令下载 cann-recipes-infer 源码。

```bash
mkdir -p /home/code
cd /home/code
git clone https://gitcode.com/cann/cann-recipes-infer.git
cd cann-recipes-infer
```

### 下载数据集

Hy3 样例默认使用 `dataset/default_prompt.json` 中的内置 prompt。如需使用 LongBench 长序列数据集，请在各个节点上准备 `dataset/LongBench` 目录。

```bash
mkdir -p dataset/LongBench
huggingface-cli download --repo-type dataset THUDM/LongBench --local-dir dataset/LongBench
```

使用 LongBench 时，将 YAML 中的 `data_config.dataset` 修改为 `LongBench`。若本地不存在 `dataset/LongBench`，框架会尝试在线读取 `THUDM/LongBench`。

> 说明：LongBench 或自定义数据集默认走文本摘要 prompt 模板，可在 `executor/utils/data_utils.py` 的 `build_dataset_input` 中按需修改。

### 下载权重

请下载 Hy3 权重并上传到各节点相同路径。BF16 原始权重用于 A3 bf16 配置及量化权重转换输入；FP8 W8A8 量化权重可直接用于 A5/950 fp8 配置。

- BF16：[Tencent-Hunyuan/Hy3](https://modelscope.cn/models/Tencent-Hunyuan/Hy3) → `/data/models/Hy3-BF16`
- FP8 ：[Tencent-Hunyuan/Hy3-FP8](https://modelscope.cn/models/Tencent-Hunyuan/Hy3-FP8) → `/data/models/Hy3-FP8`

示例下载命令：

```bash
pip install modelscope
modelscope download --model Tencent-Hunyuan/Hy3 --local_dir /data/models/Hy3-BF16
modelscope download --model Tencent-Hunyuan/Hy3-FP8 --local_dir /data/models/Hy3-FP8
```

> 说明：A5/950 的 MXFP8+MXFP4（mxfp48）权重需从 BF16 转换，见[转换权重](#转换权重)章节。

### 环境准备

#### Atlas A3 部署

1. 安装 CANN 软件包。

   本样例依赖 CANN 开发套件包与 CANN 二进制算子包，支持的 CANN 软件版本为 `CANN 9.1.0.beta.1`。

   请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.1.0-beta.1)下载 `Ascend-cann-toolkit_${version}_linux-${arch}.run` 与 `Ascend-cann-A3-ops_${version}_linux-${arch}.run` 软件包，并参考 [CANN 安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum)进行安装。

   - `${version}` 表示 CANN 包版本号，如 `9.1.0.beta.1`。
   - `${arch}` 表示 CPU 架构，如 `aarch64`、`x86_64`。

2. 安装 Ascend Extension for PyTorch（torch_npu）。

   本样例支持的 torch_npu 版本为 `v26.0.0`，PyTorch 版本为 `2.8.0`。

   请从[软件包下载地址](https://gitcode.com/Ascend/pytorch/releases/v26.0.0-pytorch2.8.0)下载 `torch_npu-2.8.0.post4-cp311-cp311-manylinux_2_28_${arch}.whl` 安装包，并参考 [torch_npu 安装文档](https://www.hiascend.com/document/detail/zh/Pytorch/2600/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md)进行安装。

3. 安装 Python 依赖。

   ```bash
   cd /home/code/cann-recipes-infer
   pip3 install -r ./models/hy3/requirements.txt
   ```

4. 配置运行环境。

   修改 `executor/scripts/set_env.sh` 中的如下字段：

   - `IPs`：配置所有节点的 IP，按照 rank id 排序，多个节点的 IP 通过空格分开，例如 `('xxx.xxx.xxx.xxx' 'xxx.xxx.xxx.xxx')`。
   - `cann_path`：CANN 软件包安装路径，例如 `/usr/local/Ascend/ascend-toolkit/latest`。

   > 说明：HCCL 相关配置，如 `HCCL_SOCKET_IFNAME`、`HCCL_OP_EXPANSION_MODE`，可以参考[集合通信文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/maintenref/envvar/envref_07_0001.html)并在 `executor/scripts/function.sh` 中自定义配置。

#### Atlas A5 Docker 部署

1. 获取 Docker 镜像。

   从[x86镜像地址](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-quantization/Hy3/cann9.1.0.pt2.9.0_hy3_x86_image_v1.tar)中下载 docker 镜像，然后上传到A5服务器的每个节点上，并通过命令导入镜像 `docker load -i cann9.1.0.pt2.9.0_hy3_x86_image_v1.tar`。

2. 拉起 Docker 容器。

   在各个节点上通过如下脚本拉起容器，默认容器名为 cann_recipes_infer。注意：需要将权重路径和源码路径挂载到容器里。以下示例挂载 8 个 NPU 设备，并挂载源码与权重目录。

   ```bash
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
       -v /home/:/home \
       -v /data:/data \
       -v /etc/localtime:/etc/localtime \
       -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
       -v /etc/ascend_install.info:/etc/ascend_install.info -v /var/log/npu/:/usr/slog \
       -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
       -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/sbin:/usr/local/sbin \
       -v /etc/hccn.conf:/etc/hccn.conf -v /root/.pip:/root/.pip -v /etc/hosts:/etc/hosts \
       -v /usr/bin/hostname:/usr/bin/hostname \
       --net=host \
       --shm-size=128g \
       --privileged \
       cann9.1.0.pt2.9.0_hy3_x86_image:v1 /bin/bash
   ```

3. 在各个节点上通过如下命令进入容器：

   ```bash
   docker attach cann_recipes_infer
   cd /home/code/cann-recipes-infer
   ```

   同步修改 `executor/scripts/set_env.sh` 中的 `IPs` 和 `cann_path`。


### 转换权重

  HY3 权重转换过程中会同步完成权重文件和 `config.json` 的转换，因此无需单独执行 `config.json` 转换步骤。转换前请先拉取 AMCT 仓库，并使用其中的权重转换工具：

  ```shell
  git clone https://gitcode.com/cann/amct.git
  cd amct
  ```

  如果权重转换的运行环境为NPU，需要先执行：

  ```shell
  cann_path=/usr/local/Ascend/cann  # cann包安装路径
  source ${cann_path}/bin/setenv.bash
  ```

  >入参介绍：`model`：原始权重路径；`model_name`：AMCT 内部模型适配器名称，HY3 使用 `hy_v3`；`device`：权重转换使用的 NPU 设备；`granularity`：转换粒度，HY3 tensorwise 权重转换使用 `tensor`；`quant_target`：量化目标模块；`quant_dtype`：量化数据类型；`bit_config`：量化位宽配置文件；`output_dir`：转换后输出的权重路径。
  
  混合精度的yaml文件没有上传至`amct`仓库，请用户在`amct_pytorch/configs/`路径下创建`mxfp_moe_w4a8_attn_w8a8.yaml`，文件配置如下：
  
   ```shell
w_bits: 8
a_bits: 8

moe:
  routed:
    w_bits: 4
    a_bits: 8
  shared:
    w_bits: 8
    a_bits: 8
  ```

  权重转换拉起示例：

  ```shell
  python3 amct_pytorch/cli/llm/deploy.py \
      --model ./Hy3-BF16 \
      --model_name hy_v3 \
      --device npu:0 \
      --granularity tensor \
      --quant_target moe attn-linear \
      --quant_dtype mxfp \
      --bit_config amct_pytorch/configs/mxfp_moe_w4a8_attn_w8a8.yaml \
      --output_dir ./output
  ```


### 修改配置

在各个节点上修改需要执行的 YAML 文件，将 `model_config.model_path` 设置为权重实际路径。YAML 通用参数说明可参考 [YAML 参数描述](../../docs/common/inference_config_guide.md)。

当前仓内提供的 Hy3 配置如下：

| 平台 | YAML 文件 | 默认 `model_path` | 精度/特性 | 说明 |
| --- | --- | --- | --- | --- |
| A3 | `ci_a3/hy3_rank16_bf16.yaml` | `/data/models/Hy3-BF16` | BF16 | 8卡 16rank，`npugraph_ex` |
| A3 | `ci_a3/hy3_rank16_bf16_mtp.yaml` | `/data/models/Hy3-BF16` | BF16 + MTP | 8卡 16rank，`next_n=1` |
| A5/950 | `ci_950/hy3_rank4_mxfp48.yaml` | `/data/models/Hy3-MXFP4` | MXFP8 + MXFP4 | 4卡，`npugraph_ex` |
| A5/950 | `ci_950/hy3_rank4_mxfp48_mtp.yaml` | `/data/models/Hy3-MXFP4` | MXFP8 + MXFP4 + MTP | 4卡，`next_n=1` |
| A5/950 | `ci_950/hy3_rank4_fp8.yaml` | `/data/models/Hy3-FP8` | FP8 | 4卡，`npugraph_ex` |
| A5/950 | `ci_950/hy3_rank4_fp8_mtp.yaml` | `/data/models/Hy3-FP8` | FP8 + MTP | 4卡，`next_n=1` |

> 说明：A5/950 配置当前面向量化权重，快速启动阶段只下载 BF16 权重；量化权重准备方式见[转换权重](#转换权重)章节。

除框架统一配置外，Hy3 还支持以下特性，放置在 YAML 文件 `model_config` 的 `custom_params` 字段下：

| 参数名 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `enable_online_split_weight` | bool | `True` | 启用在线权重切分。未启用时需提前离线转换切分好的权重。 |
| `enable_multi_streams` | bool | `True` | 启用多流并行，重叠计算与通信以提升推理性能。 |
| `enable_sp` | bool | `True` | 启用序列并行（Sequence Parallel），`attn_tp_size>1` 时按 token 切分。 |
| `enable_fia_fp8` | bool | `True` | 启用 FIA FP8 C8 分页注意力（FP8 全量化 Flash Attention 配 C8 KV Cache），量化配置生效。 |
| `enable_qkv_fused_kscale` | bool | `True` | 启用 `qkv_rms_norm_rope_cache_with_k_scale` 融合算子（QKV split + RMSNorm + RoPE + 动态量化 + Cache 写回）。 |


### 拉起多卡推理

请先进入模型目录，再执行统一推理入口，避免日志输出到错误目录。

A3 BF16 示例：

```bash
cd /home/code/cann-recipes-infer/models/hy3
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
bash ../../executor/scripts/infer.sh --model hy3 --yaml ci_a3/hy3_rank16_bf16.yaml
```

A3 BF16 + MTP 示例：

```bash
cd /home/code/cann-recipes-infer/models/hy3
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
bash ../../executor/scripts/infer.sh --model hy3 --yaml ci_a3/hy3_rank16_bf16_mtp.yaml
```

A5/950 量化配置示例。该命令仅在已准备对应量化权重，并将 YAML 中的 `model_path` 修改为真实路径后执行：

```bash
cd /home/code/cann-recipes-infer/models/hy3
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
bash ../../executor/scripts/infer.sh --model hy3 --yaml ci_950/hy3_rank4_mxfp48.yaml
```

> 说明：多机环境需要在每个节点上执行同一条推理命令。脚本会根据 YAML 中的 `world_size` 与 `executor/scripts/set_env.sh` 中的 IP 列表计算各节点 rank 信息。

推理输出会打印在 rank 0 日志中，日志保存到 `models/hy3/res/${DATE}/${CASE_NAME}/log_*.log`。

---
