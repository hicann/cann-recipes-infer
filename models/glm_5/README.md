# GLM-5模型在NPU实现高性能推理

## 概述

GLM-5是智谱团队发布的大语言模型。本样例基于GLM-5开源代码完成迁移，并适配到cann-recipes-infer统一推理框架，可在华为Atlas A3集群上运行。

- GLM-5模型结构与DeepSeek-V3.2-Exp基本一致，本样例的并行策略与主要性能优化方案沿用DeepSeek-V3.2-Exp。详细方案可参考[基于Atlas A3集群的DeepSeek-V3.2-Exp模型推理优化实践](../../docs/models/deepseek_v3_2_exp/deepseek_v3.2_exp_inference_guide.md)。

## 支持的产品型号

<term>Atlas A3 系列产品</term>

## 环境准备

本样例推荐使用Docker镜像方式运行。镜像内已包含模型运行所需的CANN、PyTorch、torch_npu以及自定义算子包，用户无需手工编译和安装自定义算子。运行前，请确保宿主机已正确安装Ascend NPU固件与驱动，且版本为Ascend HDK 25.2.0。

> `npu-smi info`可用于检查Ascend NPU固件和驱动是否正确安装。若未安装或版本不满足要求，请先下载[固件和驱动包](https://www.hiascend.com/hardware/firmware-drivers/community?product=7&model=33&cann=All&driver=Ascend+HDK+25.2.0)，并参考[安装指导](https://hiascend.com/document/redirect/CannCommunityInstSoftware)完成安装。

### 获取Docker镜像

从[ARM镜像地址](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-quantization/GLM/cann9.1_pt2.8.0_glm_aarch_image_v0.2.tar)下载Docker镜像，上传到A3服务器的每个节点，并通过如下命令导入镜像：

```bash
docker load -i cann9.1_pt2.8.0_glm_aarch_image_v0.2.tar
```

在各个节点上通过如下脚本拉起容器，默认容器名为`cann_recipes_infer`。需要将权重路径和源码路径挂载到容器内。

```bash
docker run -u root -itd --name cann_recipes_infer --ulimit nproc=65535:65535 --ipc=host \
    --device=/dev/davinci0     --device=/dev/davinci1 \
    --device=/dev/davinci2     --device=/dev/davinci3 \
    --device=/dev/davinci4     --device=/dev/davinci5 \
    --device=/dev/davinci6     --device=/dev/davinci7 \
    --device=/dev/davinci8     --device=/dev/davinci9 \
    --device=/dev/davinci10    --device=/dev/davinci11 \
    --device=/dev/davinci12    --device=/dev/davinci13 \
    --device=/dev/davinci14    --device=/dev/davinci15 \
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
    cann9.1_pt2.8.0_glm_aarch_image:v0.2 /bin/bash
```

进入容器：

```bash
docker attach cann_recipes_infer
```

### 下载源码并配置环境

进入容器后，在各个节点上执行如下命令下载cann-recipes-infer源码：

```bash
mkdir -p /home/code
cd /home/code
git clone https://gitcode.com/cann/cann-recipes-infer.git
cd cann-recipes-infer
```

修改`executor/scripts/set_env.sh`中的如下字段：

- `IPs`：离线推理场景的节点IP列表，按照rank id排序，多个节点的IP通过空格分开，例如`('xxx.xxx.xxx.xxx' 'xxx.xxx.xxx.xxx')`。
- `PREFILL_IPS` / `DECODE_IPS`：在线PD分离推理场景的prefill/decode节点IP列表，按实例和rank顺序填写。
- `cann_path`：CANN软件包安装路径，例如`/usr/local/Ascend/ascend-toolkit/latest`。

> 说明：HCCL相关配置，如`HCCL_SOCKET_IFNAME`、`HCCL_OP_EXPANSION_MODE`，可以参考[集合通信文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/maintenref/envvar/envref_07_0001.html)并在`executor/scripts/function.sh`中自定义配置。

## 权重准备

请根据所使用的模型类型自行下载原始权重到本地路径，例如`/data/models/GLM-5-FP8`。

GLM-5原始权重下载地址如下：

- [GLM-5 Bfloat16权重](https://huggingface.co/zai-org/GLM-5)
- [GLM-5 FP8权重](https://huggingface.co/zai-org/GLM-5-FP8)

## 权重转换

在各个节点上使用`weight_convert.sh`脚本完成FP8到Int8权重转换。

> 入参介绍：`input_fp8_hf_path`为原始FP8权重路径；`output_hf_path`为转换后输出的权重路径；`quant_mode`为量化模式。

如果权重转换的运行环境为NPU，需要先执行：

```bash
cann_path=/usr/local/Ascend/ascend-toolkit/latest
source ${cann_path}/bin/setenv.bash
```

执行权重转换前，先切换到GLM-5模型目录：

```bash
cd models/glm_5
```

权重转换拉起示例：

```bash
# 转换为W8A8C16权重
bash utils/weight_convert.sh --input_fp8_hf_path /data/models/GLM-5-FP8 --output_hf_path /data/models/GLM-5-W8A8 --quant_mode w8a8c16

# 转换为MXFP8权重
bash utils/weight_convert.sh --input_fp8_hf_path /data/models/GLM-5-FP8 --output_hf_path /data/models/GLM-5-W8A8-MXFP8 --quant_mode w8a8-mx
```

## 推理执行

1. 配置推理执行需要加载的权重文件以及YAML文件。

   - 修改YAML文件中的参数。

     在`models/glm_5/config/`目录下已提供YAML样例供您参考，您可以根据集群规模、平台类型、推理模式以及量化类型选择对应的YAML文件。本文以`models/glm_5/config/glm_5_rank_128_128ep_w8a8_decode_benchmark.yaml`文件为例，修改其中的`model_path`参数，将其设置为[权重转换](#权重转换)阶段准备好的权重文件存储路径，例如`/data/models/GLM-5-W8A8`。

     - YAML文件中的配置说明可见[YAML参数描述](../../docs/common/inference_config_guide.md)。

     - 除框架统一配置之外，还额外支持以下特性，放置在 YAML 文件 `model_config` 的 `custom_params` 字段下：

     | 参数名 | 位置 | 含义 |
     | --- | --- | --- |
     | `moe_chunk_max_len` | `model_config.custom_params` | prefill阶段MoE按token维度分块的最大长度，用于降低routing/GMM峰值显存，默认65536。 |
     | `enable_multi_streams` | `model_config.custom_params` | 启用多流优化。 |
     | `enable_offload` | `model_config.custom_params` | 启用KV cache offload，当前仅支持offline推理场景。 |

   - 配置`executor/scripts/infer.sh`脚本中的参数。

     离线推理模式下，将`--yaml`设置为`config`文件夹下YAML文件名称，例如`glm_5_rank_128_128ep_w8a8_decode_benchmark.yaml`。
     在线推理模式下，将`--mode`设置为`online`，`--pd-role`设置为`prefill`或`decode`，可通过`--p-yaml-name`和`--d-yaml-name`指定prefill/decode的YAML文件。

   - 需注意当前仅prefill路径支持Context Parallel。在线PD分离推理时，router会读取prefill和decode YAML中的`parallel_config.world_size`来计算各实例的leader地址，因此prefill主节点上的prefill/decode YAML必须与对应角色节点保持一致。

2. 准备输入prompt。

   - 使用内置prompt。

     本样例已在`dataset/default_prompt.json`中内置输入prompt。若您直接使用内置prompt，可将YAML中的`data_config.dataset`设置为`default`。

     您也可以在`dataset/default_prompt.json`中自定义prompt输入。

   - 使用长序列prompt（默认）。

     若需要使用长序列prompt，可以选择InfiniteBench数据集。需要从[链接](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/blob/main/longbook_qa_eng.jsonl)下载长序列输入数据集`longbook_qa_eng`，并上传到各个节点的`dataset/InfiniteBench`目录下。

     ```bash
     mkdir -p dataset/InfiniteBench
     ```

     使用InfiniteBench时，将YAML中的`data_config.dataset`设置为`InfiniteBench`。

     > 说明：在使用InfiniteBench或其他自定义数据集时，默认执行文本摘要任务，可在`cann-recipes-infer/executor/utils/data_utils.py`的`build_dataset_input`函数中修改默认system prompt。

3. 执行统一推理脚本。

   以下命令需在仓库根目录下执行。

   统一入口脚本位于`executor/scripts/infer.sh`，通过以下参数控制启动：

   | 参数 | 含义 | 取值示例 |
   | --- | --- | --- |
   | `--model` | 模型目录名，对应`models/`下的子目录 | `glm_5` |
   | `--mode` | 推理模式 | `offline`（离线推理） / `online`（在线PD分离推理） |
   | `--yaml` | 离线模式：YAML文件名 | `glm_5_rank_128_128ep_w8a8_decode_benchmark.yaml` |
   | `--pd-role` | 在线模式：PD部署角色 | `prefill` / `decode` |
   | `--p-yaml-name` | 可选，在线模式：prefill YAML文件名，不传则默认`glm_5_pd/prefill.yaml` | `glm_5_pd/prefill.yaml` |
   | `--d-yaml-name` | 可选，在线模式：decode YAML文件名，不传则默认`glm_5_pd/decode.yaml` | `glm_5_pd/decode.yaml` |

   > 在线模式IP等更多配置参见[executor设计文档 §5.1 启动方式](../../docs/design/executor_design.md#51-启动方式)。

   **使用方式一：命令行传参**

   ```bash
   # offline模式
   bash executor/scripts/infer.sh --model glm_5 --yaml glm_5_rank_128_128ep_w8a8_decode_benchmark.yaml

   # online模式
   bash executor/scripts/infer.sh --model glm_5 --mode online --pd-role prefill

   # online模式（指定prefill/decode YAML）
   bash executor/scripts/infer.sh --model glm_5 --mode online --pd-role prefill --p-yaml-name glm_5_pd/prefill.yaml --d-yaml-name glm_5_pd/decode.yaml
   ```

   如需查看参数说明，可以执行`bash executor/scripts/infer.sh --help`。

   **使用方式二：直接修改脚本默认值后执行**

   编辑`executor/scripts/infer.sh`，按照需求修改`MODEL`、`MODE`、`YAML_FILE`、`PD_ROLE`、`P_YAML_NAME`、`D_YAML_NAME`等参数的默认值，例如：

   ```bash
   MODEL=glm_5
   MODE=offline
   YAML_FILE=glm_5_rank_128_128ep_w8a8_decode_benchmark.yaml
   ```

   保存后直接执行：

   ```bash
   bash executor/scripts/infer.sh
   ```

   > 说明：
   > - 如果是多机环境，需要在每个节点上执行。
   > - 推理日志和结果保存在`models/glm_5/res/`路径下。

## Benchmark

基于Atlas A3，本实践使用`config/glm_5_rank_128_128ep_w8a8_decode_benchmark.yaml`作为运行配置文件，对GLM-5 W8A8量化版本进行了性能Benchmark测试。

| Quant Mode | Global Batch Size | Seq Length | Chips | TPOT (ms) | Throughput (tokens/p/s) |
| --- | --- | --- | --- | --- | --- |
| W8A8 | 256 | 65536 | 64 | 22.54 | 177.46 |

> 注：性能数据基于MTP3与force eplb配置采集，平均3个draft token中accept token为1.44个。

### 精度评测

在GSM8K数据集上对GLM-5 W8A8量化版本进行精度评测，结果如下：

| Quant Mode | GSM8K Accuracy |
| --- | --- |
| W8A8 | 96.59% |

## 附录

### 常见问题处理

**长序列请求out of memory问题处理**

长序列请求可能导致device内存out of memory，尤其是在prefill阶段。可通过以下方式降低峰值内存：

- 通过YAML中的`scheduler_config.max_prefill_tokens`限制单次prefill batch的总prompt token数，从而控制prefill阶段峰值内存。
- 可通过`model_config.custom_params.moe_chunk_max_len`设置prefill MoE chunk大小，降低routing/GMM峰值显存占用。当前框架默认值为65536，设置较小的值可以有效降低峰值显存，但会增加MoE处理轮次，导致通信开销增加和性能下降，需要根据实际显存压力权衡。
