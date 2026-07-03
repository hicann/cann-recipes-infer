# DeepSeek-V3.2-Exp模型在NPU实现高性能推理

## 概述

DeepSeek-V3.2-Exp是DeepSeek团队发布的大语言模型。本样例基于DeepSeek开源代码完成迁移，并适配到cann-recipes-infer统一推理框架，可在华为Atlas A3集群上运行。

- 本样例的并行策略和性能优化点详细介绍可参见[基于Atlas A3集群的DeepSeek-V3.2-Exp模型推理优化实践](../../docs/models/deepseek_v3_2_exp/deepseek_v3.2_exp_inference_guide.md)。

---

## 支持的产品型号

<term>Atlas A3 系列产品</term>

## 环境准备

本样例统一使用Docker镜像方式运行。镜像内已包含模型运行所需的CANN、PyTorch、torch_npu以及自定义算子包，用户无需手工编译和安装自定义算子。运行前，请确保宿主机已正确安装 Ascend NPU 固件与驱动，且版本为 Ascend HDK 25.2.0。
> npu-smi info 检查Ascend NPU固件和驱动是否正确安装。如果已安装，通过命令`npu-smi info`确认版本是否为 25.2.0。如果未安装或者版本不是 25.2.0，请先下载[固件和驱动包](https://www.hiascend.com/hardware/firmware-drivers/community?product=7&model=33&cann=All&driver=Ascend+HDK+25.2.0)，然后根据[指导](https://hiascend.com/document/redirect/CannCommunityInstSoftware)自行安装。

### 获取Docker镜像

镜像版本和链接如下，后续如有新镜像版本请以最新发布信息为准。

从[ARM镜像地址](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-quantization/DeepSeek-V3.2-Exp/cann9.0_pt2.8.0_dsv3.2_aarch_image_v0.1.tar)中下载Docker镜像，然后上传到A3服务器的每个节点，并通过如下命令导入镜像：

```bash
docker load -i cann9.0_pt2.8.0_dsv3.2_aarch_image_v0.1.tar
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
    cann9.0_pt2.8.0_dsv3.2_aarch_image:v0.1 /bin/bash
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

- `IPs`：配置所有节点的IP，按照rank id排序，多个节点的ip通过空格分开，例如：`('xxx.xxx.xxx.xxx' 'xxx.xxx.xxx.xxx')`。
- `cann_path`：CANN软件包安装路径，例如`/usr/local/Ascend/ascend-toolkit/latest`。

> 说明：HCCL相关配置，如`HCCL_SOCKET_IFNAME`、`HCCL_OP_EXPANSION_MODE`，可以参考[集合通信文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/maintenref/envvar/envref_07_0001.html)并在`executor/scripts/function.sh`中自定义配置。

## 权重准备

请根据所使用的模型类型自行下载原始权重到本地路径，例如`/data/models/DeepSeek-V3.2-Exp-FP8`。

DeepSeek-V3.2-Exp原始权重下载地址：[DeepSeek-V3.2-Exp权重](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp)。

## 权重转换

在各个节点上使用`weight_convert.sh`脚本完成FP8到Bfloat16/Int8权重转换。

> 入参介绍：`input_fp8_hf_path`为原始FP8权重路径；`output_hf_path`为转换后输出的权重路径；`quant_mode`为量化模式。

如果权重转换的运行环境为NPU，需要先执行：

```bash
cann_path=/usr/local/Ascend/ascend-toolkit/latest
source ${cann_path}/bin/setenv.bash
```

执行权重转换前，先切换到DeepSeek-V3.2-Exp模型目录：

```bash
cd models/deepseek_v3_2_exp
```

权重转换拉起示例：

```bash
# 转换为Bfloat16权重
bash utils/weight_convert.sh --input_fp8_hf_path /data/models/DeepSeek-V3.2-Exp-FP8 --output_hf_path /data/models/DeepSeek-V3.2-Exp-Bfloat16 --quant_mode bfloat16

# 转换为W8A8C16权重
bash utils/weight_convert.sh --input_fp8_hf_path /data/models/DeepSeek-V3.2-Exp-FP8 --output_hf_path /data/models/DeepSeek-V3.2-Exp-W8A8C16 --quant_mode w8a8c16

# 转换为W8A8C8权重
bash utils/weight_convert.sh --input_fp8_hf_path /data/models/DeepSeek-V3.2-Exp-FP8 --output_hf_path /data/models/DeepSeek-V3.2-Exp-W8A8C8 --quant_mode w8a8c8

# 转换为W4A8C8权重
bash utils/weight_convert.sh --input_fp8_hf_path /data/models/DeepSeek-V3.2-Exp-FP8 --output_hf_path /data/models/DeepSeek-V3.2-Exp-W4A8C8 --quant_mode w4a8c8
```

## 推理执行

1. 配置推理执行需要加载的权重文件以及YAML文件。

   - 修改YAML文件中的参数。

     在`models/deepseek_v3_2_exp/config/`目录下已提供了较优性能的YAML样例供您参考，您可以根据模型类型选择对应的YAML文件。本文以`models/deepseek_v3_2_exp/config/deepseek_v3.2_exp_rank_128_128ep_w8a8c8_decode_benchmark.yaml`文件为例，修改其中的`model_path`参数，将其设置为[权重转换](#权重转换)阶段准备好的权重文件存储路径，例如`/data/models/DeepSeek-V3.2-Exp-W8A8C8`。

     - YAML文件中的配置说明可见[YAML参数描述](../../docs/common/inference_config_guide.md)。

     - 除框架统一配置之外，还额外支持以下特性，放置在 YAML 文件 `model_config` 的 `custom_params` 字段下：

     | 参数名 | 位置 | 含义 |
     | --- | --- | --- |
     | `enable_offload` | `model_config.custom_params` | 启用KV cache offload。 |
     | `enable_multi_streams` | `model_config.custom_params` | 启用多流优化。 |

   - 配置`executor/scripts/infer.sh`脚本中的参数。

     离线推理模式下，将`--yaml`设置为`config`文件夹下YAML文件名称，例如`deepseek_v3.2_exp_rank_128_128ep_w8a8c8_decode_benchmark.yaml`。
     在线推理模式下，将`--mode`设置为`online`，`--pd-role`设置为`prefill`或`decode`，可通过`--p-yaml-name`和`--d-yaml-name`指定prefill/decode的YAML文件。

   - 需注意当前仅prefill路径支持Context Parallel，约束为`world_size == cp_size`。KV cache offload当前仅支持offline推理场景。

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

     > 说明：
     > - 在使用InfiniteBench数据集或其他自定义数据集时，默认执行文本摘要任务，可在`cann-recipes-infer/executor/utils/data_utils.py`的`build_dataset_input`函数里修改默认的system prompt。

3. 执行统一推理脚本。

   以下命令需在仓库根目录下执行。

   统一入口脚本位于`executor/scripts/infer.sh`，通过以下参数控制启动：

   | 参数 | 含义 | 取值示例 |
   | --- | --- | --- |
   | `--model` | 模型目录名，对应`models/`下的子目录 | `deepseek_v3_2_exp` |
   | `--mode` | 推理模式 | `offline`（离线推理） / `online`（在线PD分离推理） |
   | `--yaml` | 离线模式：YAML文件名 | `deepseek_v3.2_exp_rank_128_128ep_w8a8c8_decode_benchmark.yaml` |
   | `--pd-role` | 在线模式：PD部署角色 | `prefill` / `decode` |
   | `--p-yaml-name` | 可选，在线模式：prefill YAML文件名，不传则默认`deepseek_v3_2_exp_pd/prefill.yaml` | `deepseek_v3_2_exp_pd/prefill.yaml` |
   | `--d-yaml-name` | 可选，在线模式：decode YAML文件名，不传则默认`deepseek_v3_2_exp_pd/decode.yaml` | `deepseek_v3_2_exp_pd/decode.yaml` |

   > 在线模式IP等更多配置参见[executor设计文档 §5.1 启动方式](../../docs/design/executor_design.md#51-启动方式)。

   **使用方式一：命令行传参**

   ```bash
   # offline模式
   bash executor/scripts/infer.sh --model deepseek_v3_2_exp --yaml deepseek_v3.2_exp_rank_128_128ep_w8a8c8_decode_benchmark.yaml

   # online模式
   bash executor/scripts/infer.sh --model deepseek_v3_2_exp --mode online --pd-role prefill

   # online模式（指定prefill/decode YAML）
   bash executor/scripts/infer.sh --model deepseek_v3_2_exp --mode online --pd-role prefill --p-yaml-name deepseek_v3_2_exp_pd/prefill.yaml --d-yaml-name deepseek_v3_2_exp_pd/decode.yaml
   ```

   如需查看参数说明，可以执行`bash executor/scripts/infer.sh --help`。

   **使用方式二：直接修改脚本默认值后执行**

   编辑`executor/scripts/infer.sh`，按照需求修改`MODEL`、`MODE`、`YAML_FILE`、`PD_ROLE`、`P_YAML_NAME`、`D_YAML_NAME`等参数的默认值，例如：

   ```bash
   MODEL=deepseek_v3_2_exp
   MODE=offline
   YAML_FILE=deepseek_v3.2_exp_rank_128_128ep_w8a8c8_decode_benchmark.yaml
   ```

   保存后直接执行：

   ```bash
   bash executor/scripts/infer.sh
   ```

   > 说明：
   > - 如果是多机环境，需要在每个节点上执行。
   > - 推理日志和结果保存在`models/deepseek_v3_2_exp/res/`路径下。

## 附录

### 常见问题处理

**HCCL_BUFFSIZE不足问题**

如果报错日志中出现关键字`HCCL_BUFFSIZE is too SMALL, ..., NEEDED_HCCL_BUFFSIZE..., HCCL_BUFFSIZE=200MB, ...`，可通过配置环境变量`HCCL_BUFFSIZE`解决，所有rank上的该环境变量需保持一致。HCCL_BUFFSIZE参数介绍可参考[昇腾资料](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/maintenref/envvar/envref_07_0080.html)。

**自定义算子导入失败**

如果报错日志中出现类似`'_OpNamespace' 'custom' object has no attribute`的关键字，说明当前环境缺少所需自定义算子。请确认使用的是本样例提供的Docker镜像，或确认镜像内自定义算子包安装完整。offload路径依赖`npu_gather_selection_kv_cache`算子，W4A8C8路径依赖`npu_swiglu_clip_quant`。
