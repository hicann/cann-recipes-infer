# GPT-OSS模型在NPU上推理

## 概述
本样例基于Transformers库的[GPT-OSS](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_oss/modeling_gpt_oss.py)模型，在<term>Atlas A2系列产品</term>实现了单机单batch推理，其中GPT-OSS-120B模型可以采用8卡部署，GPT-OSS-20B模型可以在单device上进行部署。
- 本样例采用的详细优化点介绍可参见[基于Atlas A2系列产品的GPT-OSS模型推理性能优化实践](../../docs/models/gpt-oss/gpt_oss_optimization.md)。

下面详细介绍GPT-OSS的推理样例在NPU上的执行步骤。

## 支持的产品型号
<term>Atlas A2 系列产品</term>

## 环境准备

1. 安装CANN软件包。

   本样例的编译执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），支持的CANN软件版本为`CANN 9.0.0`。

   请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0)下载`Ascend-cann-toolkit_${version}_linux-${arch}.run`与`Ascend-cann-A3-ops_<version>_linux-<arch>.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0090.html?OS=Ubuntu&InstallType=localpack)进行安装。

    - `${version}`表示CANN包版本号，如9.0.0。
    - `${arch}`表示CPU架构，如aarch64、x86_64。

2. 安装Ascend Extension for PyTorch（torch_npu）。

   Ascend Extension for PyTorch（torch_npu）为支撑PyTorch框架运行在NPU上的适配插件，本样例支持的Ascend Extension for PyTorch版本为`v26.0.0`，PyTorch版本为`2.8.0`。
   请从[软件包下载地址](https://gitcode.com/Ascend/pytorch/releases/v26.0.0-pytorch2.8.0)下载`torch_npu-2.8.0.post4-cp311-cp311-manylinux_2_28_${arch}.whl`安装包，参考[torch_npu安装文档](https://www.hiascend.com/document/detail/zh/Pytorch/2600/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md)进行安装。


3. 下载项目源码并安装依赖的python库。
    ```bash
    # 下载项目源码，以master分支为例
    git clone https://gitcode.com/cann/cann-recipes-infer.git

    # 安装依赖的python库
    cd cann-recipes-infer/models/gpt_oss
    pip3 install -r requirements.txt
    ```

4. 配置样例运行所需环境信息。

   修改`executor/scripts/set_env.sh`脚本中的如下字段：
   - `cann_path`: CANN软件包安装路径，例如`/usr/local/Ascend/ascend-toolkit/latest`。
    > 说明：HCCL相关配置，如：`HCCL_SOCKET_IFNAME`、`HCCL_OP_EXPANSION_MODE`，可以参考[集合通信文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/maintenref/envvar/envref_07_0001.html)并在`executor/scripts/function.sh`中自定义配置。

## 权重准备

本样例对GPT-OSS开源模型的原始权重进行了切分与调整，GPT-OSS提供了两个原始权重：
- [GPT-OSS-20B权重](https://huggingface.co/openai/gpt-oss-20b/tree/main)
- [GPT-OSS-120B权重](https://huggingface.co/openai/gpt-oss-120b/tree/main)

开发者可以根据模型任务进行选择，并将原始权重下载至本地路径，例如：`/data/models/gpt-oss-20b-bf16`。

> 注意：原始权重为mxfp4格式，本推理脚本仅支持bf16格式，可参考[官方代码](https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/weights.py)自行转换。


## 推理执行

1. 配置推理执行需要加载的权重文件以及YAML文件。

   - 修改YAML文件中的`model_path`参数。

     在`models/gpt_oss/config`目录下已提供了较优性能的YAML样例供您参考，您可以根据权重的不同选择对应的YAML文件，本文以`gpt_oss_20b.yaml`文件为例，修改其中的`model_path`参数，将其设置为[权重准备](#权重准备)阶段准备好的权重文件存储路径，例如`/data/models/gpt-oss-20b-bf16`。

     关于YAML文件中的更多配置说明可参见[YAML参数描述](../../docs/common/inference_config_guide.md)。

   - 配置`executor/scripts/infer.sh`脚本中的参数。

     离线推理模式下，将`--yaml`设置为`config`文件夹下YAML文件名称，例如`gpt_oss_20b.yaml`。
     在线推理模式下，将`--mode`设置为`online`，`--pd-role`设置为`prefill`或`decode`，可通过`--p-yaml-name`和`--d-yaml-name`指定prefill/decode的YAML文件。

2. 准备输入prompt。

   - 使用内置prompt。

     本样例已在`dataset/default_prompt.json`中内置了输入prompt，若您直接使用内置prompt，本步骤可直接跳过。

     当然，您也可以在`dataset/default_prompt.json`文件中自定义prompt输入。

   - 使用长序列prompt。

     本样例默认使用内置prompt，若您需要使用长序列prompt，需要执行以下操作：

     1. 修改YAML文件中的`dataset`参数，将其修改为`dataset: "LongBench"`，使用LongBench数据集作为长序列prompt。

     2. 若您的机器无法联网，需要您从[huggingface](http://huggingface.co/datasets/zai-org/LongBench/tree/main)手动下载数据集至`dataset/LongBench`目录下，`LongBench`文件夹需手工创建，目录中包含`LongBench.py`和`data`目录，并需要在`LongBench.py`中修改数据集加载路径；若您的机器可正常联网，样例执行过程中会自动在线读取LongBench数据集，您无需手工下载。
      > 说明：使用LongBench数据集时，默认执行文本摘要任务，可在`cann-recipes-infer/executor/utils/data_utils.py`的`build_dataset_input`函数里修改默认的system prompt。

3. 执行统一推理脚本。

   统一入口脚本位于 `executor/scripts/infer.sh`，通过以下参数控制启动：

   | 参数 | 含义 | 取值示例 |
   | --- | --- | --- |
   | `--model` | 模型目录名，对应 `models/` 下的子目录 | `gpt_oss` |
   | `--mode` | 推理模式 | `offline`（离线推理）/ `online`（在线PD分离推理） |
   | `--yaml` | 离线模式：yaml 文件名 | `gpt_oss_20b.yaml` |
   | `--pd-role` | 在线模式：PD 角色 | `prefill` / `decode` |
   | `--p-yaml-name` | 可选，在线模式：prefill yaml 文件名，不传则默认 `gpt_oss_pd/prefill.yaml` | `gpt_oss_pd/prefill.yaml` |
   | `--d-yaml-name` | 可选，在线模式：decode yaml 文件名，不传则默认 `gpt_oss_pd/decode.yaml` | `gpt_oss_pd/decode.yaml` |

   **使用方式一：命令行传参**
   ```shell
   # offline 模式
   bash executor/scripts/infer.sh --model gpt_oss --yaml gpt_oss_20b.yaml
   # online 模式
   bash executor/scripts/infer.sh --model gpt_oss --mode online --pd-role prefill
   # online 模式（指定 prefill/decode yaml）
   bash executor/scripts/infer.sh --model gpt_oss --mode online --pd-role prefill --p-yaml-name gpt_oss_pd/prefill.yaml --d-yaml-name gpt_oss_pd/decode.yaml
   ```

   如需查看参数说明，可以执行 `bash executor/scripts/infer.sh --help`。

   **使用方式二：直接修改脚本默认值后执行**
    编辑 executor/scripts/infer.sh，修改 MODEL / MODE / YAML_FILE / PD_ROLE / P_YAML_NAME / D_YAML_NAME 等参数的默认值，例如：
    ```shell
    MODEL=gpt_oss
    MODE=offline
    YAML_FILE=gpt_oss_20b.yaml
    ```
    保存后直接执行：
   ```shell
   bash executor/scripts/infer.sh
   ```

   > 说明：推理日志和结果保存在 `models/gpt_oss/res/` 路径下。

   > **需要注意**
   > - 目前仅支持prompt的batch_size为1。
   > - 默认使用`eager`单算子模式推理。
   > - 对于20b模型提供单device推理，对于120b模型提供8卡推理（仅支持tp切分）。
   > - YAML文件中默认设置`enable_online_split_weight: True`，模型权重会在加载过程中[在线切分](../../docs/common/online_split_weight_guide.md)到各个设备上，无需离线切分。
