# GPT-OSS模型在NPU上推理

## 概述
本样例基于Transformers库的[GPT-OSS](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_oss/modeling_gpt_oss.py)模型，在<term>Atlas A2系列产品</term>实现了单机单batch推理，其中GPT-OSS-120B模型可以采用8卡部署，GPT-OSS-20B模型可以在单device上进行部署。
- 本样例采用的详细优化点介绍可参见[基于Atlas A2系列产品的GPT-OSS模型推理性能优化实践](../../docs/models/gpt-oss/gpt_oss_optimization.md)。

下面详细介绍GPT-OSS的推理样例在NPU上的执行步骤。

## 支持的产品型号
<term>Atlas A2 系列产品</term>

## 环境准备

1. 安装CANN软件包。

   本样例的编译执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels）。支持的CANN软件版本为`CANN 8.5.0.alpha002`。

   请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0.alpha002)下载`Ascend-cann-toolkit_${version}_linux-${arch}.run`与`Ascend-cann-kernels-910b_${version}_linux-${arch}.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Debian&Software=cannToolKit)进行安装。

    - `${version}`表示CANN包版本号，如8.5.0.alpha002。
    - `${arch}`表示CPU架构，如aarch64、x86_64。


2. 安装Ascend Extension for PyTorch（torch_npu）。

   Ascend Extension for PyTorch（torch_npu）为支撑PyTorch框架运行在NPU上的适配插件。
   请从[软件包下载地址](https://gitcode.com/Ascend/pytorch/tree/v2.6.0-7.3.0)下载`v2.6.0-7.3.0`源码，参考[源码编译安装](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0005.html)。


3. 下载项目源码并安装依赖的python库。
    ```bash
    # 下载项目源码，以master分支为例
    git clone https://gitcode.com/cann/cann-recipes-infer.git

    # 安装依赖的python库
    cd cann-recipes-infer/models/gpt-oss
    pip3 install -r requirements.txt
    ```

4. 配置样例运行所需环境信息。

   修改`executor/scripts/set_env.sh`脚本中的如下字段：
   - `cann_path`: CANN软件包安装路径，例如`/usr/local/Ascend/ascend-toolkit/latest`。
    > 说明：HCCL相关配置，如`HCCL_SOCKET_IFNAME`、`HCCL_OP_EXPANSION_MODE`，可以参考[集合通信文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/maintenref/envvar/envref_07_0001.html#ZH-CN_TOPIC_0000002449945377__section163522499503)并在`executor/scripts/function.sh`中自定义配置。

## 权重准备

本样例对GPT-OSS开源模型的原始权重进行了切分与调整，GPT-OSS提供了两个原始权重：
- [GPT-OSS-20B权重](https://huggingface.co/openai/gpt-oss-20b/tree/main)
- [GPT-OSS-120B权重](https://huggingface.co/openai/gpt-oss-120b/tree/main)

开发者可以根据模型任务进行选择，并将原始权重下载至本地路径，例如：`/data/models/gpt-oss-20b-bf16`。

> 注意：原始权重为mxfp4格式，本推理脚本仅支持bf16格式，可参考[官方代码](https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/weights.py)自行转换。


## 推理执行

1. 配置推理执行需要加载的权重文件以及YAML文件。

   - 修改YAML文件中的`model_path`参数。

     在`models/gpt-oss/config`目录下已提供了较优性能的YAML样例供您参考，您可以根据权重的不同选择对应的YAML文件，本文以`gpt_oss_20b.yaml`文件为例，修改其中的`model_path`参数，将其设置为[权重准备](#权重准备)阶段准备好的权重文件存储路径，例如`/data/models/gpt-oss-20b-bf16`。

   - 修改`models/gpt-oss/infer.sh`脚本中`YAML`参数。

     将`YAML`设置为`config`文件夹下YAML文件名称，例如`gpt_oss_20b.yaml`。

2. 准备输入prompt。

   - 使用内置prompt。

     本样例已在`dataset/default_prompt.json`中内置了输入prompt，若您直接使用内置prompt，本步骤可直接跳过。

     当然，您也可以在`dataset/default_prompt.json`文件中自定义prompt输入。

   - 使用长序列prompt。

     本样例默认使用内置prompt，若您需要使用长序列prompt，需要执行以下操作：

     1. 修改YAML文件中的`dataset`参数，将其修改为`dataset: "LongBench"`，使用LongBench数据集作为长序列prompt。

     2. 若您的机器无法联网，需要您从[huggingface](http://huggingface.co/datasets/zai-org/LongBench/tree/main)手动下载数据集至`dataset/LongBench`目录下，`LongBench`文件夹需手工创建，目录中包含`LongBench.py`和`data`目录，并需要在`LongBench.py`中修改数据集加载路径；若您的机器可正常联网，样例执行过程中会自动在线读取LongBench数据集，您无需手工下载。
      > 说明：使用LongBench数据集时，默认执行文本摘要任务，可在`cann-recipes-infer/executor/utils/data_utils.py`的`build_dataset_input`函数里修改默认的system prompt。

3. 执行推理脚本。

    ```shell
    cd models/gpt-oss
    bash infer.sh
    ```

   > **需要注意**
   > - 目前仅支持prompt的batch_size为1。
   > - 默认使用`eager`单算子模式推理。
   > - 对于20b模型提供单device推理，对于120b模型提供8卡推理（仅支持tp切分）。
   > - YAML文件中默认设置`enable_online_split_weight: True`，模型权重会在加载过程中[在线切分](../../docs/common/online_split_weight_guide.md)到各个设备上，无需离线切分。


