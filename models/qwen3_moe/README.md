#  Qwen3-MoE模型在NPU实现低时延推理

## 概述
Qwen3-MoE模型是2025年开源的大语言模型，包括Qwen3-235B-A22B与Qwen3-30B-A3B两个版本。本样例基于transformers库[modeling_qwen3_moe.py](https://github.com/huggingface/transformers/blob/v4.53.0/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)完成Qwen3-235B-A22B模型的适配优化。


## 支持的产品型号
<term>Atlas A3 系列产品</term>

## 环境准备

1. 安装CANN软件包。

   本样例的编译执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），支持的CANN软件版本为`CANN 8.3.RC1.alpha002`。

   请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.3.RC1.alpha002)下载`Ascend-cann-toolkit_${version}_linux-${arch}.run`与`Ascend-cann-kernels-${chip_type}_${version}_linux-${arch}.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Debian&Software=cannToolKit)进行安装。

2. 安装Ascend Extension for PyTorch（torch_npu）。

   Ascend Extension for PyTorch（torch_npu）为支撑PyTorch框架运行在NPU上的适配插件，本样例支持的Ascend Extension for PyTorch版本为`7.2.RC1.alpha002`，PyTorch版本为`2.6.0`。

   请从[软件包下载地址](https://gitee.com/ascend/pytorch/tree/v7.2.RC1.alpha002-pytorch2.6.0)下载`v7.2.RC1.alpha002-pytorch2.6.0`源码，参考[源码编译安装](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0005.html)。

3. 下载项目源码并安装依赖的python库。
    ```bash
    # 下载项目源码，以master分支为例
    git clone https://gitcode.com/cann/cann-recipes-infer.git

    # 安装依赖的python库
    cd cann-recipes-infer/models/qwen3_moe
    pip3 install -r requirements.txt
    ```

4. 配置样例运行所需环境信息。

   修改`executor/scripts/set_env.sh`中的如下字段：
   - `IPs`：配置所有节点的IP，按照rank id排序，多个节点的ip通过空格分开，例如：`('xxx.xxx.xxx.xxx' 'xxx.xxx.xxx.xxx')`。
   - `recipes_path`: 当前代码仓根目录，例如`/home/cann-recipes-infer`。
   - `cann_path`: CANN软件包安装路径，例如`/usr/local/Ascend/ascend-toolkit/latest`。
   - `driver_path`: 固件驱动包安装路径，例如`/usr/local/Ascend/driver`。
    > 说明：HCCL相关配置，如：`HCCL_SOCKET_IFNAME`、`HCCL_OP_EXPANSION_MODE`，可以参考[集合通信文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/maintenref/envvar/envref_07_0001.html#ZH-CN_TOPIC_0000002449945377__section163522499503)并在`executor/scripts/function.sh`中自定义配置。

## 权重准备

请根据所使用的模型类型自行下载原始权重到本地路径，例如`/data/models/qwen3_origin_weight/`。

以Qwen3-235B-A22B为例，权重下载地址：[Qwen3-MoE权重](https://huggingface.co/Qwen/Qwen3-235B-A22B/tree/main)。

## 推理执行

1. 配置推理执行需要加载的权重文件以及YAML文件。

   - 修改YAML文件中`model_path`参数。

     在`models/qwen3_moe/config`目录下已提供了较优性能的YAML样例供您参考，您可以根据模型类型选择对应的YAML文件，本文以`models/qwen3_moe/config/qwen3_235b_16tp.yaml`文件为例，修改其中的`model_path`参数，将其设置为[权重准备](#权重准备)阶段准备好的权重文件存储路径，例如`/data/models/qwen3_origin_weight/`。

   - 修改`models/qwen3_moe/infer.sh`脚本中`YAML_FILE_NAME`参数。

     将`YAML_FILE_NAME`设置为`config`文件夹下YAML文件的名字，例如`qwen3_235b_16tp.yaml`。

2. 准备输入prompt。

   - 使用内置prompt。

     本样例已在`dataset/default_prompt.json`中内置了输入prompt，若您直接使用内置prompt，本步骤可直接跳过。

     当然，您也可以在`dataset/default_prompt.json`文件中自定义prompt输入。

   - 使用长序列prompt。

     本样例默认使用内置prompt，若您需要使用长序列prompt，需要执行以下操作：

     1. 修改YAML文件中的`dataset`参数，将其修改为`dataset: "LongBench"`，使用LongBench数据集作为长序列prompt。

     2. 若您的机器无法联网，需要您从[huggingface](http://huggingface.co/datasets/zai-org/LongBench/tree/main)手动下载数据集至`dataset/LongBench`目录下，`LongBench`文件夹需手工创建，目录中包含`LongBench.py`和`data`目录，并需要在`LongBench.py`中修改数据集加载路径；若您的机器可正常联网，样例执行过程中会自动在线读取LongBench数据集，您无需手工下载。
      > 说明：使用LongBench数据集时，默认执行文本摘要任务，可在`cann-recipes-infer/executor/utils/data_utils.py`的`build_dataset_input`函数里修改默认的system prompt。


2. 执行推理脚本。

   ```shell
   cd models/qwen3_moe
   bash infer.sh
   ```

   > **需要注意**
   > - 本样例支持多卡推理（支持TP和EP切分）。
   > - 默认输入长度为4080，最大输出token数为32，decode为单batch推理。
   > - 默认使用`eager`单算子模式推理。

## 优化点参考

本样例采用的详细优化点介绍可参见[基于Atlas A3的Qwen3-MoE模型低时延推理性能优化实践](../../docs/qwen3_moe_optimization.md)。