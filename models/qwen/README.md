# Qwen Dense Models NPU Inference

统一的 Qwen2/Qwen3 Dense（非MoE）模型推理适配，支持以下模型变体：

| 模型 | model_name | 特性 |
|---|---|---|
| Qwen3-8B | `qwen3_8b` | QK-Norm, attention_bias=False |
| Qwen2.5-7B-Instruct | `qwen25_7b_instruct` | 无QK-Norm, attention_bias=True |

## 特性

- 统一建模代码，通过 HuggingFace config.json 自动识别模型变体
- 支持在线权重切分，无需离线预处理
- 支持可选的多卡TP并行部署
- 支持 Packed Sequence（TND格式），Prefill/Decode 阶段均使用打包序列
- 支持 Page Attention 块式KV Cache管理

## 已验证特性

| 特性 | 状态 |
|---|---|
| ge_graph 图模式 | ✅ 已验证 |
| npugraph_ex（含static_kernel） | ✅ 已验证 |
| Packed Sequence (TND) | ✅ 已支持 |
| Page Attention | ✅ 已支持 |

## 支持的产品型号
<term>Atlas A2 系列产品</term>
<term>Atlas A3 系列产品</term>

## 软件版本

|  软件  |    版本    |
|--------|------------|
| CANN   | 8.5.0 |
| torch_npu | 2.8.0 |
| transformers | 4.55.0 |

## 快速开始

### 环境准备

1. 安装CANN软件包。

   本样例的编译执行依赖CANN开发套件包与CANN二进制算子包，支持的CANN软件版本为`CANN 8.5.0`。

   请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0)下载`Ascend-cann-toolkit_${version}_linux-${arch}.run`与`Ascend-cann-${soc}-ops_${version}_linux-${arch}.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Debian&Software=cannToolKit)进行安装。
   
      - `${soc}`表示芯片版本，如910b、A3
      - `${version}`表示CANN包版本号，如8.5.0。
      - `${arch}`表示CPU架构，如aarch64、x86_64。

2. 安装Ascend Extension for PyTorch（torch_npu）。

   Ascend Extension for PyTorch（torch_npu）为支撑PyTorch框架运行在NPU上的适配插件，本样例支持的Ascend Extension for PyTorch版本为`v26.0.0`，PyTorch版本为`2.8.0`。

   请从[软件包下载地址](https://gitcode.com/Ascend/pytorch/releases/v26.0.0-pytorch2.8.0)下载`torch_npu-2.8.0.post4-cp311-cp311-manylinux_2_28_${arch}.whl`安装包，并参考[torch_npu安装文档](https://www.hiascend.com/document/detail/zh/Pytorch/2600/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md)进行安装。

    - `${arch}`表示CPU架构，如aarch64、x86_64。


3. 下载项目源码并安装依赖的python库。
    ```bash
    # 下载项目源码，以master分支为例
    git clone https://gitcode.com/cann/cann-recipes-infer.git

    # 安装依赖的python库，仅支持python 3.11
    cd cann-recipes-infer
    pip3 install -r ./models/qwen/requirements.txt
    ```

4. 配置样例运行所需环境信息。

   修改`executor/scripts/set_env.sh`中的如下字段：
   - `cann_path`: CANN软件包安装路径，例如`/usr/local/Ascend/ascend-toolkit/latest`。
    > 说明：HCCL相关配置，如`HCCL_SOCKET_IFNAME`、`HCCL_OP_EXPANSION_MODE`，可以参考[集合通信文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/envvar/envref_07_0001.html#ZH-CN_TOPIC_0000002449945377__section163522499503)并在`executor/scripts/function.sh`中自定义配置。

### 权重准备

从 HuggingFace 获取原始权重，例如：
- [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

### 配置与执行

1. 配置推理执行需要加载的权重文件以及YAML文件。

   - 修改YAML文件中`model_path`参数。关于YAML文件中的更多配置说明可参见[InferenceConfig使用指南](../../docs/common/inference_config_guide.md)。

     在`models/qwen/config`目录下已提供了不同模型和并行度的YAML样例供您参考：

     **Qwen3-8B:**
     - `qwen3_8b_1tp.yaml`：单卡部署
     - `qwen3_8b_2tp.yaml`：2卡TP并行部署

     **Qwen2.5-7B-Instruct:**
     - `qwen25_7b_instruct_1tp.yaml`：单卡部署
     - `qwen25_7b_instruct_2tp.yaml`：2卡TP并行部署

     将YAML文件中的`model_path`参数设置为权重文件存储路径。

   - 修改`models/qwen/infer.sh`脚本中`YAML_FILE_NAME`参数。

2. 执行推理：

   ```shell
   cd models/qwen
   bash infer.sh
   ```

