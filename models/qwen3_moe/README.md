#  Qwen3-MoE模型在NPU实现低时延推理

## 概述
Qwen3-MoE模型是2025年开源的大语言模型，包括Qwen3-235B-A22B与Qwen3-30B-A3B两个版本。本样例基于transformers库[modeling_qwen3_moe.py](https://github.com/huggingface/transformers/blob/v4.53.0/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)完成Qwen3-235B-A22B模型的适配优化。


## 支持的产品型号
<term>Atlas A3、950 系列产品</term>  
在950系列产品中，Qwen3-MoE Prefill支持低时延优化、Decode支持W4A8混精推理。

## A3系列产品环境准备

1. 安装CANN软件包。

   本样例的编译执行依赖CANN开发套件包与CANN二进制算子包，支持的CANN软件版本为`CANN 9.0.0`。

   请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0)下载`Ascend-cann-toolkit_${version}_linux-${arch}.run`与`Ascend-cann-A3-ops_${version}_linux-${arch}.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0090.html?OS=Ubuntu&InstallType=localpack)进行安装。

   - `${version}`表示CANN包版本号，如9.0.0。
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
    pip3 install -r ./models/qwen3_moe/requirements.txt
    ```

4. 配置样例运行所需环境信息。

   修改`executor/scripts/set_env.sh`中的如下字段：
   - `IPs`：配置所有节点的IP，按照rank id排序，多个节点的ip通过空格分开，例如：`('xxx.xxx.xxx.xxx' 'xxx.xxx.xxx.xxx')`。
   - `cann_path`: CANN软件包安装路径，例如`/usr/local/Ascend/ascend-toolkit/latest`。
    > 说明：HCCL相关配置，如：`HCCL_SOCKET_IFNAME`、`HCCL_OP_EXPANSION_MODE`，可以参考[集合通信文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/maintenref/envvar/envref_07_0001.html)并在`executor/scripts/function.sh`中自定义配置。

## 950系列产品环境准备
1. 安装CANN软件包。

   本样例的编译执行依赖CANN开发套件包与CANN二进制算子包，支持的CANN软件版本为`CANN 9.1.0-beta.1`。

   请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann)下载`Ascend-cann-toolkit_${version}_linux-${arch}.run`与`Ascend-cann-A3-ops_${version}_linux-${arch}.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/softwareinst/instg/instg_0090.html?OS=openEuler&InstallType=localpack)进行安装。

   - `${version}`表示CANN包版本号，如9.1.0-beta.1。
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
    pip3 install -r ./models/qwen3_moe/requirements.txt
    ```

4. 配置样例运行所需环境信息。

   修改`executor/scripts/set_env.sh`中的如下字段：
   - `IPs`：配置所有节点的IP，按照rank id排序，多个节点的ip通过空格分开，例如：`('xxx.xxx.xxx.xxx' 'xxx.xxx.xxx.xxx')`。
   - `cann_path`: CANN软件包安装路径，例如`/usr/local/Ascend/ascend-toolkit/latest`。
    > 说明：HCCL相关配置，如：`HCCL_SOCKET_IFNAME`、`HCCL_OP_EXPANSION_MODE`，可以参考[集合通信文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/maintenref/envvar/envref_07_0001.html)并在`executor/scripts/function.sh`中自定义配置。
## 权重准备

请根据所使用的模型类型自行下载原始权重到本地路径，例如`/data/models/qwen3_origin_weight/`。

以Qwen3-235B-A22B为例，权重下载地址：[Qwen3-MoE权重](https://huggingface.co/Qwen/Qwen3-235B-A22B/tree/main)。

## 权重转换

本样例在950系列产品上支持Qwen3-MoE模型量化推理，基于`models\qwen3_moe\utils\convert_model.py`可以完成从Bfloat16到W4A8 MXFP8和A4W4 MXFP4的权重转换。默认不增加额外参数时，转换脚本保持原有W4A8 MXFP8流程；只有显式指定`--quant_type w4a4c16`时，才进入A4W4 MXFP4转换。

> 入参介绍：`input_bf16_hf_path`：原始Bfloat16权重路径；`output_hf_path`：转换后输出的权重路径。

如果权重转换的运行环境为NPU，需要先执行：

```bash
cann_path=/usr/local/Ascend/ascend-toolkit/latest # cann包安装路径
source ${cann_path}/bin/setenv.bash
```

权重转换执行示例：

```bash
# 转换为W4A8权重
python models/qwen3_moe/utils/convert_model.py --input_bf16_hf_path /data/models/Qwen3-235B-A22B --output_hf_path /data/models/Qwen3-235B-A22B-MXFP48
```

### MXFP4 + Hadamard 权重转换

当 Attention Linear 采用 A4W4 MXFP4 量化时，低 bit 量化对 outlier 更敏感，可能带来较明显的精度损失；MoE MXFP4 对精度影响相对较小。为降低 Attention Linear 低 bit 量化中的 outlier 影响，可以在离线权重转换阶段融合 Hadamard 变换参数。

1. 下载并解压 Hadamard 参数。

```bash
wget -O quant_params_attn-linear_4_24.rar https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-quantization/quant_params_attn-linear_4_24.rar
mkdir -p /data/models/quant_params_attn-linear_4_24
unrar x quant_params_attn-linear_4_24.rar /data/models/quant_params_attn-linear_4_24/
```

解压后的目录中应包含按层保存的 `layer_<layer_idx>_self_attn.pt` 文件；如果解压后多了一层子目录，请在后续命令中使用实际包含该文件的目录。

2. 转换为 A4W4 MXFP4 权重，并在转换阶段融合 Hadamard 参数。

```bash
cd /path/to/cann-recipes-infer

python models/qwen3_moe/utils/convert_model.py \
  --input_bf16_hf_path /data/models/Qwen3-235B-A22B \
  --output_hf_path /data/models/Qwen3-235B-A22B-W4A4-Hadamard \
  --quant_type w4a4c16 \
  --hadamard_weight_path /data/models/quant_params_attn-linear_4_24
```

3. 推理时使用 MXFP4 + Hadamard YAML 样例。

```yaml
model_config:
  model_path: "/data/models/Qwen3-235B-A22B-W4A4-Hadamard"
  enable_hadamard: True
  hadamard_weight_path: "/data/models/quant_params_attn-linear_4_24"
```

完整 YAML 可参考：

```text
models/qwen3_moe/config/qwen3_235b_mxfp4_hadamard.yaml
```

> 说明：Hadamard 只在 Attention Linear A4W4 MXFP4 通路启用。旧的 W4A8 MXFP8 推理链路即使配置了 `enable_hadamard` 和 `hadamard_weight_path`，也不会进入 Hadamard 分支。

## 推理执行

1. 配置推理执行需要加载的权重文件以及YAML文件。

   - 修改YAML文件中`model_path`参数。关于YAML文件中的更多配置说明可参见[YAML参数描述](../../docs/common/inference_config_guide.md)。

     在`models/qwen3_moe/config`目录下已提供了较优性能的YAML样例供您参考，您可以根据模型类型选择对应的YAML文件，本文以`models/qwen3_moe/config/qwen3_235b_16tp.yaml`文件为例，修改其中的`model_path`参数，将其设置为[权重准备](#权重准备)阶段准备好的权重文件存储路径，例如`/data/models/qwen3_origin_weight/`。

   - 配置`executor/scripts/infer.sh`脚本中的参数。

     离线推理模式下，将`--yaml`设置为`config`文件夹下YAML文件名称，例如`qwen3_235b_16tp.yaml`。
     在线推理模式下，将`--mode`设置为`online`，`--pd-role`设置为`prefill`或`decode`，可通过`--p-yaml-name`和`--d-yaml-name`指定prefill/decode的YAML文件。

2. 准备输入prompt。

   - 使用长序列prompt（默认）。

     本样例默认使用LongBench数据集作为长序列prompt，YAML配置文件中`dataset`参数已设置为`"LongBench"`。若您的机器无法联网，需要从[huggingface](https://huggingface.co/datasets/zai-org/LongBench/tree/main)手动下载数据集至`dataset/LongBench`目录下，`LongBench`文件夹需手工创建，目录中包含`LongBench.py`和`data`目录，并需要在`LongBench.py`中修改数据集加载路径；若您的机器可正常联网，样例执行过程中会自动在线读取LongBench数据集，您无需手工下载。
     > 说明：在使用LongBench数据集或其他自定义数据集时，默认执行文本摘要任务，可在`cann-recipes-infer/executor/utils/data_utils.py`的`build_dataset_input`函数里修改默认的system prompt。

   - 使用内置prompt。

     若您需要使用内置prompt，需要执行以下操作：

     1. 修改YAML文件中的`dataset`参数，将其修改为`dataset: "default"`。

     2. 本样例已在`dataset/default_prompt.json`中内置了输入prompt，您可以直接使用或在该文件中自定义prompt输入。


3. 执行统一推理脚本。

   统一入口脚本位于 `executor/scripts/infer.sh`，通过以下参数控制启动：

   | 参数 | 含义 | 取值示例 |
   | --- | --- | --- |
   | `--model` | 模型目录名，对应 `models/` 下的子目录 | `qwen3_moe` |
   | `--mode` | 推理模式 | `offline`（离线推理）/ `online`（在线PD分离推理） |
   | `--yaml` | 离线模式：yaml 文件名 | `qwen3_235b_16tp.yaml` |
   | `--pd-role` | 在线模式：PD 角色 | `prefill` / `decode` |
   | `--p-yaml-name` | 可选，在线模式：prefill yaml 文件名，不传则默认 `qwen3_moe_pd/prefill.yaml` | `qwen3_moe_pd/prefill.yaml` |
   | `--d-yaml-name` | 可选，在线模式：decode yaml 文件名，不传则默认 `qwen3_moe_pd/decode.yaml` | `qwen3_moe_pd/decode.yaml` |

   > 在线模式 IP 等更多配置参见 [executor 设计文档 §5.1 启动方式](../../docs/design/executor_design.md#51-启动方式)。

   **使用方式一：命令行传参**
   ```shell
   # offline 模式
   bash executor/scripts/infer.sh --model qwen3_moe --yaml qwen3_235b_16tp.yaml
   # online 模式
   bash executor/scripts/infer.sh --model qwen3_moe --mode online --pd-role prefill
   # online 模式（指定 prefill/decode yaml）
   bash executor/scripts/infer.sh --model qwen3_moe --mode online --pd-role prefill --p-yaml-name qwen3_moe_pd/prefill.yaml --d-yaml-name qwen3_moe_pd/decode.yaml
   ```

   如需查看参数说明，可以执行 `bash executor/scripts/infer.sh --help`。

   **使用方式二：直接修改脚本默认值后执行**
    编辑 executor/scripts/infer.sh，修改 MODEL / MODE / YAML_FILE / PD_ROLE / P_YAML_NAME / D_YAML_NAME 等参数的默认值，例如：
    ```shell
    MODEL=qwen3_moe
    MODE=offline
    YAML_FILE=qwen3_235b_16tp.yaml
    ```
    保存后直接执行：
   ```shell
   bash executor/scripts/infer.sh
   ```

   > 说明：
   > - 如果是多机环境，需要在每个节点上执行。
   > - 推理日志和结果保存在 `models/qwen3_moe/res/` 路径下。

## 优化点参考

本样例采用的详细优化点介绍可参见[基于Atlas A3、950的Qwen3-MoE模型低时延推理性能优化实践](../../docs/models/qwen3-moe/qwen3_moe_optimization.md)。
