# Kimi-K2-Thinking 模型在 NPU 实现高性能推理

## 概述
本样例基于 CANN 平台对 Kimi-K2-Thinking 模型进行迁移，可在华为 Atlas A3 集群上运行。并行策略和性能优化点详细介绍可参见 [NPU Kimi-K2-Thinking 推理优化实践](../../docs/models/kimi-k2-thinking/kimi_k2_thinking_inference_guide.md)。

## 支持的产品型号
<term>Atlas A3 系列产品</term>

## 环境准备

1. 安装 CANN 软件包。

   本样例的编译执行依赖 CANN 开发套件包与 CANN 二进制算子包，支持的 CANN 软件版本为 `CANN 9.0.0`。

   请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0)下载 `Ascend-cann-toolkit_${version}_linux-${arch}.run` 与 `Ascend-cann-A3-ops_${version}_linux-${arch}.run` 软件包，并参考 [CANN 安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0090.html?OS=Ubuntu&InstallType=localpack)进行安装。

    - `${version}` 表示 CANN 包版本号，如 9.0.0。
    - `${arch}` 表示 CPU 架构，如 aarch64、x86_64。

2. 安装 Ascend Extension for PyTorch（torch_npu）。

   Ascend Extension for PyTorch（torch_npu）为支撑 PyTorch 框架运行在 NPU 上的适配插件，本样例支持的 Ascend Extension for PyTorch 版本为 `v26.0.0`，PyTorch 版本为 `2.8.0`。

   请从[软件包下载地址](https://gitcode.com/Ascend/pytorch/releases/v26.0.0-pytorch2.8.0)下载 `torch_npu-2.8.0.post4-cp311-cp311-manylinux_2_28_${arch}.whl` 安装包，并参考 [torch_npu 安装文档](https://www.hiascend.com/document/detail/zh/Pytorch/2600/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md)进行安装。

    - `${arch}` 表示 CPU 架构，如 aarch64、x86_64。

3. 下载项目源码并安装依赖的 python 库。
    ```bash
    # 下载项目源码，以 master 分支为例
    git clone https://gitcode.com/cann/cann-recipes-infer.git

    # 安装依赖的 python 库，仅支持 python 3.11
    cd cann-recipes-infer
    pip3 install -r ./models/kimi_k2_thinking/requirements.txt
    ```

4. 配置样例运行所需环境信息。

   修改 `executor/scripts/set_env.sh` 中的如下字段:
   - `IPs`：配置所有节点的 IP，按照 rank id 排序，多个节点的 ip 通过空格分开，例如：`('xxx.xxx.xxx.xxx' 'xxx.xxx.xxx.xxx')`。
   - `cann_path`: CANN 软件包安装路径，例如 `/usr/local/Ascend/ascend-toolkit/latest`。

   > 说明：HCCL 相关配置，如：`HCCL_SOCKET_IFNAME`、`HCCL_OP_EXPANSION_MODE`，可以参考[集合通信文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/maintenref/envvar/envref_07_0001.html)并在 `executor/scripts/function.sh` 中自定义配置。

## 权重准备

请下载 [Kimi-K2-Thinking 原始权重](https://huggingface.co/moonshotai/Kimi-K2-Thinking)，并上传到 Atlas A3 各节点某个固定的路径下，比如 `/data/models/Kimi-K2-Thinking`。

## 推理执行

1. 配置推理执行需要加载的权重文件以及 YAML 文件。

   - 修改 YAML 文件参数。

     本文以 `models/kimi_k2_thinking/config/kimi_k2_thinking.yaml` 文件为例，修改其中的 `model_path` 参数，将其设置为权重文件存储路径，例如 `/data/models/Kimi-K2-Thinking`。
     在 `models/kimi_k2_thinking/config` 目录下已提供了较优性能的 YAML 样例供您参考，您可以根据集群规模选择对应的 YAML 文件，具体参数说明如下：
      - YAML 文件中的配置说明可见 [YAML 参数描述](../../docs/common/inference_config_guide.md)。
      - 除框架统一配置之外，还额外支持以下特性，放置在 YAML 文件 `model_config` 的 `custom_params` 字段下：

        | 参数名 | 类型 | 默认值 | 含义 |
        | --- | --- | --- | --- |
        | `enable_mla_prolog` | bool | `false` | 启用 MLA prolog 优化，用融合算子处理 Attention 的前置计算，提高吞吐。 |
        | `enable_multi_streams` | bool | `false` | 在图模式下，启用 MoE 共享专家多流并行计算，提升吞吐。 |
        | `enable_superkernel` | bool | `false` | 在 `ge_graph` 模式下，启用 superkernel 加速，将多个算子融合为大核以提高执行效率（`npugraph_ex` 模式不支持）。 |

   - 配置 `executor/scripts/infer.sh` 脚本中的参数。

     离线推理模式下，将 `--yaml` 设置为 `config` 文件夹下 YAML 文件名称，例如 `kimi_k2_thinking.yaml`。
     在线推理模式下，将 `--mode` 设置为 `online`，`--pd-role` 设置为 `prefill` 或 `decode`，可通过 `--p-yaml-name` 和 `--d-yaml-name` 指定 prefill/decode 的 YAML 文件。

2. 准备输入 prompt。

   本样例默认使用 InfiniteBench 数据集进行长序列推理。需要从[链接](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/blob/main/longbook_qa_eng.jsonl)中下载长序列输入数据集 longbook_qa_eng，并上传到各个节点上新建的路径 `dataset/InfiniteBench` 下。
   ```shell
   mkdir -p dataset/InfiniteBench
   ```

   若您需要使用内置 prompt 或自定义 prompt，可以修改 YAML 文件中的 `dataset` 参数为 `dataset: "default"`，并在 `dataset/default_prompt.json` 文件中自定义 prompt 输入。

3. 执行统一推理脚本。

   统一入口脚本位于 `executor/scripts/infer.sh`，通过以下参数控制启动：

   | 参数 | 含义 | 取值示例 |
   | --- | --- | --- |
   | `--model` | 模型目录名，对应 `models/` 下的子目录 | `kimi_k2_thinking` |
   | `--mode` | 推理模式 | `offline`（离线推理）/ `online`（在线 PD 分离推理） |
   | `--yaml` | 离线模式：yaml 文件名 | `kimi_k2_thinking.yaml` |
   | `--pd-role` | 在线模式：PD 部署角色 | `prefill` / `decode` |
   | `--p-yaml-name` | 可选，在线模式：prefill yaml 文件名，不传则默认 `pd/prefill.yaml` | `pd/prefill.yaml` |
   | `--d-yaml-name` | 可选，在线模式：decode yaml 文件名，不传则默认 `pd/decode.yaml` | `pd/decode.yaml` |

   > 在线模式 IP 等更多配置参见 [executor 设计文档 §5.1 启动方式](../../docs/design/executor_design.md#51-启动方式)。

   **使用方式一：命令行传参**
   ```shell
   # offline 模式
   bash executor/scripts/infer.sh --model kimi_k2_thinking --yaml kimi_k2_thinking.yaml
   # online 模式
   bash executor/scripts/infer.sh --model kimi_k2_thinking --mode online --pd-role prefill
   # online 模式（指定 prefill/decode yaml）
   bash executor/scripts/infer.sh --model kimi_k2_thinking --mode online --pd-role prefill --p-yaml-name pd/prefill.yaml --d-yaml-name pd/decode.yaml
   ```

   如需查看参数说明，可以执行 `bash executor/scripts/infer.sh --help`。

   **使用方式二：直接修改脚本默认值后执行**
    编辑 `executor/scripts/infer.sh`，按照需求修改 MODEL / MODE / YAML_FILE / PD_ROLE / P_YAML_NAME / D_YAML_NAME 等参数的默认值，例如：
    ```shell
    MODEL=kimi_k2_thinking
    MODE=offline
    YAML_FILE=kimi_k2_thinking.yaml
    ```
    保存后直接执行：
   ```shell
   bash executor/scripts/infer.sh
   ```

   > 说明：
   > - 如果是多机环境，需要在每个节点上执行。
   > - 推理日志和结果保存在 `models/kimi_k2_thinking/res/` 路径下。


