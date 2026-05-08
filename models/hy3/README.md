# Hunyuan3 (Hy3-preview) 模型在NPU实现高性能推理

## 概述

Hunyuan3 (Hy3-preview) 是腾讯混元团队开发的大规模 MoE 语言模型，总参数量 295B，激活参数量约 21B/token。本样例基于 Hunyuan3 开源代码完成模型迁移与 NPU 优化适配，在 Atlas A3 8卡上实现了 ge_graph 图模式高性能推理，Decode 性能达到 28.44 ms/t（对比原始 eager 基线 293 ms/t，10.3x 加速）。

模型关键特性：
- MoE 架构：80 层（1 Dense FFN + 79 MoE），192 专家，top-8 sigmoid routing
- GQA Attention：64 Q heads / 8 KV heads，QK Norm before RoPE
- Shared Expert per MoE layer
- 120K 大词表
- 最大上下文 256K

## 支持的产品型号

<term>Atlas A3 系列产品</term>

## 环境准备

1. 安装 CANN 软件包。

   本样例依赖 CANN 开发套件包（cann-toolkit）与 CANN 二进制算子包（cann-kernels），支持的 CANN 软件版本为 `CANN 9.0.0`。

   请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0)下载 `Ascend-cann-toolkit_${version}_linux-${arch}.run` 与 `Ascend-cann-A3-ops_${version}_linux-${arch}.run` 软件包，并参考 [CANN 安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html?OS=openEuler&InstallType=netyum) 进行安装。

    - `${version}` 表示 CANN 包版本号，如 9.0.0。
    - `${arch}` 表示 CPU 架构，如 aarch64、x86_64。

2. 安装 Ascend Extension for PyTorch（torch_npu）。

   Ascend Extension for PyTorch（torch_npu）为支撑 PyTorch 框架运行在 NPU 上的适配插件，本样例支持的 torch_npu 版本为 `v2.8.0.post6`，PyTorch 版本为 `2.8.0`。

   请从[软件包下载地址](https://gitcode.com/Ascend/pytorch/releases/v26.0.0-pytorch2.8.0)下载 `torch_npu-2.8.0.post4-cp311-cp311-manylinux_2_28_${arch}.whl` 安装包，参考 [torch_npu 安装文档](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0004.html) 进行安装。

    - `${arch}` 表示 CPU 架构，如 aarch64、x86_64。

3. 下载项目源码并安装依赖的 Python 库。
    ```bash
    # 下载项目源码，以 master 分支为例
    git clone https://gitcode.com/cann/cann-recipes-infer.git

    # 安装依赖的 Python 库，仅支持 Python 3.11
    cd cann-recipes-infer
    pip3 install -r ./models/hy3/requirements.txt
    ```

4. 配置样例运行所需环境信息。

   修改 `executor/scripts/set_env.sh` 中的如下字段：
   - `IPs`：配置所有节点的 IP，按照 rank id 排序，多个节点的 IP 通过空格分开，例如：`('xxx.xxx.xxx.xxx' 'xxx.xxx.xxx.xxx')`。
   - `cann_path`: CANN 软件包安装路径，例如 `/usr/local/Ascend/ascend-toolkit/latest`。

   > 说明：HCCL 相关配置，如 `HCCL_SOCKET_IFNAME`、`HCCL_OP_EXPANSION_MODE`，可以参考[集合通信文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/envvar/envref_07_0001.html) 并在 `executor/scripts/function.sh` 中自定义配置。

## 权重准备

本样例使用 BF16 原始权重，无需额外权重转换（YAML 中已配置 `enable_online_split_weight: True`，推理启动时自动完成多卡切分）。

请将 Hunyuan3 原始权重下载至本地路径，例如 `/data/models/hy3-preview/`。

## 推理执行

1. 配置推理执行需要加载的权重文件以及 YAML 文件。

   - 修改 YAML 文件中 `model_path` 参数。关于 YAML 文件中的更多配置说明可参见 [YAML 参数描述](./config/README.md)。

     在 `models/hy3/config` 目录下已提供了较优性能的 YAML 样例供您参考，您可以根据场景选择对应的 YAML 文件：

     | YAML 文件 | 执行模式 | 说明 |
     |-----------|---------|------|
     | `hy3_rank16_4tp_16ep_gegraph_bf16.yaml` | ge_graph | **推荐**，图模式 + SuperKernel + compile cache |
     | `config/ci/hy3_rank_16_4tp_16ep_eager.yaml` | eager | 单算子模式，用于调试对比 |
     | `config/ci/hy3_rank_16_4tp_16ep_gegraph.yaml` | ge_graph | CI 配置，batch_size=16 |
     | `config/ci/hy3_rank_16_4tp_16ep_gegraph_longbench.yaml` | ge_graph | LongBench 长序列数据集 |

     本文以 `hy3_rank16_4tp_16ep_gegraph_bf16.yaml` 文件为例，修改其中的 `model_path` 参数，将其设置为[权重准备](#权重准备)阶段准备好的权重文件存储路径，例如 `/data/models/hy3-preview/`。

   - 修改 `models/hy3/infer.sh` 脚本中 `YAML_FILE_NAME` 参数。

     将 `YAML_FILE_NAME` 设置为 `config` 文件夹下 YAML 文件名称，例如 `hy3_rank16_4tp_16ep_gegraph_bf16.yaml`。

2. 准备输入 prompt。

   - 使用内置 prompt。

     本样例已在 `dataset/default_prompt.json` 中内置了输入 prompt，若您直接使用内置 prompt，本步骤可直接跳过。

     当然，您也可以在 `dataset/default_prompt.json` 文件中自定义 prompt 输入。

   - 使用长序列 prompt。

     本样例默认使用内置 prompt，若您需要使用长序列 prompt，可以选择 LongBench 数据集。需要执行以下操作：

     1. 修改 YAML 文件中的 `dataset` 参数，将其修改为 `dataset: "LongBench"`，使用 LongBench 数据集作为长序列 prompt。

     2. 若您的机器无法联网，需要您从 [HuggingFace](https://huggingface.co/datasets/zai-org/LongBench/tree/main) 手动下载数据集至 `dataset/LongBench` 目录下，`LongBench` 文件夹需手工创建，目录中包含 `LongBench.py` 和 `data` 目录，并需要在 `LongBench.py` 中修改数据集加载路径；若您的机器可正常联网，样例执行过程中会自动在线读取 LongBench 数据集，您无需手工下载。

     > 说明：在使用 LongBench 数据集或其他自定义数据集时，默认执行文本摘要任务，可在 `cann-recipes-infer/executor/utils/data_utils.py` 的 `build_dataset_input` 函数里修改默认的 system prompt。

3. 执行推理脚本。

   ```shell
   cd models/hy3
   bash infer.sh
   ```
   > 说明：如果是多机环境，需要在每个节点上执行。

## Benchmark

基于 Atlas A3 8卡，使用 `hy3_rank16_4tp_16ep_gegraph_bf16.yaml` 配置，BF16 精度。

| 指标 | 数值 | 测试条件 |
|------|------|---------|
| Prefill 耗时 | 1,452 ms | 1024 tokens，batch=16 |
| Decode 单步耗时 | **28.44 ms/t** | range: 27.4-29.4 ms/t，32 tokens |
| Decode warmup（首次） | ~91 s | 含图编译 |
| Decode warmup（cached） | ~8 s | compile cache 命中 |
| 对比 eager 基线 | **10.3x 加速** | 原始 eager Decode 293 ms/t |

> 注：性能数据基于 ge_graph 模式 + SuperKernel + compile cache 采集。
