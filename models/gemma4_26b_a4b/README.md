# Gemma-4模型在NPU上推理

## 概述

Gemma-4-26B-A4B是Google于2026年开源的多模态MoE大语言模型，包含视觉塔与语言MoE解码器两部分。本样例基于[transformers](https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/models/gemma4/modeling_gemma4.py)的实现，仅适配Language MoE Decoder路径（视觉塔暂不适配），并完成对应的NPU优化适配，包括KVCache异构连续缓存管理、融合算子替换、GE图模式与npugraph_ex图模式加速、MoE EP并行（local_experts / MC2 dispatch_v2 双路径自动切换）。

模型规模：总参数量26.5B，每token激活~3.8B（128 experts top-8），权重~51.6 GB（BF16）。架构特点：双模式Attention（25层sliding GQA + 5层full GQA, k_eq_v）、每层Dense MLP与MoE并联。

- 本样例采用的详细优化点介绍可参见[Gemma-4模型推理性能优化实践](optimization_report.md)。

## 支持的产品型号
<term>Atlas A2 系列产品</term>
<term>Atlas A3 系列产品</term>

## 环境准备

1. 安装CANN软件包。

   本样例的编译执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），支持的CANN软件版本为`CANN 8.5.0`。

   请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0)下载`Ascend-cann-toolkit_${version}_linux-${arch}.run`与`Ascend-cann-A3-ops_<version>_linux-<arch>.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0090.html?Mode=PmIns&InstallType=localpack&OS=Ubuntu)进行安装。

    - `${version}`表示CANN包版本号，如8.5.0。
    - `${arch}`表示CPU架构，如aarch64、x86_64。

2. 安装Ascend Extension for PyTorch（torch_npu）。

   Ascend Extension for PyTorch（torch_npu）为支撑PyTorch框架运行在NPU上的适配插件，本样例支持的Ascend Extension for PyTorch版本为`v7.3.0`，PyTorch版本为`2.8.0`。

   请从[软件包下载地址](https://gitcode.com/Ascend/pytorch/releases/v7.3.0-pytorch2.8.0)下载`torch_npu-2.8.0.post2-cp311-cp311-manylinux_2_28_${arch}.whl`安装包，参考[torch_npu安装文档](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md)进行安装。

3. 下载项目源码并安装依赖的python库。
    ```bash
    # 下载项目源码，以master分支为例
    git clone https://gitcode.com/cann/cann-recipes-infer.git

    # 安装依赖的python库
    cd cann-recipes-infer
    pip3 install -r ./models/gemma4_26b_a4b/requirements.txt
    ```

4. 配置样例运行所需环境信息。

   修改`executor/scripts/set_env.sh`中的如下字段：
   - `IPs`：配置所有节点的IP，按照rank id排序，多个节点的ip通过空格分开，例如：`('xxx.xxx.xxx.xxx' 'xxx.xxx.xxx.xxx')`。
   - `cann_path`: CANN软件包安装路径，例如`/usr/local/Ascend/ascend-toolkit/latest`。

## 权重准备

请自行下载Gemma-4-26B-A4B原始权重到本地路径，例如`/data/models/gemma-4-26B-A4B`：
- [Gemma-4-26B-A4B权重](https://huggingface.co/google/gemma-4-26B-A4B/tree/main)

> 注意：单卡部署不可行（51.6 GB > 32 GB HBM），最少 4 卡（BF16），推荐 8 卡。

## 推理执行

1. 配置推理执行需要加载的权重文件以及YAML文件。

   在`models/gemma4_26b_a4b/config`目录下已提供YAML配置供参考：
   - `gemma4_rank_8_8ep_gegraph_decode.yaml`：8卡 ge_graph 模式 + EP=8（默认）
   - `gemma4_rank_8_8ep_decode.yaml`：8卡 eager 模式 + EP=8
   - `gemma4_rank_8_8ep_npugraph_decode.yaml`：8卡 npugraph_ex 模式 + EP=8
   - `gemma4_rank_1_eager.yaml`：单卡 eager 基线（仅用于单卡 functional 验证，非性能场景）

   修改 YAML 文件中`model_config.model_path`参数，将其设置为权重文件存储路径。

2. 执行推理。`infer.sh` 接受 `$1=mode`、`$2=yaml`，与仓库其他模型保持一致。
    ```bash
    cd models/gemma4_26b_a4b
    bash infer.sh                                                  # 默认 8卡 ge_graph + EP=8
    bash infer.sh offline gemma4_rank_8_8ep_decode.yaml            # 8卡 eager + EP=8
    bash infer.sh offline gemma4_rank_8_8ep_npugraph_decode.yaml   # 8卡 npugraph_ex + EP=8
    bash infer.sh offline gemma4_rank_1_eager.yaml                 # 单卡 eager 基线
    ```

## 优化点参考

详见 [optimization_report.md](optimization_report.md)，主要优化包括：
1. 并行化：EP=8（128 experts / 8 = 16 expert/rank）+ embed/lmhead_tp=8 + attn_tp=1
2. KVCache 异构连续缓存（sliding 层与 full 层 head_dim 不同），FA v1 + sliding sparse_mode=4
3. 融合算子全量替换：`add_rms_norm`（60 处/step）/ `rotary_mul`（25 层 sliding 全量 RoPE + 5 层 full attention partial RoPE）/ `moe_gating_top_k_softmax`
4. GE 图模式 Decode 整图 + npugraph_ex 模式（双路径，YAML 切换）
5. MoE EP decode 路径 `local_experts` 与 MC2 `dispatch_v2/combine_v2` 双方案，按 `experts_per_rank ≤ 24` 自动切换

## 性能基线

A2 / A3 实测性能（input_len=256，max_new_tokens=32，BS=8）：

| 平台 | 模式 | Prefill | Decode/token |
|---|---|---|---|
| A2 (Ascend 910B4) | eager 基线 | 312 ms | 98.5 ms |
| A2 (Ascend 910B4) | ge_graph + EP8 | 189 ms | 15 ms |
| A3 (Ascend 910_93) | ge_graph + EP8 | 76 ms | 10.2 ms |
| A3 (Ascend 910_93) | npugraph_ex + EP8 | 102 ms | 11.6 ms |

> 详细性能拆解（含逐阶段贡献、A3 MoE 双路径 mc2 vs local_experts 对比）参见 [optimization_report.md](optimization_report.md)。
