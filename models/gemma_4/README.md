# Gemma-4 模型在 NPU 上推理

## 概述

Gemma-4 是 Google 于 2026 年开源的多模态稀疏 MoE 大语言模型，包含视觉编码器与语言 MoE 解码器两部分。本样例仅适配 Language MoE Decoder 路径，完成对应的 NPU 推理优化适配；视觉编码器暂不覆盖。

- HuggingFace: https://huggingface.co/google/gemma-4-26B-A4B
- 架构: MoE（128 experts top-8，sliding/full 双模式 GQA Attention，每层 Dense MLP 与 MoE 两条支路并行、输出相加）
- 参数量: 26.5B（活跃 ~3.8B/token，BF16 权重约 51.6 GB）

## 支持的产品型号

<term>Atlas A2 系列产品</term>
<term>Atlas A3 系列产品</term>

## Agent 优化说明

本样例由 NPU 推理优化 Agent Skills 完成迁移与优化。

- **部署模式**：框架部署，接入 `executor/core/`
- **优化点参考**：本样例已落地的主要优化点（详细方案见 [agentic/optimization_report.md](agentic/optimization_report.md)）：
  - 混合并行部署：MoE 走专家并行（EP），Embedding 与 LM Head 沿词表张量并行（TP），Attention 走数据并行（DP）
  - KVCache 与 Attention：KV Cache 接入 Paged Attention 管理，按 attention 类型分别管理 sliding / full 两套 block pool（两类缓存长度语义不同）；Flash Attention 按层类型选 layout——sliding 走 TND（packed、无 padding），full 因 head_dim=512 暂不在 TND 非 MLA 白名单、保留 BNSD
  - 算子融合：RMSNorm、RoPE、MoE 路由、KV Cache 写入等替换为 torch_npu 提供的融合算子
  - 图模式：Decode 阶段支持 `ge_graph` 与 `npugraph_ex` 两种后端（通过 yaml `exe_mode` 切换），开启 `enable_cache_compile` 后二次启动 warmup 约节省 20×
  - MoE 通信优化：Decode 路径采用 MC2 dispatch/combine 算子完成跨卡路由通信，Prefill 路径采用 double routing（双重路由），与仓内其他 MoE 模型保持一致

agent 各优化阶段过程归档在 [agentic/](agentic/) 目录：

- [agentic/optimization_report.md](agentic/optimization_report.md) — 优化报告
- [agentic/progress.md](agentic/progress.md) — agent 各优化阶段过程归档

## 环境准备

1. 安装 CANN 软件包。

   本样例的编译执行依赖 CANN 开发套件包与 CANN 二进制算子包，支持的 CANN 软件版本为 `CANN 9.0.0`。

   请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0)下载 `Ascend-cann-toolkit_${version}_linux-${arch}.run` 与 `Ascend-cann-A3-ops_${version}_linux-${arch}.run` 软件包，并参考 [CANN 安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0090.html?OS=Ubuntu&InstallType=localpack) 进行安装。

   - `${version}` 表示 CANN 包版本号，如 9.0.0。
   - `${arch}` 表示 CPU 架构，如 aarch64、x86_64。

2. 安装 Ascend Extension for PyTorch（torch_npu）。

   Ascend Extension for PyTorch（torch_npu）为支撑 PyTorch 框架运行在 NPU 上的适配插件，本样例支持的 torch_npu 版本为 `v26.0.0`，PyTorch 版本为 `2.8.0`。

   请从[软件包下载地址](https://gitcode.com/Ascend/pytorch/releases/v26.0.0-pytorch2.8.0)下载 `torch_npu-2.8.0.post4-cp311-cp311-manylinux_2_28_${arch}.whl` 安装包，并参考 [torch_npu 安装文档](https://www.hiascend.com/document/detail/zh/Pytorch/2600/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md) 进行安装。

   - `${arch}` 表示 CPU 架构，如 aarch64、x86_64。

3. 下载项目源码并安装依赖的 Python 库。

    ```bash
    # 下载项目源码，以 master 分支为例
    git clone https://gitcode.com/cann/cann-recipes-infer.git

    # 安装依赖的 Python 库，仅支持 Python 3.11
    cd cann-recipes-infer
    pip3 install -r ./models/gemma_4/requirements.txt
    ```

4. 配置样例运行所需环境信息。

   修改 `executor/scripts/set_env.sh` 中的如下字段：
   - `IPs`：配置所有节点的 IP，按照 rank id 排序，多个节点的 IP 通过空格分开，例如：`('xxx.xxx.xxx.xxx' 'xxx.xxx.xxx.xxx')`。
   - `cann_path`：CANN 软件包安装路径，例如 `/usr/local/Ascend/ascend-toolkit/latest`。

   > 说明：`HCCL_SOCKET_IFNAME` 等网络相关配置可参考[集合通信文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/maintenref/envvar/envref_07_0001.html) 在 `executor/scripts/function.sh` 中按需调整；`HCCL_OP_EXPANSION_MODE` 已由框架按平台默认设置（910c 启用 AIV 展开），一般无需改动。

## 权重准备

从 HuggingFace 下载 Gemma-4 原始权重到本地路径（例如 `/data/models/Gemma-4`），并将 yaml 内 `model_config.model_path` 改为该路径：

- HuggingFace 仓库：https://huggingface.co/google/gemma-4-26B-A4B

```bash
huggingface-cli download google/gemma-4-26B-A4B --local-dir /data/models/Gemma-4
```

下载完成后将仓内提供的对话模板拷贝到权重目录，AutoTokenizer 在 transformers 5.0+ 会自动从权重路径加载该独立的对话模板文件，无需任何代码修改：

```bash
cp models/gemma_4/chat_template.jinja /data/models/Gemma-4/
```

## 推理执行

1. 配置 yaml：`model_config.model_path` 改为本地权重路径（必改）；其他参数按需调整，各字段含义见 [YAML 参数描述](../../docs/common/inference_config_guide.md)。

   配置文件：`config/gemma4_rank_8_8ep_decode.yaml`（8 卡，默认 GE 图模式）。切换执行模式改 `exe_mode` 字段即可（`eager` / `ge_graph` / `npugraph_ex`，npugraph_ex 下可同时开启 `enable_static_kernel`）。

2. 准备推理 prompt：
   - **内置 prompt**：使用 `dataset/default_prompt.json`，无需额外准备
   - **自定义 / 长序列**：替换 `dataset/default_prompt.json` 内容，或在 yaml 内将 `data_config.dataset` 改为对应 benchmark（如 `"LongBench"` / `"InfiniteBench"`）

   > 注意：full 层走 BNSD layout 要求同批次内输入等长（详见 [agentic/optimization_report.md](agentic/optimization_report.md) §9 算子需求 #1），目前 batch 内 prompt 需保持相同长度；待 CANN 后续把 head_dim=512 加入 FA v2 TND 非 MLA 白名单后可支持变长输入。

3. 执行推理：

   统一入口脚本位于 `executor/scripts/infer.sh`，通过 `--model` / `--yaml` 指定模型目录与配置文件：

   ```bash
   bash executor/scripts/infer.sh --model gemma_4 --yaml gemma4_rank_8_8ep_decode.yaml
   ```

## 性能基线

| Quant Mode | Global Batch Size | Seq Length (input/output) | Chips | TPOT (ms) | Throughput (tokens/s) |
|------------|-------------------|---------------------------|-------|-----------|------------------------|
| BF16 | 8 | 256/32 | 8×A3 (ge_graph) | 10.2 | 784 |
| BF16 | 8 | 256/32 | 8×A3 (npugraph_ex) | 11.6 | 690 |
| BF16 | 8 | 256/32 | 8×A2 (ge_graph) | 15.0 | 533 |
| BF16 | 8 | 256/32 | 8×A2 (eager) | 98.5 | 81 |

> 各阶段性能报告见 [agentic/optimization_report.md](agentic/optimization_report.md)。