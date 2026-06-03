# LongCat-Flash-Lite模型在NPU上推理

## 概述

基于 MLA + Sparse MoE + N-gram Embedding 架构的稀疏专家混合大语言模型。本样例基于 LongCat-Flash-Lite 开源代码进行迁移，并完成对应的 NPU 优化适配，覆盖 Paged Attention 缓存管理、融合算子替换、图模式加速、专家并行路径与多 batch 支持。

- HuggingFace: [meituan-longcat/LongCat-Flash-Lite](https://huggingface.co/meituan-longcat/LongCat-Flash-Lite)
- 架构: MoE LLM（MLA + Sparse MoE + N-gram Embedding）
- 参数量: 约 69 B（N-gram Embedding ~31 B，MoE ~34 B）

## 支持的产品型号

<term>Atlas A2 系列产品</term>
<term>Atlas A3 系列产品</term>

## Agent 优化说明

本样例由 NPU 推理优化 Agent Skills 完成迁移与优化。

- **部署模式**：框架部署，接入 `executor/core/`
- **优化点参考**：本样例已落地的主要优化点（详细方案见下方 agentic 报告）：
  - **packed sequence 布局**：MoE、MLA、N-gram Embedding、LM Head 全链路按 token 维拍平的 packed sequence 组织，减少各模块之间的布局转换。
  - **MLA 压缩 KV + Paged Attention**：通过 Paged Attention 框架统一管理 KV cache，每 token 仅缓存 576 维 latent，相比原 4-KV-head GQA 展开的 1280 维节省 55% 显存；Decode 走 absorb 路径，K/V 不再各自展开为多头形态。
  - **融合算子替换**：把 RMSNorm、SwiGLU、MoE 路由 / 分发 / GMM 等热点全部换成 torch_npu 融合算子；A3 Decode 额外启用 `npu_mla_prolog_v3`，把 Q/KV 投影、RMSNorm、RoPE 与 cache 写入合并为一次算子调用。
  - **EP 路径序列并行**：EP 部署下 Prefill 把 attention 输出端的 AllReduce 拆为 ReduceScatter + AllGather，层间张量保持在 TP 维切分，紧随其后的 RMSNorm 与 Dense MLP 算力同步缩减（仅在 `attn_tp > 1 且 attn_dp > 1` 时触发）。
  - **图模式整图**：GE 图把 Decode 前向完整入图，包括 N-gram Embedding 这种含约 200 个小算子的子模块，消除 host 下发的串行开销；同一份模型代码可切换到 `npugraph_ex` 后端（torch_npu 注册的 torch.compile backend）。
  - **MoE EP Decode 通信**：Decode 阶段 MoE 的跨卡专家路由由 MC2 通信算子完成，需独占通信域；A2 平台单卡专家数超过算子上限时，框架自动回退到 double routing，无需改 yaml。

agent 各优化阶段过程归档在 [agentic/](agentic/) 目录：

- [agentic/optimization_report_tp.md](agentic/optimization_report_tp.md) — TP 路径优化报告
- [agentic/optimization_report_ep.md](agentic/optimization_report_ep.md) — EP 路径优化报告
- [agentic/progress_tp.md](agentic/progress_tp.md) / [agentic/progress_ep.md](agentic/progress_ep.md) — 各优化阶段过程归档

## 环境准备

| 项 | 版本 |
|----|----|
| CANN | 9.0.0 |
| torch_npu | v26.0.0 |
| PyTorch | 2.8.0 |
| Python | 3.11 |

1. 安装 CANN 软件包：从 [软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0) 下载 `Ascend-cann-toolkit_${version}_linux-${arch}.run` 与 `Ascend-cann-${soc}-ops_${version}_linux-${arch}.run`，按 [CANN 安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0090.html?OS=Ubuntu&InstallType=localpack) 操作。
   - `${soc}`：芯片版本，如 `910b` / `A3`
   - `${version}`：CANN 包版本号，如 `9.0.0`
   - `${arch}`：CPU 架构，如 `aarch64` / `x86_64`

2. 安装 Ascend Extension for PyTorch（torch_npu，配套 `v26.0.0` / PyTorch `2.8.0`）：从 [软件包下载地址](https://gitcode.com/Ascend/pytorch/releases/v26.0.0-pytorch2.8.0) 下载 `torch_npu-2.8.0.post4-cp311-cp311-manylinux_2_28_${arch}.whl`，按 [torch_npu 安装文档](https://www.hiascend.com/document/detail/zh/Pytorch/2600/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md) 操作。

3. 下载项目源码并安装依赖：

   ```bash
   git clone https://gitcode.com/cann/cann-recipes-infer.git
   cd cann-recipes-infer
   pip3 install -r ./models/longcat_flash_lite/requirements.txt
   ```

4. 配置 `executor/scripts/set_env.sh`：`cann_path` 改为本机 CANN 安装路径（例如 `/usr/local/Ascend/ascend-toolkit/latest`）；多节点部署时按 rank 顺序填 `IPs`，单节点忽略。
5. 通过环境变量 `ASCEND_RT_VISIBLE_DEVICES` 指定与 yaml `parallel_config.world_size` 等量的卡。

   > 说明：多机部署时 `HCCL_SOCKET_IFNAME` 需按本机网卡前缀（如 `enp` / `eth`）在 `executor/scripts/function.sh` 中配置，可参考[集合通信文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/maintenref/envvar/envref_07_0001.html)。

## 权重准备

从 HuggingFace 下载 LongCat-Flash-Lite 原始权重到本地路径（例如 `/data/models/LongCat-Flash-Lite`），并将 yaml 内 `model_config.model_path` 改为该路径：

- HuggingFace 仓库：[meituan-longcat/LongCat-Flash-Lite](https://huggingface.co/meituan-longcat/LongCat-Flash-Lite)

下载方式（任选其一）：

```bash
# 通过 huggingface-cli
pip install -U huggingface_hub
huggingface-cli download meituan-longcat/LongCat-Flash-Lite --local-dir /data/models/LongCat-Flash-Lite

# 或通过 git lfs（需先 git lfs install）
git clone https://huggingface.co/meituan-longcat/LongCat-Flash-Lite /data/models/LongCat-Flash-Lite
```

## 推理执行

1. 配置 yaml：`model_config.model_path` 改为本地权重路径（必改）；其他参数按需调整，本模型特有的开关位于 `model_config.custom_params`。

   配置文件清单：
   - `config/longcat_flash_lite_rank_8_8tp.yaml` — 8 卡 TP，低延迟
   - `config/longcat_flash_lite_rank_8_8ep.yaml` — 8 卡 EP，平衡延迟与吞吐

   > 长序列 / 多 batch 无需单独 yaml：长输入改 `data_config.input_truncated_len` 与 `scheduler_config.max_new_tokens`，多 batch 改 `scheduler_config.batch_size` 即可。

   > 切换图模式：把 yaml 里 `model_config.exe_mode` 从 `"ge_graph"` 改成 `"npugraph_ex"` 即可换到 `npugraph_ex` 后端（torch_npu 提供的 torch.compile backend）；该后端建议同时加 `model_config.enable_static_kernel: true` 以获得最优性能。默认 `ge_graph` 时延更低。
   >
   > 编译缓存：yaml 加 `model_config.enable_cache_compile: true` 将图编译产物缓存到 `config/cache_compile/`，重复启动时复用缓存、跳过重新编译。

2. 准备推理 prompt：
   - **内置 prompt**：使用 `dataset/default_prompt.json`（~256 token 关于 attention 的短句），无需额外准备，适用于 1k yaml。
   - **长序列**：将 `data_config.input_truncated_len` 调到目标输入长度（如 4096），并把 `dataset/default_prompt.json` 替换为相应长度的 prompt；或将 `data_config.dataset` 改为 `"LongBench"` / `"InfiniteBench"` 使用公开 benchmark（需自行下载到本地或通过 HF 拉取）。

   > 若长序列场景出现 OOM，参考 [AGENTS.md 注意事项](../../AGENTS.md) 的 OOM 缓解顺序调整 batch_size / kvp_size / 量化模式。

3. 执行统一推理脚本。

   统一入口 `executor/scripts/infer.sh` 通过 `--model` / `--yaml` 指定模型与配置（本样例仅支持离线推理，`--mode` 默认 `offline`）：

   | 参数 | 含义 | 取值示例 |
   | --- | --- | --- |
   | `--model` | 模型目录名，对应 `models/` 下的子目录 | `longcat_flash_lite` |
   | `--yaml` | `config/` 下的 yaml 文件名 | `longcat_flash_lite_rank_8_8tp.yaml` |

   ```bash
   bash executor/scripts/infer.sh --model longcat_flash_lite --yaml longcat_flash_lite_rank_8_8tp.yaml
   bash executor/scripts/infer.sh --model longcat_flash_lite --yaml longcat_flash_lite_rank_8_8ep.yaml
   ```

   如需查看参数说明，可执行 `bash executor/scripts/infer.sh --help`。

## 性能基线

A3 8 卡实测性能（input_len=1024）。低延迟 / 平衡两行对应 `config/` 下两份 yaml，高吞吐行在 EP yaml 基础上将 `batch_size` 改为 8：

| 配置 | 量化模式 | Global Batch Size | Seq Length (in/out) | 卡数 | Prefill (ms) | TPOT (ms) | 吞吐 (tokens/s) |
|------|---------|-------------------|---------------------|------|--------------|-----------|----------------|
| `rank_8_8tp.yaml`（TP，低延迟） | BF16 | 1 | 1024/128 | 8×A3 | 47.84 | 5.76 | 174 |
| `rank_8_8ep.yaml`（EP，平衡） | BF16 | 2 | 1024/128 | 8×A3 | 67 | 8.10 | 247 |
| `rank_8_8ep.yaml` + `batch_size=8`（EP，高吞吐） | BF16 | 8 | 1024/128 | 8×A3 | 129.38 | 9.54 | 838 |

> 详细性能拆解（含逐阶段贡献、A2 / A3 跨硬件对比）参见 [optimization_report_tp.md](agentic/optimization_report_tp.md)（TP 路径）与 [optimization_report_ep.md](agentic/optimization_report_ep.md)（EP 路径）。
