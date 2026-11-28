<h1 align="center">CANN-RECIPES-INFER</h1>

<p align="center">
  Practical inference optimization for large models on CANN<br>
  Get started with Ascend NPU inference and reproduce high-performance deployments for DeepSeek, Qwen, GLM, Hunyuan, and more
</p>

<p align="center">
  <a href="README.md">简体中文</a> | <strong>English</strong>
</p>

<p align="center">
  <a href="https://gitcode.com/cann/cann-recipes-infer/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/Models-23%2B-blueviolet.svg" alt="Models">
</p>

<p align="center">
  <a href="#overview">📖 Overview</a> ·
  <a href="#quick-start">🚀 Quick Start</a> ·
  <a href="#examples">📦 Examples</a> ·
  <a href="https://gitcode.com/cann/cann-recipes-infer/issues">💬 Community Discussions</a> ·
  <a href="https://etherpad-cann.meeting.osinfra.cn/p/sif-recipes">🗓️ Biweekly Meetings</a>
</p>

---

## 📰 Latest News

- [2026/11] [Kimi K3.1 Pro](models/kimi_k3_1/README.md) has **day-zero inference support** on Ascend 950PR/DT, with DSA fused operators, Indexer multistream execution, LI MXFP8 and SFA FP8 KV quantization, and prefill context parallelism (CP).
- [2026/10] [Kimi K3](models/kimi_k3_1/README.md) adds MegaKDA, ReplaySSM, SuperKernel, and W8A8C8 quantization on Ascend 950PR/DT. Combined with multistream optimizations, these deliver an **approximately 2.8× inference speedup**.
- [2026/10] [LongCat-Next](models/longcat_next/README.md) now supports NPU inference on Ascend Atlas A3, including text, image, and speech generation.
- [2026/09] A new [DeepSeek-V4.1-Flash ultra-low-latency inference example](docs/models/deepseek_v4_1/deepseek_v4.1_low_latency_tp_guide.md) uses tensor parallelism (TP) with a global batch size (GBS) of 1 to achieve an equivalent latency of 2.5 ms.
- [2026/09] A new [DeepSeek-V4 Attention–FFN Disaggregation example](integration/vllm/dsv4-afd/deepseek_v4_afd_deployment_guide.md) uses vLLM and afd-plugin on Ascend A5, with synchronous P2P and asynchronous Window modes, dual microbatches, ACLGraph, and DSpark.
- [2026/09] [DeepSeek-V4.1-Flash](integration/vllm/dsv41-flash-single-npu-moe-offload/README.md) now supports **single-card inference** on Ascend 950PR/DT.
- [2026/09] [GLM-5.3-Flash](models/glm_5_3/README.md) now supports **HiF8 inference** on Ascend 950DT.
- [2026/09] [DeepSeek-V4.1-Flash](models/deepseek_v4_1/README.md) now supports **FP8-FP4 inference** on Ascend 950PR/DT, with multimodal understanding and Engram offload.
- [2026/08] [DeepSeek-V4-Flash](integration/sglang/dsv4-flash-single-npu-moe-offload/README.md) now supports **single-card inference** on Ascend 910B/Atlas A3. Attention and hot experts run on the NPU; the remaining MoE experts are offloaded to a Kunpeng CPU for native MXFP4 computation. Decode throughput is approximately 19–22.5 tok/s. **Deployment with the CANNLab lite-infer-and-train image is supported.**
- [2026/08] [BaiLing-V3](integration/vllm/ling-3.0-flash/README.md) has **day-zero inference support** on Ascend Atlas A2/A3.

<details>
<summary>📜 Earlier Updates</summary>

- [2026/07] [daVinci-MagiHuman](models/davinci-magihuman/README.md) now supports inference on Ascend Atlas A2/A3.
- [2026/07] [Kimi K3](models/kimi_k3_1/README.md) has **day-zero inference support** on Ascend 950PR/DT.
- [2026/07] [Ling-2.5](models/bailing_2_5/README.md) now supports inference on Ascend 950DT.
- [2026/07] [Hy3](models/hy3/README.md) has **day-zero inference support** on Atlas A3 and Ascend 950PR/DT.
- [2026/06] [GLM-5.2](models/glm_5_2/README.md) now supports inference on Ascend Atlas A3.
- [2026/06] [HunyuanVideo](models/hunyuan-video/README.md) supports **MXFP8 A8W8 quantization** and Flash Attention MXFP8 activation quantization on Ascend 950PR/DT.
- [2026/06] [DeepSeek-V4](models/deepseek_v4/README.md) now supports **HiF8 inference** on Ascend 950PR/DT.
- [2026/06] [Qwen3.5](models/qwen3_5/README.md) now supports text-to-text inference on Ascend Atlas A3.
- [2026/05] [Gemma4-26B-A4B](models/gemma_4/README.md) and [LongCat-Flash-Lite](models/longcat_flash_lite/README.md) have completed end-to-end adaptation and optimization using the model optimization agent, enabling inference on Ascend Atlas A2/A3.
- [2026/04] [DeepSeek-V4](models/deepseek_v4/README.md) has **day-zero inference support** on Ascend Atlas A3 and 950PR/DT.
- [2026/03] [SANA-Video](models/sana-video/README.md) now supports inference on Ascend Atlas A2.
- [2026/03] Model optimization Agent Skills are now open source, enabling agents to perform end-to-end inference optimization and deployment on Ascend Atlas A2/A3.
- [2026/03] [HunyuanImage-3.0](models/hunyuan-image-3.0/README.md) now supports expert-parallel (EP) inference on Ascend Atlas A2/A3.
- [2026/03] [Qwen3-8B](models/qwen/README.md) and [Qwen2.5-7B-Instruct](models/qwen/README.md) now support inference on Ascend Atlas A2/A3.
- [2026/02] [GLM-5](models/glm_5/README.md) now supports inference on Ascend Atlas A3.
- [2026/01] [Qwen3-next](integration/sglang/qwen3-next/README.md) supports sequence parallelism, MTP, GDN fused operators, and W8A8C8 quantization with SGLang.
- [2026/01] [HSTU](models/hstu/README.md) now supports inference on Ascend Atlas A2.
- [2026/01] [DeepSeek-OCR-2](integration/vllm/deepseek-ocr-2/README.md) now supports inference on Ascend Atlas A2.
- [2026/01] [LongCat-Flash](models/longcat_flash/README.md) now supports **Attention–FFN Disaggregation (AFD)** on Ascend Atlas A3.
- [2025/12] [HunyuanImage-3.0](models/hunyuan-image-3.0/README.md) now supports inference on Ascend Atlas A2/A3.
- [2025/12] [LongCat-Flash](models/longcat_flash/README.md) now supports low-latency inference on Ascend Atlas A3.
- [2025/12] [GPT-OSS-20B](models/gpt_oss/README.md) and [GPT-OSS-120B](models/gpt_oss/README.md) now support inference on Ascend Atlas A2.
- [2025/11] [Kimi-K2-Thinking](models/kimi_k2_thinking/README.md) has **day-zero support for 256K-context inference** on Ascend Atlas A3, with native W4A16 quantization.
- [2025/10] [DeepSeek-R1](models/deepseek_r1/README.md) and [Kimi-K2](models/deepseek_r1/README.md) now support low-latency and high-throughput inference on Ascend Atlas A3.
- [2025/10] [Wan2.2-I2V](models/wan2.2-i2v/README.md) supports Ulysses sequence parallelism, CFG parallelism, and VAE parallelism. The inference code is now open source.
- [2025/10] [HunyuanVideo](models/hunyuan-video/README.md) supports Ulysses and RingAttention sequence parallelism, plus TeaCache acceleration. The inference code is now open source.
- [2025/10] [DeepSeek-V3.2-Exp](models/deepseek_v3_2_exp/README.md) supports **W8A8C8 quantization**. The quantization algorithms and inference code are now open source.
- [2025/10] [Qwen3-MoE](models/qwen3_moe/README.md) now supports inference on Ascend Atlas A3.
- [2025/09] [DeepSeek-V3.2-Exp](models/deepseek_v3_2_exp/README.md) has **day-zero inference support** on Ascend Atlas A3.

</details>

---

<a id="overview"></a>

## 📖 Overview

cann-recipes-infer provides CANN-based optimization examples for representative large language models (LLMs), multimodal models, and inference acceleration algorithms. It helps developers run model inference on CANN quickly and efficiently.

The repository contains two types of examples:

| Type | Description | Directory |
|------|-------------|-----------|
| 🔬 Native optimization | Full-stack optimization, from operators to algorithms, using PyTorch and this repository's lightweight inference framework (`executor` / `module`). Designed for learning and reproducibility. | `models/` |
| 🌐 External framework integration | Examples of enabling optimization features in vLLM, SGLang, and other frameworks. Suitable for developers who already use these frameworks and want to optimize inference on Ascend. | `integration/` |

### 🗺️ Find Your Starting Point

| Your goal | Recommended starting point | Estimated time |
|-----------|----------------------------|:--------------:|
| 👋 Try Ascend for the first time | [CANNLab platform](#quick-start) → get started in your browser | 10 min |
| 🏗️ Deploy in your own environment | [Examples](#examples) → find your model | 30 min |
| 🚀 Learn optimization techniques | [Model performance tuning guides](./docs/models/) | As needed |
| ✨ Contribute code | [Contribution guide](CONTRIBUTION.md) | 15 min |

### Why cann-recipes-infer

| Feature | Description |
|---------|-------------|
| ⚡ Day-zero model support | Ascend adaptations for popular models such as DeepSeek-V4 and Kimi-K2 are available on release day. |
| 🚀 Inference performance | Fused operators, quantization, hybrid parallelism, multistream execution, and compute-core allocation optimizations target production throughput and latency requirements. |
| 🔗 Framework compatibility | Native optimizations are built on PyTorch and can be ported to PyTorch-based production frameworks such as vLLM and SGLang. |
| 🖥️ Multiple hardware generations | Support for Atlas A3, Atlas A2, and Ascend 950PR/DT across different cluster sizes. |
| 🤖 Agent-assisted optimization | A growing collection of model optimization Agent Skills helps AI agents automate model adaptation and optimization. |

---

<a id="quick-start"></a>

## 🚀 Quick Start

### Run Your First Model on CANNLab

The CANNLab development platform provides NPU environments with CANN preinstalled. The corresponding example READMEs include simplified startup instructions to help you complete an end-to-end NPU inference run with minimal setup.

**Currently supported examples:**

| Example | Description |
|---------|-------------|
| [DeepSeek-V4-Flash on a single card](integration/sglang/dsv4-flash-single-npu-moe-offload/README.md#快速开始) | Run DeepSeek-V4-Flash on a single Atlas A3 card, offloading MoE experts to a Kunpeng CPU. The CANNLab instructions start from an unmodified CANN 9.0.0 image and walk through staged installation of dependencies and custom operators. |
| [GLM-5.3-Flash on a single card](integration/sglang/glm53-flash-single-npu-moe-offload/README.md#1-配置文件) | Run GLM-5.3-Flash on a single Atlas A3 card, offloading routed MoE experts to a Kunpeng CPU in MXFP4. Both patches are based on their respective upstream mainlines, with copy-and-paste instructions from source checkout and patching through accuracy evaluation. |
| [SANA-Video](models/sana-video/README.md#cannlab一站式开发平台的快速启动) | Run single-card text-to-video inference with PyTorch on Atlas A2/A3 using a simplified CANNLab startup workflow. |
| [HunyuanVideo](models/hunyuan-video/README.md#cannlab一站式开发平台的快速启动) | Run single-card text-to-video inference with PyTorch on Atlas A2/A3 using a simplified CANNLab startup workflow. |
| [Wan2.2-I2V](models/wan2.2-i2v/README.md#cannlab一站式开发平台的快速启动) | Run single-card image-to-video inference with PyTorch on Atlas A2/A3 using a simplified CANNLab startup workflow. |
| [DeepSeek-V4](models/deepseek_v4/README.md#cannlab一站式开发平台指南) | Run DeepSeek-V4 Flash inference on eight Atlas A3 cards, with standard CANNLab startup instructions and configuration examples. |
| [Qwen3-8B](models/qwen/README.md#cannlab一站式开发平台指南) | Run Qwen3-8B inference on a single Atlas A3 card, with AMCT W8A8 quantization and an end-to-end CANNLab startup workflow. |

> 🧩 More examples are being added. Use [Issues](https://gitcode.com/cann/cann-recipes-infer/issues) to suggest models you would like to see supported.

### Deploy in Your Own Environment

**Environment requirements**

| Component | Requirement |
|-----------|-------------|
| Hardware | Atlas A3 / Atlas A2 / Ascend 950PR/DT |
| CANN | 9.0.0 recommended |
| Python | 3.11 recommended |

> 💡 The commands below are examples. Some models require specific hardware or CANN versions. See the [model-specific READMEs](./models) for compatibility requirements, environment setup, and weight preparation. Linked guides may be in Chinese.

**Large language models: Qwen3-MoE example**

Follow the [Qwen3-MoE deployment guide](models/qwen3_moe/README.md) to prepare the environment and weights. Before launching, set `model_path` in `models/qwen3_moe/config/qwen3_235b_16tp.yaml` to your local weight directory.

```bash
# Step 1: Clone the repository
git clone https://gitcode.com/cann/cann-recipes-infer.git
cd cann-recipes-infer

# Step 2: Install model dependencies from the repository root
pip install -r models/qwen3_moe/requirements.txt

# Step 3: Launch inference through the repository's shared entry point
bash executor/scripts/infer.sh --model qwen3_moe --yaml qwen3_235b_16tp.yaml
```

**Multimodal generation: SANA-Video example**

```bash
# Start from the repository root
cd models/sana-video
# Prepare the weights and environment as described in the model README
bash infer.sh
```

---

<a id="examples"></a>

## 📋 Examples

| Example | Description |
|---------|-------------|
| [DeepSeek-V4.1-Flash](models/deepseek_v4_1/README.md) | Native multimodal inference on Ascend 950PR/DT, with Engram offload and other optimizations. |
| [daVinci-MagiHuman](models/davinci-magihuman/README.md) | Deployment with Transformers on Atlas A2/A3, with fused operators and other optimizations. |
| [Kimi-K3](models/kimi_k3_1/README.md) | Hybrid-attention MoE inference with Transformers on Ascend 950PR/DT. KDA integrates with the framework's Mamba Cache; routed experts use native MXFP4 weights and dynamic MXFP8 activations. Supports graph-mode compilation and multistream execution. |
| [Ling-2.5](models/bailing_2_5/README.md) | Hybrid-attention MoE inference with Transformers on Ascend 950DT. Alternates GLA linear attention and MLA full attention across layers, with MXFP W8A8 linear layers, MXFP W4A8 routed experts, graph-mode compilation, and multistream execution. |
| [GLM-5.2](models/glm_5_2/README.md) | Built on Transformers with the DSA + MoE + MTP architecture. Adds IndexShare for cross-layer top-k reuse, W8A8 quantization, and KV offload for long-context deployment. |
| [DeepSeek-V4](models/deepseek_v4/README.md) | Inference on Atlas A3 and Ascend 950PR/DT, combining 1M-token context support with low interaction latency for agentic applications. |
| [DeepSeek-V4 AFD (Attention–FFN Disaggregation)](integration/vllm/dsv4-afd/deepseek_v4_afd_deployment_guide.md) | Deploys Attention and FFN separately on Ascend A5 using vLLM, vLLM-Ascend, and afd-plugin. Includes synchronous P2P and asynchronous Window configurations, dual microbatches, ACLGraph, and DSpark. See the [design guide](docs/models/deepseek_v4/deepseek_v4_afd_guide.md). |
| [DeepSeek-V4-Flash on a single card (NPU + CPU MoE offload)](integration/sglang/dsv4-flash-single-npu-moe-offload/README.md) | SGLang inference on a single Atlas A3 card (or 910B) with a Kunpeng CPU. Attention, shared experts, routers, and hot experts use NPU W8A8; other experts use native MXFP4 on the CPU. Includes dynamic hot-expert residency, streaming long-context prefill, online conversion with AscendC MXFP4 operators, and CPU–NPU multistream overlap. Measured A3 decode throughput is approximately 19–22.5 tok/s. See the [deployment guide](docs/integration/sglang/dsv4-flash-single-npu-moe-offload/dsv4_flash_single_card_inference_guide.md). |
| [GLM-5.3-Flash on a single card (NPU + CPU MoE offload)](integration/sglang/glm53-flash-single-npu-moe-offload/README.md) | SGLang inference on a single Atlas A3 die with a Kunpeng CPU. Attention, dense layers, and resident experts use NPU INT8 W8A8; other routed experts are offloaded to host DDR for MXFP4 computation. Supports streaming prefill and dynamic hot-expert residency. Both patches use upstream mainline baselines (ktransformers `6d460cc1` / sglang `5aab054e`). Reported perplexity over 32 windows is 3.5888, with all 11 acceptance checks passing. See the [example README](integration/sglang/glm53-flash-single-npu-moe-offload/README.md#1-配置文件) and [design document](docs/integration/sglang/glm53-flash-single-npu-moe-offload/glm53_flash_single_card_design.md). |
| [DeepSeek-V4.1-Flash on a single card (NPU + CPU MoE offload)](integration/vllm/dsv41-flash-single-npu-moe-offload/README.md) | vLLM deployment on a single Ascend 950PR card with an x86 CPU, supporting CPU MoE offload, dynamic hot experts, Engram storage in RAM or on SSD, and DSpark K2/K5. See the [deployment guide](integration/vllm/dsv41-flash-single-npu-moe-offload/README.md) and [design document](docs/models/deepseek_v4_1/dsv41-flash-950-single-card.md). |
| [Qwen3.5](models/qwen3_5/README.md) | Text-to-text inference adaptation with Transformers on Atlas A3, supporting TP/EP deployment, fused operators, and graph-mode compilation. |
| [HunyuanVideo](models/hunyuan-video/README.md) | Built on xDiT for Atlas A2, with Ulysses and RingAttention sequence parallelism, FBCache, and TeaCache acceleration. |
| [Qwen Dense (Qwen3-8B / Qwen2.5-7B-Instruct)](models/qwen/README.md) | Qwen2/Qwen3 Dense inference with Transformers on Atlas A2/A3. Automatically identifies model variants from their configuration, with fused operators, graph-mode compilation, packed sequences in TND format, and paged attention. |
| [SANA-Video](models/sana-video/README.md) | PyTorch-based adaptation and optimization on Atlas A2/A3 with NPU fused operators. Supports single-node, single-card inference and single-node, multi-card data parallelism (DP). |
| [Qwen3-MoE](models/qwen3_moe/README.md) | Qwen3-235B-A22B adaptation with Transformers on Atlas A3, supporting tensor-parallel (TP) or expert-parallel (EP) deployment. |
| [HunyuanImage-3.0](models/hunyuan-image-3.0/README.md) | Transformers-based deployment on Atlas A2/A3, with TP/EP, multistream execution, CFG parallelism, VAE parallelism, fused operators, and redundant-operator elimination. |
| [GLM-5](models/glm_5/README.md) | High-throughput inference with Transformers on Atlas A3. Uses long-context-friendly context parallelism (CP) for prefill and large-scale EP for decode. |
| [LongCat-Flash](models/longcat_flash/README.md) | Low-latency inference with Transformers on Atlas A3, with TP/EP, multistream execution, compute-core allocation, and weight prefetching. |
| [GPT-OSS](models/gpt_oss/README.md) | Transformers-based inference on Atlas A2. GPT-OSS-120B supports eight-card deployment; GPT-OSS-20B can run on a single device. |
| [HSTU](models/hstu/README.md) | RecSDK-based inference on Atlas A2, supporting single-node, single-card and multi-card deployment, hierarchical KV cache management, ACLGraph, and the `hstu_paged` fused operator. |
| [DeepSeek-OCR-2](integration/vllm/deepseek-ocr-2/README.md) | Deployment with vLLM and vLLM-Ascend on Atlas A2, supporting single images, PDF documents, and batch evaluation. |
| [Kimi-K2-Thinking](models/kimi_k2_thinking/README.md) | Transformers-based deployment of the 256K-context model on Atlas A3, using native W4A16 MoE computation and BF16 attention. Requires at least one node and also supports multi-node, large-scale EP deployment. |
| [DeepSeek-R1/Kimi-K2](models/deepseek_r1/README.md) | Low-latency and high-throughput inference with Transformers on Atlas A3. Supports DP or TP + sequence parallelism (SP) for prefill, large-scale EP for decode, fused operators, and multistream execution. |
| [Wan2.2-I2V](models/wan2.2-i2v/README.md) | Inference adaptation and optimization with Transformers on Atlas A2. |
| [DeepSeek-V3.2-Exp](models/deepseek_v3_2_exp/README.md) | High-throughput inference with Transformers on Atlas A3, using long-context-friendly CP for prefill and large-scale EP for decode, plus model-wide NPU fused kernels and multistream optimizations. |
| [Hy3](models/hy3/README.md) | NPU adaptation of Tencent Hunyuan Hy3 with Transformers on Atlas A3 and Ascend 950PR/DT. A 295B-parameter MoE model with approximately 21B active parameters per token. |
| [Gemma-4](models/gemma_4/README.md) | Transformers-based NPU inference optimization for the Language MoE Decoder path of Google's open-source multimodal sparse MoE model. |
| [Gemma-4-E2B](models/gemma_4_e2b/README.md) | BF16 text-only offline inference on a single logical Atlas A2 device, supporting per-layer inputs, cross-layer KV sharing, double-width MLPs, and eager / `npugraph_ex` execution modes. |
| [LongCat-Flash-Lite](models/longcat_flash_lite/README.md) | Transformers-based NPU adaptation of the MLA + Sparse MoE + N-gram Embedding architecture. Includes paged-attention cache management, fused operators, graph-mode acceleration, EP, W8A8 quantization, and multistream overlap during decode. |
| [LongCat-Next](models/longcat_next/README.md) | Transformers-based NPU inference on Atlas A3, with image and audio inputs, text/image/speech generation, and TP/EP deployment. |
| [Step-3.7-Flash](models/step3p7_flash/README.md) | Transformers-based inference for a multimodal MoE model on Atlas A3: a vision encoder, an MoE text backbone, and three MTP layers, totaling approximately 197B parameters. Uses eight cards / 16 ranks and supports text-only and image-text inputs. |
| [Pangu7B](models/pangu_7b/README.md) | Low-latency NPU inference adaptation and optimization for Huawei's open-source Pangu 7B model, based on its official Hugging Face implementation. |
| [DeepSeek-V4-TileLang/Inductor](models/deepseek_v4_flash_tilelang_and_inductor_af/README.md) | Demonstrates DeepSeek acceleration on Ascend A3 NPUs with TileLang-Ascend operator replacements and Inductor + AutoFuse compilation. |

---

## 📂 Repository Structure

<details>
<summary>Expand the directory tree</summary>

```text
├── .agents                                     # Agent skills and subagent sources
├── docs                                        # Documentation
│   ├── agent                                   # Agent design documents
│   ├── cann                                    # CANN platform documentation
│   ├── common                                  # Shared guides (inference configuration, new-model checklist, etc.)
│   ├── design                                  # Design documents (executor, KV cache, multistream execution, etc.)
│   ├── integration                             # Documentation for external framework integrations
│   │   ├── ling-3.0-flash                        # BaiLing-V3 integration documentation
│   │   └── sglang                              # SGLang integration documentation
│   │       ├── dsv4-flash-single-npu-moe-offload
│   │       ├── glm53-flash-single-npu-moe-offload
│   │       └── qwen3-next
│   └── models                                  # Native model optimization documentation
│       ├── deepseek_r1
│       ├── deepseek_v3_2_exp
│       ├── deepseek_v4
│       ├── deepseek_v4_1
│       └── ...
├── accelerator                                 # Acceleration algorithm examples
├── benchmark                                   # EvalScope guides and scripts for online inference accuracy evaluation
│   └── evalscope_scripts                       # EvalScope scripts (LongBench long-context evaluation)
├── dataset                                     # Datasets and default prompts
├── executor                                    # Inference execution framework
│   ├── core                                    # Core modules
│   │   ├── config                              # Inference configuration management
│   │   ├── engine                              # Execution engine
│   │   ├── forward_data_info                   # Forward-pass data management
│   │   ├── kv_cache                            # KV cache managers
│   │   ├── model_worker                        # Model workers and MTP workers
│   │   └── scheduler                           # Task scheduler
│   ├── model_loader                            # Weight loading and format adaptation
│   ├── offline                                 # Offline inference entry point and execution logic
│   ├── online                                  # Online inference service
│   │   ├── kv_transfer                         # KV transfer engines
│   │   └── scheduler                           # Prefill/decode scheduling
│   ├── scripts                                 # Environment setup and test scripts
│   ├── utils                                   # Shared utilities
│   └── model_runner.py                         # ModelRunner class
├── integration                                 # External framework integrations
│   ├── custom                                  # Custom integrations
│   │   └── deepseek-v3-ascend310p
│   ├── sglang                                  # SGLang integrations
│   │   ├── dsv4-flash-single-npu-moe-offload
│   │   ├── glm53-flash-single-npu-moe-offload
│   │   ├── llada2.x
│   │   ├── qwen3-moe
│   │   └── qwen3-next
│   └── vllm                                    # vLLM integrations
│       ├── deepseek-ocr-2
│       ├── dsv4-afd                            # DeepSeek-V4 AFD deployment and startup guides
│       ├── dsv41-flash-single-npu-moe-offload
│       ├── ling-3.0-flash
│       ├── minimax_m2.5_mxfp4
│       └── pd-hybrid-dp-ep
├── models                                      # Model implementations
│   ├── deepseek_v4                             # DeepSeek-V4 model implementation and configurations
│   ├── deepseek_v4_flash_tilelang_and_inductor_af# DeepSeek-V4 Flash TileLang/Inductor operator examples
│   ├── deepseek_r1                             # DeepSeek-R1 model implementation and configurations
│   ├── deepseek_v3_2_exp                       # DeepSeek-V3.2-Exp model implementation and configurations
│   ├── gemma_4                                 # Gemma4 model implementation and configurations
│   ├── gemma_4_e2b                              # Gemma-4-E2B model implementation and configurations
│   ├── glm_5                                   # GLM-5 model implementation and configurations
│   ├── glm_5_2                                 # GLM-5.2 model implementation and configurations
│   ├── gpt_oss                                 # GPT-OSS model implementation and configurations
│   ├── hstu                                    # HSTU model implementation and configurations
│   ├── hunyuan-image-3.0                       # HunyuanImage-3.0 model implementation and configurations
│   ├── hunyuan-video                           # HunyuanVideo model implementation and configurations
│   ├── hy3                                     # Hy3 model implementation and configurations
│   ├── kimi_k2_thinking                        # Kimi-K2-Thinking model implementation and configurations
│   ├── kimi_k3_1                               # Kimi-K3 model implementation and configurations
│   ├── longcat_flash                           # LongCat-Flash model implementation and configurations
│   ├── longcat_flash_lite                      # LongCat-Flash-Lite model implementation and configurations
│   ├── longcat_next                            # LongCat-Next model implementation and configurations
│   ├── pangu_7b                                # PanGu-7B model implementation and configurations
│   ├── qwen                                    # Unified Qwen2/Qwen3 Dense implementation and configurations
│   ├── qwen3_5                                 # Qwen3.5 model implementation and configurations
│   ├── qwen3_moe                               # Qwen3-MoE model implementation and configurations
│   ├── sana-video                              # SANA-Video model implementation and configurations
│   ├── step3p7_flash                           # Step3p7-Flash model implementation and configurations
│   ├── wan2.2-i2v                              # Wan2.2-I2V model implementation and configurations
│   └── ...
├── module                                      # Base layers, including Linear
│   ├── blockwise_sparse                        # Sparse computation modules
│   ├── dit_cache                               # DiT cache modules
│   ├── fa_quant                                # Flash Attention quantization
│   ├── quantization                            # Quantization (FP8, MXFP8, compressed tensors, etc.)
│   ├── unified_sp                              # Unified sequence parallelism
│   ├── fuse_moe_gmm.py                         # Fused MoE GMM operator
│   ├── linear.py                               # Linear class
│   ├── utils.py                                # Utility functions
│   └── vae_patch_parallel.py                   # VAE parallelism patch
├── ops                                         # Operators
│   ├── ascendc                                 # AscendC operators (HC, Indexer, MoE, etc.)
│   ├── pypto                                   # PyPTO operators (Lightning Indexer, etc.)
│   └── ...
├── recipes_ops                                 # Operator implementations for KDA and other models (cannbot_dsl)
├── scripts                                     # Repository utilities (agent client initialization, etc.)
├── tests                                       # Core inference and online service tests (scheduling, KV transfer, etc.)
│   ├── core                                    # Core tests (sequence packing, continuous batching, PD request fields)
│   └── online                                  # Online inference service tests
│       ├── kv_transfer                         # KV transfer, metadata buffer, and bootstrap service tests
│       ├── scheduler                           # Prefill/decode disaggregated scheduler tests
│       └── server                              # Server helpers, PD routing, and DP dispatch tests
├── AGENTS.md                                   # Agent instructions
├── CONTRIBUTION.md                             # Contribution guide
├── DISCLAIMER.md                               # Disclaimer
├── LICENSE                                     # License
├── README.md                                   # Project README in Chinese
└── README_EN.md                                # Project README in English
```

</details>

---

## ✨ Contributing

Contributions are welcome, including new model adaptations, performance optimizations, documentation improvements, and bug reports.

See the [contribution guide](CONTRIBUTION.md) for the submission process and coding conventions.

---

## 📝 License and Disclaimer

### License Applicability

- This repository is licensed under Apache 2.0. See [LICENSE](./LICENSE).
- If a directory contains its own license file, that license applies to the directory. Otherwise, Apache 2.0 applies.

### Disclaimer

- See [DISCLAIMER](DISCLAIMER.md) for the full disclaimer.

---

<p align="center">
  <sub>Made with ❤️ by the CANN Team · <a href="https://gitcode.com/cann">More CANN Projects</a></sub>
</p>
