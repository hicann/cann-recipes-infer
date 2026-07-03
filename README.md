<h1 align="center">CANN-RECIPES-INFER</h1>

<p align="center">
  基于 CANN 平台的大模型推理优化实践<br>
  覆盖 DeepSeek、Qwen、GLM、Hunyuan 等主流模型，克隆即用，快速上手昇腾 NPU 推理，复现生产级高性能
</p>

<p align="center">
  <a href="https://gitcode.com/cann/cann-recipes-infer/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/Models-23%2B-blueviolet.svg" alt="Models">
</p>

<p align="center">
  <a href="#-概述">📖 概述</a> ·
  <a href="#-快速开始">🚀 快速开始</a> ·
  <a href="#-样例列表">📦 样例列表</a> ·
  <a href="https://gitcode.com/cann/cann-recipes-infer/issues">💬 社区讨论</a>
</p>

---

## 📰 最新动态

- [2026/06] GLM-5.2模型在昇腾Atlas A3系列上已支持推理部署
- [2026/06] HunyuanVideo 模型在昇腾 950PR/DT 系列上支持 **mxfp8 a8w8 量化** 和 Flash Attention mxfp8 激活值量化
- [2026/06] DeepSeek-V4 模型在昇腾 950PR/DT 系列上已支持 **HiF8 推理部署**
- [2026/06] Qwen3.5 模型文生文通路在昇腾 Atlas A3 系列上已完成推理部署
- [2026/05] Hy3-preview 模型基于模型优化 Agent（底座 DeepSeek-V4-Pro），在昇腾 Atlas A3 系列上已完成端到端优化适配，支持推理部署

<details>
<summary>📜 更多历史动态</summary>

- [2026/05] Gemma4-26B-A4B / LongCat-Flash-Lite 模型基于模型优化 Agent，在昇腾 Atlas A2/A3 系列上已完成端到端优化适配，支持推理部署
- [2026/04] DeepSeek-V4 模型在昇腾 Atlas A3 系列和 950PR/DT 系列上已 **0day 支持推理部署**
- [2026/03] SANA-Video 模型在昇腾 Atlas A2 系列上已支持推理部署
- [2026/03] 模型优化 Agent Skills 已开源，支持 Agent 在昇腾 Atlas A2/A3 系列上完成端到端推理优化部署
- [2026/03] HunyuanImage-3.0 模型在昇腾 Atlas A2/A3 系列上已支持 EP 推理部署方案
- [2026/03] Qwen3-8B / Qwen2.5-7B-Instruct 模型在昇腾 Atlas A2/A3 系列上已支持推理部署
- [2026/02] GLM-5 模型在昇腾 Atlas A3 系列上已支持推理部署
- [2026/01] Qwen3-next 模型支持 SGLang 框架下序列并行、MTP、GDN 融合算子、W8A8C8 量化
- [2026/01] HSTU 模型在昇腾 Atlas A2 系列上已支持推理部署
- [2026/01] DeepSeek-OCR-2 模型在昇腾 Atlas A2 系列上已支持推理部署
- [2026/01] LongCat-Flash 模型在昇腾 Atlas A3 系列上已支持 **Attention-FFN Disaggregation (AFD)** 部署模式
- [2025/12] HunyuanImage-3.0 模型在昇腾 Atlas A2/A3 系列上已支持推理部署
- [2025/12] LongCat-Flash 模型在昇腾 Atlas A3 系列上已支持低时延的推理部署
- [2025/12] GPT-OSS-20B / GPT-OSS-120B 在昇腾 Atlas A2 系列上已支持推理部署
- [2025/11] Kimi-K2-Thinking 模型在昇腾 Atlas A3 系列上已 **0day 支持 256K 序列推理部署**，适配原生 W4A16 量化
- [2025/10] DeepSeek-R1 / Kimi-K2 模型在昇腾 Atlas A3 系列上已支持低时延、高吞吐的推理部署
- [2025/10] Wan2.2-I2V 模型支持 Ulysses 序列并行、CFG 并行、VAE 并行，推理代码已开源
- [2025/10] HunyuanVideo 模型支持 Ulysses 序列并行、RingAttention 序列并行、TeaCache 加速，推理代码已开源
- [2025/10] DeepSeek-V3.2-Exp 模型支持 **W8A8C8 量化**，量化算法和推理代码已开源
- [2025/10] Qwen3-MoE 模型在昇腾 Atlas A3 系列上已支持推理部署
- [2025/09] DeepSeek-V3.2-Exp 模型在昇腾 Atlas A3 系列上已 **0day 支持推理部署**

</details>

---

## 📖 概述

cann-recipes-infer 仓库旨在针对 LLM 与多模态模型推理业务中的典型模型、加速算法，提供基于 CANN 平台的优化样例，方便开发者简单、快速、高效地使用 CANN 平台进行模型推理。

仓库包含两种类型的实践：

| 类型 | 说明 | 目录 |
|------|------|------|
| 🔬 原生深度优化 | 基于 PyTorch + 仓库轻量化推理框架（executor / module），从算子到算法全栈优化，方便复现和学习 | `models/` |
| 🌐 外部框架集成 | 展示如何基于 vLLM、SGLang 等外部框架使能优化特性，适合已熟悉对应框架的开发者快速进行昇腾优化 | `integration/` |

### 🗺️ 用户导航

| 你是... | 推荐入口 | 预计耗时 |
|---------|---------|:-------:|
| 👋 初次接触昇腾 | [一站式平台](#-快速开始) → 浏览器即可体验 | 10 min |
| 🏗️ 自有环境部署 | [样例列表](#-样例列表) → 按模型名查找 | 30 min |
| 🚀 掌握优化方案 | [各模型目录下的性能调优文档](./docs/models/) | 按需 |
| ✨ 贡献代码 | [贡献指南](CONTRIBUTION.md) | 15 min |

### 为什么使用 cann-recipes-infer

| 维度 | 说明 |
|------|------|
| ⚡ Day-0 新模型支持 | DeepSeek-V4、Kimi-K2 等热门模型发布当天即完成昇腾适配 |
| 🚀 极致推理性能 | 融合算子、量化、混合切分并行、多流控核等全栈优化，满足生产级吞吐和时延要求 |
| 🔗 生态无缝兼容 | 原生优化只依赖 PyTorch，可快速迁移到基于 Pytorch 的 vLLM / SGLang 等各类生产级框架 |
| 🖥️ 多代际硬件覆盖 | 支持 Atlas A3 / A2 / 950PR·DT，支持不同规模的集群部署 |
| 🤖 Agent 时代就绪 | 持续沉淀模型优化的 Agent Skills，支持 AI Agent 自动化完成新模型的适配与优化 |

---

## 🚀 快速开始

### 一站式平台快速跑通第一个模型

「CANNLab 一站式开发平台」是为开发者提供的 NPU 环境，内部已集成完整的 CANN 环境，可以直接使用。cann-recipes-infer 针对该平台在相应样例 README 中提供了简化的「快速启动」路径，帮助用户最小步骤完成 NPU 推理体验。

**当前支持的模型：**

| 模型实践 | 简介 |
|-----|-----|
|[SANA-Video](models/sana-video/README.md#cannlab一站式开发平台的快速启动)|基于PyTorch框架，在Atlas A2/A3环境中完成SANA-Video单卡文生视频推理，针对CANNLab一站式开发平台场景提供简化的启动流程，帮助用户快速上手完成一次端到端 NPU 推理体验。|
|[HunyuanVideo](models/hunyuan-video/README.md#cannlab一站式开发平台的快速启动)|基于PyTorch框架，在Atlas A2/A3环境中完成HunyuanVideo单卡文生视频推理，针对CANNLab一站式开发平台场景提供简化的启动流程，帮助用户快速上手完成一次端到端 NPU 推理体验。|
|[Wan2.2-I2V](models/wan2.2-i2v/README.md#cannlab一站式开发平台的快速启动)|基于PyTorch框架，在Atlas A2/A3环境中完成Wan2.2-I2V单卡图生视频推理，针对CANNLab一站式开发平台场景提供简化的启动流程，帮助用户快速上手完成一次端到端 NPU 推理体验。|
|[DeepSeek-V4](models/deepseek_v4/README.md#cannlab一站式开发平台指南)|在 Atlas A3 环境中完成 DeepSeek-V4 Flash 模型的8卡推理，针对CANNLab一站式开发平台场景提供标准启动流程和相关配置，帮助用户快速上手完成一次端到端 NPU 推理体验。|
|[Qwen3-8B](models/qwen/README.md#cannlab一站式开发平台指南)|在 Atlas A3 单卡环境中完成 Qwen3-8B 模型推理，针对CANNLab一站式开发平台场景集成 AMCT W8A8 量化与端到端启动流程，帮助用户快速上手完成一次端到端 NPU 推理体验。|

> 🧩 更多模型实践持续扩展中，欢迎在 [Issues](https://gitcode.com/cann/cann-recipes-infer/issues) 反馈优先支持的模型实践。

### 在自有环境上部署

**环境要求**

| 项目 | 要求 |
|------|------|
| 硬件 | Atlas A3 / A2 / 950PR·DT |
| CANN | 推荐 9.0.0 |
| Python | 推荐 3.11 |

> 💡 **说明**：以下样例仅作参考，部分模型需特定硬件或 CANN 版本支持，具体兼容性要求请查看 [各模型目录下的 README 文档](./models)。

**大语言模型（以 Qwen3-MoE 为例）**

```bash
# Step 1: 克隆仓库
git clone https://gitcode.com/cann/cann-recipes-infer.git
cd cann-recipes-infer

# Step 2: 进入模型目录，安装依赖
cd models/qwen3_moe
pip install -r requirements.txt

# Step 3: 执行根目录下的脚本一键启动推理
bash ../../executor/scripts/infer.sh --model qwen3_moe --yaml qwen3_235b_16tp.yaml
```

**多模态生成模型（以 SANA-Video 为例）**

```bash
cd models/sana-video
# 按模型 README 完成权重与环境配置后启动
bash infer.sh
```

---

## 📋 样例列表

| 模型实践 | 简介 |
|------|------|
| [GLM-5.2](models/glm_5_2/README.md)                     |基于Transformers库，沿用 DSA + MoE + MTP 结构，新增 IndexShare（跨层 top-k 复用），支持 W8A8 量化与 KV Offload 长序列部署。
| [DeepSeek-V4](models/deepseek_v4/README.md)             |支持Atlas A3和950PR/DT多代际昇腾芯片，兼具1M长序列推理能力与超低交互时延表现，为DeepSeek模型支持Agentic应用提供计算底座，满足千行百业灵活要求。
| [Qwen3.5](models/qwen3_5/README.md)                 |基于Transformers库，在Atlas A3环境中完成Qwen3.5模型文生文通路适配优化，支持TP/EP并行部署，使能融合算子、图模式编译等优化特性。
| [HunyuanVideo](models/hunyuan-video/README.md)          |基于xDiT框架，在Atlas A2环境中采用了Ulysses序列并行和RingAttention序列并行策略，同时适配了FBCache和TeaCache加速。
| [Qwen Dense (Qwen3-8B / Qwen2.5-7B-Instruct)](models/qwen/README.md)|基于Transformers库，在Atlas A2/A3环境中完成Qwen2/Qwen3 Dense模型推理适配，通过config自动识别模型变体，使能融合算子、图模式编译、Packed Sequence（TND格式）、Page Attention等优化特性。
| [SANA-Video](models/sana-video/README.md)               | 基于PyTorch框架，在Atlas A2/A3环境中完成SANA-Video模型适配和优化，使能NPU融合算子，实现较高的推理性能，支持单机单卡以及单机多卡DP部署。
| [Qwen3-MoE](models/qwen3_moe/README.md)                 |基于Transformers库，在Atlas A3环境中完成Qwen3-235B-A22B模型的适配优化，支持TP或EP部署。
| [HunyuanImage-3.0](models/hunyuan-image-3.0/README.md)  |基于Transformers库，在Atlas A2/A3环境中完成HunyuanImage-3.0模型部署，支持TP和EP并行部署，使能多流并行、CFG并行、VAE并行，同时结合了融合算子、消除冗余算子等优化特性。
| [GLM-5](models/glm_5/README.md)                         |基于Transformers库，在Atlas A3环境中Prefill阶段采用了长序列亲和的CP并行策略，Decode阶段沿用大EP并行，实现了较高的吞吐推理性能。
| [LongCat-Flash](models/longcat_flash/README.md)         |基于Transformers库，在Atlas A3环境中完成LongCat-Flash模型低时延场景适配优化，支持TP或EP部署，使能多流并行、控核、权重预取等优化特性。
| [GPT-OSS](models/gpt_oss/README.md)                     |基于Transformers库，在Atlas A2环境中完成gpt-oss模型部署，其中GPT-OSS-120B模型可以采用8卡部署，GPT-OSS-20B模型可以在单device上进行部署。
| [HSTU](models/hstu/README.md)                           |基于RecSDK库，在Atlas A2环境中完成HSTU模型部署，支持单机单卡和单机多卡部署，使能KV Cache多级缓存管理、支持aclgraph，同时结合了hstu_paged融合算子等优化特性。
| [DeepSeek-OCR-2](integration/vllm/deepseek-ocr-2/README.md) |基于vllm + vllm-ascend库，在Atlas A2环境中完成DeepSeek-OCR-2模型部署，支持单图、PDF文档和批量评估。
| [Kimi-K2-Thinking](models/kimi_k2_thinking/README.md)   |基于Transformers库，在Atlas A3环境中完成Kimi-K2-Thinking 256K模型部署，支持原生量化模式，MOE采用W4A16计算，Attention保留BF16精度模式。最小部署单元为单机，同时支持多机大EP部署模式。
| [DeepSeek-R1/Kimi-K2](models/deepseek_r1/README.md)     |基于Transformers库，在Atlas A3环境中完成DeepSeek-R1/Kimi-K2模型低时延、高吞吐两种场景的适配优化，在Prefill阶段支持DP或TP+SP并行部署，在Decode阶段沿用大EP并行，同时还结合了融合算子和多流并行等优化特性。
| [Wan2.2-I2V](models/wan2.2-i2v/README.md)               |基于Transformers库，在Atlas A2环境中完成Wan2.2-I2V模型的适配优化。
| [DeepSeek-V3.2-Exp](models/deepseek_v3_2_exp/README.md) |基于Transformers库，在Atlas A3环境中Prefill阶段采用了长序列亲和的CP并行策略，Decode阶段沿用大EP并行，同时从整网上设计了新的NPU融合Kernel和多流并行优化，实现了较高的吞吐推理性能。

---

## 📂 目录结构

<details>
<summary>点击展开完整目录树</summary>

```
├── .agents                                     # Agent skills 与 subagent 源码
├── docs                                        # 文档目录
│   ├── agent                                   # Agent 相关设计文档
│   ├── cann                                    # CANN 平台相关文档
│   ├── common                                  # 公共文档（推理配置指南、新模型清单等）
│   ├── design                                  # 设计文档（executor、KV Cache、多流并行等）
│   ├── integration                             # 外部框架集成模型文档目录
│   │   └── sglang                              # SGLang 框架集成模型文档
│   │       ├── dsv4-flash-single-npu-moe-offload
│   │       └── qwen3-next
│   └── models                                  # 原生深度优化模型文档
│       ├── deepseek_r1                         
│       ├── deepseek_v3_2_exp                   
│       ├── deepseek-v4                         
│       └── ...                                 
├── accelerator                                 # 加速算法样例
├── dataset                                     # 数据集和默认 prompt
├── executor                                    # 推理执行框架
│   ├── core                                    # 核心模块
│   │   ├── config                              # 推理配置管理
│   │   ├── engine                              # 执行引擎
│   │   ├── forward_data_info                   # 前向数据信息管理
│   │   ├── kv_cache                            # KV Cache 管理器
│   │   ├── model_worker                        # 模型 worker 与 MTP worker
│   │   └── scheduler                           # 任务调度器
│   ├── model_loader                            # 权重加载与格式适配
│   ├── offline                                 # 离线推理入口与执行逻辑
│   ├── online                                  # 在线推理服务
│   │   ├── kv_transfer                         # KV Transfer 引擎
│   │   └── scheduler                           # Prefill/Decode 调度
│   ├── scripts                                 # 环境与测试脚本
│   ├── utils                                   # 通用工具模块
│   └── model_runner.py                         # ModelRunner 类定义
├── integration                                 # 外部框架集成
│   ├── custom                                  # 自定义集成方案
│   │   └── deepseek-v3-ascend310p  
│   ├── sglang                                  # SGLang 框架集成
│   │   ├── dsv4-flash-single-npu-moe-offload   
│   │   ├── llada2.x                            
│   │   ├── qwen3-moe                           
│   │   └── qwen3-next                          
│   └── vllm                                    # vLLM 框架集成
│       ├── deepseek-ocr-2                      
│       ├── minimax_m2.5_mxfp4                  
│       └── pd-hybrid-dp-ep                     
├── models                                      # 模型脚本目录
│   ├── deepseek_v4                             # DeepSeek-V4 的模型脚本及执行配置
│   ├── deepseek-v4-flash-tilelang-and-inductorAF  # DeepSeek-V4 Flash TileLang/Inductor 算子样例
│   ├── deepseek_r1                             # DeepSeek-R1 的模型脚本及执行配置
│   ├── deepseek_v3_2_exp                       # DeepSeek-V3.2-Exp 的模型脚本及执行配置
│   ├── gemma_4                                 # Gemma4 模型脚本及执行配置
│   ├── glm_5                                   # GLM-5 的模型脚本及执行配置
│   ├── glm_5_2                                 # GLM-5.2 的模型脚本及执行配置
│   ├── gpt_oss                                 # GPT-OSS 的模型脚本及执行配置
│   ├── hstu                                    # HSTU 的模型脚本及执行配置
│   ├── hunyuan-image-3.0                       # HunyuanImage-3.0 的模型脚本及执行配置
│   ├── hunyuan-video                           # HunyuanVideo 的模型脚本及执行配置
│   ├── hy3_preview                             # Hy3-preview 的模型脚本及执行配置
│   ├── kimi_k2_thinking                        # Kimi-K2-Thinking 的模型脚本及执行配置
│   ├── longcat_flash                           # LongCat-Flash 的模型脚本及执行配置
│   ├── longcat_flash_lite                      # LongCat-Flash-Lite 的模型脚本及执行配置
│   ├── pangu_7b                                # PanGu-7B 的模型脚本及执行配置
│   ├── qwen                                    # Qwen2/Qwen3 Dense 模型统一脚本及执行配置
│   ├── qwen3_5                                 # Qwen3.5 的模型脚本及执行配置
│   ├── qwen3_moe                               # Qwen3-MoE 的模型脚本及执行配置
│   ├── sana-video                              # SANA-Video 的模型脚本及执行配置
│   ├── step3p7_flash                           # Step3p7-Flash 的模型脚本及执行配置
│   ├── wan2.2-i2v                              # Wan2.2-I2V 的模型脚本及执行配置
│   └── ...                                     
├── module                                      # Linear 等基础 layer 的类定义
│   ├── blockwise_sparse                        # 稀疏处理模块
│   ├── dit_cache                               # DiT Cache 模块
│   ├── fa_quant                                # FA 量化模块
│   ├── quantization                            # 量化模块（FP8/MXFP8/压缩张量等）
│   ├── unified_sp                              # 统一序列并行模块
│   ├── fuse_moe_gmm.py                         # MoE GMM 融合算子
│   ├── linear.py                               # Linear 类定义
│   ├── utils.py                                # 工具函数
│   └── vae_patch_parallel.py                   # VAE 并行补丁
├── ops                                         # 算子目录
│   ├── ascendc                                 # AscendC 算子（HC、Indexer、MoE 等）
│   ├── pypto                                   # PyPTO 算子（Lightning Indexer 等）
│   └── ...
├── AGENTS.md                                   # Agent 使用说明
├── CONTRIBUTION.md                             # 贡献指南
├── DISCLAIMER.md                               # 免责声明
├── LICENSE                                     # 许可证
└── README.md                                   # 项目说明文档
```

</details>

---

## ✨ 参与贡献

欢迎各种形式的贡献：新模型适配、性能优化、文档改进、Bug 反馈。

请参阅 [贡献指南](CONTRIBUTION.md) 了解提交流程和代码规范。

---

## 📝 许可证与声明

### 许可协议适用规则

- 本仓库整体遵循 Apache 2.0 协议，详见 [LICENSE](./LICENSE)。
- 目录包含独立 License 文件时，以该 License 为准；如不存在 License 文件，则遵循 Apache 2.0 协议。

### 其他声明

- 完整免责声明见 [DISCLAIMER](DISCLAIMER.md)

---

<p align="center">
  <sub>Made with ❤️ by the CANN Team · <a href="https://gitcode.com/cann">More CANN Projects</a></sub>
</p>