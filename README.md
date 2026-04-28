# cann-recipes-infer

## 🚀Latest News
- [2026/04] DeepSeek-V4模型在昇腾Atlas A3系列和950PR/DT系列上已0day支持推理部署。
- [2026/03] SANA-Video模型在昇腾Atlas A2系列上已支持推理部署。
- [2026/03] 模型优化 Agent Skills已开源，支持 Agent 在昇腾Atlas A2/A3系列上完成端到端推理优化部署。
- [2026/03] HunyuanImage-3.0模型在昇腾Atlas A2/A3系列上已支持EP推理部署方案。
- [2026/02] GLM-5模型在昇腾Atlas A3系列上已支持推理部署。
- [2026/01] Qwen3-next模型支持SGLang框架下序列并行、MTP、GDN融合算子、W8A8C8量化。
- [2026/01] HSTU模型在昇腾Atlas A2系列上已支持推理部署。
- [2026/01] DeepSeek-OCR-2模型在昇腾Atlas A2系列上已支持推理部署。
- [2026/01] LongCat-Flash模型在昇腾Atlas A3系列上已支持Attention-FFN Disaggregation(AFD)部署模式。
- [2025/12] HunyuanImage-3.0模型在昇腾Atlas A2/A3系列上已支持推理部署。
- [2025/12] LongCat-Flash模型在昇腾Atlas A3系列上已支持低时延的推理部署。
- [2025/12] GPT-OSS-20B/GPT-OSS-120B在昇腾Atlas A2系列上已支持推理部署。
- [2025/11] Kimi-K2-Thinking模型在昇腾Atlas A3系列上已0day支持256K序列推理部署，适配原生W4A16量化。
- [2025/10] DeepSeek-R1/Kimi-K2模型在昇腾Atlas A3系列上已支持低时延、高吞吐的推理部署。
- [2025/10] Wan2.2-I2V模型支持Ulysses序列并行、CFG并行、VAE并行，推理代码已开源。
- [2025/10] HunyuanVideo模型支持Ulysses序列并行、RingAttention序列并行、TeaCache加速，推理代码已开源。
- [2025/10] DeepSeek-V3.2-Exp模型支持W8A8C8量化，量化算法和推理代码已开源。
- [2025/10] Qwen3-MoE模型在昇腾Atlas A3系列上已支持推理部署。
- [2025/09] DeepSeek-V3.2-Exp模型在昇腾Atlas A3系列上已0day支持推理部署。


## 🎉概述
cann-recipes-infer仓库旨在针对LLM与多模态模型推理业务中的典型模型、加速算法，提供基于CANN平台的优化样例，方便开发者简单、快速、高效地使用CANN平台进行模型推理。


## ✨样例列表
|实践|简介|
|-----|-----|
|[DeepSeek-V3.2-Exp](models/deepseek-v3.2-exp/README.md)|基于Transformers库，在Atlas A3环境中Prefill阶段采用了长序列亲和的CP并行策略，Decode阶段沿用大EP并行，同时从整网上设计了新的NPU融合Kernel和多流并行优化，实现了较高的吞吐推理性能。
|[Qwen3-MoE](models/qwen3_moe/README.md)|基于Transformers库，在Atlas A3环境中完成Qwen3-235B-A22B模型的适配优化，支持TP或EP部署。
|[HunyuanVideo](models/hunyuan-video/README.md)|基于xDiT框架，在Atlas A2环境中采用了Ulysses序列并行和RingAttention序列并行测量，同时适配了FBCache和TeaCache加速。
|[Wan2.2-I2V](models/wan2.2-i2v/README.md)|基于Transformers库，在Atlas A2环境中完成Wan2.2-I2V模型的适配优化。
|[DeepSeek-R1/Kimi-K2](models/deepseek_r1/README.md)|基于Transformers库，在Atlas A3环境中完成DeepSeek-R1/Kimi-K2模型低时延、高吞吐两种场景的适配优化，在Prefill阶段支持DP或TP+SP并行部署，在Docede阶段沿用大EP并行，同时还结合了融合算子和多流并行等优化特性。
|[Kimi-K2-Thinking](models/kimi-k2-thinking/README.md)|基于Transformers库，在Atlas A3环境中完成Kimi-K2-Thinking 256K模型部署，支持原生量化模式，MOE采用W4A16计算，Attention保留BF16精度模式。最小部署单元为单机，同时支持多机大EP部署模式。
|[GPT-OSS](models/gpt_oss/README.md)|基于Transformers库，在Atlas A2环境中完成gpt-oss模型部署，其中GPT-OSS-120B模型可以采用8卡部署，GPT-OSS-20B模型可以在单device上进行部署。
|[LongCat-Flash](models/longcat-flash/README.md)|基于Transformers库，在Atlas A3环境中完成LongCat-Flash模型低时延场景适配优化，支持TP或EP部署，使能多流并行、控核、权重预取等优化特性。
|[HunyuanImage-3.0](models/hunyuan-image-3.0/README.md)|基于Transformers库，在Atlas A2/A3环境中完成HunyuanImage-3.0模型部署，支持TP和EP并行部署，使能多流并行、CFG并行、VAE并行，同时结合了融合算子、消除冗余算子等优化特性。
|[DeepSeek-OCR-2](contrib/vllm-deepseek-ocr2/README.md)|基于vllm + vllm-ascend库，在Atlas A2环境中完成DeepSeek-OCR-2模型部署，支持单图、PDF文档和批量评估。
|[HSTU](models/hstu/README.md)|基于RecSDK库，在Atlas A2环境中完成HSTU模型部署，支持单机单卡和单机多卡部署，使能KV Cache多级缓存管理、支持aclgraph，同时结合了hstu_paged融合算子等优化特性。
|[GLM-5](models/glm-5/README.md)|基于Transformers库，在Atlas A3环境中Prefill阶段采用了长序列亲和的CP并行策略，Decode阶段沿用大EP并行，实现了较高的吞吐推理性能。
|[SANA-Video](models/sana-video/README.md)| 基于PyTorch框架，在Atlas A2/A3环境中完成SANA-Video模型适配和优化，使能NPU融合算子，实现较高的推理性能，支持单机单卡以及单机多卡DP部署。
|[DeepSeek-V4](models/deepseek-v4/README.md)|支持Atlas A3和950PR/DT多代际昇腾芯片，兼具1M长序列推理能力与超低交互时延表现，为DeepSeek模型支持Agentic应用提供计算底座，满足千行百业灵活要求。

## 🏃 一站式平台快速体验

「一站式平台」是为开发者提供的 NPU 环境，内部已集成完整的 CANN 环境，可以直接使用。cann-recipes-infer 针对该平台在相应样例 README 中提供了简化的「快速启动」路径，帮助用户最小步骤完成 NPU 推理体验。当前支持的模型正在持续扩展中，敬请关注：

|实践|简介|
|-----|-----|
|[SANA-Video](models/sana-video/README.md#一站式平台的快速启动)|基于PyTorch框架，在Atlas A2/A3环境中完成SANA-Video单卡文生视频推理，针对一站式平台场景提供简化的启动流程，帮助用户快速上手完成一次端到端 NPU 推理体验。|

## 📖目录结构说明
```
├── docs                                        # 文档目录
|  ├── models                                   # 模型文档目录
|  |  ├── deepseek-v4                         # DeepSeek-V4相关文档
|  |  ├── deepseek-v3.2-exp                     # DeepSeek-V3.2-Exp相关文档
|  |  ├── qwen3-moe                             # Qwen3-MoE相关文档
|  |  ├── hunyuan-video                         # HunyuanVideo相关文档
|  |  ├── wan2.2-i2v                            # Wan2.2-I2V相关文档
|  |  ├── deepseek-r1                           # DeepSeek-R1相关文档
|  |  ├── kimi-k2-thinking                      # Kimi-K2-Thinking相关文档
|  |  ├── gpt-oss                               # gpt-oss相关文档
|  |  ├── longcat-flash                         # LongCat-Flash相关文档
|  |  ├── hunyuan-image-3.0                     # HunyuanImage-3.0相关文档
|  |  ├── sana-video                            # SANA-Video相关文档
|  |  └── ...
|  └── common                                   # 公共文档目录
├── accelerator                                 # 加速算法样例
├── executor                                    # ModelRunner等模型执行相关的类定义
|  ├── model_runner.py                          # ModelRunner类定义
│  └── ...
├── models                                      # 模型脚本目录
|  ├── deepseek-v4                            # DeepSeek-V4的模型脚本及执行配置
|  ├── deepseek-v3.2-exp                        # DeepSeek-V3.2-Exp的模型脚本及执行配置
|  ├── qwen3_moe                                # Qwen3-MoE的模型脚本及执行配置
|  ├── qwen3-moe-sglang                         # Qwen3-MoE在sglang上的修改patch及执行配置
|  ├── hunyuan-video                            # HunyuanVideo的模型脚本及执行配置
|  ├── wan2.2-i2v                               # Wan2.2-I2V的模型脚本及执行配置
|  ├── deepseek_r1                              # DeepSeek-R1的模型脚本及执行配置
|  ├── kimi-k2-thinking                         # Kimi-K2-Thinking的模型脚本及执行配置
|  ├── gpt_oss                                  # gpt-oss的模型脚本及执行配置
|  ├── longcat-flash                            # LongCat-Flash的模型脚本及执行配置
|  ├── hunyuan-image-3.0                        # HunyuanImage-3.0的模型脚本及执行配置
|  ├── sana-video                               # SANA-Video的模型脚本及执行配置
│  └── ...
├── module                                      # Linear等基础layer的类定义
│  └── linear.py                                # Linear类定义
│  └── ...
├── ops                                         # 算子目录
|  ├── ascendc                                  # ascendc算子
|  ├── pypto                                    # pypto算子
│  └── tilelang                                 # tilelang算子
└── CONTRIBUTION.md
└── README.md
└── ...
```

## 📝相关信息

- [贡献指南](./CONTRIBUTION.md)
- [许可证](./LICENSE)

    cann-recipes-infer仓涉及的模型，如模型目录下存在License的以该License为准。如模型目录下不存在License的，遵循CANN 2.0协议，对应协议文本可查阅[LICENSE](./LICENSE)
- [免责声明](./DISCLAIMER.md)
