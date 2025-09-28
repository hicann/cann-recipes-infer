# cann-recipes-infer

## 🚀Latest News
- [2025/09] DeepSeek-V3.2模型在昇腾Atlas A3系列上已0day支持推理部署。


## 🎉概述
cann-recipes-infer仓库旨在针对LLM与多模态模型推理业务中的典型模型、加速算法，提供基于CANN平台的优化样例，方便开发者简单、快速、高效地使用CANN平台进行模型推理。


## ✨样例列表
|实践|简介|
|-----|-----|
|[DeepSeek-V3.2](models/deepseek-v3.2/README.md)|基于Transformers库，在Atlas A3环境中Prefill阶段采用了长序列亲和的CP并行策略，Decode阶段沿用大EP并行，同时从整网上设计了新的NPU融合Kernel和多流并行优化，实现了较高的吞吐推理性能。


## 📖目录结构说明
```
├── docs                                        # 文档目录
|  ├── models                                   # 模型文档目录
|  |  ├── deepseek-v3.2                         # deepseek-v3.2相关文档
|  └── common                                   # 公共文档目录
├── accelerator                                 # 加速算法样例
├── executor                                    # ModelRunner等模型执行相关的类定义
|  ├── model_runner.py                          # ModelRunner类定义
│  └── ...
├── models                                      # 模型脚本目录
|  ├── deepseek-v3.2                            # deepseek-v3.2的模型脚本及执行配置
├── modules                                     # Linear等基础layer的类定义
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
- [免责声明](./DISCLAIMER.md)
