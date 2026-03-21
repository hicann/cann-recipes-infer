# NPU 模型优化 Agent Skills

## 项目概述

### 项目定位

面向昇腾 NPU 推理优化场景的 Agent Skills 集合，基于 CANN 平台能力和 [cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer) 仓库的模型优化经验，将路径知识、阶段依赖和验证流程模块化，支持 Agent 按流程完成端到端推理优化。

在 AI 智能体（Agent）上下文中，技能（Skills）是为扩展 Agent 能力而设计的模块化功能单元。每个 Skill 封装了指令、元数据及可选资源，当 Agent（如 OpenCode 等）通过意图识别匹配到相关上下文时，自动调用对应的 Skill。

### 核心能力

- **端到端优化编排**：从基线分析到 KVCache 优化、融合算子替换、图模式适配，按阶段推进
- **每阶段验证驱动**：精度验证和性能验证通过后再进入下一阶段，避免问题叠加
- **专有知识注入**：torch_npu 融合算子接口、FA 参数约束、MoE GMM pipeline、图模式前置条件等
- **精度问题诊断**：KVCache 与 FA 精度问题的系统化排查流程

### 核心目标

- 将 NPU 推理优化的知识、流程和经验组织为可复用的 Skill
- 建立昇腾社区 NPU 推理优化的 Skill 能力
- 促进 Skill 的协同开发、共享和创新

### 目标用户

- 使用昇腾 NPU 进行模型推理优化的开发者
- 需要 Agent 辅助完成 NPU 推理适配的团队

## Skills 列表

| Skill | 功能 | 触发场景 |
|-------|------|---------|
| [model-optimize](model-optimize/) | 端到端优化编排入口 | 完整 NPU 推理优化、全流程优化 |
| [kvcache-optimization](kvcache-optimization/) | KVCache 优化 + FA 替换 | KVCache 管理、Paged Attention、FA 融合算子 |
| [torch-npu-fusion-optimizer](torch-npu-fusion-optimizer/) | 融合算子分析与替换 | torch_npu 融合算子替换、MoE/Attention 适配 |
| [graph-mode-adaptation](graph-mode-adaptation/) | 图模式适配 | torch.compile、GE 图模式、图中断修复 |
| [kvcache-fa-precision-debug](kvcache-fa-precision-debug/) | 精度调试 | KVCache/FA 替换后精度异常 |

### 调度关系

```
model-optimize（编排入口）
  ├── 阶段 1：kvcache-optimization
  ├── 阶段 2：torch-npu-fusion-optimizer
  ├── 阶段 3：graph-mode-adaptation
  └── 按需触发：kvcache-fa-precision-debug
```

## 目录结构

```
.agent/skills/
├── README.md                                # 本文件
├── model-optimize/                          # 端到端优化编排
│   ├── SKILL.md
│   └── templates/
│       └── optimization_report_template.md  # 优化报告模板
├── kvcache-optimization/                    # KVCache 优化
│   ├── SKILL.md
│   └── references/
│       └── fa-code-examples.md              # FA 代码示例
├── torch-npu-fusion-optimizer/              # 融合算子优化
│   ├── SKILL.md
│   └── references/
│       ├── module-attention-gqa.md          # GQA Attention 参考链路
│       ├── module-attention-mla-absorb.md   # MLA Absorb 参考链路
│       ├── module-attention-mla-indexer.md  # MLA Indexer 参考链路
│       ├── moe-patterns.md                  # MoE 算子模式详解
│       ├── rotary-embedding-pattern.md      # RoPE 预计算模式
│       └── torch_npu_API/
│           └── torch_npu_list.md            # 算子总表（索引到在线文档）
├── graph-mode-adaptation/                   # 图模式适配
│   ├── SKILL.md
│   └── references/
│       ├── ge-graph-guide.md                # GE 图模式指南
│       ├── llm-model-guide.md               # LLM 模型改造指南
│       └── npugraph_ex-guide.md             # npugraph_ex 指南
└── kvcache-fa-precision-debug/              # 精度调试
    ├── SKILL.md
    └── references/
        └── fa_debug_utils.py                # 精度调试工具函数
```

## SKILL 命名规范

- `SKILL.md` 文件必须严格命名为 `SKILL.md`（区分大小写）
- Skill 文件夹使用**烤串命名法（kebab-case）**，如 `kvcache-optimization`
- Skill 文件夹内不包含 `README.md`，所有文档内容在 `SKILL.md` 中或存放于 `references/` 目录下

## 在线文档引用

Skill 中的算子接口和图模式文档通过在线链接引用，不包含离线副本：

| 文档来源 | 在线地址 |
|---------|---------|
| torch_npu 算子 API | [op-plugin/docs/context/](https://gitcode.com/Ascend/op-plugin/tree/7.3.0/docs/context/) |
| TorchAir 图模式文档 | [torchair/docs/zh/](https://gitcode.com/Ascend/torchair/tree/master/docs/zh) |

## 使用方式

### 安装

将 `.agent/skills/` 目录放置到项目根目录下（本项目已放置，clone 后可直接使用）。支持 [Agent Skills 规范](https://agentskills.io/home) 的工具（如 OpenCode 等）会自动扫描并加载这些 Skill。

### 触发方式

Skill 有两种触发方式：

1. **场景匹配**：当用户的任务描述命中 Skill 的触发场景时，Agent 自动识别并调用对应 Skill。例如用户说"优化这个模型的推理性能"或"帮我把 Attention 替换成 FA 算子"，Agent 会匹配到对应的 Skill。
2. **指定调用**：用户直接指定使用某个 Skill，例如 `/model-optimize` 或 `/torch-npu-fusion-optimizer`。

每个 Skill 的 `SKILL.md` frontmatter 中的 `description` 字段定义了触发场景关键词，具体见 [Skills 列表](#skills-列表) 中的"触发场景"列。

### 示例

**端到端优化**（触发 model-optimize，自动调度三个阶段）：

```
对目标模型进行端到端 NPU 推理优化
```

**单阶段使用**（直接触发特定 Skill）：

```
使用 torch-npu-fusion-optimizer 分析模型的融合算子替换方案
```

```
将模型的 Decode 阶段适配图模式
```


## 免责声明

1. 本目录中的 Agent Skills 内容仅供技术参考和学习使用，不代表其适用于任何生产环境或关键业务系统。
2. 开发者在使用时应自行评估其安全性、兼容性和适用性。作者及贡献者不对因使用本内容导致的任何直接或间接损失承担责任。
3. 本内容可能涉及第三方依赖或接口调用，相关权限及合规性需由开发者自行核实。
4. 除非另有明确约定，本目录所有内容均基于开源协议发布，不提供任何形式的技术支持或担保。
