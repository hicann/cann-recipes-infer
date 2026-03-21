# NPU 模型优化 Agent Skills 设计文档

## 1. 项目概述

### 1.1 项目定位

本项目是 [cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer) 仓库的 Agent Skills 扩展，基于 CANN 平台能力和仓库已有模型的优化经验，将 NPU 推理优化中的路径知识、阶段依赖和验证流程模块化，支持 Agent 按流程完成端到端推理优化。

### 1.2 核心目标

- 将分散的 NPU 推理优化知识总结组织为可按需加载的 Skill
- 通过阶段化编排和验证驱动，降低优化路径走偏的风险
- 沉淀仓库模型的优化经验为可复用的参考链路
- 建立昇腾社区 NPU 推理优化的 Skill 能力
- 促进 Skill 的协同开发、共享和创新

### 1.3 目标用户

- 使用昇腾 NPU 进行模型推理优化的开发者
- 需要 Agent 辅助完成 NPU 推理适配的团队

## 2. 项目架构设计

### 2.1 整体结构

```
cann-recipes-infer/
├── docs/
│   └── agent/                  # Agent 相关文档
└── .agent/
    ├── skills/                 # Skills 目录
    │   ├── README.md
    │   ├── skill-name-1/
    │   ├── skill-name-2/
    │   └── skill-name-n/
    └── tests/                  # Skill 测试
```

### 2.2 Skill 调度架构

Skill 体系采用 **Agent Team** 模式——由一个编排 Agent 协调多个专业 Agent 协作完成复杂任务。每个 Agent 加载不同的 Skill，各自专注于特定阶段的优化工作，通过共享状态文件协同推进。具体实现为"主 Agent + SubAgent"的调度方式：

- **主 Agent**：加载总入口 Skill，负责全局编排、阶段推进和进度管理
- **SubAgent**：由主 Agent 按需启动，加载特定阶段的 Skill 执行具体任务，完成后将结果返回主 Agent

以当前版本的 5 个 Skill 为例，调度流程如下：

```
用户任务
  │
  ▼
主 Agent（加载 model-optimize）
  │
  ├── 0. 模型分析 + 性能基线（主 Agent 执行）
  │
  ├── 阶段 1：启动 SubAgent ──→ 加载 kvcache-optimization
  │                                │
  │                                ├── 执行 KVCache + FA 优化
  │                                ├── 精度验证 ──失败──→ 启动 SubAgent 加载 kvcache-fa-precision-debug
  │                                └── 结果返回主 Agent
  │
  ├── 主 Agent 确认阶段 1 通过
  │
  ├── 阶段 2：启动 SubAgent ──→ 加载 torch-npu-fusion-optimizer
  │                                │
  │                                ├── 执行融合算子分析与替换
  │                                └── 结果返回主 Agent
  │
  ├── 主 Agent 确认阶段 2 通过
  │
  └── 阶段 3：启动 SubAgent ──→ 加载 graph-mode-adaptation
                                   │
                                   ├── 执行图模式适配
                                   └── 结果返回主 Agent ──→ 输出优化报告
```

这种调度方式的设计考虑：

- **隔离上下文**：每个 SubAgent 只加载对应阶段的 Skill 和 references，避免上下文窗口被无关信息占满
- **阶段间状态传递**：通过 `progress.md` 共享文件传递阶段间的设计决策和验证结果，支持断点接力
- **验证卡点**：主 Agent 在每个阶段之间做验证判断，决定是否进入下一阶段

### 2.3 Skill 分类

| 类型 | Skill | 说明 |
|------|-------|------|
| 编排类 | model-optimize | 总入口，管理阶段流程和进度 |
| 阶段执行类 | kvcache-optimization、torch-npu-fusion-optimizer、graph-mode-adaptation | 各阶段的具体优化实施 |
| 辅助类 | kvcache-fa-precision-debug | 按需触发的精度调试 |

## 3. Skill 标准化设计

### 3.1 遵循标准

本项目遵循 [Agent Skills 规范](https://agentskills.io/home)，确保 Skill 可被支持该规范的工具（如 OpenCode 等）识别和加载。

### 3.2 Skill 目录结构

每个 Skill 采用自包含结构，所有相关资源集中在 Skill 文件夹内：

```
skill-name/
├── SKILL.md              # 技能定义文件（必需）
├── references/           # 参考文档（可选）
│   ├── domain-doc-1.md   #   领域知识、参考链路等
│   └── domain-doc-2.md
├── templates/            # 模板文件（可选）
│   └── report-template.md
└── scripts/              # 辅助脚本（可选）
    └── utils.py          #   调试工具、验证脚本等
```

| 目录/文件 | 必需 | 说明 |
|-----------|------|------|
| `SKILL.md` | 是 | 技能定义，包含 frontmatter 和工作流程 |
| `references/` | 否 | 领域知识文档，供 Agent 按需读取 |
| `templates/` | 否 | 报告模板、代码模板等 |
| `scripts/` | 否 | 辅助工具脚本（调试、验证等） |

### 3.3 SKILL.md 规范

frontmatter 必需字段：

```yaml
---
name: skill-name          # kebab-case，与文件夹名一致
description: 单行描述      # 触发场景关键词，< 1024 字符
user-invocable: true      # 是否可由用户直接调用
---
```

### 3.4 命名规范

- `SKILL.md` 文件名严格区分大小写
- Skill 文件夹使用 kebab-case（如 `kvcache-optimization`）
- Skill 文件夹内不放 `README.md`，文档内容在 `SKILL.md` 或 `references/` 中

## 4. 知识组织设计

### 4.1 三层知识结构

```
SKILL.md（工作流程层）
  │  定义"做什么、怎么做、什么顺序"
  │
  ▼
references/（领域知识层）
  │  仓库模型实现经验的结构化提炼
  │
  ▼
在线文档（接口参考层）
     torch_npu 算子 API、图模式文档
```

SKILL.md 控制流程，references 提供领域知识，在线文档提供接口细节。Agent 按需逐层深入，避免一次性加载全部文档。

### 4.2 在线文档引用

算子接口和图模式文档通过在线链接引用上游仓库，不包含离线副本：

| 文档来源 | 在线地址 |
|---------|---------|
| torch_npu 算子 API | [op-plugin/docs/context/](https://gitcode.com/Ascend/op-plugin/tree/7.3.0/docs/context/) |
| TorchAir 图模式文档 | [torchair/docs/zh/](https://gitcode.com/Ascend/torchair/tree/master/docs/zh) |

### 4.3 references 设计说明

references 是对仓库内模型实现经验的结构化提炼，而非算子文档的离线副本。每个参考文档从仓库已有模型中提取标准链路和最佳实践，让 Agent 在分析新模型时能快速找到最接近的参考实现。

## 5. 验证设计

### 5.1 Skill 有效性验证

采用"有 Skill / 无 Skill"对比测试方法：

- 相同模型、相同基线、相同硬件、相同 Agent 能力
- 唯一变量：是否加载 Skill
- 对比维度：最终性能、中间阶段质量、优化路径完整性

### 5.2 每阶段验证机制

每个阶段完成后必须通过两项验证：

- **精度验证**：Prefill logits cosine similarity、Decode token match 等
- **性能验证**：Prefill/Decode 耗时、显存占用，与上一阶段对比

未通过精度验证时，触发精度调试 Skill 进行排查。

## 6. 技能全景与演进规划

### 6.1 技能全景

当前 Skill 覆盖推理优化的核心三阶段。完整的 NPU 模型适配与优化流程还包括前置阶段和后续进阶特性，规划如下：

```
模型适配全流程：

[前置阶段]                    [当前覆盖]                         [进阶特性]

模型基线迁移适配               阶段 1：KVCache + FA               性能采集与分析
  │                            │                                │
并行化改造                     阶段 2：融合算子优化                多流并行
  │(TP/EP/DP)                  │                                │
  │                            阶段 3：图模式适配                 SuperKernel
  ▼                            │                                │
  ──────────────────────────── ▼ ──────────────────────────────  │
         待添加                    已开源                        Prefetch
                                                                │
                                                              KV Offload
                                                                │
                                                              新增融合算子设计与开发
                                                                │
                                                                ▼
                                                              待添加
```

### 6.2 演进原则

- 每个新 Skill 独立开发、独立验证，不影响已有 Skill
- 前置阶段 Skill 完成后，纳入 model-optimize 的编排流程
- 进阶特性 Skill 作为主要优化流程之后的可选扩展
- 持续通过真实模型测试积累经验，迭代 Skill 编排与 references 中的参考链路

## 7. 协作与贡献

### 7.1 贡献流程

1. 在上游仓库提交 RFC（Issue）描述技能方案
2. Fork 仓库，创建特性分支
3. 按照 Skill 标准化规范开发
4. 提交 PR 并关联 RFC Issue
5. 代码审核与合并

### 7.2 审核标准

- SKILL.md frontmatter 格式正确
- 命名规范符合 kebab-case
- references 内容准确，在线链接有效
- 不包含离线副本文档和二进制文件
