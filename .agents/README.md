# NPU 模型优化 Agent Skills

## 项目概述

### 项目定位

面向昇腾 NPU 推理优化场景的 Agent Skills 集合。基于 CANN 平台能力和 [cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer) 仓库的模型优化经验，将路径知识、阶段依赖和验证流程模块化，支持 Agent 按流程完成端到端推理优化。

在 AI 智能体（Agent）上下文中，技能（Skills）是为扩展 Agent 能力而设计的模块化功能单元。每个 Skill 封装了指令、元数据及可选资源，当 Agent（如 OpenCode 等）通过意图识别匹配到相关上下文时，自动调用对应的 Skill。

### 仓库能力概览

cann-recipes-infer 仓库支持的核心优化技术：

- **并行策略**：TP（张量并行）、EP（专家并行）、DP（数据并行）等多种并行组合
- **优化技术**：多流并行、融合算子、消除冗余算子
- **加速算法**：FBCache、TeaCache、权重预取、KVP（KV缓存并行）
- **量化支持**：W8A8、W4A16、W8A8C8 等多种量化模式
- **执行模式**：`ge_graph`（图执行）和 `eager`（即时执行）
- **硬件平台**：昇腾 Atlas A2 / A3 系列

### 核心能力

- **端到端优化编排**：从框架适配、基线建立到并行化改造、KVCache 优化、融合算子替换、图模式适配，按阶段推进
- **每阶段验证驱动**：精度验证和性能验证通过后再进入下一阶段，避免问题叠加
- **专有知识注入**：torch_npu 融合算子接口、FA 参数约束、MoE GMM pipeline、图模式前置条件、并行切分策略等
- **问题诊断**：KVCache/FA 精度问题和 NPU 运行时错误的系统化排查流程

### 核心目标

- 将 NPU 推理优化的知识、流程和经验组织为可复用的 Skill
- 建立昇腾社区 NPU 推理优化的 Skill 能力
- 促进 Skill 的协作开发和经验共享

### 目标用户

- 使用昇腾 NPU 进行模型推理优化的开发者
- 需要 Agent 辅助完成模型适配和性能调优的团队

支持单卡和多卡部署。

## Skills 列表

### 主流程 Skills

由 model-infer-optimize 编排调度，按阶段依次执行：

| Skill | 功能 | 触发场景 |
|-------|------|---------|
| [model-infer-optimize](skills/model-infer-optimize/) | 端到端优化编排入口 | 完整 NPU 推理优化、全流程优化 |
| [model-infer-migrator](skills/model-infer-migrator/) | 框架适配与基线建立 | 新模型适配、部署基线采集 |
| [model-infer-parallel-analysis](skills/model-infer-parallel-analysis/) | 并行策略分析 | 确定 TP/EP/DP 并行配置 |
| [model-infer-parallel-impl](skills/model-infer-parallel-impl/) | 并行切分实施 | 并行化代码改造、权重转换 |
| [model-infer-kvcache](skills/model-infer-kvcache/) | KVCache 优化 + FA 替换 | KVCache 管理、Paged Attention、FA 融合算子 |
| [model-infer-fusion](skills/model-infer-fusion/) | 融合算子分析与替换 | torch_npu 融合算子替换、MoE/Attention 适配 |
| [model-infer-graph-mode](skills/model-infer-graph-mode/) | 图模式适配 | torch.compile、GE 图模式、图中断修复 |

### 辅助与调试 Skills

按需触发，不在主流程阶段中：

| Skill | 功能 | 触发场景 |
|-------|------|---------|
| [model-infer-precision-debug](skills/model-infer-precision-debug/) | NPU 推理精度诊断 | 精度验证未通过、NaN/Inf、MoE 精度偏差（当前主要覆盖 KVCache/FA） |
| [model-infer-runtime-debug](skills/model-infer-runtime-debug/) | NPU 运行时错误诊断 | aicore timeout、HCCL 错误、OOM、推理卡住 |

### 独立进阶优化 Skills

独立使用的进阶优化技术，尚未编排进主流程：

| Skill | 功能 | 触发场景 |
|-------|------|---------|
| [model-infer-multi-stream](skills/model-infer-multi-stream/) | 多流并行优化 | NPU 多流表达、算子分流、计算与计算并行 |
| [model-infer-prefetch](skills/model-infer-prefetch/) | 权重预取适配 | 权重预取、prefetch 适配 |
| [model-infer-superkernel](skills/model-infer-superkernel/) | SuperKernel 适配 | SuperKernel scope 划分、NPU 算子融合 |


### 调度关系

```
model-infer-optimize（编排入口）
  ├── 阶段 0：model-infer-migrator（框架适配与基线建立）
  ├── 阶段 1：model-infer-parallel-analysis + model-infer-parallel-impl（并行策略，多卡部署时）
  ├── 阶段 2：model-infer-kvcache（KVCache 优化 + FA 替换）
  ├── 阶段 3：model-infer-fusion（融合算子替换）
  ├── 阶段 4：model-infer-graph-mode（图模式适配）
  └── 按需触发：model-infer-precision-debug（精度诊断）/ model-infer-runtime-debug（运行时错误）
```

### Agent 协作架构

主流程采用"编排 Agent + 三类专业 SubAgent"的协作模式，通过分析/实施/验证的职责隔离，降低单个 Agent 同时承担多角色导致的越界与漏检问题：

| Agent | 角色 | 职责 |
|-------|------|------|
| 主 Agent | 编排者 | 阶段推进、用户确认、进度管理，不直接修改代码 |
| model-infer-analyzer | 分析专家 | 架构分析、方案设计，只读代码，只写 progress.md |
| model-infer-implementer | 实施工程师 | 代码改造、调试修复，读写全部文件 |
| model-infer-reviewer | 审查专家 | 精度验证、性能对比，不修改模型代码 |

每阶段协作流程：主 Agent 派发 analyzer 分析 → 用户确认方案 → 派发 implementer 实施 → 派发 reviewer 验证 → 通过后进入下一阶段。SubAgent 之间通过共享状态文件 `progress.md` 传递上下文。

## 目录结构

```
.agents/
├── README.md                                    # 本文件
├── agents/                                      # SubAgent 角色定义
│   ├── model-infer-analyzer.md                  # 分析专家
│   ├── model-infer-implementer.md               # 实施工程师
│   └── model-infer-reviewer.md                  # 审查专家
├── hooks/                                       # Hook 脚本
├── settings.json                                # Agent 配置
└── skills/                                      # Skills 目录
    ├── model-infer-optimize/                    # 端到端优化编排
    ├── model-infer-migrator/                    # 框架适配与基线建立
    ├── model-infer-parallel-analysis/           # 并行策略分析
    ├── model-infer-parallel-impl/               # 并行切分实施
    ├── model-infer-kvcache/                     # KVCache 优化
    ├── model-infer-fusion/                      # 融合算子优化
    ├── model-infer-graph-mode/                  # 图模式适配
    ├── model-infer-precision-debug/             # NPU 推理精度诊断
    ├── model-infer-runtime-debug/               # NPU 运行时调试
    ├── model-infer-multi-stream/                # 多流并行（独立进阶优化）
    ├── model-infer-prefetch/                    # 权重预取（独立进阶优化）
    ├── model-infer-superkernel/                 # SuperKernel（独立进阶优化）
```

## 在线文档引用

Skill 中引用的外部文档：

| 文档来源 | 在线地址 |
|---------|---------|
| torch_npu 算子 API | [op-plugin/docs/context/](https://gitcode.com/Ascend/op-plugin/tree/7.3.0/docs/context/) |
| TorchAir 图模式文档 | [torchair/docs/zh/](https://gitcode.com/Ascend/torchair/tree/master/docs/zh) |

## 使用方式

### 安装

根据使用的客户端选择对应命令：

```bash
bash scripts/init-agent.sh --claude     # Claude Code 用户
bash scripts/init-agent.sh --opencode   # OpenCode 用户
bash scripts/init-agent.sh              # 两个客户端都要用时（可选）
```

脚本作用：
- 在 `.claude/` 下创建 skills/agents/hooks 的 symlink（指向 `.agents/`），并复制 `settings.json`
- 在 `.opencode/` 下创建 skills/agents 的 symlink
- 根目录生成 `CLAUDE.md` → `AGENTS.md` 的 symlink

生成物（`.claude/` / `.opencode/` / `CLAUDE.md`）已加入仓库 `.gitignore`，不会被提交。

### 触发方式

Skill 有两种触发方式：

1. **场景匹配**：当用户的任务描述命中 Skill 的触发场景时，Agent 自动识别并调用对应 Skill。例如用户说"优化这个模型的推理性能"或"帮我把 Attention 替换成 FA 算子"，Agent 会匹配到对应的 Skill。
2. **指定调用**：用户直接指定使用某个 Skill，例如 `/model-infer-optimize` 或 `/model-infer-fusion`。

每个 Skill 的 `SKILL.md` frontmatter 中的 `description` 字段定义了触发场景关键词，具体见 [Skills 列表](#skills-列表) 中的"触发场景"列。

### 示例

**端到端优化**（触发 model-infer-optimize，按阶段编排执行）：

```
对目标模型进行端到端 NPU 推理优化
```

**单阶段使用**（直接触发特定 Skill）：

```
使用 model-infer-fusion 分析模型的融合算子替换方案
```

```
将模型的 Decode 阶段适配图模式
```

**问题诊断**（按需触发调试 Skill）：

```
FA 替换后精度对不上，帮我排查
```

```
推理报 aicore timeout 错误
```


## 贡献指南

欢迎贡献新的 Skill 或改进现有 Skill：

1. 在上游仓库提交 Issue 描述技能方案
2. Fork 仓库，创建特性分支
3. 按以下规范开发，提交 PR 并关联 Issue

### 命名规范

- `SKILL.md` 文件必须严格命名为 `SKILL.md`（区分大小写）
- Skill 文件夹使用 kebab-case，如 `model-infer-kvcache`
- Skill 文件夹内不包含 `README.md`，文档内容在 `SKILL.md` 或 `references/` 中

### 文档引用规范

- 大型外部参考文档（如算子接口、图模式指南等）通过在线链接引用，不包含离线副本
- references 存放仓库模型实现经验的结构化提炼，而非外部文档的离线副本

详细的架构设计见 `docs/agent/skill-design.md`。

## 免责声明

1. 本目录中的 Agent Skills 内容仅供技术参考和学习使用，不代表其适用于任何生产环境或关键业务系统。
2. 开发者在使用时应自行评估其安全性、兼容性和适用性。作者及贡献者不对因使用本内容导致的任何直接或间接损失承担责任。
3. 本内容可能涉及第三方依赖或接口调用，相关权限及合规性需由开发者自行核实。
4. 除非另有明确约定，本目录所有内容均基于开源协议发布，不提供任何形式的技术支持或担保。
