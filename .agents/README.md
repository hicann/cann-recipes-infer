# NPU 模型优化 Agent Skills

## 项目概述

### 项目定位

面向昇腾 NPU 推理优化场景的 Agent Skills 集合。基于 CANN 平台能力和 [cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer) 仓库的模型优化经验，将路径知识、阶段依赖和验证流程模块化，支持 Agent 按流程完成端到端推理优化。

在 AI 智能体（Agent）上下文中，技能（Skills）是为扩展 Agent 能力而设计的模块化功能单元。每个 Skill 封装了指令、元数据及可选资源，支持该规范的客户端可根据任务加载对应 Skill。

### 核心能力

- **端到端优化编排**：从框架适配、基线建立到并行化改造、KVCache 优化、融合算子替换、量化与图模式适配，按适用阶段推进
- **每阶段验证驱动**：精度验证和性能验证通过后再进入下一阶段，避免问题叠加
- **技术指导与知识查询**：保留并行、KVCache、融合、量化与图模式的关键判断和模板，按需查询知识库中的接口约束、部署样例与排障经验
- **问题诊断**：NPU 推理精度问题和运行时错误的系统化排查流程

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
| [model-infer-quantization](skills/model-infer-quantization/) | 量化适配改造 | compressed-tensors 量化方案接入、权重/激活/KVCache 量化适配 |
| [model-infer-graph-mode](skills/model-infer-graph-mode/) | 图模式适配 | torch.compile、npugraph_ex / GE 图模式、图中断修复 |

### 辅助与调试 Skills

按需触发，不在主流程阶段中：

| Skill | 功能 | 触发场景 |
|-------|------|---------|
| [model-infer-precision-debug](skills/model-infer-precision-debug/) | NPU 推理精度诊断 | 精度验证未通过、NaN/Inf、输出质量退化、跨 batch 不一致等 |
| [model-infer-runtime-debug](skills/model-infer-runtime-debug/) | NPU 运行时错误诊断 | aicore timeout、HCCL 错误、OOM、推理卡住 |
| [model-infer-profiling](skills/model-infer-profiling/) | 性能数据采集 | 采集 NPU profiling、torch_npu.profiler 使用、解析失败排查 |
| [model-infer-perf-breakdown](skills/model-infer-perf-breakdown/) | 性能数据拆解 | 拆解 kernel_details.csv、按模块定位耗时分布与抖动 layer |

### 进阶性能调优 Skills

在已有 baseline 之上探索优化方向，也可按任务单独调用技术 skill：

| Skill | 功能 | 触发场景 |
|-------|------|---------|
| [model-infer-sota-approach](skills/model-infer-sota-approach/) | 进阶性能调优编排入口 | 在已有 baseline 之上由 profiling 驱动、多方向发现候选并按 Plan 收敛 |
| [model-infer-multi-stream](skills/model-infer-multi-stream/) | 多流并行优化 | NPU 多流表达、算子分流、计算与计算并行 |
| [model-infer-prefetch](skills/model-infer-prefetch/) | 权重预取适配 | 权重预取、prefetch 适配 |


### 调度关系

```
model-infer-optimize（编排入口）
  ├── 阶段 0：model-infer-migrator（框架适配与基线建立）
  ├── 阶段 1：model-infer-parallel-analysis + model-infer-parallel-impl（并行策略，多卡部署时）
  ├── 阶段 2：model-infer-kvcache（KVCache 优化 + FA 替换）
  ├── 阶段 3：model-infer-fusion（融合算子替换）
  ├── 阶段 4：model-infer-quantization（量化适配改造，可选）
  ├── 阶段 5：model-infer-graph-mode（图模式适配）
  └── 按需触发：model-infer-precision-debug（精度诊断）/ model-infer-runtime-debug（运行时错误）
```

### Agent 协作架构

主流程采用"编排 Agent + 三类专业 SubAgent"的协作模式，通过分析/实施/验证的职责隔离，降低单个 Agent 同时承担多角色导致的越界与漏检问题：

| Agent | 角色 | 职责 |
|-------|------|------|
| 主 Agent | 编排者 | 阶段推进、用户确认、进度管理与验收裁决，模型改造交 implementer |
| model-infer-analyzer | 分析专家 | 架构分析、方案设计，只读模型代码与配置，写分析产物 |
| model-infer-implementer | 实施工程师 | 按已确认方案改造代码、调试修复并自验证 |
| model-infer-reviewer | 审查专家 | 独立验证精度、性能与仓库规范，可准备测试文件，模型修复交 implementer |

每阶段协作流程：分析 → 用户确认方案 → 实施 → 独立验证 → 阶段总结，再进入下一适用阶段；已有有效结果与明确授权可复用。SubAgent 优先通过调用方指定的 `progress.md` 传递上下文，默认使用模型 `agentic/progress.md`；独立调用按任务要求交付。

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
    ├── model-infer-quantization/                # 量化适配改造
    ├── model-infer-graph-mode/                  # 图模式适配
    ├── model-infer-precision-debug/             # NPU 推理精度诊断
    ├── model-infer-runtime-debug/               # NPU 运行时调试
    ├── model-infer-profiling/                   # 性能数据采集
    ├── model-infer-perf-breakdown/              # 性能数据拆解
    ├── model-infer-sota-approach/               # 探索式优化编排
    ├── model-infer-multi-stream/                # 多流并行
    └── model-infer-prefetch/                    # 权重预取
```

## 知识库与参考资料

通过 `knowledge-query` 查询 cannbot-knowledge 中与目标框架、平台和版本匹配的接口约束、部署样例与排障经验；需要官方资料时按部署版本追踪来源。通用查询与经验回流入口见 [AGENTS.md](../AGENTS.md#知识库)。

## 使用方式

### 安装

根据使用的客户端选择对应命令，默认安装 cannbot-knowledge 的查询技能。首次安装需联网，使用 Git、curl 和 Python 3.10+；知识库安装器会补齐 Python 依赖、恢复参考资源并准备索引：

```bash
bash scripts/init-agent.sh --codex      # Codex 用户
bash scripts/init-agent.sh --claude     # Claude Code 用户
bash scripts/init-agent.sh --opencode   # OpenCode 用户
bash scripts/init-agent.sh              # 多客户端都要用时（可选）
```

已有知识仓和安装模式从 `.cannbot/knowledge.env` 复用；首次安装默认使用 `~/.cannbot/cannbot-knowledge` 和 `consumer` 模式。可用 `--knowledge-root /path/to/cannbot-knowledge` 指定位置；已有 checkout 保持当前版本。需要经验回流时追加 `--knowledge-profile contributor`。离线安装需预备知识仓和 Python 依赖，并显式追加 `--skip-resources`；此时仅使用已有知识和资源。更多选项背景见[知识库安装说明](https://gitcode.com/cann/cannbot-knowledge/blob/master/docs/installation_and_usage.md)。

脚本作用：

- 在 `.codex/agents/` 下由 `.agents/agents/*.md` 生成 `*.toml`，注册 `.codex/hooks.json`；Codex 直接读取 `.agents/skills`
- 在 `.claude/` 下创建 skills/agents/hooks 的 symlink（指向 `.agents/`），将共享 Hook 合并到 `settings.json`，保留已有用户配置
- 在 `.opencode/` 下创建 skills/agents 的 symlink
- 初始化 Claude Code 时，生成通过 `@AGENTS.md` 导入公共指引的 `CLAUDE.md`
- 知识库技能共用 `.agents/skills/` 入口，配置写入 `.cannbot/knowledge.env`，查询说明统一维护在 `AGENTS.md`

生成的客户端视图、知识库技能链接及 `.cannbot/` 配置已加入 `.gitignore`。

Codex 复杂优化任务建议在任务入口明确写明“请使用 Codex subagent workflow”，再指定工作流与角色分工。

Claude Code 与 Codex 共用 `.agents/settings.json` 的事件配置和 `.agents/hooks/` 脚本，初始化时转换为绝对路径。Codex 需信任项目，再通过 `/hooks` 审核启用，详见[官方 Hook 文档](https://learn.chatgpt.com/docs/hooks)。OpenCode 当前仅安装 skill 与 agent 视图，尚未注册 Hook。

### 触发方式

Skill 有两种触发方式：

1. **场景匹配**：当用户的任务描述命中 Skill 的触发场景时，Agent 自动识别并调用对应 Skill。例如用户说"优化这个模型的推理性能"或"帮我把 Attention 替换成 FA 算子"，Agent 会匹配到对应的 Skill。
2. **指定调用**：用户直接指定使用某个 Skill，例如“使用 `model-infer-optimize` 优化模型”。

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

### 内容与引用规范

- Skill 保留关键判断、真实约束和完整模板；references 承载按需读取的必要流程、接口契约与代码样例。
- 具体部署经验、版本相关接口资料和排障案例通过知识库查询，入口就近说明查什么；重要约束可在 skill 中保留兜底。
- 样例注明适用框架与条件，保留仓库推荐写法；引用文件、脚本和接口随修改同步核对。

完整的开发与审查准则见[基础工作流设计](../docs/agent/model-infer-optimize-design.md)；进阶编排见[SOTA 工作流设计](../docs/agent/sota-approach-skill-design.md)。

## 免责声明

1. 本目录中的 Agent Skills 内容仅供技术参考和学习使用，不代表其适用于任何生产环境或关键业务系统。
2. 开发者在使用时应自行评估其安全性、兼容性和适用性。作者及贡献者不对因使用本内容导致的任何直接或间接损失承担责任。
3. 本内容可能涉及第三方依赖或接口调用，相关权限及合规性需由开发者自行核实。
4. 除非另有明确约定，本目录所有内容均基于开源协议发布，不提供任何形式的技术支持或担保。
