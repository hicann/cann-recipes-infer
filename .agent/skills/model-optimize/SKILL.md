---
name: model-optimize
description: 模型端到端优化技能。依次执行 KVCache 优化、融合算子优化和图模式适配，每阶段完成后进行精度和性能验证，确认达标后再进入下一阶段。触发场景包括：完整 NPU 推理优化、端到端优化、全流程优化、依次完成 KVCache/融合算子/图模式适配。通过 subagent 依次调用 kvcache-optimization、torch-npu-fusion-optimizer 和 graph-mode-adaptation 技能。
user-invocable: true
---

# 模型端到端优化技能

## 概述

本技能对目标模型按阶段执行完整的 NPU 推理优化。工作流程为：

```
模型分析 → KVCache静态化+FA替换 → 验证 → 用户确认 → 融合算子优化（Attention之外） → 验证 → 用户确认 → 图模式适配 → 验证 → 用户确认 → 输出报告
```

每个优化阶段完成后，必须验证精度和性能，并经用户确认后才可进入下一阶段。

---

## 重要原则

- **逐阶段推进**：先 KVCache 优化，再融合算子，最后图模式，不可跳过或并行
- **验证驱动**：每阶段必须完成精度验证和性能验证后才能进入下一阶段
- **用户确认**：每阶段的验证结果必须经用户确认后才能继续
- **完整记录**：全程使用优化报告模板记录问题和结果
- **保持模型完整性**：不为通过验证而简化模型实现
- **subagent 委托**：各阶段的子技能通过 Agent tool 启动独立 subagent 执行，主 agent 不直接加载子技能内容自行实施
- **主 agent 职责边界**：主 agent 负责编排（分析、派 subagent、跑测试、用户确认），不直接修改模型代码
- **失败不接手**：subagent 未完成时，派新 subagent 继续（读 progress.md 从断点接力），主 agent 不自己接手实施或调试
- **进度持久化**：所有 subagent 必须在返回前将设计决策、实施记录、调试记录更新到 progress.md
- **性能计时边界**：性能测试只统计模型 forward 耗时（含 KVCache 更新、FA 调用等），不包括输入准备时间（attention_mask 构造、position_ids 生成等）

---

## 共享状态文件规范

每个模型目录下维护一个 `progress.md` 文件，作为主 agent 和各 subagent 之间的共享状态。

文件路径：`{model_dir}/progress.md`

格式规范：

```markdown
## 阶段 0：模型分析
- 模型路径: ...
- 架构: GQA/MHA/MLA (具体参数)
- KVCache 模式: PA / 非PA / MLA压缩
- FA 算子: npu_fused_infer_attention_score / v2
- 量化: BF16 / W8A8 / ...
- 基线: Prefill Xms, Decode Xms/token, XMB

## 阶段 N：标题
### 设计决策
- 关键参数和选型理由

### 实施记录
- [完成] 描述 — 文件:行号
- [进行中] 描述
- [失败] 描述 — 失败原因

### 当前代码状态
- 简要记录当前代码的关键状态（如 tensor layout、cache 格式、已替换/未替换的模块），供接力 subagent 直接了解现状，不必重新读代码推断

### 调试记录
- [已查] 检查项 ✓
- [发现] 问题描述
- [放弃] 方案描述 — 放弃原因
- [修复] 修复措施 — 文件:行号
- [待验证] 待确认事项

### 精度验证
- 状态: 通过/未通过
- 具体数据

### 性能验证
- 优化前后对比数据
```

写入规则：
- 阶段 0 由主 agent 写入
- 实施记录、设计决策由开发 subagent 写入
- 调试记录由调试 subagent 写入
- 精度/性能验证由主 agent 写入
- 各阶段内容只追加不清空，保留完整历史
- 写入前先读取现有内容，追加到对应 section 末尾，避免覆盖其他角色的记录

---

## 工作流程

### 阶段 0：模型分析

在进行任何优化之前，先对模型进行全面分析，建立优化基线。

#### 0.1 模型结构分析

- 识别模型架构类型（LLM / MoE / Diffusion / 多模态等）
- 拆解网络结构为独立模块（Embedding → Transformer Blocks → Output Head）
- 标注 Prefill / Decode 分支差异
- 记录关键模块：Attention 类型（GQA/MHA/MLA）、FFN/MoE 结构、特殊模块

#### 0.2 运行环境与配置分析

- 硬件平台：Atlas A2 / A3 / A5
- 量化模式：BF16 / W8A8 / W8A8C8 / W4A16
- 执行模式：当前 exe_mode（eager / ge_graph）
- **当前范围为单卡模型适配，多卡并行策略不在本技能范围内**

> 后续可扩展并行策略优化阶段（TP / EP / DP 配置优化），作为单卡优化完成后的进阶流程。

#### 0.3 建立性能基线

在优化前，先确认模型能正常运行（加载、推理、输出无报错），再记录基线指标：

- **精度基线**：使用标准输入运行模型，记录输出结果（用于后续精度对比）
- **性能基线**：记录以下指标（可获取的部分）
  - Prefill 阶段耗时
  - Decode 阶段单步耗时
  - 端到端吞吐量（tokens/s）
  - 显存占用

#### 0.4 输出分析报告

将分析结果整理为结构化文档，并在模型目录下创建优化报告文件：

```
models/{model_name}/optimization_report.md
```

使用 `templates/optimization_report_template.md` 模板初始化报告文件，填入模型分析和基线数据。

同时创建 `{model_dir}/progress.md`，写入阶段 0 section（模型分析结果、基线数据）。

---

### 阶段 1：KVCache 静态化 + FA 算子替换

通过 subagent 调用 `kvcache-optimization` 技能，完成 KVCache 静态化改造并替换 Attention Core 为 FA 融合算子。

> KVCache 模式选型（连续缓存 / PA / MLA 压缩）会影响 FA 算子的调用方式和参数配置（layout、block_table 等），具体 FA 版本选择由 kvcache-optimization skill 指导。
> 静态化完成后 cache metadata（slot_mapping、block_table、actual_seq_lengths_kv）已就绪，FA 替换只需对齐接口直接换上。
> 后续阶段 2 的融合算子优化专注于 Attention Core 之外的模块（MoE、FFN、Norm 等）。

#### 1.1 启动开发 subagent

使用 Agent tool 启动 subagent：

```
启动 subagent（类型：general-purpose），提示词包含：
1. 说明目标模型路径和代码位置
2. 传递阶段 0 的模型分析摘要（注意力架构、序列长度、batch 大小等），指向 progress.md 获取完整上下文
3. **你必须首先调用 /kvcache-optimization 技能，严格按技能流程执行。必须先读取技能指定的参考实现文件，再动手写代码**
4. 要求 subagent 根据模型类型选择合适的 KVCache 模式（连续缓存 / 分页注意力 / MLA 压缩），并配合选择对应的 FA 算子
5. 要求 subagent 完成以下工作：
   - 实现 KVCache 的初始化（缓存 tensor、block_table 等）
   - 实现缓存更新逻辑（scatter_update_ 或 npu_kv_rmsnorm_rope_cache）
   - 构造 FA 算子所需入参（slot_mapping、actual_seq_lengths、block_table 等）
   - 适配 Prefill 和 Decode 两个阶段的差异
6. **遇到报错时**：可自行调试修复，但需要更换方案方向（如换 layout、换 FA 版本）时，先重新查阅 skill 和参考实现确认方向正确，再动手改。将关键决策和放弃的方案简要写入 progress.md
   - 静态化完成后，将 Attention Core 替换为 KVCache 模式对应的 FA 算子
7. 要求 subagent 将设计决策和实施记录更新到 progress.md 阶段 1 section
```

#### 1.2 精度验证

开发 subagent 返回后，主 agent 执行精度验证：

1. 使用与基线相同的标准输入运行模型
2. 对比优化前后的输出结果
3. 判定标准：
   - 文本生成模型：输出 token 序列一致或语义等价
   - 数值对比：关键 tensor 的相对误差 < 1e-3（BF16）或 < 1e-2（量化模式）
4. 将结果写入 progress.md 阶段 1 的精度验证 section
5. 不通过时，将失败详情（症状、误差数据、出错阶段）写入 progress.md，然后触发 1.3

#### 1.3 精度调试（精度验证未通过时触发）

> 运行时报错（crash）由开发 subagent 重新读取 /kvcache-optimization 技能后自行修复。精度问题由专门的调试 subagent 处理。

**触发条件**（满足任一即触发）：
- 精度验证未通过（token 不一致或数值误差超阈值）
- 输出包含 NaN/Inf
- Prefill 和 Decode 阶段精度表现不一致

**启动精度调试 subagent**：

```
启动 subagent（类型：general-purpose），提示词包含：
1. 说明目标模型路径和代码位置
2. 指向 progress.md 获取模型分析结果、KVCache 实施记录和精度验证失败详情
3. **你必须首先调用 /kvcache-fa-precision-debug 技能，严格按技能流程执行，不要自行发明调试方法**
4. 提供精度验证的失败详情（症状描述、误差数据、出错阶段）和基线输出文件路径
5. 若为接力调试，说明"前一轮调试未完成，读 progress.md 调试记录，从断点继续，已排查项不要重复"
6. 要求 subagent 按技能流程完成排查和修复
7. 要求 subagent 将调试记录更新到 progress.md 阶段 1 的调试记录 section
```

**调试结果处理**：

```
调试 subagent 返回后 → 主 agent 重跑精度测试 → 结果写入 progress.md
精度达标 → 继续 1.4 性能验证
精度仍未达标 → 派新调试 subagent（读 progress.md 从断点接力）→ 最多接力 5 次
接力 5 次仍未解决 → 回退 KVCache 改动 → 报告给用户决策
调试发现需要修改 KVCache 方案 → 重新执行 1.1（更换 KVCache 模式）
```

#### 1.4 性能验证

精度通过后，主 agent 执行性能验证：

1. 使用相同配置和输入运行性能测试
2. 记录优化后的性能指标（与基线相同的指标维度）
3. 计算性能变化百分比
4. 将结果写入 progress.md 阶段 1 的性能验证 section

#### 1.5 阶段检查与用户确认

**阶段 1 检查项**：
- [ ] KVCache 优化 subagent 已完成 KVCache 模式选型、静态化实现及 FA 算子替换
- [ ] 缓存初始化、更新、FA 调用入参构造均已实现
- [ ] eager attention 已替换为对应 FA 算子
- [ ] Prefill 和 Decode 阶段的缓存逻辑差异已正确处理
- [ ] 精度验证通过（若曾触发精度调试，调试报告已记录）
- [ ] 性能验证完成并记录
- [ ] progress.md 阶段 1 各 section 已填写完整
- [ ] 将优化结果（KVCache 模式、精度、性能变化）汇总呈现给用户，经用户确认后进入下一阶段。精度/性能异常时参照「异常处理」章节

---

### 阶段 2：融合算子优化

通过 subagent 调用 `torch-npu-fusion-optimizer` 技能，对模型执行融合算子替换。

> 若阶段 1 已完成 Attention Core 的 FA 算子替换，则本阶段跳过 Attention Core，聚焦其余模块（RoPE 融合、KV write 融合、MoE、FFN、Norm 等）。
> 若阶段 1 未完成 FA 替换，本阶段需包含 Attention Core 的融合算子适配。

#### 2.1 启动开发 subagent

使用 Agent tool 启动 subagent：

```
启动 subagent（类型：general-purpose），提示词包含：
1. 说明目标模型路径和代码位置
2. 指向 progress.md 获取模型分析结果和阶段 1 完成情况
3. 明确指示调用 /torch-npu-fusion-optimizer 技能
4. progress.md 中阶段 1 记录了 FA 是否已替换，若已替换则本阶段跳过 Attention Core
5. 要求 subagent 完成融合算子的分析和替换实施
6. 要求 subagent 将实施记录更新到 progress.md 阶段 2 section
```

#### 2.2 精度验证

融合算子 subagent 返回后，主 agent 执行精度验证：

1. 使用与基线相同的标准输入运行模型
2. 对比融合前后的输出结果
3. 判定标准：
   - 文本生成模型：输出 token 序列一致或语义等价
   - 数值对比：关键 tensor 的相对误差 < 1e-3（BF16）或 < 1e-2（量化模式）
4. 将结果写入 progress.md 阶段 2 的精度验证 section

#### 2.3 性能验证

精度通过后，主 agent 执行性能验证：

1. 使用相同配置和输入运行性能测试
2. 记录优化后的性能指标（与基线相同的指标维度）
3. 计算性能变化百分比
4. 将结果写入 progress.md 阶段 2 的性能验证 section

#### 2.4 阶段检查与用户确认

**阶段 2 检查项**：
- [ ] 融合算子优化 subagent 已完成所有模块的分析与替换决策
- [ ] 每个替换模块均记录了精度对比结果
- [ ] 每个替换模块均记录了性能对比结果
- [ ] 对失败/跳过的模块：审查跳过理由是否为硬约束，需前置改造但未尝试的应要求 subagent 先尝试
- [ ] progress.md 阶段 2 各 section 已填写完整
- [ ] 将优化结果（替换算子清单、精度、性能变化）汇总呈现给用户，经用户确认后进入下一阶段。精度/性能异常时参照「异常处理」章节

---

### 阶段 3：图模式适配优化

通过 subagent 调用 `graph-mode-adaptation` 技能，对模型执行图模式适配。

#### 3.1 启动开发 subagent

使用 Agent tool 启动 subagent：

```
启动 subagent（类型：general-purpose），提示词包含：
1. 说明目标模型路径和代码位置
2. 指向 progress.md 获取全部历史阶段的上下文
3. 明确指示调用 /graph-mode-adaptation 技能
4. 说明当前的执行模式和期望的图模式类型（若用户未指定则要求 subagent 询问用户）
5. 要求 subagent 完成图模式的方案设计、用户确认和实施开发
6. 要求 subagent 将实施记录更新到 progress.md 阶段 3 section
```

**注意**：图模式仅适用于 Decode 阶段，Prefill 阶段禁止使用图模式。

#### 3.2 精度验证

图模式 subagent 返回后，主 agent 执行精度验证：

1. 使用与基线相同的标准输入运行模型
2. 对比图模式适配前后的输出结果
3. 判定标准与阶段 1 相同
4. 将结果写入 progress.md 阶段 3 的精度验证 section

#### 3.3 性能验证

精度通过后，主 agent 执行性能验证：

1. 使用相同配置和输入运行性能测试
2. 记录优化后的性能指标
3. 计算相对于 **阶段 2 完成后** 的性能变化
4. 计算相对于 **原始基线** 的累计性能变化
5. 将结果写入 progress.md 阶段 3 的性能验证 section

#### 3.4 阶段检查与用户确认

**阶段 3 检查项**：
- [ ] 图模式适配 subagent 已完成方案设计和实施
- [ ] Decode 阶段已启用图模式
- [ ] Prefill 阶段未使用图模式
- [ ] 精度验证通过
- [ ] 性能验证完成并记录（本阶段增量 + 累计）
- [ ] progress.md 阶段 3 各 section 已填写完整
- [ ] 将优化结果（改造内容、精度、性能变化）汇总呈现给用户，经用户确认后进入总结阶段。精度/性能异常时参照「异常处理」章节

---

### 阶段 4：优化总结

使用 `templates/optimization_report_template.md` 模板，将完整的优化报告写入模型目录：

```
models/{model_name}/optimization_report.md
```

报告应包含：
- 模型分析概要
- 各阶段的优化措施、精度验证结果、性能验证结果
- 功能问题记录表（问题描述、影响范围、处理方式、状态）
- 性能问题记录表（瓶颈描述、优化措施、优化前后数据、增益）
- 累计优化效果（相对原始基线的总提升）
- 遗留问题与后续建议

> 报告内容可从 progress.md 各阶段的记录中提取整理。

---

## 异常处理

### 精度不达标

```
精度验证未通过
    │
    ├─ 定位问题模块 → 回退该模块的优化 → 重新验证
    │
    ├─ 若回退后精度恢复 → 记录该模块为"不适配"，继续其他模块
    │
    └─ 若整体精度无法恢复 → 回退当前阶段全部改动 → 报告问题 → 请求用户决策
```

### 性能无增益

```
性能验证无增益
    │
    ├─ 分析性能瓶颈原因 → 记录到报告
    │
    ├─ 若精度达标 → 告知用户，由用户决定是否保留改动
    │
    └─ 若精度也未达标 → 按精度不达标流程处理
```

### subagent 执行失败

```
subagent 返回错误或未完成
    │
    ├─ 主 agent 读 progress.md 了解已完成部分
    │
    ├─ 派新 subagent 从断点继续（读 progress.md 接力）→ 最多接力 5 次
    │
    └─ 若无法继续 → 报告当前进度和阻塞点 → 请求用户决策
```

