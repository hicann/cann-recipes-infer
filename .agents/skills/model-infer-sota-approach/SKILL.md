---
name: model-infer-sota-approach
description: 模型推理进阶性能调优编排技能。用于从推理场景、精度基线、profiling 采集与性能分析开始，派发多个方向的 subagent 生成不限于多流的优化候选，再按 Plan / round 编排 implementer 和 reviewer subagent 实施、验收、派生新 Plan，并最终判断哪些 Plan 保留、哪些淘汰。适用于融合算子、prefetch、图模式、多流、KVCache、量化、并行等优化项的统一调测编排。
---

# 模型推理优化编排技能（进阶性能调优流程）

本技能由 profiling 驱动候选发现，按 Plan / round 编排实施、独立复查、派生与最终验收。

## 定位：基础优化与进阶性能调优

- **基础流程 `model-infer-optimize`**：按并行 → KVCache/FA → 融合 → 量化 → 图模式推进适用阶段，经过分析、方案确认、实施、独立验证与总结；已有有效结论和授权可复用。
- **本技能（进阶性能调优流程）**：在已有 baseline 之上探索优化方向与组合，按 profiling 证据推进 Plan / round，不预设固定阶段。

**前置条件**：模型已完成框架适配，并有可运行、精度可复现的 baseline。条件未满足时，先通过 `model-infer-optimize` 阶段 0（模型分析与基线建立）或 `model-infer-migrator` 完成适配与基线建立。

**领域 skill 调用边界**：第 7 步按 Plan 内容选择单点技术 skill 实施，不嵌套调用其他编排流程。

## 流程总览

```text
+--------------------------------------------------------------+
| 1. 确认推理场景与性能目标                                     |
|    性能目标可选；必须明确目标推理场景                         |
+-----------------------------+--------------------------------+
                              |
                              v
+--------------------------------------------------------------+
| 2. 构造推理输入并跑通精度基线                                 |
|    派发 scenario subagent，生成输入、验证基线、记录精度口径   |
+-----------------------------+--------------------------------+
                              |
                              v
+--------------------------------------------------------------+
| 3. 采集 baseline profiling (round0)                          |
|    派发 profiling-instrumenter（非交互），采集可关闭或回退    |
+-----------------------------+--------------------------------+
                              |
                              v
+--------------------------------------------------------------+
| 4. 分析 baseline profiling                                   |
|    profile-analyzer 按代码与数据拆解，复用适用的分析规则     |
+-----------------------------+--------------------------------+
                              |
                              v
+--------------------------------------------------------------+
| 5. 候选发现                                                  |
|    按瓶颈选择来源，独立方向可并行派发 candidate              |
+-----------------------------+--------------------------------+
                              |
                              v
+--------------------------------------------------------------+
| 6. 初始化 Plan Dashboard（主 agent 唯一写者）                |
|    归并候选、裁定跨方向互斥/叠加、确定状态与验收口径         |
+-----------------------------+--------------------------------+
                              |
                              v
+--------------------------------------------------------------+
| 7. Plan 实施 / review / 派生循环                             |
|    每 Plan 分配新的全局 roundN；按需重采 profiling           |
|    implementer 用单点 skill 实施，reviewer 复核验收          |
|    两者均可建议派生，状态由主 agent 裁决                     |
+-----------------------------+--------------------------------+
                              |
            +-----------------+-----------------+
            |                                   |
            v                                   v
+--------------------------+        +--------------------------+
| 仍有值得验证的 Plan      |        | 本次探索已结束            |
| 回到实施 / review 循环   |        | 进入最终验收             |
+--------------------------+        +--------------------------+
```

## 执行规则

按下述流程推进并记录当前步骤与 round。每次派发 subagent 前，从 [`references/subagent-prompt-templates.md`](references/subagent-prompt-templates.md) 选择对应模板、替换占位符，可补充影响执行的上下文。

Plan Dashboard 使用 [`references/plan-dashboard-template.md`](references/plan-dashboard-template.md)；状态裁决、互斥/叠加判定、是否重采 profiling 和派生规则见 [`references/decision-rules.md`](references/decision-rules.md)。

### 性能分析栈

profiling 的采集与分析交给两个 skill，本流程只调用、不重复它们的内部契约：

- **`model-infer-profiling`**：采集 profiling 数据（`kernel_details.csv` 等）。
- **`model-infer-perf-breakdown`**：拆解 profiling 的时间分布、异常层与算子，支撑候选发现和收益归因。理论对比是可选步骤，依赖外部 skill，启用后才额外产出「逐算子实测 / 理论 gap」。

baseline 和重采 round 的采集、分析均由 subagent 执行。主 agent 传入场景、分析目标及已有规则；分析 subagent 根据模型实现和数据确定拆解口径，影响结果且无法消解的疑问返回主 agent。重采轮按 perf-breakdown 的复用规则更新分析，并与 baseline 同口径对照。

### 全局约束

- **subagent 配置继承主 agent**：所有 subagent 的 model、thinking 和上下文强度均与主 agent 一致。
- **主 agent 只做编排**：负责阶段推进、用户交互、派发、Dashboard 维护和验收裁决；采集、分析、实施与独立复查交 subagent。实施前向用户呈现并确认方案与取舍，已有授权可复用；新增取舍或超出授权的变更由主 agent 向用户确认，subagent 返回待决事项。
- **Dashboard 与 Plan 分工**：Dashboard 是 Plan 状态的唯一来源，由主 agent 维护总览与裁决。candidate 写方向级分析 `analysis/<source>.md`；主 agent 归并草案并初始化 `plans/plan-<id>.md`，implementer / reviewer 追加方案细节与 round 结论。主 agent 按需核查明细及证据。路径、字段与写者见 [`references/plan-dashboard-template.md`](references/plan-dashboard-template.md)。
- **共享状态使用 progress.md**：优先使用调用方指定文件，否则使用模型 `agentic/progress.md`，缺失时按[状态模板](../model-infer-optimize/templates/progress_template.md)创建。主 agent 维护常驻区的场景、精度口径、原始基线、已通过 Plan 与当前 round 目标；implementer / reviewer 先读后追加工作区，关键节点及时回写。每轮结束时先更新常驻区，再用[归档脚本](../model-infer-optimize/scripts/archive_progress.py)将工作区归入 `progress_history.md`。
- **Dashboard 初始化时机**：场景、精度基线、baseline profiling 和候选发现完成后初始化 Dashboard。
- **性能证据统一口径**：按目标指标的同场景、同测量方式实测判定收益，记录代码与配置、预热范围、测量结果及波动；分别记录相对本轮实施前采纳版本的增量和相对原始 baseline 的累计收益。profiling 报告用于瓶颈定位与收益归因，新实现或配置的收益需有对应测量。
- **roundN 全局递增、不复用**：每次尝试 Plan 或调整强度时分配新的全局 `roundN`；派生 Plan 也在开始尝试时分配，保留各轮独立记录。
- **状态只有三种**：`待实现` / `通过` / `淘汰`。
- **探索结束条件**：依据任务范围、目标达成情况与继续验证的预期价值决定是否结束；尚未验证的 Plan 保留 `待实现` 并注明后续建议。最终采用方案仍须完成独立验收。

### 1. 确认推理场景与性能目标

主 agent 先从任务、配置和已有记录确认场景，缺失或有冲突的事项再向用户确认，详见 [`references/scenario-confirm.md`](references/scenario-confirm.md)。需要明确：

- 模型 / case 名称、代码目录和推理入口；
- 要优化的推理阶段或 workload（prefill、decode、图像生成、长序列、batch、并发等）；
- 精度或功能的判定口径；
- 性能目标（可选）；未指定时不阻塞后续流程；
- 输出归档目录，例如 `optimization-analysis/<case>/`。

用户未指定具体场景时，主 agent 先从推理脚本、配置和已有报告中调查候选场景；无法确定的事项向用户确认。

### 2. 构造推理输入并跑通精度基线

使用 scenario 模板派发 `scenario` subagent，详细操作见 [`references/scenario-setup.md`](references/scenario-setup.md)。产出可复现的推理输入（或输入构造脚本）、基线推理命令、精度/功能结果，以及场景记录文件路径（`scenario.md`）。已有证据与当前场景一致时复用。主 agent 核对摘要及必要证据，确认场景能稳定复现后再进入 profiling。

### 3. 采集 baseline profiling（round0）

派发 `profiling-instrumenter` subagent，使用 `model-infer-profiling` 为已跑通的场景插入或启用 profiling 并采集。baseline 和重采轮均非交互执行，详细采集输出由 subagent 保存。产出采集入口/开关、采集命令、产物路径和回退方式；保留采集开关，关闭后恢复普通推理路径。**本轮产物为 baseline（round0）profile**，作为后续 round 同口径对照的基准。

### 4. 分析 baseline profiling

派发 `profile-analyzer` subagent，根据模型实现、profiling 和已有分析规则运行 `model-infer-perf-breakdown`。产出时间分布（各模块与层的 wall、bubble 中位数与异常层）和洞察（模块 bubble、算子抖动、连续 vector 区间、冗余搬运）；启用理论对比时补充逐算子实测 / 理论 gap。具体文件与分析规则的复用范围由该 skill 定义。

这份 baseline 分析标定了性能结构和优化空间，既是候选发现的主要依据，也是最终验收做同场景、同口径对照的基准。

### 5. 候选发现

候选来源在下表注册。主 agent 根据瓶颈与任务范围选择来源，独立方向可并行派发。prompt 使用 [`references/subagent-prompt-templates.md`](references/subagent-prompt-templates.md) 的通用 candidate 模板，按所选来源的「说明 / 输入 / 产物文件」填占位符。
若有候选来源当前没有挂载关联的工具/skill，则跳过。

| 候选来源（source） | 说明（方法） | 输入 | 产物文件 | 下游衔接 |
| --- | --- | --- | --- | --- |
| multi-stream | 用 `model-infer-multi-stream` 按分析范围拆解模块 / 算子 DAG、判断并行性，提出有实质差异的编排候选 | 代码目录、场景记录、baseline 分析 | `analysis/multi-stream.md`（DAG + 候选，结构见[模块拆解模板](../model-infer-multi-stream/references/module-decomposition.md)） | 各 Plan 引用本文件的 DAG，implement / review 据此实施 |
| knowledge | 通过 `knowledge-query` 查询 cannbot-knowledge 中同框架、平台与负载的优化样例和适用约束 | 场景记录（模型 / 硬件 / 阶段）、baseline 瓶颈摘要 | `analysis/knowledge.md`（候选 + 知识库条目 + 适用条件） | 各 Plan 保留条目引用与适用条件，implement / review 按需查询 |
| perf-insight | 根据 insight 核查候选：连续 vector 区间评估融合，冗余搬运检查重复拷贝与格式转换，模块 bubble 与抖动定位重点算子；权重读取受限且有预取窗口时评估 prefetch；理论对比启用时补充理论偏离热点 | baseline 报告与 insight 目录、场景记录 | `analysis/perf-insight.md`（候选 + insight 证据） | 各 Plan 引用本文件的 insight 证据 |

各来源分别写入指定产物文件，`plan-dashboard.md` 由主 agent 维护。候选草案以摘要返回主 agent，在第 6 步统一归并并裁定互斥 / 叠加关系；每个草案包含方案描述、预期收益、风险与验证口径、互斥 / 可叠加、推荐优先级。用户提供的优化列表作为初始候选分配给对应来源的 candidate，增删或合并时说明原因。

### 6. 初始化 Plan Dashboard

候选发现完成后，主 agent 按 `plan-dashboard-template.md` 初始化 Dashboard，写入约定路径并完成以下事项：

- 归并重复或等价的候选；
- 给每个候选分配稳定的 `plan_id`，初始状态统一为 `待实现`，并为每个 Plan 建一份 `plans/plan-<id>.md`（保存 candidate 草案的方案描述、方案细节、知识库条目与实现参考及适用条件）；
- 在 Dashboard 中为每个 Plan 填写关键信息、状态和 Plan 文件链接；
- **裁定候选之间的互斥与叠加关系**（例如两个 Plan 修改同一段 attention），标明互斥组和可叠加性；
- 填写每个 Plan 的优化类型和验收口径；实施所用的单点 skill 在第 7 步由主 agent 按 Plan 内容确定；
- 记录每个候选的来源（哪个 candidate subagent）和支撑它的 baseline profiling 证据。

### 7. Plan 实施 / review / 派生循环

选择一个 `待实现` Plan，分配新的全局 `roundN`，执行以下循环：

```text
选定 Plan / 分配 roundN
  -> 主 agent 按 Plan 内容选择实施所用的单点 skill
  -> 主 agent 判断本轮是否需要重采 profiling（见下）
  -> implementer subagent 用选定的 skill 实施当前 Plan
  -> （需要时）依次派发 profiling-instrumenter、profile-analyzer，采集本轮 profiling 数据并生成分析报告
  -> reviewer subagent 复核并验收 implementer 的工作
  -> 主 agent 按 decision-rules 更新状态、处理派生建议
  -> 根据任务范围与继续验证的价值，选择下一 round 或进入最终验收
```

**实施 skill 由主 agent 按 Plan 内容确定**，作为 `领域 skill` 传给 implementer subagent；多流 Plan 使用 `model-infer-multi-stream`。reviewer 使用同一 skill 复核。

**是否重采 profiling 由主 agent 判断，不强制每轮都重采。** 上一轮结束、选择下一 Plan 时，按以下依据决定：

- 本轮改动是否改变了**算子下发时序、计算图结构、dtype 或 layout**——多流、融合、图模式、量化通常会改，需要重采；
- reviewer 是否反映现有 profile 已不足以支撑收益判断；
- 是否转向新的优化方向或热点模块；
- 是否要与 baseline 做同口径对照——**最终方案验收前必须重采一次**。

需要重采时，依次派发 `profiling-instrumenter` 和 `profile-analyzer`，采集本轮 profiling 数据并生成分析报告，按 perf-breakdown 的复用规则核对并更新分析配置。不重采时，已有 profile 可作为分析参考，收益仍按当前执行路径的实测判定。详细判据见 [`references/decision-rules.md`](references/decision-rules.md) 的"round 之间是否重采 profiling"。

implementer 可以建议派生新 Plan，保留当前 Plan 的历史；reviewer 独立验证，可准备测试文件与诊断，模型代码修复及回退交 implementer。新 Plan 另起编号、初始状态为 `待实现`，开始尝试时分配新的 `roundN`。

每轮结束时，implementer / reviewer 将实施、问题与验证过程写入 `progress.md` 工作区，方案细节、round 结论与派生建议写入 `plans/plan-<id>.md`；主 agent 核查证据并裁决，更新 Dashboard 的证据摘要、状态、采纳实现和 Round 记录。

### 8. 最终验收

主 agent 根据任务范围、目标达成情况与继续验证的价值决定是否结束探索，记录依据和后续方向。已尝试的 Plan 保留实施与复查记录；尚未验证的 Plan 保留 `待实现`，注明未纳入本次方案。

最终验收前对最终代码路径重采 profiling，与 round0（baseline）做同场景、同口径对照；由 reviewer 独立验收最终组合，主 agent 汇总证据并裁决：

- 所有 `通过` Plan 都已应用到最终代码路径；
- 可叠加 Plan 的组合效果已经实测验证；
- 同一互斥组内状态自洽（最终只保留一个或一组明确兼容的 `通过` Plan）；
- 未采纳的试验改动已回退或被开关关闭，不影响最终运行；
- 最终精度/功能仍满足第 1 步确认的口径；
- 最终性能按 baseline 同场景、同指标口径实测，并结合 profile-analyzer 报告说明收益来源与副作用。

## 参考文档索引

- **场景确认（已有信息、前置检查与待决事项）**：[`references/scenario-confirm.md`](references/scenario-confirm.md)
- **Subagent prompt 模板**：[`references/subagent-prompt-templates.md`](references/subagent-prompt-templates.md)
- **scenario 操作细则（构造输入、验证基线与确定判据）**：[`references/scenario-setup.md`](references/scenario-setup.md)
- **Plan Dashboard 模板**：[`references/plan-dashboard-template.md`](references/plan-dashboard-template.md)
- **Plan 文件模板（plans/plan-<id>.md）**：[`references/plan-file-template.md`](references/plan-file-template.md)
- **状态裁决与 round 推进规则**：[`references/decision-rules.md`](references/decision-rules.md)
