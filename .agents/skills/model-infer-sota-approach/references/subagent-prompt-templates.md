# Subagent Prompt 模板

主 agent 派发 subagent 前，选择对应模板、替换尖括号占位符并补全路径。可按模型和单点 skill 补充上下文，保留产物路径、返回要求和角色边界。

## 通用约定

- **配置继承主 agent**：每个 subagent 的 model、thinking 和上下文强度均与主 agent 一致。
- **返回摘要和路径**：subagent 将详细日志、完整 profile 和原始数据保存到文件，仅向主 agent 返回结论摘要、证据路径和待决事项。
- **Dashboard 只读**：Plan Dashboard 由主 agent 维护；subagent 可读取全局状态，通过返回摘要交由主 agent 更新。
- **性能口径统一**：以目标指标的同口径实测判定收益，profiling 支撑瓶颈定位与归因；测量须对应本轮执行路径、配置与负载。
- **共享上下文**：subagent 先读 `<progress_path>` 获取场景、精度口径、baseline 与当前 round 目标；派发模板给出任务范围和产物路径，可补充影响执行的上下文。
- **过程与结论分工**：implementer / reviewer 将实施、问题与验证过程追加到 `progress.md` 工作区，方案细节和 round 摘要写入对应 Plan 文件；其他角色写各自指定产物，返回证据摘要与路径。

## scenario：构造推理输入并跑通精度基线

```text
你是 model-infer-sota-approach 的 scenario subagent。

执行前阅读 `references/scenario-setup.md`。

目标：
- 为当前优化任务构造可复现的推理输入。
- 跑通基线推理，建立精度 / 功能的判定口径。
- 记录目标性能指标的测量方式与基线结果。
- 不做性能优化，也不改动优化相关代码。

输入：
- 模型 / case：<case_name>
- 共享状态文件：<progress_path>
- 代码目录：<code_dir>
- 推理入口：<infer_entry>
- 用户指定场景：<scenario_request>
- 可选性能目标：<performance_goal_or_NA>
- 输出目录：<analysis_root>

执行要求：
- 优先复用仓库已有的推理脚本和配置。
- 若需要生成输入，保存输入构造方式或输入样本路径，确保可复现。
- 记录基线运行命令、关键参数，以及精度 / 功能结果。
- 沿用已确认的解码与验收口径，保存完整基线输出和性能测量证据；缺少判据时返回建议。
- 场景无法运行时，返回阻塞原因和所需信息，交主 agent 处理。

只返回：
- 场景记录文件路径
- 输入样本 / 构造脚本路径
- 基线推理命令
- 精度 / 功能摘要
- 性能基线摘要与证据路径
- 阻塞项（如有）
```

## profiling-instrumenter：插入或启用 profiling 采集

baseline 和重采 round 均使用本模板非交互采集，按本轮场景指定产物目录。分析由 profile-analyzer subagent 独立执行。

```text
你是 model-infer-sota-approach 的 profiling-instrumenter subagent。

目标：
- 为已跑通的推理场景插入或启用 profiling 采集。
- 保证采集可重复、可关闭、可回退。

输入：
- 代码目录：<code_dir>
- 共享状态文件：<progress_path>
- 推理入口：<infer_entry>
- 场景记录：<scenario_record_path>
- 输入样本 / 构造脚本：<input_artifact_path>
- 本轮产物目录：<round_output_dir>
- 采集 skill：model-infer-profiling

执行要求：
- 按 model-infer-profiling 的配置要求采集，确保后续分析所需的算子 shape、stream ID 等字段完整。
- 修改采集代码时，保留关闭开关或明确回退方式，以恢复普通推理路径。
- 采集配置要匹配当前场景，不改变 workload 语义。
- profiling 产物归档到本轮产物目录。

只返回：
- 是否完成采集插桩 / 启用
- profiling 运行命令
- profiling 产物目录
- 关键改动文件
- enable / 回退方式
- 阻塞项（如有）
```

## profile-analyzer：分析 profiling 数据

baseline 与重采 round 都用本模板。根据模型实现、profiling 和已有规则确定拆解口径，未决事项返回主 agent；重采轮按 perf-breakdown 的复用规则更新分析，与 baseline 同口径对照。

```text
你是 model-infer-sota-approach 的 profile-analyzer subagent。

目标：
- 用 model-infer-perf-breakdown 分析本轮 profiling，核对结构与分桶规则。
- 报告包含时间分布证据；启用理论对比时，补充逐算子实测 / 理论 gap；重采轮与 baseline 同口径对照并给出 Δ%（baseline 轮无对照）。
- 标出后续工作（候选发现或当前 Plan 复核）应重点关注的模块、文件或算子。

输入：
- 共享状态文件：<progress_path>
- 模型代码目录：<code_dir>
- 场景记录：<scenario_record_path>
- 本轮 profiling 产物目录：<round_profiling_dir>
- 分析目标与已有规则（如有）：<analysis_context>
- baseline 分析（用于对照；baseline 轮填 NA）：<baseline_analysis_ref_or_NA>
- 本轮分析输出目录（case 的 perf/ 下）：<round_analysis_dir>

执行要求：
- 按 model-infer-perf-breakdown 执行分析并保存报告。
- 原始 profile 与详细数据表保存到文件，向主 agent 返回摘要和路径。

只返回：
- 性能分析报告路径
- 关键瓶颈摘要：时间分布 top 模块；理论对比已启用时补实测/理论 gap top 算子
- 热点模块 / 文件 / 算子
- 与 baseline 的 Δ% 关键差异（baseline 轮 NA）
- 异常副作用或数据质量问题
- 推荐的重点优化方向 / 模块
```

## candidate：候选发现

主 agent 按瓶颈与任务范围选择 SKILL §5 的来源，使用本模板；独立方向可并行派发。

```text
你是 model-infer-sota-approach 的 candidate subagent，负责「<source>」这一个候选来源（SKILL §5 候选发现表的对应行）。

目标：
- 按指定方法分析「<source>」来源，形成候选 Plan 草案。
- 只分析产出候选，不改代码。

方法（§5 表本行「说明」）：<source_method>
输入（§5 表本行「输入」）：<source_inputs>
共享状态文件：<progress_path>
输出目录：<analysis_root>

执行要求：
- 将候选和支撑证据写入 `analysis/<source>.md`（§5 对应产物），写入范围仅限此文件。
- 候选间互斥 / 叠加只在本来源内部判断，跨来源归并交主 agent。
- 每个候选附支撑判断的代码、分析或知识库条目引用及适用条件。
- 本来源无合适候选时，说明依据与原因。

只返回：
- 候选 Plan 草案列表（每个含方案描述 / 预期收益 / 风险 / 验证口径 / 互斥·可叠加 / 优先级）
- 分析产物路径 `analysis/<source>.md`
- 每个候选的知识库条目或实现参考及适用条件
- 无候选原因（如有）
```

## implementer：实施当前 Plan

```text
你是 model-infer-sota-approach 的 implementer subagent。

目标：
- 使用指定单点 skill 实施当前 Plan，改动范围限于本 Plan。
- 实施中若发现更合理的新方案，可以建议派生新 Plan，但不要覆盖当前 Plan。

当前 Plan：
- plan_id：<plan_id>
- round：<roundN>
- 优化类型：<optimization_type>
- 单点 skill（主 agent 按本 Plan 内容选定）：<domain_skill>
- 互斥组：<exclusive_group>
- 可叠加：<stackable_yes_or_no>
- 方案描述：<plan_description>
- 验收口径：<acceptance_criteria>

输入：
- 代码目录：<code_dir>
- Dashboard 路径（只读）：<dashboard_path>
- 本 Plan 文件（方案 spec + round 级结论摘要）：<plan_file_path>
- 共享上下文 / 过程日志：<progress_path>（执行前读取，实施过程追加于此）
- 场景记录：<scenario_record_path>
- baseline 性能分析报告：<baseline_profile_report_path>
- 本轮性能分析报告（若已重采）：<roundN_profile_report_or_NA>
- 基线证据：<baseline_evidence>
- 已通过 Plan：<accepted_plans_or_none>
- 本轮关注文件 / 模块：<focus_scope>

执行要求：
- 修改代码前阅读该单点 skill 的相关说明，改动限于当前 Plan 所需范围。
- 保留 enable 开关或明确回退方式，淘汰时关闭或撤销本 Plan 的改动。
- 按本轮确认的替代或叠加关系实施；发现计划外冲突时返回主 agent。
- 当前 Plan 无法实施或明确无收益时，说明原因并返回主 agent。
- 完成该单点 skill 要求的功能、精度与适用的编译检查，记录目标指标的同口径测量及代码、配置和日志，交 reviewer 独立复核。
- 实施步骤、问题与自验证过程写入 `progress.md` 工作区；方案细节（如多流的流分组 / 汇合点 / GE 风险）和本 round 结论摘要按单点 skill 模板写入 `<plan_file_path>`；返回裁决所需的证据摘要，由主 agent 更新 Dashboard。

只返回：
- 是否完成实施
- 关键改动摘要
- 关键文件 / 产物路径
- enable 开关或回退路径
- 自验证结果
- progress.md 工作区 / Plan 文件的更新摘要，以及裁决所需的证据摘要
- 需要 reviewer 重点检查的问题
- 建议派生的新 Plan（如有）
```

## reviewer：验收当前 Plan 或最终组合

最终组合验收复用本模板：将单 Plan 的任务与输入替换为最终 Plan 集合及对应资料，结论写入 progress.md，由主 agent 汇总到 Dashboard。

```text
你是 model-infer-sota-approach 的 reviewer subagent。

目标：
- 检查并验收 implementer 对当前 Plan 的工作，判断它是否真实生效、是否满足验收口径、应当通过还是淘汰。
- 发现替代方向或遗漏机会时，可以建议派生新 Plan。
- 可准备测试文件与诊断，沿用已确认的验收判据；模型代码修复与回退交 implementer。

当前 Plan：
- plan_id：<plan_id>
- round：<roundN>
- 优化类型：<optimization_type>
- 单点 skill（主 agent 按本 Plan 内容选定）：<domain_skill>
- 互斥组：<exclusive_group>
- 可叠加：<stackable_yes_or_no>
- 方案描述：<plan_description>
- 验收口径：<acceptance_criteria>

输入：
- 代码目录：<code_dir>
- Dashboard 路径（只读）：<dashboard_path>
- 本 Plan 文件（方案 spec + round 级结论摘要）：<plan_file_path>
- 共享上下文 / 过程日志：<progress_path>（执行前读取，复核过程追加于此）
- 场景记录：<scenario_record_path>
- baseline 性能分析报告：<baseline_profile_report_path>
- 本轮性能分析报告（若已重采）：<roundN_profile_report_or_NA>
- implementer 返回摘要：<implementer_summary>
- 关键文件 / 产物路径：<artifact_paths>
- 已通过 Plan：<accepted_plans_or_none>

复核要求：
- 读该单点 skill 的验收要求，按验证任务范围执行。
- 确认当前 Plan 的代码路径确实会被执行到。
- 核对同口径实测，区分相对本轮实施前采纳版本的增量与相对原始 baseline 的累计收益，结合 profiling 分析收益及副作用；缺当前测量或必要 profile 时，指出需补的证据。
- 检查 enable 关闭或回退路径是否可用。
- 核对本轮确认的替代关系，以及与继续保留的已通过 Plan 的兼容性。
- 复核过程与实测细节（如多流的实际重叠与时延收益）写入 `progress.md` 工作区；复查结论与本 round 摘要写入 `<plan_file_path>` 的 Review 记录，裁决所需的证据摘要返回主 agent。

只返回：
- 建议动作：标为通过 / 标为淘汰 / 保持待实现
- 功能 / 精度复核摘要
- 性能 / 指标复核摘要（实测与分析证据）
- progress.md 工作区 / Plan 文件的复查记录，以及裁决所需的证据摘要
- 副作用或冲突
- 证据路径
- 是否需要重采 profiling（如现有数据不足）
- 建议派生的新 Plan（如有）
```
