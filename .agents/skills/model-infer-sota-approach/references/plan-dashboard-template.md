# Plan Dashboard 模板

Dashboard 汇总推理场景、baseline、候选、Plan 状态与最终验收。约定落在 `optimization-analysis/<case>/plan-dashboard.md`，候选发现完成后初始化，回填此前的场景与 baseline 证据索引。

## 产物目录

产物按以下结构组织，默认目录为 `optimization-analysis/<case>/`：

```text
optimization-analysis/<case>/
├── plan-dashboard.md          场景、Plan 状态与裁决总览
├── scenario.md                场景记录（scenario subagent 产出，§1 链接）
├── analysis/                  各候选来源的分析产物（一来源一文件，文件名取 §5 表的来源名）
│   ├── <source>.md             一来源一文件，来源名见 SKILL §5 表
│   └── …
├── plans/                     每个 Plan 的方案、实施、review 与派生记录
│   ├── plan-A.md
│   └── …
└── perf/                      采集与分析产物，按 round 区分；内部结构由对应 skill 定义
```

`progress.md` 优先使用调用方指定文件，否则使用模型 `agentic/progress.md`。

引用关系（一份分析派生多个 Plan，不重复）：

```text
baseline 性能报告 ──证据──► analysis/<source>.md ──派生──► plans/plan-A.md / plan-B.md / …
（时间分布，理论对比已启用时含理论 gap）        （一份分析，多 Plan 共享）       每 Plan 一份完整记录
```

采集目录 `<round_output_dir>` / `<round_profiling_dir>` 和分析目录 `<round_analysis_dir>` 指向 `perf/` 下本轮的对应目录，保留历史测量与报告；`<analysis_root>` 是 case 根 `optimization-analysis/<case>/`。

## 文件职责与写入权限

- **dashboard**：Plan 状态的唯一来源，由主 agent 维护场景与基线索引、Plan 状态、采纳实现、round 裁决和最终验收。
- **明细（每个 Plan 一份 `plans/plan-<id>.md`）**：主 agent 根据 candidate 草案初始化；implementer / reviewer 追加方案细节和 round 结论，格式见 `plan-file-template.md`，领域细节沿用对应 skill 的模板。
- **方向级分析产物**：module/op DAG、候选等，一个来源一份 `analysis/<source>.md`，被该来源所有 Plan 共享，由 candidate 产出（见 §3）。

**证据摘要**：汇总每个 Plan 的实际生效情况、实测收益、精度、enable / 回退、副作用及证据路径。

`progress.md` 保存共享上下文与实施、验证过程；主 agent 核查明细及证据后更新总览。

## 1. 场景与基线

- 模型 / case：
- 代码目录：
- 推理入口：
- 推理场景：
- 性能目标（可选）：
- 精度 / 功能口径：
- 场景记录文件：
- 输入样本 / 构造脚本：
- 基线运行命令：
- 基线精度 / 功能结果：
- 目标指标的基线实测与证据路径：

## 2. Baseline Profiling 与性能分析（round0）

本节对应 round0：采集由 profiling-instrumenter 执行，分析由 profile-analyzer 根据代码、数据及已有规则完成，供后续 round 同口径对照。

- profiling 采集命令 / 产物路径：
- perf-breakdown 工作目录：
- 性能分析报告路径：
- 关键瓶颈摘要（时间分布 top 模块；理论对比已启用时补实测/理论 gap top 算子）：
- 热点模块 / 文件 / 算子：
- 数据质量或采集风险：

## 3. 候选发现记录

候选由所选来源的 candidate subagent 产出，独立方向可并行派发，来源见 SKILL §5。每个来源一行（`<source>` 取 §5 表的来源名）。候选之间的互斥 / 叠加关系在第 4 节由主 agent 裁定；实施 skill 在主流程第 7 步按 Plan 内容确定。

| 来源 | 分析产物 | 候选数量 | 摘要 |
| --- | --- | --- | --- |
| <source> | analysis/<source>.md |  |  |

## 4. Plan Dashboard

每个 Plan 一行，通过 `plan 文件` 列关联明细。

| plan_id | 来源 | 优化类型 | 单点 skill | 互斥组 | 可叠加 | 状态 | round | 证据摘要 | 淘汰 / 保留原因 | plan 文件 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Plan-A | candidate-1 |  |  | group-1 | yes/no | 待实现 |  |  |  | plans/plan-A.md |

字段说明：

- `plan_id`：稳定 ID，派生新 Plan 时递增，不复用旧 ID。
- `来源`：候选发现 subagent，或 `derived-from Plan-X/roundN`。
- `优化类型`：多流、融合算子、prefetch、图模式、KVCache、量化、并行等。
- `单点 skill`：实施该 Plan 用的单点技术 skill，由主 agent 在第 7 步按 Plan 内容判断填入（多流 Plan 已知是 `model-infer-multi-stream`，不是编排器）；候选发现阶段可留空。
- `互斥组`：跨方向的互斥关系由主 agent 在初始化时裁定；同组内最终只保留一个或一组兼容的 `通过` Plan，互不影响的 Plan 分属不同组。
- `可叠加`：填 `yes` 或 `no`；标 `yes` 的 Plan 即便各自通过，仍要在验收时验证叠加效果。
- `状态`：只允许 `待实现` / `通过` / `淘汰`。
- `round`：当前状态对应的全局 round 编号。
- `证据摘要`：从 Plan 文件提取可回查的结论与证据路径，详细日志和原始数据保存在对应产物中。
- `淘汰 / 保留原因`：一句话写核心原因；若派生了新 Plan，写明派生目标。
- `plan 文件`：该 Plan 的完整记录 `plans/plan-<id>.md`。

## 5. 当前采纳的实现

当前 `通过`、构成最终方案的 Plan 集合，随通过 / 淘汰 / 替代结果更新。互斥组内仅列出采纳方案，可叠加 Plan 并列记录。

| plan_id | 优化类型 | 单点 skill | 采纳的实现（代码改动与启用方式） | 收益摘要 | plan 文件 |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

## 6. Round 记录

记录每个 round 选择的 Plan、是否重采及裁决结论；实施 / review / profile 明细见对应 `plans/plan-<id>.md`。

| round | plan_id | 是否重采 profiling | profile 产物目录 | 主 agent 裁决 |
| --- | --- | --- | --- | --- |
| round1 |  | 是 / 否（沿用 roundX） |  |  |

## 7. 最终验收

进入本节前，先对最终代码路径重采一次 profiling，与 round0（baseline）做同场景、同口径对照。

- 本次停止依据与尚待验证的 Plan：
- 所有 `通过` Plan 是否都已应用到最终代码：
- 可叠加 Plan 的组合效果是否已实测验证：
- 同一互斥组是否只保留最终采用的 Plan：
- 未采纳的试验改动是否已回退或被开关关闭：
- 最终精度 / 功能是否仍满足第 1 节的场景口径：
- 最终目标指标是否按 baseline 同口径实测，profile 是否支撑收益归因：
- 最终采用方案：
- 淘汰方案摘要：
- 剩余风险：

---

每个 Plan 的完整记录 `plans/plan-<id>.md` 结构见 [`plan-file-template.md`](plan-file-template.md)。
