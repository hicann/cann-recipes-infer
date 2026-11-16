# Plan 文件模板（`plans/plan-<id>.md`）

主 agent 根据 candidate 草案初始化方案描述、细节与参考资料；implementer / reviewer 追加方案细节和 round 结论。主 agent 核查证据后将状态与摘要写入 dashboard。

实施、问题与验证过程写入 `progress.md` 工作区；本文件保留方案与 round 结论。

按以下结构记录；多流方案细节使用[多流 Plan 模板](../../model-infer-multi-stream/references/plan-detail-fragment.md)：

```md
# <plan_id>：<方案名>

## 方案描述

<说明优化对象、预期收益依据、验证方法与主要风险，内容可自由组织。>

## 知识库与实现参考

<支撑本 Plan 的知识库条目、代码或文档引用及适用条件，供 implement / review 按需查阅。>

## 方案细节

<按该 Plan 的单点 skill 模板记录。例如多流方向：流分组 / 汇合点 / 方案 DAG / 实施安排 / GE 调度风险（Ascend IR 路径）/ overlap_pct 实测 / enable·回退 / 副作用。裁决所需信息汇总为证据摘要，由主 agent 写入 Dashboard。>

## 实施记录

| round | implementer 摘要 | 关键文件 / 产物 | enable / 回退路径 | 派生建议 |
| --- | --- | --- | --- | --- |

## Review 记录

| round | reviewer 结论 | 功能 / 精度 | 性能 / 指标（实测与分析证据） | 副作用 | 建议动作 | 派生建议 |
| --- | --- | --- | --- | --- | --- | --- |

## 派生记录

| round | 触发现象 | 派生 Plan | 来源 | 说明 |
| --- | --- | --- | --- | --- |
```
