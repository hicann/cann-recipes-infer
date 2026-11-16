# 多流 Plan 方案细节片段

用于记录单个多流方案，也可作为编排报告中 Plan 的「方案细节」。**形成方案时**给出设计部分，**实施 / 复查后**回填实测部分。

一个多流 Plan 的方案细节包含：

- **并行对象与流分组**：参与并行的算子 / 模块及其流归属（主流 / 副流 1 / 副流 2）；tag 粒度（同一副流串行或多个独立副流）。
- **汇合点**：汇合位置与同步方式（Ascend IR 用 `npu_wait_tensor` / tagged event；aclgraph 用 `Event` / `wait_stream`）。
- **方案 DAG**：Mermaid，标清主 / 副流归属和 event 边（`-.->`）。
- **实施安排**：包含必要同步的最小实现范围，以及待验证的同步位置或控核调整。
- **GE 调度风险**（Ascend IR 路径设计期填）：副流输出 → 主流消费者及前处理 → 等待位置 → 对并行窗口的影响与应对；实施后核对编译调度。
- **overlap_pct 实测**（实施 / 复查后填，按 [`timeline-overlap-check.md`](timeline-overlap-check.md)）：主副流目标算子与执行区间 / 副流有效执行时长 / 实际重叠时长及比例 / 结合 wall 变化的判断。
- **enable 开关与回退路径**：开关名及关闭后使用的原路径。
- **风险与副作用**：`TransData` / `BroadcastTo` / `MemSet` 长尾、核资源与带宽争抢、shape 劣化等。

**证据摘要**：`overlap_pct` 判定 + wall Δ%（以性能分析报告为准）+ 关键副作用，供调用方决定方案取舍。

> 共享的模块 / 算子 DAG 与并行性判断按 [`module-decomposition.md`](module-decomposition.md) 单独记录，在此引用。
