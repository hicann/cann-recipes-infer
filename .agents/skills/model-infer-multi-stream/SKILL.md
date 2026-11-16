---
name: model-infer-multi-stream
description: NPU 多流技术知识技能。提供整网模块 / 算子 DAG 拆解、模块间与模块内并行性判断、多流候选编排派生、TorchAir(Ascend IR/GE) 与 npugraph_ex/aclgraph 的多流 API 路由、切流 / 同步 / 控核实现，以及假并行（overlap_pct）排查与 Profile 验证等技术规则。供两类工作引用：形成多流优化候选方案，以及实施和复查多流改造。触发场景包括：多流、双流、stream overlap、控核、limit_core_num、整网 DAG、模块拆解、npu_stream_switch、npu_wait_tensor、TorchAir 多流、record_stream、多流假并行排查。
---

# NPU 多流技术知识

本技能提供多流方案的分析、实现和验证方法；候选取舍、轮次推进与性能数据采集由调用方组织。

它支持两类工作：

- **方案分析**：结合模型代码与 profiling，识别多流机会，提出值得验证的编排候选。
- **实施 / 复查**：实现多流方案，包括选择执行路径、切流、同步和控核，并验证是否实际并行。

## 使用入口

| 任务 | 参考章节 | 要点 |
| --- | --- | --- |
| 多流方案分析 | 下文「方案分析」一节 + [`references/module-decomposition.md`](references/module-decomposition.md) + [`references/plan-detail-fragment.md`](references/plan-detail-fragment.md) | 按任务范围拆模块 / 算子 → 判断并行性 → 派生有实质差异的编排 |
| 多流方案实施 | 下文「实施」一节 + [`references/api-routing.md`](references/api-routing.md) | 承接方案与依赖分析，按执行路径切流 / 同步 / 控核 |
| 多流方案复查与调试 | 下文「调试与验证」一节 + [`references/timeline-overlap-check.md`](references/timeline-overlap-check.md) + [`references/plan-detail-fragment.md`](references/plan-detail-fragment.md) | 验证实际重叠与时延收益，按问题类型排查 |

产出包括可共享的**方向级分析**（模块 / 算子 DAG 与并行性判断，模板见 [`references/module-decomposition.md`](references/module-decomposition.md)）和**各方案细节**（流分组、GE 风险及实施后的实测，模板见 [`references/plan-detail-fragment.md`](references/plan-detail-fragment.md)）。各方案引用共享分析。

## 核心原则

- **按实际时间线验证并行**：结合算子落流、`Start Time(us)` 和 `Duration(us)` 检查预期分支的执行重叠，再对比时延收益。测量口径见 [`references/timeline-overlap-check.md`](references/timeline-overlap-check.md)。
- **按任务范围拆解**：全局候选发现先看代表性完整 decoder layer 的模块关系；指定模块或已有方案围绕目标及其生产者、消费者、共享状态和通信依赖分析，按并行性判断需要下钻算子。
- **开关必须可关闭**：多流路径要保留 enable 开关和原始回退路径，便于调试、复查及停用优化路径。
- **先证明正确，再追性能**：先验证依赖、功能和精度，再看 overlap 和时延收益。
- **overlap 不等于收益**：出现拖尾、资源争抢、host bound、shape 劣化时，overlap 真实成立也可能不降 wall，要继续评估控核和图模式限制。

## 执行路径定界

分析时确认当前执行模式及多流约束，实施时按对应 API 路径接入，见 [`references/api-routing.md`](references/api-routing.md)。

这里的 Ascend IR 指 TorchAir 的 GE 图模式路径。

- `torchair.CompilerConfig()` + `torchair.get_npu_backend()`：走 **Ascend IR / GE 图模式** 路径。
- `torch.compile(..., backend="npugraph_ex")`：走 **npugraph_ex / aclgraph** 路径。

## 方案分析

详细规则及完整模板见 [`references/module-decomposition.md`](references/module-decomposition.md)：

1. **拆模块、画模块级 DAG、判模块间并行**：在分析范围内沿子网络 / 通信 / 状态 / 同步 / 资源边界拆成语义完整、输入输出明确、可独立讨论调度的模块；标注依赖类型（`data` / `state` / `event` / `collective_order`），据此把模块对判成 `serial` / `parallel_candidate` / `parallel_pending_validation`。
2. **按需拆算子、画算子 DAG、判模块内并行**：需要判断模块内分流或细化依赖、资源占用时，沿计算 / 通信 / 同步 / 状态边界拆，识别可并行的算子和需要保持的依赖边。
3. **派生候选编排**：把共享同一输入、彼此无 `data` 依赖的模块 / 算子看成一个集合，按集合划分思考分流。结合并行对象、流分组、汇合点、tag 粒度或跨阶段 overlap，提出并行安排或收益假设有实质差异的候选。

计算与通信串行且等待明显时，可评估独立 micro-batch 的流水，结合切分后的算子效率与 host 开销判断收益。已有 KVCache offload 或分离式部署时，检查搬运 / 通信与本地独立计算的重叠窗口，核对读写边界、带宽与额外 buffer / clone 开销。

按当前模型结构和执行模式，查询 cannbot-knowledge 中同框架、同阶段的多流部署样例，如共享专家、Attention 前处理、KVCache 异步搬运或 micro-batch 通信计算流水。

常见派生维度见 [`references/module-decomposition.md`](references/module-decomposition.md)；同一编排的控核等配置调优记录在对应方案内。

**GE 调度风险**：检查副流输出在主流上的消费者及其前处理；若等待被放到原本可并行的主流计算之前，可能使两条分支串行。设计时明确依赖与汇合位置，实施后核对编译后的落流和时间线。

## 实施

承接已有方案与依赖分析，对照当前代码及 [`references/api-routing.md`](references/api-routing.md) 确认执行路径、仓库封装及切流、同步、控核约束。

1. **先做最小可验证实现**：按方案设置流与汇合点，完成必要同步，保留原路径和 enable 开关。
2. **显式 stream 路径检查内存生命周期**：短生命周期 tensor 若被其他 stream 继续消费，补 `record_stream()`；模型权重、常驻 cache 这类长生命周期对象一般不需要。
3. **根据实测调整编排与资源**：结合等待位置、资源争抢和拖尾调整同步位置或控核配置；核数结合目标硬件与实测调整。
4. **验证后再扩大改动**：用 profiler 检查实际落流与 overlap；充分预热后，在同负载下多次对比单流 / 多流时延，取稳定运行的中位数。

## 调试与验证

按现象选择排查方向：

1. **依赖错误 / 同步错误**：重点查事件记录、等待顺序、跨流汇合点、共享状态写入次序；典型现象是读到未完成结果、死等、结果偶发错误。
2. **精度 / 功能异常**：先对比优化前基线，再按 `prefill/decode → 模块 → 算子` 缩小范围，判断是不是多流引入的状态时序问题，而不是算子本身的问题。
3. **性能无收益**：结合落流、执行时间线与资源占用定位：

   - **3a 未按预期切流**：检查执行阶段、enable 开关和切流上下文是否实际生效，并核对编译后的算子落流。
   - **3b 主副流重叠不足**：按 [`references/timeline-overlap-check.md`](references/timeline-overlap-check.md) 检查执行区间、等待链和资源占用；GE 路径还需核对前处理算子的调度位置，再调整分流集合与汇合点。
   - **3c 已有重叠但 wall 无收益**：看 shape 变化、task 数增加、host bound、带宽与核资源争抢、拖尾。资源判断结合实际核配置与 pipeline 利用率；跨流 tensor 较大时关注 `TransData` / `BroadcastTo` / `MemSet` 的耗时变化。
4. **图模式 / runtime 限制**：按当前后端核对 graph break、图捕获 API 约束及 stream 语义。
5. **aclgraph 生命周期错误**：重点查短生命周期 tensor 是否跨流继续使用、是否遗漏 `record_stream()`；典型现象是结果偶发错误、数据踩踏、段错误或异常内存问题。

具体性能字段及 GE 重排、同步阻塞、跨流内存问题的排查经验，按 profiler 版本和执行模式查询 cannbot-knowledge。

**现象 → 下一步技术方向**：

| 现象 | 技术方向 |
| --- | --- |
| 改造点之外的 TransData / BroadcastTo / MemSet wall 显著上升 | 结合 shape 和调度变化核对额外搬运或 GE 重排；缩小副流、换集合划分，或先做算子合并让分流单元更独立 |
| 副流已落 side stream，但 layer wall 几乎不变 | 检查实际重叠、等待位置与资源占用，再决定调整分流集合、汇合点或资源分配 |
| 主流出现新空洞 / 拖尾，副流已提前结束 | 核对等待与汇合位置；必要时调整汇合点，或将具备并行条件的下游算子安排到副流继续执行 |
| 主 / 副流 cube 利用率叠加高于单流，但 wall 没降 | 结合核占用与算子耗时核对争抢；评估资源互补的集合划分，如 cube 与 vector / memory 路径组合 |
| 多流后某个原本未关注的 Cast / Reshape / DynamicQuant 位置变化 | 核对该算子的依赖、等待和耗时，再决定是否将其纳入候选集并调整分流集合 |

**Profile 验证要点**：

1. **Stream ID 落点检查**：副流算子的 `stream_id` 确实落到目标 stream、没被 GE 挪回主流；异常时按 3a 排查。
2. **overlap_pct 时间线验证**：按 [`references/timeline-overlap-check.md`](references/timeline-overlap-check.md) 计算实际执行重叠，并结合关键路径与 wall 变化判断收益。
3. 控核场景下按算子 Core 类型检查核配置是否生效，结合同时段的 Cube / Vector 利用率和耗时变化判断资源争抢。
4. 检查汇合点新增的 `EVENT_WAIT` 或空闲区间，判断等待位置对关键路径的影响。
5. 跨流 tensor 名称和维度与设计一致，没把大 tensor 搬运变成新瓶颈。

**功能 / 同步复查要点**：enable 前后输出一致、开关关闭后原路径正常；汇合点前后无缺失等待，共享状态 / KVCache / 通信结果没有读写乱序。

## 参考文档索引

- **模块 / 算子拆解与候选派生（方向级分析产物结构）**：[`references/module-decomposition.md`](references/module-decomposition.md)
- **多流 Plan 方案细节片段（Plan 自由块结构）**：[`references/plan-detail-fragment.md`](references/plan-detail-fragment.md)
- **多流 / 控核 API 路由**：[`references/api-routing.md`](references/api-routing.md)
- **overlap 时间线复盘（执行重叠与收益）**：[`references/timeline-overlap-check.md`](references/timeline-overlap-check.md)
