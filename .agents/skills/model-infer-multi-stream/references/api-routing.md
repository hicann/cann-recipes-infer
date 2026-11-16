# 多流与控核 API 路由

按当前执行模式选切流与同步方式。具体接口、图捕获兼容性及控核支持范围，按部署的 CANN / torch_npu / TorchAir 版本查询 cannbot-knowledge 或官方文档。

## 执行路径

| 路径 | 切流与同步 | 控核 |
| --- | --- | --- |
| Ascend IR / GE 图模式 | `torchair.scope.npu_stream_switch`；跨流时序用 `torchair.scope.npu_wait_tensor`，已有 tagged event 风格时沿用 `torchair.ops` 下的 `npu_record_tagged_stream` / `npu_tagged_event_record` / `npu_tagged_event_wait` | 算子级 `torchair.scope.limit_core_num`，优先级高于全局 `config.ge_config.aicore_num` |
| npugraph_ex / aclgraph | `torch.npu.Stream` / `torch.npu.stream`，配合 Event 或 Stream 的 record / wait 接口 | Stream 级 `torch.npu.npugraph_ex.scope.limit_core_num` |
| eager / 外部框架 patch | 沿用目标框架的 stream / event 管理和生命周期；显式流使用 `torch.npu.Stream` / `Event` | 按目标运行时选择资源控制方式 |

涉及 `enable_single_stream`、SuperKernel 或 micro-batch 时，核对多流组合限制与算子控核支持范围。核数上限以目标设备为准，参数结合实际资源占用与拖尾调优。

## infer 仓框架部署

优先复用 [`executor/utils/stream_utils.py`](../../../../executor/utils/stream_utils.py)，按接口传入 `exe_mode`，与目标模型的流和事件初始化保持一致：

| 封装 | 使用要点 |
| --- | --- |
| `create_stream` / `create_event` | 初始化阶段创建并复用；GE 使用流 tag，其他路径使用原生 Stream / Event |
| `npu_stream_switch` | 根据 `exe_mode` 选择 GE scope 或显式 stream；开关关闭时使用空上下文 |
| `record_event` / `wait_event` | 用于原生 Event 路径，在生产流记录、消费流等待；GE 路径通过数据依赖或 `wait_tensor` 表达时序 |
| `record_stream` | 显式流路径中，短生命周期 tensor 被其他流继续使用时跟踪其生命周期 |
| `limit_core_num` | 显式传入 `exe_mode`，分别选择 GE 算子级或 npugraph_ex Stream 级控核 |

现有 GE 实现也可能使用 [`executor/utils/common_utils.py`](../../../../executor/utils/common_utils.py) 的 tagged event 封装；同名 `record_event` / `wait_event` / `record_stream` 的语义与上述原生接口不同，按目标实现的 import 与调用关系选用。

## 同步与资源判断

- 副流消费主流产生的输入前建立依赖，主流在实际消费副流结果的位置汇合；流 / event 的创建与复用覆盖完整前向生命周期。
- 同步确定执行顺序，`record_stream` 管理内存释放时机；模型权重、常驻 cache 等长生命周期对象通常无需额外跟踪。
- 结合实际执行重叠、Cube / Vector 占用和拖尾决定是否控核；核数调整后复查算子耗时与端到端收益。
- 预取候选需有可掩盖搬运的计算窗口，且前序路径不会明显争抢带宽；具体改造由 `model-infer-prefetch` 处理。
