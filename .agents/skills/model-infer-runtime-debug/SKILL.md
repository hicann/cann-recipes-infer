---
name: model-infer-runtime-debug
description: 基于 PyTorch 框架的昇腾 NPU 模型推理运行时错误诊断与修复技能。系统化排查模型加载、初始化、推理执行全链路的运行时错误，包括 aicore timeout、HCCL 通信错误、OOM、算子约束违反、推理卡住等。触发场景：NPU 运行时错误（RuntimeError、aicore timeout 507014、HCCL timeout、device synchronize 失败、kernel crash、EZ9999/EE9999 错误码）、推理过程卡住不返回、权重加载阶段 crash、模型加载成功但 forward 失败、分布式推理某些 rank 挂死等。
---

# NPU 推理运行时错误诊断与修复

**先定位再修复**，逐级缩小故障范围。精度问题见 model-infer-precision-debug。

仅诊断时返回故障位置、证据和修复建议；包含修复的任务再实施并返回验证结果。诊断操作沿用调用方的权限约定。

> **infer 仓部署参考**：框架部署 / 独立部署的实现差异：
> - **启动协议**：框架部署走 `executor/scripts/function.sh::launch` 多卡 fork python，通过 `LOCAL_RANK` / `RANK_ID` 区分 rank；独立部署走 `torchrun --nproc_per_node=N infer.py`。卡住 / crash 时核对实际进程模型。
> - **通信组管理**：框架部署由模型通过 `CommManager.register_group()` 注册所需通信组，用 `comm_manager.get_group(name)` 取用；独立部署 Runner 内 inline `ParallelContext`，对外接口对齐 `get_group(name)` / `get_group_name(name)` / `get_rank(name)`。HCCL 错误诊断时确认是哪条路径。
> - **KV cache 接入**：框架部署由 `executor/core/kv_cache/` 管理 `cache_entries` 及各缓存分组的 `block_table` / `slot_mapping`；独立部署 Runner 自管 KV tensor + `block_table` + `slot_mapping`。详见 model-infer-kvcache。
> - **权重加载**：框架部署 `enable_online_split_weight: True` 由 ParallelLinear.weight_loader 自动按 rank 切；独立部署 Runner 自管或调 `executor.model_loader.weight_utils.default_weight_loader`。

---

## 诊断路径总览

```
问题发生
  │
  ├── 有明确错误信息 → 通用诊断流程
  │     ├── aicore timeout (507014) → 逐级定位
  │     │     ├── 定位到阶段 → 层 → 模块 → 算子
  │     │     └── 核对算子约束 → 修复策略 → 验证
  │     ├── OOM → 检查参数量 / batch_size / seq_len / 中间 buffer
  │     ├── Shape 不匹配 → 检查 TP/EP 切分维度 / Prefill vs Decode 分支
  │     └── HCCL timeout → 检查各 rank 代码路径一致性 / 通信组创建
  │
  └── 无明确错误 → npu-smi 状态检查
       ├── 部分 rank 缺失 / HBM 不均 → 多卡部署诊断
       │     ├── 权重加载 crash → TP 切分越界 / vocab pad
       │     ├── Config 字段缺失 → Config 兼容性
       │     └── 静默失败 → 逐 rank 检查日志
       ├── 有进程 + HBM 不变 + CPU 高占用 → 检查模型构造 / 初始化进度与进程栈
       ├── 有进程 + HBM 不变 + CPU 低占用 → 推理卡住诊断
       │     ├── 检查残留进程
       │     ├── eager 也卡 → sync+print 定位（通用流程第二步）
       │     └── 仅 graph 卡 → 比较 eager/graph 代码路径差异
       └── 无 NPU 进程 → 核对主机进程、退出状态与各 rank 日志
```

### npu-smi 状态检查

当没有明确错误信息时，用设备状态缩小排查范围，并结合进程栈和日志确认：

```bash
npu-smi info
```

| 设备 / 进程表现 | 排查方向 | 确认方式 |
|-----------------|---------|---------|
| 部分 rank 进程缺失或 HBM 不均 | 部分 rank 提前退出、加载进度或分工不同 | 核对预期 rank 与日志，定位最早异常 |
| 预期进程仍在，AICore 利用率低 | CPU 阶段、I/O 或通信等待 | 对照进程栈与各 rank 最后检查点 |
| HBM 不变，CPU 高占用 | 模型构造、参数初始化或编译耗时 | 检查当前调用栈和阶段进展 |
| HBM 不变，CPU 低占用 | I/O 或通信等待 | 检查阻塞位置与对端 rank 状态 |
| HBM 持续增长 | 权重 / 缓存加载或其他分配 | 对照加载日志与内存增长阶段 |
| 无 NPU 进程 | 尚未使用 NPU 或进程已退出 | 核对主机进程、退出状态和各 rank 日志 |

---

# 通用诊断流程

根据已有证据选择错误分类、同步检查点或专项排查；定位到具体算子后核对约束。

## 第一步：错误分类

### A. aicore timeout（错误码 507014）

**特征**：`AclrtSynchronizeDeviceWithTimeout`、`error code is 507014`、`aicore timeout`、`fftsplus aivector error`、`Kernel task happen error, retCode=0x25`

**含义**：某个 NPU 算子在设备上执行时超时未返回。这是最常见也最难排查的错误，因为报错的位置（synchronize 调用处）通常不是出错的位置（某个具体算子）。

**常见根因**：
- 算子入参违反当前平台 / 版本约束（见第三步）
- 算子 shape 超出硬件限制（如单次 matmul 的 M/N/K 维度超限）
- 死锁：部分 rank 走了不同的通信路径，导致集合通信永远等不齐
- 内存越界：slot_mapping/block_table 索引超出 KV cache 分配范围

### B. HCCL 通信错误

**特征**：`HCCL_CONNECT_TIMEOUT`、`HCCL error`、`AllReduce/AllToAll timeout`、`HCCL_EXEC_TIMEOUT`

**含义**：分布式通信操作超时或失败。

**常见根因**：
- 各 rank 进入通信操作的顺序/次数不一致（代码分支导致部分 rank 跳过某次通信，常见于 modeling forward 内有 `if forward_metadata.is_prefill` 等分支）
- 通信组创建时参数错误：框架部署核查模型调用 `CommManager.register_group()` 时的成员分组参数；独立部署核查 `ParallelContext.build_parallel_context()` 内 `init_comm_group` 调用参数
- MoE EP 通信组名称缺失或不匹配：dispatch_v2 / combine_v2 所用 HCCL group name 应对应模型实际注册的通信组，通过 `get_group_name(name)` 获取
- 网络问题（跨节点时 HCCL_IF_IP 配置错误）
- 通信缓冲区不足（结合报错、输入规模与当前算子的缓冲区要求确认）
- 某些 rank 已经 crash 但其他 rank 还在等它参与通信

通信超时或卡住时，按实际报错、部署方式和版本查询 cannbot-knowledge 中的 HCCL 建链、通信缓冲区配置及逐 rank 日志 / 进程栈定位方法。

### C. OOM（显存不足）

**特征**：`NPU out of memory`、`Tried to allocate X GiB`、`ENOMEM`

**含义**：NPU HBM 不够。

**常见根因**：
- 模型参数/KV cache/activation 总和超过单卡显存
- 中间 tensor 未及时释放（常见于 MoE EP all-to-all 中间 buffer）
- batch_size 或 seq_len 超出预期
- migrator 阶段单卡 OOM：编排层会标记"需多卡"，并行化（parallel-impl skill）后再补采基线，不是必修问题

### D. Shape 不匹配

**特征**：`RuntimeError: shape mismatch`、`expected size X but got Y`、`mat1 and mat2 shapes cannot be multiplied`

**含义**：tensor 维度不对。

**常见根因**：
- TP 切分后的维度或 rank 不匹配：核对模块的 `tp_size` / `tp_rank`；infer 仓按模块所属通信组通过 `comm_manager.get_rank(name)` 取组内 rank
- KV cache 的 `num_head` 与每卡实际 KV 布局不匹配：标准头切分 / 复制与 MLA latent cache 的口径见 model-infer-kvcache，改变 Attention TP 后同步核对
- EP 切分后每卡专家数计算错误：按实际专家分配核对，均匀切分时为 `experts_per_rank = num_experts // ep_size`
- Prefill/Decode 分支传入了错误 shape 的 tensor：核对阶段标识与实际执行分支，infer 仓 modeling 使用 `forward_metadata.is_prefill`
- packed sequence 协议错误：将 `[TotalTokens, hidden_size]` 直接 reshape 成 `[B, S, H]`，破坏变长 batch 的布局，详见 model-infer-kvcache
- 多缓存分组模型：向 FA 传入了整个 `block_table` 字典，应按当前层的缓存分组取对应 Tensor，详见 model-infer-kvcache

### E. 算子约束违反

**特征**：可能直接报参数错误，也可能表现为 aicore timeout 或 SIGABRT。根据报错或检查点定位后，查阅算子文档确认。

**含义**：NPU 自定义算子对入参有隐含约束（数据类型、维度范围、硬件平台限制），违反时行为未定义。

---

## 第二步：逐级定位（同步检查点法）

核心方法。当错误发生在 `torch.npu.synchronize()` 时，错误信息只告诉你"设备上有东西出错了"，不告诉你具体哪个算子。通过插入检查点逐步缩小范围。

### 2.1 定位到阶段（Prefill vs Decode）

在实际推理入口的 Prefill 完成、Decode 开始处插入同步检查：

```python
# 在 prefill 之后、decode 之前
torch.npu.synchronize()
logging.info("prefill passed")
```

**关键陷阱**：NPU 操作是异步的。`model.prefill()` 返回不代表 prefill 完成 —— 只有 `torch.npu.synchronize()` 返回才代表所有已提交的操作完成。所以报错位置（synchronize 调用处）不一定是出错位置（某个具体算子）。确保检查点覆盖所有可能的异步操作。

### 2.2 定位到层

确认出错阶段后（如 Decode），在每个 decoder layer 之间插入检查点：

```python
for i, layer in enumerate(self.layers):
    hidden_states = layer(hidden_states, ...)
    torch.npu.synchronize()
    logging.info(f"layer {i} passed")
```

排查最后成功检查点至首次失败检查点之间的执行区间。

### 2.3 定位到模块

进入出错层，在各子模块之间插入检查点：

```python
# 在 DecoderLayer.forward 中
hidden_states = self.input_layernorm(hidden_states)
torch.npu.synchronize(); logging.info("  norm passed")

hidden_states = self.self_attn(hidden_states, ...)
torch.npu.synchronize(); logging.info("  attention passed")

hidden_states = self.mlp(hidden_states)  # MoE
torch.npu.synchronize(); logging.info("  MoE passed")
```

### 2.4 定位到算子

进入出错模块，在每个 NPU 算子调用之间插入检查点。此时通常可以定位到具体算子及其入参，按第三步核对约束。

### 2.5 注意事项

- **先用 `torch.npu.synchronize()` 定位本 rank 的出错区间**。额外的 `dist.barrier()` 会让其他 rank 等待已失败的 rank，干扰定位
- **怀疑多 rank 不同步时**，对照组内检查点；需要超时报告未到达的 rank 时，用独立 Gloo 组调用 `torch.distributed.monitored_barrier(group=group_gloo, timeout=...)`（[PyTorch 文档](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.monitored_barrier)）
- **查看所有 rank 的日志**，从最早分歧处区分提前失败与通信分支不一致
- **不要一次插入太多检查点** —— 每次 sync 有开销，且大量 sync 可能改变时序。先粗粒度，再细粒度
- **保留日志**：把每轮定位的检查点输出保存，供后续分析

---

## 第三步：核对算子约束

定位到算子后，按 API、硬件和部署版本查询 cannbot-knowledge 中的约束与故障案例：MoE 查专家分布、路由参数和回退实现；FA 查每卡 Q/KV heads、dtype/layout、mask/sparse_mode、序列长度参数类型与 KV cache 格式。

### 通信算子

| 约束 | 说明 |
|------|------|
| `all_to_all_single` 的 split sizes | input_splits 和 output_splits 之和须分别等于 input 和 output 的第 0 维 |
| 通信组内调用须匹配 | 参与 rank 的 collective 顺序、次数及数据收发关系须匹配；分支跳过通信会让其他 rank 等待 |
| HCCL group 创建顺序 | 所有 rank 须以相同顺序创建相同的通信组 |

---

## 第四步：修复策略模式

定位根因后，选择合适的修复策略。

### 模式 1：算子回退（Operator Fallback）

确认某个高性能算子在当前硬件 / 配置下不可用时，回退到语义等价且满足当前配置的替代路径，并验证精度与性能。

**适用场景**：硬件约束、EP/TP size 不满足算子要求
**注意**：回退通常有性能损失

### 模式 2：参数修正（Parameter Fix）

算子入参计算错误（如 FA `head_num` 用了 KV heads 而非 Q heads、TP 切分后维度未除以 tp_size）。定位到具体算子后，逐个核对入参与算子文档/参考模型的差异。

**适用场景**：shape 不匹配、dtype 错误、维度计算错误

### 模式 3：路径统一（Path Unification）

部分 rank 因分支跳过集合通信时，从首次分歧处核对相关通信组的控制状态（如 `is_prefill`）及其来源，修正组内状态传递和 collective 调用顺序。独立处理不同请求的 DP 副本按各自请求推进。

**适用场景**：多 rank 死锁、HCCL timeout

### 模式 4：配置降级（Configuration Downgrade）

配置组合不兼容时，调整 YAML 参数（如降低 `moe_chunk_max_len`、减小 `batch_size`）。

**适用场景**：OOM、性能异常

---

## 第五步：验证清单

修复后，按以下清单验证：

- [ ] **局部验证**：复跑原失败路径，确认错误或卡住现象消失
- [ ] **全流程验证**：移除调试检查点，按目标输入规模、并行配置和执行模式运行完整 warmup + inference
- [ ] **全 rank 验证**：结合本次任务进程的退出状态、各 rank 日志与推理入口的完成记录，确认所有参与 rank 正常结束
- [ ] **输出正确性**：按请求与基准比较结果；不同 DP 副本可处理不同请求，按并行角色检查应一致的副本输出
- [ ] **性能记录**：记录修复后的 Prefill/Decode 耗时；有可用的成功基线时，按相同负载对比
- [ ] **回退结果**：如果使用了算子回退，在交付结果中说明性能影响和未来可恢复条件

---

# 特定场景诊断

按症状选择以下排查路径，必要时结合前述同步检查点定位。

## 多卡部署诊断

本节关注多卡加载阶段的权重切分、vocab 对齐、config 兼容性，以及部分 rank 失败后其他 rank 等待的现象。

加载或初始化异常时，查询 cannbot-knowledge 中同推理框架、相近模型结构的 TP 分片、词表填充与加载配套、Config 字段映射及参数初始化耗时处理案例。

### 权重加载 TP 切分越界

遇到 `narrow()` 越界或维度不可整除错误时，对照 checkpoint 实际维度、模型参数维度、对应 TP 度及本 rank 的切分范围。高位 rank 的切片起点更大，若仅高位 rank 失败，应优先核对其 `start + length` 是否超出源权重维度。

Embedding 与 LMHead 可能使用不同词表大小和 TP 度。均匀词表切分时，词表维度须分别满足 `embed_tp_size` / `lmhead_tp_size` 的整除要求。采用 vocab padding 时，模型参数与加载权重的填充口径须一致，并保持有效 token 范围。

### Config 兼容性问题

从 HuggingFace 原始模型加载时，核对实际加载的 Config 类、config.json 字段语义与模型构造代码，按模型定义处理字段缺失、格式或层数口径差异。

### 部分 Rank 静默失败

有时 rank crash 了但错误被吞掉，存活的 rank 卡在下一个 `dist.barrier()` 或集合通信上，看起来像"推理卡住"。

对照本次任务的预期 rank、实际进程与各 rank 日志，找出最早失败的 rank 及其最后完成的阶段，再检查仍在等待的 rank 是否依赖它。

### 多卡权重加载检查

加载异常时，按失败位置核对：

- **ignore 规则覆盖**：加载器的忽略规则只覆盖当前推理路径不需要的权重
- **模型路径**：核对各 rank 按其工作目录实际解析的模型路径
- **enable_online_split_weight 与 weight_loader 实现**：infer 仓 YAML 设 True 时，ParallelLinear / FusedMoEGMM 的 weight_loader 须正确处理 tp_rank / ep_rank 切分

---

## 推理卡住诊断

进程仍在运行但不产出结果时，结合设备状态、各 rank 日志和进程栈，区分阶段耗时与阻塞。

### eager 模式复现

```
卡住 → 保持模型、输入和并行配置，切换 eager 复现
  ├── eager 也卡 → 用同步检查点定位共用执行路径
  ├── eager 正常 → 重点检查图模式路径
  │     ├── 图中断或编译报错 → 按 model-infer-graph-mode 定位具体原因
  │     └── 无报错但慢/卡 → 比较 eager 和 graph 模式代码路径差异
  └── 两者都正常但上次卡了 → 对比运行条件，检查残留进程与偶发异常
```

### 初始化 / 编译耗时

日志暂时不更新、HBM 不增长而 CPU 持续忙碌时，结合进程栈区分模型构造、参数初始化、权重加载与编译。若耗时集中在随后会被 checkpoint 完整覆盖的参数初始化，可跳过对应随机初始化；具体处理参考前述同框架初始化案例。

### 残留进程导致 HCCL 卡住

**症状**：新启动的推理卡在 HCCL init，不报错不超时

**排查方向**：上次推理的进程可能未正确退出，仍占用 NPU 设备或 HCCL 端口。

**修复**：核对残留 PID 的启动命令、任务归属与 rank，仅清理本次任务遗留的进程；确认其设备与端口资源释放后重新启动。
