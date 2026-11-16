---
name: model-infer-parallel-analysis
description: 基于 PyTorch 框架的昇腾 NPU 模型推理并行策略分析技能。分析模型架构参数和昇腾硬件规格，推荐最优的 TP/EP/DP 并行配置（parallel_config）。触发场景：新模型需要确定并行策略、现有配置需要优化、部署卡数或硬件变更后需要重新评估。输出为结构化的 parallel_config 推荐及定量依据。
---

# 模型并行策略分析

> **本 skill 仅做分析和推荐，不修改任何代码。** 输出为 parallel_config 推荐和方案文档。

---

## 适用范围

- **模型类型**：Dense Transformer、MoE Transformer
- **硬件平台**：昇腾 Atlas A2 / A3 系列
- **决策范围**：`parallel_config` 中各 `*_tp_size`（TP）、`o_proj_tp_size`、`cp_size`（CP）的值，以及目标框架并行布局下的 DP/EP 度

## recipes 部署约束

适用于 cann-recipes-infer 部署。并行度推导与组合限制以目标版本的 [ParallelConfig](../../../executor/core/config/inference_config.py) 为准，以下用于当前常规布局的分析：

```
attn_dp_size  = world_size // attn_tp_size
dense_dp_size = world_size // dense_tp_size
moe_ep_size   = world_size // moe_tp_size    # EP 度由 moe_tp_size 决定，与 attn_tp_size 无关
cp_prefill_dp_size = world_size // (attn_tp_size * cp_size)
```

各模块独立计算并行度。启用 CP 时，Prefill 的请求分配使用 `cp_prefill_dp_size`，不能直接沿用 Decode 的 `attn_dp_size`。

当前 CP 实现要求：`cp_size > 1` 时，`cp_size == world_size` 且 `attn_tp_size == 1`。`model-infer-parallel-impl` 当前未覆盖 CP 接入步骤，采用 CP 的方案需单独评估目标模型的适配工作。

## 分析流程

```
第一步：提取模型参数与模块链路
    ↓
第二步：模块级拆解 → 定性分类 → 候选并行模式
    ↓
第三步：定量估算
    ↓
第四步：方案审查 → 输出推荐
    ↓
（可选）第五步：Profiling 数据校准
```

---

## 第一步：提取模型参数与模块链路

### 1.1 基础参数

从模型 config（`config.json` 或 `configuration_*.py`）中提取：

| 参数 | 符号 | 获取字段 | 说明 |
|------|------|---------|------|
| 架构类型 | - | 模型文档 | Dense / MoE / MoE+MLA / MoE+DSA |
| 总层数 | L | `num_hidden_layers` | |
| MoE 层数 | L_moe | `num_moe_layers` 或推断 | Dense 模型为 0 |
| Hidden size | H | `hidden_size` | |
| FFN 中间维度 | H_ffn | `intermediate_size` | |
| 注意力头数 | N_h | `num_attention_heads` | |
| KV 头数 | N_kv | `num_key_value_heads` | GQA: N_kv < N_h |
| Head dim | D_h | H / N_h 或 `head_dim` | |
| 专家总数 | E | `num_experts` | Dense 模型为 0 |
| 激活专家数 | E_act | `num_experts_per_tok` | |
| 词表大小 | V | `vocab_size` | |

MLA 模型额外提取：

| 参数 | 获取字段 | 说明 |
|------|---------|------|
| KV 压缩维度 | `kv_lora_rank` | MLA 的 compressed KV dim |
| Q 压缩维度 | `q_lora_rank` | MLA 的 compressed Q dim |
| V head dim | `v_head_dim` | oproj_tp 约束：`N_h * v_head_dim % oproj_tp == 0` |

### 1.2 模块链路识别与并行拆解

按模块识别 Prefill 和 Decode 阶段的计算特性与可选并行维度：

| 模块 | Prefill 特性 | Decode 特性 | 并行维度 |
|------|-------------|-----------|---------|
| Embedding | 大词表矩阵 | 大词表矩阵 | embed_tp |
| Attention QKV | 计算密集（大 S） | 访存密集（S=1） | attn_tp / cp_size |
| Attention Core（FA） | 计算密集 | 访存密集 | attn_tp / cp_size |
| Attention O_proj | 计算密集 | 访存密集 | oproj_tp（MLA 模型独立设） |
| Dense FFN | 计算密集 | 访存密集 | dense_tp |
| MoE Router | 轻量 | 轻量 | 跟随 EP |
| MoE Expert FFN | 计算密集 | 分散访存 | moe_tp（通常=1，走 EP） |
| LM Head | 大词表矩阵 | 大词表矩阵 | lmhead_tp |

> **配置口径**：每个候选方案默认使用一套 parallel_config，兼顾 Prefill 和 Decode，并根据用户目标确定侧重。仅在明确要求 PD 分离时，按两阶段各自的资源与负载分别给出配置。

> **模块级拆解优先**：parallel_config 不是按模型名或参考配置继承，而是由模块组合得到。给出候选方案前，必须分别分析 Attention/MLA、Dense MLP、MoE Expert、Embedding、LMHead，以及其他非标准大参数模块的并行方式。不同模块只有在结构、通信边界和收益判断一致时，才默认复用同一个 tp_size 或通信组。

> **非标准大参数模块**：例如额外 embedding / 子表、多模态编码器等，需要单独判断是否 TP、复制或复用通信组。不要默认跟随 `attn_tp_size`、`embed_tp_size` 或参考模型配置。

模块级拆解至少回答：
- 该模块是否在 Decode / Prefill 热路径上
- 参数量和单卡复制显存是否可接受
- 更适合 TP、DP 复制、EP，还是保持本地计算
- 与前后模块的 TP 度不一致时是否存在明显通信代价

### 1.3 部署信息确认

**自动获取**：
- 单卡显存 M：通过 `npu-smi info` 或 `torch.npu.get_device_properties` 确认

从任务上下文、已有配置或分析记录获取：

| 参数 | 说明 |
|------|------|
| 部署卡数 W | 部署总 NPU 数及节点拓扑；PD 分离时分别记录 |
| 目标负载与指标 | 目标 batch size，以及任务关注的吞吐、首 token 时延（TTFT）或 Decode 单步时延 |
| 最大序列长度 | 评估长序列计算与显存需求，判断是否需要 CP |

> **估算基准**：下述公式默认按 BF16 估算权重显存。通信按实际节点拓扑分析；PD 分离时分别估算两阶段的资源需求，并评估 KV 传输开销。

> 沿用已有分析中仍适用的结论；卡数、序列长度、batch_size 或场景变化时，更新受影响的估算与候选。

---

## 第二步：定性分类

根据模型参数和部署场景，用决策树确定候选并行模式。选择依赖经验的 TP/EP 取值或 CP 方案时，优先查询 cannbot-knowledge 中与目标相同推理部署框架的部署样例，参考架构、硬件和负载相近的并行配置、通信调优及显存优化经验，作为候选依据。

> 若目标模型是 Lite、裁剪版、客户改造版或实现分支不同，必须以当前模型的模块结构为准重新拆解。参考模型与目标模型存在模块差异时，优先相信模块级分析。

> 决策树表达分析思路，所有判断均为基线候选而非硬约束，需结合定量估算验证。硬约束、强经验、实现特性的分类见第四步"约束检查"。

### 统一决策树

决策树分三层：架构主干（由模型结构决定）→ 场景调整（由目标场景和 batch size 决定）→ 序列长度附加（按需叠加 CP）。三层共同确定候选 parallel_config。

#### 第一层：架构主干

确定非 Attention 模块的基础并行方式，以及 Attention 的默认起点。第二层再根据场景调整 Attention TP 度。

```
架构类型？
├─ Dense Transformer
│  ├─ 单卡放得下？→ 基线候选：DP
│  └─ 放不下？→ 基线候选：纯 TP，节点内 TP 起步
│
└─ MoE Transformer
   │
   ├─ MoE 层
   │  ├─ EP 和 TP 均需对比（moe_tp=1 时，EP 度 = 卡数）
   │  ├─ expert_ffn 较小时，TP 容易产生碎矩阵，优先评估保留完整专家矩阵的 EP；收益还受每专家 token 数影响
   │  └─ EP 需关注专家分布均衡和通信开销
   │
   ├─ Dense FFN
   │  ├─ 有独立 Dense 层 → 优先考虑节点内 TP 分摊权重与计算，再核对通信开销
   │  └─ 全 MoE 无独立 Dense → dense_tp=1
   │
   ├─ O_proj（MLA 模型）→ 独立评估 oproj_tp
   │  不默认跟随 attn_tp，需结合计算收益和边界通信决定
   │
   ├─ Embed/LMHead
   │  ├─ V×H 权重复制显存或 LMHead 计算开销较大 → 优先评估独立 TP
   │  └─ 否则 → 对比本地复制与复用已有 TP 组，计入边界通信代价
   │
   └─ 单机部署规模（卡数 W 由 1.3 确认）
      ├─ 单卡 → 无需并行
      ├─ 少卡（W < 节点满配）→ 优先以纯 TP 作为基线候选
      └─ 满卡（W = 节点满配）→ 对比纯 TP 与模块级差异化并行（如 MoE EP + Dense/LMHead TP），关注 TP 碎矩阵退化
```

#### 第二层：场景调整

根据目标场景和 batch size 调整 Attention TP 度。

```
目标场景？
├─ 高吞吐（大 batch）
│  ├─ MLA → 优先以 attn_tp=1 作为高吞吐基线，最大化 DP；核对权重复制显存与单卡计算负载
│  └─ GQA → Decode 默认以 attn_tp=1 起步，最大化 DP；
│     Prefill 需结合 batch 和 seq_len 判断，若单卡 attention 计算成为瓶颈则保留提升 attn_tp 的候选
│
├─ 低时延（小 batch）
│  ├─ MLA → 保留小度数 attn_tp（如 2/4）作为候选，分摊 Q/O 或 Prefill 计算；
│     不因 MLA KV 压缩直接排除 attention TP
│  └─ GQA → 比较提升 attn_tp 的计算分摊收益与通信代价，评估 TTFT
│
└─ 均衡
   → 在高吞吐和低时延之间折中

背景：MLA 压缩 KV 不按 head 存，KV 侧通常不从 head TP 获得直接收益，但 Q/O 投影、Prefill 计算和低时延场景仍可能受益。
```

#### 第三层：序列长度附加

CP 作为附加机制，默认在不推翻前两层主干的前提下叠加。若引入后导致 Attention 或 O_proj 的并行约束改变，需回到第二层重新校正。

```
目标序列长度下，Prefill 计算或显存是否成为瓶颈？
├─ 否 → 无需因序列长度单独增加 CP
└─ 是 → 对比提升 Attention TP 与引入 CP
   ├─ 标准 GQA → 优先评估 Attention TP，受 head 切分约束或收益不足时考虑 CP
   └─ MLA → 优先评估 CP，同时保留 Q/O 投影 TP 候选；核对当前实现的序列切分支持与通信代价
```

> **模块间切换代价（经验判断）**：不同 TP 度边界需 AllGather/ReduceScatter 重排，通常远小于统一 TP 度带来的浪费，但切换频繁时需关注。

---

## 第三步：定量估算

根据候选模式，估算显存与通信开销，筛选 parallel_config 具体值。性能排序结合相近条件的实测经验或 profiling；仅有估算时，保留为待验证候选。

### 3.0 查阅参考配置

对采用的参考配置，读取对应实现的实际配置和设计文档，核对推理部署框架及版本、模型结构、硬件、精度、负载及部署模式。保留适用于当前模型的候选组合，不直接继承整套方案。

### 3.1 参数量计算

**标准 GQA Attention**：
```
Attention/层 = H × D_h × (2 × N_h + 2 × N_kv)    # Q, K, V, O_proj
```

**MLA Attention**：按实际 Q/KV/O 投影矩阵逐项核算，区分低秩、RoPE 与 V 的维度。

**FFN / MoE / Embed**：
```
FFN/层       = 3 × H × H_ffn                        # gate + up + down
MoE/层       = E × 3 × H × H_ffn                    # 所有专家
Embed        = V × H
LMHead       = V × H

P_total = (L - L_moe) × (P_attn + P_ffn)
        + L_moe × (P_attn + P_moe)
        + P_embed + P_lmhead
```

### 3.2 单卡显存估算

> 以下公式为 rough sizing，用于判断候选策略是否显存可行。精确值需要实际运行或参考已验证配置的 profiling 数据。

按目标框架的并行布局确定各模块 TP/EP 度与当前阶段实际 DP 度；recipes 见「recipes 部署约束」。

**参数权重显存**（下式各 `P_*` 为对应模块在全模型中的参数总量，按均匀切分估算）：
```
P_per_card = P_attn / attn_tp_size
           + P_ffn / dense_tp_size          # 非 MoE 层
           + P_moe / (moe_ep_size * moe_tp_size)  # 路由专家同时按 EP 和 TP 切分
           + P_embed / embed_tp_size
           + P_lmhead / lmhead_tp_size

参数显存 = P_per_card × bytes_per_param
```
bytes_per_param：BF16 = 2；其他精度按实际权重格式取值。独立切分的 O_proj、共享专家等按各自并行度单列；非均匀切分或 padding 按实际分片核算。

**KV Cache 显存**（每卡 B_local = B / 当前阶段实际 DP 度；缓存分片或复制按实现核对）：
```
标准 GQA：B_local × S × L × 2 × N_kv_local × D_h × bytes_per_kv
MLA absorb：B_local × S × L × (kv_lora_rank + qk_rope_head_dim) × bytes_per_kv
```
> `N_kv_local` 为每卡实际缓存的 KV 头数，按目标实现的 head 切分或复制方式确定。MLA absorb 同时缓存 nope（`kv_lora_rank`）和 rope（`qk_rope_head_dim`）两部分，仅按前者估算会漏掉 rope 缓存。
> 注意：实际 KV Cache 还受 page/block 分配粒度、prefix cache、chunked prefill 等影响

**其他显存开销**（公式无法精确估算，需关注）：
- 量化附加状态（使用量化权重时需计入）
- 通信 buffer（AllReduce / AllToAll 临时缓冲）
- graph 模式 workspace
- 激活临时 buffer（Decode 通常较小；Prefill 与 S、chunk 大小、attention backend 强相关）

**显存余量参考**：参数显存 + KV Cache 超过卡容量的 0.8 时，重点核算通信、workspace 和激活开销；总显存以不超过容量的 0.95 作为经验目标，结合实测峰值和运行余量判断可行性。

### 3.3 通信分析

结合实际拓扑、消息规模、调用频率和计算重叠分析通信代价。

**拓扑**
- 按通信组的 rank 分布识别节点内与跨节点链路，结合目标机器的带宽和延迟估算。

**通信原语类型**

| 原语 | 触发场景 | 特点 |
|-----|---------|------|
| AllReduce | TP 的 RowParallel 输出 | 均匀 ring 算法的每 rank 发送量约为 tensor_size × 2(tp-1)/tp |
| AllToAll | EP 的 token dispatch/combine | 数据量取决于激活专家数和 token 分布 |
| AllGather / ReduceScatter | 模块间布局转换或序列并行 | 按实际分片大小和组规模估算 |
| Send/Recv | CP 的 KV 分片交换、AFD 通信 | 点对点，可与计算重叠 |

**发生频率和热路径**
- 不只看通信字节量，还要看 collective 发生次数。每层高频小通信可能在 Decode 中累积成主要延迟。
- 区分 batch=1 Decode、小 batch Decode 和大 S Prefill，不要用单一通信直觉外推所有阶段。
- 节点内小消息 AllToAll 未必比高频 AllReduce 更差；不能只根据原语名称判断优劣。
- 模块 TP 度不一致时，检查边界通信是否抵消模块 TP 收益。

**计算重叠**
- 按数据依赖区分可与计算重叠的通信，以及消费端必须等待的通信。

### 3.4 根据估算结果调整策略

估算完成后，按以下判据检查和调整：

**显存可行性**：
- 总显存 > 卡容量 × 0.95 → 显存余量偏紧，优先评估增大并行度或减小 batch
- KV Cache 占比 > 50% 可用显存 → 优先评估 CP / KVCache Offload；MoE 激活峰值较大时考虑分块计算

**通信可行性**：
- 跨节点 TP 时，对比计算分摊收益与新增通信代价。

**Batch 吞吐**：
- B_local = B / dp_size 太小（如 < 16）→ 单卡计算效率可能受限
- 固定全局 B 时，增大 DP 会进一步减小 B_local；可在显存和时延允许时增大 B，或对比更低 DP、更高 TP 的计算与通信开销

**Prefill TTFT**：
- 单卡 Prefill 计算成为瓶颈 → 对比 Attention TP 与 CP 的分摊收益及通信开销
- MLA 长序列优先评估 CP，标准 GQA 优先评估 TP；按当前实现与目标负载复核收益

> 如果估算显示已验证配置的参考值合理（显存可行 + 通信在节点内），可将其保留为高优先级候选，但仍需说明它与目标模型模块拆解结果是否一致。

---

## 第四步：方案审查

输出前检查以下约束：

### 约束检查

**[A] 硬约束**（不满足则配置非法）：
- [ ] 各模块 `world_size % *_tp_size == 0`
- [ ] `num_attention_heads % attn_tp_size == 0`
- [ ] `num_key_value_heads % attn_tp_size == 0`（按 KV head 切分的实现）
- [ ] `num_experts % ep_size == 0`（要求均匀专家划分的实现）

**[B] 强经验检查**（通常应满足，不满足需有充分理由）：
- [ ] 单卡显存估算 ≤ 实际显存 × 0.95
- [ ] Decode 阶段 attn_tp 不大于必要值（避免 S=1 时的无效通信）

**[C] 目标框架可行性**：
- [ ] 估算采用的 DP/EP 度及 CP 请求分配方式与目标实现一致
- [ ] 已核对目标框架的部署约束与能力范围（recipes 见「recipes 部署约束」）
- [ ] `N_h * v_head_dim % oproj_tp_size == 0`（MLA o_proj 切分约束）

现有框架支持的候选可作为部署推荐；需要扩展框架能力的候选单独列出适配前提和工作量。数学或模型结构上不可行的方案排除。

### 输出格式

各模块的 TP 度是独立决策，候选方案应充分组合（如 attn_tp=8 + moe_tp=1 + dense_tp=8），不要只考虑"全 TP"或"全 EP"两个极端。按定量估算给出推荐方案及有实质差异的备选并排序，说明对高吞吐/低时延/长序列场景的取舍，以及推导出的 DP/EP 度与实施适配点。

下例沿用 recipes 的配置字段；配置生成与代码适配交由对应部署方式的实施流程处理：

```yaml
# 候选 A（推荐）：{模式名称}
# 理由：{为什么推荐}
# 参考：{最接近的已验证配置；无合适参考时注明}
candidate_a:
  parallel_config:
    attn_tp_size: {value}       # attn_dp_size = world_size // attn_tp_size
    dense_tp_size: {value}
    moe_tp_size: {value}        # moe_ep_size = world_size // moe_tp_size
    embed_tp_size: {value}
    lmhead_tp_size: {value}
    o_proj_tp_size: {value}     # MLA 模型需要，非 MLA 可省略
    cp_size: {value}            # 长序列；需单独评估实施适配
  estimation:
    params_per_card_gb: {value}
    kv_cache_per_card_gb: {value}
    memory_feasible: true/false
    cross_node_comm: true/false
  tradeoff:
    throughput: {优势/中性/风险}
    latency: {优势/中性/风险}
    long_context: {优势/中性/风险}

# 候选 B（备选）：{模式名称}
# 理由：{适用场景 / 与 A 的权衡}
candidate_b:
  parallel_config: ...
```

**排序维度**（按优先级）：
1. 显存可行性（不可行直接排除）
2. 目标负载下的预期性能（计入跨节点通信与计算重叠的影响）
3. 与已验证配置的距离（仅作为性能差异不明显时的 tie-breaker）

> 当定量分析明确显示某方案有结构性性能优势（如 EP 的完整矩阵 vs TP 的碎矩阵），即使该方案未经验证，也应标注为"推荐对比测试"，不因已验证距离自动降级。

并行度变更可能影响预切权重的兼容性，由对应实施流程检查并处理（recipes 使用 model-infer-parallel-impl）。

---

## （可选）第五步：Profiling 数据校准

当用户或外部 profiling skill 提供 profiling 数据时，校准第三步的估算。本 skill 不负责采集 profiling，也不要求每次分析都执行 profiling。

### 三个关键指标

**指标 1：通信对单步延迟的影响**

在同一 rank、同一步的 trace 中检查通信与计算的重叠及消费端等待。`Hccl*` 耗时之和包含被计算隐藏的部分，多个通信重叠时还会重复计时。

> 查看位置：TensorBoard Trace View，筛选 `HcclAllReduce`、`HcclAllToAll` 等算子

**指标 2：显存峰值**

通过 `torch.npu.max_memory_allocated()` 获取各 Rank 峰值：
- 验证第三步的估算偏差
- 确定 batch_size 可以开多大

> 若外部 profiling 数据未包含显存信息，可跳过该项，或由 profiling 步骤单独补充采集。

**指标 3：Rank 间负载差异**

同步等待可能掩盖计算量差异；分开对比各 rank 同一步的计算与等待时间，MoE EP 再结合各专家 token 数定位负载偏斜。

> 查看位置：各 Rank 独立 trace 文件。

按实测的 batch 和序列长度复算。显存偏差先核对峰值 rank 的权重和 KV 实际分片；通信耗时偏高时，从 trace 区分与计算重叠的通信和阻塞下一步计算的等待。仍难解释时，可查询 cannbot-knowledge 中同框架、相近拓扑和负载的显存或通信调优案例，再用当前实测判断是否调整候选方案。
