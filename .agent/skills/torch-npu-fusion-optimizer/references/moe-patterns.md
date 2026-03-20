# MoE 算子模式详解

本文档详细描述 MoE（Mixture of Experts）场景下的完整算子链和不同并行策略的变体。

---

## 目录

- [MoE 概览](#moe-概览)
- [TP 模式 MoE 全流程](#tp-模式-moe-全流程)
- [EP+TP 混合模式（MC2）](#eptp-混合模式mc2)
- [共享专家处理](#共享专家处理)
- [MoE 门控策略](#moe-门控策略)
- [量化模式变体](#量化模式变体)
- [EP 负载均衡](#ep-负载均衡)

---

## MoE 概览

### MoE 前向计算流程

```
输入 hidden_states
    ↓
[Gate] 门控打分 → Top-K 专家选择
    ↓
[Init Routing] 按专家展开 token → 可选量化
    ↓
[AllToAll] EP 场景下跨卡分发 token（可选）
    ↓
[Re-Routing] EP 场景下按本地专家重排（可选）
    ↓
[Expert Compute] 专家 FFN 计算（grouped_matmul + activation）
    ↓
[AllToAll] EP 场景下跨卡回收 token（可选）
    ↓
[Finalize Routing] 按原始顺序恢复 + 加权聚合
    ↓
输出 hidden_states（+ 共享专家输出）
```

### 并行策略与算子选择

| 并行策略 | 适用阶段 | 核心算子 |
|---------|---------|---------|
| 纯 TP | Prefill | `init_routing_v2` → `grouped_matmul` → `finalize_routing` |
| EP + TP | Decode | `distribute_dispatch_v2` → `grouped_matmul` → `distribute_combine_v2` |
| EP（含 re_routing） | Prefill + EP | `init_routing_v2` → AllToAll → `re_routing` → `grouped_matmul` → AllToAll → `finalize_routing` |

---

## TP 模式 MoE 全流程

### 适用场景

- Prefill 阶段，纯张量并行
- 所有专家在每张卡上都有副本（或按 TP 切分）

### 完整代码流程

```python
# 来源：models/deepseek-v3.2-exp/models/modeling_deepseek.py

# ===== Step 1: 门控 =====
logits = F.linear(hidden_states.view(-1, h), self.gate.weight)
topk_weight, topk_idx, _ = torch_npu.npu_moe_gating_top_k(
    logits, k=self.top_k,
    bias=self.gate.e_score_correction_bias.float(),
    k_group=self.topk_group, group_count=self.n_group,
    group_select_mode=1, renorm=0, norm_type=1,
    routed_scaling_factor=self.routed_scaling_factor,
    eps=float(1e-20)
)
topk_idx = topk_idx.to(torch.int32)

# ===== Step 2: 路由初始化 =====
expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = \
    torch_npu.npu_moe_init_routing_v2(
        hidden_states.view(-1, h),
        expert_idx=topk_idx,
        active_num=topk_idx.shape[0] * topk_idx.shape[1],
        scale=smooth_scale_1 if "a8" in quant_mode else None,
        expert_num=num_experts,
        expert_tokens_num_type=1,  # 1=count 模式
        expert_tokens_num_flag=True,
        active_expert_range=[0, num_experts],
        quant_mode=1 if "a8" in quant_mode else -1
    )

# ===== Step 3: 专家计算 =====
# 内部调用 npu_grouped_matmul + activation
expert_output = experts(expanded_x, tokens_per_expert, pertoken_scale=pertoken_scale)

# ===== Step 4: 路由结果聚合 =====
hidden_states = torch_npu.npu_moe_finalize_routing(
    expert_output,
    skip1=shared_expert_output,  # 共享专家输出作为残差
    skip2=None, bias=None,
    scales=topk_weight.to(expert_output.dtype),
    expanded_src_to_dst_row=expanded_row_idx,
    export_for_source_row=None,
    drop_pad_mode=2
)
```

### 关键参数说明

**`expert_tokens_num_type`**：
- `0`：cumsum 模式 — tokens_per_expert 是累积和
- `1`：count 模式 — tokens_per_expert 是每个专家的 token 数量

**`quant_mode`**：
- `-1`：无量化 → expanded_x 保持 BF16
- `1`：动态量化 → expanded_x 为 INT8，返回 pertoken_scale

**`drop_pad_mode`**：
- `2`：标准模式，按 expanded_src_to_dst_row 映射恢复

---

## EP+TP 混合模式（MC2）

### 适用场景

- Decode 阶段，专家并行 + 张量并行混合
- 通信与计算融合，减少通信开销

### 完整代码流程

```python
# 来源：models/deepseek-v3.2-exp/models/modeling_deepseek.py

# ===== Step 1: MC2 分发 =====
# 融合 AllToAll 通信和 token 按专家分发
output = torch_npu.npu_moe_distribute_dispatch_v2(
    x=hidden_states.view(-1, h),
    expert_ids=topk_ids,
    **dispatch_kwargs  # 包含 group、moePara 等配置
)
expand_x, dynamic_scale, expand_idx, expert_token_num, ep_recv, tp_recv = output[:6]

# ===== Step 2: 专家计算 =====
gmm_args = {
    "x": expand_x,
    "expert_tokens": expert_token_num,
    "group_list_type": 1,
}
if "a8" in quant_mode:
    gmm_args["pertoken_scale"] = dynamic_scale
expert_output = experts(**gmm_args)

# ===== Step 3: MC2 聚合 =====
# 融合专家输出聚合和 AllToAll 回收
hidden_states = torch_npu.npu_moe_distribute_combine_v2(
    expand_x=expert_output,
    shared_expert_x=shared_expert_output,  # 共享专家直接在 combine 中融合
    expert_ids=topk_ids,
    assist_info_for_combine=expand_idx,
    expert_scales=topk_weight.float(),
    ep_send_counts=ep_recv,
    tp_send_counts=tp_recv,
    **combine_kwargs
)
```

### dispatch_v2 与 init_routing 的区别

| 特性 | `init_routing_v2` | `distribute_dispatch_v2` |
|------|-------------------|--------------------------|
| 通信 | 不含通信 | 融合 AllToAll |
| 适用阶段 | Prefill（纯 TP） | Decode（EP+TP 混合） |
| 量化 | 通过 quant_mode 参数 | 自动处理 |
| 返回 | 展开 token + row_idx | 展开 token + expand_idx + 通信计数 |

### combine_v2 与 finalize_routing 的区别

| 特性 | `finalize_routing` | `distribute_combine_v2` |
|------|-------------------|------------------------|
| 通信 | 不含通信 | 融合 AllToAll |
| 共享专家 | 通过 skip1 参数加 | 通过 shared_expert_x 参数加 |
| 路由权重 | scales 参数 | expert_scales 参数 |
| 返回 | 聚合后的 hidden_states | 聚合后的 hidden_states |

---

## 共享专家处理

### 共享专家计算

共享专家（shared expert）独立于路由专家，所有 token 都经过共享专家：

```python
# 来源：models/deepseek-v3.2-exp/models/modeling_deepseek.py
def forward_shared_expert(self, hidden_states, is_prefill):
    if self.n_shared_experts > 0:
        hidden_states_share = self.shared_experts(hidden_states.view(-1, hidden_states.shape[-1]))
    else:
        hidden_states_share = None
    return hidden_states_share
```

### 共享专家输出融合

**TP 模式**：通过 `finalize_routing` 的 `skip1` 参数融合
```python
hidden_states = torch_npu.npu_moe_finalize_routing(
    expert_output, skip1=shared_expert_output, ...
)
# 等效于：output = route_aggregate(expert_output) + shared_expert_output
```

**EP+TP 模式**：通过 `combine_v2` 的 `shared_expert_x` 参数融合
```python
hidden_states = torch_npu.npu_moe_distribute_combine_v2(
    expand_x=expert_output, shared_expert_x=shared_expert_output, ...
)
```

---

## MoE 门控策略

### `npu_moe_gating_top_k` 支持的门控方式

#### noaux_tc（DeepSeek-V3 风格）

```python
topk_weight, topk_idx, _ = torch_npu.npu_moe_gating_top_k(
    logits, k=top_k,
    bias=e_score_correction_bias.float(),  # 专家负载平衡偏置
    k_group=topk_group,       # 组内选择数
    group_count=n_group,       # 专家分组数
    group_select_mode=1,       # 分组选择
    renorm=0,                  # 先 sigmoid 再 top-k
    norm_type=1,               # 使用 sigmoid
    routed_scaling_factor=routed_scaling_factor,
    eps=1e-20
)
```

#### 标准 softmax Top-K

当 `npu_moe_gating_top_k` 不适用时，使用 PyTorch 原生实现：

```python
scores = logits.softmax(dim=-1, dtype=torch.float32)
topk_weight, topk_idx = torch.topk(scores, k=top_k, dim=-1, sorted=False)
```

### `npu_moe_gating_top_k_softmax`

带 softmax 归一化的 Top-K 变体，适用于使用 softmax 打分的模型。

---

## 量化模式变体

### MoE 路由中的量化

`npu_moe_init_routing_v2` 可集成量化，避免额外的 `npu_dynamic_quant` 调用：

```python
# 无量化模式
expanded_x, row_idx, tokens_per_expert, _ = torch_npu.npu_moe_init_routing_v2(
    hidden_states, expert_idx=topk_idx,
    active_num=active_num, expert_num=num_experts,
    quant_mode=-1  # 无量化
)
# expanded_x 是 BF16

# W8A8 量化模式
expanded_x, row_idx, tokens_per_expert, pertoken_scale = torch_npu.npu_moe_init_routing_v2(
    hidden_states, expert_idx=topk_idx,
    active_num=active_num,
    scale=smooth_scale_1,  # SmoothQuant scale
    expert_num=num_experts,
    quant_mode=1  # 动态量化
)
# expanded_x 是 INT8，pertoken_scale 可直接传给 grouped_matmul
```

### 不同量化模式的 MoE FFN 算子链

**BF16**：
```
init_routing(quant_mode=-1) → grouped_matmul(BF16→BF16) → npu_swiglu → grouped_matmul(BF16→BF16)
```

**W8A8**：
```
init_routing(quant_mode=1) → grouped_matmul(INT8→INT32) → npu_dequant_swiglu_quant → grouped_matmul(INT8→BF16)
```

**W8A8C8**：
```
init_routing(quant_mode=1) → grouped_matmul(INT8→BF16, with scale) → npu_swiglu_clip_quant → grouped_matmul(INT8→BF16, with scale)
```

---

## EP 负载均衡

### Perfect EPLB

完美 EP 负载均衡（Perfect Expert Parallelism Load Balancing），确保每张卡处理相同数量的 token：

```yaml
# YAML 配置
model_config:
  perfect_eplb: True
```

### EP 场景下的二次路由

当使用 EP 时，AllToAll 通信后需要 `npu_moe_re_routing` 按本地专家重新排列：

```python
# 来源：models/deepseek-v3.2-exp/models/modeling_deepseek.py

# AllToAll 后的 gathered_tokens 按源卡分组，需重排为按本地专家分组
hidden_states, gathered_scale, gathered_ids_unsort, tokens_per_local_expert = \
    torch_npu.npu_moe_re_routing(
        gathered_tokens,
        tokens_per_expert_group.view(self.moe_ep_size, -1),
        per_token_scales=gathered_pertoken_scale
    )

# 专家计算（使用重排后的 tokens_per_local_expert）
expert_output = experts(hidden_states, tokens_per_local_expert, pertoken_scale=gathered_scale)

# 恢复 AllToAll 前的顺序
new_x = torch.index_select(expert_output, 0, gathered_ids_unsort.float().argsort().int())
```

### 完整 EP 流程

```
init_routing_v2
    ↓
AllToAll（分发 token 到对应 EP rank）
    ↓
npu_moe_re_routing（按本地专家重排）
    ↓
grouped_matmul + activation（本地专家计算）
    ↓
index_select（恢复 re_routing 前的顺序）
    ↓
AllToAll（回收 token）
    ↓
finalize_routing（按原始 token 顺序聚合）
```

---

## 多流并行中的 MoE

### Decode 阶段多流配置

MoE 计算和共享专家计算可以并行执行：

```python
# 来源：models/deepseek-v3.2-exp/models/modeling_deepseek.py
use_aclgraph_event = self.enable_multi_streams and self.enable_aclgraph
if use_aclgraph_event:
    tng.ops.npu_tagged_event_wait(moe_npu_events[1])
```

共享专家与路由专家在不同 stream 上并行计算，通过 event 同步。
