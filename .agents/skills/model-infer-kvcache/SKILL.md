---
name: model-infer-kvcache
description: 基于 PyTorch 框架的昇腾 NPU 模型推理 KVCache 优化技能。分析并改造 LLM 推理模型的 KVCache 实现，覆盖 Legacy 连续缓存与分页注意力（Paged Attention）配 FA 融合算子、MLA 压缩缓存、SlidingWindow / 多 attn_type 混合。触发场景：KVCache 管理实现、分页注意力接入、KV 压缩、FA 融合算子、OOM / 性能问题、block_table / slot_mapping 构造。支持框架部署与独立部署两种模式。
---

# KVCache 优化技能

按模型 attention 形态选择 KVCache 与 FA 路径。需要选型时使用第一层，按需查第二层的数据布局与算子约束。已有确认方案时，从第三层对应的部署方式实施；若目标框架或算子无法满足方案，说明具体冲突并回到选型。

---

## 部署模式判定

沿用任务中明确的部署模式。已注册到 `executor/core/support_models.py`、接入 infer 仓 `executor/core/` 的模型走框架部署；自管 Runner 和推理循环且未接入该框架的模型走独立部署。新模型或仍无法判定时优先框架部署。

---

## 第一层：快速选型

部署模式决定由谁管理缓存，两种部署方式都可使用 Legacy 或 Paged。先核对模型当前的缓存路径，再按下表选型。优先采用 Paged，受架构或算子限制时保留 Legacy 并说明原因；混合注意力模型按层或 cache 分别判断。

### 1.1 KV 模式选型

> **改造目标：Paged 模式（FA + TND layout）**
> - **Paged**：infer 仓框架用 `BlockPool` 动态分配 block，支撑 batch 内序列长度差异（设计文档 §1 / §2 的核心机制）
> - **FA**：NPU 融合算子（`npu_fused_infer_attention_score{,_v2}`），用于加速 attention 计算
> - **TND layout**：变长 batch 拼一维，与 `executor/core/` ExecutionEngine 的 `input_ids: [TotalTokens]` packed 输入协议契合；MLA 用 `TND_NTD`

| 模型类型 | KV 模式 | 框架部署接入 | 独立部署接入 |
|---|---|---|---|
| 标准 LLM（MHA/GQA，含 MoE 变体） | **Paged 模式（FA + TND）** | `attn_type="FullAttention"` | 模式二（手动 Paged + FA） |
| 含滑窗 attention | **Paged 模式（FA + TND + 滑窗约束）** | `attn_type="SlidingWindow"`（含 `sliding_window`） | 模式二（手动 Paged + sparse_mode=4） |
| MLA 架构 | **Paged 模式（FA + TND_NTD + MLA absorb）** | `attn_type="FullAttention"`（nope + rope 两 entry，模型类设 `is_mla_backend=True`） | 模式三（MLA absorb） |
| 已有 Legacy 基线 | Legacy + SDPA + BSH | `cache_unit` 静态预分配 + `scatter_update_`（不进 `ATTN_TYPE_MANAGER_MAP`） | 模式一（list of (k,v)） |

需要接入样例时，查询 cannbot-knowledge 中同推理部署框架、相近注意力形态的 KVCache 实现。

> infer 仓的 Legacy 基线用于 offline 推理。目标是 online / PD 服务时，按框架的 `KVCacheManager` 契约接入。

> **MLA absorb 路径补充**：
> - latent KV 共享，**不沿 `attn_tp_size` 切分**（每 attn_tp rank 持完整 latent KV；Q 按 head 切、attn_dp 按 batch 切）。
> - V 由 latent 经 `kv_b_proj` 重投影获得，自动消除 `qk_head_dim ≠ v_head_dim` 时 V 需 pad 到 qk_head_dim 的问题。

### 1.2 算子改造

| 改造项 | 触发条件 | 框架部署 | 独立部署 |
|---|---|---|---|
| SDPA → NPU FA 融合算子（FA v1 / v2） | 目标路径支持并采用 FA | 接入所选 FA 调用（选型见 §2.7） | modeling 内手动接入 FA 调用 |
| 标准 indexing → `npu_scatter_nd_update_` 或 `npu_kv_rmsnorm_rope_cache` | 写入分页缓存 | 标准/滑窗用 `npu_scatter_nd_update_`，MLA 用 `npu_kv_rmsnorm_rope_cache` 融合写入 | Runner 自管 cache + slot_mapping，modeling 内手动选写入算子 |
| MLA absorb（key=value=cache_nope，rope 分离传入） | 模型架构是 MLA | modeling forward 内实现，cache_entries 配 nope_cache + rope_cache | 同左，cache 由 Runner 自管 |
| 多 attn_type 混合处理 | 模型含混合层 | 各层 `__init__` 按条件设置 `attn_type`，按条目的分组键取用缓存索引 | 较少见，可手动模拟 |

### 1.3 选型输出

给出各类 cache 的模式与布局（含 MLA absorb 或混合层的选择）、Prefill / Decode 的 KV 读取路径、采用 FA 时的版本及适用条件。核对目标框架能否管理所需缓存、算子能否保持原模型的注意力语义。

估算目标并发与序列长度下的每卡缓存占用。对普通分页缓存，各请求按有效长度向上取整到整块，再按组汇总所需块数；同组各层、各 cache 条目的每块字节数相加。按实际每卡头数、维度和 dtype 计算，MLA 分别计入 nope 和 rope。上述块数是负载需求。估算实际分配显存时，infer 仓离线普通分页池按配置的最大长度、本地并发及空块预留；滑窗按窗口预留，Legacy 按预分配张量估算。

---

## 第二层：核心数据结构与算子

### 2.1–2.3 分页寻址与读写分工

`slot_mapping` 给出本步写入 K/V 的物理 slot，`block_table` 将请求的逻辑块映射到物理块，供 FA 读取分页缓存。infer 仓框架按当前 batch 构造二者，模型从 `forward_metadata` 用同一 `manager_key`（默认 `attn_type`）取用。动态分配的块可能不连续，不能按请求序号推算物理地址。

独立 Runner 的静态预分配与索引构造见 `references/standalone_kv_reference.md` §2。

### 2.4 actual_seq_lengths 构造

对于 packed TND 输入，Q 长度传各请求本次 query 的累计结束位置。KV 长度取决于 FA 实际读取的数据：

| KV 读取路径 | 长度含义 | 两个请求的示例 |
|---|---|---|
| Prefill 直读 packed K/V | 各段 KV 的累计结束位置 | 长度 512、256 → `[512, 768]` |
| 通过 block_table 读取分页缓存 | 每个请求的有效 KV 长度，包含本步已写入的 token | Prefill 为 `[512, 256]`，各生成一个 token 后为 `[513, 257]` |

框架的 `actual_seq_lengths_{q,kv}` 保存逐请求长度，`actual_seq_lengths_cu_{q,kv}` 保存累计长度。普通单 token Decode 的 packed Q 长度为 `[1, 2]`；BSH 按请求分别传 Q/KV 长度。

### 2.5 kv_len 生命周期

同一步各层用同一份长度写入各自缓存，由框架或 Runner 随请求推进；在 attention 层内递增长度会导致写入位置逐层偏移。

### 2.7 FA 算子配置

按目标路径所需的滑窗、sink、分离 RoPE 或量化 KV 能力选择 FA 版本。按目标版本、硬件和 layout 查询 cannbot-knowledge 中 FA v1/v2 的参数签名、mask 限制与缓存格式。

> **模型相关约束补充**：
> - **YaRN / `mscale_all_dim` 模型的 scale**：scale = `head_dim^-0.5 × mscale²`（mscale 按 YaRN 公式计算，`mscale_all_dim` 启用时生效）。漏写 mscale² 静默偏低不报错。
> - **MLA absorb 路径**：FA 调用必须 `num_key_value_heads = 1`（latent 单 head），与 `cache_entries.num_head = 1` 严格一致；不一致触发 GQA 比例错。

#### input_layout 选择

| layout | Q 格式 | KV 格式 | 适用场景 |
|--------|--------|---------|---------|
| **`"TND"`** ★ | [T, N, D] | [T, N, D] | **Paged 模式标准 LLM 默认推荐**，变长 batch 拼一维 |
| **`"TND_NTD"`** ★ | [T, N, D] | [N, T, D] | **Paged 模式 MLA 默认推荐**，NZ 格式缓存 |
| `"BSH"` | [B, S, N*D] | [B, S, N*D] | Legacy 模式（框架 `cache_unit` / 独立模式一） |

> ★ TND / TND_NTD 是变长 batch packed `[TotalTokens, ...]` 形态；框架部署 Paged 模式下与 `executor/core/` ExecutionEngine 的 `input_ids: [TotalTokens]` packed 输入契合，独立部署模式二同样推荐。

#### Paged 模式 KV cache 实际 shape

`input_layout` 字符串与 paged KV 的实际 tensor shape 是两个概念，传 FA 前不要按字面 `[T,N,D]` 套到 cache 上：

| Attention | cache_entries 分配 | 传给 FA 的 KV shape |
|---|---|---|
| GQA / MHA | `[bn, bs, num_kv_heads_per_rank, head_dim]` | `view(*shape[:2], -1)` → `[bn, bs, H]` 3D |
| MLA（nope_cache / rope_cache） | 逻辑 `[bn, bs, 1, D]`；写入算子 `cache_mode="PA_NZ"` 实际按 NZ 排布 | view 为 5D NZ `[bn, 1, D/NZ_DIM, bs, NZ_DIM]`，分块宽度与写入布局一致 |

Q 在两种模式下均为 packed `[T, N, D]` 3D。`input_layout="TND"` / `"TND_NTD"` 描述的是 FA 算子对各张量的解释方式，paged KV 的实际形态如上表。采用 PA_NZ 缓存的 MLA 路径下，`npu_kv_rmsnorm_rope_cache` / `npu_mla_prolog_v3` 写入必须配 `cache_mode="PA_NZ"`，FA 调用前 view 到 5D NZ；参考 `models/deepseek_r1/models/modeling_deepseek.py` 搜 `KV_CACHE_NZ_DIM`。

#### sparse_mode 与 atten_mask

- 因果 Prefill 和多 token Decode（如 MTP）要屏蔽未来 token。infer 仓的 causal 路径可复用 `get_init_attn_mask`，其中 True 表示遮蔽。
- 普通全注意力单 token Decode 在有效 KV 范围内可用 `sparse_mode=0`、`atten_mask=None`；也可沿用受支持的 causal 路径。参数按模型原有注意力语义配套选择。
- 滑窗层在 Prefill 和 Decode 均需限制可见历史。使用 band 路径时配套设置 `sparse_mode=4`、窗口跨度与 `next_tokens=0`；若目标算子路径不执行窗口限制，应保证实际读取的 KV 对应正确窗口。用 `KV_len > sliding_window` 的输入验证，短序列无法区分滑窗和全注意力。

直读 K/V、分页读和 MLA absorb 的调用示例见 `references/fa-code-examples.md`。KV 量化的 cache dtype 与 FA 反量化参数由 model-infer-quantization 接入。

---

## 第三层：实施流程

### 3.1 框架部署模式实施

保留 Legacy 时，沿用连续缓存并对比 Prefill / Decode 基线。选用 Paged 时，按以下步骤接入 infer 仓框架。完整骨架见 `references/framework_kv_reference.md`，接口约定见 `docs/design/kv_cache_design.md §5`。

> **packed sequence 协议（modeling forward 必须遵守）**：框架部署模式下 modeling 接收 packed 一维输入：
> - `input_ids: [TotalTokens]`（Prefill 各 prompt 串接、普通 Decode 每请求 1 token；变长 batch 不做 padding）
> - `position_ids: [TotalTokens]`（**packed 1D**，模型内 RoPE 直接用此索引取 cos/sin）
> - **hidden_states 全程保持 `[TotalTokens, hidden_size]` 二维**（不要 reshape 成 `[B, S, H]`——变长 batch 不能简单 reshape；用 `actual_seq_lengths_cu_q` 在 FA 内表达 batch 边界）
> - **Prefill / Decode 分支判断统一用 `forward_metadata.is_prefill`**
> - Prefill 按 `actual_seq_lengths_cu_q - 1` 取每个请求的末位 hidden state，再计算 logits，输出 `[batch_size, 1, vocab_size]`

#### 步骤 1：将已选缓存类型映射到 manager

普通全注意力用 `FullAttentionManager` 按序列增长追加块，滑窗用 `SlidingWindowManager` 按窗口预留块。混合模型按已选分组接入；若现有 manager 都无法表达目标缓存行为，见步骤 4。

#### 步骤 2：attention 类内定义 cache_entries

每个 attention 类的 `__init__` 内：

1. 定义 cache 占位 tensor：`self.k_cache = torch.Tensor([])`（框架在初始化时通过 `tensor_setter` 注入实际 storage）
2. 构造 `self.cache_entries` 列表，每个 cache 一个 `CacheEntry`

完整骨架（含 GQA / SlidingWindow / MLA 三种）见 `references/framework_kv_reference.md` §2。

> `num_head` 按每卡实际 KV 布局填写。`QKVParallelLinear` 的标准头切分/复制路径使用 `max(num_kv_heads // attn_tp_size, 1)`；MLA absorb 的 nope / rope 缓存各为一个 latent head。改变 Attention TP 时同步核对。

> 模型自有 cache（非 KVCache 状态，如 N-gram window / hash buffer / speculative buffer 等）由 model class 内部自管，不上报 `cache_entries`，与框架 KV 解耦。

#### 步骤 3：模型类实现 get_cache_info

`ModelWorker` 通过 `get_cache_info()` 获取各层的缓存声明。每个参与分页的 `CacheEntry` 声明正数 `block_size`，`ModelCacheInfo` 按实际 attention 子层汇总条目；层容器为字典时遍历层实例，多 attention 子层按实际顺序展开。MLA latent KV 在 TP rank 间复制时设置 `is_mla_backend=True`，供 PD 传输选择对应 rank。

完整示例见 `references/framework_kv_reference.md` §3。

#### 步骤 4：（可选）新增 SingleTypeKVCacheManager 子类

仅当现有 `ATTN_TYPE_MANAGER_MAP` 中的 manager 无法满足目标缓存行为时执行：

1. 在 `executor/core/kv_cache/single_type_kv_cache_manager.py` 新增子类
2. 注册到 `ATTN_TYPE_MANAGER_MAP`
3. 如新 attn_type 是「固定 block 数」类型（如类似 SlidingWindow），扩展 `calculate_block_num()` 加入 `FIXED_BLOCK_ATTN_TYPES` 列表，并实现 `calculate_fixed_block_memory_bytes()` 分支

详见 `docs/design/kv_cache_design.md §5.3 / §5.4` 和 `references/framework_kv_reference.md` §5。

#### 步骤 5：attention forward 接入 block_table / slot_mapping

模型顶层从 `forward_metadata` 透传索引，各 attention 层按缓存分组键取用。直读 packed K/V 的 Prefill 可先算 attention 再写缓存；从分页缓存读取本步 K/V 时，须先按 slot 写入再用 block table 读取。完整 forward 和写入模板见 `references/framework_kv_reference.md` §4。

#### 步骤 6：验证三链路

按 `docs/design/kv_cache_design.md §5.6`：

1. **初始化链路**：`get_cache_info()` 返回的 `ModelCacheInfo` 字段正确，`cache_utils.allocate_cache_tensors()` 后每层 cache 已被注入 storage（不再是 `torch.Tensor([])` 占位）
2. **运行期链路**：分页读取时使用当前 cache 分组的 block table；FA 调用前后 dtype / shape 与 `cache_entries` 声明一致；普通单 token Decode 第一步的 `actual_seq_lengths_kv` 等于 `prompt_length + 1`。用不等长请求对比改造前后的 Prefill 和 Decode 输出。
3. **释放链路**：请求完成后 `KVCacheManager` 释放 block，下一请求复用 block 不出现脏数据；长跑无显存泄漏

### 3.2 独立部署模式实施

Runner 内自管 KV 数据结构、`block_table` 静态预分配、`slot_mapping` 动态计算、`forward_metadata` 构造。完整代码骨架见 `references/standalone_kv_reference.md`。

#### 步骤 1：按已选方案使用 Runner 模板

已选 KV 路径对应以下 Runner 骨架：
- 模式一（连续缓存）：migrator 阶段最简骨架，无 paging，FA 直接读取整个 cache
- 模式二（手动 Paged 模式 + FA）：性能主路径，自管 `block_table` + `slot_mapping`
- 模式三（MLA absorb 压缩缓存）：MLA 模型，叠加在模式一或模式二之上，缓存 `cache_nope` + `cache_rope`

#### 步骤 2：Runner 内自管 KV tensor

在 `_init_kvcache(...)` 内一次性分配所有层的 KV cache：
- 模式一：每层 `(k, v)` BSH layout，shape `[batch, max_seq_len, num_kv_heads * head_dim]`
- 模式二：每层 `(k, v)` 物理 cache，shape `[total_blocks, block_size, num_kv_heads, head_dim]`，并构造 `self.block_table = arange().reshape(batch, -1).int().npu()`
- 模式三：每层 `(cache_nope, cache_rope)`，dim 远小于完整 KV

完整骨架见 `references/standalone_kv_reference.md` §1 / §2 / §3。

#### 步骤 3：Runner 内构造 slot_mapping（模式二/三）+ forward_metadata

每步推理前 Runner 调 `_build_slot_mapping_prefill / decode` 构造 `slot_mapping`，并组装 `ForwardMetaData`（含 `is_prefill` / `kv_len` / `slot_mapping` / `block_table` / `actual_seq_lengths_kv` / `actual_seq_lengths_q` / `attention_mask`）。

`ForwardMetaData` 直接 import 仓内 `executor.utils.forward_metadata.ForwardMetaData`，或参考字段在 runner 内 inline 一个轻量 dataclass。

完整骨架见 `references/standalone_kv_reference.md` §4。

#### 步骤 4：modeling 内 scatter_update_ + FA 调用

modeling forward 接收 `forward_metadata`（或自定义 `past_key_values + kv_len + attention_mask`），按 `is_prefill` 分支：
- 写入：`scatter_update_` 或 `npu_kv_rmsnorm_rope_cache`
- 读取：FA 算子调用（详见第二层 §2.7）

完整骨架见 `references/standalone_kv_reference.md` 各章节代码块 + `references/fa-code-examples.md`。

#### 步骤 5：验证

1. Prefill 输出与基线对比（migrator 阶段已建立的基线）
2. Decode 单步耗时合理（与基线对比无显著回归）
3. 多卡场景：各 rank 输出形状一致，KV 切分无错位
4. 长跑无显存泄漏
