# DeepSeek V3.2 KVCache Offload 当前实现说明

## 1. 总体方案

当前 DeepSeek V3.2 KVCache offload 采用 **Hybrid + Memory Accounting** 方案：

- 长期 KV cache 交给新框架 `KVCacheManager` 管理。
- offload 专用临时 workspace 和算子执行时序仍由模型侧 `OffloadCache` 管理。
- 当前仅限制支持 offline，`disaggregation_mode != "NONE"` 时会直接报错。

这个方案不是旧代码的纯模型自管理，也不是完整框架化 offload。它的目标是先让长期 KV cache 接入新框架的 block 生命周期，同时尽量复用旧 offload 的 temp/selected cache 和 gather-selection 执行逻辑。

## 2. Cache 归属

### 2.1 框架管理的 cache

这些 cache 由 `KVCacheManager / CacheEntry` 创建和管理，参与 block table、slot mapping、block pool 和 request 生命周期：

| Cache | offload 下位置 | 说明 |
|---|---|---|
| `nope_cache` | swapped memory | MLA full KV 的 nope 部分，长期保存 decode 需要的 KV。 |
| `rope_cache` | swapped memory | MLA full KV 的 RoPE 部分，长期保存 decode 需要的 KV。 |
| `indexer_key_cache` | HBM | Lightning Indexer cache，不做 offload。 |
| `indexer_key_scale_cache` | HBM | int8 KV cache 场景下的 indexer scale cache，不做 offload。 |

`nope_cache / rope_cache` 在 `enable_offload=True` 时使用：

```python
CacheAllocator.SWAPPED_MEMORY
```

实际分配通过：

```python
torch_npu.empty_with_swapped_memory(...)
```

### 2.2 模型 OffloadCache 管理的 workspace

这些资源仍由模型侧 `OffloadCache` 管理，不参与框架 block pool：

| 资源 | 位置 | 用途 |
|---|---|---|
| `temp_kv_cache` | HBM | prefill 临时 KV；`npu_mla_prolog_v3` 写入，prefill attention 读取。 |
| `selected_key_values` | HBM | decode 时从 full KV gather 出 topK KV，供 SFA 读取。 |
| `selection_kv_block_table` | HBM | `npu_gather_selection_kv_cache` 的 selected cache block table。 |
| `selection_kv_block_status` | HBM | selected cache 状态，标记 block 是否已准备/复用。 |
| `default_topk_indices` | HBM | decode selected KV 后构造 sparse indices。 |
| `d2h_stream / d2h_event` | NPU runtime object | 控制 offload copy / sync 时序，保留旧模型侧逻辑。 |

这些 workspace 的 HBM 占用通过模型接口：

```python
get_offload_workspace_memory_info()
```

上报给框架，框架在计算 block 数前把它们计入 reserved HBM。swapped full KV 本身不计入 HBM 预算。

## 3. Prefill 流程

### 3.1 非 CP prefill

offload 开启后，prefill 不直接把 `npu_mla_prolog_v3` 的 KV 写入 swapped full KV，而是先写 HBM 上的 `temp_kv_cache`：

```text
hidden_states
  -> npu_mla_prolog_v3
  -> temp_kv_cache
```

随后：

- prefill attention 读取 `temp_kv_cache`；
- 需要持久化的 KV 再写入框架管理的 swapped `nope_cache / rope_cache`。

这样做保留了旧代码中 prefill 使用 HBM 临时 KV 的执行方式，避免 prefill FA 直接依赖 swapped full KV。

### 3.2 CP prefill

CP prefill 下，每个 rank 先计算本 rank local token 的 KV。随后：

```text
local latent KV
  -> cp_group all_gather
  -> restore 到 global padded token 顺序
```

restore 后会分成两套 cache 使用：

1. `full_*_cache`
   - 临时 full KV cache；
   - 只服务当前层 prefill FA / indexer；
   - 包含 CP padded 坐标，用于保证 sparse_mode=3 下 Q/KV 坐标对齐。

2. `decode_*_cache`
   - 后续 decode 需要持久化的 KV；
   - offload 下对应 `offload_cache.temp_kv_cache`，后续再写入 swapped full KV；
   - 写入哪些 token 由 `cp_metadata.persistent_valid_indices` 决定；
   - 写到哪些 PA slot 由 `cp_metadata.persistent_slot_mapping` 决定。

offline 和 online prefill 的持久化策略不同：

- offline：只持久化本 rank owner request 的真实 KV。
- online PREFILL：持久化当前 batch 的全部真实 KV，供 cp_rank0 做 PD 传输。

两种场景都不会持久化 CP pad token。

## 4. Decode 流程

decode 阶段长期 full KV 在框架管理的 swapped `nope_cache / rope_cache` 中。

为了让 SFA 在 HBM 上执行，decode 先调用：

```python
torch_npu.npu_gather_selection_kv_cache(...)
```

从 swapped full KV 中按 topK gather 出 selected KV，写入模型侧：

```text
selected_key_values
selection_kv_block_table
selection_kv_block_status
```

然后 SFA 读取 selected KV：

```text
swapped full KV
  -> GatherSelectionKvCache
  -> selected_key_values in HBM
  -> Sparse FA
```

`selection_kv_block_status` 是 stateful 的，当前仍由模型侧维护和 reset。prefill 后会调用 `reinit_status()` 清理 selected 状态，避免复用旧 selected block。

## 5. 与旧代码的区别

旧代码中，offload 相关 cache 基本都由模型侧管理：

- full KV cache 由模型侧 `OffloadCache` 分配；
- block / slot 生命周期也更偏模型内部逻辑；
- temp KV、selected KV、status、stream/event 都在模型侧。

当前新框架实现的主要变化是：

- full KV cache 已迁入 `KVCacheManager`；
- `nope_cache / rope_cache` 的物理 allocator 变成 `SWAPPED_MEMORY`；
- block table、slot mapping、request block 生命周期由框架统一管理；
- indexer cache 也进入框架 cache entry，但继续常驻 HBM；
- 模型侧 `OffloadCache` 不再管理 full KV，只保留 offload workspace 和执行时序；
- workspace 的 HBM 占用通过 memory accounting 显式反馈给框架。

因此当前实现比旧代码更贴近新框架，但还没有把 selected workspace 和 offload runtime 完全框架化。

## 6. 当前限制和后续演进

当前限制：

- 只支持 offline，online PD + offload 暂不支持。
- selected workspace 生命周期仍是模型私有逻辑。
- `selection_kv_block_status` 的 request slot reset 还没有纳入 scheduler 通用生命周期。
- swapped full KV 是否能直接参与 PD transfer 仍需单独验证。

后续演进方向：

1. workspace 容量由框架真实 cache capacity 驱动，而不是模型侧独立公式推导。
2. selected workspace 生命周期接入 scheduler，支持 online continuous batching。
3. 明确 swapped full KV 在 PD transfer 中的传输方式和对齐要求。
4. 最终把 selected workspace 声明、分配、reset 也沉淀为框架通用能力。
