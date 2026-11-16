# 并行层替换代码示例

> **API 速查**：`module/linear.py` 中的 `ColumnParallelLinear` / `RowParallelLinear` / `QKVParallelLinear` / `VocabParallelEmbedding` 均使用 `tp_size: int` + `tp_rank: int` 参数（不接受 `tp_group`）。通信组对象通过 `comm_manager.get_group("xxx_group")` 获取，用于显式 collective（all_gather / reduce_scatter 等）。
>
> `RowParallelLinear.forward` **不内置 AllReduce**，调用方按输出布局执行 AllReduce 或 ReduceScatter；保留完整 token 布局时可用 AllReduce，token 切分路径用 ReduceScatter。

并行 Linear 替换时沿用原层的 bias 设置，并透传量化配置 `quant_config` 与完整模块 `prefix`。

## Attention 层

| 原组件 | 替换为 | 通信组 | 说明 |
|-------|-------|--------|------|
| QKV Linear | `QKVParallelLinear` | `attn_tp_group` | 列切分，Q/K/V 按头数分 |
| O Linear | `RowParallelLinear` | `attn_tp_group` | 行切分，调用方按布局归约（AllReduce / ReduceScatter） |

```python
from module.linear import QKVParallelLinear, RowParallelLinear

attn_tp_rank = comm_manager.get_rank("attn_tp_group")

# QKV 投影
self.qkv_proj = QKVParallelLinear(
    hidden_size=config.hidden_size,
    head_size=config.head_dim,
    total_num_heads=config.num_attention_heads,
    total_num_kv_heads=config.num_key_value_heads,
    tp_size=self.attn_tp_size,
    tp_rank=attn_tp_rank,
)

# O 投影
self.o_proj = RowParallelLinear(
    config.num_attention_heads * config.head_dim, config.hidden_size,
    tp_size=self.attn_tp_size,
    tp_rank=attn_tp_rank,
)
```

**o_proj_tp_size 独立配置**（当 `o_proj_tp_size ≠ attn_tp_size` 时）：输入维度按实际 Attention 输出确定，以下以 MLA 的 `num_attention_heads × v_head_dim` 为例。

```python
oproj_tp_rank = comm_manager.get_rank("oproj_tp_group")

self.o_proj = RowParallelLinear(
    config.num_attention_heads * config.v_head_dim, config.hidden_size,
    tp_size=self.o_proj_tp_size,
    tp_rank=oproj_tp_rank,
)
```

---

## Dense FFN 层

| 原组件 | 替换为 | 通信组 | 说明 |
|-------|-------|--------|------|
| Gate Linear | `ColumnParallelLinear` | `dense_tp_group` | 列切分 |
| Up Linear | `ColumnParallelLinear` | `dense_tp_group` | 列切分 |
| Down Linear | `RowParallelLinear` | `dense_tp_group` | 行切分，调用方按布局归约（AllReduce / ReduceScatter） |

```python
from module.linear import ColumnParallelLinear, RowParallelLinear

dense_tp_rank = comm_manager.get_rank("dense_tp_group")

self.gate_proj = ColumnParallelLinear(
    config.hidden_size, config.intermediate_size,
    tp_size=self.dense_tp_size,
    tp_rank=dense_tp_rank,
)
self.up_proj = ColumnParallelLinear(
    config.hidden_size, config.intermediate_size,
    tp_size=self.dense_tp_size,
    tp_rank=dense_tp_rank,
)
self.down_proj = RowParallelLinear(
    config.intermediate_size, config.hidden_size,
    tp_size=self.dense_tp_size,
    tp_rank=dense_tp_rank,
)
```

---

## Embedding / LMHead

```python
from module.linear import VocabParallelEmbedding, ColumnParallelLinear

# Embedding
self.embed_tokens = VocabParallelEmbedding(
    config.vocab_size,
    config.hidden_size,
    self.padding_idx,
    torch.bfloat16,
    tp_size=self.embed_tp_size,
    tp_rank=comm_manager.get_rank("embed_tp_group"),
)

# LMHead
self.lm_head = ColumnParallelLinear(
    input_size=config.hidden_size,
    output_size=config.vocab_size,
    bias=False,
    tp_size=self.lmhead_tp_size,
    tp_rank=comm_manager.get_rank("lmhead_tp_group"),
)
```

---

## 模块间数据重排

Embedding 示例对应 `embed_tp_size > 1`、`attn_tp_size = 1`，各 rank 持有等长 token 分片的 Decode 路径；Dense FFN 示例同样以 token 分片作为输入和输出布局。

```python
embed_tp_group = comm_manager.get_group("embed_tp_group")
dense_tp_group = comm_manager.get_group("dense_tp_group")

# Gather token IDs, then reduce vocabulary-shard contributions to local tokens.
all_input_ids = input_ids.new_empty(input_ids.numel() * self.embed_tp_size)
dist.all_gather_into_tensor(all_input_ids, input_ids, group=embed_tp_group)
embed_tp_rank = comm_manager.get_rank("embed_tp_group")
local_ids = all_input_ids - embed_tp_rank * self.vocab_size_per_rank
mask = (local_ids >= 0) & (local_ids < self.vocab_size_per_rank)
embed_output = self.embed_tokens(local_ids * mask) * mask.unsqueeze(-1)
attn_input = embed_output.new_empty(input_ids.numel(), embed_output.shape[-1])
dist.reduce_scatter_tensor(attn_input, embed_output, group=embed_tp_group)

# Gather token shards for FFN, then reduce and scatter output partial sums.
dist.all_gather_into_tensor(x_output, x, group=dense_tp_group)  # 输入聚合
# ... FFN 计算 ...
dist.reduce_scatter_tensor(mlp_res, down_proj, group=dense_tp_group)  # 输出分散
```

参考实现：
- `models/qwen3_moe/models/modeling_qwen3_moe.py`、`models/gpt_oss/models/modeling_gpt_oss.py`（标准 TP 替换）
- `models/deepseek_r1/models/modeling_deepseek.py`（MLA + 模块间数据重排 + EP）
