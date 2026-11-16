# FA 调用完整代码示例

按 FA 实际读取的 KV 形态选择示例。部分模型 Prefill 直读 KV（不传 `block_table`）、Decode 走 Paged 模式（传 `block_table`），与独立部署模式一（无 paging、FA 直读整个 cache）不是同一形态。

## 标准 LLM TND / 滑窗 TND 路径

Prefill 直读 packed K/V（KV 长度取累计值）、Decode 经分页缓存读取（KV 长度按请求传入）的滑窗与 sink 示例见[框架接入模板 §4.2](framework_kv_reference.md#42-attention-层按-attn_type-取出对应-tensor)。

### Prefill / Decode 均读取分页缓存（标准 GQA，FA v2）

```python
# Eager path in models/qwen/models/modeling_qwen.py; select the layer's mapping first.
for cache, states in ((self.k_cache, key_states), (self.v_cache, value_states)):
    torch_npu.npu_scatter_nd_update_(
        cache.view(-1, self.num_key_value_heads_per_rank, self.head_dim),
        slot_mapping.view(-1, 1),
        states,
    )
attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score_v2(
    query_states,
    self.k_cache.view(*self.k_cache.shape[:2], -1),
    self.v_cache.view(*self.v_cache.shape[:2], -1),
    num_query_heads=self.num_heads_per_rank,
    num_key_value_heads=self.num_key_value_heads_per_rank,
    input_layout="TND",
    sparse_mode=3,
    atten_mask=attention_mask,
    actual_seq_qlen=forward_metadata.actual_seq_lengths_cu_q,
    actual_seq_kvlen=forward_metadata.actual_seq_lengths_kv,
    softmax_scale=self.scale_fa,
    block_table=block_table,
    block_size=self.block_size,
)
```

> migrator 阶段过渡形态（独立部署 BSH + 连续缓存 + 不传 block_table）参考 `references/standalone_kv_reference.md` §1。

## MLA 系列（TND_NTD layout + MLA absorb）

本例使用 Paged 模式（传 `block_table`）+ MLA（key=value=cache_nope, query_rope/key_rope 分离）。

### 单 token Decode（FA v2，MLA absorb）

```python
# 参考: models/deepseek_r1/models/modeling_deepseek.py
attn_output, _ = self.fa_ops.npu_fused_infer_attention_score_v2(
    q_nope, k_nope, k_nope,
    query_rope=q_pe, key_rope=k_rope,
    atten_mask=None,
    actual_seq_kvlen=actual_seq_lengths_kv,
    actual_seq_qlen=actual_seq_lengths_q,
    block_table=block_table,
    num_query_heads=self.num_heads_per_rank,
    num_key_value_heads=self.num_key_value_heads_per_rank,
    softmax_scale=self.softmax_scale,
    input_layout="TND_NTD",
    sparse_mode=0,
    block_size=self.block_size,
    query_quant_mode=0, key_quant_mode=0, value_quant_mode=0,
)
```

因果 Prefill 或多 token Decode 需使用与路径匹配的因果 mask 和 sparse_mode。

若 absorb 需从 `kv_b_proj` 派生 K/V 投影，须在所依赖的权重加载和布局变换完成后使用派生权重。融合写入 nope/rope cache 的示例见 [框架接入参考 §4.3](framework_kv_reference.md)。
