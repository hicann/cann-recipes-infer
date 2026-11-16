# RotaryEmbedding 预计算与调用模式参考

## 目的

本文档用于指导 RoPE 前置改造，收敛出更成熟的 `RotaryEmbedding` 模块化写法和调用方式。

## 适用场景

当目标算子或目标链路需要 RoPE 相关前置改造时，可优先参考本文档。例如：

- 需要将分散的 `freqs_cis` / `cos/sin` 计算收敛成统一模块
- 需要为 `npu_mla_prolog_v3`、`npu_kv_rmsnorm_rope_cache_v2`、`npu_fused_infer_attention_score_v2` 等路径准备稳定的 `cos/sin` 或 `query_rope / key_rope`
- 需要把 `Prefill / Decode` 的 RoPE 取值逻辑从各层局部切片改成模型级统一准备
- 需要为 `PA / 非PA` 路径整理更清晰的 RoPE metadata 组织方式

## 推荐模式

1. 在模型初始化时预计算 `cos/sin`，用 `register_buffer` 保存。
2. 在统一入口按模型的位置语义取值（如 `position_ids`），集中处理 Prefill / Decode、PA / 非 PA 和 BNSD / TND 的差异。各层复用准备好的 `cos/sin`。

## 最小样例

复用模板时，频率计算、旋转维度的排列和 cos/sin 布局沿用目标模型的约定。

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, dtype):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, hidden_states, position_ids):
        cos = torch.index_select(self.cos_cached, 0, position_ids.view(-1)).unsqueeze(1).unsqueeze(1)
        sin = torch.index_select(self.sin_cached, 0, position_ids.view(-1)).unsqueeze(1).unsqueeze(1)
        return cos.to(hidden_states.dtype), sin.to(hidden_states.dtype)
```

```python
def _init_rope(self):
    self.rotary_emb = RotaryEmbedding(
        dim=self.config.qk_rope_head_dim,
        max_position_embeddings=self.config.max_position_embeddings,
        base=self.config.rope_theta,
    )
```

```python
def forward(self, hidden_states, position_ids, ...):
    cos_sin = self.rotary_emb(hidden_states, position_ids)
    for layer in self.layers:
        hidden_states = layer(hidden_states, cos_sin=cos_sin, ...)
    return hidden_states
```
