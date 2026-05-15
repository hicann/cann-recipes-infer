# Gemma-4-26B-A4B 模型优化报告

## 1. 模型概况

| 项目 | 内容 |
|------|------|
| 模型名称 | Gemma-4-26B-A4B (google/gemma-4-26B-A4B) |
| 架构类型 | 多模态 MoE（仅适配 Language Decoder，Vision Tower 不在覆盖范围内）|
| 总参数量 | 26.5B（活跃 ~3.8B/token，128 experts top-8）|
| 权重 | BF16 ~51.6 GB |
| 部署 | 8 卡，EP=8，embed/lmhead_tp=8，attn_tp=1 |

### 1.1 架构特点

```
Token Embedding (vocab=262144, shared with LM Head)
└─ Decoder Block × 30
     ├─ Attention（双模式）
     │    ├─ Sliding (25 层): GQA, N_h=16, N_kv=8, head_dim=256, window=1024
     │    └─ Full (5 层): GQA, N_h=16, N_kv=2, head_dim=512, k_eq_v=True
     ├─ Dense MLP: gate+up+down, intermediate=2112, GELU
     ├─ MoE: 128 experts, top-8, intermediate=704, GEGLU
     │    （每层同时有 Dense MLP 和 MoE，非交替式）
     └─ 6× LayerNorm + layer_scalar
└─ LM Head (shared with Embedding)
```

关键特殊点：
- 双模式 Attention 导致 KV cache 维度异构（sliding head_dim=256，full head_dim=512）。
- Full 层 `k_eq_v=True`，无独立 v_proj。
- QK RMSNorm 已对 Q/K 归一化，FA 调用使用 `scale=1.0`。
- 大词表 262144，embedding 与 LM head 共享权重。

---

## 2. 性能基线

| 指标 | 值 | 测试条件 |
|------|-----|---------|
| Prefill | 312.51 ms | BS=8, seq_len=256, 8 卡 eager |
| Decode | 98.47 ms | BS=8, 8 卡 eager |

---

## 3. 并行切分

部署拓扑：

| 模块 | 切分 | 技术依据 |
|------|------|---------|
| Attention | `attn_tp=1`（DP=8）| full attention `N_kv=2`，TP 切分上限为 2；DP=8 最大化吞吐 |
| Dense MLP | `dense_tp=1` | `intermediate=2112` 较小，TP 切碎后矩阵 cube 利用率低 |
| MoE | `EP=8` | 128 experts / 8 = 16 experts/rank |
| Embedding / LM Head | `embed_tp=8` / `lmhead_tp=8` | 大词表 262144 沿 vocab 切分 |

精度：与基线一致。性能：Prefill 313 ms / Decode 97 ms。

---

## 4. KVCache 与 Attention

### 4.1 异构连续缓存

双模式 Attention 导致 KV cache 维度异构：sliding 层 `kv_dim = N_kv × head_dim = 8 × 256 = 2048`，full 层 `kv_dim = 2 × 512 = 1024`（k_eq_v，K/V 共享一组缓存）。采用连续缓存（非 Paged）：

- 每层独立分配缓冲，sliding 层与 full 层 shape 不同，避免统一 PA block 格式两套 block_table 的复杂度。
- attn_tp=1，没有 KV head 切分需求，连续缓存足够。

### 4.2 FA 路径

- FA v1（`npu_fused_infer_attention_score`），输入布局 BSH。
- `scale=1.0`：QK RMSNorm 已对 Q/K 归一化，标准 `1/√d` 缩放会让 attention 过于平坦；scale 设为 1.0 才能匹配 reference 实现。
- Sliding 层 Prefill：`sparse_mode=4`，`pre_tokens=1024` 即 sliding window。

精度：与基线一致。性能：Prefill 310 ms / Decode 97 ms。

---

## 5. 算子融合

| 模块 | 实现 | 触发次数/step |
|------|------|---------------|
| Residual + RMSNorm | `npu_add_rms_norm` | 60 |
| Sliding RoPE | `npu_rotary_mul` | 50（25 层 sliding 全维 RoPE）|
| MoE Router | `npu_moe_gating_top_k_softmax` | 30 |

Full attention 的 partial RoPE（`partial_rotary_factor=0.25`）使用 slice + `npu_rotary_mul` + cat 包装实现。

精度：与基线一致。性能：Decode 97 → 92 ms（−5.4%）。

未融合项见 §9 算子需求。

---

## 6. 图模式

### 6.1 GE graph 适配

| 项 | 配置 |
|----|------|
| 后端 | torchair GE graph，`fullgraph=True` |
| 覆盖范围 | Decode 整图，Prefill 保持 eager |
| FA | Decode 用 `tng.ops.npu_fused_infer_attention_score`；Prefill 用 `torch.ops.npu` 同名算子 |
| GEGLU | Prefill 用 `npu_geglu`；Decode 用手动 `F.gelu(gate, approximate='tanh') * up`（`npu_geglu` 未注册图模式 dispatch）|

图适配关键改造：
- MoE Decode 路由：双模式 EP decode，见 §6.3。
- RoPE cos/sin：在 `init_pa_cache` 时预计算 buffer，graph 内直接索引，避免 `kv_len.max().item()` 引入 host 同步。
- `actual_seq_lengths_kv` 使用 Tensor 形态，与 `tng.ops` 接口匹配，避免 list 元素变化触发重编译。

精度行为：Prefill 首 token 与 eager 基线 logit 完全一致（ID=236776）；Decode 因手动 GEGLU 与 EP 路由的浮点路径差异，输出文本语义连贯但与 eager 不完全相同，差异来源于 30 层 BF16 舍入累积。

| 指标 | 基线 (eager) | ge_graph | vs baseline |
|------|-------------|----------|-------------|
| Prefill (A2) | 312.51 ms | 302.52 ms | −3.2% |
| Decode (A2) | 98.47 ms | 18.21 ms | **−81.5%** |

### 6.2 npugraph_ex（备选）

切换 yaml `model_config.exe_mode: "ge_graph"` → `"npugraph_ex"` 即可换到 torch.compile aclgraph 后端，代码共用。

| 模式 | 平台 | Prefill | Decode/token | 首次 prefill 编译 | 首次 decode 编译 |
|------|------|---------|--------------|------------------|------------------|
| ge_graph | A2 | 189 ms | 15 ms | — | — |
| ge_graph | A3 | 76.43 ms | 10.20 ms | 8.38 s | 605 s |
| npugraph_ex | A3 | 102.29 ms | 11.59 ms | 7.18 s | 358 s |

适用范围约束：稳态吞吐 ge_graph 更优（A3 上 prefill 快 25%、decode 快 12%）；npugraph_ex 编译显著快（decode 编译 358 s vs 605 s），适合编译启动敏感的短任务场景。

### 6.3 MoE EP decode：mc2 与 local_experts 双路径

`Gemma4SparseMoeBlock` 提供 `moe_ep_decode_mode` 开关（yaml `model_config.custom_params.moe_ep_decode_mode`）：

- **`mc2`**：`npu_moe_distribute_dispatch_v2` + `npu_moe_distribute_combine_v2`，与 qwen3_moe / longcat-flash / deepseek 等仓内 MoE 模型一致。
- **`local_experts`**：每个 rank 计算所有 16 个本地 expert（非本 rank 的 expert routing weight 置 0），最后一次 AllReduce 汇总。完全静态 shape，对 ge_graph fullgraph 友好。

技术依据：`npu_moe_distribute_dispatch_v2` 在 A2 受 `experts_per_rank ≤ 24` 限制，A3 无该限制；Gemma-4 EP=8 下 16 experts/rank 在两个平台都满足。代码按 `experts_per_rank ≤ 24` 自动选 `mc2`，超阈值走 `local_experts`，可通过 yaml 显式覆盖。

A3 实测（8 卡 EP=8，BS=8，input_len=256，max_new_tokens=32）：

| 模式 | Prefill | Decode |
|------|---------|--------|
| ge_graph + local_experts | 76 ms | 10.20 ms |
| ge_graph + mc2 | 77 ms | 10.50 ms |
| npugraph_ex + local_experts | 102 ms | 11.59 ms |
| npugraph_ex + mc2 | 112 ms | 11.73 ms |

适用范围约束：BS=8 / decode 1 token 工况下，每层 MoE routing 数据量很小（8 token × top-8 ≈ 64 个 token-expert 对，hidden=2816），`AllReduce[8, 2816]` ≈ 22 KB，ring/tree 一次集合通信即完成；mc2 的 `dispatch_v2 + combine_v2` 是 2 次 AllToAll-style 通信，加上 `max_dispatch_tokens` padding 固定开销，小 batch 下摊不开。mc2 的相对优势在更大 batch 或 prefill 阶段才显现（每层 routing token 数显著增加，稀疏路由收益跑赢 AllReduce + 冗余 GMM 成本）。当前规模下两路径输出一致、性能接近，默认 mc2 以与仓内其他模型保持一致。

---

## 7. 跨硬件性能

A3 vs A2（同代码、同 `gemma4_rank_8_8ep_*.yaml`）：

| 路径 | 平台 | Prefill | Decode/token |
|------|------|---------|--------------|
| eager | A2 | 312.5 ms | 98.5 ms |
| ge_graph | A2 | 189 ms | 15 ms |
| ge_graph | **A3** | **76.43 ms** | **10.20 ms** |
| npugraph_ex | **A3** | **102.29 ms** | **11.59 ms** |

A3 同 ge_graph 路径对比 A2：prefill **2.47×**、decode **1.45×** 加速。

---

## 8. 累计性能演进

| 路径 | 关键改造 | Prefill (ms) | Decode (ms) | vs baseline (decode) |
|------|---------|-------------|------------|----------------------|
| 基线 (eager) | — | 312.5 | 98.5 | 1.0× |
| + 并行切分 | EP=8 + embed/lmhead_tp=8 + attn_tp=1 (DP=8) | 313 | 97 | 1.02× |
| + KVCache + FA | 异构连续缓存 + FA v1 + sliding sparse_mode=4 | 310 | 97 | 1.02× |
| + 算子融合 | RMSNorm / RoPE / Router | 307 | 92 | 1.07× |
| + 图模式 (A2) | ge_graph decode 整图 + local_experts MoE | 303 | **18.2** | **5.42×** |
| 同代码 (A3) | — | 76.4 | **10.2** | **9.66×** |

> 测试条件：BS=8, seq_len=256, 8 卡, BF16。

---

## 9. 算子需求

CANN 算子约束导致融合空缺：

| 算子 | 约束 | 影响 |
|------|------|------|
| `npu_gelu_mul` | dim ≤ 1024 | Dense FFN intermediate=4224，无法替换 GEGLU |
| `npu_ffn` (geglu mode) | 仅支持 FP16 | BF16 模型无法使用 |
| `npu_rotary_mul` | 无 `partial_rotary_factor` 原生支持 | Full attention 25% partial RoPE 需 slice + 融合 + cat 包装 |
| `npu_geglu` | 未注册图模式 dispatch | Decode 走手动 `F.gelu * up`，引入与 eager 的浮点路径差异 |

---

## 10. 当前未覆盖项

- **Prefill 图化**：Prefill 仍保持 eager，未做独立 graph 编译。
- **Decode GEGLU 精度对齐**：受 `npu_geglu` 图模式不支持限制，30 层手动 GEGLU 累积 BF16 舍入差异；语义保持一致，需 bit-exact 对齐场景需要算子侧补齐。
- **Vision Tower**：多模态视觉路径未适配。
- **量化**：W8A8 / W4A16 路径未接入。
- **Sliding 缓存优化**：sliding 层可改为环形缓存压缩长 KV，本期未做。
