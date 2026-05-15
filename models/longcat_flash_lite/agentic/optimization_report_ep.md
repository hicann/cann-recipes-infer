# LongCat-Flash-Lite EP 部署优化报告

## 1. 模型信息

| 项目 | 值 |
|------|------|
| 模型名称 | LongCat-Flash-Lite |
| 架构 | MoE + MLA + N-gram Embedding（LongcatFlashNgramForCausalLM）|
| 硬件 | Atlas A2 / A3，8 卡（HCCL 节点内）|
| 量化 | BF16 |
| 总参数 | ~80 B（NgramEmbedding 占 31 B，MoE 占 31 B）|

### 1.1 模型结构

```
embedding 层:
  embed_tokens         VocabParallelEmbedding(155136 → 3072)            0.4 B
  ngram_embedders[12]  VocabParallelEmbedding(~10.2 M → 256) × 12       31 B
14 × decoder_layer:
  ├── 2 × LongcatFlashMLA  (q_lora=1536, kv_lora=512, num_heads=32)
  ├── 2 × LongcatFlashMLP  (gate/up/down 3072↔6144)
  └── 1 × LongcatFlashMoE  (256 routed + 128 zero, top-12, expert_intermediate=1024)
final RMSNorm + lm_head(3072 → 155136)
```

特殊点：
- **Ngram embedding**：12 个子表，每表 ~10.2M 行 × 256 列，是显存大头，必须 TP 切分。
- **MoE 双层结构**：256 routed expert（EP=8 时每卡 32 个）+ 128 zero（identity 副本，每卡持有，不参与 EP 切分）。
- **Dual sub-layer attention**：每 layer 含 2 个 attention block + 1 个共享 MoE。

---

## 2. 性能基线

| 指标 | 值 | 测试条件 |
|------|-----|---------|
| Steady prefill | 198 ms | input_len=1024, batch=1, attn_tp=8 |
| Decode avg / token | 161 ms | batch=1, max_new=128 |
| 单卡 HBM 峰值 | 19.6 GB | EP=8, dense=DP |

> 基线为本 EP 拓扑（attn_tp=8 + dense DP + ngram TP + EP=8）的初次部署测试。

---

## 3. 并行切分

部署拓扑（`config/longcat_flash_lite_rank_8_8ep.yaml`）：

| 模块 | 切分 | 通信形式 |
|------|------|---------|
| MLA | attn_tp=4，attn_dp=2 | RS+AG（prefill）/ AllReduce（decode）|
| Dense MLP | DP=1（每卡完整权重）| 无 |
| MoE | EP=8 | dispatch_v2 / combine_v2（decode）；double-routing AllToAll（prefill）|
| 主 Embedding | DP=1（每卡完整主表 0.8 GB）| 无 |
| Ngram 子表 | TP=8 | AllReduce |
| LM Head | TP=8 | AllGather |

技术依据：
- Dense MLP DP-replicated：避免 AllReduce 同步开销；单卡 HBM 容纳每卡多 1.4 GB 权重。
- MoE EP=8：MoE expert_intermediate=1024，TP=8 切分后单卡 128 维，矩阵碎片化严重；EP 让每卡持完整 32 个 expert 计算。
- Ngram 子表独立 TP=8（通过 `model_config.custom_params.ngram_embed_tp_size=8`）：12 个子表共 ~31 B，必须切；与主 embed 解耦后主表可走 DP，省去主 embed 的 AllReduce。
- attn_tp=4 + attn_dp=2 拓扑使 q/k/v/o 矩阵保持较大尺寸提升 cube 利用率；batch=2 时 MoE GMM 接近峰值。

性能数据：

| 配置 | Prefill | Decode | 输出对齐 |
|------|---------|--------|---------|
| EP8 + attn_tp=4 + batch=2 | 207 ms | 150 ms | 与 §2 基线一致 |

---

## 4. KVCache 与 Attention

### 4.1 MLA 压缩缓存 + Paged Attention

- 缓存格式：`cache_mode="PA_NZ"`，block_size=128。
- 每 token 仅缓存 latent KV：`kv_lora_rank=512` + `qk_rope_head_dim=64` = 576 维，较 vanilla MHA 全展开 1280 维节省 55%。
- Prefill 路径：非 absorb，展开 K/V，FA 走 `NTD_TND` 布局、sparse_mode=3，actual_seq_lengths 用 cumulative offsets（支持 batch>1 packed prefill）。
- Decode 路径：absorb（key=value=cache_nope），FA 走 `BSND_NBSD` 布局、sparse_mode=0，scale=`softmax_scale`。
- `npu_kv_rmsnorm_rope_cache` 在 cache 写入同时完成 KV layernorm + RoPE，输出当前 step 的 k_nope / k_rope 直接喂 FA。

### 4.2 MLA Prolog 融合（decode）

使用 `npu_mla_prolog_v3` 一次性完成 `q_a_proj → RMSNorm → q_b_proj → split → RoPE → absorb-via-K` 与 `kv_a_proj_with_mqa → RMSNorm → RoPE → cache 写入`，替代 6–7 个独立 small op。Q-/KV-LoRA scale 通过 `qc_qr_scale` / `kc_scale` 参数折进 kernel，cache 仍存未缩放值。

启用条件：`exe_mode="ge_graph"` 且 `q_lora_rank > 0`；可通过 yaml `enable_mla_prolog: false` 关闭。

| 项 | 无 prolog | 启用 prolog | Δ |
|---|---|---|---|
| Decode mean | 8.36 ms | 8.10 ms | −3.1% |
| Prefill (post-warmup) | 239 ms | 239 ms | ~0 |

适用范围约束：fused kernel 内 BF16 累加顺序与原拆解 op 不同，从第 4–5 decode token 起出现 token-level 选词差异，语义保持一致。严格逐字对齐场景可关闭 `enable_mla_prolog`。

### 4.3 Prefill TP+SP attention（RS + AG 替代 AllReduce）

Prefill 把 attention 输出端的 `AllReduce(O)` 拆成 `ReduceScatter(O) + 下一层入口 AllGather(input)`，层间 hidden 维度从 `[T, H]` 缩到 `[T/tp, H]`：
- RMSNorm / Dense MLP 计算量 ÷ attn_tp（= 4）
- 层间残差激活显存同等比例减少
- 通信总量与 AllReduce 数学等价（ring AllReduce = RS + AG 两阶段）

启用条件：`attn_tp > 1` 且 `attn_dp > 1`（`is_sp` 守卫），prefill only；decode 不触（`batch_per_dp_rank` 可能小于 attn_tp，pad 浪费抵消收益）。

| 配置 | Prefill (无 SP) | Prefill (SP) | Δ |
|---|---|---|---|
| ep8 b=8 | 199.91 ms | 145.22 ms | −27% |
| ep8 b=2 | 66 ms | 65 ms | 持平 |

适用范围约束：b=2 单请求 256 token T 小，SP 节省的 norm/MLP 算力被 RS+AG 同步开销抵消；收益需要更大 batch 或更长输入才显著。

---

## 5. 算子融合

| 模块 | 实现 |
|------|------|
| RMSNorm | `npu_rms_norm` |
| Residual + RMSNorm | `npu_add_rms_norm` |
| Dense MLP | `MergedColumnParallelLinear` 合并 gate/up，`npu_swiglu` 激活 |
| MoE Router | `npu_moe_gating_top_k` |
| MoE Token 分发 (decode) | `npu_moe_distribute_dispatch_v2` + `npu_moe_distribute_combine_v2` |
| MoE Token 分发 (prefill) | `_dispatch_double_routing`（两轮 AllToAll）|
| MoE Expert 计算 | `FusedMoEGMM`（`npu_grouped_matmul` + `npu_swiglu`）|

技术依据：
- Decode 走 MC2 `dispatch_v2 / combine_v2`：A3 上无 experts/rank 限制；A2 受 `experts_per_rank ≤ 24` 算子限制，EP=8 下 32/rank 不满足，A2 部署需通过 yaml `custom_params.enable_moe_mc2_dispatch: false` 走 double_routing 路径，或设 `exe_mode: eager`。
- Prefill 单层 token 数受 dispatch_v2 BS 上限（~512）限制，使用 `_dispatch_double_routing` 两轮 AllToAll 维持 prefill 正确性。

精度：与基线逐字匹配。

---

## 6. 图模式

适配范围：Decode 整图编译（fullgraph=True），Prefill 保持 eager。

NgramEmbedding 整体进图：
- `ngram_context` 使用 `register_buffer` 持久化，`init_ngram_cache(B, device)` 在 `init_pa_cache` 末尾预分配。
- `_shift_right_ignore_eos` 用 Python 静态展开 prefix-max 链（GE 后端不支持 cummax）。
- `forward` 用 `copy_()` 完成 in-place 写入以满足 graph 跟踪。

其它图适配：
- LM head：使用 `dist.all_gather_into_tensor(buf, x)`，避免 list 物化触发 graph break。
- `process_weights_after_loading` 对 `kv_b_proj_w_k/v` 等 absorb 权重以及 prolog 权重 `mark_static_address`。
- `LongcatFlashLiteForCausalLM._can_compile_fullgraph = True`。

性能（attn_tp=4, batch=2）：

| 项 | eager (融合算子后) | ge_graph | Δ |
|---|---|---|---|
| Decode mean | 43.2 ms | 9.10 ms | −79% |
| Prefill (steady) | 161 ms | 239 ms | +48% |

适用范围约束：Prefill 在图模式下退化 +48%；MoE prefill 仍走 `_dispatch_double_routing` eager（含 host sync），且 `dispatch_v2` 受 BS 上限限制无法替换。

---

## 7. 吞吐扩展性

固定 `attn_tp=4 + attn_dp=2`，扫多组 `(input_len, batch_size)`：

| 配置 | 输入 | 输出 | batch | per-DP batch | Prefill | Decode mean | 吞吐 (tokens/s) | vs b=2 1K |
|------|------|------|-------|--------------|---------|-------------|----------------|-----------|
| 短输入 baseline | 1024 | 128 | 2 | 1 | 62 ms | 8.10 ms | 247 | 1.0× |
| 短输入大 batch | 1024 | 128 | 8 | 4 | 129 ms | 9.54 ms | **838** | **3.39×** |
| 长输入大 batch | 4096 | 1024 | 8 | 4 | 1536 ms | 10.62 ms | **753** | **3.05×** |

观察：
- batch=8 vs batch=2 吞吐扩展 3.0–3.4×（理论 4×），EP+DP 拓扑中等 batch 扩展性良好。
- 长输入 (1K → 4K) decode latency 仅增 +12%，attention 走 PA block 分块，每步只需读取 KV cache 的活跃 block，O(seq) 增长被通信开销稀释。
- Prefill 超线性增长（b=8 时 seq 1K → 4K 即 ×4，prefill 129 → 1536 ms 为 ×11.9），attention O(seq²) 计算在长输入下占主导。

---

## 8. 累计性能演进

| 路径 | 关键改造 | Decode (ms/token) | vs 基线 |
|------|---------|-------------------|---------|
| 基线 (TP=8 eager) | manual SDPA + Python MoE loop | 273 | 1.0× |
| 并行切分 | EP=8 + dense DP + ngram TP | 84.1 | 3.25× |
| + KVCache + FA | PA + `npu_fused_infer_attention_score` | 75.5 | 3.62× |
| + 算子融合 | RMSNorm / SwiGLU / AddRMSNorm | 68.5 | 3.99× |
| + MC2 dispatch_v2 | `npu_moe_distribute_dispatch/combine_v2` | 43.2 | 6.32× |
| + GE graph 整图 | torchair full-graph decode | 9.10 | 30.0× |
| + MLA prolog 融合 | `npu_mla_prolog_v3` | **8.10** | **33.7×** |

---

## 9. 当前未覆盖项

- **Prefill 图化**：MoE prefill 路径仍走 `_dispatch_double_routing` eager，受 `dispatch_v2` BS 上限（~512）限制无法直接用于 prefill；prefill 独立 graph 编译未做。
- **MLA prolog BF16 漂移**：fused kernel 累加顺序差异导致 token-level 选词分歧，语义不受影响；可按需关闭 `enable_mla_prolog` 严格对齐。
- **多流 / dispatch overlap**：MoE dispatch 与 GEMM 计算重叠未做。
- **权重量化**：W8A8 / W4A16 路径未接入。
