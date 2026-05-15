# LongCat-Flash-Lite TP 部署优化报告

## 1. 模型信息

| 项目 | 内容 |
|------|------|
| 模型名称 | LongCat-Flash-Lite |
| 模型架构 | MoE LLM (MLA + Sparse MoE + N-gram Embedding) |
| 硬件平台 | Atlas A2 / A3 |
| 卡数 (world_size) | 8 |
| 量化模式 | BF16 |

### 1.1 模型结构

```
VocabParallelEmbedding (131072 → 3072, TP-split)
  └─ NgramEmbedding (4-gram, 4-split, 12 sub-embedders + post_proj)
  └─ YarnRoPE (interleaved, theta=5e6, factor=10, max_pos=327680)
  └─ DecoderLayer × 14 (dual sub-layer + shortcut MoE)
       ├─ Sub-layer 0:
       │    ├─ RMSNorm → MLA Attention[0] (q_lora=1536, kv_lora=512, 32 heads)
       │    ├─ RMSNorm → MoE (256 routed + 128 identity, top-12, scale=6.0)
       │    ├─ RMSNorm → Dense MLP (3072→6144→3072, SiLU)
       ├─ Sub-layer 1:
       │    ├─ RMSNorm → MLA Attention[1]
       │    ├─ RMSNorm → Dense MLP → + shortcut_MoE_output
  └─ RMSNorm (final)
  └─ LM Head (3072 → 131072, ColumnParallel + all_gather_into_tensor)
```

---

## 2. 性能基线

| 指标 | 值 | 测试条件 |
|------|-----|---------|
| Prefill 耗时 | ~2580 ms | input_len=1024, batch=1 |
| Decode 单步耗时 | ~273 ms | batch=1, eager, manual SDPA, Python MoE for-loop |
| 吞吐 | ~3.66 tok/s | input_len=1024, output_len=128, batch=1 |

### 2.1 精度基线

```
输入：An attention function can be described as mapping a query and a set of
      key-value pairs to an output, where the query, keys, values, and output
      are all vectors. The output is
输出：computed as a weighted sum of the values, where the weight assigned to
      each value is computed by a compatibility function of the query with the
      corresponding key. ...
```

---

## 3. 并行切分

部署拓扑（`config/longcat_flash_lite_rank_8_8tp.yaml`）：

| 模块 | 切分 |
|------|------|
| MLA | TP=8 |
| Dense MLP | TP=8 |
| MoE | TP=8 |
| Embedding (主表 + Ngram 子表) | TP=8 |
| LM Head | TP=8 |

技术依据：单 batch、低延迟场景下全模块 TP=8 平铺最大化单 token 算力；Ngram 子表 12 份共 ~31 B，TP 切分是显存约束。

---

## 4. KVCache 与 Attention

### 4.1 MLA 压缩缓存 + Paged Attention

- 缓存格式：`cache_mode="PA_NZ"`，`block_size=128`。
- 每 token 仅缓存 latent KV：`kv_lora_rank=512` + `qk_rope_head_dim=64` = 576 维，较 vanilla MHA 全展开 1280 维节省 55%。
- Prefill 路径：非 absorb，展开 K/V，FA 走 `NTD_TND` 布局、`sparse_mode=3`。
- Decode 路径：absorb（key=value=cache_nope），FA 走 `BSND_NBSD` 布局、`sparse_mode=0`，配合 `softmax_scale`。
- `npu_kv_rmsnorm_rope_cache` 在 cache 写入同时完成 KV layernorm + RoPE，返回当前 step 的 `k_nope` / `k_rope` 直接喂 FA。
- Decode absorb 路径 KV LoRA scale 因子 `mla_scale_kv_lora = sqrt(hidden / kv_lora_rank) = sqrt(6)` 在 FA 调用前对 `cache_nope_nz` 应用，保持与基线数值等价。

| 指标 | 基线 | KVCache 后 | vs baseline |
|------|------|-----------|------------|
| Prefill | 2580 ms | 2540 ms | −1.6% |
| Decode | 273 ms | 267 ms | −2.2% |
| KVCache 维度 | 1280 维/token/层 | 576 维/token/层 | **−55%** |
| NPU 显存峰值 | — | 20140 MB | — |

精度：与基线逐字匹配。

### 4.2 MLA Prolog 融合（A3 decode）

使用 `npu_mla_prolog_v3` 一次性完成 `q_a_proj → RMSNorm → q_b_proj → split → RoPE → absorb-via-K` 与 `kv_a_proj_with_mqa → RMSNorm → RoPE → cache 写入`，替代 6–7 个独立 small op；Q-/KV-LoRA scale 通过 `qc_qr_scale` / `kc_scale` 参数折进 kernel，cache 仍存未缩放值。

启用条件：`exe_mode="ge_graph"` 且 `q_lora_rank > 0`；可通过 yaml `enable_mla_prolog: false` 关闭。

| 项 | 无 prolog | 启用 prolog | Δ |
|---|---|---|---|
| Decode mean (A3) | 5.86 ms | 5.49 ms | −6% |
| 吞吐 (tok/s, A3) | 170 | 182 | +7% |

适用范围约束：fused kernel 内 BF16 累加顺序与原拆解 op 不同，从第 8 decode token 起出现 token-level 选词差异，语义保持等价。严格逐字对齐场景可关闭 `enable_mla_prolog`。

---

## 5. 算子融合

| 模块 | 实现 |
|------|------|
| RMSNorm | `npu_rms_norm` |
| Residual + RMSNorm（56 处）| `npu_add_rms_norm` |
| Dense MLP | `MergedColumnParallelLinear` 合并 gate/up + `npu_swiglu` |
| MoE Router | `npu_moe_gating_top_k` |
| MoE Token 分发 | `npu_moe_init_routing_v2` + `npu_moe_finalize_routing` |
| MoE Expert 计算 | `FusedMoEGMM`（`npu_grouped_matmul` + `npu_swiglu`）|
| Q RMSNorm | `npu_rms_norm` |

| 指标 | KVCache 后 | 融合算子后 | 本节变化 | vs baseline |
|------|-----------|-----------|---------|------------|
| Prefill | 2540 ms | 105 ms | **−95.9%** | −95.9% |
| Decode | 267 ms | 92 ms | **−65.5%** | **−66.3%** |
| 吞吐 | 3.7 tok/s | 10.9 tok/s | **+195%** | +195% |
| NPU 显存 | — | 17340 MB | — | — |

技术依据：Prefill 1024 token 几乎激活全部 384 个 expert，原 Python for-loop 单层 ~384 次迭代 × 14 层 ≈ 5376 次调度，融合算子合并后 Prefill 加速幅度（−95.9%）远大于 Decode（−65.5%）。

精度：与基线逐字匹配。

---

## 6. 图模式

适配范围：Decode 整图编译（`fullgraph=True`），Prefill 保持 eager。

GE 图适配清单：
- NgramEmbedding 整体入图：`ngram_context` 使用 `register_buffer` 持久化，`init_ngram_cache(B, device)` 在 `init_pa_cache` 末尾预分配；`forward` 用 `copy_()` 完成 in-place 写入以满足 graph 跟踪；`_shift_right_ignore_eos` 用 Python 静态展开 `torch.maximum` 链（GE 后端不支持 cummax）。
- LM head：使用 `dist.all_gather_into_tensor(buf, x)`，避免 list 物化触发 graph break。
- FA 调用：Decode 使用 `tng.ops.npu_fused_infer_attention_score`，接受 Tensor 类型 `actual_seq_lengths_kv`；`dynamic=False` 避免 list 元素变化导致重编译。
- PA cache：`block_table` 做 `mark_static`；cache 保留原地更新语义，使用 `npu_kv_rmsnorm_rope_cache` 返回值作为下游输入（GE 图数据流依赖）。
- MoE identity expert mask：使用乘法掩码（GE graph 不支持布尔索引赋值）。
- `process_weights_after_loading` 对 `kv_b_proj_w_k/v` 等 absorb 权重 `mark_static_address`。
- `LongcatFlashLiteForCausalLM._can_compile_fullgraph = True`。

| 指标 | 融合算子后 (eager) | 当前 (ge_graph) | 本节变化 | vs baseline |
|------|-------------------|----------------|---------|------------|
| Prefill | 105 ms | 107 ms | 持平 | −95.9% |
| Decode (NgramEmbedding eager) | 92 ms | 18.5 ms | **−79.9%** | **−93.2%** |
| Decode (NgramEmbedding 入图) | — | **7.14 ms** | **−61.4% vs 18.5** | **−97.4%** |
| 吞吐 | 10.9 tok/s | 140 tok/s | +1185% | +3681% |
| NPU 显存 | 17340 MB | 17448 MB | 持平 | — |

技术依据（NgramEmbedding 入图）：NgramEmbedding 在 eager 路径下单步 forward 触发 ~200 个小 op（12 个 sub-embedder × multi-op + 12 次 all_reduce + post_proj + shift_right + ngram-id 累加），每 op host dispatch + collective launch ~50–150 µs，累计 ~11 ms。入图后这部分串行开销消除，wall-clock decode 从 18.5 ms 降至 7.14 ms，差值与 PROFILE 实测的 host overhead（11.20 ms）相吻合。

精度行为：前 ~20 个 decode token 与基线逐字一致；之后因 BF16 累加顺序在 graph 内不同（GE 算子调度路径异于 eager）出现 token 级分歧，语义保持等价。

---

## 7. 长输入与跨硬件

### 7.1 同代码跨硬件对比（A2 vs A3）

| 路径 | 硬件 | Prefill | Decode | 吞吐 (tok/s) |
|------|------|---------|--------|--------------|
| ge_graph (NgramEmbedding 图外) | A2 | 107 ms | 18.50 ms | 54.1 |
| ge_graph (NgramEmbedding 图外) | **A3** | **57.1 ms** | **12.01 ms** | **83.9** |
| ge_graph (NgramEmbedding 入图) | A2 | 106 ms | 7.14 ms | 140 |
| ge_graph (NgramEmbedding 入图) | **A3** | **~62 ms** | **5.86 ms** | **~170** |
| MLA prolog 融合 | **A3** | ~62 ms | **5.49 ms** | **~182** |

A3 相对 A2 加速：Decode −18% ~ −35%，吞吐 +21%；NgramEmbedding 入图收益在两个硬件上量级一致（A2 −61%，A3 −51%）。

### 7.2 长输入场景（A3，4K input / 1K output）

| 项 | 1K input / 128 output | 4K input / 1K output |
|---|---|---|
| Prefill (post-warmup) | ~62 ms | 216.5 ms |
| Decode mean | 5.49 ms | 6.00 ms |
| 吞吐 | ~182 tok/s | ~167 tok/s |

适用范围约束：输入从 1K → 4K，单步 decode 仅增 ~9%（KV cache 更大导致 PA 加载与 FA 计算开销），prefill 近似线性增长（×3.5，对应 input 增长 ×4）。长输入下吞吐仍保持 165+ tok/s。

---

## 8. 累计性能演进

A2 演进（同硬件、累计软件优化）：

| 路径 | 关键改造 | Prefill (ms) | Decode (ms) | 吞吐 (tok/s) | vs baseline (decode) |
|------|---------|-------------|------------|-------------|----------------------|
| 基线 | manual SDPA + Python MoE loop | 2580 | 273 | 3.7 | 1.0× |
| + KVCache | MLA 压缩缓存 + PA + FA | 2540 | 267 | 3.7 | 1.02× |
| + 算子融合 | RMSNorm / SwiGLU / MoE GMM | 105 | 92 | 10.9 | 2.97× |
| + GE graph (NgramEmbedding 图外) | torchair decode 整图 | 107 | 18.5 | 54.1 | 14.8× |
| + NgramEmbedding 入图 | host dispatch overhead 消除 | 106 | **7.14** | **140** | **38.2×** |

A3 路径（同代码迁移 + MLA prolog 单独贡献）：

| 路径 | Prefill (ms) | Decode (ms) | 吞吐 (tok/s) | vs A2 同代码 |
|------|-------------|------------|-------------|-------------|
| A2 NgramEmbedding 入图（参考）| 106 | 7.14 | 140 | — |
| A3 同代码（NgramEmbedding 入图） | ~62 | 5.86 | ~170 | A3 硬件代差 |
| A3 + MLA prolog | ~62 | **5.49** | **~182** | A3 + −6% |

> 测试条件：input_len=1024, output_len=128, batch_size=1, TP=8, BF16。

---

## 9. 当前未覆盖项

- **冷启动**：首次图编译耗时 ~80 s；可通过 `enable_cache_compile` 缓存编译产物。
- **Prefill 图化**：Prefill 仍保持 eager，未做单独 graph 编译。
- **MLA prolog BF16 漂移**：fused kernel 累加顺序差异导致 token-level 选词分歧，语义不受影响；严格逐字对齐场景可关闭 `enable_mla_prolog`。
- **批处理与通信重叠**：`batch_size > 1` 路径、TP AllReduce 与 GEMM overlap、多流 dispatch 未覆盖。
- **权重量化**：W8A8 / W4A16 路径未接入。
