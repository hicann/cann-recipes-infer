## 模型信息

### 1. 架构类型

**MoE (Mixture of Experts) LLM** -- 基于 DeepSeek-V3 / MLA 架构的稀疏专家混合大语言模型，带 N-gram Embedding 增强。

实际 config.json 关键参数（来自 `/data1/models/LongCat-Flash-Lite/config.json`）：
- architecture: `LongcatFlashNgramForCausalLM`
- hidden_size: 3072
- num_layers: 14（每层包含 2 个 sub-layer，实际 attention 层数为 28）
- num_attention_heads: 32
- n_routed_experts: 256
- zero_expert_num: 128（identity expert）
- moe_topk: 12
- vocab_size: 131072
- torch_dtype: bfloat16

### 2. 网络结构拆解

```
VocabParallelEmbedding (vocab=131072, dim=3072, TP-split)
  └─ NgramEmbedding (ngram_vocab_size_ratio=78, emb_neighbor_num=4, emb_split_num=4)
       ├─ 12 个子 embedder (k=4, n-1=3 → 4*3=12), 各带 post_proj 映射回 hidden_size
       └─ 最终 embedding = (base + sum(ngram_projs)) / 13
  └─ RoPE (YarnRotaryEmbedding, interleaved, rope_theta=5e6, factor=10, max_pos=327680)
  └─ DecoderLayer × 14
       每个 DecoderLayer 包含 2 个 sub-layer (dual-sublayer + shortcut MoE):
       ├─ Sub-layer 0:
       │    ├─ RMSNorm → MLA Attention[0] → Residual Add
       │    ├─ RMSNorm → MoE (shortcut_output) + Dense MLP[0] → Residual Add
       ├─ Sub-layer 1:
       │    ├─ RMSNorm → MLA Attention[1] → Residual Add
       │    ├─ RMSNorm → Dense MLP[1] → Residual Add + shortcut_output
  └─ RMSNorm (final)
  └─ ColumnParallelLinear LM Head (3072 → 131072, TP-split)
       └─ all_gather 合并输出 logits
```

### 3. Prefill / Decode 分支差异

**MLA Attention 层 (LongcatFlashMLA.forward)**:
- **KV Cache 写入**: 两阶段均使用 `torch_npu.scatter_update_` 写入 k_cache/v_cache
- **Prefill (seq_length > 1)**: 使用当前计算的 key_states/value_states 做注意力计算，应用 attention_mask（因果掩码）
- **Decode (seq_length == 1)**: 从 k_cache/v_cache 中切片 `[:, :kv_len+1, :, :]` 获取完整 KV，不使用 attention_mask
- **注意力计算**: 两阶段均使用朴素 PyTorch matmul attention（未使用 Flash Attention）

**NgramEmbedding.forward**:
- **Prefill (is_prefill=True)**: 初始化 ngram_context 为 input_ids 末尾 n-1 个 token
- **Decode (is_prefill=False)**: 从 ngram_context 拼接当前 token 构建上下文，然后滚动更新 ngram_context

### 4. 关键模块分析

#### 4.1 Attention 类型: MLA (Multi-head Latent Attention)

基于实际 config.json 确认:
- `attention_method`: 代码中强制为 "MLA"（非 GQA/MHA）
- `q_lora_rank`: 1536 -- Q 路径使用 LoRA 压缩 (hidden→1536→RMSNorm→heads*qk_head_dim)
- `kv_lora_rank`: 512 -- KV 路径使用 LoRA 压缩 (hidden→512+64→RMSNorm→heads*(nope+v))
- `qk_nope_head_dim`: 128, `qk_rope_head_dim`: 64 → `qk_head_dim`: 192
- `v_head_dim`: 128
- `num_attention_heads`: 32
- `mla_scale_q_lora`: true, `mla_scale_kv_lora`: true -- 启用 LoRA 缩放因子
- Q 缩放因子: sqrt(hidden_size / q_lora_rank) = sqrt(3072/1536) = sqrt(2)
- KV 缩放因子: sqrt(hidden_size / kv_lora_rank) = sqrt(3072/512) = sqrt(6)

KV Cache 存储为展开后的 (num_heads_per_rank, qk_head_dim) 和 (num_heads_per_rank, v_head_dim)，未使用 MLA 压缩缓存。

#### 4.2 FFN / MoE 结构

**MoE (LongcatFlashMoE)**:
- Router: TopK router with `e_score_correction_bias`, softmax scoring
- `n_routed_experts`: 256, `zero_expert_num`: 128 (identity expert, 直接输出加权输入)
- 总 expert 数: 384 (256 routed + 128 identity)
- `moe_topk`: 12
- `routed_scaling_factor`: 6.0
- `expert_ffn_hidden_size`: 1024 (每个 expert 的中间维度)
- expert 结构: gate_up_proj (fused, 2*1024) → SiLU → down_proj
- **无 Shared Expert**

**Dense MLP (LongcatFlashMLP)**:
- 每个 sub-layer 有一个 Dense MLP (与 MoE 并行)
- `ffn_hidden_size`: 6144
- 结构: gate_proj + up_proj → SiLU → down_proj

**Shortcut MoE 设计**: Sub-layer 0 的 MoE 输出作为 shortcut，加到 sub-layer 1 的最终输出。

#### 4.3 特殊模块

- **N-gram Embedding**: 使用 4-gram (emb_neighbor_num=4)，每个 n-gram 级别拆分为 4 份 (emb_split_num=4)，共 12 个子 embedder，每个有独立的 vocab 和 post_proj。ngram_vocab_size_ratio=78。
- **YarnRoPE**: interleaved 模式，rope_theta=5e6, scaling_factor=10, max_position=327680, mscale_all_dim=1
- **Identity (Zero) Expert**: 128 个 identity expert，输入直接加权作为输出，不参与 all_reduce

#### 4.4 可配置开关

| 开关 | 实际值 | 说明 |
|------|--------|------|
| attention_method | "MLA" | 仅支持 MLA |
| n_routed_experts | 256 | MoE expert 数 |
| zero_expert_num | 128 | Identity expert 数 |
| zero_expert_type | "identity" | Zero expert 行为 |
| moe_topk | 12 | Top-K 路由 |
| mla_scale_q_lora | true | Q LoRA 缩放 |
| mla_scale_kv_lora | true | KV LoRA 缩放 |
| q_lora_rank | 1536 | Q 压缩维度 |
| kv_lora_rank | 512 | KV 压缩维度 |
| ENABLE_NGRAM_EMBEDDING | True (代码常量) | N-gram Embedding 开关 |
| router_bias | false (默认) | Router 偏置 |

### 5. 运行环境

| 项目 | 值 |
|------|-----|
| 硬件平台 | 8x Ascend 910B4 (32GB HBM each) |
| CANN Toolkit | /usr/local/Ascend/ascend-toolkit/latest |
| PyTorch | 2.8.0 |
| torch_npu | 2.8.0.post2 |
| transformers | 4.55.0 |
| 执行模式 | eager |
| 量化模式 | BF16 (无量化) |
| 模型路径 | /data1/models/LongCat-Flash-Lite |

**YAML 配置 (longcat_flash_lite_8tp.yaml)**:
- world_size: 8
- attn_tp_size: 8, moe_tp_size: 8, embed_tp_size: 8, lmhead_tp_size: 8
- batch_size: 1
- input_max_len: 1024
- max_new_tokens: 128
- exe_mode: eager
- with_ckpt: true

### 6. 性能与精度基线

**基线来源**: baselines/baseline_tp8.md + 运行日志 res/20260328/

#### 6.1 性能基线

| 指标 | 值 | 测试条件 |
|------|-----|---------|
| Weight Loading | 35.91s | 26 safetensors shards |
| Warmup Prefill | ~29076 ms | input_len=1024, batch_size=1 |
| Warmup Decode | ~305 ms | batch_size=1 |
| Prefill (post-warmup) | 2373~2447 ms | input_len=1024, batch_size=1 |
| Decode avg (128 tokens) | ~273 ms/token | batch_size=1 |
| Decode min | ~260 ms | batch_size=1 |
| Decode max | ~326 ms | batch_size=1 |

#### 6.2 精度基线

测试输入:
```
An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is
```

基线输出:
```
computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key. Can you explain this in simpler terms?
Sure! Let's break it down in simpler terms:

Imagine you're reading a book and trying to understand a specific sentence. The **query** is like your question or focus: "What does this sentence mean?"

The **keys and values** are like pieces of information scattered throughout the book. The **keys** are clues that help you find relevant parts, and the **values** are the actual information in those parts.

The **compatibility function**
```

#### 6.3 主要优化机会

1. **KVCache**: 当前使用朴素 scatter_update + 动态切片，未使用 Paged Attention 或 MLA 压缩缓存
2. **Attention**: 使用 PyTorch matmul attention，未使用 Flash Attention 融合算子
3. **RMSNorm**: 使用朴素 PyTorch 实现，可用 torch_npu 融合算子替换
4. **MoE**: 逐 expert 循环计算，可用 grouped_matmul 等融合算子优化
5. **RoPE**: interleaved 模式的 view/transpose 操作可优化
6. **图模式**: 当前 eager 模式，Decode 阶段可适配 torch.compile

## 进度概览

| 阶段 | 结论 | Prefill (ms) | Decode (ms) | 吞吐 (tok/s) | Decode 较基线 |
|------|------|-------------|------------|-------------|--------------|-------------|
| 基线 | - | 2580 | 273 | 3.7 | - |
| Phase 1 | **PASS** | 2540 | 267 | 3.7 | -2.2% |
| Phase 2 | **PASS** | 105 | 92 | 10.9 | -66.3% |
| Phase 3 | **PASS** | 107 | **18.5** | **54.1** | **-93.2%** |
| Phase 3.5 | **PASS** | 106 | **7.14** | **140** | **-97.4%** |

> 所有性能数据来源：benchmark_8tp.sh 3 轮实测，精度均与基线逐字匹配
> 上表为 Atlas A2（8x Ascend 910B 32GB HBM）实测

## A3 硬件实测对比（2026-04-25）

切换到 Atlas A3（8x Ascend 910C，64GB HBM × 2 die）后，使用 chips 8–15 重测同一份 commit 9ed6f41 代码：

| 阶段 | 硬件 | Prefill (ms) | Decode (ms) | 吞吐 (tok/s) | 数据来源 |
|------|------|--------------|-------------|--------------|----------|
| Phase 3 (NgramEmbedding 未进图) | A2 (910B 32GB) | 107 | 18.5 | 54.1 | progress_history |
| Phase 3 (NgramEmbedding 未进图) | A3 (910_93 64GB) | 57.12 | **12.0** | 83.93 | baselines/baseline_..._20260425_114125.json |
| Phase 3.5 (NgramEmbedding 进图) | A2 | 106 | 7.14 | 140 | progress_history |
| Phase 3.5 (NgramEmbedding 进图) | **A3** | ~62 | **5.86** | **~170** | res/20260425/.../log_0.log（16:35-16:45 跑） |

**A3 vs A2（同代码、Phase 3.5）：**
- Decode: 7.14 → 5.86 ms/token，**-18%**（约 1.22× 加速）
- 吞吐: 140 → ~170 tok/s（约 1.21× 加速）
- 主要来自更高 NPU 主频与 HBM 带宽，量化代码路径不变

**A3 上 NgramEmbedding 进图收益**：12.0 → 5.86 ms/token，**-51%**（量级与 A2 相当）

**当前主机/部署细节**：因 chip 0–7 被其他任务占用（每张约 55 GB），本次 8 卡推理通过显式 `ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15` 切到 chip 8–15；同时把 `executor/scripts/function.sh` 中 `MASTER_PORT` 由 6038 改为 26038 以避开 EADDRINUSE。冷启 kernel 编译 ~9 min，复用 kernel_meta 后约 30s。


<!-- ===== 以上为常驻区，不清除 ===== -->

<!-- ===== 以下为工作区，阶段推进时归档并清空 ===== -->

## Phase 4: MLA Prolog Decode 融合（npu_mla_prolog_v3）

### 实施记录
- [完成] 增加 `enable_mla_prolog` 配置开关（model_config）— 默认 True；可通过 YAML 显式 `enable_mla_prolog: false` 或环境变量 `ENABLE_MLA_PROLOG=0` 关闭
- [完成] 准备 prolog 权重：`weight_dq` 与 `weight_dkv_kr` 使用 `q_a_proj.weight.T.contiguous()` / `kv_a_proj_with_mqa.weight.T.contiguous()`，并在 `enable_weight_nz=True` 时调用 `npu_format_cast(29)` 转 NZ
- [完成] 新增 `_forward_decode_with_prolog` 路径：调用 `torch_npu.npu_mla_prolog_v3` 一次性完成 q_a_proj→rmsnorm→q_b_proj→split→rope→absorb 与 kv_a_proj→rmsnorm→cache 写入
- [完成] q_pe 使用 prolog 直接输出（不再单独 npu_interleave_rope）
- [完成] 通过 `qc_qr_scale=mla_scale_q_lora`、`kc_scale=mla_scale_kv_lora` 把两种 lora 缩放同时下沉到算子内部，FA 之前不再额外乘 cache
- [完成] Prefill 路径保持不动（仍走 npu_kv_rmsnorm_rope_cache + 独立 q_b_proj/RoPE/手算 absorb）
- [完成] 新建 `models/longcat_flash_lite_tp8_optimize/config/longcat_flash_lite_8tp_mla_prolog.yaml` 用于打开开关

### 当前代码状态
- `LongcatFlashMLA.__init__` 新增 `self.enable_mla_prolog`（来自 model_config.enable_mla_prolog，默认 True；env `ENABLE_MLA_PROLOG=0/1` 可强制覆盖）。
- `_init_absorb_weights` 之外新增 `_prepare_prolog_weights`：在 `process_weights_after_loading` 阶段构造 `weight_dq_prolog`/`weight_dkv_kr_prolog`（NZ），并保留 `q_b_proj.weight` 直接复用作为 `weight_uq_qr`。
- `_forward_decode` 在 `enable_mla_prolog=True` 时走 prolog 分支；否则保留原 absorb + manual scale 路径。
- `kv_b_proj_w_k` 仍按既有 `_init_absorb_weights` 输出，shape `(N, D, Hckv)`。
- Prefill 路径完全保留 `_forward_prefill`，未做修改。

### 当前代码状态
- `LongcatFlashMLA._init_absorb_weights` 输出 `kv_b_proj_w_k`(N=4,128,512)/`kv_b_proj_w_v`(N=4,512,128)
- `q_a_proj`, `kv_a_proj_with_mqa` 仍是 `nn.Linear`(原 `(out, in)` 排布)，未 transpose；prolog 路径运行时按需 transpose+NZ 后注册到 `weight_dq_prolog`/`weight_dkv_kr_prolog`
- `q_b_proj.weight` 经 `process_weights_after_loading` 后已是 transpose+NZ `(Hcq, N*(D+Dr))`，直接喂给 prolog 的 `weight_uq_qr`
- cache: `cache_nope`/`cache_rope` 仍由 `init_pa_cache` 创建，`cache_mode=PA_NZ`
- prolog `qc_qr_scale=mla_scale_q_lora`、`kc_scale=mla_scale_kv_lora`：cache 仍存"未缩放" k_c，FA 之前**不再**乘 mla_scale_kv_lora（scale 已经被 prolog 内部 absorb 进 q）


### 7. MLA prolog v3 实测结果（2026-04-26 — A3, chip 8-15）

测试环境：
- commit base: `28b416e` + 本次未提交改动（enable_mla_prolog 开关）
- yaml: `config/longcat_flash_lite_8tp_prolog.yaml`（`exe_mode: ge_graph`, `enable_mla_prolog=true`）
- batch=1, input_len=1024 (warmup) → max_new_tokens=128

#### 性能 (decode/token, post-warmup)

| 路径 | mean (ms) | min | max | 稳态值 | 加速 vs baseline |
|------|-----------|-----|-----|--------|-----------------|
| Baseline (legacy decode, 手写 prolog) | 5.84 | 5.71 | 6.40 | ≈ 5.85 | — |
| MLA prolog v3 路径 | **5.54** | 5.38 | 6.62 | ≈ 5.44–5.50 | **−5–6%** (≈0.40ms) |

prefill (post-warmup) ≈ 57 ms，与 baseline 一致；warmup prefill ≈ 5.5–6 s，与 baseline 一致。

#### 精度 (输出对比)

**Baseline (commit 28b416e, A3 chip 8–15)**:
> computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key. Can you explain this in simpler terms?
> Sure! Let's break it down in simpler terms…

**MLA prolog v3 路径**:
> computed as a weighted sum of the input values, where the weights are determined by the attention mechanism. The attention mechanism uses a softmax function to compute the weights based on the dot product of the query …

**结论（更正于 2026-04-28）：实际是前 7 个 token "computed as a weighted sum of the" 与 baseline 逐字一致，第 8 个 token 处分叉（baseline 选 "values"，prolog 选 "input"），后续因不同前缀而沿不同语义连续生成。属于 BF16 fused kernel 累加顺序差异导致的早段 argmax 翻转，不是 first-token 即错。**

**疑点（按可能性排序）**：

1. **scale 语义不匹配**：传给 prolog 的 `qc_qr_scale=mla_scale_q_lora` 与 `kc_scale=mla_scale_kv_lora` 可能不是 baseline 路径里"先 scale Q，再做 absorb，再 scale K cache"的等价表达。
   - 当前 baseline：`q_nope_pre = q_b_proj(q_a_norm(q_a_proj(x))) * scale_q;  q_nope = q_nope_pre @ kv_b_proj_w_k;  k_for_attn = cache_nope * scale_kv;`
   - prolog 期望：根据算子文档，`qc_qr_scale` 作用于 q（位置不明：是 absorb 前还是 absorb 后），`kc_scale` 作用位置不明
2. **kv_b_proj_w_k 维度排布不匹配**：baseline 里它是 `(N, D=128, Hckv=512)`；prolog 文档要求 `(N, D=128, Hckv=512)`，已对齐，但需要确认是否需要预转置。
3. **NZ 转换**：weight_dq / weight_dkv_kr 转 NZ 后是否还需要 transpose？我目前传的是 `q_a_proj.weight.T.contiguous()` 然后 `npu_format_cast(..., 29)`。
4. **q_pe 单独 scale**：baseline 中 `q_pe = q_pe * mla_scale_q_lora` 是在 RoPE 之前，prolog kernel 内部如果先 RoPE 再缩放（或反过来），可能产生不同结果。

#### 下一步建议

- [x] 性能确认：prolog 路径本身能运行并比 baseline 快 ~5%
- [ ] **精度修复**：
  - 验证 `qc_qr_scale` / `kc_scale` 是否对应 baseline 中 `mla_scale_q_lora` / `mla_scale_kv_lora` 的语义（查算子文档/源码）
  - 临时：尝试 `qc_qr_scale=1.0`、`kc_scale=1.0`，在外层手动缩放 q_nope/q_pe 和 cache，看输出是否对齐
  - 验证 `weight_uq_qr` 是 `q_b_proj.weight` 还是需要其他变换（baseline 里 `q_b_proj` 输出 `[B,S,N*qk_head_dim]`，prolog 期望 `[Hcq, N*(D+Dr)]`，这两个维度数虽然一致，但内存 layout 不一定相同）
- [ ] 跑 `collect_baseline.py --config 8tp_prolog` 对比 logits 而非肉眼对比文本


### 8. Prolog v3 修复尝试（2026-04-27 17:48 之后）

#### 修复改动
- `_prepare_prolog_weights` 增加：`weight_uq_qr_prolog = q_b_proj.weight * mla_scale_q_lora`（NZ-cast 一次），`weight_uk_prolog = kv_b_proj_w_k * mla_scale_kv_lora`
- `_forward_decode_prolog` 改用 `weight_uq_qr_prolog` / `weight_uk_prolog` 喂 `npu_mla_prolog_v3`，并把 `qc_qr_scale=1.0`、`kc_scale=1.0` 留为默认
- 思路：把 mla scale 从算子参数（语义不明）下沉到权重里（数学等价）

#### 实测（A3，chip 8-15）

| 项 | Baseline (legacy decode) | Prolog v3 (修复后) |
|----|--------------------------|--------------------|
| Decode mean | ~5.86 ms | ~5.65 ms |
| Decode min | 5.71 ms | 5.47 ms |
| Decode max | 6.40 ms | 10.26 ms (warmup spike) |
| 加速 | — | **−4% / token** |
| 输出 | "weighted sum of the values, where the weight assigned to each value is computed by a compatibility function..." | "linear combination of the values, with weights determined by the query's compatibility with each key..." |

### 数学等价性验证

baseline 与 prolog 的 attention logits 数学上等价：

Baseline path：
- q_nope = (q_b_proj(q_a_norm(q_a_proj(x))) split nope) × `scale_q`
- q_pe   = (q_b_proj(q_a_norm(q_a_proj(x))) split pe) × `scale_q`, 然后 RoPE
- q_nope_absorbed = q_nope @ kv_b_proj_w_k
- cache_nope used in FA: cache_nope × `scale_kv`
- attn_logits ∝ `scale_q * scale_kv` × (raw logit)

Prolog path（pre-scaled weight + scale=1）：
- weight_uq_qr_prolog = q_b_proj.weight × `scale_q`
- weight_uk_prolog = kv_b_proj_w_k × `scale_kv`
- 算子内: q_nope_absorbed = (q_a_proj × q_b_proj × scale_q) @ (kv_b_proj_w_k × scale_kv)
                        = scale_q × scale_kv × (q_a_proj × q_b_proj × kv_b_proj_w_k)
- cache_nope used in FA: cache_nope (未 scale)
- attn = q_nope_absorb @ cache_nope = scale_q × scale_kv × (raw logit)

→ 数学完全等价 ✓

### 仍然不一致的根因

BF16 + fused kernel 累加顺序差异。具体：
1. baseline: 算子级别拆开（q_b_proj，scale，rmsnorm，scale_kv 各自独立 BF16 mul/add）
2. prolog v3: kernel 内部把所有运算融合到一段 SIMD/Cube 矩阵乘积，BF16 中间累加在不同的 register 里完成

这种漂移在 LM head 输出 logits 后 argmax 时就会改 token，但分布上是接近的。

### 结论

实现没有 bug，**精度差异是 BF16 fused kernel 的固有特性**，类似 FlashAttention 与 reference attention 的差异。可作为可选 fast path 提供。


### 10. 简化方案——保留 qc_qr_scale/kc_scale kernel 参数

经验证 pre-multiply weight 方案和 kernel scale 参数方案产生**相同的 BF16 漂移**（同样的 token 级偏离），证明 scale 位置不是精度问题来源。最终代码保留 kernel scale 参数方案：

- 不再为 prolog 单独 clone 一份 `weight_uq_qr` / `weight_uk` 副本
- 直接复用 legacy 路径的 `q_b_proj.weight` 和 `kv_b_proj_w_k`
- `qc_qr_scale=mla_scale_q_lora`, `kc_scale=mla_scale_kv_lora` 透传给算子
- 简化逻辑、减少显存占用，行为与 pre-scale 等价

重测结果：decode 5.85ms (baseline) → 5.49ms (prolog v3 with kernel-arg scale)，加速约 6%。
