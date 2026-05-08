## 模型信息

### 运行环境
- NPU 型号: Ascend 910B4 (Atlas A2)
- 单卡 HBM: 32 GB
- 部署卡数: 8 (单节点)
- 量化模式: BF16 (未量化)
- 执行模式: eager (未适配)

### 模型架构

**架构类型**: 多模态 MoE (Vision + Language MoE)

**模型名称**: Gemma-4-26B-A4B (google/gemma-4-26B-A4B)
- 总参数量: 26.5B, 活跃参数量/token: ~3.8B
- 权重大小: ~51.6 GB (BF16)
- 权重路径: /data2/models/gemma-4-26B-A4B

**Language Model (Text Decoder)**:
- 层数: 30 (25 sliding_attention + 5 full_attention)
- Hidden size: 2816
- 词表大小: 262144 (大词表)
- tie_word_embeddings: True (Embedding 与 LM Head 共享权重)

**Attention 结构 (双模式)**:
- Sliding Attention 层 (25 层): GQA, N_h=16, N_kv=8, head_dim=256, sliding_window=1024, RoPE theta=10000
- Full Attention 层 (5 层, 位于 layer 5/11/17/23/29): GQA, N_h=16, N_kv=2 (num_global_key_value_heads), global_head_dim=512, attention_k_eq_v=True (K=V, 无独立 v_proj), partial_rotary_factor=0.25, RoPE theta=1000000
- QK Norm: 有 (q_norm, k_norm)
- final_logit_softcapping: 30.0

**FFN 结构 (每层均有 Dense FFN + MoE)**:
- Dense MLP: gate_proj + up_proj + down_proj, intermediate_size=2112, activation=gelu_pytorch_tanh
- MoE: 128 experts, top_k=8, moe_intermediate_size=704, gate_up_proj fused
- Router: proj (H->E=2816->128) + per_expert_scale + scale
- 每层同时包含 Dense FFN 和 MoE (非交替式, 全部 30 层均有 experts 和 mlp)
- 每层有 6 个 LayerNorm (input_layernorm, post_attention_layernorm, pre_feedforward_layernorm, post_feedforward_layernorm, pre_feedforward_layernorm_2, post_feedforward_layernorm_2 等) + layer_scalar

**Vision Tower (SigLIP-like)**:
- 层数: 27
- Hidden size: 1152, head_dim=72, N_h=16, N_kv=16 (MHA)
- FFN intermediate: 4304
- patch_size: 16, pooling_kernel_size: 3
- 参数量: ~549M
- embedding_projection: 1152 -> 2816 (投影到 LM hidden_size)
- vision_soft_tokens_per_image: 280

**网络结构拆解**:
```
Image Input
  └─ Vision Tower (27 layers, MHA, ~549M params)
       ├─ Patch Embedder (Conv2D 16x16 -> 1152)
       ├─ Position Embedding
       ├─ Transformer Block x27 (MHA, FFN)
       └─ Embedding Projection (1152 -> 2816)
  └─ Soft Token Merge (280 tokens/image)

Text Input
  └─ Token Embedding (262144 x 2816, shared with LM Head)

Combined Token Sequence
  └─ Language Decoder Block x30
       ├─ Self-Attention
       │    ├─ Sliding Attention (25 layers): GQA, N_h=16, N_kv=8, D_h=256, window=1024
       │    └─ Full Attention (5 layers): GQA, N_h=16, N_kv=2, D_h=512, k_eq_v
       ├─ Dense FFN: gate+up+down (H=2816, H_ffn=2112)
       ├─ MoE: Router -> Top-8/128 experts (H=2816, H_moe=704)
       └─ Residual + LayerNorms (6 per layer) + layer_scalar
  └─ Final RMSNorm
  └─ LM Head (shared embedding, 2816 -> 262144)
```

**Prefill / Decode 分支差异**:
- Prefill: 处理多 token, sliding_attention 层使用 sliding_window=1024 因果 mask, full_attention 层使用全因果 mask; Vision 编码仅在 Prefill 执行
- Decode: 每步 1 token, sliding_attention 层只需最近 1024 个 KV, full_attention 层需全部历史 KV
- attention_k_eq_v (full_attention 层): K 和 V 共用投影, KV Cache 只需存 K (显存节省)

**特殊模块与注意事项**:
1. 双 Attention 模式 (sliding + full): sliding_window=1024 限制 KV 长度, full 层无限制但仅 5 层
2. attention_k_eq_v: full_attention 层无 v_proj, V=K, KV Cache 实现需特殊处理
3. 每层同时有 Dense FFN 和 MoE: 非标准结构, 不是 Dense FFN 和 MoE 交替, 而是每层都有
4. global_head_dim=512 vs head_dim=256: 两种 attention 层的 QKV 维度不同
5. partial_rotary_factor=0.25 (full_attention): 仅 25% 维度应用 RoPE
6. 大词表 262144: Embedding/LM Head 显存占比高 (~738M params, ~1.4 GB BF16)
7. Vision Tower 约 549M 参数, 不参与文本推理循环

**KV Cache 显存分析**:
- Sliding attention (25 层): 每 batch 最多缓存 1024 token, 2 * N_kv * D_h = 2 * 8 * 256 = 4096 bytes/token/layer (BF16), 总计 ~210 MB/batch
- Full attention (5 层): 缓存全部历史, 若 k_eq_v 仅存 K: N_kv_global * D_h_global = 2 * 512 = 1024 * 2 bytes = 2048 bytes/token/layer; 若存 K+V: 4096 bytes/token/layer
- S=4096, BS=4: sliding ~210MB + full ~42-84MB = ~252-294 MB

**参数量明细**:
| 模块 | 每层参数量 | 层数 | 小计 |
|------|-----------|------|------|
| Sliding Attention (Q+K+V+O) | 34.6M | 25 | 865M |
| Full Attention (Q+K+O, no V) | 49.0M | 5 | 245M |
| Dense FFN (gate+up+down) | 17.8M | 30 | 535M |
| MoE (128 experts) | 761.3M | 30 | 22,839M |
| Router + Norms + scalar | ~0.4M | 30 | 12M |
| Embedding (262144 x 2816) | 738M | 1 | 738M |
| Vision Tower | 549M | 1 | 549M |
| Final Norm | ~0.003M | 1 | ~0M |
| **估算总计** | | | **~25.8B** |
| **实际总计 (weight index)** | | | **26.5B** |

**单卡显存部署分析 (BF16, 无量化)**:
| 部署卡数 | 每卡参数显存 | 剩余显存 (32GB卡) | 可行性 |
|---------|------------|-----------------|--------|
| 1 | ~51.6 GB | 不可行 | 不可行 |
| 2 | ~25.8 GB | ~6.2 GB | 极紧张 |
| 4 | ~12.9 GB | ~19.1 GB | 可行 |
| 8 | ~6.5 GB | ~25.5 GB | 充裕 |

### 模型当前状态

**状态: 单卡框架适配完成, 需多卡并行化**

已完成项:
1. 完整的单卡框架适配代码 (text-only, Vision Tower 跳过):
   - `models/modeling_gemma4.py`: HF Gemma4 text decoder 移植, 含双模式 attention (sliding+full), k_eq_v, Dense FFN+MoE 并行, final_logit_softcapping, 连续 KV Cache + FA v1, 权重加载 (处理 model.language_model. 前缀映射和 tie_word_embeddings)
   - `models/configuration_gemma4.py`: 简化的 Gemma4TextConfig (从嵌套 config.json 的 text_config 读取)
   - `runner_gemma4.py`: 继承 ModelRunner, model_generate/input_prepare/output_process 完整实现
   - `infer.py` / `infer.sh`: 标准框架入口
   - `config/gemma4_rank_1_eager.yaml`: 单卡 eager 配置 (world_size=1)
   - `models/model_setting.py`: check_vars + update_vars
   - `requirements.txt` / `README.md`
2. NPU 前向推理验证通过 (tiny model test: Prefill + Decode 均 OK)
3. 代码可作为后续并行化改造的基础

需多卡:
- BF16 权重 ~51.6 GB, 超出单卡 32 GB HBM
- 推荐 8 卡部署 (EP 适合 128 experts)
- 后续进入阶段 1 parallel-impl 进行并行化改造

**最接近的仓库参考模型**:
- MoE 结构: qwen3-moe (MoE 路由/专家计算参考)
- 多模态: 暂无直接参考 (仓库以 LLM 为主)

### 部署基线

- 基线已采集 (2026-04-15)
- 配置: 8x Ascend 910B4, eager, BS=8, input_len=256, decode_steps=32
- Prefill: 312.51 ms
- Decode avg: 98.47 ms
- 输出: "A model is a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
- 基线文件: baseline/baseline_metadata.json

## 进度概览
| 阶段 | 结论 | 性能变化 |
|------|------|---------|
| 阶段 0 | 分析完成, 单卡框架适配完成 (需多卡) | 基线已采集: Prefill 312.51ms, Decode avg 98.47ms |
| 阶段 1 | 并行策略分析完成, 推荐方案 A (attn_tp=1 dense_tp=1 moe_tp=1 ep=8) | 实施完成 |
| 阶段 1 实施 | 8 卡并行化改造完成, Prefill+Decode 跑通, 输出正常 | Prefill 313ms, Decode avg 97ms |
| 阶段 2 | KVCache 连续缓存 + FA v1 BSH, sliding 层 sparse_mode=4 修复 | Prefill 310ms (-0.7%), Decode 97ms (-1.1%), 精度无损 |
| 阶段 3 | 融合算子: npu_add_rms_norm + npu_rotary_mul + npu_moe_gating_top_k_softmax | Prefill 307ms (-1.8%), Decode 92ms (-6.4%), 精度无损 |
| 阶段 4 | GE graph Decode fullgraph, local-expert MoE, torchair FA, RoPE 预计算 | Decode 18ms (-81.5%), Prefill 303ms (-3.2%). Prefill 首 token 对齐 eager; Decode 因手动 GEGLU + local-expert 路径与 eager 有 BF16 级数值偏差, 输出语义连贯但文本不完全一致 |
| 阶段 5 | Runner → new framework 迁移：模块签名切到 `(config, infer_config, comm_manager, prefix)`, 基类 `nn.Module`, KV cache 用 `cache_unit` 暴露给框架自动分配, forward 走 `ForwardMetaData`, YAML 切 4 段式 schema, `support_models.py` 注册 | 8 卡 ge_graph 稳态: prefill 189ms / decode ~14.7ms/token（与阶段 4 的 303ms / 18ms 持平或略优，差异在 CANN 环境波动范围内）. 精度：post-refactor 内部 eager↔ge_graph 的 prefill cos=0.99998951、top20 一致，与阶段 4 baseline 行为一致（decode 在 BF16 下有 token 级漂移属已知）. 算法语义未改，仅接口层重构。 |
| 阶段 6 (A3 验证) | 不改任何 framework / model 代码，只把 yaml 里的 `model_path` 指到 A3 上的实际权重，重跑 ge_graph + npugraph_ex 两套配置 | A3 8 卡稳态：**ge_graph prefill 76.4ms / decode 10.20ms**，**npugraph_ex prefill 102.3ms / decode 11.59ms**。同代码对比 A2 ge_graph (189ms / 14.7ms) prefill 提升 2.47×、decode 提升 1.44×。两路径输出末态 token 分布与 A2 一致，无回归。 |
| 阶段 6.1 (MoE MC2 启用) | EP decode 默认走 `mc2`（`npu_moe_distribute_dispatch_v2 + combine_v2`），与 qwen3_moe / longcat-flash 路径一致；唯一约束是 dispatch_v2 要求 `experts_per_rank ≤ 24`，gemma-4 EP=8 时 16/rank 满足。增加 `moe_ep_decode_mode` 开关，experts_per_rank > 24 自动回退 `local_experts`（如 longcat-flash-lite EP=8 / 32 expert/rank） | A3 BS=8 实测：mc2 与 local_experts 数值一致；mc2 ge_graph 10.5 ms / npugraph_ex 11.73 ms，local_experts 同 mode 10.2 ms / 11.59 ms。当前 BS=8 量级下两条路径接近（小消息上 AllReduce 比 2× AllToAll 略快），mc2 优势在 BS↑ 或 prefill 链路才会显现。默认走 mc2，跟 qwen3_moe / longcat-flash 风格保持一致 |
| 阶段 7 (BSND-shim 三模式跑通) | rebase 后 framework drift（f0b0fde 04-26 SchedulerConfig schema + 2c2336a 04-30 MR !315 packed 1D + chat wrap）暴露，BSND-shim 过渡方案落地：ForCausalLM 入口 reshape input_ids/position_ids，TextModel + Attention 内 kv_len/actual_seq_lengths_kv 局部 shim（packed [B*S]→BSND [B]），RoPE table size 加 max_new_tokens 修早先缩水回归，yaml 加 `moe_ep_decode_mode: local_experts` 绕开 A2 上 MC2 dispatch_v2 hang，npugraph_ex 下 actual_seq_lengths_kv 改读 framework 提供的 host list 字段（torch.compile dynamo schema 严格要求 SymInt[]），tokenizer_config.json 加最简 chat_template；transformers 4.55.4 兼容并行验证。TND 全量迁移作独立工单后续推进 | A2 8 卡三模式跑通，输出文本主题与基线一致：eager prefill 119.70 / decode avg 107.83 ms（local_experts 路径），ge_graph prefill 123.61 / decode avg 14.75 ms（与阶段 5 的 189/14.7 ms 持平且 prefill 更快），npugraph_ex prefill 215.20 / decode avg 22.00 ms |

