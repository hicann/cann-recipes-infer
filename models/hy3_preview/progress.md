## 模型信息

### 运行环境
- NPU 型号: Ascend 910C (Atlas A3)
- 单卡 HBM: 64 GB (65536 MB)
- 部署卡数: 16
- 量化模式: BF16（16×64GB=1024GB，295B BF16 参数 590 GB 可行）
- 执行模式: **ge_graph**（图模式 + MC2 dispatch/combine + GMM）

### 模型架构
- 模型名称: Hy3-preview，由 Tencent Hy Team 开发
- 架构类型: MoE Transformer（Dense + Sparse MoE with Shared Expert）
- 模型路径: models/hy3_preview（已完成适配）
- 权重路径: /data/models/hy3_preview
- HF 源码: /tmp/transformers_hy3_check/src/transformers/models/hy_v3/

#### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 总参数量 | 295B (295,033,543,488) | BF16 约 590 GB |
| 激活参数量 | ~21B | 每 token 计算用 |
| MTP 层参数 | ~3.8B | layer 80，标准推理不使用 |
| 总层数 (excl. MTP) | 80 | layers 0-79 |
| MTP 层数 | 1 | layer 80 |
| Hidden Size | 4096 | |
| Attention Heads | 64 (Q), 8 (KV) | GQA，ratio=8x |
| Head Dim | 128 | |
| Dense FFN Intermediate | 13312 | 仅 layer 0 |
| MoE Expert Intermediate | 1536 | layers 1-79 |
| 专家总数 | 192 | 每 MoE 层 |
| 每 token 激活专家 | 8 | top-8 routing |
| Shared Expert | 1 | 每 MoE 层，intermediate=1536 |
| 词表大小 | 120,832 | |
| 最大上下文长度 | 262,144 (256K) | |
| RoPE Theta | 11,158,840 | default RoPE |
| 激活函数 | SiLU | |
| 归一化 | RMSNorm, eps=1e-5 | |

#### 网络结构拆解

```
Embedding (120832 × 4096, 495M params)
  └─ Transformer Block × 80 (layers 0-79)
       ├─ Pre-Attention RMSNorm
       ├─ Attention (GQA)
       │    ├─ QKV Projection: QKVParallelLinear (TP=4)
       │    ├─ Q Norm: npu_rms_norm(128) per-head
       │    ├─ K Norm: npu_rms_norm(128) per-head
       │    ├─ RoPE: npu_apply_rotary_pos_emb(layout='BSH')
       │    ├─ KV Cache: BSH [B,S,N_kv*D], scatter_update_(axis=-2)
       │    ├─ Attention Core: FA v1 (eager) / torchair FA (ge_graph)
       │    ├─ O Projection: RowParallelLinear (TP=4)
       │    └─ Residual: npu_add_rms_norm fused
       ├─ Post-Attention RMSNorm
       ├─ FFN / MoE
       │    ├─ Layer 0: Dense FFN (gate: 4096→13312, up: 4096→13312, down: 13312→4096)
       │    └─ Layers 1-79: MoE
       │         ├─ Router: Linear(4096, 192), sigmoid activation
       │         ├─ Expert Bias: 192-dim learnable bias (e_score_correction_bias)
       │         ├─ Top-K Selection: top-8 with router_scaling_factor=2.826
       │         ├─ Expert FFN × 192 (each: gate+up via packed gate_up_proj [3072,4096], down [4096,1536])
       │         ├─ Shared Expert FFN (gate: 4096→1536, up: 4096→1536, down: 1536→4096)
       │         └─ Combine: (routed + shared expert), optional fp32 accumulation
       └─ Residual: add input
  └─ Final RMSNorm
  └─ LM Head: Linear(4096, 120832, no bias), not tied with embedding
```

#### Prefill / Decode 分支差异

已适配为 NPU FA v1 融合算子，Prefill/Decode 分支通过 `q_len == 1` 检测：
- **Prefill**: `npu_fused_infer_attention_score` + `sparse_mode=3` (causal) + `atten_mask=~tril(2048,2048)` bool
- **Decode**: `npu_fused_infer_attention_score` + `sparse_mode=0` (dense) + `atten_mask=None`
- **ge_graph Decode**: torchair FA 接口 + MC2 dispatch/combine + GMM (`npu_grouped_matmul` split_item=2)
- KV Cache: BSH layout, `scatter_update_(axis=-2)` 写入

#### 关键模块特性

| 特性 | 说明 |
|------|------|
| QK Norm | Q 和 K 在 RoPE 前分别通过 RMSNorm(128)，这是区别于标准 GQA 的特殊设计 |
| Sigmoid Router | 使用 sigmoid 而非 softmax，配合 expert_bias 和 router_scaling_factor(2.826) |
| Expert Bias | 每 MoE 层有独立学习的 192-dim bias (`e_score_correction_bias`)，加到 sigmoid 分数上 |
| Router Scaling | top-k weights 归一化后额外乘以 `router_scaling_factor=2.826` |
| FP32 Combine | 可选：shared expert 输出与 routed expert 输出在 fp32 下累加（`enable_moe_fp32_combine`） |
| MTP Layer | layer 80 (num_nextn_predict_layers=1)，标准推理跳过（`_keys_to_ignore_on_load_unexpected` 包含 layer 80） |
| Layer 0 Dense | first_k_dense_replace=1，第一层使用 Dense FFN 而非 MoE |
| Large Vocab | 120K 词表，embed/lmhead 各占 495M 参数（~1 GB BF16），需独立 TP 切分 |

#### 参数分布

| 模块 | 参数量 | 占比 | BF16 显存 |
|------|--------|------|----------|
| Embedding | 494,927,872 | 0.17% | 0.94 GB |
| Attention (80层) | 6,039,818,240 | 2.05% | 11.25 GB |
| Dense FFN (layer 0) | 163,577,856 | 0.06% | 0.30 GB |
| MoE Experts (79层×192专家) | 287,839,632,192 | 97.56% | 536.25 GB |
| RMSNorm (全模型) | 659,456 | <0.01% | ~0 GB |
| LM Head | 494,927,872 | 0.17% | 0.94 GB |
| **总计** | **295,033,543,488** | **100%** | **~590 GB** |

> 关键观察：97.6% 的参数集中在 MoE 专家中，这意味着 **EP（Expert Parallelism）是部署的必要条件**。

### 模型当前状态

- **状态**: **可运行**（16 die 多卡部署，ge_graph 图模式）
- **部署配置**:
  - 16 die (8 卡 × 2 die/卡) Atlas A3 (Ascend 910C)
  - parallel_config: attn_tp=4, dense_tp=4, moe_ep=16, embed_tp=4, lmhead_tp=4
  - 执行模式: ge_graph + MC2 dispatch/combine + GMM (npu_grouped_matmul split_item=2)
  - 量化: BF16, enable_online_split_weight=True
  - enable_superkernel=True, enable_cache_compile=True
- **已适配组件**:
  1. modeling_hy_v3.py — 完整 NPU 模型实现 (FA v1 + RoPE BSH + EP routing + MC2 + GMM)
  2. runner_hy3.py — HYV3Runner (prefill/decode 分离调用, graph_compile, SuperKernel)
  3. infer.py / infer.sh — 推理入口
  4. config/hy3_bf16.yaml — 16 die 并行配置
- **baseline**: baseline_metadata.json 未正式采集（待补采）

### 权重命名差异（Checkpoint vs Code）

| 模块 | Checkpoint 命名 | 建模代码命名 | 说明 |
|------|----------------|-------------|------|
| 专家 gate/up | `experts.{id}.gate_proj.weight`, `experts.{id}.up_proj.weight` | `gate_up_proj` (packed [2×1536, 4096]) | `@use_experts_implementation` 处理转换 |
| Shared Expert | `shared_mlp.gate_proj.weight` 等 | `shared_experts.gate_proj` | 名称不同 |
| Router | `router.gate.weight` | `gate.weight` | 名称结构不同 |
| Expert Bias | `expert_bias` | `e_score_correction_bias` | 参数名不同 |

适配时需处理这些命名映射。

### 仓库参考模型

| 参考模型 | 相似点 | 差异 |
|---------|--------|------|
| **qwen3-moe** | MoE + GQA + Shared Expert + 大规模 | Hy3-preview 总参数量更大 (295B vs 235B)，有 QK Norm，sigmoid router |
| **deepseek-r1** | MoE + 大规模部署 | R1 使用 MLA Attention（非 GQA），架构差异大 |
| **kimi-k2-thinking** | 大 MoE 模型 | K2 使用 MLA + DSA，架构差异大 |

最接近的参考模型是 **qwen3-moe**，两者均使用 GQA Attention + MoE + Shared Expert 架构。

## 进度概览
| 阶段 | 结论 | 关键措施 | 最终性能 |
|------|------|---------|---------|
| 阶段 0: 模型分析 | **完成** | 架构分析、参数拆解、硬件确认 | — |
| 阶段 1: 并行化 | **通过** | attn_tp=4, dense_tp=4, moe_ep=16, embed/lmhead_tp=4 | Prefill 2.07s, Decode 293ms/t (eager) |
| 阶段 2: KVCache + FA | **通过** | FA v1 (BSH), 连续缓存, scatter_update_, sparse_mode=3/0 | 与 qwen3-moe 对齐 |
| 阶段 3: 融合算子 | **通过** (P0+P1) | npu_rms_norm, npu_add_rms_norm, npu_swiglu, npu_moe_gating_top_k | Decode 16%↓ (eager) |
| 阶段 4: 图模式 | **通过** | ge_graph + MC2 dispatch/combine + GMM + compile cache + SuperKernel | Decode 33→30.8ms/t |
| 多流 (shared expert) | **退化** | enable_multi_streams=True, shared expert 太小无法 overlap | 32.0ms/t（代码保留, flag 关闭） |
| **阶段 3 P2** | **已评估** | npu_ffn 因 API 硬约束不可行；shared_expert_x 与 SuperKernel 冲突已回退 | 待 CANN 更新后启用 |
| **累计最终** | **稳定** | 16die A3 ge_graph + SuperKernel + compile cache | Prefill 1452ms (1024t), **Decode 28.44ms/t**, **10.3x** vs 原始 eager |

<!-- ===== 以上为常驻区，不清除 ===== -->
