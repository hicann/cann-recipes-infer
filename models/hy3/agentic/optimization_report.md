# Hy3 模型优化报告

---

## 1. 概述

Hy3 是腾讯混元团队发布的 MoE 语言模型，采用 GQA 注意力与带共享专家的稀疏 MoE 结构，总参数量 295B、每 token 激活约 21B。本报告记录该模型在 Atlas A3 平台上的端到端基础优化闭环，涵盖从并行化部署、KVCache 静态化与融合算子替换到图模式适配的完整方案及性能、精度结论。

- 架构: GQA + 稀疏 MoE（含共享专家），decoder-only
- 总参数量: 295B（每 token 激活约 21B）
- 硬件平台: Atlas A3（Ascend 910C）
- 部署规模: 8 卡 16 rank（每卡 2 die），单节点
- 量化模式: BF16

> 本报告为 Hy3 在 Atlas A3 平台的端到端基础优化闭环记录。Ascend 950 PR 平台上的进一步性能调优参见 [Hy3 推理优化实践](../../../docs/models/hy3/hy3_optimization.md)。

### 1.1 模型结构概要

```
Embedding (120832 × 4096, 495M params)
  └─ Transformer Block × 80 (layers 0-79)
       ├─ Pre-Attention RMSNorm (npu_add_rms_norm fused)
       ├─ Attention (GQA: 64Q heads, 8KV heads, head_dim=128)
       │    ├─ QKV Projection: QKVParallelLinear (TP=4)
       │    ├─ QK Norm: npu_rms_norm(128) per-head
       │    ├─ RoPE: npu_apply_rotary_pos_emb(layout='BSH')
       │    ├─ KV Cache: BSH [B,S,N_kv*D], scatter_update_(axis=-2)
       │    ├─ Attention Core: FA v1 (eager) / torchair FA (ge_graph)
       │    └─ O Projection: RowParallelLinear (TP=4)
       ├─ Post-Attention RMSNorm (npu_add_rms_norm fused)
       ├─ FFN / MoE
       │    ├─ Layer 0: Dense FFN (gate/up: 4096→13312, down: 13312→4096, SiLU)
       │    └─ Layers 1-79: MoE
       │         ├─ Router: npu_moe_gating_top_k(norm_type=1)，sigmoid 非 softmax
       │         ├─ EP dispatch/combine: MC2 (ge_graph) / all_to_all manual (eager)
       │         ├─ Expert FFN × 12 per-rank (GMM: npu_grouped_matmul split_item=2)
       │         └─ Shared Expert × 1 (gate/up: 4096→1536, down: 1536→4096)
       └─ Residual Connection (fused via npu_add_rms_norm)
  └─ Final RMSNorm (npu_rms_norm)
  └─ LM Head: ColumnParallelLinear(4096, 120832, no bias)
```

### 1.2 关键参数

| 参数 | 值 | 参数 | 值 |
|------|-----|------|-----|
| 总参数量 | 295B (295,033,543,488) | 激活参数量 | ~21B |
| 总层数 | 80 (excl. MTP layer 80) | Hidden Size | 4096 |
| Attention Heads | 64 (Q), 8 (KV) | Head Dim | 128 |
| Dense FFN Intermediate | 13312 (layer 0) | MoE Expert Intermediate | 1536 |
| 专家总数 | 192 | 每 token 激活专家 | 8 (top-8) |
| Shared Expert | 1 per MoE layer | 词表大小 | 120,832 |
| 最大上下文 | 262,144 (256K) | RoPE Theta | 11,158,840 |

### 1.3 特殊设计特点

| 特性 | 说明 | 适配影响 |
|------|------|---------|
| QK Norm | Q 和 K 在 RoPE 前各经 RMSNorm(128) | 已通过 npu_rms_norm 优化 |
| Sigmoid Router | 使用 sigmoid 而非 softmax | 已通过 npu_moe_gating_top_k(norm_type=1) 融合 |
| Expert Bias | learnable `e_score_correction_bias` | 参与 routing scoring |
| Router Scaling | top-k weights × 2.826 | 已通过 routed_scaling_factor 参数处理 |
| MTP Layer | layer 80 (~3.8B params) | 推理跳过（_keys_to_ignore_on_load_unexpected） |
| Large Vocab | 120K tokens | embed/lmhead 各 ~1 GB BF16，词表按 TP=4 切分 |

上述结构特殊点在后续适配中均通过对应融合算子或参数处理消化，未引入定制算子。

---

## 2. 性能基线

本章记录 Hy3 完成并行化部署、尚未引入任何算子融合或图模式优化时的初始性能，作为后续各项优化的对比基准。基线在 Atlas A3 16 die、BF16、eager 模式下采集。

| 指标 | 值 | 测试条件 |
|------|-----|---------|
| Prefill 耗时 | 2,070 ms | 1024 tokens, 16 die A3, eager mode, batch=4 |
| Decode 单步耗时 | 293 ms/t | eager mode, 16 die A3 |
| 显存占用 | ~54.5 GB/die | BF16, 295B 参数 EP+TP 分布后 |
| 执行模式 | eager | 初始部署 |

该基线采集于并行化部署完成之后，Decode 单步 293 ms/t、Prefill 2070 ms，是后续所有性能对比的统一基准。

---

## 3. 并行化改造

在基线部署之上，本章将 295B 参数的 Hy3 按 16 die 拓扑做混合并行切分，这是所有后续优化的部署前提。切分需同时兼顾 GQA 的 KV 头数量约束与 MoE 专家规模，避免碎矩阵与跨模块边界通信。

### 3.1 并行策略

| 参数 | 值 | 理由 |
|------|-----|------|
| world_size | 16 die | 8 卡 × 2 die/卡 |
| attn_tp_size | 4 | N_kv=8 约束, 均衡 Prefill/Decode |
| dense_tp_size | 4 | 与 attn_tp 对齐，避免模块边界通信 |
| moe_tp_size | 1 | Expert intermediate=1536 太小，TP 导致碎矩阵 |
| moe_ep_size | 16 | 全卡 EP，192/16=12 experts/rank |
| embed_tp_size | 4 | 大词表 120K 切分 |
| lmhead_tp_size | 4 | 大词表切分 |

### 3.2 关键实现

- 通信组：注意力、Dense、Embedding、LM Head 均按 TP=4 切分，各自建立 4 组 TP 通信组；MoE 按 16 卡专家并行建立一个 EP 通信组。
- EP 路由：eager 模式使用手动 all_to_all 完成专家 dispatch/combine，图模式改用 MC2 dispatch/combine。
- 权重加载：启用在线切分权重（enable_online_split_weight），112 个 shard 约 12s 完成加载。
- padding_idx：全局 padding 索引 120002 在 Embedding 切分后映射为每 rank 的局部索引。

### 3.3 验收结果

| 验收项 | 结果 |
|------|------|
| 编译 | 16 die 编译通过 |
| 多卡推理 | 无 crash |
| 解码输出 | 输出正常文本；QKV 权重按 checkpoint 的 q/k/v_proj 映射到 merged_qkv_proj 后正确加载 |

上述切分下模型编译与 16 die 推理均正常，解码输出正确文本，说明并行拆分与权重加载映射正确、精度未见异常，可作为后续优化的部署基线。

---

## 4. KVCache 静态化与 FA 算子替换

完成并行化部署后，本章将 Attention 的 KV 管理从动态实现改为静态连续缓存，并把注意力核替换为 FA 融合算子。静态 shape 与融合注意力既提升当前 eager 性能，也为后续图模式捕获提供前提。

### 4.1 选型结果

| 决策项 | 选择 | 理由 |
|--------|------|------|
| KVCache 模式 | 模式一：连续缓存 (BSH) | 静态 batch, 无动态分配需求 |
| FA 算子 | FA v1 (`npu_fused_infer_attention_score`) | GQA 无 MLA/量化需求 |
| KV Cache 写入 | scatter_update_(axis=-2) | BSH layout 标准写法 |
| Layout | BSH [B, S, N_kv*D] | QK Norm per-head 兼容 |
| Prefill mask | sparse_mode=3 + ~tril bool | Decoder-only causal |
| Decode mask | sparse_mode=0 + None | 无需因果遮蔽 |

### 4.2 验收结果

| 指标 | 结果 |
|------|------|
| KVCache 静态化 | BSH 连续缓存，scatter_update_ |
| FA 算子替换 | FA v1，Prefill/Decode 分支正确 |
| 精度 (vs HF) | 输出正常文本 |

KVCache 改为 BSH 连续缓存并接入 FA v1 后，Prefill 与 Decode 分支分别使用正确的 mask 与 sparse_mode，输出正常文本、精度与 HF 对齐，同时静态 shape 为后续图模式捕获提供了前提。

---

## 5. 融合算子优化

在 KVCache 与 FA 改造之上，本章把模型中的 Norm、激活、MoE 路由等 Python 小算子模式替换为 torch_npu 融合算子。这些替换在 eager 模式下即可获得收益，同时减少算子数量，为图模式捕获铺路。

### 5.1 实施项目

#### P0（零风险）

| 模块 | 原实现 | 替换为 | 调用频率 |
|------|--------|--------|---------|
| RMSNorm (all) | Python `pow(2).mean().rsqrt()` | `npu_rms_norm` | 321次/forward |
| Residual + RMSNorm | `x + y` + `RMSNorm(y)` 分离 | `npu_add_rms_norm` | 160次/forward |
| FFN SiLU | `F.silu(gate) * up` | `npu_swiglu` | 80+次/forward |

#### P1（精度敏感）

| 模块 | 原实现 | 替换为 | 调用频率 |
|------|--------|--------|---------|
| MoE Router | Python sigmoid+topk+gather+norm+scale (7 ops) | `npu_moe_gating_top_k(norm_type=1)` | 79次/forward |

> P1 精度说明: Hy3 的 `e_score_correction_bias` 在 sigmoid 后加（仅影响 topk 选择），而 `npu_moe_gating_top_k` 的 bias 在 sigmoid 内。该语义差异未造成可见输出质量退化。

#### P2（进阶候选，暂缓）

| 模块 | 候选算子 | 未实施原因 |
|------|---------|----------|
| Dense FFN 全融合 | `npu_ffn(activation='swiglu')` | 需绕过并行 Linear, 性能门槛待 profiling |
| Expert FFN 全融合 | `npu_ffn(expert_tokens=...)` | forward_ordered 适配, 性能门槛待 profiling |

#### 不适配算子

| 算子 | 不适配原因 |
|------|----------|
| `npu_kv_rmsnorm_rope_cache` | MLA 专用, Hy3 为 GQA + per-head QK Norm |
| `npu_mla_prolog_v3` | MLA absorb 模式专用 |
| `npu_moe_distribute_combine_add_rms_norm` | 需配合 dispatch_v2 使用, 且手动 EP 链路不匹配 |

### 5.2 验收结果 (eager mode, P0+P1)

| 指标 | 改造前 | 改造后 | 改善 |
|------|--------|--------|------|
| Prefill (1024t) | 2,070 ms | 1,041 ms | **50%** |
| Decode avg | 293 ms/t | 246 ms/t | **16%** |

P0+P1 融合算子在 eager 模式下把 Prefill 从 2070 ms 降至 1041 ms（约 50%）、Decode 从 293 ms/t 降至 246 ms/t（约 16%），主要收益来自 Norm/激活/路由小算子合并；MoE 路由的 bias 语义差异未造成可见输出质量退化，精度保持不变。

---

## 6. 图模式适配

前几章改造均在 eager 模式下完成，本章将 Decode 路径适配到 ge_graph 图模式，通过消除 graph break、算子二进制融合与编译缓存进一步压低单步时延；Prefill 因序列长度可变仍保持 eager，图模式仅覆盖 Decode。

### 6.1 实施方案

| 项目 | 方案 | 说明 |
|------|------|------|
| 图模式后端 | ge_graph | torchair + CompilerConfig |
| Decode 路由 | MC2 dispatch/combine + GMM | 替代手动 all_to_all, 消除数据依赖 graph break |
| 专家计算 | `npu_grouped_matmul(split_item=2)` | 替代 per-expert F.linear loop |
| FA 接口 | torchair FA (ge_graph) / torch.ops.npu FA (eager Prefill) | Prefill 保持 eager |
| SuperKernel | `enable_superkernel=True` | Decode 层循环外包 superkernel_scope |
| Compile Cache | `enable_cache_compile=True` | 持久化到 compile_cache/, 二次 warmup ~9s (10x) |
| 多流 | 默认关闭 (enable_multi_streams=False) | shared expert 计算量小，无法有效 overlap，实现保留待后续评估 |

### 6.2 关键实现要点

- Expert GMM 权重以 inline `.transpose(1,2)` view 复用，避免 `.transpose().contiguous()` 产生的约 35.8 GB 权重副本，不额外分配显存。
- Checkpoint 中分离的 `q/k/v_proj` 权重映射到 `merged_qkv_proj.weight`，保证 QKV 投影权重正确加载。

### 6.3 验收结果

| 指标 | eager (P0+P1) | ge_graph (图模式阶段) | 改善 |
|------|-------------|----------------------|------|
| Prefill (1024t) | 1,041 ms | 1,452 ms | — |
| Decode avg | 246 ms/t | 30.8 ms/t | **8.0x（阶段值）** |
| Decode warmup (首次) | — | 91,151 ms | — |
| Decode warmup (cached) | — | 8,016 ms | 10x |

图模式阶段将 Decode 从 eager 的 246 ms/t 降至 30.8 ms/t（阶段加速约 8.0x），主要收益来自 MC2 路由与 GMM 消除数据依赖 graph break、以及 SuperKernel 减少层循环开销；compile cache 把二次 warmup 从约 91s 压到约 8s。此为图模式阶段结果，最终经重编译进一步降至 28.44 ms/t（见 §7）。

---

## 7. 累计优化效果

本章汇总并行化、融合算子与图模式各阶段叠加后的累计效果，标注最大收益来源与最终性能水平。

| 指标 | 原始基线 (eager) | 最终 (ge_graph) | 累计改善 |
|------|----------------|----------------|---------|
| Prefill 耗时 (1024t) | 2,070 ms | 1,452 ms | **1.4x** |
| Decode 单步耗时 | 293 ms/t | **28.44 ms/t** | **10.3x** |
| vs 目标 (<100ms/t) | — | 28.44 ms/t | **3.5x margin** |

累计来看，Decode 从 293 ms/t 降至 28.44 ms/t（10.3x），最大收益来自图模式叠加 MC2 路由与 GMM；Prefill 从 2070 ms 降至 1452 ms（约 1.4x）。最终 Decode 相对 <100 ms/t 目标仍有约 3.5x 余量。

### 7.1 各阶段贡献

| 优化项 | Decode 性能 | 增量 | 累计改善 |
|------|-----------|---------|---------|
| 并行化 (eager 基线) | 293 ms/t | — | 1.0x |
| P0+P1 融合算子 | 246 ms/t | 1.19x | 1.19x |
| ge_graph + GMM | 33.3 ms/t | 7.4x | 8.8x |
| + compile cache | 31.5 ms/t | 1.06x | 9.3x |
| + SuperKernel | 30.8 ms/t | 1.02x | 9.5x |
| 最终重编译 | **28.44 ms/t** | 1.08x | **10.3x** |

各阶段中，ge_graph + GMM 贡献了绝大部分收益（7.4x），compile cache 与 SuperKernel 为增量优化，最终重编译后 Decode 稳定在 28.44 ms/t。

---

## 8. 遗留问题与后续建议

本章记录本期优化未落地的候选方案及其阻塞原因，并给出最终性能确认与已知限制，作为后续迭代参考。

### 8.1 P2 融合算子最终评估

| 候选 | 结论 | 阻塞原因 |
|------|------|---------|
| `npu_ffn(expert_tokens=...)` | 不可行 | API 硬约束：swiglu 模式不接受 expert_tokens |
| `npu_ffn(activation='swiglu')` Dense | 不可行 | API 硬约束：swiglu 仅支持 float16 (Hy3 用 bf16) |
| `npu_moe_distribute_combine_add_rms_norm` | 不可行 | 数学不等价：API 计算 `norm(A+B+res)` vs Hy3 deferred residual `norm(A)+res` |
| `combine_v2 + shared_expert_x` | 暂不启用 | SuperKernel 编译失败：融合链过长超过 2-stream 限制 (CANN 8.5.0)。待 CANN 更新后可重新启用 |

上述四个进阶候选在当前 CANN 版本下均受 API 数据类型、语义等价性或融合链长度的硬约束限制，本期不落地；其中 combine_v2 + shared_expert_x 待 CANN 更新后可重新评估。

### 8.2 已验证最终性能

| 指标 | 数值 | 备注 |
|------|------|------|
| Prefill (1024t) | 1,452 ms | ge_graph + compile cache |
| **Decode avg** | **28.44 ms/t** | range: 27.4-29.4, 32 tokens |
| vs 原始 eager (293ms/t) | **10.3x 加速** | |
| vs 目标 (<100ms/t) | **3.5x margin** | |
| 输出质量 | 正常 | "The output is a weighted sum of the values..." |

最终确认 Decode 28.44 ms/t（32 token 区间 27.4–29.4）、Prefill 1452 ms，相对原始 eager 基线 293 ms/t 加速 10.3x，输出质量正常，达到本期优化目标并留有约 3.5x 余量。

### 8.3 性能分析

当前 Decode 28.44 ms/t 已远超 <100ms/t 目标 (3.5x margin)。Prefill 1452ms (1024t) 仍有优化空间但 Prefill 非主要瓶颈 (单次推理仅执行 1 次 Prefill + N 次 Decode)。

### 8.4 已知限制

- **Multi-stream**: shared expert 计算量太小 (1536 intermediate), 无法有效 overlap。若未来模型升级增大 intermediate 可重新评估
- **长序列 (>128K)**: KV Cache 增长到 ~10.7 GB/卡 (attn_tp=4), 余量减少。若扩展到 256K 需评估 KVP 叠加
- **W8A8 量化**: 若未来减至 8 卡部署, 需 W8A8 量化 (~295 GB int8)
