# Gemma-4-26B-A4B 模型优化报告

> 初版生成: 2026-04-15（基于 Atlas A2 / 910B4，32 GB HBM）
> 更新: 2026-04-27（追加 Atlas A3 / 910 A3 8 卡迁移性能数据，见第 7.1 节）

---

## 1. 模型概况

| 项目 | 内容 |
|------|------|
| 模型名称 | Gemma-4-26B-A4B (google/gemma-4-26B-A4B) |
| 架构类型 | 多模态 MoE，本次仅适配 Language Decoder（跳过 Vision Tower） |
| 总参数量 | 26.5B（活跃 ~3.8B/token，128 experts top-8） |
| 权重 | BF16 ~51.6 GB |
| 部署 | 8 卡，EP=8，embed/lmhead_tp=8，attn_tp=1 |
| 执行模式 | eager（Prefill）+ GE graph（Decode） |

### 架构特点

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

关键特殊点：双模式 Attention 导致 KV cache 维度异构；k_eq_v 使 full 层无 v_proj；QK RMSNorm 使 FA scale=1.0。

---

## 2. 性能基线

| 指标 | 值 | 条件 |
|------|-----|------|
| Prefill | 312.51 ms | BS=8, seq_len=256, 8 卡 eager |
| Decode | 98.47 ms | BS=8, 8 卡 eager |

---

## 3. 阶段 1：并行化改造

| 配置 | 值 | 理由 |
|------|-----|------|
| attn_tp | 1 | full_attention N_kv=2 限制 TP 上限；DP=8 最大化吞吐 |
| dense_tp | 1 | intermediate=2112 过小，TP 碎片化无收益 |
| EP | 8 | 128 experts / 8 = 16/rank |
| embed/lmhead_tp | 8 | 大词表 262144 切分 |

精度: 通过 | Prefill 313ms, Decode 97ms

---

## 4. 阶段 2：KVCache + FA

| 项目 | 内容 |
|------|------|
| KVCache | 连续缓存，按层异构（sliding kv_dim=2048, full kv_dim=1024） |
| FA | FA v1, BSH, scale=1.0 |
| 改动 | sliding 层 Prefill sparse_mode=4 + pre_tokens=1024 |
| PA 不采用 | 双 Attention head_dim 不同（256/512），需两套 block_table，复杂度高于收益 |

精度: 与基线一致 | Prefill 310ms, Decode 97ms

---

## 5. 阶段 3：融合算子

### 已实施

| 模块 | 替换为 | 频次/step |
|------|--------|----------|
| Residual + Norm | `npu_add_rms_norm` | 60 |
| Sliding RoPE (25 层) | `npu_rotary_mul` | 50 |
| MoE Router | `npu_moe_gating_top_k_softmax` | 30 |

### 不适配项

| 模块 | 候选算子 | 阻塞原因 |
|------|---------|---------|
| Dense FFN GEGLU | `npu_gelu_mul` | dim 上限 1024，需 4224 |
| Dense FFN GEGLU | `npu_ffn` geglu | 不支持 bf16 |

精度: 无损 | Decode: 97 → 92ms (-5.4%)

---

## 6. 阶段 4：图模式适配

### 6.1 方案

| 项目 | 内容 |
|------|------|
| 后端 | GE graph (torchair), fullgraph=True, Decode only |
| MoE | Decode: local-expert + AllReduce; Prefill: double_routing |
| FA | Decode: torchair FA; Prefill: torch.ops.npu FA |
| GEGLU | Prefill: npu_geglu; Decode: 手动 F.gelu（图模式兼容） |

**MoE 图模式方案**：EP 场景下 double_routing 的 AllToAll split sizes 是 data-dependent 的，无法图化。改为 local-expert 模式：每 rank 只计算本地 expert，非本地权重置零，AllReduce 聚合（固定 shape）。数学上与 double_routing 严格等价。非 EP 场景下 init_routing 链路本身图模式兼容，不需要此改造。

### 6.2 图中断修复

| 问题 | 修复 |
|------|------|
| MoE AllToAll 动态 shape | local-expert + AllReduce |
| RoPE kv_len.max().item() | cos/sin 预计算为 buffer |
| actual_seq_lengths_kv list | 改为 Tensor |
| npu_geglu 不支持图编译 | Decode 回退手动 GEGLU |

### 6.3 精度

- **Prefill**: 首 token 与 eager 基线一致（ID=236776）
- **Decode**: 因手动 GEGLU 和 local-expert 路由的数值路径与 eager 不同，输出语义连贯但文本不完全一致。差异来源：(1) 手动 GEGLU vs npu_geglu 的 BF16 舍入路径不同，30 层累积；(2) AllReduce 树形归约 vs AllToAll 的浮点加法顺序不同

### 6.4 性能

| 指标 | 基线 | 阶段 4 | 变化 |
|------|------|--------|------|
| Prefill | 312.51 ms | 302.52 ms | -3.2% |
| Decode | 98.47 ms | 18.21 ms | **-81.5%** |

---

## 7. 总结

| 阶段 | Prefill (ms) | Decode (ms) | Decode 累计变化 |
|------|-------------|------------|----------------|
| 基线 | 312.5 | 98.5 | — |
| 1. 并行化 | 313 | 97 | -1.5% |
| 2. KVCache/FA | 310 | 97 | -1.5% |
| 3. 融合算子 | 307 | 92 | -6.6% |
| **4. 图模式** | **303** | **18.2** | **-81.0%** |

**最终: Decode 98.5 → 18.2 ms, 5.4× 加速**

---

## 7.1 A3 回归测试（2026-04-27）

切换到 Atlas 800 A3（Ascend 910 系列，64 GB HBM/chip）后对 framework 做了一次回归。代码零修改，只调整了 yaml 里的 `model_path` 指向本机权重路径。

### 测试条件
- BS=8，input_len=256，max_new_tokens=32
- TP=1，EP=8（128 experts / 8 = 16 expert/rank）
- 同一份 framework 代码、同一份 `gemma4_rank_8_8ep_*.yaml`

### 稳态性能

| 路径 | 平台 | Prefill | Decode/token | 来源 |
|---|---|---|---|---|
| eager | A2 (910B4) | 312.5 ms | 98.5 ms | baseline (2026-04-15, 阶段 0) |
| ge_graph | A2 (910B4) | 189 ms | 15 ms | framework 阶段 5 (2026-04-22) |
| **ge_graph** | **A3 (910 A3)** | **76.43 ms** | **10.20 ms** |
| **npugraph_ex** | **A3 (910 A3)** | **102.29 ms** | **11.59 ms** |

A3 vs A2（同 ge_graph 路径）：prefill **2.47×**、decode **1.45×** 加速，符合硬件代差预期。

ge_graph 与 npugraph_ex 在 A3 上的对比：ge_graph decode 仍快 ~12%，prefill 快 ~25%，与 A2 上结论一致——该模型规模下 ge_graph 图优化更充分。

### 冷启动 / 编译开销（A3，仅供参考）

| 路径 | 首次 prefill | 首次 decode |
|------|------|------|
| ge_graph | 8.38 s | 605 s（含 graph compile） |
| npugraph_ex | 7.18 s | 358 s |

> npugraph_ex 编译显著快，但稳态略慢；适合短任务/启动敏感场景。

---

## 7.2 MoE EP decode 路径：mc2 vs local_experts (A3, 2026-04-28)

`Gemma4SparseMoeBlock` 暴露 `moe_ep_decode_mode` 开关（yaml 的 `model_config.custom_params.moe_ep_decode_mode`），可选两条 EP decode 路径：

- **`mc2`**: `npu_moe_distribute_dispatch_v2` + `npu_moe_distribute_combine_v2`，与 qwen3_moe / longcat-flash / deepseek 等模型 EP 解码路径一致。A3 上无 expert-per-rank 限制；A2 上要求 `experts_per_rank ≤ 24`（gemma-4 EP=8 时 16/rank 满足）。
- **`local_experts`**: 每个 rank 跑所有 16 个 local expert（非本 rank 的 expert 输出 routing weight 设为 0），最后做一次 AllReduce 汇总。完全静态形状，对 ge_graph fullgraph 友好。

默认值的判定：A2 (Atlas 800/800I A2) 上 `npu_moe_distribute_dispatch_v2` 有 `experts_per_rank ≤ 24` 的算子限制（超过则编译失败），A3 没有这层限制。当前代码按 `experts_per_rank ≤ 24` 自动选 `mc2`，超过则回退 `local_experts`，可在 yaml 里显式覆盖。Gemma-4 在 EP=8 下 16/rank，A2 / A3 都满足条件。

### A3 实测（8 卡 EP=8，BS=8，input_len=256，max_new_tokens=32）

| 模式 | Prefill | Decode |
|------|---------|--------|
| ge_graph + local_experts | 76 ms | 10.20 ms |
| ge_graph + mc2 | 77 ms | 10.50 ms |
| npugraph_ex + local_experts | 102 ms | 11.59 ms |
| npugraph_ex + mc2 | 112 ms | 11.73 ms |

两条路径输出一致。在 BS=8 / decode 1 token 这个工况下 mc2 没快出来，反而比 local_experts 慢 0.2-0.3 ms。原因是这个 shape 下：

- 每层 MoE 实际 routing 的数据量很小（8 个 token × top-8 ≈ 64 个 token-expert 对，hidden=2816）
- `AllReduce` `[8, 2816] ≈ 22 KB` 对 HCCL 来说极小，ring/tree 一次集合通信即完成
- mc2 的 `dispatch_v2` + `combine_v2` 是 2 次 AllToAll-style 通信，加上输出 padding 到 `max_dispatch_tokens` 的固定开销，小 batch 下摊不开

mc2 的优势期望体现在更大 batch 或 prefill 阶段（每层 MoE 实际处理 token 数显著增加时，dispatch+combine 的稀疏路由收益才能跑赢 AllReduce + 冗余 GMM 的成本）。当前规模下两路径接近，但代码默认走 mc2，与仓内其他 MoE 模型保持一致。

---

## 8. 功能问题记录

| 问题 | 阶段 | 根因 | 修复 |
|------|------|------|------|
| 首 token EOS + 高频词重复 | 1 | MoE 激活函数用错（框架默认 SiLU，应为 GEGLU） | Gemma4GegluMoEGMM |
| 修复后仍重复 | 1 | Router 输入双重 norm（pre_feedforward_layernorm_2 + Router 内部 norm） | Router 使用 raw residual |
| 仍无意义输出 | 1 | FA scale=1/√d，但 QK RMSNorm 已归一化，attention 过于平坦 | scale=1.0 |
| Decode 输出异常 | 1 | chat template 缺 BOS token | 添加 `{{ bos_token }}` |
| ge_graph Prefill 首 token 偏移 7.7 分 | 4 | use_fused_geglu 全局设 False，Prefill 也用了手动 GEGLU | 按 is_prefill 动态选择 |
| GE graph 编译失败 | 4 | npu_geglu 未注册图模式 dispatch | Decode 回退手动 GEGLU |

---

## 9. 算子需求

| 算子 | 约束 | 影响 | 建议 |
|------|------|------|------|
| `npu_gelu_mul` | dim ≤1024 | Dense FFN 需 4224 | 放宽维度限制 |
| `npu_ffn` geglu | 仅 fp16 | BF16 模型无法使用 | 增加 bf16 |
| `npu_rotary_mul` | 无 partial rotation 原生支持 | full attention 用 slice+融合+cat 包装可用（实测 prefill -5.3%） | 增加 partial_rotary_factor 原生支持，去掉 slice/cat 包装 |
| `npu_geglu` | 不支持 GE graph | Decode 回退手动实现，引入精度差异 | 支持图模式 |

---

## 10. 遗留与后续

| 方向 | 说明 |
|------|------|
| MC2 MoE | 已验证 (§7.2): A3 + experts_per_rank=16 时 mc2 dispatch_v2 路径可用，已默认启用，BS=8 decode 与 local_experts 持平 |
| npu_geglu 图模式 | 支持后可消除 Decode 手动 GEGLU 精度差异 |
| 多模态 | Vision Tower 未适配 |
| 量化 | W8A8/W4A16 可进一步加速 |
| 长序列 KV | sliding 层可用环形缓存 |

---

## 11. Skill 反馈

| 类型 | 问题 | 建议 |
|------|------|------|
| 执行 | implementer 自验 logit 差距报告不准（<2 vs 实际 7.7） | reviewer 应独立测量 |
| 执行 | npu_geglu 图模式不兼容在 analyzer 阶段未识别 | 融合算子分析应检查图模式兼容性 |
| 流程 | 阶段 2 实质改动极小，migrator 已完成 KVCache+FA | migrator 已含 FA+KVCache 时可合并阶段 |
