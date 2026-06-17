# LongCat-Flash-Lite TP 部署优化报告

## 1. 概述

LongCat-Flash-Lite 在 8 卡 TP 部署下面向单 batch、低延迟场景。下文给出该路径的并行切分、KVCache 与 Attention 改造、算子融合、图模式适配的最终方案，并在 Atlas A2 / A3 两代硬件上给出实测性能与精度结论。

- HuggingFace: [meituan-longcat/LongCat-Flash-Lite](https://huggingface.co/meituan-longcat/LongCat-Flash-Lite)
- 架构: MoE LLM（MLA + Sparse MoE + N-gram Embedding）
- 总参数量: 约 69 B（N-gram Embedding ~31 B，MoE ~34 B）
- 硬件平台: Atlas A2 / A3
- 部署规模: 8 卡（单节点）
- 量化模式: BF16

---

## 2. 模型结构

LongCat-Flash-Lite 采用双子层（dual sub-layer）Transformer 结构，每个 Decoder Layer 内部含两个 MLA Attention 与两个 Dense MLP，并通过 shortcut 通道把 MoE 模块的输出与第二个子层 Dense MLP 相加；Embedding 端使用 12 路 N-gram 子表，配合 4 路移位累加获得 token-level 上下文先验。

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

关键特殊点：

- **N-gram Embedding 子表占比**：12 个 sub-embedder 共约 31 B 参数（占总参数量约 46%），需在 TP 维度切分以满足单卡显存约束；每步 forward 调用约 200 个小算子，host 下发是 Decode 主要瓶颈。
- **MLA 压缩 KV**：每 token 仅缓存 `kv_lora=512` + `qk_rope=64` = 576 维 latent KV，较原 GQA 风格展开（4 KV head × (192+128) = 1280 维/token）节省 55% 显存。
- **MoE shortcut**：第一个子层的 MoE 输出通过 shortcut 通道与第二个子层 Dense MLP 的输出相加，引入跨子层的依赖。
- **MoE Zero Expert**：128 路 identity expert 通过乘法掩码实现，避免布尔索引赋值（GE 图不支持）。

---

## 3. 性能基线

基线在 Atlas A2 上以 eager 模式采集，未启用任何 NPU 融合算子、PA 框架或图编译：MLA 走手写 SDPA，MoE 走 Python for-loop 逐 expert 计算，KVCache 走 transformers 原生连续显存格式。该基线作为后续各项优化的统一对照。

| 指标 | 值 | 测试条件 |
|------|-----|---------|
| Prefill 耗时 (ms) | ~2580 | input_len=1024, batch_size=1 |
| Decode 单步耗时 (ms) | ~273 | batch_size=1, eager, manual SDPA, Python MoE for-loop |
| 端到端吞吐 (tokens/s) | ~3.66 | input_len=1024, output_len=128, batch_size=1 |
| 显存占用 | — | 未启用 PA，KVCache 占用主导 |

基线 Decode 单步耗时 273 ms 主要由 Python MoE for-loop 与 attention 路径的 manual SDPA 占据，是后续优化的主要瓶颈。

### 3.1 精度基线

```
输入：An attention function can be described as mapping a query and a set of
      key-value pairs to an output, where the query, keys, values, and output
      are all vectors. The output is
输出：computed as a weighted sum of the values, where the weight assigned to
      each value is computed by a compatibility function of the query with the
      corresponding key. ...
```

---

## 4. 并行切分

本样例 TP 路径面向单 batch、低延迟场景，全模块按 Tensor Parallel 维度平铺到 8 卡以最大化单 token 算力。N-gram Embedding 的 12 个子表合计约 31 B 参数，TP 切分既分摊算力也是单卡显存的硬性约束。

部署拓扑（`config/longcat_flash_lite_rank_8_8tp.yaml`）：

| 模块 | 切分 | 通信形式 |
|------|------|---------|
| MLA Attention | TP=8 | AllReduce |
| Dense MLP | TP=8 | AllReduce |
| MoE | TP=8 | AllReduce |
| Embedding（主表 + N-gram 子表） | TP=8 | AllReduce（输出端） |
| LM Head | TP=8 | AllGather |

如上表所示，全部模块按 TP=8 平铺，模型整体只走一种通信模式（AllReduce 为主、LM Head 输出 AllGather），结构简单清晰。两点细节值得说明：

- 单 batch 路径下序列并行守卫（`is_sp`）不触发，因为 `attn_dp=1` 不满足 `attn_tp > 1 且 attn_dp > 1` 条件，Prefill / Decode 两阶段均走完整 AllReduce，不拆分为 RS+AG。
- LM Head 在 ColumnParallel 输出后用 `all_gather_into_tensor` 替代 `dist.all_gather`，规避后者产生 Python list 导致图模式下的图中断（graph break）。

并行切分本身不带来直接收益，时延变化体现在后续 KVCache、算子融合与图模式阶段。

---

## 5. KVCache 与 Attention

MLA 压缩 KV 与 Paged Attention 配合 FA v2，将每 token KV 缓存从原 1280 维压到 576 维；Prefill 与 Decode 各走一条独立 FA 路径，5.3 单独介绍 Decode 上的 MLA Prolog 融合。

### 5.1 KVCache 管理

MLA 压缩缓存与 framework 的 Paged Attention 框架对接，仅缓存 latent KV，避免在 K/V 头维度展开。

| 参数 | 值 | 说明 |
|------|-----|------|
| KV 模式 | MLA 压缩 + Paged Attention | 仅缓存 latent KV |
| cache_mode | `PA_NZ` | NZ 内存排布，配合 FA v2 直读 |
| block_size | 128 | Paged 块粒度 |
| 单 token 缓存维度 | 576 维（512 + 64） | 较原 GQA 风格展开 1280 维节省 55% |
| Prefill cache 写入 | `npu_kv_rmsnorm_rope_cache` | 同时完成 KV layernorm + RoPE + cache 写入，并返回当前 step 的 `k_nope` / `k_rope` |
| Decode cache 写入 | `npu_mla_prolog_v3` | 见 5.3 |

KV-LoRA scale 始终在 FA 调用之外应用（Prefill 在 `k_nope` 输出端、Decode 在 `npu_mla_prolog_v3` 的 `q_nope` 输出端），cache 在 Prefill / Decode 写入下均保持未缩放，跨槽尺度一致。

### 5.2 Flash Attention 算子选型

Prefill 与 Decode 走两条不同的 FA v2 路径，布局与 mask 模式独立选择。

| 项 | Prefill | Decode |
|---|---|---|
| FA op | `npu_fused_infer_attention_score_v2`（非 absorb） | `npu_fused_infer_attention_score_v2`（absorb） |
| K/V 输入 | 当前 batch 的展开 K/V | `key=value=cache_nope`，由 PA 框架按 `block_table` 拉取 |
| input_layout | `NTD_TND` | `TND_NTD` |
| sparse_mode | 3（causal mask） | 0（full attention） |
| softmax_scale | 标准 | 显式传入，配合 absorb |

两条路径接入 KVCache 后输出与基线对齐；Prefill 从 2580 ms 降至 2540 ms（−1.6%），Decode 单步从 273 ms 降至 267 ms（−2.2%），单卡 NPU 显存峰值 20140 MB。本节主要收益来自缓存维度减半与显存峰值下降，时延收益受限于 MoE / Norm 等仍走 eager 的影响，需待算子融合阶段释放。

### 5.3 MLA Prolog 融合

Decode 路径以 `npu_mla_prolog_v3` 一次性完成 `q_a_proj → RMSNorm → q_b_proj → split → RoPE → absorb-via-K` 与 `kv_a_proj_with_mqa → RMSNorm → RoPE → cache 写入`，替代 6–7 个独立小算子。Q-LoRA scale 通过 `qc_qr_scale` 折进 kernel；`kc_scale=1.0` 让 cache 与 Prefill 写入路径保持同一未缩放约定，KV-LoRA scale 改在 `q_nope` 输出上应用。

启用条件：图模式 decode（`exe_mode` 为 `"ge_graph"` 或 `"npugraph_ex"`）且 `q_lora_rank > 0`，默认开启。启用后 Decode 单步实测稳定在 5.76 ms / 174 tok/s 水平（同代码关闭 prolog 实测约 6.1 ms / 165 tok/s，差异约 5–6%）；精度与关闭路径输出在前 ~20 token 一致，之后因 BF16 累加顺序略有 token 级分歧、语义保持等价。

---

## 6. 算子融合

TP 路径上替换为 torch_npu 融合算子的全部位置如下；第 5 节已介绍的 FA / MLA Prolog 类算子不再展开，此处补全 Norm、激活、MoE 路由 / GMM 等其余融合点。

| 模块 | 实现 | 来源 | 触发位置 |
|------|------|------|---------|
| RMSNorm | `npu_rms_norm` | 算子融合 | 每层 Q-Norm / pre-Norm 等首层无 residual 位置 |
| Residual + RMSNorm（56 处） | `npu_add_rms_norm` | 算子融合 | 层间残差 |
| Dense MLP | `MergedColumnParallelLinear`（gate/up 合并）+ `npu_swiglu` | 算子融合 | Dense MLP gate+up+down |
| MoE Router | `npu_moe_gating_top_k` | 算子融合 | MoE 路由打分 |
| MoE Token 分发 | `npu_moe_init_routing_v2` + `npu_moe_finalize_routing` | 算子融合 | Token 重排 / 聚合 |
| MoE Expert 计算 | `FusedMoEGMM`（`npu_grouped_matmul` + `npu_swiglu`） | 算子融合 | Expert 并行 GMM |
| Flash Attention（Prefill / Decode） | `npu_fused_infer_attention_score_v2` | KVCache 与 Attention | Attention 计算 |
| MLA Prolog（Decode） | `npu_mla_prolog_v3` | KVCache 与 Attention | MLA 前置 |
| KV layernorm + RoPE + cache 写入（Prefill） | `npu_kv_rmsnorm_rope_cache` | KVCache 与 Attention | MLA 前置 |

替换覆盖了 Norm、激活、MoE 路由 / 分发 / GMM 等所有可融合的热点。叠加上一阶段的 KVCache 改造后，端到端实测如下：

| 指标 | KVCache 后 | 融合算子后 | 本阶段变化 | vs baseline |
|------|-----------|-----------|---------|------------|
| Prefill (ms) | 2540 | 105 | **−95.9%** | −95.9% |
| Decode (ms) | 267 | 92 | **−65.5%** | **−66.3%** |
| 吞吐 (tok/s) | 3.7 | 10.9 | **+195%** | +195% |
| NPU 显存 (MB) | — | 17340 | — | — |

技术依据：Prefill 1024 token 几乎激活全部 384 个 expert，原 Python for-loop 单层约 384 次迭代 × 14 层 ≈ 5376 次调度；融合算子合并后 Prefill 加速幅度（−95.9%）远大于 Decode（−65.5%）。本阶段精度与基线对齐，最大收益来自 MoE GMM 替换 Python for-loop。

---

## 7. 图模式

本样例同时支持 GE 图（默认）与 npugraph_ex 两个后端；同一份模型代码通过 yaml `model_config.exe_mode` 切换，两个后端共享融合算子与 KVCache 接口，仅在 Decode 整图编译路径上差异化。

### 7.1 GE 图

Decode 路径以 `fullgraph=True` 整图编译，Prefill 保持 eager。下列改造均围绕"避免图中断、避免重编译"两条主线展开：

- **N-gram Embedding 入图**：把 `ngram_context` 从普通属性改为 `register_buffer`，在 batch 已知后预分配；前向写入用 `copy_()` 原位更新，避免新建张量打断图跟踪。`_shift_right_ignore_eos` 内的前缀 max 用 Python 静态展开为一串 `torch.maximum`，绕开 GE 后端尚不支持的 `cummax`。
- **LM Head 通信**：用 `dist.all_gather_into_tensor` 替代 `dist.all_gather`，后者会生成 Python list 触发图中断。
- **Flash Attention 入口**：Decode 切到 `tng.ops.npu_fused_infer_attention_score_v2`，其 schema 接受 Tensor 类型的 `actual_seq_lengths_kv`，避免每步 kv 长度变化引发的重编译；编译时关闭 dynamic 模式，让 shape 在图内为常量。
- **Paged Attention cache**：cache 保留原地更新语义，但下游一律以 `npu_kv_rmsnorm_rope_cache` 的返回值作为输入，让 KV 写入与读取在图内形成显式数据流；只对 `block_table` 做 `mark_static`，cache 本身保持可变。
- **MoE Zero Expert mask**：用乘法掩码代替布尔索引赋值（GE 后端尚不支持后者）。
- **静态权重标记**：`process_weights_after_loading` 阶段对 MLA absorb 权重与 prolog 权重统一 `mark_static_address`，整图编译时不会被当作动态输入处理。
- 模型类设 `_can_compile_fullgraph = True`，框架自动按整图路径编译。

| 指标 | 融合算子后（eager） | GE 图（N-gram 图外） | GE 图（N-gram 入图） | vs baseline |
|------|---------------------|------------------------|------------------------|------------|
| Prefill (ms) | 105 | 107 | 106 | −95.9% |
| Decode (ms) | 92 | 18.5 | **7.14** | −97.4% |
| 吞吐 (tok/s) | 10.9 | 54 | **140** | +3681% |
| NPU 显存 (MB) | 17340 | 17448 | 17448 | 持平 |

技术依据：N-gram Embedding 在 eager 路径下单步 forward 触发约 200 个小算子（12 个 sub-embedder × multi-op + 12 次 AllReduce + post_proj + shift_right + ngram-id 累加），每个算子的 host 下发 + 集合通信下发约 50–150 µs，累计约 11 ms。入图后这部分串行开销被合并到一次图下发，Decode 实测时延从 18.5 ms 降至 7.14 ms，与 profile 测得的 host 下发开销（11.20 ms）量级吻合，主要增益即来自此处。

精度行为：前约 20 个 Decode token 与基线一致；之后因 BF16 累加顺序在 GE 图调度下与 eager 不同，出现 token 级分歧，语义保持等价。

### 7.2 npugraph_ex

`npugraph_ex` 是 torch_npu 注册的 torch.compile backend，框架在 `executor/utils/graph_utils.py` 内以 `torch.compile(..., backend="npugraph_ex", ...)` 接入。切换路径只需把 yaml 的 `model_config.exe_mode` 从 `"ge_graph"` 改成 `"npugraph_ex"`，模型代码无需改动——MLA 模块在初始化时按执行模式分别绑定 `tng.ops` 或 `torch.ops.npu` 下的 FA 算子，Decode 路径内 `actual_seq_lengths_kv` 自动按后端切换 list / Tensor 形态。

| 路径 | Prefill (ms) | Decode (ms) |
|------|--------------|-------------|
| GE 图 | 47.84 | 5.76 |
| npugraph_ex | ~59 | ~7.6 |

两个后端共享同一份模型代码与算子接口，Decode 时延以 GE 图更优（该模型规模下整图优化更充分），npugraph_ex 编译启动更轻量、更适合编译开销敏感场景；上表中 npugraph_ex 数据来自早期对比实验。

---

## 8. 累计性能演进

A2 上各阶段的累计优化效果与 A3 同代码迁移收益分别汇总于下，最大收益来源为 N-gram Embedding 入图（消除 host 下发开销）与 MoE GMM 融合（替换 Python for-loop）。

### 8.1 A2 同硬件累计优化路径

| 路径 | 关键改造 | Prefill (ms) | Decode (ms) | 吞吐 (tok/s) | vs baseline (decode) |
|------|---------|--------------|-------------|--------------|----------------------|
| 基线 | manual SDPA + Python MoE loop | 2580 | 273 | 3.7 | 1.0× |
| + KVCache | MLA 压缩缓存 + PA + FA | 2540 | 267 | 3.7 | 1.02× |
| + 算子融合 | RMSNorm / SwiGLU / MoE GMM | 105 | 92 | 10.9 | 2.97× |
| + GE 图（N-gram 图外） | torchair Decode 整图 | 107 | 18.5 | 54.1 | 14.8× |
| + N-gram Embedding 入图 | host 下发开销消除 | 106 | **7.14** | **140** | **38.2×** |

最终 A2 上 Decode 单步从 273 ms 降至 7.14 ms，吞吐由 3.7 tok/s 提升至 140 tok/s，累计加速 38.2×。

### 8.2 A3 跨硬件路径

| 路径 | Prefill (ms) | Decode (ms) | 吞吐 (tok/s) | vs A2 同代码 |
|------|--------------|-------------|--------------|--------------|
| A2 N-gram Embedding 入图（参考） | 106 | 7.14 | 140 | — |
| A3 同代码（默认开启 MLA Prolog） | 47.84 | **5.76** | **174** | Decode −19%，吞吐 +24% |

A3 相对 A2 在同代码下 Decode 单步缩短约 19%、吞吐提升 24%，主要来自 A3 算力代差；MLA Prolog 默认随 ge_graph 启用，已计入上表。

> 测试条件：batch_size=1，TP=8，BF16，input_len=1024 / output_len=128。Decode 单步在多次实测下稳定在 5.76–6.14 ms 区间。

### 8.3 长输入场景

把输入序列从 1K 扩展到 4K（输出 128 → 1K）后，A3 上的实测如下：

| 项 | 1K input / 128 output | 4K input / 1K output |
|---|---|---|
| Prefill (ms, 预热后) | 47.84 | 216.5 |
| Decode mean (ms) | 5.76 | 6.00 |
| 吞吐 (tok/s) | 174 | ~167 |

输入从 1K 增至 4K 时，单步 Decode 仅增约 4%（KV cache 更大带来 PA 加载与 FA 计算开销上升），Prefill 近似线性增长（×4.5，对应输入 ×4）；长输入场景吞吐仍稳定在 165 tok/s 以上。
