# LongCat-Flash-Lite EP 部署优化报告

## 1. 概述

LongCat-Flash-Lite 在 8 卡 EP 部署下面向中等 batch、吞吐优先场景。下文给出 EP 路径的并行切分、KVCache 与 Attention 改造、算子融合、图模式适配的最终方案，并在其上叠加 W8A8 量化与 Decode 多流 MoE 重叠两层进阶优化，附 A2 / A3 两代硬件上的实测性能与精度结论。MoE 模块按专家并行（EP）切分到 8 卡，N-gram Embedding 单独按 TP 切分，Dense MLP 与主 Embedding 保持 DP-replicated 以减少同步通信。

- HuggingFace: [meituan-longcat/LongCat-Flash-Lite](https://huggingface.co/meituan-longcat/LongCat-Flash-Lite)
- 架构: MoE LLM（MLA + Sparse MoE + N-gram Embedding）
- 总参数量: 约 69 B（N-gram Embedding ~31 B，MoE ~34 B）
- 硬件平台: Atlas A2 / A3
- 部署规模: 8 卡（单节点）/ 16 卡（等比扩展）
- 量化模式: BF16 / W8A8

---

## 2. 模型结构

LongCat-Flash-Lite 采用双子层（dual sub-layer）Transformer 结构，每个 Decoder Layer 内部含两个 MLA Attention、两个 Dense MLP 与一个共享 MoE；Embedding 端在主 vocab 表之外引入 12 路 N-gram 子表，配合移位累加获得 token-level 上下文先验。

```
Embedding:
  embed_tokens         VocabParallelEmbedding(131072 → 3072)            0.4 B
  ngram_embedders[12]  Embedding(10.22 M → 256) × 12 + post_proj × 12   31 B
14 × DecoderLayer:
  ├── 2 × LongcatFlashMLA  (q_lora=1536, kv_lora=512, num_heads=32)
  ├── 2 × LongcatFlashMLP  (gate/up/down 3072 ↔ 6144)
  └── 1 × LongcatFlashMoE  (256 routed + 128 zero, top-12, expert_intermediate=1024)
final RMSNorm + LM Head (3072 → 131072)
```

关键特殊点：

- **N-gram Embedding 子表占比**：12 个子表每表 10.22 M 行 × 256 列，共约 31 B 参数（占总参数量约 46%），是单卡显存的硬约束，必须按 TP 切分。
- **MoE 双层结构**：256 个 routed expert（EP=8 时每卡 32 个）+ 128 个 zero expert（identity 副本，每卡持有，不参与 EP 切分），后者通过乘法掩码实现以兼容 GE 图。
- **Dual sub-layer attention**：每 layer 含 2 个 attention block + 1 个共享 MoE，MoE 输出通过 shortcut 通道与第二个子层 Dense MLP 相加，跨子层存在依赖。

---

## 3. 性能基线

基线在 EP 拓扑（attn_tp=4 + attn_dp=2 + dense DP + N-gram TP + EP=8）下以 eager 模式采集，未启用任何 NPU 融合算子、PA 框架或图编译。该基线作为后续 KVCache、算子融合、图模式各阶段优化效果的统一对照。

| 指标 | 值 | 测试条件 |
|------|-----|---------|
| Prefill 耗时 (ms) | 198 | input_len=1024, batch_size=1, attn_tp=8 |
| Decode 单步耗时 (ms) | 161 | batch_size=1, max_new=128 |
| 单卡 HBM 峰值 (GB) | 19.6 | EP=8, Dense DP |

EP 拓扑基线的 Decode 已较 TP 基线显著下降（273 → 161 ms），主要来自 MoE 模块由 TP weight-shard 改成 EP expert-shard——每卡承担的 expert 数从 256 降至 32，每 token 的 MoE for-loop 迭代规模同步缩小；后续优化在此基础上继续压缩通信、算子与 host 下发开销。

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

EP 路径将专家并行作为 MoE 模块的主要切分方式，每卡持完整的 32 个 routed expert，Dense MLP 与主 Embedding 走 DP-replicated 以避免逐层 AllReduce；N-gram Embedding 因显存约束单独按 TP 切分。该拓扑在中等 batch 下能保持 MoE GMM 接近峰值算力。

部署拓扑（`config/longcat_flash_lite_rank_8_8ep.yaml`）：

| 模块 | 切分 | 通信形式 |
|------|------|---------|
| MLA Attention | attn_tp=4，attn_dp=2 | RS+AG（prefill）/ AllReduce（decode） |
| Dense MLP | DP=1（每卡完整权重） | 无 |
| MoE | EP=8 | dispatch_v2 / combine_v2（decode）；double-routing AllToAll（prefill） |
| 主 Embedding | DP=1（每卡完整主表 0.8 GB） | 无 |
| N-gram 子表 | TP=8 | AllReduce |
| LM Head | TP=8 | AllGather |

整体思路是把同步通信尽量限制在必要的模块上，具体决策如下：

- Dense MLP 走 DP-replicated：避免逐层 AllReduce 同步开销，单卡多放 1.4 GB 权重对 HBM 仍有富余。
- MoE 走 EP=8：MoE `expert_intermediate=1024`，若 TP=8 切分后单卡只剩 128 维，矩阵碎片化会拖累 cube 利用率；EP 让每卡持完整 32 个 expert，GMM 单次调度的有效算力更高。
- N-gram 子表独立 TP=8（由 `model_config.custom_params.ngram_embed_tp_size=8` 启用）：12 个子表共约 31 B，必须切；与主 embed 解耦后主表可走 DP，省去主 embed 的 AllReduce。
- attn_tp=4 + attn_dp=2 拓扑使 q/k/v/o 矩阵保持较大尺寸提升 cube 利用率；batch=2 时 MoE GMM 接近峰值。

| 配置 | Prefill (ms) | Decode (ms) | 精度 |
|------|--------------|-------------|------|
| EP=8 + attn_tp=4 + batch=2 | 207 | 150 | 与 § 3 基线一致 |

切换到 EP 拓扑本身不带来显式时延加速，Prefill 与 Decode 量级与基线持平；其作用在于打开后续 KVCache、MC2 与图模式的可用空间。

---

## 5. KVCache 与 Attention

MLA 压缩 KV 与 Paged Attention 配合 FA v2，把每 token KV 缓存压到 576 维；Prefill 与 Decode 走两条独立 FA 路径；Decode 路径默认启用 MLA Prolog 融合；EP 路径在 attention 输出端把 AllReduce 拆成 ReduceScatter + AllGather，配合 `attn_dp=2 + attn_tp=4` 形成 TP+SP 优化。

### 5.1 KVCache 管理

MLA 压缩缓存与 framework 的 Paged Attention 框架对接，每 token 仅缓存 latent KV，避免在 K/V 头维度展开。

| 参数 | 值 | 说明 |
|------|-----|------|
| KV 模式 | MLA 压缩 + Paged Attention | 仅缓存 latent KV |
| cache_mode | `PA_NZ` | NZ 内存排布，配合 FA v2 直读 |
| block_size | 128 | Paged 块粒度 |
| 单 token 缓存维度 | 576 维（512 + 64） | 较原 GQA 风格展开 1280 维节省 55% |
| Prefill cache 写入 | `npu_kv_rmsnorm_rope_cache` | 同时完成 KV layernorm + RoPE + cache 写入 |
| Decode cache 写入 | `npu_mla_prolog_v3` | 见 5.3 |

KV-LoRA scale 在 FA 调用之外应用（Prefill 在 `k_nope` 输出端、Decode 在 `npu_mla_prolog_v3` 的 `q_nope` 输出端），cache 在两条写入路径下都保持未缩放，跨槽尺度一致。

### 5.2 Flash Attention 算子选型

Prefill 与 Decode 走两条不同的 FA v2 路径，布局、mask 模式与 cache 拉取方式独立选择；batch>1 的 packed prefill 通过 cumulative offsets 送入 `actual_seq_lengths`。

| 项 | Prefill | Decode |
|---|---|---|
| FA op | `npu_fused_infer_attention_score_v2`（非 absorb） | `npu_fused_infer_attention_score_v2`（absorb） |
| K/V 输入 | 当前 batch 的展开 K/V | `key=value=cache_nope`，由 PA 框架按 `block_table` 拉取 |
| input_layout | `NTD_TND` | `TND_NTD` |
| sparse_mode | 3（causal mask） | 0（full attention） |
| actual_seq_lengths | cumulative offsets，支持 packed prefill | tensor |
| softmax_scale | 标准 | 显式传入，配合 absorb |

KVCache 接入后两条路径输出与基线对齐；缓存维度减半与 PA 接入是显存改善的主要来源，时延收益受限于 MoE / Norm 等仍走 eager 的影响，需待算子融合阶段释放。

### 5.3 MLA Prolog 融合

Decode 路径以 `npu_mla_prolog_v3` 一次性完成 `q_a_proj → RMSNorm → q_b_proj → split → RoPE → absorb-via-K` 与 `kv_a_proj_with_mqa → RMSNorm → RoPE → cache 写入`，替代 6–7 个独立小算子。Q-LoRA scale 通过 `qc_qr_scale` 折进 kernel；`kc_scale=1.0` 让 cache 与 Prefill 写入路径保持同一未缩放约定，KV-LoRA scale 改在 `q_nope` 输出上应用。

启用条件：图模式 decode（`exe_mode` 为 `"ge_graph"` 或 `"npugraph_ex"`）且 `q_lora_rank > 0`。

| 项 (b=2, A3) | 无 prolog | 启用 prolog | Δ |
|---|---|---|---|
| Decode mean (ms) | 8.36 | 8.10 | −3.1% |
| Prefill 预热后 (ms) | 239 | 239 | ~0 |

启用 MLA Prolog 后 Decode 单步在 A3 上额外缩短约 0.26 ms，Prefill 无变化；精度与无 prolog 路径输出对齐。

### 5.4 Prefill TP+SP attention

Prefill 阶段把 attention 输出端的 `AllReduce(O)` 拆成 `ReduceScatter(O)` + 下一层入口 `AllGather(input)`，层间 hidden 维度从 `[T, H]` 缩到 `[T/tp, H]`：

- RMSNorm / Dense MLP 计算量按 `attn_tp` 倍数缩减
- 层间残差激活显存同等比例减少
- 通信总量与 AllReduce 数学等价（ring AllReduce ≡ RS + AG 两阶段）

启用条件：`attn_tp > 1` 且 `attn_dp > 1`（`is_sp` 守卫），仅 Prefill 触发；Decode 不触发，因为 `batch_per_dp_rank` 可能小于 `attn_tp`，pad 浪费会抵消收益。

| 配置 | Prefill 无 SP (ms) | Prefill SP (ms) | Δ |
|---|---|---|---|
| EP=8 batch=8 | 199.91 | 145.22 | −27% |
| EP=8 batch=2 | 66 | 65 | 持平 |

适用范围约束：batch=2 时单请求 256 token T 偏小，SP 节省的 norm / MLP 算力被 RS+AG 同步开销抵消；收益需要更大 batch 或更长输入才能显著体现。

---

## 6. 算子融合

EP 路径上替换为 torch_npu 融合算子的全部位置如下；第 5 节已介绍的 FA / MLA Prolog 类算子不再展开，此处补全 Norm、激活、MoE 路由 / 分发 / GMM 等其余融合点。

| 模块 | 实现 | 来源 | 触发位置 |
|------|------|------|---------|
| RMSNorm | `npu_rms_norm` | 算子融合 | 每层 Q-Norm / pre-Norm 等首层无 residual 位置 |
| Residual + RMSNorm | `npu_add_rms_norm` | 算子融合 | 层间残差 |
| Dense MLP | `MergedColumnParallelLinear`（gate/up 合并）+ `npu_swiglu` | 算子融合 | Dense MLP gate+up+down |
| MoE Router | `npu_moe_gating_top_k` | 算子融合 | MoE 路由打分 |
| MoE Token 分发（Decode） | `npu_moe_distribute_dispatch_v2` + `npu_moe_distribute_combine_v2` | 算子融合 | MC2 路径 |
| MoE Token 分发（Prefill） | `_dispatch_double_routing`（double routing） | 算子融合 | Prefill 路径 |
| MoE Expert 计算 | `FusedMoEGMM`（`npu_grouped_matmul` + `npu_swiglu`） | 算子融合 | Expert 并行 GMM |
| Flash Attention（Prefill / Decode） | `npu_fused_infer_attention_score_v2` | KVCache 与 Attention | Attention 计算 |
| MLA Prolog（Decode） | `npu_mla_prolog_v3` | KVCache 与 Attention | MLA 前置 |
| KV layernorm + RoPE + cache 写入（Prefill） | `npu_kv_rmsnorm_rope_cache` | KVCache 与 Attention | MLA 前置 |

MoE 路径上的两条通信策略需要单独说明，分别覆盖 Decode 与 Prefill：

- **Decode 通信走 MC2**：MoE 的 token 分发与聚合由 MC2 `dispatch` / `combine` 一对 all-to-all 算子完成，expert GMM 计算独立于通信之外。MC2 算子要求独占通信域，因此单独建 `moe_ep_group_mc2`（连 prefill 的 EP / CP 通信域也不能复用），按 fullmesh 拓扑组织流量。eager、GE 图、npugraph_ex 三种后端下均可工作，默认启用。A2 上单卡 expert 数（EP=8 / 每卡 32 个 expert）超过算子上限（24），yaml 标 `platform_version: "A2"` 后框架自动回退到 double routing，无需改模型代码。
- **Prefill 仍走 double routing**：Prefill 单层 token 数超出 MC2 dispatch 的 batch 上限（约 512），由 `_dispatch_double_routing` 维持正确性。

算子融合阶段精度与基线对齐；最大收益来自 MoE GMM 替换 Python for-loop 与 MC2 dispatch_v2 替换 double routing。

---

## 7. 图模式

本样例同时支持 GE 图与 npugraph_ex 两个后端，同一份模型代码通过 yaml `model_config.exe_mode` 切换；两个后端共享融合算子与 KVCache 接口，仅在 Decode 整图编译路径上差异化。EP 路径下 Prefill 始终保持 eager。

### 7.1 GE 图

Decode 路径以 `fullgraph=True` 整图编译，Prefill 保持 eager。改造与 TP 路径同源，要点如下：

- **N-gram Embedding 入图**：把 `ngram_context` 从普通属性改为 `register_buffer`，在 batch 已知后预分配；前向写入用 `copy_()` 原位更新，避免新建张量打断图跟踪。`_shift_right_ignore_eos` 内的前缀 max 用 Python 静态展开为一串 `torch.maximum`，绕开 GE 后端尚不支持的 `cummax`。
- **LM Head 通信**：用 `dist.all_gather_into_tensor` 替代 `dist.all_gather`，避免后者生成 Python list 触发图中断（graph break）。
- **静态权重标记**：`process_weights_after_loading` 阶段对 MLA absorb 权重与 prolog 权重统一 `mark_static_address`。
- 模型类设 `_can_compile_fullgraph = True`，框架自动按整图路径编译。

性能（attn_tp=4，batch=2）：

| 项 | eager（融合算子后） | GE 图 | Δ |
|---|---|---|---|
| Decode mean (ms) | 43.2 | **9.10** | −79% |
| Prefill 稳态 (ms) | 161 | 239 | +48% |

适用范围约束：MoE Prefill 仍走 `_dispatch_double_routing` eager（含 host sync），且 `dispatch_v2` 受 BS 上限限制无法替换，因此 Prefill 在图模式下 +48%。Decode 走 MC2 + 整图编译收益足以抵消，整体仍是 EP 路径的最优后端。

### 7.2 npugraph_ex

`npugraph_ex` 是 torch_npu 注册的 torch.compile backend，框架在 `executor/utils/graph_utils.py` 内以 `torch.compile(..., backend="npugraph_ex", ...)` 接入。切换路径只需把 yaml 的 `model_config.exe_mode` 从 `"ge_graph"` 改成 `"npugraph_ex"`，模型代码无需改动——MLA 模块在初始化时按执行模式分别绑定 `tng.ops` 或 `torch.ops.npu` 下的 FA 算子，Decode 路径内 `actual_seq_lengths_kv` 自动按后端切换 list / Tensor 形态。

A3 实测：

| 模式 | yaml | Prefill (ms) | Decode mean (ms) |
|------|------|--------------|------------------|
| GE 图 | rank_8_8ep（b=2） | 62 | 8.10 |
| npugraph_ex | rank_8_8ep（b=2） | 77.92 | 9.11 |
| GE 图 | rank_8_8ep（b=8） | 129 | 9.54 |
| npugraph_ex | rank_8_8ep（b=8） | 95.61 | 11.17 |

适用范围约束：稳态 Decode 在 npugraph_ex 上慢 12–17%（GE 图整图优化更充分）；Prefill 不入图，未享受 SP + 图整体优化，b=2 反而慢 26%，b=8 因 GE 图路径上 batch 处理 overhead 较多反而快 26%。整体稳态吞吐 GE 图占优，npugraph_ex 更适合编译启动敏感场景。

---

## 8. W8A8 量化

接入 compressed-tensors W8A8：MoE GMM（256 个 routed expert）与 Dense MLP 走 int8，MLA 保持 BF16（在 deploy 的 `ignore` 集内，`npu_mla_prolog_v3` 不接 W8A8 入参）。yaml 仅把 `model_config.model_path` 切到 W8A8 checkpoint，框架从 HF `quantization_config` 自动注入 `quant_config`，无需额外 yaml 字段。

### 8.1 量化路径

| 模块 | 量化模式 | 实现 |
|------|---------|------|
| MLA Attention | 保持 BF16 | 在 `ignore` 集内，权重不量化；`npu_mla_prolog_v3` 无 W8A8 入参 |
| Dense MLP | `mm_quant_mode = w8a8int8` | `gate_up → int32 + per-token scale → npu_dequant_swiglu_quant 融合 → down(int8)` |
| MoE GMM | `gmm_quant_mode = w8a8int8` | `_forward_ep_decode` / `_forward_ep_prefill` 在 GMM 前 `npu_dynamic_quant`，per-token scale 随路送入 `FusedMoEGMM` |

设计选择：MoE 的 AllToAll dispatch 仍走 BF16（`dispatch_v2` 的 `quant_mode=0`），在 re-routing 之后按 rank 做一次 `npu_dynamic_quant` 再送入 GMM。原因——`dispatch_v2` 内建的 W8A8 路径需要跨全部 routed expert 的全局 `smooth_scale`，而本 rank 只持有一个 shard，gather 全局 smooth_scale 的代价高于一次 rank 内 `npu_dynamic_quant`；Prefill 的 double-routing 路径同理保持 BF16 alltoall。

### 8.2 W8A8 性能

W8A8 是正收益。在 Atlas A3、专家并行八卡相同配置下，由 BF16 切换到 W8A8 实测如下：

| 配置 | BF16 Decode (ms) | W8A8 Decode (ms) | Δ Decode | Prefill Δ |
|------|------------------|------------------|----------|-----------|
| 批量 2，输入 1K / 输出 128 | 8.38 | 7.18 | −14.2%（吞吐 +16.4%） | — |
| 批量 8，输入 4K / 输出 1K | 9.80 | 8.21 | −16.2%（吞吐 +19.4%） | −36.1%（1K 输入） |

量化把 MoE GMM 与 Dense MLP 的 cube 有效算力翻倍，Decode 在中等批量下稳定下降 14% 至 16%；Prefill 在 1K 输入下实测下降约 36%，4K 输入下 attention 保持 BF16、占比升高，量化收益相应降低（未单独实测）。

适用范围约束：W8A8 推荐用于专家并行、批量不小于 2 的算力受限场景；张量并行、单批量等场景下计算单元未饱和、调度开销占主导，量化收益有限，不推荐（见 `optimization_report_tp.md`）。

---

## 9. 多流 MoE 重叠

LongCat-Flash-Lite 的双子层结构——每层包含两个 MLA Attention 和一个共享 MoE，MoE 的输出经 shortcut 通道与第二个子层相加——天然提供了计算重叠的机会：把整段 MoE（含 dispatch、GMM、combine）调度到副流，主流并行执行第一个子层（dense_a、mla[1]、dense_b），在层末通过 shortcut 通道三路相加汇合。该优化仅在 Decode 阶段启用，Prefill 保持串行。

### 9.1 设计

| 项 | 内容 |
|------|------|
| 重叠对象 | MoE 整段（副流）与第一个子层（主流）并行 |
| 流切换 | `npu_stream_switch(..., "moe")`，代码中唯一显式声明的副流 |
| 核划分 | `limit_core_num` 把 AI Core 与 Vector Core 在主流和 MoE 副流之间切分（主流分得 16 个 AI Core、32 个 Vector Core，MoE 副流分得 8 个 AI Core、16 个 Vector Core），避免两条流争用同一批计算核 |
| 同步点 | 层末 `npu_wait_tensor` 后通过 shortcut 通道三路相加 |
| 开关 | `custom_params.enable_moe_stream_overlap`（默认关闭）、`enable_limit_core_num`（默认开启） |

### 9.2 多流性能

在专家并行八卡、批量 8、约 4K 输入与 1K 输出的 W8A8 配置下，于串行基线上叠加多流重叠与核数限制：

| 路径 | 设备侧耗时 (ms) | 端到端 (ms/step) |
|------|------------------------|------------------|
| W8A8 串行 | 7.67 | 8.21 |
| 叠加多流重叠与核数限制 | **6.62** | **7.12**（主机空闲时约 6.7） |

- **设备侧耗时下降约 14%**（不受主机负载干扰的可靠口径）：MoE 整段约九成被另一子层的计算隐藏，仅约 0.33 ms 暴露在关键路径上，主流计算（约 5.0 ms）成为新的瓶颈。
- 端到端（8.21 → 7.12 ms，约 13%）与设备侧（7.67 → 6.62 ms，约 14%）口径量级一致，设备侧为去除主机干扰后的稳定值；同一软件栈实测，主机空闲时端到端回落到接近设备侧（约 6.7 ms），主机被其它租户抢占时抬到约 7.5 ms。
- 精度与 BF16 对齐：同 prompt 前 16 step logits cosine ≥0.998、top1 全一致，输出语义等价。

适用范围约束：多流是可选开关，默认关闭时走稳定的串行路径。SuperKernel（torchair 的 `super_kernel` scope 融合）与权重 L2 预取在本模型规模上实测均无增益——SuperKernel 因融合反而拖慢主流，预取的收益落在多次运行的波动区间内，故均未纳入；二者属规模相关优化，在参数量更大的 LongCat-Flash 上才体现收益。

---

## 10. 累计性能演进

### 10.1 BF16 优化路径

A3 平台批量 2 的各阶段实测如下，行间是累加关系，最右一列对照初始 TP=8 eager 基线：

| 路径 | 关键改造 | Decode (ms/token) | vs 基线 |
|------|---------|-------------------|---------|
| 基线 (TP=8 eager) | manual SDPA + Python MoE loop | 273 | 1.0× |
| + 并行切分 | EP=8 + Dense DP + N-gram TP | 84.1 | 3.25× |
| + KVCache + FA | PA + `npu_fused_infer_attention_score_v2` | 75.5 | 3.62× |
| + 算子融合 | RMSNorm / SwiGLU / AddRMSNorm | 68.5 | 3.99× |
| + MC2 dispatch_v2 | `npu_moe_distribute_dispatch_v2` / `combine_v2` | 43.2 | 6.32× |
| + GE 图整图 | torchair full-graph decode | 9.10 | 30.0× |
| + MLA Prolog 融合 | `npu_mla_prolog_v3` | **8.10** | **33.7×** |

BF16 路径 A3 Decode 单步从 273 ms 降至 8.10 ms，累计加速 33.7×。

### 10.2 最终版本

在大批量长序列部署配置（专家并行八卡、批量 8、约 4K 输入与 1K 输出）下，于 BF16 路径之上再叠加 W8A8 量化与多流重叠，单步 Decode 端到端汇总如下：

| 阶段 | Decode 端到端 (ms) | 设备侧耗时 (ms) | vs BF16 |
|------|---------------------|------------------|---------|
| BF16 | 9.80 | — | 1.0× |
| + W8A8 量化 | 8.21 | 7.67 | 1.19× |
| + 多流重叠 | **7.12** | **6.62** | **1.38×** |

最终部署配置端到端 7.12 ms（主机空闲时约 6.7）、设备侧 6.62 ms，相对 BF16 同配置累计提速约 1.38×。

### 10.3 吞吐扩展性

A3 平台固定 EP 拓扑（attention 走 attn_tp=4 + attn_dp=2），扫多组输入长度与批量验证吞吐扩展性。

| 配置 | input_len | output_len | batch | per-DP batch | Prefill (ms) | Decode mean (ms) | 吞吐 (tokens/s) | vs b=2 1K |
|------|-----------|------------|-------|--------------|--------------|------------------|----------------|-----------|
| 短输入 baseline | 1024 | 128 | 2 | 1 | 62 | 8.10 | 247 | 1.0× |
| 短输入大 batch | 1024 | 128 | 8 | 4 | 129 | 9.54 | **838** | 3.39× |
| 长输入大 batch | 4096 | 1024 | 8 | 4 | 1536 | 10.62 | 753 | 3.05× |

- 批量 8 相比批量 2 吞吐扩展约 3.0–3.4×（理论上限 4×），中等批量下扩展性良好，没有出现 EP 通信瓶颈。
- 长输入行与 8.2 / 10.2 节的 BF16 数值（10.62 与 9.80）来自不同轮次实测，相差约 8%，属多次运行的正常波动区间；各表内部均为同轮可比数据。
- 输入从 1K 增至 4K，Decode 时延仅增约 12%；attention 走 PA 分块，每步只读活跃 KV cache，序列增长被通信开销稀释。
- Prefill 在长输入下超线性增长（输入翻 4 倍时 Prefill 涨约 12 倍），attention 的二次复杂度在此规模占主导。
