# Gemma-4 模型优化报告

> 生成时间：2026-05-21
> 优化执行者：agent

---

## 1. 概述

Gemma-4 是 Google 于 2026 年开源的多模态稀疏 MoE 大语言模型，本样例仅适配 Language MoE Decoder 路径，针对昇腾 NPU 完成并行切分、KVCache、算子融合与图模式四个方向的优化适配，最终在 Atlas A3 八卡部署下达到稳态 Decode 单步约 10 ms、Prefill 约 76 ms 的性能水平。

- HuggingFace: https://huggingface.co/google/gemma-4-26B-A4B
- 架构: MoE（128 experts top-8，sliding/full 双模式 GQA Attention）
- 总参数量: 26.5B（活跃 ~3.8B/token）
- 硬件平台: Atlas A2 / Atlas A3
- 部署规模: 8 卡，single-node
- 量化模式: BF16

---

## 2. 模型结构

本样例覆盖的是 Gemma-4 的语言解码器部分；视觉编码器不在改造范围。语言解码器最显著的两个结构特征是双模式 Attention 和"Dense MLP 与 MoE 并行（输出相加）"的层结构，这两点共同决定了 KVCache 与算子融合阶段的处理思路。

```
Token Embedding (vocab=262144, 与 LM Head 共享权重)
└─ Decoder Block × 30
     ├─ Attention（双模式，按层类型路由）
     │    ├─ Sliding (25 层): GQA, head_dim=256, sliding_window=1024
     │    └─ Full (5 层): GQA, head_dim=512, K 与 V 共享投影
     ├─ Dense MLP: gate+up+down, intermediate=2112, GELU
     ├─ MoE: 128 experts, top-8, intermediate=704, GEGLU
     │    （每层同时含 Dense MLP 与 MoE，并行计算后相加，而非逐层交替）
     └─ 7 × RMSNorm + layer_scalar
└─ LM Head (与 Embedding tied)
```

关键特殊点：

- 双模式 Attention 导致 KV cache 维度异构，sliding 层与 full 层各自独立管理 block pool；Full 层每 6 层出现一次，分布在固定位置。
- Full 层启用 `attention_k_eq_v`，Key 与 Value 共用同一份投影，整段省去独立 v_proj 计算与存储。
- QK RMSNorm 已对 Q/K 归一化，Flash Attention 调用直接使用 `softmax_scale=1.0`，无需额外 1/√d 缩放。
- 大词表 262144，Embedding 与 LM Head tied weights，并行切分时统一沿 vocab 维度切。

---

## 3. 性能基线

基线在 Atlas A2 八卡 eager 模式下采集，作为后续并行切分、KVCache、算子融合、图模式四个方向逐项叠加的对比基准。

| 指标 | 值 | 测试条件 |
|------|-----|---------|
| Prefill 耗时 (ms) | 312.51 | BS=8, input_len=256, BF16 |
| Decode 单步耗时 (ms) | 98.47 | BS=8, BF16 |

基线 yaml：`config/gemma4_rank_8_8ep_decode.yaml`（exe_mode=eager）。基线建立后，Decode 单步时延 98 ms 为主要待优化对象。

### 3.1 精度基线

测试输入：仓内 `dataset/default_prompt.json` 内置 prompt（关于 Transformer Attention 公式的简短问答），输入长度截断至 256 tokens，BS=8。

基线输出（首条请求生成 32 tokens）：

> "An attention function can be described as a query, keys, values, and an output, where the query, keys, values, and output are all vectors."

后续优化阶段的精度验证以该 token 序列为对照，要求字节级一致。

---

## 4. 并行切分

本样例在八卡部署下采用"MoE 走专家并行、Embedding 与 LM Head 沿词表切分、Attention 与 Dense MLP 保持单卡"的混合并行策略，目的是在显存允许范围内最大化 Decode 吞吐并避免引入冗余通信。

部署拓扑（`config/gemma4_rank_8_8ep_decode.yaml`）：

| 模块 | 切分 | 通信形式 |
|------|------|---------|
| Attention | 单卡（沿 batch 做数据并行 DP）| 无 |
| Dense MLP | 单卡 | 无 |
| MoE | 专家并行（EP），128 专家切到 8 卡，每卡承载 16 路由专家 | dispatch_v2 / combine_v2（decode）或 AllToAll 两轮（prefill）|
| Embedding | 沿词表维度切到 8 卡 | 取 Embedding 时按 token 落点选择 |
| LM Head | 沿词表维度切到 8 卡 | AllGather 汇总 logits |

技术依据：

- Attention 采用数据并行：Full 层 KV head 数为 2，张量并行（TP）切分上限受限，且本样例规模下 Decode 吞吐受 batch 主导，DP 模式无需引入额外通信。
- Dense MLP 保持单卡：intermediate 维度仅为 2112，再做 TP 切分后矩阵 cube 利用率反而下降。
- MoE 采用专家并行：128 专家可均匀分给 8 卡，每卡 16 个专家，是该规模下显存与通信平衡的最优配置。
- Embedding 与 LM Head 沿词表切分：词表规模 262144 较大，TP 切分后单卡显存可控；LM Head 出 logits 时通过 AllGather 汇总。

该改造保持精度不变，相对基线时延几乎无变化（Prefill 313 ms / Decode 97 ms），主要意义在于打开后续 Paged Attention、Flash Attention 等优化所依赖的多卡部署形态。

---

## 5. KVCache 与 Attention

本样例 KV Cache 接入推理框架的 Paged Attention 管理，并按层类型选择 Flash Attention 的 layout，目的是在双模式 Attention 结构下统一显存管理、消除 Decode 单步主要的访存与计算开销。

### 5.1 KVCache 管理

KV Cache 采用 Paged Attention 方案，由框架统一负责块的分配与回收。sliding 与 full 是两类不同的 attention，缓存长度语义不同，框架按 attention 类型分别管理两套 block pool：sliding 层按 sliding_window 长度约束 quota，full 层按全长度 quota（两类 head_dim 也不同，故每 token 缓存尺寸不同；head_dim 对 Flash Attention layout 选择的影响见 §5.2）。每个 Attention 模块通过 `cache_entries` 声明所需缓存条目，`Gemma4ForCausalLM.get_cache_info()` 统一暴露给框架供初始化。

| 参数 | 值 | 说明 |
|------|-----|------|
| KV 模式 | Paged Attention | 双 attn_type 分别管理 block pool |
| block_size | 128 | 框架默认分页粒度 |
| sliding 层 head_dim | 256 | 缓存长度按 sliding_window=1024 约束 |
| full 层 head_dim | 512 | K 与 V 共享投影（attention_k_eq_v）|

### 5.2 Flash Attention 算子选型

Flash Attention 选用 v2 算子（`npu_fused_infer_attention_score_v2`），Prefill 与 Decode 通过同一接口完成。算子在两类层上的 layout 选择不同：sliding 层走 TND（packed 一维），full 层因头维度更大、暂不在当前 TND 非 MLA 白名单覆盖范围内，过渡形态保留 BNSD。Prefill 路径在 eager 模式调用 torch 算子入口；Decode 路径在 GE 图模式下走 torchair 入口，两条路径共用同名算子但分别走各自的图编译后端。

| 项 | Sliding 层 | Full 层 |
|---|---|---|
| input_layout | TND | BNSD |
| sparse_mode | 4（band，前向窗口受 sliding_window 约束）| Prefill: 3（全因果） / Decode: 0 |
| KV cache 物理视图 | 扁平到 `(blocknum, blocksize, num_kv * head_dim)` | 转置到 `(blocknum, num_kv, blocksize, head_dim)` |

该改造保持精度不变，相对算子融合阶段 Decode 单步从 92 ms 降到约 11 ms，是整段优化收益最大的一步，主要源自 Paged Attention 取代连续缓存后访存模式更紧凑以及 Flash Attention 整图替代手写 attention。若 §9 #1 算子约束后续支持，full 层可统一走 TND 路径，去掉中间 transpose。

### 5.3 MoE Decode 通信适配

MoE Decode 路径使用 MC2 `dispatch_v2 / combine_v2` 算子完成跨卡路由通信，每个 expert 的 token 在 owning rank 单点累加，bf16 reduction 顺序与单卡基线一致；Prefill 路径使用 double routing（双重路由），覆盖变长输入下的 dispatch 需求。

---

## 6. 算子融合

本样例除 §5 引入的 Flash Attention 与 MoE 通信类算子外，进一步替换 Norm、RoPE、MoE 路由与 KV Cache 写入等子链路上的标准 PyTorch 实现，统一走 torch_npu 提供的融合算子。

| 模块 | 实现 | 来源 | 触发位置 |
|------|------|------|---------|
| Residual + RMSNorm | `npu_add_rms_norm` | 算子融合 | 每层 Attention 与 MLP 前后的残差归一化 |
| Sliding RoPE | `npu_rotary_mul`（half 模式）| 算子融合 | sliding 层 Q/K |
| Full partial RoPE | `npu_apply_rotary_pos_emb` 包装 slice/cat | 算子融合 | full 层 Q/K 前 rotary_dim=128 段 |
| MoE Router | `npu_moe_gating_top_k_softmax` | 算子融合 | MoE 路由打分与 top-k |
| MoE 分发 (Decode) | `npu_moe_distribute_dispatch_v2` + `npu_moe_distribute_combine_v2` | KVCache 与 Attention | MoE 跨卡通信 |
| MoE 分发 (Prefill) | double routing 双重路由（`_dispatch_double_routing`）| KVCache 与 Attention | MoE 跨卡通信 |
| Cache 写入 | `npu_scatter_nd_update_` | KVCache 与 Attention | 每层 K/V 写入 paged 缓存 |

sliding 层的融合 RoPE 调用存在一处过渡形态：sliding 路径在 forward 入口是三维 packed 张量，而 `npu_rotary_mul` 在图模式下要求四维输入，因此调用前先零拷贝视图升到四维、调用后还原为三维。若 `npu_apply_rotary_pos_emb` 后续支持 head_dim=256（详见 §9 #2），可去掉该视图绕行。

该改造保持精度不变，相对并行切分阶段 Decode 单步从 97 ms 降到 92 ms，单独贡献约 5% 加速；与 §5 的 Paged Attention 叠加后，Decode 单步降至约 11 ms。

---

## 7. 图模式

本样例支持 GE 图模式与 npugraph_ex 两种后端，覆盖 Decode 阶段；Prefill 保持 eager，避免变长输入导致的图重编译。两种后端通过 yaml `model_config.exe_mode` 切换，modeling 代码共用。

### 7.1 npugraph_ex

| 项 | 配置 |
|----|------|
| 后端 | torch.compile，`backend="npugraph_ex"`，`dynamic=True` |
| 覆盖范围 | Decode 整图，Prefill 保持 eager |
| Flash Attention | Decode 走 `torch.ops.npu.npu_fused_infer_attention_score_v2`（`actual_seq_lengths_kv` 走 List[int] 路径，避开 dynamo 拦截 `aten._local_scalar_dense`）|

| 平台 | Prefill (ms) | Decode 单步 (ms) |
|---|---|---|
| Atlas A3 | 102.29 | 11.59 |

npugraph_ex 后端 Decode 单步约 11.6 ms，略慢于 GE 图模式，作为图模式备选部署路径，与 GE 图模式共用同一份 modeling 代码。

### 7.2 GE 图模式

| 项 | 配置 |
|----|------|
| 后端 | torchair GE 图模式，`fullgraph=True` |
| 覆盖范围 | Decode 整图，Prefill 保持 eager |
| Flash Attention | Decode 走 `torchair.ops.npu_fused_infer_attention_score_v2`，`actual_seq_lengths_kv` 用 Tensor 入图 |
| 编译缓存 | 支持 `enable_cache_compile`（默认关闭），开启后二次启动 warmup 节省约 20× |

| 平台 | Prefill (ms) | Decode 单步 (ms) |
|---|---|---|
| Atlas A2 | 189 | 15.0 |
| Atlas A3 | 76.43 | 10.20 |

图模式适配过程中有两处共用关键改造：RoPE 的 cos/sin 在 `Gemma4RotaryEmbedding.__init__` 阶段通过 `register_buffer` 预计算到最大位置数，forward 内只做 `index_select`，避免运行时 `kv_len.max().item()` 触发的 host 同步；GEGLU 在 Decode 路径下用 `npu_fast_gelu(gate) * up` 替代 `npu_geglu`，规避 `npu_geglu` 在两种图后端下 dispatch 注册不完整的问题（详见 §9 #3）。

GE 图模式是本样例的推荐部署路径：A3 Decode 单步 10.20 ms，相对 A2 ge_graph 同代码部署再提升约 45%。

---

## 8. 累计性能演进

下表按改造叠加顺序列出 Prefill / Decode 时延变化，所有数据均在 BS=8、input_len=256、BF16 条件下采集。基础数据取自 Atlas A2 平台，最末两行为同代码在 Atlas A3 上不同图模式后端下的最终性能。

| 路径 | 关键改造 | Prefill (ms) | Decode (ms) | vs baseline (decode) |
|------|---------|--------------|-------------|----------------------|
| 基线 (A2 eager) | — | 312.5 | 98.5 | 1.0× |
| + 并行切分 | MoE 走专家并行，Embedding/LM Head 沿词表切分 | 313 | 97 | 1.02× |
| + KVCache + Flash Attention | 异构 KVCache 接入，Flash Attention 算子化 | 310 | 97 | 1.02× |
| + 算子融合 | Norm、RoPE、MoE Router 等子链路融合 | 307 | 92 | 1.07× |
| + Paged Attention + FA v2 | KV 改 Paged 管理，FA 升级至 v2，layout 切到 packed 路径 | 303 | 11.4 | 8.64× |
| + Sliding 融合 RoPE | sliding 层融合 RoPE 恢复 | 303 | 11.1 | 8.87× |
| + MoE 通信优化 | Decode 走 MC2 dispatch/combine | 303 | 11.1 | 8.87× |
| 同代码（A3 GE 图模式）| — | 76.43 | 10.20 | — |
| 同代码（A3 npugraph_ex）| — | 102.29 | 11.59 | — |

整段优化的主要收益来源是 §5 的 Paged Attention 与 Flash Attention v2 改造，单步将 Decode 从 92 ms 压到 11 ms 一档；同代码切到 A3 GE 图模式后端进一步将 Decode 单步降至约 10 ms、Prefill 降至 76 ms，是当前推荐的部署形态。

---

## 9. 算子需求

改造过程中遇到以下 head_dim 不在白名单或图后端覆盖不全的算子约束，已用替代实现绕过；若 CANN 后续支持，本样例的部分过渡形态可进一步简化。

| # | 算子 | 当前约束（CANN 9.0.0） | 期望支持 | 简化效果 |
|---|------|---------------------|---------|---------|
| 1 | `npu_fused_infer_attention_score_v2`（TND layout, 非 MLA 场景）| `head_dim ∈ {128, 192, 256}`，非 MLA 场景拒绝 D=512 | 把 D=512 加入 TND 非 MLA 白名单 | full 5 层与 sliding 统一走 TND，去掉中间转置视图 |
| 2 | `npu_apply_rotary_pos_emb`（TND 3D 通用 RoPE）| `head_dim ∈ {64, 128}` | 扩到 256 / 512 | sliding 全维 + full partial 统一用同一融合 RoPE，免去 sliding 当前四维视图绕行 |
| 3 | `npu_geglu` | 图模式 dispatch 缺失 / meta 签名不匹配 | 注册 GE 图模式与 torch.compile 后端 | Decode GEGLU 可统一用 fused 算子，替换当前 `npu_fast_gelu(gate) * up` |

---

## 10. 当前未覆盖项

- **Prefill 图化**：Prefill 仍保持 eager，未做独立图编译，后续在 prompt 形态收敛后可评估接入。
- **Full 层 BNSD 过渡形态**：Full 层因 §9 #1 算子约束保留 BNSD layout，缓存物理视图需 transpose，且 BNSD 要求同批次输入等长，目前不支持变长 batch；若算子白名单后续支持可统一走 TND，去掉 transpose 并打开变长输入。
- **Sliding RoPE 维度过渡**：sliding 层为兼容 `npu_rotary_mul` 图模式四维输入要求，调用前后各做一次零拷贝视图，若 §9 #2 算子约束后续支持可直接走 TND 三维。
- **视觉编码器**：多模态视觉路径未适配，本样例仅覆盖 Language Decoder。
- **量化**：W8A8 / W4A16 等量化路径未接入。
- **Sliding 缓存压缩**：sliding 层目前按 sliding_window 长度直接分配 block pool，后续可评估环形缓存等更紧凑形式以释放显存。
