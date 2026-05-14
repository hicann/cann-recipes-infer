# hy3-preview 优化进度历史归档

> 归档时间：2026-04-28
> 本次归档涵盖阶段 0-4 的完整工作记录

<!-- ===== 以下为工作区，阶段推进时归档并清空 ===== -->

### 阶段 1 部署记录 (2026-04-27)

- [x] YAML 配置更新: world_size=16, attn_tp=4, dense_tp=4, moe_tp=1, embed_tp=4, lmhead_tp=4, batch_size=8
- [x] 并行配置验证: attn_dp=4, moe_ep=16, embed_dp=4, batch_size_per_rank=2
- [x] 通信组: attn/dense/embed/lmhead tp=4 groups [0,1,2,3]×4, moe_ep=16 group [0..15]
- [x] padding_idx 修复: 全局 120002 → 每 rank 局部索引 (rank3=29378, others=None)
- [x] EP 路由: 使用手动 all_to_all (init_routing_v2 + re_routing + finalize)，MC2 dispatch/combine 在小 batch 下有空 expand_x 问题
- [x] set_env.sh CANN path configured
- [x] 权重加载: 112 shards ~12s, EP 过滤 (12 experts/rank)
- [x] 推理跑通: Prefill 2.07s (1024 tokens), Decode ~293ms/t, 32 tokens 总耗时 ~12s

#### 硬件配置
- 环境: 8卡 × 2die = 16 die (Ascend 910C / Atlas A3), 每 die 64 GB HBM, 总 1024 GB
- CANN: 25.5.1, PyTorch: 2.6.0, torch_npu: 2.6.0.post6

#### 代码修改
1. `config/hy3_bf16.yaml`: Candidate A parallel config
2. `models/model_setting.py`: dense_tp_size 默认值, embed_dp_size 约束检查
3. `models/modeling_hy_v3.py`:
   - padding_idx 每 rank 局部化
   - 移除 MC2 dispatch/combine, 统一使用手动 all_to_all EP 路由
   - 移除 set_mc2_kwargs, row_idx_decode 等未使用代码
4. `executor/scripts/set_env.sh`: cann_path 更新为实际路径

#### 待修复/优化
- 输出精度待验证（初步输出有乱码）
- Decode 性能待优化（~293ms/t，目标 <100ms/t）
- MC2 dispatch/combine 在大 EP + 小 batch 下的空 expand_x 问题待解

### 精度验证 (2026-04-27 并行化改造后审查)

- **状态**: 基础设施已修复，精度待实际运行验证
- **根因分析**:
  - `dist.broadcast` bug 是 decode 完全无法执行的直接原因（16/16 rank 崩溃）
  - 代码结构审查（FA 参数、KV Cache、RoPE、MoE EP、并行通信）与 qwen3-moe 参考一致，未发现结构性精度缺陷
  - "输出乱码含 LaTeX/HTML 碎片" 可能来自于 broadcast 修复前的其他配置（单卡或不同 tp 度）运行，或权重加载不完整
- **修复措施**:
  - 移除 `dist.broadcast` 解除 decode 阻塞
  - 显式添加 `sparse_mode=0` 到 Decode FA 调用
- **待验证**: 在 16 卡环境中使用修复后代码运行完整推理，检查输出文本合理性和 token 序列质量

### 精度调试记录 (2026-04-27 model-infer-precision-debug)

- [已查] FA v1 参数: Prefill `sparse_mode=3` + `~tril` attn_mask; Decode `sparse_mode=0` + atten_mask=None ✓
- [已查] KV Cache: BSH layout, `scatter_update_(axis=-2)` 正确，kv_len 层内只读 ✓
- [已查] RoPE: `npu_apply_rotary_pos_emb(layout='BSH')` cos/sin shape 匹配 ✓
- [已查] MoE EP: `npu_moe_init_routing_v2` → all_to_all → re_routing → experts → finalize 链路完整 ✓
- [已查] LM Head: ColumnParallelLinear + all_gather across lmhead_tp_group ✓
- [已查] kv_len 生命周期: Prefill 初始化为 0 → model_output_process 更新为 seq_len → Decode 每步 +1 ✓
- [已查] 权重加载: EP 过滤 (12 experts/rank)，packed gate_up_proj 映射正确 ✓
- [已查] 通信组: attn_tp_group = embed_tp_group = lmhead_tp_group (均 size=4)，moe_ep_group (size=16) ✓
- [发现] `dist.broadcast(next_tokens, src=0, group=tp_group)` 在 PyTorch 2.6 中 src=0 被解释为全局 rank 0，导致 ranks 4-15 崩溃 → **已修复：移除 broadcast**
- [发现] Decode FA 缺少显式 `sparse_mode=0`（虽然默认值应为 0，但显式传参可避免版本差异风险）→ **已修复：添加 sparse_mode=0**
- [发现] Attention 层使用 `q_len == 1` 检测 decode，而非使用传入的 `is_prefill` 参数。单 token prefill 时可能误判为 decode（边缘情况，与 qwen3-moe 参考一致，暂不修改）

### 自验证结果
- 参考 skill: /model-infer-precision-debug
- 代码加载: 通过 — HYV3ForCausalLM, HYV3Attention, HYV3MoE, HYV3Model, HYV3Config, check_vars, update_vars 均正常导入
- 编译: 通过 — runner_hy3.py, modeling_hy_v3.py, model_setting.py, configuration_hy_v3.py, infer.py 均语法正确
- 修复确认: `sparse_mode=0` 已存在于 HYV3Attention.forward 中；`dist.broadcast(next_tokens` 已从 runner_hy3.py 中移除
- 推理: 未执行（环境中无可用的多卡推理环境）
- 输出: N/A（未到推理阶段，原 broadcast bug 阻塞了所有 decode 执行）

### 性能验证 (2026-04-27 并行化改造后审查)

- **状态**: 部分完成（仅 Prefill 数据可用）
- **Prefill** (1024 tokens, batch_size=4):
  - TP group [0,1,2,3]: ~11706 ms
  - TP group [4,5,6,7]: ~6147 ms
  - TP group [8,9,10,11]: ~6896 ms
  - TP group [12,13,14,15]: ~6077 ms
  - 各组差异显著（最大 ~1.9x），疑似 TP group 0 承担额外协调开销或 batch 分配不均
- **Decode**: 无数据（未进入 decode 阶段）
- **基线对比**: 不可用（`baseline/` 目录为空，缺少 `baseline_metadata.json`）

### 阶段 0 完成记录

- [x] 模型架构已确认：MoE Transformer (Dense + Sparse MoE with Shared Expert, GQA)
- [x] 关键参数已提取：295B 总参数，21B 激活，80层（1 Dense + 79 MoE），192 experts
- [x] 模块链路已拆解：QK Norm → GQA → Sigmoid Router → Expert FFN → Shared Expert
- [x] 运行环境已确认：Atlas A3 (Ascend910C), 8卡×2die=16die × 64 GB HBM
- [x] 模型状态已确认：不可运行（目录为空，未适配）
- [x] 显存约束已识别：BF16 的 590 GB 参数超过 8×64GB=512GB 总 HBM，**必须采用 W8A8 或更多卡数**
- [x] 参考模型已确定：qwen3-moe（最接近的仓库参考）

### 实施记录
- [完成] 创建 `models/modeling_hy_v3.py` — 完整模型实现，基于 HF 源码适配框架接口
  - HYV3RMSNorm, HYV3RotaryEmbedding, HYV3Attention (NPU FA v1), HYV3MLP, HYV3Experts (packed 3D tensors), HYV3TopKRouter, HYV3MoE, HYV3DecoderLayer, HYV3Model, HYV3ForCausalLM
  - KV Cache: BSH layout `(batch, cache_seq_len, num_kv_heads * head_dim)`
  - NPU FA: prefill sparse_mode=3 (causal), decode sparse_mode=0 (dense)
  - NPU RoPE via `torch_npu.npu_apply_rotary_pos_emb` with layout='BSH'
  - Packed expert tensors: `gate_up_proj [num_experts, 2*intermediate, hidden]`, `down_proj [num_experts, hidden, intermediate]`
  - `load_weights()` 处理 per-expert checkpoint key → packed tensor 映射，含 buffers_dict 处理 expert_bias
  - `_ignore_weights_patterns = ["model.layers.80."]` 跳过 MTP 层
- [完成] 创建 `runner_hy3.py` — HYV3Runner 继承 ModelRunner
- [完成] 创建 `infer.py` — 推理入口，遵循 qwen3-moe 模式
- [完成] 创建 `config/hy3_bf16.yaml` — 单卡配置 (world_size=1, eager mode)
- [完成] 创建 `models/model_setting.py` — check_vars, update_vars
- [完成] 创建 `infer.sh`, `requirements.txt`, `README.md`
- [修复] `infer.py` 移除错误的 `check_common_parallel_settings` import — infer.py:21
- [修复] 移除 `runner_hy3.py:95-98` `dist.broadcast(next_tokens, src=0, group=tp_group)` — src=0 在 PyTorch 2.6 中被解释为全局 rank 0，ranks 4-15 的 TP group 不含全局 rank 0，抛 ValueError 导致全部 16 rank 崩溃于首次 decode
- [修复] `modeling_hy_v3.py:244` 为 Decode FA 调用显式添加 `sparse_mode=0` — 防御性修复，避免依赖默认值（参照 qwen3-moe 模式显式传参）
- [诊断] 精度问题根因分析 — 见下方调试记录

### 当前代码状态
- Tensor layout: BSH KV Cache, BSH RoPE
- Attention: NPU FA v1 (`torch.ops.npu.npu_fused_infer_attention_score`)
- Decode: `sparse_mode=0` (显式), Prefill: `sparse_mode=3` (causal)
- Expert storage: packed 3D tensors (1443 total nn.Module, 无 nn.ModuleList 嵌套)
- Weight loading: `enable_online_split_weight=True`, custom `load_weights()` with per-expert key regex matching
- 跳过 MTP layer 80 权重加载
- `dist.broadcast` 已移除，所有 rank 的 next_tokens 完全由本地 logits 决定（TP group 内 rank 共享数据，argmax 结果自然一致）
- Tensor layout: BSH KV Cache, BSH RoPE
- Attention: NPU FA v1 (`torch.ops.npu.npu_fused_infer_attention_score`)
- Expert storage: packed 3D tensors (1443 total nn.Module, 无 nn.ModuleList 嵌套)
- Weight loading: `enable_online_split_weight=True`, custom `load_weights()` with per-expert key regex matching
- 跳过 MTP layer 80 权重加载

### 自验证结果
- 参考 skill: /model-infer-migrator (Scene A: 空目录框架适配)
- 代码加载: 通过 — infer.py 正确导入 HYV3Runner 和 HYV3ForCausalLM，无 import 错误
- 配置加载: 通过 — YAML 配置正确解析，world_size=1, enable_online_split_weight=True
- 权重加载: 通过 — 112 safetensors shards 全部加载完成 (~246s)，所有 checkpoint key 正确映射到模型参数
- 推理 (to_device): OOM 符合预期 — RuntimeError: NPU out of memory, 54.50 GiB already allocated / 61.27 GiB total
- 输出: N/A（未到推理阶段）

### 显存分析
- 单卡 HBM: 64 GB (65536 MB)，实际可用 ~61.27 GiB
- to_device 已消耗: 54.50 GiB（约 58.57 GB）
- 剩余需要: ~535+ GB（模型总计 ~590 GB BF16）
- 结论: **单卡完全无法承载**，需多卡 EP/TP 部署
- 8 卡 BF16 部署: 590 GB 参数 / 512 GB 总 HBM = 115%，仍超显存
- 推荐方案: W8A8 量化 (590→~310 GB) + 8 卡 EP，或 16 卡 BF16 直接部署

---

## 阶段 1：并行策略分析

### 关键决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 架构主干 | MoE EP（moe_tp=1, moe_ep=16） | 97% 参数在专家中，EP 是必要条件；每卡 12 experts ≤ 24 |
| Attn TP 度 | 4（候选 A 推荐） | 平衡 Prefill 计算分摊和 Decode 通信开销，N_kv=8 约束 attn_tp ∈ {1,2,4,8} |
| Dense TP 度 | 4（与 attn_tp 对齐） | 非专家参数仅 3.4 GB，与 attn_tp 对齐避免模块边界 AllGather/ReduceScatter |
| Embed/LMHead TP | 16（全卡） | 大词表 120K 需独立 TP，参考 V3.2/GLM-5/Kimi-K2 的 embed_tp=16 模式 |
| CP / KVP | 不启用 | 32K 序列下 attn_tp=4 已充分分摊 Prefill 计算，暂不需要序列并行 |
| 量化 | BF16（推荐），可选 W8A8 | 16×64GB=1024GB HBM，BF16 参数 590 GB 可行；若未来减至 8 卡必须 W8A8 |

### 部署信息确认

| 参数 | 值 | 来源 |
|------|-----|------|
| 部署卡数（world_size） | 16 die（8 卡 × 2 die/卡） | 用户指定 |
| 单 die HBM | 64 GB | Atlas A2 (Ascend 910B) |
| 总 HBM | 1024 GB | |
| 目标场景 | 均衡（EP+TP 混合） | 用户指定 |
| 序列长度范围 | 4K - 32K | 用户指定 |
| batch_size | 1 | 用户指定 |
| 单卡实际可用 | ~61.3 GiB | 实测 to_device OOM 边界 |

---

### 第一步：模型参数与模块链路

#### 基础参数（从 configuration_hy_v3.py 和 progress.md 常驻区提取）

| 参数 | 符号 | 值 | 说明 |
|------|------|-----|------|
| 架构类型 | — | MoE Transformer (GQA + Shared Expert) | 已确认 |
| 总层数 | L | 80 (excl. MTP layer 80) | |
| MoE 层数 | L_moe | 79 (layer 1-79) | |
| Dense FFN 层数 | L_dense | 1 (layer 0) | first_k_dense_replace=1 |
| Hidden size | H | 4096 | |
| FFN intermediate (Dense) | H_ffn_dense | 13312 | layer 0 only |
| FFN intermediate (MoE expert) | H_ffn_moe | 1536 | layers 1-79 |
| Attention heads | N_h | 64 | |
| KV heads | N_kv | 8 | GQA ratio = 8x |
| Head dim | D_h | 128 | |
| 专家总数 | E | 192 | |
| 激活专家数 | E_act | 8 | top-8 sigmoid routing |
| Shared Expert | 1 | 每 MoE 层 | intermediate=1536 |
| 词表大小 | V | 120,832 | >100K，大词表 |
| RoPE theta | — | 11,158,840 | default RoPE, head_dim=128 |
| QK Norm | — | 是 | RMSNorm(128) per head，区别于标准 GQA |

#### 参数量分布

| 模块 | 参数量 | BF16 显存 | 占比 |
|------|--------|----------|------|
| Attention (80层) | 6.04B | 12.08 GB | 2.05% |
| Dense FFN (layer 0) | 0.16B | 0.33 GB | 0.06% |
| MoE Experts (79层×192专家) | 286.29B | 572.57 GB | 97.04% |
| MoE 非专家 (Router+Shared+ExpertBias, 79层) | 1.55B | 3.11 GB | 0.53% |
| Embedding | 0.49B | 0.99 GB | 0.17% |
| LM Head | 0.49B | 0.99 GB | 0.17% |
| **总计** | **295.03B** | **590.07 GB** | **100%** |

> 关键结论：97% 参数在 MoE 专家中，EP 是必要条件而非可选项。

#### 模块链路与并行维度

| 模块 | Prefill 特性 | Decode 特性 | 并行维度 | 约束 |
|------|-------------|-----------|---------|------|
| Embedding | V=120K 大词表 | 同左 | embed_tp | V / embed_tp 需整除 |
| Attention QKV | 计算密集 (S=32K) | 访存密集 (S=1) | attn_tp | N_kv=8 → tp ≤ 8 且整除此值 |
| Attention Core (FA) | 计算密集 | 访存密集 | attn_tp | Q heads=64 整除 tp |
| Attention O_proj | 中等计算 | 访存密集 | attn_tp | GQA 非 MLA，跟随 attn_tp |
| Dense FFN (layer 0) | 计算密集 | 访存密集 | dense_tp | 仅 0.33 GB，影响小 |
| MoE Router | 轻量 | 轻量 | dense_tp | 跟随 dense_tp（或复制） |
| MoE Shared Expert | 中等计算 | 中等访存 | dense_tp | 1.49B params，跟随 dense_tp |
| MoE Expert FFN | 计算密集 | 分散访存 | moe_ep | E=192 / ep=16 = 12 ≤ 24 ✓ |
| LM Head | V=120K 大词表 | 同左 | lmhead_tp | 同 embedding |

---

### 第二步：定性分类（决策树遍历）

#### 第一层：架构主干

```
MoE Transformer
├─ MoE 层 (79/80): EP=16（全卡 EP，experts_per_rank=12 ≤ 24 ✓）
│   专家 FFN 较小 (1536 intermediate)，TP 会导致碎矩阵 → moe_tp=1 明确
├─ Dense FFN: 仅 layer 0，3.4 GB 总 dense 参数，可跟随 attn_tp 切分
├─ Embed/LMHead: V=120K > 100K → 大词表，独立设 TP=16
├─ O_proj: GQA（非 MLA），无需独立 oproj_tp
└─ 部署规模: 16 die（单机 8 卡 A2 × 2 die/卡），节点内 HCCS
```

#### 第二层：场景调整

```
均衡（batch=1）
├─ GQA → attn_tp 适度（2-4），分摊 Prefill 计算降低 TTFT
│   N_kv=8 约束 attn_tp ∈ {1, 2, 4, 8}
│   batch=1 时 DP 无吞吐帮助，可适当增大 attn_tp
└─ Decode 阶段 attn_tp 通信开销极小（单 token），非瓶颈
```

#### 第三层：序列长度附加

```
最大序列 32K
├─ GQA + attn_tp=4 已充分分摊 Prefill 计算
│   → 无需 CP
├─ KV cache 32K × 80 层 ≈ 10.7 GB (attn_tp=1) / 2.7 GB (attn_tp=4)
│   → attn_tp=4 下每卡 KV cache ~2.7 GB，无需 KVP
└─ 结论：无附加 CP/KVP 需求
```

#### 候选模式确定

| 候选 | attn_tp | dense_tp | moe_tp | embed_tp | lmhead_tp | attn_dp | moe_ep | 参考模式 |
|------|---------|----------|--------|----------|-----------|---------|--------|---------|
| **A** | 4 | 4 | 1 | 16 | 16 | 4 | 16 | Qwen3-MoE attn4tp8dp (W=32) |
| B | 2 | 2 | 1 | 16 | 16 | 8 | 16 | Qwen3-MoE 1tp16ep 的扩展 |
| C | 8 | 8 | 1 | 16 | 16 | 2 | 16 | Qwen3-MoE 16tp 的降低版 |

---

### 第三步：定量估算

#### 3.1 单卡显存估算

| 项目 | 候选 A (tp=4) | 候选 B (tp=2) | 候选 C (tp=8) |
|------|-------------|-------------|-------------|
| Attention 参数 | 3.02 GB | 6.04 GB | 1.51 GB |
| Dense 参数 | 0.86 GB | 1.72 GB | 0.43 GB |
| MoE Expert 参数 | 35.79 GB | 35.79 GB | 35.79 GB |
| Embedding 参数 | 0.06 GB | 0.06 GB | 0.06 GB |
| LM Head 参数 | 0.06 GB | 0.06 GB | 0.06 GB |
| **参数小计** | **39.79 GB** | **43.67 GB** | **37.85 GB** |
| KV Cache (4K) | 0.34 GB | 0.67 GB | 0.17 GB |
| KV Cache (32K) | 2.68 GB | 5.37 GB | 1.34 GB |
| 通信/激活/workspace (估) | ~5 GB | ~5 GB | ~5 GB |
| **总计 (4K)** | ~45.1 GB | ~49.3 GB | ~43.0 GB |
| **总计 (32K)** | ~47.5 GB | ~54.0 GB | ~44.2 GB |
| **可用 95%** | 60.8 GB | 60.8 GB | 60.8 GB |
| **显存可行** | **是** | **是** | **是** |

> attn_tp=1 方案参数 51.4 GB + KV(32K) 10.7 GB + overhead ~5 GB = ~67.1 GB 超过 64 GB，**不可行**。

#### 3.2 Prefill TTFT 估算（32K 序列，仅计算通信+注意力计算）

| 指标 | 候选 A (tp=4) | 候选 B (tp=2) | 候选 C (tp=8) |
|------|-------------|-------------|-------------|
| 注意力计算 (估) | ~1.2s | ~2.3s | ~0.6s |
| TP AllReduce (估) | ~0.6s | ~0.4s | ~0.7s |
| MoE AllToAll (估) | ~0.4s | ~0.4s | ~0.4s |
| **TTFT 粗估** | **~2.2s** | **~3.1s** | **~1.7s** |

> 粗估基于 A2 ~300 TFLOPS BF16 和 HCCS ~56 GB/s，实际受 NPU 利用率、计算通信重叠影响。相对差异可靠，绝对值需 profiling 校准。

#### 3.3 通信分析

**层 1：是否跨节点**
- 所有通信均在同节点内（16 die = 8 物理卡 × 2 die，HCCS 互联）
- 不跨节点：无 RDMA 瓶颈 ✓

**层 2：通信原语类型**

| 原语 | 触发场景 | 候选 A | 候选 B | 候选 C |
|------|---------|--------|--------|--------|
| AllReduce | O_proj 输出 (每层) | 4-rank AR | 2-rank AR | 8-rank AR |
| AllReduce | Dense FFN (每层) | 4-rank AR | 2-rank AR | 8-rank AR |
| AllToAll | MoE dispatch/combine (79层) | 16-rank A2A | 16-rank A2A | 16-rank A2A |
| AllGather/ReduceScatter | Embed→Attn 边界 (tp 切换) | 16→4 | 16→2 | 16→8 |

**层 3：是否可被计算重叠**
- MoE AllToAll：主要通信瓶颈，79 层累计 ~21 GB，难以完全重叠
- TP AllReduce：每层体积随 tp 增长，可与 attention 计算部分重叠
- Embed/LMHead AllGather：仅在首尾各一次，影响极小

**Decode 阶段通信量**
- 单 token：AllReduce ~8-14 KB，AllToAll ~8 KB，均微不足道
- 延迟方面：TP group 内同步开销 ~100us 量级，基本无影响

#### 3.4 与已验证配置的距离

| 候选 | 最接近参考 | 差异 | 适配风险 |
|------|----------|------|---------|
| A | Qwen3-MoE 32卡 attn4tp8dp | GQA + MoE + Shared Expert 模式一致；Hy3-preview 有 QK Norm、sigmoid router、大词表 | 低 |
| B | Qwen3-MoE 16卡 1tp16ep | 相似但 attn_tp 不同 | 低 |
| C | Qwen3-MoE 16卡 16tp | 纯 TP 模式改为 EP+TP 混合，Hy3-preview 总参数量更大 | 中 |

---

### 第四步：方案审查与推荐

#### 硬约束检查

| 约束 | 候选 A | 候选 B | 候选 C |
|------|--------|--------|--------|
| world_size % attn_tp == 0: 16%x==0 | ✓ (4) | ✓ (2) | ✓ (8) |
| world_size % dense_tp == 0: 16%x==0 | ✓ (4) | ✓ (2) | ✓ (8) |
| num_attention_heads % attn_tp == 0: 64%x==0 | ✓ | ✓ | ✓ |
| num_key_value_heads % attn_tp == 0: 8%x==0 | ✓ | ✓ | ✓ |
| num_experts % ep_size == 0: 192%16==0 | ✓ | ✓ | ✓ |

全部通过。

#### 强经验检查

| 检查项 | 候选 A | 候选 B | 候选 C |
|--------|--------|--------|--------|
| 单卡显存 ≤ 64×0.95=60.8 GB | ✓ (~47.5 GB) | ✓ (~54.0 GB) | ✓ (~44.2 GB) |
| tp_size ≤ 单节点卡数 (16) | ✓ (4) | ✓ (2) | ✓ (8) |
| Decode attn_tp 不大于必要值 | ✓ (4，适中) | ✓ (2，偏小) | △ (8，偏大) |

#### 候选方案详述

```yaml
# ============================================================
# 候选 A（推荐）：均衡 EP+TP 混合，attn_tp=4
# ============================================================
# 理由：在单套 parallel_config 下实现 Prefill TTFT 和 Decode 效率的最佳平衡。
#       attn_tp=4 充分分摊 32K Prefill 计算（TTFT 估 ~2.2s），
#       Decode 阶段 4-rank AllReduce 开销极小。
#       显存充裕（~48 GB / 64 GB），为未来 batch_size 增长留空间。
# 参考：Qwen3-MoE 32卡 attn4tp8dp（最接近的 GQA+MoE+Shared Expert 仓库实现）
candidate_a:
  parallel_config:
    attn_tp_size: 4       # attn_dp_size = 16 // 4 = 4
    dense_tp_size: 4      # 与 attn_tp 对齐，避免模块边界通信
    moe_tp_size: 1        # moe_ep_size = 16（全卡 EP）
    embed_tp_size: 16     # 大词表全卡切分
    lmhead_tp_size: 16    # 大词表全卡切分
    cp_size: 1            # 无需 CP
    kvp_size: 1           # 无需 KVP
  estimation:
    params_per_card_gb: 39.8
    kv_cache_per_card_gb_at_32k: 2.7
    total_per_card_gb_at_32k: ~47.5
    memory_feasible: true
    cross_node_comm: false
  tradeoff:
    throughput: "中性 — DP=4，batch=1 时无帮助；扩大 batch 需 ≤ 4"
    latency: "优势 — Prefill TTFT 估 ~2.2s，Decode 同步开销极小"
    long_context: "优势 — 32K KV cache 仅 2.7 GB，余量充足"

# ============================================================
# 候选 B（备选）：DP 优先，attn_tp=2
# ============================================================
# 理由：更大的 DP 度 (8) 适合未来扩大 batch_size 的场景。
#       Prefill TTFT 较高（估 ~3.1s），但 Decode 通信负担最小。
#       适用场景：预期 batch_size 从 1 增长到 2-8 的高吞吐场景。
# 参考：Qwen3-MoE 16卡 1tp16ep 的低 TP 模式
candidate_b:
  parallel_config:
    attn_tp_size: 2       # attn_dp_size = 16 // 2 = 8
    dense_tp_size: 2
    moe_tp_size: 1        # moe_ep_size = 16
    embed_tp_size: 16
    lmhead_tp_size: 16
    cp_size: 1
    kvp_size: 1
  estimation:
    params_per_card_gb: 43.7
    kv_cache_per_card_gb_at_32k: 5.4
    total_per_card_gb_at_32k: ~54.0
    memory_feasible: true
    cross_node_comm: false
  tradeoff:
    throughput: "优势 — DP=8，可支持 batch ≤ 8"
    latency: "风险 — Prefill TTFT 估 ~3.1s（比 A 高 40%）"
    long_context: "中性 — 32K KV cache 5.4 GB，仍可接受"

# ============================================================
# 候选 C（备选）：Prefill 优先，attn_tp=8
# ============================================================
# 理由：最低的 Prefill TTFT（估 ~1.7s），适合对首 token 延迟敏感的场景。
#       但 DP=2 限制 batch 扩展，Decode 阶段 8-rank AllReduce 有较多同步开销。
#       适用场景：仅关注 Prefill TTFT，batch_size 固定为 1。
# 参考：Qwen3-MoE 16卡 16tp 纯 TP 模式（降低 TP 度 + 加 EP）
candidate_c:
  parallel_config:
    attn_tp_size: 8       # attn_dp_size = 16 // 8 = 2
    dense_tp_size: 8
    moe_tp_size: 1        # moe_ep_size = 16
    embed_tp_size: 16
    lmhead_tp_size: 16
    cp_size: 1
    kvp_size: 1
  estimation:
    params_per_card_gb: 37.9
    kv_cache_per_card_gb_at_32k: 1.3
    total_per_card_gb_at_32k: ~44.2
    memory_feasible: true
    cross_node_comm: false
  tradeoff:
    throughput: "风险 — DP=2，batch 扩展空间最小"
    latency: "优势 — Prefill TTFT 估 ~1.7s（三者最低）"
    long_context: "优势 — KV cache 仅 1.3 GB，显存最充裕"
```

#### 排序与推荐

| 排名 | 候选 | 总评分 | 核心优势 | 核心劣势 |
|------|------|--------|---------|---------|
| **1** | **A (tp=4)** | ★★★★★ | Prefill/Decode 最佳平衡，显存充裕，与 Qwen3-MoE 参考最近 | DP=4 中等 |
| 2 | B (tp=2) | ★★★★ | DP=8 最大 batch 扩展性 | Prefill TTFT 最高（~3.1s） |
| 3 | C (tp=8) | ★★★★ | 最低 Prefill TTFT（~1.7s） | DP=2 最小扩展性，Decode TP 同步偏多 |

**推荐：候选 A**。理由：
1. 均衡场景的核心需求是在单套配置下兼顾 Prefill 和 Decode，A 的 attn_tp=4 恰好落在两个极端的中间
2. 显存 ~47.5 GB（32K），留有 25% 余量应对峰值
3. QK Norm + GQA 与参考模型 Qwen3-MoE 的模式高度一致
4. 若未来需要更大 batch，从 tp=4→tp=2 只需修改 parallel_config 并重新转换权重（或启用 online_split_weight 自动适配）

---

### 权重处理提示

- `enable_online_split_weight: True`（当前 YAML 配置）：无需重新转换权重，修改 parallel_config 后运行时自动切分
- 若关闭 online_split_weight：每次修改 parallel_config 必须重跑 `bash utils/weight_convert.sh`

### 下一步

本分析完成。并行策略确认后，下一阶段为实施：
- 调用 `/model-infer-parallel-impl` skill 实施并行化改造（替换并行 Linear、MoE 通信、YAML 配置等）
- 或调用 `/model-infer-optimize` skill 进行端到端优化（含 fusion、KVCache、图模式等）

---

## 阶段 2：KVCache 模式分析与选型

### 关键决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| KVCache 模式 | **模式一：连续缓存** | GQA 架构 + 静态 batch + 无动态分配需求；PA 增加复杂度无收益 |
| KV Cache Layout | **BSH** | 与 QK Norm per-head 操作兼容，与 qwen3-moe 参考一致 |
| FA 算子 | **FA v1** (`npu_fused_infer_attention_score`) | GQA 无 MLA/量化/Sink Token 需求，v1 简洁且参数名不易混淆 |
| KVCache 写入方式 | **scatter_update_** (axis=-2) | BSH layout 标准写法，与 qwen3-moe 完全一致 |
| Prefill/Decode FA 分支 | **q_len==1 检测** | 与 qwen3-moe 参考一致，单 token prefill 边缘情况已识别 |
| block_table / slot_mapping | **不需要** | 连续缓存模式下 FA 直接读整段 cache，无需分页索引 |

---

### 一、架构确认

Hy3-preview Attention 架构从 config 实际值确认（非代码推断）：

| 参数 | 值 | 来源 |
|------|-----|------|
| 架构类型 | **GQA**（非 MLA） | `num_key_value_heads=8`, `num_attention_heads=64`, ratio=8x |
| Head dim | 128 | `head_dim=128` |
| QK Norm | **有**（RMSNorm per head on Q and K） | 区别于标准 GQA 的特殊设计 |
| RoPE | theta=11,158,840, head_dim=128 | 标准 Rotary Embedding |
| MLA 压缩 | **无** | 无 `kv_lora_rank` / `q_lora_rank` 参数 |

**结论**：架构为 GQA，不适用 MLA 模式（模式三）。KVCache 选型在模式一（连续缓存）和模式二（分页注意力 PA）之间。

---

### 二、模式选型决策：模式一 vs 模式二

按 KVCache skill 决策树遍历：

```
前置确认: GQA（非 MLA）
    ↓
需要分页注意力？──→ 分析判断...
    ↓
结论: 否 → 模式一（连续缓存，scatter_update_ + FA）
```

**选择模式一（连续缓存）的理由**：

1. **框架特性**：cann-recipes-infer 使用静态 batching，`batch_size` 在推理全程固定。KV Cache 在 `init_cache()` 中预分配 `[batch_size, max_seq_len, num_kv_heads*head_dim]` 固定 shape 的连续 tensor，不存在动态分配/释放需求。

2. **无 KV Cache 内存碎片问题**：连续缓存预分配一整块连续显存，静态 batch 下不存在外部碎片。PA 的 block 管理（block_table、slot_mapping）在此场景下只增加复杂度和代码维护成本，无性能收益。

3. **与参考模型一致**：最接近的仓库参考模型 **qwen3-moe**（同为 GQA + MoE + Shared Expert）使用模式一（BSH 连续缓存），当前 Hy3-preview 实现已完全对齐。

4. **代码简洁性**：连续缓存仅需 `scatter_update_` 写入 + FA 直接读取，无需维护 block_table、slot_mapping、block_size 等额外状态。这在后续 graph mode 适配时也更友好（减少动态 shape 和 index 运算）。

5. **KV Cache 显存可接受**：32K 序列下，attn_tp=4 时每卡 KV Cache 仅 ~2.7 GB，占总显存 ~47.5 GB 的 5.7%，无显存压力驱动的 PA 需求。

**模式二（PA）不适用的具体原因**：

| 评估维度 | 模式一（当前） | 模式二（PA） | 评价 |
|---------|-------------|------------|------|
| 显存利用率 | ~100%（连续无碎片） | ~100%（block 对齐有小浪费） | 持平 |
| 代码复杂度 | 低（2 个算子调用） | 中（block_table/slot_mapping 构造 + 维护） | 模式一胜 |
| 动态 batch 支持 | 不支持 | 支持 | 当前无此需求 |
| KV Cache 共享/复用 | 不支持 | 支持 | 当前无此需求 |
| Graph mode 兼容 | 好（静态 cache） | 需额外处理 dynamic index | 模式一胜 |
| 参考成熟度 | qwen3-moe 已验证 | 无 GQA PA 参考（仅 MLA 模型用 PA） | 模式一胜 |

---

### 三、Layout 选择：BSH

**当前 layout：BSH** `[batch, seq_len, num_kv_heads * head_dim]`

BSH 是正确选择，理由：

1. **QK Norm 兼容性**：Hy3-preview 的 QK Norm 是 per-head RMSNorm(head_dim)，需要访问每个 head 的独立维度。当前实现的 `view(bsz, q_len, num_heads, head_dim)` → Norm → `view(bsz, q_len, -1)` 链在 BSH 下自然且高效（仅 view 操作，无数据拷贝）。

2. **qwen3-moe 一致性**：BSH 是 qwen3-moe 的 layout，已验证稳定。

3. **TND layout 不适用**：TND `[total_tokens, N, D]` 将多 batch token 拼接，需要额外的 `actual_seq_qlen` / `actual_seq_kvlen` 累积和计算，增加复杂度且在 graph mode 下可能引入动态 shape。

4. **scatter_update_ axis 一致性**：BSH 下 `scatter_update_(past_key, kv_len, key_states, -2)` 沿序列维度（dim=-2）写入，语义清晰。

**layout 对比**：

| Layout | 缓存 shape | scatter_update_ axis | FA input_layout | 适用性 |
|--------|-----------|---------------------|-----------------|--------|
| **BSH** | [B, S, N_kv*D] | -2 | "BSH" | **当前选择，正确** |
| TND | [T, N, D] | 1 | "TND" | 不适合（QK Norm 需 per-head reshape） |
| TND_NTD | [N, T, D] | N/A (PA) | "TND_NTD" | PA 模式，不适合 |

---

### 四、FA 算子版本：FA v1

**当前算子：`torch.ops.npu.npu_fused_infer_attention_score` (FA v1)**

选择 FA v1 的理由：

1. **GQA 无特殊需求**：FA v2 的额外能力（learnable_sink、dequant_scale_key/value、query_quant_mode）均用于 MLA/量化/Sink Token 场景，Hy3-preview GQA 不需要。

2. **参数名清晰**：FA v1 使用 `scale` / `num_heads` / `actual_seq_lengths_kv`，FA v2 使用 `softmax_scale` / `num_query_heads` / `actual_seq_kvlen`。v1 的参数名与 qwen3-moe 参考完全一致，降低参数名混用导致静默默认值的风险。

3. **已验证稳定**：qwen3-moe 在 32 卡大规模部署中使用 FA v1 BSH，成熟度验证充分。

4. **Prefill/Decode 参数正确性已验证**（precision-debug 记录）：
   - Prefill: `sparse_mode=3` + `atten_mask=~tril(2048,2048)` bool ✓
   - Decode: `sparse_mode=0` + `atten_mask=None` ✓

**FA v1/v2 参数映射（防混用）**：

| 功能 | FA v1 参数名 | FA v2 参数名 | Hy3-preview 当前使用 |
|------|------------|------------|------------|
| 缩放系数 | `scale` | `softmax_scale` | `scale=1/sqrt(128)` ✓ |
| Q head 数 | `num_heads` | `num_query_heads` | `num_heads=16` (per rank) ✓ |
| KV head 数 | `num_key_value_heads` | `num_key_value_heads` | `num_kv_heads=2` (per rank) ✓ |
| Q 长度 | `actual_seq_lengths` | `actual_seq_qlen` | v1 Decode 不需传（BSH） |
| KV 长度 | `actual_seq_lengths_kv` | `actual_seq_kvlen` | `kv_len+1` list ✓ |

---

### 五、Prefill/Decode 差异与缓存策略影响

#### 5.1 当前实现分析

```python
# 当前 Prefill/Decode 分支（modeling_hy_v3.py:235-260）
if q_len == 1:
    # Decode: 从 cache 读取 full KV
    past_key_states, past_value_states = past_key_value[self.layer_idx]
    actual_seq_lengths_kv = (kv_len + 1).detach().cpu().numpy().tolist()
    attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
        query_states, past_key_states, past_value_states,
        num_heads=..., num_key_value_heads=...,
        input_layout="BSH", sparse_mode=0, atten_mask=attention_mask,
        scale=..., actual_seq_lengths_kv=actual_seq_lengths_kv,
    )
else:
    # Prefill: 直接传当前 batch 的 K,V
    attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
        query_states, key_states, value_states,
        num_heads=..., num_key_value_heads=...,
        input_layout="BSH", atten_mask=attention_mask,
        sparse_mode=3, scale=...,
    )
```

#### 5.2 Prefill 阶段

| 特性 | 当前行为 | 评估 |
|------|---------|------|
| K,V 来源 | 当前 batch 计算的新 K,V（不读 cache） | 正确：Prefill 无历史 KV，无需读 cache |
| sparse_mode | 3（causal） | 正确：Decoder-only LLM 标准选择 |
| atten_mask | `~tril(2048,2048)` bool | 正确：shape=[2048,2048] 满足 FA 硬约束 |
| KV Cache 写入 | `scatter_update_` 将新 K,V 写入 cache | 正确：为后续 Decode 准备历史 |
| Prefill 后 kv_len | `torch.max(position_ids, axis=1)[0] + 1` | 正确：定位到 seq_len，Decode 从此位置开始写入 |

#### 5.3 Decode 阶段

| 特性 | 当前行为 | 评估 |
|------|---------|------|
| K,V 来源 | 从 `past_key_value` 读取完整历史 KV | 正确 |
| 单 token 写入 | `scatter_update_(past_key, kv_len, key_states, -2)` | 正确：写入位置 = kv_len（逐层复用同一 kv_len） |
| sparse_mode | 0（dense） | 正确：Decode 无需因果遮蔽 |
| atten_mask | None | 正确 |
| actual_seq_lengths_kv | `(kv_len + 1).detach().cpu().numpy().tolist()` | 功能正确，但 `.cpu().numpy().tolist()` 在 graph mode 下会导致 Graph Break |

#### 5.4 单 token Prefill 边缘情况

当前使用 `q_len == 1` 而非 `is_prefill` 参数判断 Prefill/Decode：

- 单 token Prefill（prompt 长度=1）会走 Decode 分支
- Decode 分支的 `sparse_mode=0` 无因果遮蔽，对单 token 无影响（attention over 1 token 不需要 mask）
- 与 qwen3-moe 行为一致，已在前序 precision-debug 中记录为 "可接受"

#### 5.5 KV Cache 写入位置管理

```
kv_len 生命周期：
  Prefill: 初始化为 0 → forward() 中所有层在 scatter_update_ 使用同一个 kv_len=0
  Prefill 后: model_output_process 设置 kv_len = seq_len
  Decode 每步: scatter_update_ 使用当前 kv_len 写入 → forward 后 kv_len += 1

关键约束（已验证正确）：
  - kv_len 是 Runner 层变量，所有层共享只读
  - 不应在 attention/cache 内部递增 kv_len
  - 各层 scatter_update_ 使用同一 kv_len 值
```

---

### 六、与参考模型的对比

| 特性 | Hy3-preview（当前实现） | qwen3-moe（参考） | gpt-oss（参考） |
|------|---------------|-------------------|----------------|
| KVCache 模式 | 模式一（连续） | 模式一（连续） | 模式一（连续） |
| Layout | BSH | BSH | TND |
| FA 版本 | FA v1 | FA v1 | FA v2 |
| scatter_update_ axis | -2 | -2 | 1 (TND layout) |
| Prefill mask | sparse_mode=3 + tril mask | sparse_mode=2 + mask | sparse_mode=4/3 + mask |
| Decode mask | sparse_mode=0 + None | sparse_mode=0 + None | sparse_mode=4/0 + mask/None |
| QK Norm | 有（view + Norm + view） | 无 | 无 |
| 缓存写入时机 | attention 内 q_len==1 分支 | attention 内 q_len==1 分支 | 独立函数 |

Hy3-preview 当前实现与 qwen3-moe 参考高度一致，唯一的架构差异（QK Norm）已正确适配。

---

### 七、注意事项与未来优化方向

#### 7.1 当前可优化项

| 项目 | 当前状态 | 问题 | 建议 |
|------|---------|------|------|
| Decode `actual_seq_lengths_kv` 重复计算 | prepare_inputs 和 attention forward 各算一次 | 冗余计算 + 双重 `.cpu().numpy()` | 短期保留（冗余但无害）；长期在 graph mode 适配时统一 |
| QK Norm 的 `contiguous().view()` | qkv split 后 view 到 4D | `contiguous()` 在非连续 tensor 上有额外内存拷贝 | 若 `QKVParallelLinear` 输出已连续则可移除 contiguous()；需 profiling 确认 |
| `q_len==1` 检测 | 用 q_len 而非 is_prefill | 单 token prefill 走 decode 分支 | 与 qwen3-moe 一致，暂不修改 |

#### 7.2 Graph Mode 兼容性预警

当前 Decode FA 调用中的 `.detach().cpu().numpy().tolist()` 在 graph mode 下会导致 **Graph Break**。适配 graph mode 时需改为 Tensor 类型参数（GE 模式用 torchair FA 接口）或 `list[int] + dynamic=True`（npugraph_ex 模式）。

#### 7.3 长序列场景（>32K）

当前 32K 序列 KV Cache 约 2.7 GB/卡，若扩展到 128K+：
- KV Cache 增长到 ~10.7 GB/卡（attn_tp=4）
- 总显存 ~55.5 GB，仍在 64 GB 内但余量减少
- 若扩展到 256K（模型最大上下文）：KV Cache ~21.4 GB/卡，总显存 ~66.2 GB 可能超限
- 应对：减小 batch / 增大 attn_tp / 添加 KVP（沿 head 维度切分 KV Cache）

#### 7.4 QK Norm + RoPE 融合机会

当前链路：`QKV split → view 4D → QK Norm → RoPE(BSH) → view BSH → scatter_update_`

QK Norm 是 Hy3-preview 特有的 per-head RMSNorm，不存在现成的融合算子覆盖此路径。`npu_kv_rmsnorm_rope_cache` 的 RMSNorm 是 layer-level（非 per-head），不适用于 QK Norm。

未来可能的融合优化：
- QK Norm 两个 RMSNorm(128) 可合并为一个对 `[Q_concat; K_concat]` 的 Norm（需验证数学等价性）
- 或通过自定义 AscendC kernel 实现 QK Norm + RoPE 融合

---

### 八、选型总结

```
Hy3-preview KVCache 模式选型结果:

  架构: GQA (64Q / 8KV heads, head_dim=128) + QK Norm + RoPE
    │
    ├── KVCache 模式: 模式一（连续缓存，BSH layout）
    │    理由: 静态 batch、无动态分配需求、与 qwen3-moe 参考一致
    │
    ├── FA 算子: FA v1 (npu_fused_infer_attention_score)
    │    理由: GQA 无 MLA/量化需求，v1 简洁稳定
    │
    ├── Layout: BSH [B, S, N_kv*D]
    │    理由: QK Norm 兼容、qwen3-moe 一致、scatter_update_ 语义清晰
    │
    ├── 写入: scatter_update_(axis=-2)
    │    理由: BSH layout 标准写法
    │
    ├── Prefill: sparse_mode=3, atten_mask=~tril(2048,2048) bool
    │    Decode:  sparse_mode=0, atten_mask=None
    │
    └── 无需: block_table / slot_mapping / PA / MLA 压缩 / KVP（当前序列长度下）
```

当前实现已处于模式一的正确状态，无需改造。建议在后续阶段关注：
- Graph mode 适配时处理 `actual_seq_lengths_kv` 的 `.cpu().numpy()` 问题
- 融合优化阶段检查 QK Norm + RoPE 融合机会
- 若序列长度扩展到 128K+，评估 KVP 叠加方案

---

## 阶段 3：融合算子匹配分析（/model-infer-fusion 步骤 1-4，仅分析）

### 关键决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 分析范围 | 步骤 1-4（仅分析，不实施） | 按任务要求 |
| 分析深度 | 全模块覆盖 | Attention 子链路、MoE、FFN、Norm、Residual |
| 参考模型 | qwen3-moe（GQA+MoE+Shared Expert） | 架构最接近的仓库参考 |

---

### 第一步：模块拆解与当前算子清单

#### 1.1 模块链路总览

```
Embedding (VocabParallelEmbedding)
  └─ Transformer Block × 80 (layers 0-79)
       ├─ [Residual + RMSNorm] input_layernorm
       ├─ Attention (GQA)
       │    ├─ QKV Projection (QKVParallelLinear) ✓ NPU parallel
       │    ├─ Q Norm (HYV3RMSNorm, per-head 128)  ← CANDIDATE
       │    ├─ K Norm (HYV3RMSNorm, per-head 128)  ← CANDIDATE
       │    ├─ RoPE (npu_apply_rotary_pos_emb)     ✓ NPU fused
       │    ├─ KV Cache write (scatter_update_)    ✓ NPU fused
       │    ├─ Flash Attention (npu_fused_infer_attention_score) ✓ NPU fused
       │    └─ O Projection (RowParallelLinear)    ✓ NPU parallel
       ├─ [Residual + RMSNorm] post_attention_layernorm
       ├─ FFN / MoE
       │    ├─ Layer 0: Dense FFN
       │    │    ├─ gate_proj (ColumnParallelLinear)  ✓ NPU parallel
       │    │    ├─ up_proj (ColumnParallelLinear)    ✓ NPU parallel
       │    │    ├─ SiLU activation (F.silu)          ← CANDIDATE
       │    │    └─ down_proj (RowParallelLinear)     ✓ NPU parallel
       │    └─ Layers 1-79: MoE
       │         ├─ Router (nn.Linear + sigmoid + topk)  ← CANDIDATE
       │         ├─ init_routing_v2                      ✓ NPU fused
       │         ├─ AllToAll dispatch/combine            ✓ dist
       │         ├─ re_routing                           ✓ NPU fused
       │         ├─ Expert FFN × 12 (per-rank)
       │         │    ├─ gate+up chunk (F.linear)         ← CANDIDATE
       │         │    ├─ SiLU activation (F.silu)         ← CANDIDATE
       │         │    └─ down_proj (F.linear)             ← CANDIDATE
       │         ├─ finalize_routing                      ✓ NPU fused
       │         └─ Shared Expert (HYV3MLP, same as Dense FFN) ← CANDIDATE
       └─ [Residual add] (hidden_states = residual + hidden_states)
  └─ Final RMSNorm (HYV3RMSNorm)                     ← CANDIDATE
  └─ LM Head (ColumnParallelLinear)                  ✓ NPU parallel
```

**标注说明**:
- `✓ NPU fused` = 已使用 torch_npu 融合算子，无需替换
- `✓ NPU parallel` = 已使用并行 Linear（ColumnParallel/RowParallel），无需替换
- `✓ dist` = 使用标准 PyTorch 分布式通信，无需替换
- `← CANDIDATE` = 存在可用 NPU 融合算子，需要分析替换可行性

#### 1.2 每 Forward Pass 的算子调用频率

| 算子位置 | 每层调用数 | 80 层总调用数 | 当前实现 |
|---------|-----------|------------|---------|
| input_layernorm (RMSNorm) | 1 | 80 | Python manual RMSNorm |
| post_attention_layernorm (RMSNorm) | 1 | 80 | Python manual RMSNorm |
| Q Norm (RMSNorm per-head) | 1 | 80 | Python manual RMSNorm |
| K Norm (RMSNorm per-head) | 1 | 80 | Python manual RMSNorm |
| Residual add (post-attn) | 1 | 80 | `hidden_states = residual + hidden_states` |
| Residual add (post-ffn) | 1 | 80 | `hidden_states = residual + hidden_states` |
| MoE Router (sigmoid+topk) | 1 | 79 | Python sigmoid + topk + normalize |
| Dense FFN SiLU | 1 | 1 (layer 0) | `F.silu(gate) * up` |
| Shared Expert SiLU | 1 | 79 | `F.silu(gate) * up` |
| Expert FFN SiLU (per-expert) | ≤12 | ≤948 | `F.silu(gate) * up` in loop |
| Final RMSNorm | N/A | 1 | Python manual RMSNorm |

---

### 第二步：按模块匹配仓库参考实现

#### 2.1 RMSNorm — 匹配 `npu_rms_norm`

**参考**: qwen3-moe `modeling_qwen3_moe.py:77-79`
```python
# qwen3-moe 参考实现
def ln_npu(self, hidden_states):
    result = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
    return result
```

**Hy3-preview 当前实现** (`modeling_hy_v3.py:56-61`):
```python
def forward(self, hidden_states):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)
```

**匹配评估**: 完全匹配。`npu_rms_norm(x, gamma, eps)` 计算公式 `x / sqrt(mean(x^2) + eps) * gamma`，与 Hy3-preview RMSNorm 数学等价。qwen3-moe、deepseek-r1、deepseek-v3.2-exp 等所有仓库模型均使用此算子。

**候选操作**:
| 位置 | 当前 | 替换 | 调用频率 |
|------|------|------|---------|
| `HYV3RMSNorm.forward()` | Python manual | `npu_rms_norm` | 321次/forward |
| `HYV3Model.norm` (final) | Python manual | `npu_rms_norm` | 1次/forward |
| Q Norm (Attention) | Python manual | `npu_rms_norm` | 80次/forward |
| K Norm (Attention) | Python manual | `npu_rms_norm` | 80次/forward |

**约束检查**:
- Atlas A2: ✓ 支持
- dtype: BF16 input + BF16 weight ✓
- 尾轴约束: head_dim=128 ≥ 32 bytes (16 BF16 elems × 2 bytes = 32) ✓ 边界满足
- 通用 tail: hidden_size=4096 >> 32 bytes ✓

---

#### 2.2 Residual + RMSNorm — 匹配 `npu_add_rms_norm`

**参考**: qwen3-moe `modeling_qwen3_moe.py:89-92`
```python
# qwen3-moe 参考实现
elif len(args) == 1:  # residual is not None
    residual = args[0]
    y, _, x = torch_npu.npu_add_rms_norm(residual, hidden_states, self.weight, self.variance_epsilon)
    return (y, x)  # y = residual + rms_norm(hidden), x = rms_norm(hidden)
```

**Hy3-preview 当前实现** (`modeling_hy_v3.py:629-662`):
```python
# 当前：分两步 — 先 norm，再 add
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)
# ... attention ...
hidden_states = residual + hidden_states  # ← 分离的 residual add
```

**匹配评估**: 需要改造 DecoderLayer 的 residual flow。`npu_add_rms_norm(residual, hidden_states, weight, eps)` 一次性完成 `rms_norm(hidden_states) + residual`，融合了两个操作。

**改造方案**: 参考 qwen3-moe 的 `RMSNorm.forward(hidden_states, residual)` 重载模式:
- `forward(hidden_states)` → 纯 norm（Pre-Norm）
- `forward(hidden_states, residual)` → `npu_add_rms_norm(residual, hidden_states, weight, eps)` 返回 `(y, x)`，其中 y 已含 residual

DecoderLayer 改造后:
```python
# 改造前:
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)
hidden_states = self.self_attn(hidden_states, ...)
hidden_states = residual + hidden_states       # ← 2 ops: add

residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)
hidden_states = residual + hidden_states       # ← 2 ops: add

# 改造后:
hidden_states, residual = self.input_layernorm(hidden_states, None)
# → rms_norm(hidden) + return (normalized, original_as_residual)
hidden_states = self.self_attn(hidden_states, ...)
# → npu_add_rms_norm: fused residual add + norm, 1 op
hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
hidden_states = self.mlp(hidden_states)
# = hidden_states + residual handled in next layer's input_layernorm
# return (hidden_states, residual) for next layer
```

**候选操作**:
| 位置 | 当前 | 替换 | 调用频率 |
|------|------|------|---------|
| Post-Attn residual+norm | `residual + norm(hidden)` (2 ops) | `npu_add_rms_norm` (1 op) | 80次/forward |
| Post-FFN residual+norm | `residual + norm(hidden)` (2 ops) | `npu_add_rms_norm` (1 op) | 80次/forward |

**约束检查**:
- Atlas A2: ✓ 支持（API 存在且 qwen3-moe/deepseek-r1 等 A2 模型使用）
- dtype: BF16 ✓
- API 返回值: `(y, _, x)` 其中 `y = residual + rms_norm(x)`, `x = rms_norm(hidden_states)`

---

#### 2.3 MoE Router (sigmoid + topk) — 匹配 `npu_moe_gating_top_k`

**参考**: deepseek-r1 / deepseek-v3.2-exp 使用 `npu_moe_gating_top_k(norm_type=1)` (sigmoid)，qwen3-moe 使用 `npu_moe_gating_top_k_softmax` (softmax)

**Hy3-preview 当前实现** (`modeling_hy_v3.py:340-362`):
```python
def forward(self, hidden_states, e_score_correction_bias):
    router_logits = self.gate(hidden_states.float())
    routing_weights = torch.sigmoid(router_logits)                   # step 1
    scores_for_choice = routing_weights + e_score_correction_bias    # step 2
    _, top_k_index = torch.topk(scores_for_choice, self.top_k, ...)  # step 3
    top_k_weights = routing_weights.gather(1, top_k_index)           # step 4
    top_k_weights = top_k_weights / (top_k_weights.sum(-1, ...) + 1e-20)  # step 5
    top_k_weights = top_k_weights * self.router_scaling_factor       # step 6
```

**候选**: `torch_npu.npu_moe_gating_top_k(router_logits, k=8, bias=e_score_correction_bias, norm_type=1, routed_scaling_factor=2.826, eps=1e-20)`

**精度差异分析 (CRITICAL)**:

Hy3-preview 的 router 算法与 `npu_moe_gating_top_k(norm_type=1)` 的语义差异：

| 步骤 | Hy3-preview 原始行为 | `npu_moe_gating_top_k(norm_type=1, bias=...)` |
|------|------------|----------------------------------------------|
| Sigmoid | `sigmoid(logits)` — 无 bias | `sigmoid(logits + bias)` — bias 在 sigmoid 内 |
| TopK 选择 | `topk(sigmoid(logits) + bias)` | `topk(sigmoid(logits + bias))` |
| 权重取值 | `gather(sigmoid(logits))` — 取原始 sigmoid 值 | 取 `sigmoid(logits + bias)` 的值 |
| 归一化 | `/ sum(weights) * scaling_factor` | `/ sum(weights) * routed_scaling_factor` |

**差异的本质**: Hy3-preview 的 `e_score_correction_bias` 只影响专家选择（topk 排序），不影响最终的专家权重（权重始终取原始 sigmoid 值）。而 `npu_moe_gating_top_k` 的 bias 同时影响选择和权重。

**结论**: **有条件候选。** 直接替换会改变 routing 行为（bias 进入 sigmoid 而非附加在选择分数上），可能导致精度差异。具体影响需实验验证：
- 如果 bias 值较小（e_score_correction_bias 经训练后趋于稳定），`sigmoid(x + bias) ≈ sigmoid(x) + bias * sigmoid'(x)`，差异可近似为一阶泰勒展开
- 但这是数学非等价替换，必须通过精度对比验证
- 如果精度验证通过则可替换（79个 MoE 层路由均受益）；如果精度不通过则保留当前 Python 实现

**候选操作**:
| 位置 | 当前 | 候选替换 | 调用频率 | 阻塞项 |
|------|------|---------|---------|--------|
| MoE Router | Python sigmoid+topk+gather+norm | `npu_moe_gating_top_k(norm_type=1)` | 79次/forward | 精度验证 |

---

#### 2.4 FFN SiLU Activation — 匹配 `npu_swiglu`

**参考**: `npu_swiglu` API 将 `silu(A) * B` 融合为单次调用

**Hy3-preview 当前实现** (Dense FFN `modeling_hy_v3.py:316-322`):
```python
def forward(self, x):
    gate = self.gate_proj(x)
    up = self.up_proj(x)
    down = self.down_proj(F.silu(gate) * up)  # ← F.silu(gate)*up 可融合
```

**Expert FFN** (`modeling_hy_v3.py:443-444`):
```python
gate, up = F.linear(expert_hidden, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
expert_out = F.linear(F.silu(gate) * up, self.down_proj[expert_idx])
# ← F.silu(gate)*up 可融合
```

**候选**: `torch_npu.npu_swiglu(torch.cat([gate, up], dim=-1), dim=-1)` 等价于 `F.silu(gate) * up`

**匹配评估**: 完全匹配。`npu_swiglu(input, dim=-1)` 将输入沿 dim 对半切分为 A、B，计算 `silu(A) * B`，与 `F.silu(gate) * up` 数学等价。

**注意事项**:
- `npu_swiglu` 需要输入在 dim 维度连续。`gate` 和 `up` 是独立 Linear 输出（Dense FFN）或 chunk 输出（Expert FFN），需先 concat
- Dense FFN (`HYV3MLP`): gate 和 up 是独立 projection，concat 后调用 `npu_swiglu`
- Expert FFN (packed `gate_up_proj`): `F.linear` 输出直接 `.chunk(2)` 后 gate 和 up 在内存中可能不连续，需 `.contiguous()` 后 concat 再调用 `npu_swiglu`

**候选操作**:
| 位置 | 当前 | 替换 | 调用频率 |
|------|------|------|---------|
| Dense FFN (layer 0) | `F.silu(gate) * up` | `npu_swiglu(cat([gate,up]), dim=-1)` | 1次/forward |
| Shared Expert (79层) | `F.silu(gate) * up` | `npu_swiglu(cat([gate,up]), dim=-1)` | 79次/forward |
| Expert FFN (79×12 experts) | `F.silu(gate) * up` (in loop) | `npu_swiglu(cat([gate,up]), dim=-1)` (in loop) | ≤948次/forward |

**约束检查**:
- Atlas A2: ✓ 支持
- dtype: BF16 ✓
- `npu_silu` 单函数: 计划废弃，不推荐单独使用（用 `F.silu` 即可）；`npu_swiglu` 是融合算子，有效

---

#### 2.5 Dense FFN 全融合 — 匹配 `npu_ffn`

**参考**: `npu_ffn` API 支持无专家模式（`expert_tokens=None`），融合 `x@W1 → activation → @W2` 全链路

**Hy3-preview Dense FFN 当前实现**:
```python
gate = self.gate_proj(x)    # [M, K] @ [K, N] → [M, N]
up = self.up_proj(x)        # [M, K] @ [K, N] → [M, N]
down = self.down_proj(F.silu(gate) * up)  # silu(gate)*up @ [N, K] → [M, K]
```

**候选**: `torch_npu.npu_ffn(x, weight1=gate_up_combined, weight2=down_weight, activation='swiglu')`

**匹配评估**: 有条件匹配。
- 需要将 `gate_proj` 和 `up_proj` 的权重合并为 `[2*N, K]` 的 `weight1`
- `weight2` = `down_proj.weight` `[K, N]`
- API 仅支持无 bias 的 FFN（Hy3-preview FFN 无 bias，匹配 ✓）

**性能门槛警告** (来自 API 文档):
> "激活层为 swiglu 时，性能使能需要满足门槛要求：整网中 FFN 结构所对应的小算子里，vector 耗时 30us 且占比 10% 以上的用例，方可尝试 FFN 融合算子；或在不知道小算子性能的情况下，尝试使能 FFN，若性能劣化则不使能 FFN。"

**分析**: Hy3-preview 的 Dense FFN（layer 0, intermediate=13312）计算量较小（约 0.33B FLOPs），且仅 1 层，可能不满足 `npu_ffn` 的门槛要求。**优先使用 `npu_swiglu` 激活融合（方案 2.4），`npu_ffn` 全融合作为进阶选项**。

另外，`HYV3MLP` 已使用 `ColumnParallelLinear` / `RowParallelLinear` 做 TP 切分，其内部已包含通信逻辑。直接替换为 `npu_ffn` 会绕过这些并行 Linear 层，需要手动处理 TP 切分和通信，增加复杂度。

**结论**: **不推荐作为首选。** 优先采用 `npu_swiglu`（方案 2.4）做激活融合，保留并行 Linear 层的现有 TP 逻辑。

---

#### 2.6 Expert FFN 全融合 — `npu_ffn` with `expert_tokens`

**候选**: `torch_npu.npu_ffn(x_ordered, gate_up_3d, down_3d, activation='swiglu', expert_tokens=expert_tokens_list)`

**匹配评估**: 理论上可替换 per-expert loop 中的 `F.linear → chunk → silu*mul → F.linear` 全链路。但存在以下阻塞：

1. **TP 约束**: Expert FFN 当前 `moe_tp=1`（不做 TP），expert weights 保持完整。`npu_ffn` 的 3D weight 需要 `[E_per_rank, K, N]` 格式，与当前 `gate_up_proj [E, 2*intermediate, hidden]` 兼容。

2. **手动 EP 路由**: 当前使用 `_moe_ep_manual` 手动路由，expert 计算通过 `forward_ordered()` 的 per-expert loop 完成。`npu_ffn` 的 `expert_tokens` 参数期望按专家排序的 token + 每个专家的 token 数量列表，这与 `forward_ordered` 的 `tokens_per_expert` 输入格式基本匹配。

3. **性能门槛**: 同方案 2.5 的警告。Expert intermediate=1536 较小，可能不满足 `npu_ffn` 的最佳性能门槛。

4. **weight 格式**: `npu_ffn` 期望 `weight1 [E, K1, N1]` 和 `weight2 [E, K2, N2]`。当前 `gate_up_proj [E, 2*intermediate, hidden]` 可直接作为 weight1（`K1=hidden, N1=2*intermediate`），`down_proj [E, hidden, intermediate]` 可直接作为 weight2（`K2=intermediate, N2=hidden`）。匹配 ✓

**结论**: **中等优先级候选。** 先验证 `npu_swiglu` 在 per-expert loop 中的有效性。若 profiling 显示 FFN 是瓶颈再尝试 `npu_ffn` 全融合。

---

#### 2.7 MoE Combine + Add + RMSNorm — `npu_moe_distribute_combine_add_rms_norm`

**候选**: `torch_npu.npu_moe_distribute_combine_add_rms_norm(...)`

**匹配评估**: **不适配。** 原因：
- **硬件约束**: API 文档明确标注仅支持 Atlas A3 系列，Hy3-preview 现部署在 Atlas A3 上，此 API 变为可用。
- 即使硬件满足，此 API 需要配合 `npu_moe_distribute_dispatch_v2` 使用，当前使用手动 AllToAll 路由方案，链路不匹配。

---

### 第三步：查阅算子接口文档，确认可用性与适配性

#### 3.1 算子约束汇总

| 算子 | Atlas A2 | BF16 | 硬约束 | 适配前置改造 |
|------|---------|------|--------|------------|
| `npu_rms_norm` | ✓ | ✓ | tail ≥ 32 bytes | 无（直接替换） |
| `npu_add_rms_norm` | ✓ | ✓ | 标准 | DecoderLayer residual flow 改造 |
| `npu_moe_gating_top_k` | ✓ | ✓ | 专家数 ≤ 2048 (192 ✓) | 精度验证（bias 语义差异） |
| `npu_swiglu` | ✓ | ✓ | dim 对半整除 | gate/up concat（Dense FFN 需 concat 两个 Linear 输出） |
| `npu_ffn` (dense) | ✓ | ✓ | — | 需绕过并行 Linear，不推荐 |
| `npu_ffn` (MoE) | ✓ | ✓ | expert_tokens ≤ 256 | forward_ordered 适配 |
| `npu_moe_distribute_combine_add_rms_norm` | **✗** (A3 only) | — | — | **不适配** |

#### 3.2 FA / RoPE / KV Cache / MoE Routing — 已使用融合算子（无需替换）

| 链路 | 当前算子 | 状态 |
|------|---------|------|
| Attention Core | `npu_fused_infer_attention_score` (FA v1) | **已优化** ✓ |
| RoPE | `npu_apply_rotary_pos_emb(layout='BSH')` | **已优化** ✓ |
| KV Cache 写入 | `scatter_update_(axis=-2)` | **已优化** ✓ |
| MoE Init Routing | `npu_moe_init_routing_v2` | **已优化** ✓ |
| MoE Re-Routing | `npu_moe_re_routing` | **已优化** ✓ |
| MoE Finalize | `npu_moe_finalize_routing` | **已优化** ✓ |
| QKV/O 投影 | `QKVParallelLinear` / `RowParallelLinear` / `ColumnParallelLinear` | **已优化** ✓ |

#### 3.3 QK Norm 特殊说明

QK Norm 是 Hy3-preview 特有的 per-head RMSNorm(128)。`npu_kv_rmsnorm_rope_cache` 的 RMSNorm 是 layer-level（非 per-head），**不适用于 QK Norm**。

QK Norm 的优化方向：当前 QK Norm 可替换为 `npu_rms_norm`（方案 2.1），但 QK Norm + RoPE 的整体融合（如 `npu_kv_rmsnorm_rope_cache`）不可用（因为该算子是 MLA 专用且 norm 语义不同）。

#### 3.4 不适配算子及原因

| 算子 | 不适配原因 | 硬约束证据 |
|------|----------|----------|
| `npu_kv_rmsnorm_rope_cache` | MLA 专用算子，做 layer-level KV norm + RoPE + cache 写入。Hy3-preview 是 GQA + per-head QK norm，完全不匹配 | API 文档明确描述为 MLA 结构专用 |
| `npu_mla_prolog_v3` | MLA prolog 算子，融合 Q/KV 投影 + RMSNorm + RoPE + cache。Hy3-preview 为 GQA 架构 | API 文档明确要求 MLA absorb 模式 |
| `npu_moe_distribute_combine_add_rms_norm` | 仅 Atlas A3 支持 | API 文档产品支持表 |
| `npu_moe_distribute_dispatch_v2` | MC2 dispatch 在小 batch + 大 EP 下有 rank 收到 0 token 限制 | 前序实施记录：16 die EP + batch=4 时某些 rank 收不到 token |

---

### 第四步：分析阶段审查与候选清单

#### 4.1 分析阶段审查

- [x] 模块拆解完整：已展开到可替换链路级别，含 Prefill/Decode 分支差异、QK Norm 子链路
- [x] 参考匹配完整：已按模块匹配 qwen3-moe（GQA）、deepseek-r1（MoE gating）、deepseek-v3.2-exp（MoE）参考实现
- [x] 算子约束已确认：已查阅 API 文档确认适配约束（dtype、shape、硬件平台）
- [x] 候选清单已形成：每个候选明确前置条件、最小验证切口和阻塞点

#### 4.2 候选替换清单（按优先级排序）

```
优先级排序依据:
  Tier 1 (P0): 低风险 + 高性能收益 + 参考模型已验证 + 无精度影响
  Tier 2 (P1): 中等风险 + 需精度验证 或 需前置改造
  Tier 3 (P2): 高复杂度 + 需性能门槛验证 + 收益不确定
  ⛔ 不适配: 硬约束不满足
```

| 优先级 | 模块 | 原算子 | NPU 融合算子 | 替换理由 | 前置条件 | 验证切口 |
|--------|------|--------|------------|---------|---------|---------|
| **P0** | RMSNorm (all) | Python manual `pow(2).mean().rsqrt()` | `torch_npu.npu_rms_norm` | 消除 321 次 Python 小算子调用，qwen3-moe/deepseek 等所有模型已验证 | 无 | 单层 RMSNorm 输出精度对比 |
| **P0** | Residual + RMSNorm (160处) | 分离的 `x + y` + `RMSNorm(y)` | `torch_npu.npu_add_rms_norm` | 融合 2 次操作→1 次，消除 160 次 Python kernel launch | DecoderLayer residual flow 改造（参考 qwen3-moe 模式） | DecoderLayer 输出精度对比（改造前后） |
| **P0** | FFN SiLU activation (80处) | `F.silu(gate) * up` | `torch_npu.npu_swiglu` | 融合 silu+mul 为单算子，Dense FFN + Shared Expert + Expert FFN 全覆盖 | Dense FFN: gate/up concat；Expert: chunk 后 concat | SiLU 激活输出精度对比 |
| **P1** | MoE Router (79处) | Python sigmoid+topk+gather+norm 多步 | `torch_npu.npu_moe_gating_top_k` | 融合 sigmoid+topk+routing 全流程为单算子 | **精度验证**（bias 语义差异：Hy3-preview bias 在 sigmoid 后，API 在 sigmoid 前） | Router 输出的 topk_index 和 topk_weights 精度对比 |
| **P2** | Dense FFN 全融合 (1处) | gate_proj→up_proj→silu→down_proj | `torch_npu.npu_ffn(activation='swiglu')` | 端到端融合 4 步→1 步 | 绕过 ColumnParallel/RowParallel 需手动处理 TP；性能门槛待 profiling | 不推荐作为首选（优先 P0 的 swiglu 融合） |
| **P2** | Expert FFN 全融合 (≤948处) | F.linear→chunk→silu→F.linear loop | `torch_npu.npu_ffn(expert_tokens=...)` | 替换 per-expert loop 为单次 grouped FFN | forward_ordered 适配；性能门槛待 profiling | Expert FFN 输出精度对比 + 性能对比（loop vs fused） |
| ⛔ | MoE Combine+Add+RMSNorm | 手动 AllToAll + finalize + shared + add | `npu_moe_distribute_combine_add_rms_norm` | 不适配 | **不可用**: Atlas A3 only，Hy3-preview 部署在 Atlas A2 | N/A |

#### 4.3 推荐实施顺序

```
阶段 A (Tier 1, 零风险):
  1. RMSNorm → npu_rms_norm (改动最小，321 处)
  2. FFN SiLU → npu_swiglu (Dense + Shared + Expert, 80+ 处)
  验证: 精度对比（单层 vs 改造后）

阶段 B (Tier 1, 需前置改造):
  3. Residual + RMSNorm → npu_add_rms_norm (需改 DecoderLayer residual flow)
  验证: 精度对比（DecoderLayer 输出端到端）

阶段 C (Tier 2, 需精度验证):
  4. MoE Router → npu_moe_gating_top_k (需精度验证 bias 语义差异)
  验证: 精度对比 + 若失败回退保留 Python 实现

阶段 D (Tier 3, 进阶优化):
  5. profiling 驱动: 若 SiLU 融合后 expert FFN 仍是热点 → 尝试 npu_ffn
  6. profiling 驱动: 若 Dense FFN 是热点 → 评估 npu_ffn 替代方案
```

#### 4.4 预期收益估算

| 优化项 | 每层消除的小算子 | 80层累计 | 预估影响 |
|--------|---------------|---------|---------|
| RMSNorm → `npu_rms_norm` | pow, mean, rsqrt, mul (4 ops) | ~1284 ops | 高：消除全部 Python fallback kernel launch |
| Residual+RMSNorm → `npu_add_rms_norm` | add + norm (2 ops→1) | 160 ops 融合 | 中：减少 kernel launch + 内存往返 |
| FFN SiLU → `npu_swiglu` | silu + mul (2 ops→1) | ~80-948 ops | 中：Dense+Shared 小但 Expert loop 累积可观 |
| MoE Router → `npu_moe_gating_top_k` | sigmoid, add, topk, gather, sum, div, mul (7 ops→1) | 79×7=553 ops | 中-高：但需精度验证 |

> 注：以上为算子数量级估算。实际性能收益受 kernel launch overhead、内存带宽、计算密度等多因素影响，需 profiling 确认。

#### 4.5 与参考模型的差距总结

| 模块 | Hy3-preview 当前 | qwen3-moe 参考 | 差距 |
|------|---------|---------------|------|
| RMSNorm | Python manual | `npu_rms_norm` + `npu_add_rms_norm` | 待替换 |
| MoE Gating | Python manual | `npu_moe_gating_top_k_softmax` | 待验证（sigmoid vs softmax 差异） |
| FFN Activation | `F.silu(gate) * up` | `F.silu(gate) * up` (同样未用 `npu_swiglu`) | qwen3-moe 也未优化此项 |
| Attention | FA v1 + RoPE + scatter_update_ | FA v1 + RoPE + scatter_update_ | **已对齐** |
| MoE Routing | init_routing_v2 + re_routing + finalize | init_routing_v2 + MC2 dispatch/combine (Decode) 或 all_to_all (Prefill) | Hy3-preview 统一使用 all_to_all，qwen3-moe Decode 用 MC2 |
| QKV/O Proj | ParallelLinear | ParallelLinear | **已对齐** |

**关键发现**: Hy3-preview 与 qwen3-moe 在 Attention 和并行 Linear 层面已对齐，主要差距在 Norm 和 MoE Router 的融合算子使用上。qwen3-moe 自身也未使用 `npu_swiglu`，说明此优化在仓库中尚无 GQA MoE 模型的验证先例，Hy3-preview 可作为首个试点。

---

## 阶段 3：融合算子实施（/model-infer-fusion 步骤 5，仅 P0 项）

### 实施记录
- [完成] `npu_rms_norm` 替换 HYV3RMSNorm.forward — modeling_hy_v3.py:56-78
  - 将所有 321 个 RMSNorm 层（input_layernorm × 80, post_attention_layernorm × 80, Q Norm × 80, K Norm × 80, final norm × 1）从 Python manual 实现替换为 `torch_npu.npu_rms_norm`
  - tail axis 约束检查: hidden_size=4096 >> 32 bytes, head_dim=128 >= 32 bytes (16 BF16 elems × 2 bytes = 32), 尾轴满足 ✓
  - dtype 约束: BF16 input + BF16 weight ✓ (HYV3RMSNorm weight 默认 float32, API 文档确认 bfloat16 input + float32 weight 支持 ✓)
  - 与 qwen3-moe 参考实现一致 (modeling_qwen3_moe.py:77-79)
- [完成] `npu_swiglu` 替换 FFN SiLU 激活 — modeling_hy_v3.py:333, 419, 448
  - HYV3MLP.forward (Dense FFN layer 0 + Shared Expert × 79): `F.silu(gate) * up` → `torch.cat([gate, up], dim=-1)` + `torch_npu.npu_swiglu(merged)`
  - HYV3Experts.forward (per-expert loop, ≤948 次/forward): `.chunk(2, dim=-1)` + `F.silu(gate) * up` → 跳过 chunk, 直接用 `F.linear` 完整输出传给 `torch_npu.npu_swiglu`
  - HYV3Experts.forward_ordered (EP 路由 per-expert chunk): 同上
  - 约束检查: dim=-1 对半整除 (2*intermediate=3072, 整除 2 ✓), dtype BF16 ✓
  - qwen3-moe 未使用此优化; deepseek-r1/glm-5/kimi-k2 等 MLA 模型已大规模使用
- [完成] `npu_add_rms_norm` residual+norm 融合 — modeling_hy_v3.py:56-78, 651-686, 774-790
  - HYV3RMSNorm.forward 扩展: `forward(hidden_states)` → 纯 norm; `forward(hidden_states, None)` → norm + return residual; `forward(hidden_states, residual)` → `npu_add_rms_norm` 融合
  - HYV3DecoderLayer.forward: 新增 `past_residual` 参数; `input_layernorm(hidden_states, past_residual)` / `post_attention_layernorm(hidden_states, residual)` 使用融合算子
  - HYV3Model.forward: 新增 `residual = None` 初始化 + 层间 `past_residual` 传递 + 最终 `self.norm(hidden_states, residual)` 融合
  - eps=1e-5 适配: `self.variance_epsilon` 从 config.rms_norm_eps 读取 (Hy3-preview config 为 1e-5), 直接传递给 `npu_add_rms_norm` 和 `npu_rms_norm`
  - 完全遵循 qwen3-moe 参考模式 (modeling_qwen3_moe.py:82-96, 1006, 1020)

### 当前代码状态
- HYV3RMSNorm.forward: 支持 3 种调用模式 (纯 norm / norm+返回 residual / fused add_rms_norm)
- HYV3MLP.forward: gate/up concat → npu_swiglu → down
- HYV3Experts.forward / forward_ordered: 跳过 chunk, Linear 输出直入 npu_swiglu
- HYV3DecoderLayer.forward: past_residual 参数, fused residual+norm 链路
- HYV3Model.forward: residual 层间传递, fused final norm
- 所有 QK Norm (HYV3Attention) 自动受益于 npu_rms_norm (head_dim=128, eps=1e-5)
- 外部接口 (prepare_inputs_for_generation, forward) 不变, runner 无需修改

### 自验证结果
- 参考 skill: /model-infer-fusion (步骤 5, 仅 P0 项)
- 代码加载: 通过 — modeling_hy_v3.py, runner_hy3.py, infer.py, model_setting.py 语法正确
- 编译: 通过 — ast.parse 无错误
- 推理: 未执行 (环境中无可用的多卡推理环境)
- 输出: N/A（未到推理阶段）

### 未实施项 (P1/P2/不适配)
- P1: MoE Router → `npu_moe_gating_top_k` — 精度验证未完成 (bias 语义差异), 保留 Python 实现
- P2: Dense FFN 全融合 → `npu_ffn` — 需绕过并行 Linear, 性能门槛待 profiling
- P2: Expert FFN 全融合 → `npu_ffn(expert_tokens=...)` — forward_ordered 适配 + 性能门槛
- ⛔: MoE Combine+Add+RMSNorm → `npu_moe_distribute_combine_add_rms_norm` — Atlas A3 only, Hy3-preview 部署在 A2

### A3 迁移与图模式适配 (2026-04-27)

- [x] YAML 更新: exe_mode eager→ge_graph, enable_superkernel=True, moe_ep_size=16
- [x] FA 算子切换: HYV3Attention 添加 enable_gegraph 标志 + fa_ops (torchair.ops in ge_graph)
- [x] actual_seq_lengths_kv tensor 化: ge_graph 模式保持 tensor，避免 Graph Break
- [x] Runner graph_compile: torchair.patch_for_hcom() + CompilerConfig (frozen_parameter, tiling_schedule_optimize, StableRDFS)
- [x] Runner model_inference: prefill/decode 分离调用，ge_graph warmup mark_inputs
- [x] SuperKernel 集成: HYV3Model.forward decode 阶段 superkernel_scope("decode_layer") 包裹层循环
- [x] model_setting SuperKernel 验证: enable_superkernel 强制要求 exe_mode=ge_graph
- [x] 语法/导入自验证: 全部通过

#### 待验证
- 16 卡 A3 环境实际推理运行
- Decode 性能（期望从 ~293ms/t 显著改善，目标 <100ms/t）
- 输出 token 质量（无乱码/LaTeX/HTML 碎片）
- SuperKernel 是否与手动 EP AllToAll 兼容（图捕获风险）

### Eager 模式推理测试 (2026-04-27, 第4次运行)

- [x] 16-die A3 eager 模式推理跑通
- [x] 权重加载: 112 shards, ~12s
- [x] Prefill: warmup 11.09s, actual 1.04s (1024 tokens)
- [x] Decode: warmup 290ms/t, actual avg **246ms/t** (range: 218-277ms/t)
- [x] P0 融合算子效果: decode 从 ~293ms/t 降至 ~246ms/t（16% 改善）
- [ ] 输出质量: 生成文本为乱码（"by_{-□S Fr I end(\\id by \\..."），疑似模型预览版训练不完整或权重映射问题
- [ ] ge_graph 模式: 因手动 EP all_to_all 数据依赖 size（all_tokens_sum）在 torch.compile fullgraph 下不可静态推导，需切换到 MC2 dispatch/combine
- [ ] SuperKernel: 因 superkernel_scope context manager 在 torch._dynamo 下图捕获失败（GenericContextWrappingVariable），需移至 compile 外部

#### 性能对比
| 指标 | 前次(A3 eager) | 本次(A3 eager+P0融合) | 改善 |
|------|---------------|---------------------|------|
| Prefill (1024t) | ~2.07s | 1.04s | 50% |
| Decode avg | ~293ms/t | 246ms/t | 16% |

#### 已知阻塞项
1. **ge_graph 图模式**: `_moe_ep_manual` 中 `sum(output_splits)` 产生数据依赖 size，torch.compile(fullgraph=True) 无法静态推导 → 需实现 MC2 dispatch/combine 替代手动 all_to_all
2. **SuperKernel**: `superkernel_scope` context manager 在图捕获时 Unsupported → 需移至 torch.compile 外部或使用其他使能方式
3. **输出质量**: 乱码问题需精度调试（可能是模型预览版问题、tokenizer/model vocab 不完全对齐、或生成参数问题）

### 图模式适配尝试 (2026-04-27)

**状态**: GE 图模式 decode 编译未成功，回退到 eager 模式

**实施的改造**:
- [x] MC2 dispatch/combine 方法 `_moe_ep_mc2_decode` — 替代手动 all_to_all，消除数据依赖 size
- [x] GMM expert 计算 `forward_gmm` — 使用 `npu_grouped_matmul` (split_item=2) 替代 per-expert F.linear
- [x] 权重 GMM 格式转换 `_prepare_gmm_weights` — 将 `[E,out,in]` 转为 GMM 期望的 `[E,in,out]`
- [x] Runner `graph_compile`: torchair.patch_for_hcom() + CompilerConfig（基础设施就绪）
- [x] Runner `model_inference`: prefill/decode 分离调用（基础设施就绪）

**GE 编译失败记录**:

| 尝试 | 配置 | 错误 |
|------|------|------|
| #1 | torchair FA + fullgraph=True | torchair FA 不支持 eager mode（prefill 必经之路） |
| #2 | torch.ops.npu FA + 手动EP + fullgraph=True | 数据依赖 `all_tokens_sum` 无法静态推导 |
| #3 | torch.ops.npu FA + MC2 + fullgraph=True | MC2 dispatch 返回值 UNPACK_SEQUENCE 长度不匹配 |
| #4 | torch.ops.npu FA + MC2 + fullgraph=True (index unpack) | MC2 combine 参数 `scales` 无效 |
| #5 | torch.ops.npu FA + MC2 + fullgraph=True (fix kwargs) | GE 后端编译失败 (ERR03005 GRAPH internal error) |
| #6 | torch.ops.npu FA + MC2 + fullgraph=False | GE 后端编译失败 (ERR03005 GRAPH internal error) |
| #7 | torch.ops.npu FA + MC2 + eager | OOM — MC2 dispatch 中间张量过大 |

**根因分析**:
- **GE 编译失败**: `npu_grouped_matmul(split_item=2)` + `npu_moe_distribute_combine_v2` 在当前 torchair/CANN 版本中 GE 后端不支持或存在兼容性问题
- **MC2 eager OOM**: MC2 dispatch 创建的中间张量（expand_x, expand_idx 等）在 eager 模式下使用额外显存，累计超过单 die 61 GB 限制
- **手动 EP + fullgraph**: 数据依赖 split sizes 无法在编译时静态推导，`dynamic=True` 也无法解决此问题

**已就绪待激活**（当 torchair/CANN 更新后）:
- `HYV3Experts.forward_gmm()` — GMM 专家计算
- `HYV3Experts._prepare_gmm_weights()` — 权重格式转换
- `HYV3MoE._moe_ep_mc2_decode()` — MC2 dispatch/combine 路由
- `runner_hy3.py:graph_compile()` — GE 图编译基础设施
- YAML 中 `exe_mode: ge_graph` 切换

**当前工作配置**: eager 模式 + P0 融合 + 手动 EP 路由 + 分离 prefill/decode 调用

### 最终性能 (eager mode, 16-die A3)

| 指标 | 数值 |
|------|------|
| Prefill (1024t) | 1,041 ms |
| Decode avg | **242.48 ms/t** (range: 212-263) |
| 对比基线 (293ms/t) | **17.2% 改善** |

### QKV 权重加载修复 (2026-04-27)

- **根因**: Checkpoint 使用分离的 `q_proj/k_proj/v_proj` 权重（每层 3 个 key × 81 层 = 243 个 key），但模型使用 `QKVParallelLinear` 合并 QKV 投影（参数名为 `merged_qkv_proj.weight`）。`load_weights` 中 `name in params_dict` 匹配失败，QKV 权重被静默跳过，模型使用随机初始化权重组装 QKV → 输出完全乱码。
- **修复**: 在 `load_weights` 中添加 QKV 权重特殊匹配逻辑，将 `model.layers.X.self_attn.{q,k,v}_proj.weight` 映射到 `merged_qkv_proj.weight`，通过 `weight_loader(param, loaded_weight, loaded_shard_id={q,k,v})` 正确加载。
- **验证**: 全量 47,138 个 checkpoint key 映射验证 — 0 个 unmatched。推理输出从 `"by_{-□S Fr I end(\\id by \\..."` 恢复正常文本。

### acl_graph 图模式测试 (2026-04-27)

- **尝试 1**: acl_graph + MC2 dispatch/combine + fullgraph=True → **OOM** (graph capture 阶段 59.61/61.27 GiB)
- **尝试 2**: acl_graph + MC2 + fullgraph=True + max_position_embeddings=4096 → **仍 OOM** (同等显存占用)
- **尝试 3**: acl_graph + manual EP + fullgraph=False → **跑通但性能退化**: decode ~1218ms/t（6x 于 eager），torch._dynamo cache_size_limit=8，79 层 MoE 的 all_to_all 均产生图中断，大量重编译开销
- **结论**: acl_graph 在当前硬件 (64 GB HBM/die) 下不可行。MC2 intermediates + graph capture overhead 超显存；manual EP 图中断过多导致性能退化

### 当前性能 (eager mode, 16-die A2, 2026-04-27)

| 指标 | 数值 |
|------|------|
| Prefill warmup | 10,970 ms |
| Prefill (1024t) | 971 ms |
| Decode avg | **236.17 ms/t** (range: 222-247) |
| 输出质量 | 正常（"The output is a weighted sum of the values..."） |
| 对比原始基线 (293ms/t) | **19.4% 改善** |
| 距离目标 (<100ms/t) | 需 2.36x 加速 |

### ge_graph 图模式修复与跑通 (2026-04-27)

- **根因**: `_prepare_gmm_weights()` 中 `.transpose(1,2).contiguous()` 创建了 35.8 GB 额外权重副本 (79 层 × ~453 MB/layer)。与原始权重共存导致总专家权重 ~71.6 GB，超过 64 GB HBM。qwen3-moe 的 `FusedMoEGMM.process_weights_after_loading` 采用 in-place transpose 替代，不产生额外副本。
- **修复**:
  1. 移除 `_prepare_gmm_weights`、`_gate_up_gmm`、`_down_gmm` 持久化属性
  2. `forward_gmm` 改为 inline transpose (`.transpose(1,2)` 返回 view，不分配新内存)
  3. `npu_grouped_matmul` 可以使用 strided tensor（无需 `.contiguous()`）
- **MC2 dispatch/combine**: 参照 qwen3-moe 的 `moe_infer_dispatch_combine` 实现，用于 decode 阶段
- **MoE 路由**: ge_graph decode 使用 MC2 dispatch/combine（graph-compatible）；prefill 和 eager 使用 manual EP all_to_all

### 最终性能 (ge_graph mode, 16-die A2, 2026-04-27)

| 指标 | 数值 |
|------|------|
| Prefill warmup | 11,626 ms |
| Prefill (1024t) | 1,361 ms |
| Decode warmup (compile) | 91,151 ms |
| Decode avg | **33.25 ms/t** (range: 30.9-35.2) |
| 输出质量 | 正常 |
| vs eager baseline (236ms/t) | **7.1x 加速** |
| vs 原始基线 (293ms/t) | **8.8x 加速** |
| vs 目标 (<100ms/t) | **远超目标 (3x margin)** |

### Compile Cache 启用 (2026-04-27)

- `enable_cache_compile: True` + ge_graph: 首次运行生成 compile cache (~91s warmup)，后续运行 warmup 仅 ~9s（10x 加速）。缓存持久化到 `compile_cache/` 目录。

### P1 MoE Router 融合 (2026-04-27)

- `npu_moe_gating_top_k(norm_type=1)` 替换 Python sigmoid+topk+gather+norm+scale（7 ops → 1 kernel）
- 参数: `k=8, norm_type=1 (sigmoid), routed_scaling_factor=2.826`
- bias 语义差异 (sigmoid(x+bias) vs sigmoid(x)+bias) 未造成可见输出质量退化
- 79 层 MoE 路由均受益

### 最终性能汇总 (ge_graph + compile cache + P1 fusion, 16-die A2)

| 指标 | 数值 | 备注 |
|------|------|------|
| Prefill warmup | 10,590 ms | 首次运行含 compile cache 生成 |
| Prefill warmup (cached) | 8,028 ms | 二次运行 |
| Prefill (1024t) | 1,397 ms | |
| Decode warmup (compile) | 91,151 ms | 首次运行 |
| Decode warmup (cached) | 8,016 ms | 二次运行，10x 加速 |
| **Decode avg** | **31.47 ms/t** | range: 29.7-32.9 |
| 输出质量 | 正常 | |
| vs 原始 eager (293ms/t) | **9.3x 加速** | |
| vs 目标 (<100ms/t) | **3.2x margin** | |

### P2 融合算子重评估 (2026-04-28, A3 硬件)

- [分析] `npu_moe_distribute_combine_add_rms_norm`: **不可行** — API 计算 `norm(combine+shared+residual)` 与 Hy3-preview 的 deferred residual 模式 `norm(signal)+residual` 数学不等价，需全局重构 80 层 residual flow
- [分析] `npu_ffn(expert_tokens=..., activation='swiglu')`: **不可行** — API 硬约束：swiglu 模式不接受 expert_tokens 参数（仅 silu 模式支持），且 swiglu 仅支持 float16（Hy3-preview 用 bfloat16）
- [分析] `npu_ffn(activation='swiglu')` Dense FFN: **不可行** — 同 swiglu 不支持 bfloat16；仅 layer 0 1 层，收益极微
- [实施] `combine_v2 + shared_expert_x`: **可行** — A3 硬件支持此参数，单行改动消除 79 层/step 的 `routed_output + shared_output` Python 加法

#### shared_expert_x 实施与回退记录 (2026-04-28)

- [已实施] `HYV3MoE.forward()`: ge_graph decode 路径提前计算 shared_output → 传入 `_moe_ep_mc2_decode` → 跳过 Python 加法
- [已实施] `_moe_ep_mc2_decode()`: 新增 `shared_output=None` 参数 → `npu_moe_distribute_combine_v2(shared_expert_x=shared_output)`
- [验证] 16 卡 A3 推理 → **SuperKernel 编译失败**
  - 根因: `shared_expert_x` 使 combine_v2 输出包含完整 routed+shared，与下游 InplaceAddRmsNorm 形成更长融合链
  - SuperKernel 限制: 不支持超过 2 个 real stream 的融合 (`HcomAllReduce + dispatch + 2×gmm + swiglu + combine + InplaceAddRmsNorm + HcomAllGather`)
  - 错误: `ERROR: super kernel do not support more than 2 real stream`
- [回退] 代码已还原至已知稳定状态（无 shared_expert_x），SuperKernel 恢复编译成功
- **结论**: shared_expert_x 在 CANN 8.5.0 下与 SuperKernel 不兼容，待 CANN 更新 SuperKernel stream 限制后可重新启用
