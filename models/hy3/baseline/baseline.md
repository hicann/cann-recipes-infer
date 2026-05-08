# hy3-preview Baseline

> 采集时间：2026-04-27 ~ 2026-04-28
> 基线目的：记录 hy3-preview (Hunyuan3) 模型在 NPU 上的初始部署性能，以及各优化阶段的性能演进

---

## 1. 硬件环境

| 项目 | 规格 |
|------|------|
| NPU 型号 | Ascend 910C (Atlas A3) |
| 部署卡数 | 16 die (8 物理卡 × 2 die/卡) |
| 单 die HBM | 64 GB (65536 MB) |
| 总 HBM | 1024 GB |
| 实际可用/die | ~61.27 GiB |
| CANN 版本 | 8.5.0 (25.5.1) |
| PyTorch | 2.8.0 |
| torch_npu | 2.8.0 |
| 推理框架 | cann-recipes-infer (ModelRunner) |

---

## 2. 模型信息

| 参数 | 值 |
|------|-----|
| 模型名称 | Hunyuan3 (Hy3 preview) |
| 开发者 | Tencent Hy Team |
| 架构类型 | MoE Transformer (Dense + Sparse MoE + Shared Expert) |
| 总参数量 | 295B (295,033,543,488) |
| 激活参数量 | ~21B / token |
| BF16 参数占用 | ~590 GB |
| 总层数 | 80 (layer 0 Dense FFN + layers 1-79 MoE) |
| MTP 层 | layer 80 (~3.8B params, 推理跳过) |
| Hidden Size | 4096 |
| Attention | GQA: 64 Q heads, 8 KV heads, head_dim=128 |
| Dense FFN Intermediate | 13312 (layer 0) |
| MoE Expert Intermediate | 1536 |
| 专家总数 | 192 |
| 每 token 激活专家 | 8 (top-8 sigmoid routing) |
| Shared Expert | 1 per MoE layer |
| 词表大小 | 120,832 |
| 最大上下文长度 | 262,144 (256K) |
| RoPE Theta | 11,158,840 |
| 激活函数 | SiLU |
| 归一化 | RMSNorm, eps=1e-5 |
| 特殊设计 | QK Norm (per-head RMSNorm before RoPE), sigmoid router + expert bias |

---

## 3. 部署配置

### 3.1 并行策略

| 参数 | 值 | 说明 |
|------|-----|------|
| world_size | 16 | 8 卡 × 2 die/卡 |
| attn_tp_size | 4 | N_kv=8 约束, 均衡 Prefill/Decode |
| dense_tp_size | 4 | 与 attn_tp 对齐 |
| moe_tp_size | 1 | Expert intermediate=1536 太小 |
| moe_ep_size | 16 | 全卡 EP, 192/16 = 12 experts/rank |
| embed_tp_size | 4 | 120K 大词表切分 |
| lmhead_tp_size | 4 | 与 embed 对齐 |

### 3.2 执行配置

| 参数 | 值 |
|------|-----|
| 执行模式 | ge_graph (图模式, Decode 阶段) |
| 量化模式 | BF16 |
| enable_online_split_weight | True |
| enable_superkernel | True |
| enable_cache_compile | True |
| batch_size | 16 |

---

## 4. 性能基线

### 4.1 原始基线 (阶段 1 并行化后, eager mode)

| 指标 | 数值 | 测试条件 |
|------|------|---------|
| Prefill 耗时 | **2,070 ms** | 1024 tokens, batch=4, 16 die A3, eager |
| Decode 单步耗时 | **293 ms/t** | eager mode, 16 die A3 |
| 显存占用 | ~54.5 GB/die | BF16, 295B EP+TP 分布后 |

> 采集时间：2026-04-27。这是完成并行化部署后的第一个可运行版本，作为所有后续优化的对比基准。

### 4.2 各阶段性能演进

| 阶段 | 配置 | Prefill (1024t) | Decode (ms/t) | 累计加速 |
|------|------|----------------|---------------|---------|
| 阶段 1 | eager (并行化基线) | 2,070 ms | 293 | 1.0x |
| 阶段 3 | eager + P0+P1 融合算子 | 1,041 ms | 246 | 1.19x |
| 阶段 4 | ge_graph + MC2 dispatch/combine + GMM | — | 33.3 | 8.8x |
| 阶段 4 | + compile cache | 1,397 ms | 31.5 | 9.3x |
| 阶段 4 | + SuperKernel | 1,408 ms | 30.8 | 9.5x |
| **最终** | **ge_graph + compile cache + SuperKernel** | **1,452 ms** | **28.44** | **10.3x** |

> 最终性能采集时间：2026-04-28，32 tokens decode 实测 (range: 27.4-29.4 ms/t)。

### 4.3 最终性能 vs 目标

| 指标 | 最终值 | 目标 | 达成 |
|------|--------|------|------|
| Decode 单步耗时 | **28.44 ms/t** | <100 ms/t | **3.5x margin** |
| Prefill 耗时 (1024t) | 1,452 ms | — | — |
| 输出质量 | 正常 | — | ✓ |

---

## 5. 关键优化项

### 5.1 融合算子 (阶段 3)

| 算子 | 替换前 | 替换后 | 调用频率 |
|------|--------|--------|---------|
| RMSNorm (全部) | Python `pow(2).mean().rsqrt()` | `npu_rms_norm` | 321次/forward |
| Residual + RMSNorm | `x + y` + `norm(y)` 分离 | `npu_add_rms_norm` | 160次/forward |
| FFN SiLU 激活 | `F.silu(gate) * up` | `npu_swiglu` | 80+次/forward |
| MoE Router | Python sigmoid+topk+gather+norm (7 ops) | `npu_moe_gating_top_k(norm_type=1)` | 79次/forward |

### 5.2 图模式 (阶段 4)

| 项目 | 说明 |
|------|------|
| 图模式后端 | ge_graph (torchair + CompilerConfig) |
| Decode 路由 | MC2 dispatch/combine 替代手动 all_to_all |
| 专家计算 | `npu_grouped_matmul(split_item=2)` GMM |
| FA 接口 | torchair FA (ge_graph Decode) / torch.ops.npu FA (eager Prefill) |
| SuperKernel | `enable_superkernel=True`, decode 层循环外包 |
| Compile Cache | `enable_cache_compile=True`, 持久化 compile_cache/, 二次 warmup ~8s |

---

## 6. 关键修复记录

| 序号 | 阶段 | 问题 | 影响 | 修复 |
|------|------|------|------|------|
| F-1 | 1 | QKV 权重静默跳过 | 输出完全乱码 | q/k/v_proj → merged_qkv_proj 映射 |
| F-2 | 1 | `dist.broadcast(src=0)` ranks 4-15 崩溃 | Decode 完全阻塞 | 移除 broadcast |
| F-3 | 4 | GMM `_prepare_gmm_weights` 权重副本 OOM | ge_graph 编译失败 | inline transpose view |
| F-4 | 4 | 手动 EP all_to_all 数据依赖 graph break | fullgraph 编译失败 | MC2 dispatch/combine 替代 |
| F-5 | 4 | SuperKernel superkernel_scope 图捕获失败 | SuperKernel 未生效 | 移至 torch.compile 外部 |

---

## 7. 已知限制

| 限制项 | 说明 |
|--------|------|
| multi-stream | shared expert 计算量太小 (intermediate=1536), 无法有效 overlap, enable_multi_streams=False |
| P2 融合算子 | `npu_ffn` swiglu 模式不接受 expert_tokens + 仅支持 fp16, 不可行 |
| shared_expert_x | 与 SuperKernel 冲突 (CANN 8.5.0 stream 限制), 已回退 |
| 长序列 (>128K) | KV Cache 增长到 ~10.7 GB/卡, 256K 时 ~21.4 GB/卡 可能超限 |
| 量化 | 若减至 8 卡部署需 W8A8 量化, 当前 BF16 16 卡可行 |

---

## 8. 显存分布 (ge_graph 最终配置)

| 模块 | 参数量 | BF16 显存 | 占比 |
|------|--------|----------|------|
| Embedding | 495M | 0.99 GB | 0.17% |
| Attention (80层) | 6.04B | 12.08 GB | 2.05% |
| Dense FFN (layer 0) | 164M | 0.33 GB | 0.06% |
| MoE Experts (79层×192专家) | 286.29B | 572.57 GB | 97.04% |
| MoE 非专家 (Router+Shared+Bias) | 1.55B | 3.11 GB | 0.53% |
| LM Head | 495M | 0.99 GB | 0.17% |
| **总计** | **295.03B** | **590.07 GB** | **100%** |

> attn_tp=4 + moe_ep=16 分布后，每卡参数约 39.8 GB，KV Cache (32K) ~2.7 GB，总约 47.5 GB/die。
