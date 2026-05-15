# LongCat-Flash-Lite EP 路径改造进度归档

## 阶段 1 - 并行策略分析（首版，2026-03-28，已被新方案覆盖）

> 归档原因: 该版分析仅推导出 D1（纯 TP=8）和 D3（attn_tp=8 + EP=8）两个候选；后续按客户方案进一步细化为 dense_tp=1 + ngram_embed_tp=8 的差异化切分。

### 参数与模块链路

| 模块 | 总参数量 | BF16 显存 | 说明 |
|------|---------|----------|------|
| embed_tokens | 0.40 B | 0.8 GB | 主词表 |
| Ngram Embedding | ~31 B | 62 GB | 12 个子表 × ~10M vocab × 256，必须 TP 切 |
| MLA × 28 | 0.99 B | 0.13 GB | q_lora=1536, kv_lora=512 |
| Dense MLP × 28 | 0.5 B | 1.0 GB | gate/up/down |
| MoE × 14 | 33.8 B | 62 GB | EP8 时单卡 32 expert |
| LM Head | 0.4 B | 0.8 GB | |

### 候选方案

- **D1（纯 TP=8）**: 全部张量并行，包括 MoE。问题: MoE 切 1024/8=128 维成碎矩阵
- **D3（attn_tp=8 + ep=8）**: attn 走 TP，MoE 走 EP。最终采用此方案

---

## 归档于 2026-03-31（旧版 EP8 实施记录）

旧 EP8 配置（attn_tp=8 + dense_tp=8 + embed_tp=8）：
- LongcatFlashExperts 沿用 `FusedMoEGMM`，每 rank 持有 32 个 routed expert
- 走 EP8 双重 AllToAll（双路由），跳过 npu_moe_distribute_dispatch K=24 上限
- 实测 prefill ~250ms，decode ~170ms

被新版 ep8 + dense DP + 主 embed DP 替代后归档。

---

## 阶段 1 - 并行化改造完成归档

### 部署方案确认

按客户要求实施：
| 模块 | TP 大小 | 备注 |
|------|--------|------|
| MLA | 4 或 8 | 提供两套配置，b=2 用 4，b=1 用 8 |
| Dense MLP | 1 (DP) | 不切，每卡持有完整权重 |
| MoE | EP=8 | 走双重 AllToAll routing |
| 主 Embedding | 1 (DP) | 主表 0.8 GB 复制到每卡 |
| Ngram subtables | 8 | 沿 vocab 切，AllReduce 形式 |
| LM Head | 8 | 沿 vocab 切 |

### 与旧 EP8 实现的差异

| 模块 | 旧 ep8 (TP=8 一刀切) | 新 ep8 (本次实现) |
|------|--------------------|-------------------|
| dense MLP | TP=8 + AllReduce | DP=1（不切） |
| 主 Embedding | TP=8 + AllReduce | DP=1（不切） |
| ngram 子表 | 与主 embed 共用 embed_tp | 通过 `custom_params.ngram_embed_tp_size` 独立 TP，与主 embed 解耦 |
| MoE | EP=8 双重路由 | EP=8 双重路由（一致） |
| Attention | TP=8 | TP=8 / TP=4 双套配置 |
| BSPR check | 强制 batch % attn_tp == 0 | 已移除（与 MLA 切头不冲突） |

### 代码改动

模型侧自闭环，未改框架代码：`ngram_embed_tp_size` 通过 `model_config.custom_params` 读取，ngram 通信复用 `embed_tp_group`。

| 文件 | 变更 |
|------|------|
| `models/longcat_flash_lite/models/modeling_longcat_flash_lite.py` | `LongcatFlashMLP` 切到 `dense_tp_group`；`NgramEmbedding` 增加独立 `ngram_tp_size/rank/group` 参数（由 model `__init__` 从 `custom_params` 读出后注入） |
| `models/longcat_flash_lite/config/longcat_flash_lite_rank_8_8ep.yaml` | `dense_tp_size=1`, `embed_tp_size=1`, `custom_params.ngram_embed_tp_size=8` |

### 实测性能（验收）

| 配置 | Prefill | Decode | 对比基线 | 输出验证 |
|------|---------|--------|----------|----------|
| baseline (旧 TP8) | 4475 ms | 273 ms | — | — |
| ep8 attn_tp=8 batch=1 | 198 ms | 161 ms | -41% decode | ✓ 与 baseline 完全匹配 |
| ep8 attn_tp=4 batch=2 | 207 ms | 150 ms (per req 75 ms) | -45% decode | ✓ 完全匹配 |

### 关键观察 / 经验

1. **客户方案的合理性**: dense MLP 走 DP 节省 AllReduce，虽然每卡多 1.4 GB 权重，但通信省下来的时间显著，单卡 HBM 完全负担得起。
2. **attn_tp=4 + batch=2 比 attn_tp=8 + batch=1 总吞吐更好**: 因 attn_tp 更小时 q/k/v/o 矩阵更大，cube 利用率更高；同时 batch=2 让 MoE GMM 也更接近峰值。
3. **ngram 子表 TP=8 不必转 ReduceScatter**: 当前的 AllReduce 形式已经能并行，没有 SP 串接，转换无收益。

### 验证结果

| 指标 | attn_tp=8 + batch=1 | attn_tp=4 + batch=2 |
|------|---------------------|---------------------|
| prefill | 198 ms | 220 ms |
| decode/token | 161 ms | 150 ms |
| 单卡显存峰值 | 19.6 GB | 20.4 GB |
| 输出与 baseline 对齐 | ✓ | ✓ |

阶段 1 通过，进入阶段 2。

---

## 阶段 2 — KVCache + FA 改造（归档于 2026-04-28）

### 实施

- 替换 `_forward_legacy` 中的 manual SDPA 路径，新增 `_forward_prefill_pa`（NTD/TND FA + npu_kv_rmsnorm_rope_cache）和 `_forward_decode_pa`（BSND_NBSD FA + absorb path）
- 新增 `LongcatFlashNgramForCausalLM.process_weights_after_loading` → 调用 `_init_absorb_weights` 拆分 `kv_b_proj` 为 `kv_b_proj_w_k / kv_b_proj_w_v`
- 新增 `LongcatFlashNgramForCausalLM.init_pa_cache(...)`，按 `batch_size_per_dp_rank * ceil(max_seq/block_size)` 分配 `cache_nope`、`cache_rope`、`block_table`
- 删除 `LongcatFlashMLA._forward_legacy`、`self.k_cache/v_cache/k_cache_unit/v_cache_unit`、`self.scaling`，避免 `model_worker._init_kvcache` 误分配 ~336 MB legacy buffer
- 简化 `forward()` dispatcher，仅根据 `forward_metadata.is_prefill` 分流到 `_forward_prefill_pa` / `_forward_decode_pa`

### 验证结果

| 配置 | Prefill | Decode | 输出对齐 |
|------|---------|--------|----------|
| Stage 1 baseline | 207 ms | 151 ms | (基线) |
| Stage 2 PA + FA | 455 ms | 145 ms | ✓ 与 baseline 完全一致 |

`[MLA] use_pa=True ...` 调试日志确认 `_forward_prefill_pa` 实际被执行。

### 已知问题 / 待跟进

- **Prefill 退化 +120%**: 与 tp8 stage 1 数据（持平 -1.6%）差距巨大，未充分定位根因。怀疑：
  1. 旧 cache 残留（已修复，cleanup commit）
  2. 双 BMM 在 attn_tp=4 时（每卡 8 head）launch 开销显著（待图模式验证）
  3. `npu_kv_rmsnorm_rope_cache` 在 attn_dp=2 模式下首次 launch 被序列化
- **Decode 持平**：与 tp8 持平表现一致；图模式（stage 4）才能拿到收益

需在 stage 3/4 定位 prefill 退化问题，或引入混合方案（prefill 走 legacy SDPA，decode 走 PA absorb）。

---

## 阶段 3 - 融合算子（已完成于 2026-04-28）

### 实施内容

| 改动 | 位置 | 备注 |
|------|------|------|
| `LongcatFlashRMSNorm.forward` 用 `npu_rms_norm` / `npu_add_rms_norm` 融合 add+norm | line 33-58 | 支持 `(x, residual)` 接口，与 tp8 一致 |
| `LongcatFlashMLP` 合并 `gate_proj+up_proj`→`MergedColumnParallelLinear`，再走 `npu_swiglu` | line 491-525 | 1 GEMM + 1 fused activation 替代原 2 GEMM + 2 element-wise |
| `LongcatFlashDecoderLayer.forward` 改为残差链式传递（4 处 `npu_add_rms_norm`） | line 970-1007 | 节省 28 layer × 4 = 112 个独立 launch / step |
| `load_weights` 增加 dense MLP `gate_proj`/`up_proj` → `gate_up_proj` 装载 | line 1497 之后 | 兼容 checkpoint 格式 |

### 实测性能（attn_tp=4, batch=2, seq=1024）

| 阶段 | Prefill (steady) | Decode avg | 说明 |
|------|------------------|------------|------|
| Stage 1 baseline (legacy SDPA) | 207 ms | 151 ms | manual SDPA + 手写 RMSNorm/SiLU |
| Stage 2 (PA+FA only) | 455 ms (+120%) | 145 ms (-4%) | PA 路径，仅替换 KV cache + attention |
| **Stage 3 (PA+FA + 融合算子)** | **244 ms (+18%)** | **124 ms (-18%)** | 加上 `npu_rms_norm` / `npu_add_rms_norm` / `gate_up_proj` + `npu_swiglu` |

### 验证

- ✓ 输出与 baseline 一致（"computed as a weighted sum of the values..."）
- ✓ NPU 利用率正常，无 OOM 或超时
- 待优化：prefill 仍比 baseline 慢 ~37 ms（推断来自 `kv_b_proj_w_k`/`w_v` BMM 在 ND→NZ 转换的开销）
- Decode 已经从 151 → 123 ms，是 stage 2 + stage 3 联合收益

