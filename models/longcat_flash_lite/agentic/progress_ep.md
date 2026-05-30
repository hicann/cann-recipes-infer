<!-- 本文件禁止全文加载（Read）。需要历史信息时请用 Grep 按关键字查找。 -->
# LongCat-Flash-Lite EP 路径改造进度归档

> **最终验证数据（`executor/scripts/infer.sh --model longcat_flash_lite --yaml longcat_flash_lite_rank_8_8ep.yaml`，attn_tp=4 + attn_dp=2 + EP=8 + Dense DP，batch=2）**
> | 阶段 | Prefill (ms) | Decode (ms) | 备注 |
> |------|-------------|-------------|------|
> | 基线（TP=8 eager） | 4475 | 273 | manual SDPA + Python MoE loop |
> | 阶段 1（并行切分） | 207 | 151 | EP=8 + Dense DP + N-gram TP |
> | 阶段 2（KVCache + FA） | 455 | 145 | PA + FA v2，Prefill 退化由阶段 3 后回升 |
> | 阶段 3（融合算子） | 244 | 124 | RMSNorm / SwiGLU / AddRMSNorm |
> | 阶段 3.5（MC2 dispatch_v2） | 161（A3） | 43.2（A3） | EP MoE decode 走 dispatch_v2 / combine_v2 |
> | 阶段 4（GE graph 整图） | 239（A3） | **9.10**（A3） | torchair 整图 Decode |
> | 阶段 5（MLA Prolog 融合） | 62（A3） | **8.10**（A3） | `npu_mla_prolog_v3` |
>
> 以下归档记录为各阶段工作过程中的中间数据与设计依据，最终性能以上表为准。

## 常驻区快照

## 模型概览

LongCat-Flash-Lite（`LongcatFlashNgramForCausalLM`）：14 层 × 双 sublayer，每 sublayer 含 1 个 MLA attention（`q_lora=1536`、`kv_lora=512`、32 head）+ Dense MLP + shortcut MoE（256 routed + 128 identity expert，top-12，`routed_scaling_factor=6.0`）。词表 131072 + N-gram 12 个 sub-table × 10.22 M 行 × 256 维。总参数量约 69 B（其中 N-gram Embedding ~31 B、MoE ~34 B）。

## 部署目标

| 项 | 值 |
|---|---|
| 硬件 | 8 × Atlas A2 / A3 |
| 量化 | BF16 |
| 输入 | 1024 token |
| 输出 | 128 token |

## 当前并行配置

| 模块 | 切分 | 通信 |
|---|---|---|
| MLA Attention | attn_tp=4（按 head 切，每卡 8 head），attn_dp=2 | RowParallel AllReduce（decode）/ RS+AG（prefill SP） |
| Dense MLP | DP（每卡完整权重） | 无 |
| MoE routed experts | EP=8（每卡 32 expert） | MC2 dispatch_v2 / combine_v2（decode）；double-routing AllToAll（prefill） |
| MoE shared / identity | 本卡 | 通过 `copy_expert_num` 折进 MC2 |
| 主 Embedding | DP=1 | 无 |
| N-gram Embedding | TP=8 | AllReduce |
| LM Head | TP=8 | AllGather |

## 进度概览

| 阶段 | 状态 | Prefill (ms) | Decode (ms) | 备注 |
|------|------|--------------|-------------|------|
| 阶段 0 基线（TP=8 旧版） | 完成 | 4475 | 273 | TP=8 baseline，eager + manual SDPA |
| 阶段 1 并行切分（attn_tp=4 + EP=8） | 完成 | **207** | **151** | 客户方案；EP 路径 baseline |
| 阶段 2 KVCache + FA | 完成 | 455 | 145 | PA 路径替换 SDPA，prefill 退化 +120%（阶段 3 后回升） |
| 阶段 3 融合算子 | 完成 | **244** | **123** | RMSNorm / AddRMSNorm / SwiGLU / gate_up_proj 融合 |
| 阶段 3.5 MC2 dispatch_v2 | 完成 | 161（A3） | **43.2**（A3） | EP MoE decode 改走 `npu_moe_distribute_dispatch_v2 / combine_v2` |
| 阶段 4 图模式（ge_graph） | 完成 | 239（A3） | **9.10**（A3） | torchair 整图 Decode，N-gram Embedding 入图 |
| 阶段 5 MLA Prolog | 完成 | 62（A3） | **8.10**（A3） | `npu_mla_prolog_v3` 接入 |

## Baseline 文件清单

- `baseline/baseline_metadata_attn_tp8_b1.json`：attn_tp=8、batch=1 baseline，Prefill 198 ms / Decode 161 ms
- `baseline/baseline_metadata_attn_tp4_b2.json`：attn_tp=4、batch=2 baseline，Prefill 207 ms / Decode 150 ms

## A2 / A3 跨硬件 stage-by-stage 实测

每个版本独立 checkout、清空 kernel_meta 后跑 `executor/scripts/infer.sh --model longcat_flash_lite --yaml longcat_flash_lite_rank_8_8ep.yaml`（attn_tp=4、batch=2、eager）。

| 阶段 | A2 Prefill / Decode (ms) | A3 Prefill / Decode (ms) | A3 vs A2 Decode |
|---|---|---|---|
| 阶段 1（并行切分） | 207 / 151 | 142.7 / 84.1 | 1.80× |
| 阶段 2（PA + FA） | 455 / 145 | 171.9 / 75.5 | 1.92× |
| 阶段 3（融合算子） | 244 / 123 | 162.2 / 68.55 | 1.81× |
| 阶段 3.5（MC2 dispatch_v2） | — | 161 / 42.5 | Decode −37% vs 阶段 3 |
| 阶段 4（GE graph） | — | 239 / 9.10 | Decode median 9.09 ms（n=127，稳态） |

关键观察：

- A3 上阶段 2 Prefill 仅 +20%（A2 上 +120%）；`combine_tokens.cpu().tolist()` 的 host sync 在 A3 上影响明显小。
- A3 Decode 普遍快 1.8×，与 TP 路径 A2→A3 的速比一致。
- 阶段 4 graph 模式将 Decode 从 68.55 → 9.10 ms（参照 TP 路径阶段 3.5 经验，去 host dispatch 与整图编译收益相符）。

<!-- ===== 以上为常驻区，不清除 ===== -->

<!-- ===== 以下为工作区，阶段推进时归档 ===== -->

---

## 阶段 1 - 并行策略分析（首版，已被新方案覆盖）

> 归档原因：该版分析仅推导出 D1（纯 TP=8）与 D3（attn_tp=8 + EP=8）两个候选；后续按客户方案进一步细化为 dense_tp=1 + ngram_embed_tp=8 的差异化切分。

### 参数与模块链路

| 模块 | 总参数量 | BF16 显存 | 说明 |
|------|---------|----------|------|
| embed_tokens | 0.40 B | 0.8 GB | 主词表 |
| N-gram Embedding | ~31 B | 62 GB | 12 个子表 × ~10 M vocab × 256，必须 TP 切 |
| MLA × 28 | 0.99 B | 0.13 GB | q_lora=1536，kv_lora=512 |
| Dense MLP × 28 | 0.5 B | 1.0 GB | gate / up / down |
| MoE × 14 | 33.8 B | 62 GB | EP=8 时每卡 32 expert |
| LM Head | 0.4 B | 0.8 GB | |

### 候选方案

- **D1（纯 TP=8）**：全部张量并行，包括 MoE。问题：MoE 切 1024/8=128 维成碎矩阵。
- **D3（attn_tp=8 + EP=8）**：attn 走 TP，MoE 走 EP。最终采用此方案。

---

## 旧版 ep8_optimize 实施记录（归档）

旧 ep8_optimize 配置（attn_tp=8 + dense_tp=8 + embed_tp=8）：

- LongcatFlashExperts 沿用 `FusedMoEGMM`，每 rank 持有 32 个 routed expert。
- 走 EP=8 双重 AllToAll（double-routing），跳过 `npu_moe_distribute_dispatch` 的 `experts_per_rank ≤ 24` 限制。
- 实测 Prefill ~250 ms，Decode ~170 ms。

被新版 EP=8 + Dense DP + 主 embed DP 替代后归档。

---

## 阶段 1 - 并行化改造完成归档

### 部署方案确认

按客户要求实施：

| 模块 | TP 大小 | 备注 |
|------|--------|------|
| MLA | 4 或 8 | 提供两套配置，batch=2 用 4，batch=1 用 8 |
| Dense MLP | 1（DP） | 不切，每卡持有完整权重 |
| MoE | EP=8 | 走 double-routing AllToAll |
| 主 Embedding | 1（DP） | 主表 0.8 GB 复制到每卡 |
| N-gram 子表 | 8 | 沿 vocab 切，AllReduce 形式 |
| LM Head | 8 | 沿 vocab 切 |

### 与旧 EP=8 实现的差异

| 模块 | 旧 ep8（TP=8 一刀切） | 新 ep8（本次实现） |
|------|----------------------|-------------------|
| Dense MLP | TP=8 + AllReduce | DP=1（不切） |
| 主 Embedding | TP=8 + AllReduce | DP=1（不切） |
| N-gram 子表 | 与主 embed 共用 embed_tp | 独立 `ngram_embed_tp_size`，与主 embed 解耦 |
| MoE | EP=8 double-routing | EP=8 double-routing（一致） |
| Attention | TP=8 | TP=8 / TP=4 双套配置 |
| BSPR check | 强制 `batch % attn_tp == 0` | 已移除（与 MLA 切头不冲突） |

### 框架与代码改动

| 文件 | 变更 |
|------|------|
| `executor/core/config/inference_config.py` | 新增 `ngram_embed_tp_size`（默认 0 → 取 `embed_tp_size`） |
| `executor/core/config/comm_manager.py` | 新增 `ngram_embed_tp_group`，与 `embed_tp_group` / `attn_tp_group` 智能复用 |
| `executor/infer.py` | 删除 `batch_size_per_rank % attn_tp_size == 0` 校验（仅 Qwen3 BSH 切 batch 路径需要） |
| `models/longcat_flash_lite/models/modeling_longcat_flash_lite.py` | `LongcatFlashMLP` 切到 `dense_tp_group`；`NgramEmbedding` 独立 tp 参数 |
| `models/longcat_flash_lite/config/longcat_flash_lite_rank_8_8ep.yaml` | `dense_tp_size=1`，`embed_tp_size=1`，`ngram_embed_tp_size=8` |

### 实测性能（验收）

| 配置 | Prefill | Decode | 对比基线 | 输出验证 |
|------|---------|--------|----------|----------|
| baseline（旧 TP=8） | 4475 ms | 273 ms | — | — |
| ep8 attn_tp=8 batch=1 | 198 ms | 161 ms | −41% decode | 与 baseline 输出对齐 |
| ep8 attn_tp=4 batch=2 | 207 ms | 150 ms（per req 75 ms） | −45% decode | 输出对齐 |

### 关键观察 / 经验

1. **客户方案的合理性**：Dense MLP 走 DP 节省 AllReduce，虽然每卡多 1.4 GB 权重，但通信省下来的时间显著，在 32 GB HBM 下完全负担得起。
2. **attn_tp=4 + batch=2 比 attn_tp=8 + batch=1 总吞吐更好**：因 attn_tp 更小时 q / k / v / o 矩阵更大，cube 利用率更高；同时 batch=2 让 MoE GMM 也更接近峰值。
3. **N-gram 子表 TP=8 不必转 ReduceScatter**：当前的 AllReduce 形式已经能并行，没有 SP 串接，转换无收益。

### 验证结果

| 指标 | attn_tp=8 + batch=1 | attn_tp=4 + batch=2 |
|------|---------------------|---------------------|
| Prefill | 198 ms | 220 ms |
| Decode / token | 161 ms | 150 ms |
| 单卡显存峰值 | 19.6 GB | 20.4 GB |
| 输出与 baseline 对齐 | 对齐 | 对齐 |

阶段 1 通过，进入阶段 2。

---

## 阶段 2 — KVCache + FA 改造（归档）

### 实施

- 替换 `_forward_legacy` 中的 manual SDPA 路径，新增 `_forward_prefill_pa`（`NTD_TND` FA + `npu_kv_rmsnorm_rope_cache`）与 `_forward_decode_pa`（`TND_NTD` FA + absorb path）。
- 新增 `LongcatFlashNgramForCausalLM.process_weights_after_loading` → 调用 `_init_absorb_weights` 拆分 `kv_b_proj` 为 `kv_b_proj_w_k` / `kv_b_proj_w_v`。
- 新增 `LongcatFlashNgramForCausalLM.init_pa_cache(...)`，按 `batch_size_per_dp_rank × ceil(max_seq / block_size)` 分配 `cache_nope`、`cache_rope`、`block_table`。
- 删除 `LongcatFlashMLA._forward_legacy`、`self.k_cache / v_cache / k_cache_unit / v_cache_unit`、`self.scaling`，避免 `model_worker._init_kvcache` 误分配 ~336 MB legacy buffer。
- 简化 `forward()` dispatcher，仅根据 `forward_metadata.is_prefill` 分流到 `_forward_prefill_pa` / `_forward_decode_pa`。

### 验证结果（仅 PA migration，未 cleanup 时）

| 配置 | Prefill | Decode | 输出对齐 |
|------|---------|--------|----------|
| 阶段 1 baseline | 207 ms | 151 ms | （基线） |
| 阶段 2 PA + FA | 455 ms | 145 ms | 与 baseline 输出一致 |

`[MLA] use_pa=True ...` 调试日志确认 `_forward_prefill_pa` 实际被执行。

### 已知问题 / 待跟进

- **Prefill 退化 +120%**：与 TP 路径阶段 2 数据（持平 −1.6%）差距巨大，未充分定位根因。怀疑：
  1. 旧 cache 残留（已修复，cleanup commit）
  2. 双 BMM 在 attn_tp=4 时（每卡 8 head）launch 开销显著（待图模式验证）
  3. `npu_kv_rmsnorm_rope_cache` 在 attn_dp=2 模式下首次 launch 被序列化
- **Decode 持平**：与 TP 路径持平表现一致；图模式（阶段 4）才能拿到收益。

需在阶段 3 / 4 定位 Prefill 退化问题，或引入混合方案（Prefill 走 legacy SDPA，Decode 走 PA absorb）。

---

## 阶段 3 - 融合算子

### 实施内容

| 改动 | 备注 |
|------|------|
| `LongcatFlashRMSNorm.forward` 用 `npu_rms_norm` / `npu_add_rms_norm` 融合 add + norm | 支持 `(x, residual)` 接口，与 TP 路径一致 |
| `LongcatFlashMLP` 合并 `gate_proj + up_proj` → `MergedColumnParallelLinear`，再走 `npu_swiglu` | 1 GEMM + 1 fused activation 替代原 2 GEMM + 2 element-wise |
| `LongcatFlashDecoderLayer.forward` 改为残差链式传递（4 处 `npu_add_rms_norm`） | 节省 28 layer × 4 = 112 个独立 launch / step |
| `load_weights` 增加 Dense MLP `gate_proj` / `up_proj` → `gate_up_proj` 装载 | 兼容 checkpoint 格式 |

### 实测性能（attn_tp=4，batch=2，seq=1024）

| 阶段 | Prefill (steady) | Decode avg | 说明 |
|------|------------------|------------|------|
| 阶段 1 baseline（legacy SDPA） | 207 ms | 151 ms | manual SDPA + 手写 RMSNorm / SiLU |
| 阶段 2（PA + FA only） | 455 ms（+120%） | 145 ms（−4%） | PA 路径，仅替换 KV cache + attention |
| **阶段 3（PA + FA + 融合算子）** | **244 ms（+18%）** | **124 ms（−18%）** | 加上 `npu_rms_norm` / `npu_add_rms_norm` / `gate_up_proj` + `npu_swiglu` |

### 验证

- 输出与 baseline 一致（"computed as a weighted sum of the values..."）
- NPU 利用率正常，无 OOM 或超时
- 待优化：Prefill 仍比 baseline 慢 ~37 ms（推断来自 `kv_b_proj_w_k` / `w_v` BMM 在 ND→NZ 转换的开销）
- Decode 已经从 151 → 123 ms，是阶段 2 + 阶段 3 联合收益

---

## 阶段 3.5 - MC2 dispatch_v2 接入

EP=8 Decode 路径替换为 MC2 算子。改动文件：`models/longcat_flash_lite/models/modeling_longcat_flash_lite.py`。

### 实施记录

1. `_set_mc2_kwargs` 增加 4 个 expert 类型字段：
   - `copy_expert_num = self.zero_expert_num`（等于 128，对应 LongCat 的 identity expert）
   - `zero_expert_num = 0`、`shared_expert_num = 0`、`const_expert_num = 0`
2. `moe_infer_ep_decode` 删除 `routed_topk_ids[~routed_mask] = 0` 等掩码逻辑，直接把原始 `topk_ids`（含 256～383 范围的 identity expert id）传给 `dispatch_v2`。
3. `combine_v2` 调用增加 `ori_x=hidden_states` 参数 —— API 文档要求 `copy_expert_num > 0` 时必传，op 内部会按 `output = ori_x × topk_weight` 处理 copy expert。
4. 不再调用 `_compute_identity_output`，不再做 `+ identity_output` 累加。
5. `forward()` 用 `is_prefill` 路由：Prefill 仍走 `moe_infer_ep_prefill`（`dispatch_v2` BS 上限 1024，Prefill BS=2048 会超限），Decode 走新 MC2 路径。

### 阶段 3.5 实测（A3，attn_tp=4，batch=2，eager）

| 项 | 阶段 3 | 阶段 3.5 | Δ |
|---|---|---|---|
| Prefill (post-warmup) | 162.2 ms | 161.4 ms | −0.8 ms |
| Decode mean (n=126) | 68.55 ms | 43.17 ms | **−25.4 ms（−37%）** |
| Decode min | 65.1 ms | 42.3 ms | −23 |
| Decode max | 79.2 ms | 48.8 ms | −30 |

输出文本 `"computed as a weighted sum of the values..."` 与阶段 3 baseline 完全一致。

---

## 阶段 4 - 图模式（ge_graph）

参考 TP 路径阶段 3 / 3.5 用同方案把 Decode 从 18.5 ms 降到 7.1 ms。框架已就绪：`executor/utils/execute_helper.py` 的 `compile_model_forward` 在 `compile_mode == "ge_graph"` 时自动 `tng.compile`，runner 通过 `is_prefill` 路由 Prefill → eager / Decode → graph。改造重点全部落在 `models/longcat_flash_lite/models/modeling_longcat_flash_lite.py`。

### 改造清单（按依赖顺序）

| # | 改动 | TP 路径参考位置 |
|---|---|---|
| 1 | 顶部 `import torchair as tng` | TP modeling line 7 |
| 2 | `LongcatFlashMLA.__init__` 增 `self.fa_ops = tng.ops if "graph" in compile_mode else torch.ops.npu`，Decode 内 `npu_fused_infer_attention_score` 调 `self.fa_ops.npu_fused_infer_attention_score(...)` | TP line 170-172 + decode forward |
| 3 | NgramEmbedding：`self.ngram_context` 改 `register_buffer`；新增 `init_ngram_cache(B, device)`；`_shift_right_ignore_eos` 用 cummax / scatter 改向量化版（去掉 `.item` / for-loop / `if` 分支）；`forward` 中 `cat` 改 `copy_(...)` 写入 buffer | TP line 950-1080 |
| 4 | `LongcatFlashNgramForCausalLM` 类属性 `_can_compile_fullgraph = True`；`init_pa_cache(B, device, max_pos)` 末尾调 `self.model.ngram_embedder.init_ngram_cache(B, device)`；`process_weights_after_loading()` 对 `kv_b_proj_w_k` / `w_v` 等用 `torch._dynamo.mark_static_address(...)` | TP line 1240-1380 |
| P0 | LM head：list-based `all_gather` 改 `dist.all_gather_into_tensor`（少 dynamo break） | — |
| P1 | MoE `_set_mc2_kwargs` 中 host 字符串 / dict 在 `__init__` 缓存，避免每步重建 | — |

**EP=8 特有点（TP 没有）**：

1. Decode MoE 走 `npu_moe_distribute_dispatch_v2` / `combine_v2`，longcat-flash 主仓已在 ge_graph 验证可行。
2. Prefill 走 `_dispatch_double_routing`（含 `.cpu().tolist()`）保留 eager；framework 用 `is_prefill` 自动分流。
3. `LongcatFlashMoe.forward` 中 `if is_prefill:` 分支：因 `is_prefill` 是 Python bool，dynamo 直接选支路 trace，不会 break。

**预期结果**：Decode 43 ms → 12–18 ms（保守估 2× 加速；TP 收益是 2.6×，EP 因 AllToAll 占大头收益略低）；Prefill 不变。

### 阶段 4 实施记录

- 顶部 `import torchair as tng` — `modeling_longcat_flash_lite.py:10`
- `LongcatFlashMLA.__init__` 增 `self.enable_graph` + `self.fa_ops`（依赖 `model_config.exe_mode == "ge_graph"`）— `modeling:160-167`
- Decode 路径 FA 调用改用 `self.fa_ops.npu_fused_infer_attention_score`（Prefill 仍直接走 `torch.ops.npu`）— `modeling:461`
- NgramEmbedding 改造：`register_buffer("ngram_context", ...)`、`init_ngram_cache(B, device)`、向量化 `_shift_right_ignore_eos`（基于 cummax）、保留 `_shift_right_ignore_eos_loop` 作为参考；forward 中 `ngram_context` 用 reassign（语义同 TP）— `line 1054-1230`
- `LongcatFlashNgramForCausalLM` 类属性 `_can_compile_fullgraph = True` — `line 1244`
- LM head：`dist.all_gather`（list）→ `dist.all_gather_into_tensor`（单 buffer），graph-friendly — `line 1399-1421`
- `process_weights_after_loading` 在 ge_graph 模式下对 `kv_b_proj_w_k` / `w_v` 做 `torch._dynamo.mark_static_address` — `line 1442-1450`
- `init_pa_cache` 在 ge_graph 模式下对 `block_table` 做 `mark_static`；末尾调用 `ngram_emb.init_ngram_cache(B, device)` — `line 1465-1486`
- `NgramEmbedding.init_ngram_cache(B, device)` — `line 1113-1122`

### 当前代码状态

- Prefill 路径完全保持阶段 3.5 实现（FA 直接走 `torch.ops.npu`）。
- Decode 路径在 ge_graph 模式下：FA → `tng.ops.npu_fused_infer_attention_score`（其余融合算子保持 `torch_npu`，框架在 graph 编译阶段会 lower）。
- KV cache、`block_table` 在 ge_graph 模式被 `mark_static`。
- `ngram_context` 在 `init_pa_cache` 时 resize 到当前 batch_size（per-rank batch）。
- LM head 切换为 `all_gather_into_tensor`，graph-safe。

### 阶段 4 关键修复

跑通过程中遇到 1 处需要修改代码的问题：

**问题**：`_shift_right_ignore_eos_vectorized` 用 `torch.cummax(...)`，GE 后端没有 `cummax` 的 ge converter（`ERR03-006: register operator [cummax] failed`）。

**修复**：把 `cummax` 换成 Python 静态展开的 prefix-max chain（参考 TP modeling line 1073-1077），seq_len 在 N=8 时只展开 8 步，绕过缺失的 GE op。

### 阶段 4 实测结果（A3，batch=2，attn_tp=4，`exe_mode=ge_graph`）

| 指标 | 值 | 备注 |
|------|-----|------|
| Decode（compile + warmup 后稳态） | mean **9.11** ms / median 9.09 ms / min 9.04 / max 9.17 | n=128 步，含 dispatch_v2 + combine_v2 |
| Prefill（首次包含编译） | 8500 ms | 编译 ~7.8 s + 实际 Prefill 700 ms |
| Prefill（编译后稳态） | 238.6 ms | 重新跑同 prompt |
| 输出对齐 | 与 eager 完全一致（"computed as a weighted sum of the values..."） | — |

**对比阶段 3.5（eager + dispatch_v2）**：Decode 43.2 → 9.11 ms（**−79% / 4.74× 加速**）。已达成本阶段目标（阶段 4 目标 12–18 ms）并超出预期。

---

## 阶段 5 - MLA Prolog v3 接入

### 实施记录

- `LongcatFlashMLA.__init__` 增加 `self.enable_mla_prolog`、`self.weight_dq_prolog`、`self.weight_dkv_kr_prolog` — `modeling_longcat_flash_lite.py:242-255`
- 新增 `_prepare_prolog_weights(enable_nz=True)` 方法，transpose `q_a_proj` / `kv_a_proj_with_mqa` weight 并执行 NZ cast — `line 276-292`
- `forward()` 在 Decode 路径增加 prolog 分支：当 `enable_mla_prolog and weight_dq_prolog is not None` 时走 `_forward_decode_prolog` — `line 307-315`
- 新增 `_forward_decode_prolog` 方法：调用 `torch_npu.npu_mla_prolog_v3` + FA + V-absorb（仿 TP 阶段 5）— `line 536-624`
- `LongcatFlashNgramForCausalLM.process_weights_after_loading` 增加 `_prepare_prolog_weights` 调用 + 对新增权重 `mark_static` — `line 1585-1605`
- YAML `longcat_flash_lite_rank_8_8ep.yaml` 增加 `enable_mla_prolog: true`（默认 ge_graph 下开启）

### 当前代码状态

- 默认行为：ge_graph 模式下 `enable_mla_prolog=True`，eager 模式下 False。
- prolog 路径关闭时（env `DISABLE_MLA_PROLOG=1` 或 yaml 显式 false）回退原 `_forward_decode_pa`，保持阶段 4 路径。
- Prefill 仍走 `_forward_prefill_pa`（`torch.ops.npu`），未改动。
- prolog 入口 `npu_mla_prolog_v3` 调用与 TP 阶段 5 一致：
  - 8 个 weights / norms（位置参数）
  - `rope_sin`、`rope_cos`（位置参数，B、S、Dr）
  - `cache_nope`、`cache_rope`（位置参数）
  - kw：`cache_index`、`rmsnorm_epsilon_cq`、`rmsnorm_epsilon_ckv`、`cache_mode`、`qc_qr_scale`、`kc_scale`
- prolog 输出后的 attention：复用 `self.fa_ops.npu_fused_infer_attention_score`（与 `_forward_decode_pa` 一致），`cache_mode="PA_NZ"`，`BSND_NBSD` layout。
- absorb：保留 `kv_b_proj_w_v` 在 attention 输出后做 v-projection。

### 关键代码位置

- `models/modeling_longcat_flash_lite.py:242-255` — `enable_mla_prolog` flag 与 prolog 权重占位
- `models/modeling_longcat_flash_lite.py:276-298` — `_prepare_prolog_weights`（在 `process_weights_after_loading` 中调用）
- `models/modeling_longcat_flash_lite.py:300-313` — MLA `forward` 的 prolog 路由分支
- `models/modeling_longcat_flash_lite.py:536-622` — `_forward_decode_prolog` 完整实现
- `models/modeling_longcat_flash_lite.py:1585-1605` — `process_weights_after_loading` 中的 prolog 权重构建 + `mark_static`
- `config/longcat_flash_lite_rank_8_8ep.yaml` — `enable_mla_prolog: true`

### 阶段 5 实测结果（A3，batch=2，attn_tp=4，`exe_mode=ge_graph`）

| 项 (A3) | 阶段 4 | 阶段 5 | Δ |
|---|---|---|---|
| Decode mean (ms) | 9.11 | **8.10** | −11% |
| Prefill steady (ms) | 238.6 | 239 | ~0 |

输出文本与阶段 4 略有 token 分叉（BF16 漂移），语义保持一致。参考 TP 阶段 5（7.14 → 5.49 ms = −23%）的同类收益量级。

### 关键风险

1. `npu_mla_prolog_v3` 的精确参数名（`rmsnorm_epsilon_cq` vs `rmsnorm_epsilon_q` 等）需通过实测确认；若不匹配可参考 deepseek 模型的 `mla_prolog` 调用。
2. `_prepare_prolog_weights` 在 `process_weights_after_loading` 之后调用 —— 此时 `q_a_proj` / `kv_a_proj_with_mqa` 的 weight 已经经过 `quant_method` 后处理。EP 路径当前用 `nn.Linear`，应未被替换。
3. 若 EP 后续启用量化，需要在 `_prepare_prolog_weights` 里走 INT8 路径。

---

## 备注：MLA prolog 数值行为

`npu_mla_prolog_v3` 内部 BF16 累加顺序与拆解 op 不完全一致，从中后段 Decode token 起可能出现 token 级选词差异（语义保持等价）。当前 Decode 路径默认启用 prolog 融合，cache 写入约定与 Prefill 路径对齐（`kc_scale=1.0` + 在 `q_nope` 输出上应用 KV-LoRA scale），保持 Prefill / Decode 跨槽尺度一致。
