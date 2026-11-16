# KVCache / FA 精度排查专题

现象涉及 KVCache 写入、FlashAttention 入参、Prefill/Decode 阶段差异或分页注意力时使用本专题。

**入参校验器**：`scripts/fa_debug_utils.py` 覆盖 FIA v1/v2（`npu_fused_infer_attention_score` 系列），在调用前插入 `debug_fa_all_params()` 检查。`--self-test` 跑模型调用与 API 支持配置的合成用例，以及校验器覆盖的错误用例。mask 内容校验可能引入设备同步；观测会改变现象时，按主线 4.3 取证。

**调用方式**：`path=` 标记当前路径，`fa_version=` 声明 v1 还是 v2，`fa_call_kwargs=` 传入实际调用的参数字典才会检查 v1/v2 参数名混用。除 `sparse_mode` 外各参数带默认值，省略即按默认值校验，与实际调用不符会得到相反结论，逐个显式传入。

**日志级别**：ERROR 表示违反校验器覆盖的入参约束，WARN 表示需核对的可疑点或未覆盖的组合，OK / INFO 是过程信息。

**校验器不覆盖**：不接收 kv_len、hidden_states、`pre_tokens` / `next_tokens` / `num_key_value_heads`，kv_len 生命周期、packed 二维形态、滑窗窗口与 MLA 的 KV head 数需人工核对。Q/K/V 已传入但不查 dtype 与 device；layout 只校验 query，分页下 K/V 是 cache 视图、形态不由 layout 名决定；scale 仅提示默认值 1.0，实际缩放公式需核对模型。

---

## 部署模式与数据协议

**部署模式适配**：诊断流程对框架部署 / 独立部署两模式均适用，差异仅在参数获取路径——框架部署从 `forward_metadata.{block_table, slot_mapping, actual_seq_lengths_*, is_prefill}` 取，独立部署从 Runner 自管的 ParallelContext / forward_metadata 实例取。`block_table` / `slot_mapping` 在框架部署下是 `Dict[attn_type → Tensor]`，attention 内按 `self.attn_type` 取出对应 tensor；独立部署 Runner 自管的形态是单 Tensor。

**packed sequence 协议**（infer 仓框架部署）：`input_ids` / `position_ids` 是 `[TotalTokens]` 一维；`hidden_states` 全程保持 `[TotalTokens, hidden_size]` 二维（不要 reshape 成 `[B, S, H]`）；阶段分支统一用 `forward_metadata.is_prefill`。详见 `model-infer-kvcache` 的「框架部署模式实施」。

按实际 FA API、硬件和部署版本，查询 cannbot-knowledge 中 sparse_mode 与 mask、pre_tokens/next_tokens 的配套取值，以及 layout/缓存格式、MLA/RoPE、量化参数约束和精度案例；参考同框架、同注意力路径的调用样例。

核对 sparse_mode 时同时看因果或滑窗语义、query 长度、量化模式和缓存读取路径；配置原则见 `model-infer-kvcache` 的「FA 算子配置」。

---

## 症状分类

| 症状 | 可能原因 | 优先排查方向 |
|------|---------|-------------|
| 输出全为重复 token 或乱码 | KVCache 写入位置错误 | slot_mapping 构造 / 写入算子（PA 用 `npu_scatter_nd_update_`，BSH legacy 用 `scatter_update_`） |
| 首个 token 正确，后续偏移 | Decode 阶段 kv_len 更新错误 | kv_len 生命周期 |
| Prefill 正确，Decode 偏差大 | Decode FA 入参错位 | sparse_mode 与 q_len / 量化模式 / 缓存形态是否匹配、actual_seq_lengths_kv |
| 输出含 NaN/Inf | 数值溢出 | scale 参数、dtype 不匹配 |
| 精度逐层累积偏差 | FA 算子数值精度 | input_layout / NZ 格式 / 量化参数 |
| 多 batch 部分样本错误 | batch 间缓存交叉污染 | block_table / slot_mapping 构造 |
| 量化模式下偏差显著放大 | 量化参数不匹配 | dequant_scale / antiquant_mode |

---

## 快速验证清单

- [ ] 1. KVCache dtype 与缓存声明、写入及 FA 读取路径一致；量化 KV 同时核对 scale 和反量化参数
- [ ] 2. device 一致性：Q/K/V、mask 与 cache 位于预期设备
- [ ] 3. block_table dtype：必须为 torch.int32
- [ ] 4. sparse_mode、mask 与注意力可见范围一致：因果多 token 路径屏蔽未来 token，滑窗路径限制历史窗口；普通全注意力单 token Decode 在有效 KV 范围内可用 0 + None，或受支持的 causal 路径
- [ ] 5. Prefill 和 Decode 的 FA 调用参数按路径配置（atten_mask、actual_seq_lengths 等；阶段分支取自 `forward_metadata.is_prefill`）
- [ ] 6. scale 按模型实际公式：常规为 `1/sqrt(head_dim)`；YaRN 的 `mscale_all_dim` 生效时保留 mscale² 修正
- [ ] 7. num_heads / num_kv_heads：按实际切分或复制结果核对每 rank 头数；MLA absorb 按 latent 单 head 核对
- [ ] 8. kv_len 初始值：按部署模式核对，两模式语义差一。框架部署在 Prefill 后写入 `seq_len - 1`，即末 token 下标，Decode 入口再 +1 得到实际长度；独立部署按 Runner 自身约定。不要用同一个值核对两种模式
- [ ] 9. input_layout 与 Q 的 shape 匹配；分页 K/V 按实际 cache 格式核对，MLA absorb 的 key/value 使用同一 latent cache
- [ ] 10. actual_seq_lengths 按 packed 直读或分页读取区分累计长度与逐请求长度，详见「actual_seq_lengths 构造」
- [ ] 11. FA 版本与实际调用的参数名一致
- [ ] 12. 自定义 mask 在有效计算区域存在全遮蔽行时，核对该算子路径的 `inner_precise` 行无效修正配置
- [ ] 13. kv_len 层内只读，由框架或 Runner 随请求推进；层内递增会导致各层写入位置错位
- [ ] 14. infer 仓框架的 hidden_states 保持 `[TotalTokens, hidden_size]` 二维形态，变长 batch 不能简单 reshape

---

## 分阶段排查

### Prefill 阶段

**KVCache 写入验证**：在写入算子（PA 模式 `npu_scatter_nd_update_` / MLA 融合写入 `npu_kv_rmsnorm_rope_cache` / `npu_kv_rmsnorm_rope_cache_v2` 或 `npu_mla_prolog_v3` 的 `cache_index` 路径 / BSH legacy 骨架 `scatter_update_`）调用前后插桩，检查：

1. 按 slot_mapping 定位本次写入的有效位置，对照写入前后的缓存
2. 纯 scatter 拷贝应与写入源逐值相等；融合 RMSNorm/RoPE 或量化写入则按原计算的 epsilon、RoPE 与量化参数构造参考结果。比对方式见 `operator_compare.md`
3. 标准 PA 的 `npu_scatter_nd_update_` 路径：cache / states 都需 view 为 `[total, num_kv_heads, head_dim]`；slot_mapping 取 `forward_metadata.slot_mapping[self.attn_type].view(-1, 1)`
4. BSH legacy 路径：scatter_update_ 的 axis 与 layout 匹配（BSH → axis=1）

**FA 入参验证**：打印并校验 Q/K/V shape、actual_seq_qlen/kvlen、sparse_mode、是否含 NaN/Inf

### Decode 阶段

**KVCache 写入验证**：

1. 验证 slot_mapping 计算（独立部署连续缓存：`expected_slot = kv_len.view(-1,1) + kv_len_offset`；PA 模式：框架部署由 `prepare_slot_mapping()` 生成，独立部署 Runner 自管按 block_table 索引构造）
2. 验证写入位置未越界
3. 验证 kv_len 逐步递增正确

**FA 入参验证**（按路径分类）：

1. 按注意力可见范围核对 sparse_mode 与 atten_mask；滑窗路径用超过窗口长度的输入验证历史截断
2. 每请求 query 长度通常为 1，MTP 时按实际验证 token 数；packed TND 的 actual_seq_qlen 传累计结束位置，如两个单 token 请求为 `[1, 2]`
3. block_table dtype 必须为 int32
4. KV cache 已写入区域无 NaN/Inf

量化 KV 路径偏差放大时，用相同输入的非量化 KV 作对照，核对 scale 与量化/反量化配置。

### 阶段切分后的现象对应

主线的执行阶段二分收敛到某一阶段后，本专题下这些现象与根因的对应关系：

| 现象 | 优先怀疑 |
|------|---------|
| Prefill 输出即不一致 | KVCache 写入位置、FA 入参配置、mask 与 sparse_mode 匹配 |
| Prefill 一致，Decode 第 1 步即不一致 | FA 入参按阶段的配置、KVCache 初始状态、actual_seq_kvlen 形态 |
| Decode 第 N 步开始不一致 | kv_len 累计更新、block 分配与缓存溢出、slot_mapping 构造 |

本表只覆盖 KVCache 与 FA 相关的成因。RoPE 与 position、量化 scale、通信、MTP 状态、采样与模型状态更新都可能产生相同现象。排除本表后回主线继续逐 layer 与逐算子收敛，收敛到算子后与参考实现比对——这条通路不依赖现象与根因的对应表。

### 分页注意力（PA）专项

1. **block_table 静态校验**：独立部署静态分配为 `numel == batch_size × (max_seq_len // block_size)`；框架部署第二维由 `ceil(max_model_len / block_size)` 得出，按 floor 核对在不整除时会误报
2. **block_table 值域校验**：仅独立部署静态分配时成立，应为 `[0, total_blocks)` 的排列且无重复。框架部署由 BlockPool 动态分配并用 null block 补齐到 max_block_num，必然重复，不适用本条
3. **slot_mapping 与 block_table 交叉验证**

---

## actual_seq_lengths 构造

| layout | 读取路径 | actual_seq_qlen | actual_seq_kvlen |
|--------|------|----------------|-----------------|
| TND | 直读 packed K/V | Q 各段的累计结束位置 | KV 各段的累计结束位置 |
| TND | Prefill，分页读取 KV（含 chunked prefill） | 累积和 | 每请求有效 KV 长度 |
| TND | Decode，分页读取 KV | 累积和，单 token 时为 `[1, 2, ...]` | **原始值** `[kv1, kv2, ...]` |
| BSH / BNSD | 两阶段 | 原始值 | 原始值 |

infer 仓框架的 `actual_seq_lengths_{q,kv}` 是逐请求长度，`actual_seq_lengths_cu_{q,kv}` 是累计长度。按实际 Q/KV 读取方式选字段，构造逻辑见 `executor/core/engine/execution_engine.py` 的 `_build_model_inputs`。
