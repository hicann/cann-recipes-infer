# KVCache / FA 精度排查专题

现象涉及 KVCache 写入、FlashAttention 入参、Prefill/Decode 阶段差异或分页注意力时使用本专题。

**入参校验器**：`scripts/fa_debug_utils.py`，在 FA 调用前插入 `debug_fa_all_params()` 一键检查。`--self-test` 跑内置用例——一组按已适配模型调用参数构造的合成配置，一组已知陷阱。排查内存类问题时不要插入：mask 校验每次调用强制一次设备到主机同步，会改变现象。

**调用方式**：`path=` 声明当前路径（`std_pa` / `sliding` / `mla_absorb`，拿不准或不属于这三类就传 None 跳过路径对照），`fa_version=` 声明 v1 还是 v2，`fa_call_kwargs=` 传入实际调用的参数字典才会检查 v1/v2 参数名混用。除 `sparse_mode` 外各参数带默认值，省略即按默认值校验，与实际调用不符会得到相反结论，逐个显式传入。

**日志级别**：ERROR 表示从入参本身可自证的矛盾，WARN 表示需人工判断的可疑点（含与路径推荐值的偏离，对照「部署模式与数据协议」的偏离表看是否可接受），OK / INFO 是过程信息。

**校验器不覆盖**：不接收 kv_len、hidden_states、`pre_tokens` / `next_tokens` / `num_key_value_heads`，kv_len 生命周期、packed 二维形态、滑窗窗口与 MLA 的 KV head 数需人工核对。Q/K/V 已传入但不查 dtype 与 device；layout 只校验 query，分页下 K/V 是 cache 视图、形态不由 layout 名决定；scale 只能发现忘设，判不出 MLA 与 YaRN 要乘的 mscale²。

---

## 部署模式与数据协议

**部署模式适配**：诊断流程对框架部署 / 独立部署两模式均适用，差异仅在参数获取路径——框架部署从 `forward_metadata.{block_table, slot_mapping, actual_seq_lengths_*, is_prefill}` 取，独立部署从 Runner 自管的 ParallelContext / forward_metadata 实例取。`block_table` / `slot_mapping` 在框架部署下是 `Dict[attn_type → Tensor]`，attention 内按 `self.attn_type` 取出对应 tensor；独立部署 Runner 自管的形态是单 Tensor。

**packed sequence 协议**（框架部署 modeling 强制要求）：`input_ids` / `position_ids` 是 `[TotalTokens]` 一维；`hidden_states` 全程保持 `[TotalTokens, hidden_size]` 二维（不要 reshape 成 `[B, S, H]`）；阶段分支统一用 `forward_metadata.is_prefill`。详见 `model-infer-kvcache` 的「框架部署模式实施」。

**sparse_mode 推荐值**：按路径取自「atten_mask 与 sparse_mode 联合校验」表，滑窗层另配 `pre_tokens=sliding_window` / `next_tokens=0`。路径判定依据见 `model-infer-kvcache` 的「FA 算子配置」。

以下偏离都是本仓已适配模型的既有写法，不作缺陷：

| 偏离 | 出现在 | 驱动因素 |
|------|--------|---------|
| 单 token Decode 用 `sparse_mode=0` + `mask=None` | pangu_7b 的 legacy BSH，该路径省略 `sparse_mode` 形参，等同传 0；qwen3_moe 的 float8 KV 分支；gemma_4 全注意力层按 BSH 视图读分页 cache | 各模型按缓存形态与量化模式自定的实现选择 |
| MLA absorb 在 `q_len>1` 的 Decode 用 `sparse_mode=3` 并保留 mask | deepseek_r1、longcat_flash 的 MTP 分支；kimi_k3 的 DSpark 草稿验证分支 | q_len 大于 1 |
| MLA absorb Prefill 与 Decode 共用一份入参，`sparse_mode` 恒为 3、mask 恒传 | kimi_k2_thinking | 实现选择，与 q_len 无关 |
| BSH legacy Prefill 用 `sparse_mode=2` | pangu_7b | 实现选择 |
| 非因果注意力的 TND PA Decode 用 `sparse_mode=0` + `mask=None`，`q_len>1` | kimi_k3 的 DSpark 草稿路径 | 注意力本身非因果，与 q_len 无关 |

没有单一因素能决定 sparse_mode：qwen3_moe 的 float8 分支由量化模式驱动，但同模型 BF16 的单 token Decode 保持 3；hy3 的 fp8 路径两阶段统一用 3。判定前对照该模型自身的分支条件——q_len、量化模式、缓存形态、注意力是否因果，不要套单一规则。

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

- [ ] 1. dtype 一致性：KVCache tensor 与模型计算 dtype 一致（BF16 / FP16）
- [ ] 2. device 一致性：所有 tensor 均在 NPU 上，无意外的 CPU tensor
- [ ] 3. block_table dtype：必须为 torch.int32
- [ ] 4. sparse_mode 与路径推荐值一致，或偏离属「部署模式与数据协议」所列的既有写法之一；注意省略该形参等同于传 0
- [ ] 5. Prefill 和 Decode 的 FA 调用参数按路径配置（atten_mask、actual_seq_lengths 等；阶段分支取自 `forward_metadata.is_prefill`）
- [ ] 6. scale 按模型实际公式：常规为 `1/sqrt(head_dim)`；MLA 与启用 YaRN 的模型在此基础上乘 mscale²，漏乘静默偏低不报错
- [ ] 7. num_heads / num_kv_heads：与模型配置一致，多卡时已按 rank 切分（`num_kv_heads_per_rank = max(num_kv_heads // attn_tp_size, 1)`）
- [ ] 8. kv_len 初始值：按部署模式核对，两模式语义差一。框架部署在 Prefill 后写入 `seq_len - 1`，即末 token 下标，Decode 入口再 +1 得到实际长度；独立部署按 Runner 自身约定。不要用同一个值核对两种模式
- [ ] 9. input_layout 与 tensor shape 匹配（标准 LLM TND PA 默认 TND；MLA absorb 用 TND_NTD）
- [ ] 10. actual_seq_lengths 按 layout、字段与阶段构造：TND 的 qlen 恒为累积和，kvlen 只在 K/V 不读历史 block 的 Prefill 与 qlen 同值、其余取原始值；BSH 两个字段两阶段都取原始值。详见「actual_seq_lengths 构造」
- [ ] 11. atten_mask 与 sparse_mode 匹配（详见「atten_mask 与 sparse_mode 联合校验」）；MLA absorb 单 token Decode 用 None
- [ ] 12. FA v1 vs v2 参数名：未混用（详见「FA v1 vs v2 关键参数名映射」）
- [ ] 13. `sparse_mode=1` 下 mask 形状为 `(Q_S, KV_S)`，两维需覆盖实际序列长度。`sparse_mode=2/3/4` 用固定 `(2048, 2048)`（允许前置广播维），与 block_table 容量无关，不要按容量放大
- [ ] 14. MLA rope 参数：query_rope/key_rope 同时传或同时不传；rope D=64，query D 仅支持 512/128
- [ ] 15. inner_precise 行无效修正：Prefill + 自定义 mask 有全遮蔽行时需设为 2/3，取值按当前调用算子的 API 文档确认
- [ ] 16. kv_len 层内只读：kv_len 在 attention/cache 内部未被修改（禁止层内递增，否则各层写入位置错位导致精度彻底损坏）
- [ ] 17. hidden_states 全程保持 `[TotalTokens, hidden_size]` 二维形态，未被 reshape 成 `[B, S, H]`（变长 batch 不能简单 reshape）

---

## 分阶段排查

### Prefill 阶段

**KVCache 写入验证**：在写入算子（PA 模式 `npu_scatter_nd_update_` / MLA 融合写入 `npu_kv_rmsnorm_rope_cache` / `npu_kv_rmsnorm_rope_cache_v2` 或 `npu_mla_prolog_v3` 的 `cache_index` 路径 / BSH legacy 骨架 `scatter_update_`）调用前后插桩，检查：

1. 写入前缓存是否为零（首次 Prefill）
2. 写入后缓存内容与 key_states 是否一致。写入是拷贝不是计算，容差应当收紧到接近逐位相等，比对方式见 `operator_compare.md`
3. PA 路径：cache / states 都需 view 为 `[total, num_kv_heads, head_dim]`；slot_mapping 取 `forward_metadata.slot_mapping[self.attn_type].view(-1, 1)`
4. BSH legacy 路径：scatter_update_ 的 axis 与 layout 匹配（BSH → axis=1）

**FA 入参验证**：打印并校验 Q/K/V shape、actual_seq_qlen/kvlen、sparse_mode、是否含 NaN/Inf

### Decode 阶段

**KVCache 写入验证**：

1. 验证 slot_mapping 计算（独立部署连续缓存：`expected_slot = kv_len.view(-1,1) + kv_len_offset`；PA 模式：框架部署由 `prepare_slot_mapping()` 生成，独立部署 Runner 自管按 block_table 索引构造）
2. 验证写入位置未越界
3. 验证 kv_len 逐步递增正确

**FA 入参验证**（按路径分类）：

1. sparse_mode 与 atten_mask 按路径取自「atten_mask 与 sparse_mode 联合校验」表，滑窗层另配 `pre_tokens=sliding_window` / `next_tokens=0`
2. actual_seq_qlen 每个值通常为 1（MTP 场景为 `next_n+1`）
3. block_table dtype 必须为 int32
4. KV cache 已写入区域无 NaN/Inf

### 阶段切分后的现象对应

主线的执行阶段二分收敛到某一阶段后，本专题下这些现象与根因的对应关系：

| 现象 | 优先怀疑 |
|------|---------|
| Prefill 输出即不一致 | KVCache 写入位置、FA 入参配置、mask 与 sparse_mode 匹配 |
| Prefill 一致，Decode 第 1 步即不一致 | FA 入参按阶段的配置、KVCache 初始状态、actual_seq_kvlen 形态 |
| Decode 第 N 步开始不一致 | kv_len 累计更新、block 分配与缓存溢出、slot_mapping 构造 |

本表只覆盖 KVCache 与 FA 相关的成因。RoPE 与 position、量化 scale、通信、MTP 状态、采样与模型状态更新都可能产生相同现象。排除本表后回主线继续逐 layer 与逐算子收敛，收敛到算子后与参考实现比对——这条通路不依赖现象与根因的对应表。同一现象反复出现且定位结论稳定的，沉淀进 `known_issues.md`。

### 分页注意力（PA）专项

1. **block_table 静态校验**：独立部署静态分配为 `numel == batch_size × (max_seq_len // block_size)`；框架部署第二维由 `ceil(max_model_len / block_size)` 得出，按 floor 核对在不整除时会误报
2. **block_table 值域校验**：仅独立部署静态分配时成立，应为 `[0, total_blocks)` 的排列且无重复。框架部署由 BlockPool 动态分配并用 null block 补齐到 max_block_num，必然重复，不适用本条
3. **slot_mapping 与 block_table 交叉验证**

---

## 常见陷阱与修复

### KVCache 写入相关

| 问题 | 根因 | 修复方案 |
|------|------|---------|
| 缓存写入位置偏移（PA 模式） | `npu_scatter_nd_update_` 的 cache / states view 形态不一致或 slot_mapping 索引错误 | cache 和 states 都 view 为 `[total, num_kv_heads, head_dim]`；slot_mapping 取 `forward_metadata.slot_mapping[self.attn_type].view(-1, 1)` |
| 缓存写入位置偏移（BSH legacy 骨架） | `scatter_update_` 的 axis 与 layout 不匹配 | BSH → axis=1 |
| 多 batch 缓存交叉写入（独立部署）| kv_len_offset 未按 max_seq_len 对齐 | `kv_len_offset = arange(0, batch*max_seq_len, max_seq_len)` |
| Prefill 后缓存部分为零 | kv_len 计算错误 | 检查 kv_len 是否等于实际有效序列长度（具体值视框架约定） |
| NZ 格式下缓存异常 | cache_mode 与 NZ 支持或 CP 开关不匹配 | 分页模式下 `PA_NZ` 与 `PA_BSND` 都可用，按 NZ 格式支持情况选；开 CP 的用 `PA_BSND` 或 `BSND`。不要一律改成 `PA_NZ` |
| 融合写入算子结果不一致 | `npu_kv_rmsnorm_rope_cache` 系列或 `npu_mla_prolog_v3` 参数不匹配 | 对齐 epsilon（用 `kv_a_layernorm.variance_epsilon` 而非硬编码）和 RoPE 频率 |

### FA 算子相关

| 问题 | 根因 | 修复方案 |
|------|------|---------|
| MLA absorb Decode 偏差大 | `q_len=1` 的 Decode 仍用 sparse_mode=3 + causal mask | 切 `sparse_mode=0` + `atten_mask=None`；`q_len>1` 的 MTP Decode 保持 3 + mask |
| 标准 LLM TND PA Decode 偏差 | Decode 误改成 `sparse_mode=0` + `mask=None` | TND PA 且 KV 未量化时 Decode 保持与 Prefill 一致 `sparse_mode=3` + causal mask；KV 量化或 legacy BSH 连续缓存的单 token Decode 用 0 + None 是正确写法 |
| 滑窗层 Decode 偏差 | Decode 误改成 `sparse_mode=0`，或 pre_tokens / next_tokens 与 sliding_window 不匹配 | Prefill+Decode 统一 `sparse_mode=4`，mask 用框架统一的 2048 causal，窗口靠 `pre_tokens=sliding_window` / `next_tokens=0` 表达 |
| actual_seq_lengths 错误 | 未按字段与阶段区分：qlen 恒累积，kvlen 形态随阶段与是否读历史 block 变；框架部署字段名 `actual_seq_lengths_cu_q`（累积形态）vs `actual_seq_lengths_kv`（原始形态） | 见「actual_seq_lengths 构造」，构造依据见 `model-infer-kvcache` 的同名小节 |
| Decode 首步输出为零 | actual_seq_qlen 为 0 | Decode 时应为 1 或 cumsum([1,...]) |
| MLA absorb 模式 key≠value | cache_nope 读写不一致 | FA 中 key 和 value 都传 cache_nope |
| scale 参数错误 | FA v1 用 `scale`，v2 用 `softmax_scale` | 确认参数名与版本匹配 |

### 量化相关

| 问题 | 根因 | 修复方案 |
|------|------|---------|
| W8A8C8 下长序列偏差显著 | KV 缓存量化带来的精度损失随序列累积 | 检查 KV 量化 scale；必要时该层改回 BF16 KV |

FA 侧的量化参数名 v1 与 v2 不同，见「FA v1 vs v2 关键参数名映射」的 KV 量化行。

### Prefill/Decode 一致性

| 问题 | 根因 | 修复方案 |
|------|------|---------|
| Prefill 不走 KVCache 写入 | 条件分支逻辑错误 | 确保 Prefill 也执行写入算子 |
| Decode 使用了与路径不符的 FA 参数 | 阶段分支判断或路径分类错误 | 阶段分支取自 `forward_metadata.is_prefill`；按路径分类配 sparse_mode / atten_mask（标准 LLM TND PA 走 PA 且非单 token 时不分阶段统一配置；MLA absorb 与单 token Decode 才显式分支） |
| 切换阶段时 kv_len 未更新 | Prefill 后未执行 kv_len += seq_len | 在 Prefill→Decode 切换时由 Runner 层（独立部署）或 ExecutionEngine（框架部署）更新 |
| MLA absorb 单 token Decode 误传 atten_mask | 切到 sparse_mode=0 后没把 mask 设 None | 同时设 `sparse_mode=0` + `atten_mask=None`；`q_len>1` 时不切 |

---

## FA 入参配置参考

### atten_mask 与 sparse_mode 联合校验

本表覆盖 FIA 系列（`npu_fused_infer_attention_score` 等）。DSA 稀疏注意力算子（`npu_sparse_flash_attention` / `npu_kv_quant_sparse_flash_attention`）用 `sparse_mode=3` 且不传 mask，因果几何由 topk 索引表达，不适用本表。

| sparse_mode | 含义 | atten_mask 要求 | 适用路径 | 常见错误 |
|:-----------:|------|----------------|---------|---------|
| 0 | defaultMask | 可选，通常 **None** | MLA absorb Decode（q_len=1）；KV 量化或 legacy BSH 连续缓存的单 token Decode | 误把 TND PA 且 KV 未量化的 Decode 切到此路径 |
| 1 | allMask | **必传** `(Q_S, KV_S)` | 特殊 | 忘记传 mask |
| 2 | leftUpCausal | 必传 | 不推荐；BSH legacy Prefill 有既有实现在用 | — |
| 3 | rightDownCausal | **必传** `(2048, 2048)`，允许前置广播维；上三角不含对角为 True 即遮蔽位 | **标准 LLM TND PA（Prefill+Decode 统一）**、MLA absorb Prefill 与 `q_len>1` 的 MTP Decode | mask 极性反了，下三角置 True；MLA absorb 单 token Decode 误用 |
| 4 | band（滑窗） | **必传** `(2048, 2048)` | 滑窗层 Prefill+Decode 统一 | pre_tokens/next_tokens 错误；FA v1 + BSH + Q_S=1 Decode 路径 op 内部忽略 |

### FA v1 vs v2 关键参数名映射

| 功能 | FA v1 | FA v2 | 易错点 |
|------|-------|-------|--------|
| 缩放系数 | `scale` | `softmax_scale` | 默认 1.0，传错名精度崩溃 |
| Q head 数 | `num_heads` | `num_query_heads` | 传错名默认 1 |
| Q 长度 | `actual_seq_lengths` | `actual_seq_qlen` | 名称完全不同 |
| KV 长度 | `actual_seq_lengths_kv` | `actual_seq_kvlen` | 名称完全不同 |
| KV 量化 | `antiquant_scale/offset` | `dequant_scale_key/value` | v2 分离 key/value |

### input_layout 与 tensor shape 匹配

| layout | Q 维度 | K/V 维度 | 适用场景 |
|--------|--------|---------|---------|
| **TND** | 3D `[T, N, D]` | 分页 cache，按 block_table 索引 | **框架部署 PA 标准 LLM 默认推荐**，T = packed tokens |
| **TND_NTD** | 3D `[T, N, D]` | MLA absorb 下是 NZ 分页 cache，非 3D | **框架部署 PA + MLA absorb Decode 默认推荐** |
| NTD_TND | 3D `[N, T, D]` | 3D `[T, N, D]` | MLA 非 absorb 的 Prefill；部分 GQA FP8 路径 |
| BSH | 3D `[B, S, N*D]` | 同 Q，或按 BSH 视图读的分页 cache | legacy 连续缓存；head_dim 过大需连续 K/V 时也用它读分页 cache |
| BNSD | 4D `[B, N, S, D]` | 4D `[B, N, S, D]` | 非 PA；部分 LLM 全注意力层 Prefill 也用 |
| BSND | 4D `[B, S, N, D]` | 4D `[B, S, N, D]` | 非 PA；仓内出现在多模态子模块与 blockwise sparse |

> PA 模式下 K/V 传的是分页 cache，形态不由 layout 名决定：GQA / MHA 的物理存储是 `[total_blocks, block_size, num_kv_heads, head_dim]`，FA 调用时 view 为 `[bn, bs, num_head*dim]`；MLA absorb 的 KV cache 是 NZ 格式的 5D 形态。按 layout 名核对 K/V 维度只在非分页路径成立。

### actual_seq_lengths 构造

| layout | 阶段 | actual_seq_qlen | actual_seq_kvlen |
|--------|------|----------------|-----------------|
| TND | Prefill，K/V 不读历史 block | 累积和 `cumsum([s1, s2, ...])` | 累积和，与 qlen 同值 |
| TND | Prefill，分页且读历史 block | 累积和 | **原始值**，长度与 qlen 无关 |
| TND | Decode | 累积和，单 token 时为 `[1, 2, ...]` | **原始值** `[kv1, kv2, ...]` |
| BSH / BNSD | 两阶段 | 原始值 | 原始值 |

框架部署提供三个字段：`forward_metadata.actual_seq_lengths_cu_q` 与 `actual_seq_lengths_cu_kv` 是累积形态，`actual_seq_lengths_kv` 是原始形态。按上表取——qlen 恒取 `actual_seq_lengths_cu_q`；kvlen 在不读历史 block 的 Prefill 取 `actual_seq_lengths_cu_kv`，此时它与 `cu_q` 同值，Decode 与读历史 block 的分页 Prefill 取 `actual_seq_lengths_kv`。字段名里没有 `cu` 不等于该阶段就该用它，构造逻辑见 `executor/core/engine/execution_engine.py` 的 `_build_model_inputs`。

chunked prefill 每轮只算本轮 token、K/V 读回已写入的 block，kvlen 因此与 qlen 的对应长度无关并取原始形态，写法见 `models/kimi_k3/models/modeling_kimi_k3.py` 的 `_prefill_chunk_attention`。

Decode 时每个序列的 q 长度通常为 1，MTP 场景为 `next_n+1`。

### inner_precise 与行无效修正

| 值 | 精度模式 | 行无效修正 | 适用场景 |
|:-:|------|:---:|---------|
| 0 | 高精度 | 否 | 默认 |
| 1 | 高性能 | 否 | 精度要求不高 |
| 2 | 高精度 | **是** | Prefill + 自定义 mask 有全遮蔽行 |
| 3 | 高性能 | **是** | 同上但允许精度损失 |

> 取值语义来自算子 API 文档。仓内 LLM attention 路径没有 2/3 的用例，只有视频生成模型与序列并行模块用默认 0，调整前按当前调用算子的 API 文档确认。

### MLA rope 约束

- `query_rope` 和 `key_rope` 必须同时传或同时不传
- rope D 必须为 64，query D 仅支持 512 或 128
- D=512 时仅支持 `sparse_mode` 为 0、3、4
