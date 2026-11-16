# 典型结构量化改造卡

运行对象映射见 [量化产物接入契约 §4](quantization-contract.md)。以下按结构说明影响量化路径选择的条件。

## 结构卡 A：Dense Decoder

### 适用结构

- Decoder-only Transformer。
- Attention 是 MHA/GQA/MQA，核心投影为 `q_proj/k_proj/v_proj/o_proj` 或 fused QKV。
- FFN 是 dense MLP，核心投影为 `gate_proj/up_proj/down_proj` 或 fused gate-up。
- 无 routed experts。

### 映射

| 结构对象 | infer 对象 |
|----------|------------|
| Attention projections | `Linear` / `ReplicatedLinear` |
| MLP projections | `Linear` / `ReplicatedLinear` |
| fused QKV / gate-up | 只有所有 shard 量化策略一致时才可融合接入 |
| `ignore` 模块 | `UnquantizedLinearMethod` |

### 改造要点

- 构造可量化投影层时传入 `quant_config` 和能匹配产物 `targets`（含正则）的稳定 prefix。
- `weight_scale` 通常按 output dim 加载；`smooth_scales` 用于动态 activation quant。
- post-load 重点检查 transpose、NZ format、scale dtype。
- 合并拆开的 Q/K/V 或 gate/up 前，核对各分片的 shape、量化策略和权重 dtype。若共用一次输入动态量化，各投影的 smooth scale 必须相同。合并权重及对应的 weight scale 后再转目标布局。
- W8A8 下若 gate/up 输出 int32 累加值和 per-token scale，核对激活是否完成反量化与再量化，并将新的 scale 传给 down 投影。仅替换 Linear 无法保证这段 forward 接线正确。

## 结构卡 B：MoE Decoder

### 适用结构

- Decoder-only Transformer + routed experts。
- router/topk 通常保持浮点。
- experts 可能是 per-expert Linear，也可能是 packed expert 权重。
- 有 TP/EP 切分时，scale / smooth scale 可能需要按 rank 对齐。

### 映射

| 结构对象 | infer 对象 |
|----------|------------|
| shared attention / dense MLP | `Linear`（按结构卡 A 处理） |
| routed experts gate/up/down | 优先 `MoEGMM`；per-expert Linear 的适用条件见契约 §4 |
| router/topk | 通常浮点，不作为量化 target |
| expert smooth scale | 按 EP/TP 规则加载或聚合 |

### 改造要点

- FusedMoEGMM 的 gate/up 与 down 共用方案时，优先匹配 `MoEGMM`，仅声明 `Linear` 时复用其方案。若声明 `MoEGMMUpGate`，需配套 `MoEGMMDown`，并确认 runtime 支持两组方案的组合。
- W8A8 MoE 分别核对 expert 的 gate/up、down 投影权重及对应的 weight scale 和 smooth scale。
- W4A16 / W4A8 关注 int32 pack/unpack、group_size、offset、scale layout。
- packed expert checkpoint 的权重和对应 scale 必须按同一专家分片映射加载。
- A8 expert 有两种常见输入路径：量化后分发时，scale 随 token 路由和重排；BF16 分发后在本 rank 动态量化时，由本地生成 per-token scale。全局 smooth scale 驱动路由时，若本 rank 只加载了本地 expert 的 scale，需先补齐消费算子所需的专家范围。
- routing 的 token 分组编码（如累计数或计数）须与 GMM 的 group list 一致。含 zero/copy expert 时，还要核对实际参与分发的专家范围、top-k 权重和 finalize 规则。
- shared expert 若采用量化 gate/up 与 down 投影，按结构卡 A 核对激活和 per-token scale 的传递。

## 结构卡 C：MLA + MoE / KVCache

### 适用结构

- Attention 使用 MLA、latent KV、LoRA rank projection 或 KV cache 特殊 layout。
- 通常同时伴随 MoE。
- 量化对象除 Linear/MoEGMM 外，可能包含 `kv_cache_scheme` 或 k/v output scale。
- 子场景：长序列 + Indexer / LightningIndexer / Sparse Flash Attention。

### 映射

| 结构对象 | infer 对象 |
|----------|------------|
| MLA q/k/v/latent projections | `Linear` |
| routed experts | `MoEGMM` 或 per-expert Linear（按结构卡 B 处理） |
| KVCache quant | `kv_cache_scheme` / `get_cache_scale()` |
| MLA fused prolog / sparse attention | 按输入、输出和 cache 契约评估融合兼容性 |
| Indexer / LightningIndexer / LI cache | 独立量化闭环 |

### 改造要点

- 先拆清 prefill/decode 的 MLA 路径，不要只看类名。
- KVCache 量化必须有明确 scale 来源；`.k_proj.output_scale` / `.v_proj.output_scale` 可映射到 k/v cache scale。
- MLA projection 的 post-load 要核对 cache layout、rope/nope split、rank 切分。
- MoE 部分按结构卡 B 处理。
- C8 路径要核对 prolog 写入的 cache 布局、query/cache scale 和 FA 实际消费的量化数据。涉及 CP/offload 时，cache 与对应 scale 必须同步重排。
- Indexer 单独跟踪 q/k 量化、key cache 与 scale 的更新及 Top-K 算子的输入，分别核对 Prefill 和 Decode。普通 KVCache 已走 C8 不能证明 Indexer 量化生效。
