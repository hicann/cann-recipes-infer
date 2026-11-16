---
name: model-infer-fusion
description: 基于 PyTorch 框架的昇腾 NPU 模型推理融合算子优化技能。分析模型代码，识别可替换为 torch_npu 融合算子的计算模式，生成替换方案。触发场景：torch_npu 融合算子替换、MoE/Attention/FFN/Norm 等模块的推理算子适配、torch_npu API 使用咨询。基于仓库已有模型的融合算子经验，按计算语义推荐最佳方案。
---

# torch_npu 融合算子优化技能

分析 PyTorch 模型代码中的计算模式，匹配 torch_npu 融合算子进行替换优化。基于 torch_npu 本地 docstring 查询脚本（`scripts/torch_npu_query.py`）和仓库已有模型的融合算子使用经验，按模块逐一分析、匹配、替换、验证。

---

## 工作流程

### 第一步：分析模型代码，拆解模块

按本次任务范围拆解目标计算链路及其依赖，后续候选匹配与覆盖检查沿用这一范围。

- **Attention**：Norm、QKV 投影、RoPE、KV Cache 读写、Attention Core（FA 或稀疏 FA）、O 投影。MLA absorb 路径还包含 V 反投影。
- **Dense FFN**：输入投影、激活或门控乘法、输出投影。
- **MoE**：Gate / TopK、Routing Init / Dispatch、专家计算、Finalize / Combine，以及模型已有的 Shared Expert。结合并行配置识别 MC2 / AllToAll 通信。
- **其他链路**：Embedding、LM Head、跨层残差与 Norm 等。

**记录运行场景特征**：
- Prefill / Decode 分支差异：同一模块在两个阶段可能走不同算子路径，替换时分别处理
- 量化需求：BF16 / W8A8 / W8A8C8 / W4A16
- 分布式配置：TP / EP / DP，影响 MoE 路由算子选择
- 固定约束：layout（NZ/BSND/TND）、cache 格式、metadata
- 中间输出消费关系：候选链路内的中间张量是否被融合范围外模块、hook、graph 输出、debug 路径或多分支复用

### 第二步：按模块独立匹配仓库参考实现

按模块匹配计算语义最接近的仓库实现，标出参考链路覆盖的部分。对未命中的模块或子链路，转到第三步继续查找候选算子。需要参考实现时，可查询 cannbot-knowledge 中同部署框架、相同 Attention 或 MoE 形态的融合样例。

#### Attention 层

**注意**：若 FA 融合算子已在前置阶段替换，仅跳过 FA 调用本身。Attention 的其他子链路（RoPE、KV Cache 写入、Residual+Norm、QKV/O Projection 等）仍需逐一检查是否有可用融合算子。同时注意多步骤整体融合算子（如 `npu_kv_rmsnorm_rope_cache` 融合 RMSNorm+RoPE+Cache写入，`npu_mla_prolog_v3` 融合 Q/KV投影+RMSNorm+RoPE+Cache写入），不要将单个子模块简单归为"标准线性层无融合算子"而跳过评估。

分别沿 Prefill 和 Decode 核对 FA 的 K/V 来源（当前输入或 cache）、cache 写入时机与索引（连续位置或分页 slot），以及因果 / 滑窗掩码和有效序列长度。按实际计算链路选择候选：

- **GQA / MHA**：从 QKV 投影追踪到 RoPE 和 FA，保留模型已有的 Q/K Head Norm。
- **MLA**：区分 FA 前展开 K/V 的路径与 absorb 路径。后者将 K 的投影吸收到 Q 侧，FA 使用压缩 KV 和独立的 RoPE 分量，之后再做 V 反投影。
- **MLA + Indexer**：评估 prolog 时保留 Indexer 消费的中间输出，稀疏 FA 还依赖 Indexer 产生的 Top-K 索引。

#### MoE / FFN 层

先确认当前模块属于 Dense FFN 还是 MoE；若为 MoE，再结合 gate 形式、并行模式和阶段判断更接近哪条参考链路。

```
是否有 MoE？
  │
  ├─ 无（Dense FFN）→ 查找 Dense Linear / Activation / Norm 可用融合算子
  │
  └─ 有 MoE
       │
       ├─ Gate 算子
       │    ├─ softmax 打分 → npu_moe_gating_top_k_softmax
       │    └─ sigmoid/noaux → npu_moe_gating_top_k
       │
       └─ 路由 + 专家计算（按并行模式和阶段区分）
            ├─ Prefill（纯 TP）: init_routing_v2 → grouped_matmul → finalize_routing
            ├─ Prefill（EP）:    init_routing_v2 → AllToAll → re_routing → grouped_matmul → finalize_routing
            └─ Decode（EP dispatch/combine）: MC2 dispatch_v2 → grouped_matmul → MC2 combine_v2
```

MoE 算子适配与 EP / TP 并行、通信组、权重切分、routing 输出格式和 dispatch / combine 强耦合。本 skill 只识别 MoE 子链路和候选 `torch_npu` API。评估 A2 常规 MC2 路径时核对每 rank 的 MoE expert 数，上限、回退方案及实施细节见[并行实施参考](../model-infer-parallel-impl/references/framework_moe_parallel.md)。

#### 未匹配模块

对参考实现未覆盖的模块（如 Embedding、LM Head 或 Diffusion 特有模块），可从以下模式查找候选：

- Residual + Norm：检查 `add + rms_norm`、独立 `rms_norm` 等已有 Norm 类融合算子
- Dense / Gated FFN：检查 `Linear + Activation/SwiGLU + Linear`、`npu_ffn` 或 activation 类融合算子
- Linear + Activation：检查投影后紧随激活的短链路
- Norm / Projection / RoPE / Cache 前处理：若输出消费关系单一，检查是否能并入已有 prolog、rope-cache 或其他大范围融合路径
- 量化链路（可选）：仅在当前模型已有量化方案或任务明确要求量化时，检查量化 FFN、activation / quant、量化 / 反量化等融合算子

---

### 第三步：查阅算子接口文档，确认可用性与适配性

优先通过 `scripts/torch_npu_query.py` 查询目标版本的本地 docstring。无 torch_npu 环境时，可用 `--docs-path` 指定目标版本的 `_op_plugin_docs.py`。

```bash
# 单 API 详情
python3 scripts/torch_npu_query.py show <api_name>

# 反向搜索
python3 scripts/torch_npu_query.py search "<keyword>"

# 列出已收录的算子名
python3 scripts/torch_npu_query.py list [--prefix npu_]
```

按计算语义查找候选时，可查询 cannbot-knowledge 的 `torch_npu` 官方文档索引。接口约束以目标版本的本地文档或 op-plugin 官方详情为准。

对每个候选核对接口签名、dtype/shape/layout、cache 组织及动态 shape、控制流和 metadata 的支持情况，以及适用的量化、图模式、硬件和并行约束。若格式转换、RoPE 输入整理或 KV Cache 改造可满足要求，列为“需前置改造”并说明依赖。判定“不适配”需有无法通过合理改造满足的具体约束。

---

### 第四步：输出分析结果

按模块列出原计算链路、候选 API 及覆盖范围，说明适配结论、依据（目标版本接口文档，命中样例时附参考代码位置）、所需前置改造和最小验证方法。对任务范围内未覆盖或跳过的链路，说明具体原因及证据。仅分析的任务在此交付。

若现有 API 无法覆盖且有明确收益依据，将该链路作为新增算子需求移交后续开发，附上代码位置、子图语义与期望融合范围、输入输出约束（含 dtype/layout、相关 cache 与量化信息）及不适配证据。

---

### 第五步：逐模块实施替换

按本次任务选定的链路实施。已有确认方案时，直接从该方案接手。

若候选模块依赖以下前置改造且尚未完成，可按需查阅对应资源：

- `RoPE` 预计算、缓存和取值路径整理：[`references/rotary-embedding-pattern.md`](references/rotary-embedding-pattern.md)
- `KV Cache` 静态化 / `PA` 改造：参考 `model-infer-kvcache` skill

完成必要的前置改造后，以可独立验证的链路为单位替换算子，对比融合前后的输出和性能。精度对齐且有性能收益时保留，验证失败时回退该链路的改动，再评估替代方案。

记录每条选定链路的保留或回退结论、精度与性能对比及日志位置。受阻时注明原因和已尝试的方案。若实施中才发现现有 API 的硬限制，补入分析结论，符合第四步条件时移交新增算子需求。

---

## 参考文档索引

> 以下文档按需查阅，避免一次性加载消耗 token

| 主题 | 路径 |
|------|------|
| **torch_npu API 文档查询脚本** | [`scripts/torch_npu_query.py`](scripts/torch_npu_query.py) |
| MoE 并行与融合算子链路 | [`../model-infer-parallel-impl/references/framework_moe_parallel.md`](../model-infer-parallel-impl/references/framework_moe_parallel.md) |
| RotaryEmbedding 预计算与调用模式 | [`references/rotary-embedding-pattern.md`](references/rotary-embedding-pattern.md) |
| op-plugin 仓库（在线） | [gitcode.com/Ascend/op-plugin](https://gitcode.com/Ascend/op-plugin) |
