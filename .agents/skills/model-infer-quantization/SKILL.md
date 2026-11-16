---
name: model-infer-quantization
description: infer 仓模型量化适配改造技能。分析并接入既有 compressed-tensors 量化方案和权重，完成量化产物契约检查、结构参考匹配、量化 runtime 映射、权重加载、post-load 处理、融合算子量化冲突回退、真实生效验证和收益评估。用于量化初评估、实施前复核、改造与验证。
---

# 模型量化适配改造

在 infer 仓的离线推理路径中接入已交付的 `compressed-tensors` 方案和权重。改造限于 infer 侧的模型映射、加载、post-load 和融合回退，保留产物声明的量化范围和张量语义。

---

## 工作流程

```
1 确认输入和执行模式 → 2 检查量化产物契约 → 3 拆解模型结构与量化对象
→ 4 核对适配条件与评估接入分级 → 5 交付评估或复核结论
→ 6 实施量化接入（仅改造）→ 7 验证真实生效与收益 → 8 汇总结果与后验决策
```

## 第一步：确认输入和执行模式

以 `model_dir` 和 `quant_export_dir` 为输入，按任务模式（`mode`）分流。缺模型目录时可转 `model-infer-migrator`；缺量化产物时先由产物方交付。

| 模式 | 入口与边界 |
| --- | --- |
| 初评估 | 按第二至五步评估已交付产物与模型，使用已有证据估算显存和部署影响，不改代码，不新采基线。 |
| 实施前复核 | 以有效的初评估报告及此后的模型或量化产物变化为输入，只读复核受影响的结论，按第五步交付。报告缺失或已失效时转初评估。 |
| 改造 | 从第六步接手用户已确认的方案（`user_decision`），以非量化部署基线（`baseline`）和量化前最新基线为对照，采集量化基线。缺少改造决策时只做分析。 |
| 验证 | 从第七步验证已有改造，返回证据和阻塞点，不改代码。缺少非量化对照时先补基线再判断收益。 |

---

## 第二步：检查量化产物契约

按 `references/quantization-contract.md` §1–§3 核对导出目录中的配置、权重、索引（如有）与部署说明。关键字段缺失、张量语义不清或权重不完整时，列出受影响的模块、缺失项、预期 shape/dtype/scale 及阻塞原因，交付补充诉求。

---

## 第三步：拆解模型结构与量化对象

**读取**：`{model_dir}/agentic/progress.md`、`{model_dir}/config/*.yaml`、`{model_dir}/runner_*.py`（如有）、`{model_dir}/models/modeling_*.py` + `references/quantization-contract.md` §4-§5。

### 3.1 映射量化对象

按模型实际实现，将产物声明的量化目标逐项映射到 infer 运行对象，注明影响接入的结构特征。

### 3.2 记录运行场景

Prefill/Decode 分支差异、部署卡数 + TP/EP + 是否 online split、量化模式（`W8A8` / `W8A8C8` / `W4A8C8` 等）、量化前是否已有融合/前序优化、可能影响量化的 dtype/layout/scale/cache 组织。

---

## 第四步：核对适配条件与评估接入分级

**读取**：`references/quantization-structure-cards.md` 中与当前模块相关的章节 + `references/quantization-fusion-and-benefit.md` §A（融合兼容性）；初评估显存与部署收益时见该文 §B。

### 4.1 按模型结构核对适配条件

按 Dense / MoE / MLA / Indexer 的实际算子链核对 forward、scale 路径和 post-load 条件。需要参考实现时，可查询 cannbot-knowledge 中同部署框架、同结构与量化模式的接入样例和融合算子契约。

### 4.2 融合算子兼容性判断

按 §A.1 对照目标算子的输入输出契约，判断能否保留现有融合、是否需要替身或专属适配。列出需要确认的回退项，实施条件见 §6.5。

### 4.3 接入分级（按模块/能力逐项打级，不压成单维）

按 §3.1 的映射对各目标模块分别分级。本次方案所需模块出现 L2 或 L3 时，逐项列出运行时能力缺口或产物补充诉求，方案暂不进入改造。

| 级别 | 判定 | 所需改造或阻塞项 |
| --- | --- | --- |
| L0 | 只需配置或 YAML 对齐 | 配置对齐后可验证 |
| L1 | 需要模型映射、runner 或 post-load | 需调整模型映射或加载逻辑 |
| L2 | 缺 infer 量化 runtime object / 算子能力（compressed-tensors 无对应 method/kernel） | 停止当前适配，标 runtime gap，向主流程/用户确认是否另起 runtime 任务；不在本次适配内默默补 kernel |
| L3 | 量化契约缺失，无法安全接入 | 向产物方反馈缺失项，补齐后再评估 |

---

## 第五步：交付评估或复核结论

- **初评估**：按 `references/quantization-templates.md` 的「量化方案初评估」交付映射、逐模块分级、部署估算和融合风险，给出是否保留量化候选的建议。
- **实施前复核**：对照初评估报告，检查受影响的 targets/ignore、张量语义、prefix、运行对象、权重分片、dtype/layout/cache、post-load 或融合路径。无影响时沿用原结论；有影响时仅输出变更的映射、阻塞点或待确认的回退选项。

---

## 第六步：实施量化接入

新模型采用 infer 仓框架的加载和 post-load 入口；已有模型沿用实际加载路径，见 `references/quantization-contract.md` §5。

### 6.1 接入 `quant_config`

1. 在真实 model loading 入口把 `config.json` 的量化配置解析成 `CompressedTensorsConfig`（实现可能是 `get_quant_config(...)` / `CompressedTensorsConfig.from_config(...)` / loader 内部 helper，不假定唯一入口）。
2. 挂到模型 config（框架经 `get_quant_config` 挂到 `hf_config.quant_config`）并透传到各可量化模块；保持 prefix 稳定使 `targets` / `ignore` 命中。
3. 输出头（如 `lm_head`）按产物声明及实际导出的张量确定量化或浮点路径。命中 target 却缺少 scale 时按 §6.6 核对。

### 6.2 替换或映射 runtime object

- Dense Linear：接入 `Linear` / `ReplicatedLinear` / `RowParallelLinear` 的量化方法。
- MoE experts：优先映射到 `MoEGMM` / `FusedMoEGMM`，核对 gate/up 与 down 投影实际匹配的 target 和量化方案。
- KVCache：按 `kv_cache_scheme` 和 cache scale 接入。`ignore` 模块显式浮点回退，不计入收益。

### 6.3 处理权重加载和 post-load

核对：量化张量名↔参数前缀、TP/EP shard 与 online split、weight 转置、NZ/base format、scale dtype、smooth scale 完整性、MoE expert pack/unpack、cache scale 来源。根据消费算子确认哪些模块跳过后处理，以及拼接、转置与格式转换的顺序；遇到特殊 pack/shard 或 scale 布局时，可查询 cannbot-knowledge 中对应的加载样例。

### 6.4 恢复量化后运行时融合

Dense 优先检查量化产物是否拆散原 fused `gate/up` 或 `q/k/v`、能否在 post-load 后恢复更大的量化 matmul（减少小粒度 matmul 和动态量化开销）。不改变量化方案语义。

### 6.5 处理融合算子冲突

若融合算子不接受产物的 dtype/layout/scale，按已确认的回退原则切回该模块的非融合路径，继续加载原产物并验证。超出确认范围时暂停改造，返回待决策项。记录失败的算子、输入契约、错误、回退点及后续融合量化需求。

### 6.6 ignore 不完备的兼容处理

> 触发：产物里既不在 `ignore` 也无 `weight_scale` 的 Linear 候选模块（contract §3 *ignore 完备性启发式*）。

- 若 `deploy_quantization.md` 明确该模块保留浮点，确认实际构造了浮点运行对象且加载了浮点权重。必要时在 infer 侧补 effective ignore，保留产物 `config.json`，列出隐式浮点前缀、范围依据和补充诉求。
- 范围模糊、有冲突，或应量化却缺少 scale 时，停止改造并输出补充诉求。跳过加载告警不能证明浮点路径已生效。

---

## 第七步：验证真实生效与收益

收益对照与归因见 `references/quantization-fusion-and-benefit.md` §B。

### 7.1 功能验证 + 等价性自检

- 功能：量化模型可加载，Prefill/Decode 至少各跑通一次，核对相对基线新增的重复、异常空输出或提前结束，日志能证明走到量化 runtime；验证命令记录模型路径、产物路径、卡数、YAML。
- **等价性自检（接线正确性，非精度评测）**：绝对精度由产物方在 `deploy_quantization.md` 报告，本 skill 不评测精度。固定一组 prompt，量化基线与 infer BF16 基线各跑 greedy，文本 diff 记首个分歧 token；W8A8 允许细微差异，重点是不出现乱码/早停/与 BF16 显著走偏（抓 scale/layout 接错的「能跑但输出垃圾」）。连 BF16 参照都跑不了：标「等价性未核，转产物方精度报告」，不静默判通过。

### 7.2 生效验证（probe 优先，贴输出，不空打勾）

- **probe-B 对象级（主，零开销）**：加载后对照产物声明，抽查量化 Linear 的 `layer.scheme` 和量化 FusedMoEGMM 的 `layer.quant_method`。
- **probe-C 权重级**：按目标模块前缀，核对产物中的权重及 scale 是否实际加载到对应参数。
- **probe-A 算子级（复用 §7.3 profiling，不单独跑）**：已有 profiling 时，用 `op_statistic` 定位候选算子；浮点与量化 MoE 都会调用 `GroupedMatmul`，需结合目标层实际执行分支和张量 dtype 判定。

并记录：已量化模块 / `ignore` 浮点回退 / 融合算子回退。

### 7.3 性能和显存验证

显存 / Prefill / Decode 三项均对比「非量化基线、量化前最新基线、量化基线」，部署卡数对比「原形态、量化后形态」。性能数据多轮取中位数 + 充分 warmup，避免 single-sample。初评估判断量化后单卡可能满足时，优先给单卡验证结果。

### 7.4 返回验证问题

- 映射/加载/post-load/scale dtype/runtime object 错误：标出 infer 侧出错模块及证据。
- 融合算子冲突：报告失败的输入契约和回退建议，由改造任务按 §6.5 处理。
- 契约缺失或权重不完整：输出产物补充诉求。
- 功能生效但性能未升：保留证据，分析动态量化开销/算子粒度/部署配置/profile；需要对照案例时，可查询 cannbot-knowledge 中同执行模式、并行形态和负载下的量化 profiling 记录。

---

## 第八步：汇总结果与后验决策

### 8.1 量化结果字段

见 `references/quantization-templates.md` 的「量化结果字段」。

### 8.2 后验决策依据

量化基线产出后，根据实测收益和回退代价向用户给出建议：采用当前量化方案并记录融合回退需求、保留融合并跳过量化，或由产物方修正量化方案或补齐契约后迭代验证。最终取舍由用户确认。

---

## 参考文档索引

> 按需查阅，避免一次性加载消耗 token。

| 主题 | 路径 |
| --- | --- |
| 产物契约 / 张量语义 / 运行对象映射 / infer 仓框架接入 | `references/quantization-contract.md` |
| Dense / MoE / MLA 与 Indexer 的结构与 scale 判断 | `references/quantization-structure-cards.md` |
| 融合兼容性 / 量化收益分账与证据 | `references/quantization-fusion-and-benefit.md` |
| 初评估输出模板 / 量化结果字段 | `references/quantization-templates.md` |
