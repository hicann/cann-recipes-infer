---
name: model-infer-quantization
description: infer 仓模型量化适配改造技能。分析并接入既有 compressed-tensors 量化方案和权重，完成量化产物契约检查、结构参考匹配、量化 runtime 映射、权重加载、post-load 处理、融合算子量化冲突回退、真实生效验证和收益评估。用于模型优化流程中的量化初评估和量化改造任务；不重新设计上游量化算法，不实现 compressed-tensors 之外的量化路线。
---

# 模型量化适配改造

分析 infer 模型代码与既有 `compressed-tensors` 量化产物，按模型结构匹配仓库参考经验，完成量化接入、验证和收益评估。初评估模式只输出方案不改代码；改造模式在契约和用户决策满足后实施。

---

## 硬规则（TLDR）

> 本节是「该怎么做」的执行边界（范围 / 许可 / 流程 / 停止条件）；纯「别做啥」的禁止类统一见文末「反模式」。

1. 范围：只做 `compressed-tensors` 主线；不碰 GPTQ / AWQ / 外部 patch / service 部署，也不碰 `module/quantization/` 下 `fp8` / `mxfp4` / `mxfp8` 等仓内非 compressed-tensors route。
2. 许可：可改 infer 侧 loader、模块映射、post-load、runner 和融合回退代码。
3. 改造模式必须有用户继续量化的明确决策；缺决策只做分析（§1.2）。
4. 融合算子不支持量化时，在主流程/用户确认后回退非融合路径继续跑量化基线，不因冲突放弃量化（§6.5）。
5. 停止分两类：契约字段缺失 / 张量语义不清 / 权重不完整 → 停止并输出产物契约补充诉求（L3）；缺 infer 量化 runtime 能力 → 停止并确认是否另起 runtime 任务（L2，见 §4.3）。

---

## 工作流程

```
1 确认输入和执行模式 → 2 检查量化产物契约 → 3 拆解模型结构与量化对象
→ 4 匹配仓库参考与评估接入分级 → 5 方案审查
→ 6 实施量化接入（仅改造）→ 7 验证真实生效与收益 → 8 记录、后验决策与经验沉淀
```

每步「**读取**」标注该步要看的 reference；references 按需加载，不一次性全读。回退规则、验证要求、后验决策三段为本 SKILL 唯一真相源（§6.5 / §7 / §8.2），ref 不复述。

---

## 第一步：确认输入和执行模式

### 1.1 输入

| 输入 | 适用模式 | 说明 |
| --- | --- | --- |
| `model_dir` | 初评估 / 改造 / 验证 | infer 模型工作目录 |
| `quant_export_dir` | 初评估 / 改造 / 验证 | 量化权重和量化方案目录 |
| `mode` | 初评估 / 改造 / 验证 | 执行模式 |
| `baseline` | 改造 / 验证 | 非量化部署基线；改造还应参考量化前最新基线 |
| `user_decision` | 改造 | 用户明确继续量化的决策 |

### 1.2 模式分流

| 模式 | 允许动作 | 禁止动作 |
| --- | --- | --- |
| `初评估` | 分析量化产物、结构、显存、接入分级，写初评估报告 | 改模型代码 |
| `改造` | 按已确认方案接入 runtime、加载权重、post-load、跑量化基线 | 改量化方案或权重语义 |
| `验证` | 验证真实生效、输出可用性、性能和显存收益 | 自行调试或改代码 |

### 1.3 输入缺失

- 缺 `model_dir`：转 `model-infer-migrator` 或要求补模型目录。
- 缺 `quant_export_dir`：停止量化路径，要求补齐产物目录。
- 改造/验证缺非量化基线：先补部署基线，不进入收益验证。
- 改造缺用户决策：只做分析，不改代码。

---

## 第二步：检查量化产物契约

**读取**：`{quant_export_dir}/config.json`、`model.safetensors.index.json`、`deploy_quantization.md` + `references/quantization-contract.md` §1-§4。

### 2.1 契约检查项

- `config.json` 有 `quantization_config`（或 `compression_config`），`quant_method` = `compressed-tensors`。
- `config_groups` 能描述权重/激活量化；`targets` / `ignore` 能区分量化模块与浮点回退。
- `model.safetensors.index.json` 可索引量化与保留张量；张量语义明确（`qweight` / `weight_scale` / smooth scale / zero point 等）且可映射到 infer 参数前缀。
- `deploy_quantization.md` 写清量化对象、模式、回退策略、下游加载要求。

### 2.2 契约不满足时

任一关键项缺失：不猜测、不硬编码、不改方案，输出补充诉求（缺失项 / 影响模块和 runtime object / 期望格式 shape·dtype·scale / 阻塞原因），不进入改造。

---

## 第三步：拆解模型结构与量化对象

**读取**：`{model_dir}/agentic/progress.md`、`{model_dir}/config/*.yaml`、`{model_dir}/runner_*.py`（部分模型有，如 deepseek/glm/longcat-flash）、`{model_dir}/models/modeling_*.py` + `references/quantization-contract.md` §4-§6。

### 3.1 识别结构指纹（按真实结构，不按模型名）

| 结构指纹 | 判断依据 |
| --- | --- |
| `Dense Decoder` | decoder-only，无 routed experts，有 attention projection 和 dense MLP |
| `MoE Decoder` | 有 router/topk、routed experts、expert gate/up/down |
| `MLA + MoE / KVCache` | 有 latent KV、MLA prolog、KV cache scale 或 MLA absorb |
| `Indexer / 长序列 + MoE` | 有 Indexer、LI cache、Sparse FA、Hadamard 或长序列专属契约 |
| 特殊 packed expert | 专家权重或 scale 有特殊 pack/shard 规则 |

### 3.2 建立量化对象映射

| 量化目标 | infer runtime object |
| --- | --- |
| Dense Linear | `Linear` / `ReplicatedLinear` / `RowParallelLinear` |
| Dense MLP gate/up/down | 可量化 Linear；必要时 post-load 恢复 `gate/up` 融合 |
| Attention q/k/v/o | 可量化 Linear；必要时 post-load 恢复 `q/k/v` 融合 |
| MoE experts | `MoEGMM` / `FusedMoEGMM`，不满足时回退 per-expert Linear |
| KVCache | `kv_cache_scheme` / cache scale / cache runtime |
| `ignore` 模块 | 浮点回退，不宣称已量化 |

### 3.3 记录运行场景

Prefill/Decode 分支差异、部署卡数 + TP/EP + 是否 online split、量化模式（`W8A8` / `W8A8C8` / `W4A8C8` 等）、量化前是否已有融合/前序优化、可能影响量化的 dtype/layout/scale/cache 组织。

---

## 第四步：匹配仓库参考与评估接入分级

**读取**：`references/quantization-structure-cards.md` 结构卡 A/B/C + `references/quantization-fusion-and-benefit.md` §A（融合兼容性）、§B.3（主线路径 A1-A4）。

### 4.1 按结构匹配参考卡

匹配结构卡（Dense / MoE / MLA+MoE / Indexer），各卡重点见卡内「改造要点」。**结构卡命中 ≠ 经验全套用**：forward 替身、scale 路径、post-load 例外必须基于当前模型实际算子链独立核对。只有结构无法落入已有卡，或出现新 runtime object / 张量语义 / post-load 规则 / 融合回退模式时，才新增卡。

### 4.2 融合算子兼容性判断

按 fusion-and-benefit §A.1 分级（A 量化主线-保留接量化输入 / B 共存-契约满足时保留 / C 需替身-先 post-load 融合或换量化友好算子 / D 专属契约-先看输出 dtype·cache layout·side tensor）+ §A.2 算子映射判断。冲突处理由主流程改造前向用户确认，回退规则见 §6.5。

### 4.3 接入分级（按模块/能力逐项打级，不压成单维）

按 §3.2 映射逐项打级（Linear / MoEGMM / KVCache / Indexer 独立给级），评估卡逐项写入；整体动作按最坏级触发，但保留逐项分级以记录「哪些就绪、哪些卡住」。

| 级别 | 判定 | 动作 |
| --- | --- | --- |
| L0 | 只需配置或 YAML 对齐 | 补配置，直接验证 |
| L1 | 需要模型映射、runner 或 post-load | 改模型/runner/加载逻辑 |
| L2 | 缺 infer 量化 runtime object / 算子能力（compressed-tensors 无对应 method/kernel） | 停止当前适配，标 runtime gap，向主流程/用户确认是否另起 runtime 任务；不在本次适配内默默补 kernel |
| L3 | 量化契约缺失，无法安全接入 | 停止落代码，输出补充诉求 |

---

## 第五步：方案审查

进入代码改造前审查；未完成则回对应步骤补齐。此为改造前的统一 gate。

- [ ] 契约 Gate 通过，或已停止并输出补充诉求
- [ ] 结构指纹不是按模型名推断
- [ ] `quant_target -> infer runtime object` 映射完整
- [ ] `targets` / `ignore` 与模型参数前缀已核对
- [ ] 量化张量名、shape、dtype、scale 语义已核对
- [ ] post-load 处理项已列出
- [ ] 融合算子冲突和回退策略已列出
- [ ] 非量化基线和量化前最新基线已确认
- [ ] 改造模式下已有用户继续量化决策

初评估任务到此结束：按 `references/quantization-templates.md` 的「量化方案初评估」模板输出报告（字段对应 progress_template「阶段 0.5」超集），不进入实施。

---

## 第六步：实施量化接入

> 仅改造任务执行。每次围绕「一个模块映射或一个 post-load 问题」推进，验证通过再继续下一个风险点。

**读取**：命中结构卡的「改造要点 / 已确认模型经验 / post-load 例外」。

**接入形态**：新模型走框架（`model_worker` 调 `module/quantization` 的 `get_quant_config` 读配置 + `_process_weights_after_loading` 触发 modeling 的 `process_weights_after_loading`，无 per-model runner）；少数早期模型用 `runner_*.py` 自己读 config 调 post-load（详见 contract §5）。

### 6.1 接入 `quant_config`

1. 在真实 model loading 入口把 `config.json` 的量化配置解析成 `CompressedTensorsConfig`（实现可能是 `get_quant_config(...)` / `CompressedTensorsConfig.from_config(...)` / loader 内部 helper，不假定唯一入口）。
2. 挂到模型 config（框架经 `get_quant_config` 挂到 `hf_config.quant_config`）并透传到各可量化模块；保持 prefix 稳定使 `targets` / `ignore` 命中。
3. 输出头（如 `lm_head`）即使命中 `targets:["Linear"]` 又不在 `ignore`，构造时不传 `quant_config` 即可（输出头一般保持浮点：词表维度大、收益有限、影响采样分布，产物方通常也不导出其量化张量）。

### 6.2 替换或映射 runtime object

- Dense Linear：接入 `Linear` / `ReplicatedLinear` / `RowParallelLinear` 的量化方法。
- MoE experts：优先 `MoEGMM` / `FusedMoEGMM`；统一 `W8A8` 检查 `gmm_quant_mode` 是否继承 `mm_quant_mode`；混合位宽必须有独立 `targets=["MoEGMM"]` group。
- KVCache：按 `kv_cache_scheme` 和 cache scale 接入。`ignore` 模块显式浮点回退，不计入收益。

### 6.3 处理权重加载和 post-load

核对：量化张量名↔参数前缀、TP/EP shard 与 online split、weight 转置、NZ/base format、scale dtype、smooth scale 完整性、MoE expert pack/unpack、cache scale 来源（细则见结构卡「post-load 例外」）。

### 6.4 恢复量化后运行时融合

Dense 优先检查量化产物是否拆散原 fused `gate/up` 或 `q/k/v`、能否在 post-load 后恢复更大的量化 matmul（减少小粒度 matmul 和动态量化开销）。不改变量化方案语义。

### 6.5 处理融合算子冲突（唯一回退规则真相源）

> 本节是融合算子量化冲突的唯一回退规则真相源；其它 reference 不复述。

融合算子拒绝量化 dtype/layout/scale 时：① 不改量化方案 / `targets` / `ignore` / 张量语义；② 检查主流程是否已传入用户确认的回退原则，没有则停止改造并输出待决策项；③ 用户已接受则把该模块回退到原非融合路径；④ 继续按原量化产物加载验证；⑤ 记录融合算子名、失败输入契约、错误、回退点、后续融合量化需求。

### 6.6 ignore 不完备的兼容处理

> 触发：产物里既不在 `ignore` 也无 `weight_scale` 的 Linear 候选模块（contract §3 *ignore 完备性启发式*）。

- **前置 0 步**：模型代码若已通过常规机制（`_keys_to_ignore_on_load_unexpected` / `load_state_dict(strict=False)` / 不传 `quant_config`）让缺 scale 模块不参与量化加载，即等价合规；仍记补充诉求让产物方下版补 `ignore`。
- 读 `deploy_quantization.md` 量化范围声明，三分支：**范围明确不覆盖该模块** → 视为 implicit BF16 + ignore 漏列，走隐式回退（infer 侧把前缀注入 effective ignore set，扩展 `CompressedTensorsConfig.ignore` 走 `UnquantizedLinearMethod`；**不改产物 `config.json`**；在 `agentic/progress.md` 记 implicit BF16 前缀 + 范围依据 + 补充诉求）；**范围模糊或与 ignore 不一致** / **应量化但 scale 缺失** → 停止改造，输出对应补充诉求。

**反模式**：造假 scale 强行量化；int8 forward without scale；改产物 `config.json`；替产物方决策「漏 ignore 还是漏 scale」。

### 改造完成（进入验证前自检）

- [ ] `quant_config` 透传 + 模块映射完成
- [ ] 权重加载 / post-load / scale dtype 处理完毕
- [ ] 融合冲突已处理或记录；ignore 不完备已按 §6.6 处理
- [ ] 未擅改量化方案语义

---

## 第七步：验证真实生效与收益

> 本步是量化验证要求的唯一真相源；其它 reference 不复述。

**读取**：`references/quantization-fusion-and-benefit.md` §B.1 五维口径。量化改造不能只看 diff，必须证明真实运行。

### 7.1 功能验证 + 等价性自检

- 功能：量化模型可加载，Prefill/Decode 至少各跑通一次，输出可读（不重复/非全零/非提前 EOS），日志能证明走到量化 runtime；验证命令记录模型路径、产物路径、卡数、YAML。
- **等价性自检（接线正确性，非精度评测）**：绝对精度由产物方在 `deploy_quantization.md` 报告，本 skill 不评测精度。固定一组 prompt，量化基线与 infer BF16 基线各跑 greedy，文本 diff 记首个分歧 token；W8A8 允许细微差异，重点是不出现乱码/早停/与 BF16 显著走偏（抓 scale/layout 接错的「能跑但输出垃圾」）。连 BF16 参照都跑不了：标「等价性未核，转产物方精度报告」，不静默判通过。

### 7.2 生效验证（probe 优先，贴输出，不空打勾）

- **probe-B 对象级（主，零开销）**：加载后抽查量化目标层，`type(layer.quant_method).__name__` 不应为 `UnquantizedLinearMethod`（应是 `CompressedTensorsW8A8Int8LinearMethod` / MoEGMM method 等）。只查 load 后构造结果。
- **probe-C 权重级**：从加载日志统计命中的 `qweight`/`weight_scale` 等量化张量数，与期望量化模块数对账。
- **probe-A 算子级（复用 §7.3 profiling，不单独跑）**：已 profiling 则在 `op_statistic` grep `QuantBatchMatmul`/`GroupedMatmul`/`DequantSwigluQuant`/`DynamicQuant` 计数 >0。

并记录：已量化模块 / `ignore` 浮点回退 / 融合算子回退。

### 7.3 性能和显存验证

显存 / Prefill / Decode 三项均对比「非量化基线、量化前最新基线、量化基线」，部署卡数对比「原形态、量化后形态」。性能数据多轮取中位数 + 充分 warmup，避免 single-sample。初评估判断量化后单卡可能满足时，优先给单卡验证结果。

### 7.4 失败处理

- 映射/加载/post-load/scale dtype/runtime object 错误：优先修 infer 侧。
- 融合算子冲突：回退非融合路径继续验证并记录需求。
- 契约缺失或权重不完整：停止改造，输出补充诉求。
- 功能生效但性能未升：保留证据，分析动态量化开销/算子粒度/部署配置/profile；先做归因分离再下结论，不直接套单一根因。

### 完成标志

- [ ] 功能验证通过
- [ ] 量化输出与 BF16 等价性自检通过（或已标记未核原因）
- [ ] 量化真实生效证据完整（probe-B 至少一层 quant_method 断言通过）
- [ ] 显存、Prefill、Decode、部署卡数有对比口径
- [ ] 失败项或回退项已记录

---

## 第八步：记录内容、后验决策与经验沉淀

### 8.1 量化应记录到 progress.md 的内容

> 写入由执行的 analyzer / implementer 子代理按其 `progress.md 写入格式`（只追加、不覆盖其他角色）落盘到 `{model_dir}/agentic/progress.md`（多优化路径如 longcat 为 `agentic/progress_{path}.md`）；归档（`archive_progress.py` → `progress_history.md`）与常驻区「量化接入常驻结论」由 model-infer-optimize 主流程维护。**本 skill 只定记录内容，不负责写入机制与归档**（与兄弟 skill 一致）。

记录内容清单见 `references/quantization-templates.md` 的「量化记录内容」。

### 8.2 后验决策依据（唯一后验决策真相源）

> 本节是量化基线产出后用户决策的唯一真相源；其它 reference 不复述。

量化基线产出后，本 Skill 只输出证据和建议，不替用户做最终取舍。给主流程提供三选择依据：① 采用部分回退的量化方案（接受当前收益，回退项沉淀为融合量化需求）；② 保留融合、跳过量化（收益不足或回退代价过高，继续非量化优化）；③ 修正量化方案后迭代（需补契约 / target / ignore / 张量语义 / 上游量化能力）。不在基线产出前给最终建议，不替用户确认是否接受。

### 8.3 经验沉淀

产生可复用经验时，按 `references/quantization-structure-cards.md` 末尾「经验沉淀模板」+「新结构补卡规则」追加：只按结构沉淀不按模型名堆卡；现有卡可覆盖时只补差异点和实践数据；只有出现新结构 / runtime object / 张量语义 / post-load 规则 / 融合回退模式时才新增卡。

---

## 输出格式

结束时按序输出：① 契约结论（满足/不满足/需补充）；② 当前模式；③ 量化产物目录、模式、覆盖范围；④ 结构参考卡、接入分级、关键映射；⑤ 已量化 / 浮点回退 / 融合回退模块；⑥ 验证证据和收益结论；⑦ 后续决策选项和建议；⑧ 阻塞点或补充诉求；⑨ 已沉淀或待沉淀的参考卡位置。

---

## 参考文档索引

> 按需查阅，避免一次性加载消耗 token。

| 主题 | 路径 |
| --- | --- |
| 产物契约 / 张量语义 / 运行对象映射 / 9 步机制（两种接入形态）/ 关键入口 | `references/quantization-contract.md` |
| 结构卡 A/B/C + 模型经验 + post-load 例外 + 经验沉淀模板 + 新结构补卡规则 | `references/quantization-structure-cards.md` |
| 融合算子兼容性 + 收益判断口径 + 主线路径 A1-A4 + 文件职责 | `references/quantization-fusion-and-benefit.md` |
| 初评估输出模板 + 量化记录内容清单 | `references/quantization-templates.md` |

---

## 反模式（final guard）

> 各章节已分散给出规则；本节汇总收尾前须排除的反模式。

- 重新跑上游量化实验 / 替代 AMCT-Q 等量化算法侧。
- 替代模型迁移、并行化、KVCache/FA、融合算子或图模式 Skill。
- 修改量化方案 / `targets` / `ignore` / 张量语义 / dtype 来掩盖错误（§2.2 / §6.5 / §6.6）。
- 猜测缺失张量语义（§2.2 / 第三步）。
- 造假 scale 强行量化；int8 forward without scale（§6.6 反模式）。
- 修改产物 `config.json`（产物语义只读；§6.6 反模式）。
- 静默回退（§7.4 / §8.2）。
- 未验证真实生效就宣称量化成功（§7 / §8.2）。
- 量化基线产出前替用户做最终取舍（§8.2）。
