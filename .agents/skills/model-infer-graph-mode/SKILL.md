---
name: model-infer-graph-mode
description: 基于 PyTorch 框架的昇腾 NPU 模型推理图模式适配技能。将模型适配到 torch.compile 图模式以加速推理性能。触发场景：npugraph_ex 或 GE 图模式适配、torch.compile 在昇腾 NPU 上的使用、图中断（Graph Break）修复、aclgraph 图编译问题。LLM 模型的图模式适配优先阅读 LLM 模型改造指南。
---

# 图模式适配优化技能

提供 npugraph_ex 和 GE 两种图模式的适配指南，覆盖方案设计、图中断修复、FA 参数配置和验证测试。

---

## 重要原则

- **区分捕获、编译与回放**：前端成功捕获计算图，不能据此推断设备下发减少。评估回放收益时，单独核算算子融合或 SuperKernel 的影响。
- **保持模型完整性**：不为适配图模式而简化模型逻辑。
- **固定存储与逻辑状态分开**：回放依赖的缓冲区须保持存储稳定，有效 KV 长度、写位置和缓存内容仍按步更新。形状标记不能证明地址稳定。

---

## 工作流程

### 方案分析

1. 确定目标阶段和候选后端，按部署版本核对算子、通信和状态更新能否入图。
2. 根据实际执行路径识别图中断、重编译或回放的阻断点，向调用方返回最小改造方案和待决事项。

### 实施

以可运行的 eager 路径为基线，按已确认方案实施改造。需要变更方案时，将待决事项返回调用方。

### 验证测试

静态分析用于判断可行性。实际适配在目标 NPU 上与同一执行路径的 eager 基线对比，使用相同输入和量化配置。

- **编译与复用**：确认已预热的输入规格在稳态执行时复用已有图。infer 仓阶段 5 还须确认已选 Decode 路径完整入图、无图中断。
- **功能与性能**：对比目标阶段的输出和稳态耗时，区分首次编译开销与推理耗时。

---

## 图模式选择

用户已指定后端时沿用其选择。infer 仓新接入图模式时优先评估 `npugraph_ex`。

| 场景 | 评估入口 | 详细文档 |
|------|---------|---------|
| **LLM 稳定形状 Decode** | 核对 KV 存储稳定性及有效长度、写位置的更新 | `references/llm-model-guide.md` |
| **固定形状通用推理** | 核对 npugraph_ex 的算子与捕获支持 | `references/npugraph_ex-guide.md` |
| **已有 GE 部署或需 Ascend IR 图编译** | 核对 GE 后端能力与配置 | `references/ge-graph-guide.md` |

选择 GE 或 npugraph_ex 时，可按部署版本查阅 cannbot-knowledge 或 TorchAir 官方文档，核对捕获限制、集合通信支持和编译选项。

---

## 编译缓存使用建议

图可稳定复用后，再用 `cache_compile` 对比二次启动的编译耗时。输入规格、模型代码、通信域或软件版本变化时，重新核对缓存是否适用。infer 仓框架已在 [compile_model_forward](../../../executor/utils/graph_utils.py) 中按后端选择常规编译或持久缓存。

---

## LLM 模型适配要点

infer 仓优化流程阶段 5 仅适配 Decode，Prefill 沿用 eager。其他部署中，固定或分档的 Prefill 可按目标后端能力评估。

KV、RoPE 和长度输入的适配判断，以及 infer 仓框架的阶段入口，见 [LLM 模型改造指南](references/llm-model-guide.md)。

---

## 问题定界流程

问题尚未定界时，用相同输入复现 eager 基线，按需比较前端捕获（`backend="eager"`）、功能化（`aot_eager`）和目标后端执行，从首次出现差异的层定位。npugraph_ex 可用 `force_eager` 隔离回放路径；GE 的 `run_eagerly` 用于检查 TorchAir 转换后的 FX 图，两者的结果按各自作用解释。

若 Decode 遇到 `hit recompile_limit`，每步变化的 KV 长度是可能原因之一。若日志显示 guard 失败，先按 [LLM 指南](references/llm-model-guide.md#重编译问题定位与解决) 核对失败条件。需要具体案例时，按后端、版本和实际报错查询 cannbot-knowledge 中的图中断、重编译或回放故障。仅图模式下出现的精度偏差，见 [图模式精度定位](../model-infer-precision-debug/references/graph_mode_debug.md)。

---

## 图模式下的 FA 长度参数

按实际调用的 FA 接口、版本和 layout 核对 Q/KV 长度：逐序列有效长度与累计长度含义不同，host list 与 Tensor 是它们的传递形式。检查本轮长度在何处生成、每步 Decode 如何刷新，以及消费算子实际拿到的值。

infer 仓使用 `ForwardMetaData` 提供长度字段。例如 npugraph_ex Decode 的 Qwen 路径通过 `torch.ops.npu` 给 FA v2 传累计 Q 长度和逐序列 KV 长度的 host list；GE Decode 则调用 `torchair.ops`，传入对应的 Tensor。编译的 `dynamic` 配置应结合目标后端的 shape/guard 与回放约束确定。参见 [Qwen Attention 实现](../../../models/qwen/models/modeling_qwen.py)。

查询目标版本的 `torch_npu` 接口签名可用 [model-infer-fusion](../model-infer-fusion/SKILL.md)；需要对照实现时，查询 cannbot-knowledge 中同框架、同后端与 FA 接口版本的长度传递和 KV cache 样例。
