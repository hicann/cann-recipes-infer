# NPU 图模式优化及其在 cann-recipes-infer 框架下的使能

## 1. 文档目标

本文主要回答四个问题：

1. 什么是图模式，它解决什么问题？
2. NPU 图模式有哪些主要实现方式，它们之间是什么关系？
3. 在本仓的执行框架下，如何为模型使能图模式？
4. 适配和调试过程中最常见的问题是什么，应该如何排查？

---

## 2. 图模式基础知识

### 2.1 什么是图模式

pytorch 默认按照`eager` 模式，可以理解为“边解释边执行”的模式。模型前向走到哪里，框架就把当前位置对应的算子逐个下发到 Device 执行，开发和调试都比较直接，但在推理场景下也更容易暴露 Host 侧逐算子下发的开销。  
在图模式下，前向逻辑会先被 `torch.compile` 捕获为图，再交给 NPU 后端编译和执行。这样做的核心收益是：

- 减少 Host 逐算子下发开销，缓解 host bound。
- 让后端在更大范围上做算子融合、内存复用、调度优化。

这通常可以显著降低时延。

### 2.2 为什么图模式通常只用于 decode

对 LLM 推理来说，`prefill` 和 `decode` 的特征差异很大：

| 阶段 | 输入特征 | 是否适合图模式 | 原因 |
|------|----------|----------------|------|
| `prefill` | 序列长度动态变化，序列通常较长 | 通常不建议 | shape 和控制流容易变化，容易断图或重编译；同时 prefill 阶段单次算子执行时间更长，下发 bound 往往没有 decode 明显 |
| `decode` | 单 token 或固定小长度输入 | 推荐 | shape 更稳定，更容易形成可复用图 |

### 2.3 什么是编译缓存

图模式除了“如何编译”，还有一个经常一起出现的概念：**编译缓存**，也就是 `cache compile`。

它解决的问题是**重复启动或重复执行同一图时的启动开销**。  
如果模型结构、输入 shape、dtype 和图模式配置等保持稳定，把第一次编译的结果缓存下来，后续运行可以直接复用缓存，减少启动开销。
需要注意的是，编译缓存是否命中，强依赖模型代码、缓存目录和图模式配置是否一致。

---

## 3. NPU 图模式的支持方式

### 3.1 实现方式

NPU 图模式实现方式可以归纳为两大类：

1. **GE 图模式**：将 FX 图转换成 Ascend IR，再由 GE 引擎编译执行

2. **npugraph_ex 图模式**：基于 npugraph capture & replay，强调下沉调度和低开销执行

### 3.2 为什么文档里还会看到 `acl_graph`

仓内存在一部分较早的模型代码和文档，仍然使用 `acl_graph` 这个历史命名，实际阅读时通常可以把它对应到 `npugraph_ex` 这一路图模式。但注意：
- 新接入模型时，优先使用执行框架的 `ge_graph` / `npugraph_ex` 配置。
- 后续会逐步废弃 `acl_graph`，并清理对应实现和文档。

### 3.3 两种方式如何选择

目前建议：

- **优先功能稳定、性能更优**：选择 `ge_graph`。当然，`npugraph_ex` 也在持续优化中，本文档会持续更新。
- **优先适配更轻量，使用体验更接近 eager 模式**：选择 `npugraph_ex`，是本仓后续主推的模式。

---

## 4. cann-recipes-infer 如何使能图模式

### 4.1 使能链路

1. 配置层：
   - `executor/core/config/inference_config.py`
   - 通过 `model_config.exe_mode` 选择 `eager` / `ge_graph` / `npugraph_ex`

2. 预热层：
   - `executor/core/engine/execution_engine.py`
   - `warm_up()` 会先跑一次 `prefill`，再跑一次 `decode`，如果启用了图模式，`decode` 预热阶段会触发编译

3. 编译层：
   - `executor/core/model_worker/model_worker.py`
   - `compile_model()` 最终调用 `executor/utils/graph_utils.py` 中的 `compile_model_forward()`

4. 执行层：
   - `ModelWorker.inference()` 中，只有 `not is_prefill` 且 `exe_mode in ["ge_graph", "npugraph_ex"]` 时，才走 `self.model_compiled(**model_inputs)`

5. 元数据层：
   - `executor/utils/forward_metadata.py`
   - `ForwardMetaData` 负责把 `is_prefill`、`kv_len`、`actual_seq_lengths_*`、`attention_mask` 等动态信息传给模型

需要注意的是，图模式开关只对 decode 阶段生效，prefill 阶段统一走 eager 模式。

### 4.2 `graph_utils.compile_model_forward()` 核心逻辑解析

`OfflineInference` 最终真正执行图编译的核心函数在 `../../executor/utils/graph_utils.py` 里的 `compile_model_forward()`。
`npugraph_ex` 和 `ge_graph` 在这个函数里的主流程大体一致：先做通用准备，再根据 `exe_mode` 组织图编译配置，最后根据 `enable_cache_compile` 选择普通编译还是 `cache_compile`。差异主要在于配置承载方式，以及 `dynamic` 的默认取值不同。

#### 通用准备逻辑

```python
import torchair as tng
import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce

tng.patch_for_hcom()
torch._dynamo.config.inline_inbuilt_nn_modules = False
```

- `tng.patch_for_hcom()` 是用来处理集合通信入图的；在 PyTorch 2.6 及之后版本中，这一步通常可以省略。
- `inline_inbuilt_nn_modules = False` 用于避免内建模块被过度内联，减少部分图编译场景下的不确定性。

#### 两种图模式的主要差异

真正需要开发者关注的差异主要有下面几项：

| 项目 | `npugraph_ex` 模式 | `ge_graph` 模式 |
|------|--------------------|-----------------|
| 配置承载方式 | 通过 `options` kwargs 传入 | 通过 `CompilerConfig.experimental_config` 成员配置 |
| 普通编译后端 | `backend="npugraph_ex"` | `backend=tng.get_npu_backend(...)` |
| 缓存编译接口 | `torch.npu.npugraph_ex.inference.cache_compile(...)` | `tng.inference.cache_compile(...)` |
| `dynamic` 设置 | `True` | `False` |
| 配置项 | `static_kernel_compile` / `frozen_parameter` | `frozen_parameter` / `tiling_schedule_optimize` / `topology_sorting_strategy` |

补充说明：当前 `npugraph_ex` 模式之所以保持 `dynamic=True`，核心原因并不是图本身必须动态，而是当前配套使用的 FIA 算子接口里，`actual_seq_lengths` 等入参还不支持 Tensor 输入，只支持 `list[int]` 输入。在这种前提下如果强行使用 `dynamic=False`，容易触发重编译。后续算子接口补齐 Tensor 输入支持，这里的配置也会随之调整。

### 4.3 图模式适配里最常见的两个问题

- **Graph Break**：图捕获过程中断，部分逻辑回退到 Python/Eager 执行。一般需要通过减少 Python 控制流、避免 `.item()`、补齐入图适配来解决。
- **Recompile**：虽然能入图，但由于 shape、地址或 guard 条件变化，导致反复重新编译，性能变差。一般需要通过固定 shape、固定缓存地址，并把动态量改为显式输入来解决。

### 4.4 模型适配时必须满足的条件

仅仅打开 `exe_mode` 不够。模型本身需要满足图模式约束。

#### 条件一：模型必须先能在 eager 下稳定运行

图模式不会修复 eager 下本来就存在的错误。

#### 条件二：prefill 和 decode 要明确区分

推荐做法：

- `prefill` 保持 eager
- `decode` 使用图模式
- 用 `forward_metadata.is_prefill` 或独立的 `prefill()/decode()` 方法区分两条路径

#### 条件三：将动态信息以显式输入形式传入模型

典型动态信息包括：

- `kv_len`
- `position_ids`
- `actual_seq_lengths_q`
- `actual_seq_lengths_kv`
- `is_prefill`

这些信息应该由框架构造后传给模型，而不是在模型内部临时生成 Python 标量或依赖隐式状态推导。

#### 条件四：KV Cache 和常驻 buffer 需要预分配并原地更新

错误示例：

```python
key = torch.cat([past_key, new_key], dim=1)
```

推荐示例：

```python
torch_npu.scatter_update_(k_cache, kv_len, key_states, -2)
torch_npu.scatter_update_(v_cache, kv_len, value_states, -2)
```

目标是：

- 避免 decode 场景下 KV Cache 的 shape 或地址发生变化，以减少 dynamo guard 失败和重编译

#### 条件五：避免典型的 Graph Break 写法

尤其要避免：

- `tensor.item()`
- 基于 Tensor 值的 Python `if/while`
- 在 forward 内部临时创建影响 shape 的控制分支
- 根据 Python list/tuple 长度变化来切换图内控制流

### 4.5 FIA 融合算子适配建议

不同模式下，FIA算子的接口入参有所区别，推荐按照如下方式使用：

| 模式 | 常见 FIA 接口 | `actual_seq_lengths` 建议 | 说明 |
|------|--------------|---------------------------|------|
| `ge_graph` | `torchair.ops` | Tensor | 只适合 GE 图模式 |
| `npugraph_ex` | `torch_npu` | 常见为 `list[int]` | 当前以推理场景的 FIA 接口为主，后续算子支持变化时再调整 |

可以参考qwen3-moe的样例代码：

- `models/qwen3_moe/models/modeling_qwen3_moe.py` 在 decode 且 `enable_gegraph` 时，会切到 `torchair.ops`
- `ExecutionEngine` 在 `npugraph_ex` decode 模式下，会把 `actual_seq_lengths_*` 转成 `list`

---

## 5. 模型接入图模式的推荐步骤

建议按下面的顺序推进，而不是一次性把所有优化叠加上去。

1. 先跑通 eager 模式并确认精度正确。
2. 消除 graph break 和重编译风险，去掉 `.item()` 和动态 Python 控制流，固定 KV cache 与常驻 buffer 的地址和 shape，并明确 `actual_seq_lengths` 的类型与来源。
3. 做功能、精度验证，对比 eager 和 graph 输出，至少覆盖一轮 prefill + 多轮 decode；如果模型有 MTP，还要确认 main model 和 MTP model 的图模式输入 shape、dtype 和长度组织方式都稳定。
4. 最后再按需开启增强特性，例如 `enable_cache_compile`、`enable_static_kernel`（仅 `npugraph_ex`）、model 自带的 `enable_superkernel`（目前仅 `ge_graph`），以及多流、限核等能力。

---

## 6. 常见 Troubleshooting

### 6.1 高频问题速查表

| 现象 | 常见根因 | 处理建议 |
|------|----------|----------|
| 编译前就报错 | eager 路径本身不正确，或者模型输入的 shape、dtype、长度组织方式不稳定 | 先单独验证 eager，再检查图模式输入和前向参数组织 |
| 图捕获中断（Graph Break） | `.item()`、Tensor 驱动的 Python 分支、print、自定义算子未适配 | 改写为 Tensor 逻辑，或补齐入图适配 |
| decode 性能没有提升甚至变差 | 发生重编译；`kv_len`、`actual_seq_lengths_*`、缓存地址或输入 shape 不稳定 | 打开 `torch._logging.set_logs(recompiles=True)` 检查重编译原因，重点关注 guard 变化来源 |
| `actual_seq_lengths` 类型报错 | 图模式与 FA 接口不匹配 | `ge_graph` 优先 Tensor + `torchair.ops`，`npugraph_ex` 对齐本仓当前 `list[int]` 方案 |
| `enable_static_kernel` 报错 | 模式不对 | 该选项只允许在 `npugraph_ex` 模式使用 |
| `superkernel` 相关报错 | 在不支持的模式上开启 | 目前只在 `ge_graph` 模式尝试 |
| 通信无法入图 | 当前依赖版本要求的集合通信入图前置处理没有完成 | 结合当前 PyTorch / TorchAir 版本检查是否需要 `torchair.patch_for_hcom()` 或其他等效前置配置 |
| cache compile 不生效 | 缓存目录、输入 shape / dtype / 长度组织方式或配置变化 | 固定 cache 目录，避免模型代码或编译参数频繁变化 |
| 图模式下精度异常 | KV cache / FA / 原地更新语义变化 | 先回退到 eager 对齐，再逐项恢复优化 |

### 6.2 常用调试手段

实践中常用的调试手段包括：

- `torch._logging.set_logs(recompiles=True)`：出现重编译时打开相关日志
- eager / graph 输出比对：看功能、精度是否一致
- cache compile 开关测试：区分“图编译问题”还是“缓存命中问题”

---

## 7. 参考资料

### 仓内资料

- [Offline Inference 执行机制设计文档](../design/offline_inference_design.md)
- [InferenceConfig 类使用指南](../common/inference_config_guide.md)
- [MTP 模型接入指南](../common/mtp_model_guide.md)

### 官方资料

- TorchAir 文档总览  
  https://gitcode.com/Ascend/torchair/tree/master/docs/zh

- GE / Ascend IR 图模式  
  https://gitcode.com/Ascend/torchair/tree/master/docs/zh/ascend_ir/ascend_ir.md

- GE 图模式快速上手  
  https://gitcode.com/Ascend/torchair/tree/master/docs/zh/ascend_ir/quick_start.md

- npugraph_ex 后端  
  https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/npugraph_ex.md

- npugraph_ex 快速上手  
  https://gitcode.com/Ascend/torchair/tree/master/docs/zh/npugraph_ex/quick_start.md

- 常见案例与定位方法  
  https://gitcode.com/Ascend/torchair/tree/master/docs/zh/appendix/cases

- FAQ  
  https://gitcode.com/Ascend/torchair/tree/master/docs/zh/appendix/faq.md
