# GE 图模式使用指南

> GE图模式通过 TorchAir 的 CompilerConfig 开启，将 FX 图转换为 Ascend IR 图，并通过 GE 图引擎实现图编译和执行。

---

## 适用场景

- **生产环境**：稳定性优先
- **通用场景**：功能丰富，支持广泛
- **复杂模型**：需要更多配置选项

---

## 快速上手

```python
# 导包（必须先导 torch_npu 再导 torchair）
import torch
import torch_npu
import torchair

# Patch方式实现集合通信入图（可选）
from torchair import patch_for_hcom
patch_for_hcom()  # 集合通信入图（有 TP/EP 并行时需调用）

model = YourModel().npu()

# 图执行模式默认为 max-autotune
config = torchair.CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)

# 基于 TorchAir backend 进行 compile
opt_model = torch.compile(model, backend=npu_backend)

# 执行编译后的 Model
output = opt_model(input_tensor)
```

> 完整示例见 `context/GE图模式快速上手.md`

---

## CompilerConfig 配置

```python
config = torchair.CompilerConfig()

# debug 类功能
config.debug.xxx = ...

# export 类功能（离线导图）
config.export.xxx = ...

# dump_config 类功能
config.dump_config.xxx = ...

# fusion_config 类功能
config.fusion_config.xxx = ...

# experimental_config 类功能
config.experimental_config.xxx = ...

# inference_config 类功能
config.inference_config.xxx = ...

# ge_config 类功能
config.ge_config.xxx = ...
```

---

## 核心 API

| API | 用途 | 文档路径 |
|-----|------|---------|
| `CompilerConfig类` | 配置图模式功能 | `context/CompilerConfig类.md` |
| `get_npu_backend()` | 获取 NPU 后端 | `context/get_npu_backend.md` |
| `get_compiler()` | 获取编译器 | `context/get_compiler.md` |
| `dynamo_export()` | 导出模型 | `context/dynamo_export.md` |
| `register_fx_node_ge_converter()` | 注册转换器 | `context/register_fx_node_ge_converter.md` |
| `register_replacement()` | 自定义算子融合 | `context/register_replacement-0.md` |

---

## 功能文档索引

> 路径相对于 `resources/PyTorch图模式使用指南（TorchAir）/`

### 概述

| 功能 | 文档路径 |
|-----|---------|
| GE图模式概述 | `context/GE图模式.md` |
| GE图模式功能 | `context/GE图模式功能.md` |
| GE动-静态图概念 | `context/GE动-静态图概念.md` |
| GE动-静态图展示 | `context/GE动-静态图展示.md` |

### 基础功能

| 功能 | 文档路径 |
|-----|---------|
| TorchAir Python层日志 | `context/TorchAir-Python层日志打印.md` |
| TorchAir C++层日志 | `context/TorchAir-C++层日志打印.md` |
| 集合通信入图 | `context/集合通信入图.md` |
| 图结构dump功能 | `context/图结构dump功能.md` |
| 算子data dump（Eager模式） | `context/算子data-dump功能（Eager模式）.md` |
| 图编译Debug信息（Ascend IR） | `context/图编译Debug信息保存功能（Ascend-IR）.md` |
| run-eagerly功能 | `context/run-eagerly功能.md` |
| 冗余算子消除（Ascend IR） | `context/冗余算子消除功能（Ascend-IR）.md` |
| FX图算子融合Pass（Ascend IR） | `context/FX图算子融合Pass配置功能（Ascend-IR）.md` |
| 自定义FX图Pass（Ascend IR） | `context/自定义FX图Pass功能（Ascend-IR）.md` |

### 高级功能

| 功能 | 文档路径 |
|-----|---------|
| Dynamo导图功能 | `context/Dynamo导图功能.md` |
| 模型编译缓存（Ascend IR） | `context/模型编译缓存功能（Ascend-IR）.md` |
| 图内Tensor打印 | `context/图内Tensor打印功能.md` |
| 多流表达功能（Ascend IR） | `context/多流表达功能（Ascend-IR）.md` |
| 动态shape图分档执行 | `context/动态shape图分档执行功能.md` |
| 图编译统计信息导出 | `context/图编译统计信息导出功能.md` |
| 单流执行功能 | `context/单流执行功能.md` |
| 图编译多级优化选项 | `context/图编译多级优化选项.md` |
| 算子融合规则配置（fusion_switch_file） | `context/算子融合规则配置功能（fusion_switch_file）.md` |
| 算子融合规则配置（optimization_switch） | `context/算子融合规则配置功能（optimization_switch）.md` |
| 固定权重类输入地址（Ascend IR） | `context/固定权重类输入地址功能（Ascend-IR）.md` |
| 计算与通信并行 | `context/计算与通信并行功能.md` |
| Tiling调度优化 | `context/Tiling调度优化功能.md` |
| View类算子优化 | `context/View类算子优化功能.md` |
| 动静子图拆分场景性能优化 | `context/动静子图拆分场景性能优化.md` |
| 图内标定SuperKernel范围 | `context/图内标定SuperKernel范围.md` |
| AI Core/Vector Core限核（Ascend IR） | `context/AI-Core和Vector-Core限核功能（Ascend-IR）.md` |
| 图内算子不超时配置 | `context/图内算子不超时配置功能.md` |

### torchair.ge API

| API | 文档路径 |
|-----|---------|
| torchair.ge 模块 | `context/torchair-ge.md` |
| GE图模式API列表 | `context/GE图模式API列表.md` |
| GE图模式API参考 | `context/GE图模式API参考.md` |
| DataType类 | `context/DataType类.md` |
| Format类 | `context/Format类.md` |
| Tensor类 | `context/Tensor类.md` |
| TensorSpec类 | `context/TensorSpec类.md` |
| Const | `context/Const.md` |
| Cast | `context/Cast.md` |
| custom_op | `context/custom_op.md` |

### torchair.inference API

| API | 文档路径 |
|-----|---------|
| cache_compile | `context/cache_compile-1.md` |
| readable_cache | `context/readable_cache-2.md` |
| set_dim_gears | `context/set_dim_gears.md` |

### torchair.ops API

| API | 文档路径 |
|-----|---------|
| npu_print | `context/npu_print.md` |
| npu_fused_infer_attention_score | `context/npu_fused_infer_attention_score.md` |
| npu_fused_infer_attention_score_v2 | `context/npu_fused_infer_attention_score_v2.md` |
| record | `context/record.md` |
| wait | `context/wait.md` |

### torchair.scope API

| API | 文档路径 |
|-----|---------|
| npu_stream_switch | `context/npu_stream_switch.md` |
| npu_wait_tensor | `context/npu_wait_tensor.md` |
| super_kernel | `context/super_kernel.md` |
| limit_core_num | `context/limit_core_num-3.md` |
| op_never_timeout | `context/op_never_timeout.md` |
| data_dump | `context/data_dump.md` |

---

## 相关文档

- **LLM 模型改造指南**：`llm-model-guide.md`（LLM 适配优先阅读）