# GE 图模式使用指南

> GE图模式通过 TorchAir 的 CompilerConfig 开启，将 FX 图转换为 Ascend IR 图，并通过 GE 图引擎实现图编译和执行。

---

## 适用场景

已有 GE 部署或需将 FX 图转换为 Ascend IR 时，评估 TorchAir GE 后端。先确认目标算子有可用的 GE 转换路径，涉及集合通信时一并核对通信入图支持。

## 快速上手

以下示例使用已就绪的 NPU 模型 `model` 和输入 `input_tensor`。

```python
import torch
import torch_npu
import torchair

config = torchair.CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)

opt_model = torch.compile(model, backend=npu_backend)

output = opt_model(input_tensor)
```

infer 仓框架已在 [compile_model_forward](../../../../executor/utils/graph_utils.py) 中处理通信入图补丁和后端配置。接入框架的模型复用该入口。
