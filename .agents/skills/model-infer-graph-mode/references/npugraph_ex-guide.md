# npugraph_ex 后端使用指南

> npugraph_ex 是基于 torch.compile 的 aclgraph 图模式加速方案，通过捕获模式 (Capture & Replay) 构建模型运行实例。

---

## 适用场景

适合评估下发开销占比较高、输入形状稳定的推理路径，例如固定本轮 token 数的 LLM Decode。

---

## 使用约束

| 约束项 | 说明 |
|-------|------|
| 随机数算子 | 核对目标版本对 randn、dropout 与 RNG 状态管理的捕获支持 |
| 动态控制流 | 核对可捕获控制流与 graph break 边界，区分图编译和运行时捕获约束 |
| Stream同步 | 核对捕获区内允许的 stream/event/sync 操作及边界 |

---

## 快速上手

以下以已就绪的 NPU 模型 `model` 和固定形状输入 `input_tensor` 为例。

```python
import torch
import torch_npu

opt_model = torch.compile(model, backend="npugraph_ex", fullgraph=True, dynamic=False)
# 长度字段变化时按后端的 guard 与参数刷新机制配置；dynamic=True 不放宽回放地址约束
output = opt_model(input_tensor)
```

## 回放性能对比

单独评估捕获回放收益时，保持融合、inplace、输入/输出克隆和图池策略一致。这些选项会改变 kernel、访存或内存占用。若性能下降，先区分重新编译、重新捕获和稳态回放耗时，再对照[问题定界流程](../SKILL.md#问题定界流程)。
