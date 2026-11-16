# LLM 模型图模式改造指南

> 本文档专门针对 **LLM 推理模型** 的图模式适配，提炼核心改造思路和最佳实践。
>
> **适用场景**：GPT、Llama、DeepSeek、Qwen 等 Transformer 架构的大语言模型推理。

---

## 核心原则

先区分图编译的符号形状/guard 支持与捕获回放的参数、形状和地址约束，再识别触发 graph break、重编译或重新捕获的具体因素；不要把所有动态值都当成动态输出形状。

| 动态因素 | 问题表现 | 解决思路 |
|---------|---------|---------|
| **内存地址变化** | 回放依赖的存储可能失效 | 预分配固定大小，原地更新 |
| **Shape 变化** | 可能触发重编译或重新捕获 | 固定、分档或按后端支持处理动态 shape |
| **依赖 Tensor 值的 Python 分支** | 可能中断捕获 | 使用 Tensor 操作或模式参数 |
| **`.item()` 调用** | 从设备取标量可能引入同步或图中断 | 保持 Tensor，或将依赖该值的 Python 决策移到捕获边界外 |

---

## 重编译问题定位与解决

### 定位重编译

```python
# 开启重编译日志
torch._logging.set_logs(recompiles=True)

# 运行模型
output = compiled_model(input)
# 如果发生重编译，会打印类似：
# [recompiles] Recompiling function <func_name> for reason: <reason>
```

### 重编译解决方案

```
发现重编译
    │
    └─→ 读取具体 guard 失败原因
            ├── 缓冲地址变化 → 稳定存储并核对捕获边界
            ├── Python 长度或控制值变化 → 按后端接口外部化或刷新参数
            └── 形状变化 → 按后端能力选择固定、分档或动态编译，并验证图复用
```

先证明 guard 对应的不变量成立，再考虑专用优化；不要把跳过 guard 校验作为固定形状重编译的默认修复，`dynamic=True` 也不替代运行时捕获约束验证。

---

## 模块改造指南

### 1. KV Cache 模块改造

KV Cache 动态扩展导致形状或地址变化、影响图复用时，预分配存储并原地更新。按本步写位置写入，并用有效上下文长度控制读取范围；容量固定时每步读量仍可变化。

已原地更新时，是否还需返回 cache 取决于调用方。具体接入见 [KVCache 框架实施参考](../../model-infer-kvcache/references/framework_kv_reference.md)。

### 2. Rotary Embedding 模块改造

现有 RoPE 路径稳定入图时沿用原实现。如果图内动态构造或主机取值阻断捕获，可在外层按实际位置准备 cos/sin 并跨层复用。见 [RoPE 预计算模板](../../model-infer-fusion/references/rotary-embedding-pattern.md)。

### 3. Attention 模块改造

按阶段核对长度、mask 和 KV cache 元数据。打包输入（TND）的张量维度不能给出各请求的有效长度，FA 所需的逐序列长度或累计长度取自当前步的请求元数据。接口选择与长度刷新见[图模式下的 FA 长度参数](../SKILL.md#图模式下的-fa-长度参数)。

### 4. Buffer/Parameter 模块改造

回放依赖的 buffer 或 parameter 若在步间更换存储，保留其存储并按需用 `copy_()`、`fill_()` 原地更新。

---

## infer 仓框架的 Decode 编译入口

infer 仓框架复用已有的 Prefill/Decode 阶段标记，Decode 从 [ModelWorker](../../../../executor/core/model_worker/model_worker.py) 的编译入口执行。主模型与 draft 模型分别编译，避免两条路径共用同一缓存函数。默认入口缺失时由框架提供转发包装；自定义 worker 可指定专用接口。

```python
# 默认 Decode 入口分属主模型和 draft 模型，均沿用原 forward 的输入。
def main_decode(self, **model_inputs):
    return self.forward(**model_inputs)

def mtp_decode(self, **model_inputs):
    return self.forward(**model_inputs)
```

---

## 相关文档

- [npugraph_ex 编译模板](npugraph_ex-guide.md#快速上手)
- [GE 编译模板](ge-graph-guide.md#快速上手)
