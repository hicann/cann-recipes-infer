# 图模式精度定位

问题仅在图模式复现，或需要在图模式下取中间张量时使用。图模式的方案设计与非精度问题定界见 `model-infer-graph-mode`。

---

## 分层 backend 排查

从 eager 起逐层加机制，第一个失败的层即问题所在。

| 层次 | 相对上一层新增的机制 | 该层失败说明 |
|------|--------------------|-------------|
| eager | 无 | 问题不在图模式路径上，回主流程逐级定位 |
| `backend="eager"` | 图捕获 | 捕获或图断处理有问题，转 `model-infer-graph-mode` |
| `backend="aot_eager"` | 功能化 | 怀疑变异与别名契约，假原地算子是其中一种，见「假原地算子」 |
| 强制逐算子执行 | 图内逐算子下发 | 图编译层有问题，转 `model-infer-graph-mode` |
| 完整图模式 | 图编译优化与内存复用 | 编译优化或内存复用有问题，转 `model-infer-graph-mode` |

强制逐算子执行的开关按图模式路径不同。npugraph_ex 走 `options` 里的 `force_eager`。TorchAir GE 图模式没有完全等价的开关，最接近的是 `config.debug.run_eagerly`：它在 GE 图执行前用 Eager 模式跑一遍 FX graph，用于区分 TorchAir 的图变换与图执行器，开启后 GE 图模式相关的功能配置均不生效。该字段仓内无调用点，用法与取值按 TorchAir 官方文档确认。

### 假原地算子

该层失败还可能来自算子元信息、别名与变异契约的声明错误。判定为假原地算子需核对该算子声明的变异语义与实际写入行为是否一致。

### 固定权重地址与非连续 parameter

完整图模式层失败的高频原因。`frozen_parameter` 在本仓默认开启，模型中存在非连续的 parameter 对象时，图模式结果与 eager 不一致，关闭该功能即正常。开关位置见 `executor/model_runner.py` 与 `executor/utils/graph_utils.py`。

判定：开启 TorchAir Python 侧日志，看 `runtime inputs` 各输入的 `contiguous` 标记，出现 `contiguous=False` 即命中。修复是把该 parameter 转为连续，关闭 `frozen_parameter` 只用于定界，不作为修复。

---

## 图模式下的取值方法

图模式不能直接打印，落盘接口如下。低扰动的异步探针与图模式无关，见 `operator_compare.md`。

> 以下接口仓内无调用点，未在真实 NPU 图模式验证，使用前先在目标环境确认可用。

### 图模式张量落盘

用法类似 `torch.save`，可在图内调用。

```python
torch_npu.save_npugraph_tensor(x, save_path="./x.pt")
```

落盘产物按 `operator_compare.md` 的四项信息组织，两侧用同一份插桩，便于后续比对。
