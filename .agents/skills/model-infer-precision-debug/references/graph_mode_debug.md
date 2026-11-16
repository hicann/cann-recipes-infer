# 图模式精度定位

问题仅在图模式复现，或需要在图模式下取中间张量时使用。图模式的方案设计与非精度问题定界见 `model-infer-graph-mode`。

---

## 分层 backend 排查

用以下对照缩小范围，结合失败侧引入的机制进一步取证。

| 层次 | 对照机制 | 排查方向 |
|------|--------------------|-------------|
| eager | 原始计算 | 此处也复现时，回主流程逐级定位 |
| `backend="eager"` | 图捕获 | 捕获后的图与原始计算的差异、图断处理，参照 `model-infer-graph-mode` |
| `backend="aot_eager"` | 功能化 | 怀疑变异与别名契约，假原地算子是其中一种，见「假原地算子」 |
| 强制逐算子执行 | 图内逐算子下发 | 捕获后的 FX graph 与逐算子执行，参照 `model-infer-graph-mode` |
| 完整图模式 | 图编译优化与内存复用 | 编译优化与内存复用，参照 `model-infer-graph-mode` |

强制逐算子执行的开关按图模式路径不同。npugraph_ex 走 `options` 里的 `force_eager`。TorchAir GE 图模式没有完全等价的开关，最接近的是 `config.debug.run_eagerly`：它在 GE 图执行前用 Eager 模式跑一遍 FX graph，用于区分 TorchAir 的图变换与图执行器，开启后 GE 图模式相关的功能配置均不生效。该字段仓内无调用点，用法与取值按 TorchAir 官方文档确认。

按后端和部署版本查询 cannbot-knowledge 中固定权重地址、非连续参数与内存复用的精度案例，以及调试开关和图内取值接口。

### 假原地算子

功能化路径出现偏差时，核对算子元信息、别名与变异契约。判定为假原地算子需核对该算子声明的变异语义与实际写入行为是否一致。

---

## 图模式下的取值方法

图内取值使用对应后端支持的接口。低扰动的异步探针见 `operator_compare.md`。

### aclgraph 张量落盘

`save_npugraph_tensor` 支持 eager 和 aclgraph，可在 npugraph_ex 捕获的 forward 中调用：

```python
torch_npu.save_npugraph_tensor(x, save_path="./x.pt")
```

落盘产物按 `operator_compare.md` 的四项信息组织，两侧用同一份插桩，便于后续比对。
