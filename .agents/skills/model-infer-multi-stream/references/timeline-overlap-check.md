# 多流 overlap 时间线复盘

用算子实际执行区间确认预期分支的重叠，再结合关键路径与 wall 变化判断收益。

## 输入与范围

从同一次采集、同一 device 的 `kernel_details.csv` 或等价明细中，取目标算子的 `Stream ID`、`Name`、`Start Time(us)` 和 `Duration(us)`。按方案确定副流算子集合，以及预期与之重叠的主流算子集合。

选中的算子对应方案约定的同一并行窗口，记录层 / 模块归属和算子索引，便于复核。

## 计算实际重叠

每个算子的执行区间为 `[start, start + duration)`。分别合并两组算子的执行区间，再求交集；区间之间的空闲不计入有效执行时长。

```text
side_busy   = union(side_op execution intervals)
main_busy   = union(main_op execution intervals)
side_len    = duration(side_busy)
overlap     = duration(side_busy ∩ main_busy)
overlap_pct = overlap / side_len
```

`overlap_pct` 表示副流有效执行时间中与主流重叠的比例，取值为 0–1，展示时转为百分比；`side_len` 为 0 时记为不适用。

## 根据证据选择下一步

| 观察 | 核对与调整 |
| --- | --- |
| 目标算子未进入预期流 | 核对执行分支、开关与实际编译落流 |
| 已分流但重叠少、存在串行等待 | 对照生产者、消费者和等待位置，调整分流集合或汇合点；GE 路径核对前处理算子的编译调度 |
| 有并行窗口但一侧持续拖尾 | 结合两侧时长、核资源与带宽占用，评估窗口划分和资源分配 |
| 重叠增加但 wall 无改善 | 核对关键路径、shape 效率、额外 task / 搬运及 host 开销 |

重叠比例用于解释并行窗口的利用情况；方案取舍结合输出正确性和同负载下的时延收益，不设通用比例门槛。

## 结果记录

```text
副流算子及执行区间：...
主流目标算子及执行区间：...
side_len：... us；overlap：... us；overlap_pct：...%
同口径 wall 变化：...
主要等待 / 资源证据及下一步：...
```

跨采集对比使用同口径 wall、算子耗时及区间相对关系；各次采集的时间戳基准不同，重叠在各自采集内计算。
