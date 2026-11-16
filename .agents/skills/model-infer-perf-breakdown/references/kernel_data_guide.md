# 算子 CSV 输入契约

## 必需字段

`analyze_kernels.py` 按以下列名读取，每行对应一次算子执行：

| 字段 | 用途 |
|---|---|
| `Step Id` | 按 step 分组并提取目标 step |
| `Name` | 原始算子名称 |
| `Start Time(us)` | 开始时间，单位 µs |
| `Duration(us)` | 执行时长，单位 µs |
| `Stream ID` | 所属执行流 |

其他采集工具导出的 CSV 也须满足以上字段名称、语义及单位；有差异时，在 `<run_dir>` 生成适配后的副本，保留源数据行对应关系。

## 可选字段

| 字段 | 本工具中的用途 |
|---|---|
| `Type` | 优先用作 `normalized_name`；缺失或为空时从 `Name` 提取算子类型名 |
| `Accelerator Core` | 区分执行核心；用于 AI_CPU 过滤、通信记录去重和 Vector 候选识别 |
| `Task ID` | 结合 `Stream ID` 等字段的缺失情况，区分通信汇总与任务片段 |
| `Input Shapes` / `Output Shapes` | 辅助样本匹配和语义判断；缺失时记录为空字符串 |
| `Input Data Types` / `Output Data Types` | 供 Agent 核对量化计算与融合候选的数据类型 |
| `Input Formats` / `Output Formats` | 供 Agent 核对布局变换与融合候选的输入输出格式 |

其余采集指标可随 `raw_ops_details.json` 保留，供下钻查看。

## 行号对应

`org_index` 是传入 CSV 的 0-based 数据行号；`index` 是选定 step 内经过过滤后的致密索引。样本中的 `op_indices` 使用后者，换算见 [行号 domain 速查](phase0b_workflow.md#行号-domain-速查)。理论列合并与 CSV 拆分须使用同一份输入 CSV，保持行号对应。
