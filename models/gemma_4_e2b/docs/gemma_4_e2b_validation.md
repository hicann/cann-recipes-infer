# Gemma-4-E2B 验证报告

本文件保留仓内复现入口和结果摘要。完整基线、判定阈值、逐项验收结果、原始记录索引和仓外证据以 [RFC #530](https://gitcode.com/cann/cann-recipes-infer/issues/530) 为准。

## 验证摘要

- 目标环境：Atlas A2、Ascend 910B3、单逻辑设备、BF16、1TP、batch=1、CANN 9.0.0。
- 固定样本：10 个文本样本；eager 与 `npugraph_ex` 输出 10/10 一致，Prefill logits/top-1 对齐 10/10。
- 参考对齐：完整生成序列 9/10 一致；差异及首个分叉位置已记录在 RFC 和仓外证据中。
- 数据集：MATH-500 test 参考端 34/500，实现端 36/500，绝对差 0.4 个百分点。
- 性能：覆盖 128/512/1024/2048 输入 token、eager 与 `npugraph_ex`，每组预热 1 次、正式运行 3 次。
- 存储：2048 输入、128 输出、batch=1 两种模式均无 OOM；连续请求后 KV block 恢复，未观察到持续内存增长。

## 复现入口

先按本模型 [README](../README.md) 完成 CANN、torch_npu、Python 依赖和权重配置，再从仓库根目录运行：

```bash
bash executor/scripts/infer.sh --model gemma_4_e2b --mode offline \
  --yaml gemma_4_e2b_bf16_1tp.yaml
```

图模式将 YAML 替换为 `gemma_4_e2b_bf16_1tp_npugraph.yaml`。固定样本、参考端、性能、内存和数据集评分命令及其参数映射统一记录在 RFC #530；权重、数据集、Profiler 和大日志保存在仓外。
