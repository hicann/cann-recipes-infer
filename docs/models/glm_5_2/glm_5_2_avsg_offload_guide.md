# AVSG（Accelerated Vectorized Sparse Gather）：面向稀疏注意力 LLM 推理的高效 KV Cache 卸载

> 论文链接：<https://arxiv.org/abs/2609.37538>
> 原文作者：**Kuang Wenwei, Wang Xiangyu, Wu Chong, Wang Jun, zhang Weijie, Chen Brian K, Lan Longwen, Ken Zhang**
---

## 双机 Atlas A3 评测

与[shared_indexer_offload 技术文档](./glm_5_2_offload_guide.md)的实验设置保持一致，评测环境为生产级配置：双机 Atlas A3（共 16 卡）、GLM-5.3-W8A8 模型、64K 输入序列、MTP3。针对 64K 输入序列，在相同设备资源约束下，不使用 Offload 时，KV Cache 容量仅能支持全局 batch size 上限为 64；启用 Offload 后，该上限提升至 128。实验分别测试 8K 和 16K 两种 Device 侧 KV 缓存池容量，并覆盖 GE Graph 和 NPUGraph Ex 两种执行模式。

吞吐按两种口径报告：

- **统一假设口径**：为在相同 MTP 接收长度假设下比较不同配置，将 MTP 平均接收长度统一设为 2.7。该数值为经验假设，仅用于归一化估算，不代表各配置的实测接收长度。
- **实测口径**：使用推理过程中实际观测到的 MTP 平均接收长度。

计算方式：
```text
吞吐 = 全局 batch size × MTP 平均接收长度 /（主模型 + MTP3 单步延迟）
```

**GLM-5.3-W8A8 模型, dataset: InfiniteBench,  prompt length: 65536**
| 执行模式 | 配置 | 全局 batch | 缓存池容量 | 主+MTP3 延迟 | 吞吐（2.7 假设） | 吞吐（实测） | HBM Usage|
| --- | --- | ---: | ---: | ---: | ---: | ---: |---: |
| NPUGraph Ex | 无 Offload | 64 | - | 64.85 ms | 2,664.61 token/s | 3,038.94 token/s | - |
| NPUGraph Ex | 共享 Indexer Offload | 64 | 8K | 78.80 ms | 2,192.89 token/s | 2,500.98 token/s | - |
| NPUGraph Ex | 共享 Indexer Offload | 64 | 16K | 75.27 ms | 2,295.74 token/s | 2,619.73 token/s | - |
| NPUGraph Ex | AVSG | 64 | 8K | 76.02 ms | 2,273.09 token/s | 2,645.72 token/s | - |
| NPUGraph Ex | 共享 Indexer Offload | 128 | 8K | 99.18 ms | 3,484.57 token/s | 3,964.08 token/s | - |
| NPUGraph Ex | 共享 Indexer Offload | 128 | 16K | 90.33 ms | 3,825.97 token/s | 4,353.74 token/s | - |
| NPUGraph Ex | AVSG | 128 | 8K | 89.60 ms | 3,857.14 token/s | 4,489.65 token/s |57054/ 65536|
| NPUGraph Ex | 共享 Indexer Offload | 256|-|-|-|-|-|
| NPUGraph Ex | AVSG | 256 | 8K | 138.46 ms | 4,992.06 token/s | 4,494.38 token/s |62039/ 65536|

| 执行模式 | 配置 | 全局 batch | 缓存池容量 | 主+MTP3 延迟 | 吞吐（2.7 假设） | 吞吐（实测） | HBM Usage|
| --- | --- | ---: | ---: | ---: | ---: | ---: |---: |
| GE Graph | 无 Offload | 64 | - | 63.38 ms | 2,726.41 token/s | 3,109.82 token/s |63GB|
| GE Graph | 共享 Indexer Offload | 64 | 8K | 79.54 ms | 2,172.49 token/s | 2,477.74 token/s |55GB|
| GE Graph | 共享 Indexer Offload | 64 | 16K | 78.75 ms | 2,194.29 token/s | 2,503.91 token/s |57GB|
| GE Graph | AVSG | 64 | 8K | 76.19 ms | 2,268.01 token/s | 2,586.90 token/s |54GB|
| GE Graph | 共享 Indexer Offload | 128 | 8K | 101.86 ms | 3,392.89 token/s | 3,834.63 token/s |59GB|
| GE Graph | 共享 Indexer Offload | 128 | 16K | 97.96 ms | 3,527.97 token/s | 3,988.78 token/s |62GB|
| GE Graph | AVSG | 128 | 8K | 89.45 ms | 3,863.61 token/s | 4,367.11 token/s |57GB|
| GE Graph | 共享 Indexer Offload | 256 | 8K | OOM | - | - | - |
| GE Graph | 共享 Indexer Offload | 256 | 16K | OOM | - | - | - |
| GE Graph | AVSG | 256 | 8K | 120.23 ms | 5,748.98 token/s | 6,608.16 token/s |62GB|

## 从数据中可归纳：
### AVSG 以更低的 HBM 占用，在更小缓存池下实现更高吞吐，并将已验证并发提升至 256，最高实测吞吐较无 Offload 提升 112.49%、较共享 Indexer Offload 提升 65.67%**
1. **相比无 Offload，提高并发后，AVSG 吞吐提升 112.49%，共享 Indexer Offload 提升 28.26%。**
2. **相同缓存池容量下，AVSG 性能更优。**
	> 同为 8K 缓存池，AVSG 相比共享 Indexer Offload，在 batch 64 和 128 下的实测吞吐分别提升 4.41% 和 13.89%。这一结果表明 AVSG 的整体执行开销更低；是否源于更低的匹配开销或更高的缓存复用率，还需结合 profiling 和缓存命中率确认。
3. **AVSG 使用更小的缓存池，仍超过共享 Indexer Offload 的更大缓存池。**
	> AVSG 8K 相比共享 Indexer Offload 16K，在 batch 64 和 128 下，实测吞吐分别提升 3.31% 和 9.48%，同时 HBM 占用分别减少 3 GB 和 5 GB。
4. **相同缓存池容量下，AVSG 占用更少的 HBM。**
	> 同为 8K 缓存池，batch 64 下 HBM 从 55 GB 降至 54 GB，减少 1.82%；batch 128 下从 59 GB 降至 57 GB，减少 3.39%。数据证明了整体显存开销更低，节省量具体归因于 offload的metadata
5. **更低的显存占用支持更高并发。**
	> 在本次测试范围内，AVSG 支持 batch 256，HBM 占用为 62 GB；共享 Indexer Offload 在 batch 256 的 8K、16K 配置下均 OOM。AVSG 的最高已测吞吐相比共享 Indexer Offload 的最高已测吞吐提升 65.67%。

## 额外优势:
1. **支持原生TND格式，模型侧更亲和**
2. **支持950硬件， 950性能更佳**
