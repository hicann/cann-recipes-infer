# 基于Atlas A3集群的DeepSeek-R1模型decode阶段推理性能优化
## 概述
本文主要介绍基于Atlas A3 系列产品的DeepSeek-R1模型decode阶段的并行部署和性能优化策略。本文将介绍如何在Atlas A3 系列产品上进行大EP（Expert Parallel）推理，及其他并行技术和优化方案。

## 性能优化
### 通用优化
DeepSeek-R1结构中的非MoE部分与Llama类似，通用优化点可参考[Llama](https://gitcode.com/Ascend/torchair/tree/master/npu_tuned_model/llm/llama)的改动，如固定KV Cache大小、cos/sin优化、AddRMSNorm融合、全量优化LM Head计算量等。

### MLA (Multi-Head Latent Attention)低秩压缩优化
#### 使能融合算子
参考[Deepseek论文](https://arxiv.org/pdf/2405.04434)中提及的低秩压缩方法，可以减少KV cache占用的内存，提升推理效率。在实现MLA低秩压缩后，可以通过使能融合kernel实现性能优化，相关实现可以参考`DeepseekV3Attention`类中的 `forward_page_attention_mla_prolog`函数。

- MLA前置计算性能优化：使能[npu_mla_prolog_v2](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_mla_prolog_v2.md)融合kernel，替换attention计算前的计算，其中包含Q、K、V的线性层计算、旋转位置编码 (ROPE)、RmsNorm计算及KV Cache更新等计算处理；
- Attention性能优化：使能[npu_fused_infer_attention_score](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_fused_infer_attention_score.md)融合kernel，实现对MLA计算的加速。

### MoE模块实现Expert Parallel (EP)及使能融合算子

在MoE模块中，如采用原始的MoE实现，将通过for循环处理`expert_num`个FFN专家，效率较低。针对MoE层中的token路由和专家计算等操作，CANN提供了一系列融合算子，提升计算效率。

- Router计算优化：使用[torch_npu.npu_moe_gating_top_k](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_moe_gating_top_k.md)算子，对router计算的结果排序，并选取前top-k个专家；
- 高性能专家计算：使用[torch_npu.npu_grouped_matmul](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_grouped_matmul.md)算子，可以同时处理多个专家的计算，提高计算和搬运效率；
- 多卡间高性能通信路由：使能[torch_npu.npu_moe_distribute_dispatch_v2](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_moe_distribute_dispatch_v2.md) 和[torch_npu.npu_moe_distribute_combine_v2](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_moe_distribute_combine_v2.md)算子，实现EP并行下多卡间的通信。在使用前可参考上述的算子文档，检查`HCCL_BUFFSIZE`等环境变量的配置是否合理，了解该算子的使用场景和约束。

### MLP线性层计算合并

原始`DeepseekV3MLP`实现中，存在`gate_proj`、`up_proj`与`down_proj`三个matmul运算，可通过将`gate_proj`与`up_proj`进行合并计算，得到`gate_up_proj`提升整体计算效率。

### torchair图模式

Decode场景一般时延要求较高，在pytorch的`eager_mode`下，算子很可能面临比较严重的host下发bound，此时可以通过torch.compile使能图模式进行优化。TorchAir（Torch Ascend Intermediate Representation）是Ascend Extension for PyTorch（torch_npu）的图模式能力扩展库，提供了昇腾设备亲和的torch.compile图模式后端，实现了PyTorch网络在昇腾NPU上的图模式推理加速以及性能优化。

TorchAir提供了max-autotune和reduce-overhead两种实现模式，具体可参考[文档](https://www.hiascend.com/document/detail/zh/Pytorch/710/modthirdparty/torchairuseguide/torchair_00003.html)。本样例在decode场景下对这两种模式都支持，可以通过配置yaml文件中`exe_mode`使能，其中`ge_graph`对应max-autotune模式，`acl_graph`对应reduce-overhead模式。

### 支持Multi-Token Prediction (MTP)
根据Deepseek论文中介绍的MTP方法，实现了MTP投机推理，在未达到计算bound的场景下，MTP计算可以实现较好的推理加速效果。可通过`next_n`参数使能MTP。


### 集合通信使能AIV展开
利用Device的Vector Core计算单元来加速通信操作的执行，可参考[HCCL_OP_EXPANSION_MODE环境变量](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/maintenref/envvar/envref_07_0096.html)：

```shell
export HCCL_OP_EXPANSION_MODE=AIV
```

### SuperKernel优化
在decode使能`ge_graph`图模式的场景下，支持对模型的计算图按照用户定义的范围，进行SuperKernel优化。SuperKernel技术的详细介绍，请参考[官方文档](https://www.hiascend.com/document/detail/zh/Pytorch/710/modthirdparty/torchairuseguide/torchair_00035.html)。在本示例中，可通过`enable_superkernel`开关使能，支持将所有的decode layers优化在一个SuperKernel scope内，实现对任务调度的等待时间和调度开销的优化，提升整体性能。如希望修改SuperKernel scope的范围，可以在`modeling_deepseek.py`中调整，示例如下：
```python
from executor.utils import superkernel_scope

switch = enable_superkernel
scope = f"scope_name"
options = f"scope_options"
with superkernel_scope(switch, scope, options):
    your modules
```
其中`scope`表示当前融合范围的SuperKernel名称，`options`表示SuperKernel编译的自定义选项。关于scope和options的具体描述和使用范围请参考[SuperKernel文档](https://www.hiascend.com/document/detail/zh/Pytorch/710/modthirdparty/torchairuseguide/torchair_00035.html)。

## Benchmark

基于Atlas A3，本实践对DeepSeek-R1 W8A8C8 量化版本进行了性能Benchmark测试。
|Quant Mode| Global Batch Size | Seq Length | Chips | TPOT (ms) | Throughput (tokens/p/s) |
|-------| ----------------- | ---------- | ----- | --------- | ----------------------- |
|W8A8C8 |    6144           | 4096       | 128   | 44.9      |2138                     |

> 注：性能数据基于 MTP1 与 perfect eplb 配置采集，平均 1 个 draft token 中 accept token 为 0.7 个。
---
## 附录
[环境部署以及样例执行](../../../models/deepseek_r1/README.md)
