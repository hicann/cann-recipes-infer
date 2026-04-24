# QuantLightningIndexerMetadata

## 产品支持情况
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 推理系列产品</term>   | √  |

## 功能说明

-   API功能：QuantLightningIndexerMetadata是QuantLightningIndexer的前置算子，通过AICPU为QuantLightningIndexer算子生成分核结果，包括每个核需要处理的数据的起始点、结束点等内容，随后，QuantLightningIndexer根据该分核结果进行实际计算。

-   主要计算过程为：
    1. 获取每个`batch`的基本块大小，并计算负载。
    2. 计算所有`batch`的总负载和总的基本块个数。
    3. 为每个核分配负载，并记录分核结果，分核结果包括每个核需要处理的数据的起始点、结束点等内容。

## 函数原型

```
custom.npu_quant_lightning_indexer_metadata(num_heads_q, num_heads_k, head_dim, query_quant_mode, key_quant_mode, *, actual_seq_lengths_query=None, actual_seq_lengths_key=None, batch_size=0, max_seqlen_q=0, max_seqlen_k=0, layout_query='BSND', layout_key='BSND', sparse_count=2048, sparse_mode=3, pre_tokens=9223372036854775807, next_tokens=9223372036854775807, cmp_ratio=1, device='npu:0') -> Tensor
```
- Transformer QuantLightningIndexerMetadata算子实现参考: [QuantLightningIndexerMetadata](https://gitcode.com/cann/ops-transformer/tree/master/experimental/attention/quant_lightning_indexer_metadata)
## 参数说明
-   **num_heads_q**（`int`）：必选参数，表示公式中的$Q$的多头数，目前仅支持64。

-   **num_heads_k**（`int`）：必选参数，表示公式中的$\tilde{K}$的多头数，目前仅支持1。

-   **head_dim**（`int`）：必选参数，表示注意力头的维度。

-   **query\_quant\_mode**（`Tensor`）：必选参数，用于标识query的量化模式，当前支持Per-Token-Head量化模式，当前仅支持传入0。

-   **key\_quant\_mode**（`Tensor`）：必选参数，用于标识输入key的量化模式，当前支持Per-Token-Head量化模式，当前仅支持传入0。

- <strong>*</strong>：代表其之前的参数是位置相关的，必须按照顺序输入；之后的参数是可选参数，位置无关，不赋值会使用默认值。

-   **actual\_seq\_lengths\_query**（`Tensor`）：可选参数，表示不同Batch中`query`的有效token数，数据类型支持`int32`。如果不指定seqlen可传入None，表示和`query`的shape的S长度相同。该入参中每个Batch的有效token数不超过`query`中的维度S大小且不小于0。支持长度为B的一维tensor。<br>当`layout_query`为TND时，该入参必须传入，且以该入参元素的数量作为B值，该入参中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。不能出现负值。

-   **actual\_seq\_lengths\_key**（`Tensor`）：可选参数，表示不同Batch中压缩前原始`key`的有效token数，数据类型支持`int32`。如果不指定seqlen可传入None，表示和key的shape的S长度相同。该参数中每个Batch的原始有效token数除以压缩率后不超过`key`中的维度S大小且不小于0，支持长度为B的一维tensor。<br>当`layout_key`为TND或PA_BSND时，该入参必须传入，`layout_key`为TND，该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。

-   **batch\_size**（`int`）：可选参数，表示输入样本批量大小，默认值为None。

-   **max\_seqlen\_q**（`int`）：可选参数，表示每个Batch中的`q`的有效token数。

-   **max\_seqlen\_k**（`int`）：可选参数，表示每个Batch中的`k`的有效token数。

-   **cmp\_ratio**（`int`）：可选参数，用于稀疏计算，表示key的压缩倍数。数据类型支持`int32`。支持1/2/4/8/16/32/64/128，默认值为1。

-   **layout\_query**（`str`）：可选参数，用于标识输入`query`的数据排布格式，当前支持BSND、TND，默认值"BSND"。

-   **layout\_key**（`str`）：可选参数，用于标识输入`key`的数据排布格式，当前仅支持默认值PA_BSND。

-   **sparse\_count**（`int`）：可选参数，代表topK阶段需要保留的block数量，支持[1, 2048]，默认值为2048，数据类型支持`int32`。

-   **sparse\_mode**（`int`）：可选参数，表示sparse的模式，支持0/3，默认值为3，数据类型支持`int32`。 sparse\_mode为0时，代表defaultMask模式。sparse\_mode为3时，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。

-   **pre\_tokens**（`int`）：可选参数，用于稀疏计算，表示attention需要和前几个Token计算关联。数据类型支持`int64`，仅支持默认值2^63-1。

-   **next\_tokens**（`int`）：可选参数，用于稀疏计算，表示attention需要和后几个Token计算关联。数据类型支持`int64`，仅支持默认值2^63-1。

-   **device**（`str`）：可选参数，用于获取设备信息，默认值为None。

## 返回值说明
-   **metadata**（`Tensor`）：每个cube核上LightningIndexer计算任务的Batch、Head、以及 Q 和 K 的分块的索引，以及每个vector核上LightningDecode的规约任务索引。

## 约束说明
-   该接口支持推理场景下使用。
-   该接口支持aclgraph模式。
-   Tensor不能全传None。

## Atlas A3 推理系列产品 调用示例
- 支持单算子模式调用和aclgraph模式调用，作为QuantLightningIndexer算子的前序算子，调用示例见[QuantLightningIndexer调用示例](./custom-npu_quant_lightning_indexer.md)。
