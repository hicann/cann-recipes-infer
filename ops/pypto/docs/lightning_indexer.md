# custom.npu\_lightning\_indexer\_pto

## 产品支持情况
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 系列产品</term>   | √  |

## 功能说明

-   算子功能：用于 Deepseek IndexerAttention 中根据 Indexer 得到的 query，key，weights 计算得到索引。

Lightning Indexer 模块采用一种类似MLP的多查询注意力（Multi-Query Attention）机制来计算索引得分：

$$
I_{i, j} = \sum_h w_h^i \cdot \text{ReLU}(\bold{q}_{index, h}^i \cdot \bold{k}_{index}^j)
$$

其中 $(w_1^i, \dots, w_{N_h}^i)^T = \bold{W}_{bias}\bold{x}_i$ 表示与查询相关的、按注意力头的权重。在实际实现中，我们于 Indexer Prolog 模块中计算 $w_h^i$。

对于每个查询 token $\bold{x}_i$，基于相关性得分 $I_{i,j}$，我们仅计算得分最高的前 $k$ 个缓存项所对应的索引。

## 函数原型

```
custom.npu_lightning_indexer_pto(query, key, weights, *, actual_seq_lengths_query=None, actual_seq_lengths_key=None, block_table=None, layout_query='BSND', layout_key='PA_BSND', sparse_count=2048, sparse_mode=3) -> Tensor
```

## 参数说明

>**说明：**<br>
>
-   **query**（`Tensor`）：表示 query，必选参数，不支持非连续的Tensor，数据格式支持ND，数据类型支持`bfloat16`。

-   **key**（`Tensor`）：表示 key，必选参数必选参数，不支持非连续的Tensor，数据格式支持ND，数据类型支持`bfloat16`。

-   **weights**（`Tensor`）：表示 weights，必选参数，不支持非连续的Tensor，数据格式支持ND，数据类型支持`bfloat16`。

- <strong>*</strong>：代表其之前的参数是位置相关的，必须按照顺序输入，属于必选参数；其之后的参数是键值对赋值，与位置无关，属于可选参数（不传入会使用默认值）。

-   **actual\_seq\_lengths\_query**（`Tensor`）：表示不同 Batch 中 `query` 的有效 seqlen，可选参数，表示不同Batch中`query`的有效seqlen，数据类型支持`int32`。

-   **actual\_seq\_lengths\_key**（`Tensor`）：表示不同 Batch 中 `key` 和 `value` 的有效 seqlen，可选参数，表示不同Batch中`key`的有效seqlenK，数据类型支持`int32`。当前仅支持必传。

-   **block\_table**（`Tensor`）：表示 PagedAttention 中 KV 存储使用的 block 映射表，可选参数，表示PageAttention中KV存储使用的block映射表，数据类型支持`int32`。数据格式支持ND。当前仅支持必传。

-   **layout\_query**（`str`）：可选参数，用于标识输入`query`的数据排布格式，默认值"BSND"。当前仅支持 "BSND"。

-   **layout\_key**（`str`）：可选参数，用于标识输入`key`的数据排布格式，默认值"PA_BSND"。当前仅支持 "PA_BSND"。

-   **selected\_count**（`int`）：可选参数，代表topK阶段需要保留的block数量，数据类型支持`int32`。

-   **sparse\_mode**（`int`）：可选参数，表示sparse的模式，支持0/3，数据类型支持`int32`。

    -   sparse\_mode为0时，代表defaultMask模式。
    -   sparse\_mode为3时，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。

## 返回值说明

-   **out**（`Tensor`）：公式中的输出，数据类型支持`bfloat16`。数据格式支持ND。

## 约束说明

-   该接口支持推理场景下使用。
-   该接口支持图模式（PyTorch 2.1版本）。
-   该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。

## 算子代码执行示例
-   算子源码执行参考[test_lightning_indexer_pto.py](../examples/test_lightning_indexer_pto.py)