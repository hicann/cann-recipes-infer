# custom.npu\_gather\_selection\_kv\_cache<a name="ZH-CN_TOPIC_0000001979260729"></a>

## 产品支持情况 <a name="zh-cn_topic_0000001832267082_section14441124184110"></a>
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 推理系列产品</term>   | √  |

## 功能说明<a name="zh-cn_topic_0000001832267082_section14441124184110"></a>

`GatherSelectionKvCache`算子通过topk的稀疏索引从原始的full_kv_cache中进行gather操作，拼接成一块新的selection_kv_cache，支持DeepSeek模型在超大token场景下kv_cache的offload特性。其中，full_kv_cache支持host侧或device侧内存。


## 函数原型<a name="zh-cn_topic_0000001832267082_section45077510411"></a>

```
custom.npu_gather_selection_kv_cache(Tensor(a!) selection_k_rope, Tensor(b!) selection_kv_cache, Tensor(c!) selection_kv_block_table, Tensor(d!) selection_kv_block_status, Tensor selection_topk_indices, Tensor full_k_rope,  Tensor full_kv_cache, Tensor full_kv_block_table, Tensor full_kv_actual_seq, Tensor full_q_actual_seq, *,  int selection_topk_block_size=64) -> Tensor
```

## 参数说明<a name="zh-cn_topic_0000001832267082_section112637109429"></a>

>**说明：**<br> 
>
>- BLOCK_SIZE表示每个block中的token数，S_BLOCK_NUM表示selection_kv_cache的block num，S_MAX_BLOCK_NUM表示selection_kv_cache的最大block num，F_BLOCK_NUM表示full_kv_cache的block num，F_MAX_BLOCK_NUM表示full_kv_cache的最大block num。
>- B（Batch Size）表示输入样本批量大小、S（Sequence Nums）表示输入样本序列个数、H（Head Num）表示多头数、TOPK表示选取的token个数。

-   **selection_k_rope**（`Tensor`）：表示被选取的k_rope，必选参数，不支持非连续，数据格式支持ND，计算结果原地更新，数据类型支持`bfloat16`、`float16`、`int8`，shape为：[S_BLOCK_NUM, BLOCK_SIZE, K_ROPE]。数据类型为`int8`时，仅支持输入空tensor。
    
-   **selection_kv_cache**（`Tensor`）：表示被选取的kv_cache，必选参数，不支持非连续，数据格式支持ND，计算结果原地更新，数据类型支持`bfloat16`、`float16`、`int8`，shape为：[S_BLOCK_NUM, BLOCK_SIZE, KV_CACHE]。
    
-   **selection_kv_block_table**（`Tensor`）：表示被选取的kv_cache对应的block table映射表，必选参数，不支持非连续，数据格式支持ND，计算结果原地更新，数据类型支持`int32`，shape为：\[B\*S\*H, S_MAX_BLOCK_NUM\]。

-   **selection_kv_block_status**（`Tensor`）：表示被选取的kv_cache对应的block table的status，记录selection_kv_cache和selection_k_rope中缓存的有效topk对应的id，必选参数，不支持非连续，数据格式支持ND，计算结果原地更新，数据类型支持`int32`。当shape类型为BSND时，shape为：[B, S, H, TOPK + 1]；当shape类型为TND时，shape为：[B*S, H, TOPK + 1]。

-   **selection_topk_indices**（`Tensor`）：表示每个token选出的topk索引，必选参数，不支持非连续，数据格式支持ND，数据类型支持`int32`。当shape类型为BSND时，shape为：[B, S, H, TOPK]；当shape类型为TND时，shape为：[B*S, H, TOPK]。

-   **full_k_rope**（`Tensor`）：表示全量的k_rope，必选参数，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`、`float16`、`int8`，shape为：[F_BLOCK_NUM, BLOCK_SIZE, K_ROPE]。数据类型为`int8`时，仅支持输入空tensor。

-   **full_kv_cache**（`Tensor`）：表示全量的kv_cache，必选参数，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`、`float16`、`int8`，shape为：[F_BLOCK_NUM, BLOCK_SIZE, K_CACHE]。

-   **full_kv_block_table**（`Tensor`）：表示全量的kv_cache对应的block table映射表，必选参数，不支持非连续，数据格式支持ND，数据类型支持`int32`，shape为：[B, F_MAX_BLOCK_NUM]。

-   **full_kv_actual_seq**（`Tensor`）：表示全量的kv_cache中，不同Batch Size对应的有效token数，必选参数，不支持非连续，数据格式支持ND，数据类型支持`int32`，shape为：[B]。

-   **full_q_actual_seq**（`Tensor`）：表示不同Batch Size对应的query的有效token数，必选参数，不支持非连续，数据格式支持ND，数据类型支持`int32`，shape为：[B]。
    
- <strong>*</strong>：代表其之前的参数是位置相关的，必须按照顺序输入，属于必选参数；其之后的参数是键值对赋值，与位置无关，属于可选参数（不传入会使用默认值）。

-   **selection_topk_block_size**（`int`）：可选参数，表示被选取的topk对应的BLOCK_SIZE大小，默认值为64。

## 返回值说明<a name="zh-cn_topic_0000001832267082_section22231435517"></a>

-   **selection_kv_actual_seq**（`Tensor`）：输出selection_kv_actual_seq，数据类型支持`int32`。数据格式支持ND，shape为\[B\*S\*H\]。

## 约束说明<a name="zh-cn_topic_0000001832267082_section12345537164214"></a>

-   该接口支持推理场景下使用。
-   该接口支持aclgraph入图。
-   该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
-   BLOCK_SIZE需要保证能被selection_topk_block_size整除。
-   TOPK ≤ 2048。
-   H当前只支持1。
-   S_BLOCK_NUM ≥ B * S * H * S_MAX_BLOCK_NUM。
-   selection_kv_block_status、selection_topk_indices的shape为'TND'格式时，T需要保证能被B整除。
-   topk大于32时，selection_topk_block_size仅支持为1。

## 调用示例<a name="zh-cn_topic_0000001832267082_section14459801435"></a>

-   详见[test_npu_gather_selection_kv_cache.py](../examples/test_npu_gather_selection_kv_cache.py)