# SparseAttnSharedkvMetadata

## 产品支持情况
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 推理系列产品</term>   | √  |

## 功能说明
- API功能：`SparseAttnSharedkvMetadata`算子旨在生成一个任务列表，包含每个AIcore的Attention计算任务的起止点的Batch、Head、以及 Q 和 K 的分块的索引，供后续`SparseAttnSharedkv`算子使用。

## 函数原型

```
custom.npu_sparse_attn_sharedkv_metadata(num_heads_q, num_heads_kv, head_dim, *, cu_seqlens_q=None, cu_seqlens_ori_kv=None, cu_seqlens_cmp_kv=None, 
seqused_q=None, seqused_kv=None, batch_size=0, max_seqlen_q=0, max_seqlen_kv=0, ori_topk=0, cmp_topk=0, cmp_ratio=-1, ori_mask_mode=4, 
cmp_mask_mode=3, ori_win_left=127, ori_win_right=0, layout_q='BSND', layout_kv='PA_ND', has_ori_kv=True, has_cmp_kv=True, device='npu:0') -> Tensor
```
- Transformer SparseAttnSharedkvMetadata 算子实现参考: [SparseAttnSharedkvMetadata](https://gitcode.com/cann/ops-transformer/tree/master/experimental/attention/sparse_attn_sharedkv_metadata)

## 参数说明
-   **num_heads_q**（`int`）：必选参数，表示公式中的$Q$的多头数，目前仅支持64。

-   **num_heads_kv**（`int`）：必选参数，表示公式中的$\tilde{K}$和$\tilde{V}$的多头数，目前仅支持1。

-   **head_dim**（`int`）：必选参数，表示注意力头的维度。

- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。

-   **cu\_seqlens\_q**（`Tensor`）：可选参数，当`layout_query`为TND时，表示不同Batch中`q`的有效token数，维度为B+1，大小为参数中每个元素的值表示目前batch与之前所有batch的token数总和，即前缀和，数据类型支持`int32`。

-   **cu\_seqlens\_ori\_kv**（`Tensor`）：可选参数，当`layout_kv`为TND时，表示不同Batch中`ori_kv`的有效token数，维度为B+1，大小为参数中每个元素的值表示目前batch与之前所有batch的token数总和，即前缀和，数据类型支持`int32`。**目前layout_kv仅支持PA_ND，故设置此参数无效。**

-   **cu\_seqlens\_cmp\_kv**（`Tensor`）：可选参数，当`layout_kv`为TND时，表示不同Batch中`cmp_kv`的有效token数，维度为B+1，大小为参数中每个元素的值表示目前batch与之前所有batch的token数总和，即前缀和，数据类型支持`int32`。**目前layout_kv仅支持PA_ND，故设置此参数无效。**

-   **seqused\_q**（`Tensor`）：可选参数，表示不同Batch中`q`实际参与运算的token数，维度为B，数据格式支持ND，数据类型支持`int32`，不输入则所有token均参与运算。**目前暂不支持指定该参数。**

-   **seqused\_kv**（`Tensor`）：可选参数，表示不同Batch中`ori_kv`实际参与运算的token数，维度为B，数据格式支持ND，数据类型支持`int32`，不输入则所有token均参与运算。

-   **batch\_size**（`int`）：可选参数，表示输入样本批量大小，默认值为None。

-   **max\_seqlen\_q**（`int`）：可选参数，表示所有batch中`q`的最大有效token数。

-   **max\_seqlen\_kv**（`int`）：可选参数，表示所有batch中`ori_kv`的最大有效token数。

-   **ori_topk**（`int`）：可选参数，表示通过QLI算法从`ori_kv`中筛选出的关键稀疏token的个数。**目前暂不支持指定该参数**，默认值为None。

-   **cmp_topk**（`int`）：可选参数，表示通过QLI算法从`cmp_kv`中筛选出的关键稀疏token的个数，目前仅支持512，默认值为None。
    
-   **cmp\_ratio**（`int`）：可选参数，表示对`ori_kv`的压缩率，数据范围支持4/128，默认值为None。

-   **ori\_mask\_mode**（`int`）：可选参数，表示`q`和`ori_kv`计算的mask模式，目前仅支持输入默认值4，代表band模式的mask。

-   **cmp\_mask\_mode**（`int`）：可选参数，表示`q`和`cmp_kv`计算的mask模式，目前仅支持输入默认值3，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。

-   **ori\_win\_left**（`int`）：可选参数，表示`q`和`ori_kv`计算中`q`对过去token计算的数量，目前仅支持默认值127。

-   **ori\_win\_right**（`int`）：可选参数，表示`q`和`ori_kv`计算中`q`对未来token计算的数量，目前仅支持默认值0。

-   **layout\_q**（`str`）：可选参数，表示输入`q`的数据排布格式，默认值为BSND，目前支持传入BSND和TND。

-   **layout\_kv**（`str`）：可选参数，表示输入`ori_kv`和`cmp_kv`的数据排布格式，目前仅支持传入默认值PA_ND（PageAttention）。

-   **has\_ori\_kv**（`bool`）：可选参数，表示是否传入`ori_kv`，默认值为true。

-   **has\_cmp\_kv**（`bool`）：可选参数，表示是否传入`cmp_kv`，默认值为true。

-   **device**（`str`）：可选参数，用于获取设备信息，默认值为None。

## 返回值说明

-   **metadata**（`Tensor`）：每个cube核上FlashAttention计算任务的Batch、Head、以及 Q 和 K 的分块的索引，以及每个vector核上FlashDecode的规约任务索引。

## 约束说明

-   该接口支持推理场景下使用。
-   该接口支持aclgraph模式。
-   Tensor不能全传None。

## Atlas A3 推理系列产品 调用示例
- 支持单算子模式调用和aclgraph模式调用，作为SparseAttnSharedkv算子的前序算子，调用示例见[SparseAttnSharedkv调用示例](./custom-npu_sparse_attn_sharedkv.md)。