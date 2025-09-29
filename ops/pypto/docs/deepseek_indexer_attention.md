# custom.npu_sparse_attention_pto

## 产品支持情况
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 系列产品</term>   | √  |

## 功能说明

-   算子功能：用于 DeepSeek-V3.2-Exp 的 Attention 计算

### 简介
我们对 DeepSeek-V3.2-Exp 进行了拆解，该模型采用了一种细粒度稀疏注意力机制，将基于 token 级别的细粒度sparse attention与Lightning Indexer相结合，并将其分解为四个模块：MLA Prolog, Indexer Prolog, Lightning Indexer and Sparse Flash Attention。

### MLA prolog
MLA Prolog 模块将hidden状态 $\bold{X}$ 转换为查询投影 $\bold{q}$、键投影 $\bold{k}$ 和值投影 $\bold{v}$，其结构与 DeepSeek V3 的架构一致。在解码阶段，采用了权重吸收技术。

### indexer prolog

Indexer Prolog 模块将hidden状态 $\bold{X}$ 投影为查询索引 $\bold{q}_{index, h}$ 和键索引 $\bold{k}_{index}$ 的表示。该变换遵循如下公式：旋转位置嵌入（RoPE）仅应用于 $\bold{q}_{index, h}$ 和 $\bold{k}_{index}$ 的头维度的后半部分。

$$
\bold{q}_{index, h} = \text{RoPE}\left(\left(\text{RMSNorm}(\bold{X} \cdot \bold{W}_{qa})\right) \cdot \bold{W}_{qb}\right)
$$

$$
\bold{k}_{index} = \text{RoPE}\left(\text{LayerNorm}\left(\bold{X}\cdot \bold{W}_k \right)\right)
$$

### lightning indexer

Lightning Indexer 模块采用一种类MLP的多查询注意力（Multi-Query Attention）机制来计算索引得分：

$$
I_{i, j} = \sum_h w_h^i \cdot \text{ReLU}(\bold{q}_{index, h}^i \cdot \bold{k}_{index}^j)
$$

where $(w_1^i, \dots,w_{N_h}^i)^T = \bold{W}_{bias}\bold{x}_i$ represents query-dependent head-wise weights. In practice, we calculate $w_h^i$ in Indexer Prolog module.

### sparse flash attention

对于每个查询 token $\bold{x}_i$，索引模块会为每个键值缓存项（表示键值对或 MLA 潜在表示）计算一个相关性得分 $I_{i,j}$。然后，通过将注意力机制应用于查询 token $\bold{x}_i$ 以及得分最高的前 $k$ 个缓存项，来计算输出 $\bold{o}_i$：

$$
\bold{o}_i = \text{Attn}(\bold{x}_i, \{\bold{c}_j | j \in \text{Top-k}(\bold{I}_{i, :})\})
$$

## 函数原型

```
custom.npu_sparse_attention_pto(x, w_dq, w_uq_qr, w_uk, w_dkv_kr, gamma_cq, gamma_ckv, sin, cos, cache_index, kv_cache, kr_cache, block_table, act_seqs, w_idx_qb, w_idx_k, w_idx_proj, in_gamma_k, in_beta_k,index_k_cache) -> Tensor
```

## 参数说明

>**说明：**<br>
>

-   **x**（`Tensor`）：表示hidden状态，必选参数，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`。

-   **w_dq**（`Tensor`）：表示计算 query 的下投影权重，必选参数，不支持非连续的 Tensor，数据格式支持 NZ，数据类型支持`bfloat16`。

-   **w_uq_qr**（`Tensor`）：表示计算 query 的上投影权重，必选参数，不支持非连续的 Tensor，数据格式支持 NZ，数据类型支持`bfloat16`。

-   **w_uk**（`Tensor`）：表示权重吸收中计算 query 的权重，必选参数，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`。

-   **w_dkv_kr**（`Tensor`）：表示权重吸收中计算 ckv 的权重，必选参数，不支持非连续的 Tensor，数据格式支持 NZ，数据类型支持`bfloat16`。

-   **gamma_cq** (`Tensor`): 表示 query 的 rmsnorm 缩放，必选参数，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`。

-   **gamma_ckv** （`Tensor`）：表示 ckv 的 rmsnorm 缩放，必选参数，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`。

-   **sin** （`Tensor`）：表示用于 RoPE 的 sin，必选参数，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`。

-   **cos** （`Tensor`）：表示用于 RoPE 的 cos，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`。

-   **cache_index** （`Tensor`）：表示更新 kvCache，krCache 和 idxKCache 的位置，必选参数，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`。

-   **kv_cache** （`Tensor`）：kv cache，必选参数，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`。

-   **kr_cache** （`Tensor`）：kr cache，必选参数，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`。

-   **block_table** （`Tensor`）：表示 PagedAttention 中 KV 存储使用的 block 映射表，必选参数，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`。

-   **act_seqs** （`Tensor`）：表示不同 Batch 中 `key` 和 `value` 的有效 seqlen，必选参数，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`。

-   **w_idx_qb** （`Tensor`）：表示 Indexer 计算 query 的权重，必选参数，不支持非连续的 Tensor，数据格式支持 NZ，数据类型支持`bfloat16`。

-   **w_idx_k** （`Tensor`）：表示 Indexer 计算 key 的权重，必选参数，不支持非连续的 Tensor，数据格式支持 NZ，数据类型支持`bfloat16`。

-   **w_idx_proj** （`Tensor`）：表示 Indexer 计算 weights 的权重，必选参数，不支持非连续的 Tensor，数据格式支持 NZ，数据类型支持`bfloat16`。

-   **in_gamma_k** （`Tensor`）：表示 Indexer 计算 key 的 layernorm 缩放，必选参数，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`。

-   **in_beta_k** （`Tensor`）：表示 Indexer 计算 key 的 layernorm 偏移，必选参数，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`。

-   **index_k_cache** （`Tensor`）：表示 Indexer 中 key 的缓存，必选参数，不支持非连续的 Tensor，数据格式支持 ND，数据类型支持`bfloat16`。

## 返回值说明

-   **out**（`Tensor`）：公式中的输出，数据格式支持 ND，数据类型支持`bfloat16`。

## 约束说明

-   该接口支持推理场景下使用。
-   该接口支持图模式（PyTorch 2.1版本）。
-   该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。

## 算子代码执行示例
-   算子源码执行参考[test_deepseek_indexer_attention.py](../examples/test_deepseek_indexer_attention.py)