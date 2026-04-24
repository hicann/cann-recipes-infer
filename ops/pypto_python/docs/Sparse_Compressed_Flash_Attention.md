# SparseCompressedFlashAttention

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 推理系列产品</term>   | √  |
|<term>Atlas A3 推理系列产品</term>   | √  |
|<term>Ascend 950PR/Ascend 950DT 推理系列产品</term>   | √  |

## 功能说明

- API功能：`SparseCompressedFlashAttention`算子旨在完成以下公式描述的Attention计算，支持Sparse Compressed Attention。
- 计算公式：
  
  $$
  O = \text{softmax}(Q@\tilde{K}^T \cdot \text{softmax\_scale})@\tilde{V}
  $$
  
  其中$\tilde{K}=\tilde{V}$为基于ori_kv、cmp_kv以及cmp_kv等入参控制的实际参与计算的 $KV$。

## 函数原型

```
torch.ops.pypto.sparse_compress_flash_attention(
    query,
    q_act_seqs,
    ori_kv,
    cmp_kv,
    ori_block_table,
    cmp_block_table,
    atten_sink,
    seqused_kv,
    cmp_sparse_indices,
    softmax_scale,
    win_size,
    cmp_ratio
) -> (Tensor)
```

## 参数说明

- **query**（`Tensor`）：必选参数，对应公式中的$Q$，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`。shape为[T1*N1,D]，其中，N1仅支持64。
- **q_act_seqs**（`Tensor`）：必选参数，在`layout_query`为TND时生效。表示不同Batch中`q`的有效token数，维度为B+1，大小为参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须>=前一个元素的值，数据类型支持`int32`。
- **ori_kv**（`Tensor`）：必选参数，对应公式中的$\tilde{K}和\tilde{V}$的一部分，为原始不经压缩的KV，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，`layout_kv`为PA_ND时shape为[block\_num1* ori\_block\_size, KV\_N*D]，其中block\_num1为PageAttention时block总数，ori\_block\_size为一个block的token数，ori\_block\_size取值为128，KV_N仅支持1。
- **cmp_kv**（`Tensor`）：必选参数，对应公式中的$\tilde{K}和\tilde{V}$的一部分，为经过压缩的KV，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，`layout_kv`为PA_ND时shape为[block\_num2* cmp\_block\_size, KV\_N*D]，其中block\_num2为PageAttention时block总数，cmp\_block\_size为一个block的token数，cmp\_block\_size取值为128。
- **ori_block_table**（`Tensor`）：必选参数，表示PageAttention中oriKvCache存储使用的block映射表。数据格式支持ND，数据类型支持`int32`，shape为2维，其中第一维长度为B，第二维长度不小于所有batch中最大的S2对应的block数量，即S2\_max / block\_size向上取整。
- **cmp_block_table**（`Tensor`）：必选参数，表示PageAttention中cmpKvCache存储使用的block映射表。数据格式支持ND，数据类型支持`int32`，shape为2维，其中第一维长度为B，第二维长度不小于所有batch中最大的S3对应的block数量，即S3\_max / block\_size向上取整。
- **atten_sink**（`Tensor`）：必选参数，注意力下沉tensor，数据格式支持ND，数据类型支持`float32`，shape为[N1]。
- **seqused_kv**（`Tensor`）：必选参数，表示不同Batch中`ori_kv`实际参与运算的token数，维度为B，数据格式支持ND，数据类型支持`int32`，不输入则所有token均参与运算。
- **cmp_sparse_indices**（`Tensor`）：必选参数，代表离散取cmpKvCache的索引，不支持非连续，数据格式支持ND，数据类型支持`int32`。当`layout_query`为TND时，shape需要传入[Q\_T * KV\_N, K2]，其中K2为对`cmp_kv`一次离散选取的token数，K2仅支持512。
- **softmax_scale**（`double`）：必选参数，代表缩放系数，作为q与ori_kv和cmp_kv矩阵乘后Muls的scalar值，数据类型支持`float`。
- **win_size**（`int`）：必选参数，窗口大小，数据类型支持int32，仅支持128。
- **cmp_ratio**（`int`）：必选参数，表示对ori_kv的压缩率，数据类型支持`int`，数据支持4。

## 返回值说明

- **attention\_out**（`Tensor`）：公式中的输出。数据格式支持ND，数据类型支持`bfloat16`。shape为[T1*N1,D]。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持aclgraph模式。
- 参数q中的D和ori_kv、cmp_kv的D值相等为512。
- 参数q、ori_kv、cmp_kv的数据类型必须保持一致。
- 为了提高算子性能，当前q、ori_kv、cmp_kv、attention_out进行了高维合轴处理。
- 仅支持TND格式。
- block_size支持128。

## 调用方法

```
python3 ops/pypto_python/example/test_sparse_compressed_flash_attention_pypto.py
```
