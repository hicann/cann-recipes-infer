# CompressedFlashAttention

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 推理系列产品</term>   | √  |
|<term>Atlas A3 推理系列产品</term>   | √  |
|<term>Ascend 950PR/Ascend 950DT 推理系列产品</term>   | √  |

## 功能说明

- API功能：`CompressedFlashAttention`算子旨在完成以下公式描述的Attention计算，支持Compressed Attention。
- 计算公式：
  
  $$
  O = \text{softmax}(Q@\tilde{K}^T \cdot \text{softmax\_scale})@\tilde{V}
  $$
  
  其中$\tilde{K}=\tilde{V}$为基于kv_cache、kv_win等入参控制的实际参与计算的 $KV$。

## 函数原型

```
torch.ops.pypto.compress_flash_attention(
    q,
    cmp_kv,
    sinks,
    cmp_block_table,
    seqused_kv,
    ori_kv,
    ori_block_table,
    cmp_ratio
) -> (Tensor)
```

## 参数说明


- **q**（`Tensor`）：必选参数，对应公式中的$Q$，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`。shape为[B*S1,N1,D]，其中N1仅支持64。
- **cmp_kv**（`Tensor`）：必选参数，对应公式中的$\tilde{K}和\tilde{V}$的一部分，为经过压缩的KV，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，`layout_kv`为PA_ND时shape为[block\_num, cmp\_block\_size, KV\_N, D]，其中block\_num2为PageAttention时block总数，cmp\_block\_size为一个block的token数，cmp\_block\_size取值为16的倍数，最大支持1024。
- **sinks**（`Tensor`）：必选参数，注意力下沉tensor，数据格式支持ND，数据类型支持`float32`，shape为[N1]。
- **cmp_block_table**（`Tensor`）：必选参数，表示PageAttention中cmpKvCache存储使用的block映射表。数据格式支持ND，数据类型支持`int32`，shape为2维，其中第一维长度为B，第二维长度不小于所有batch中最大的S3对应的block数量，即S3\_max / block\_size向上取整。
- **seqused_kv**（`Tensor`）：必选参数，表示不同Batch中`ori_kv`实际参与运算的token数，维度为B，数据格式支持ND，数据类型支持`int32`。
- **ori_kv**（`Tensor`）：必选参数，对应公式中的$\tilde{K}和\tilde{V}$的一部分，为原始不经压缩的KV，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[block\_num1, ori\_block\_size, KV\_N, D]。
- **ori_block_table**（`Tensor`）：必选参数，表示PageAttention中oriKvCache存储使用的block映射表。数据格式支持ND，数据类型支持`int32`，shape为2维，其中第一维长度为B，第二维长度不小于所有batch中最大的S2对应的block数量，即S2\_max / block\_size向上取整。
- **cmp_ratio**（`int`）：必选参数，表示对ori_kv的压缩率，数据类型支持`int`，数据支持128。

## 返回值说明

- **attention\_out**（`Tensor`）：公式中的输出。数据格式支持ND，数据类型支持`bfloat16`，shape为[B*S1,N1,D]。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持aclgraph模式。
- 参数q中的D和seqused_kv、kv_cache的D值相等为512。
- 参数seqused_kv、kv_cache的数据类型必须保持一致。
- 本接口仅支持decode场景，不支持prefill场景。
- block_size支持128。

## 调用方法

```
python3 ops/pypto_python/example/test_compressed_flash_attention_pypto.py
```
