## SlidingWindowAttention

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 推理系列产品</term>   | √  |
|<term>Atlas A3 推理系列产品</term>   | √  |
|<term>Ascend 950PR/Ascend 950DT 推理系列产品</term>   | √  |

## 功能说明

- API功能：`SlidingWindowAttention`算子旨在完成以下公式描述的Attention计算，支持Sliding Window Attention。
- 计算公式：
  
  $$
  O = \text{softmax}(Q@\tilde{K}^T \cdot \text{softmax\_scale})@\tilde{V}
  $$
  
  其中$\tilde{K}=\tilde{V}$为基于kv_cache、kv_win等入参控制的实际参与计算的 $KV$。

## 函数原型

```
torch.ops.pypto.sliding_window_attention(
    q,
    ori_block_table,
    ori_kv,
    seqused_kv,
    sinks,
    win_size,
    mask,
    cu_seqlens_q
) -> (Tensor)
```

## 参数说明

- **q**（`Tensor`）：必选参数，对应公式中的$Q$，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[T1, N1,D]，其中N1仅支持64，D仅支持512。
- **ori_block_table**（`Tensor`）：必选参数，表示PageAttention中oriKvCache存储使用的block映射表。数据格式支持ND，数据类型支持`int32`，shape为2维，其中第一维长度为B，第二维长度不小于所有batch中最大的S2对应的block数量，即S2\_max / block\_size向上取整， block\_size仅支持128。
- **ori_kv**（`Tensor`）：必选参数，为原始的KV，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[block\_num1, block\_size, N2, D]，其中block\_num1为PageAttention时block总数，block\_size为一个block的token数，仅支持128，N2仅支持1。
- **seqused_kv**（`Tensor`）：必选参数，表示不同Batch中`ori_kv`的输入样本序列长度S2，维度为B，数据格式支持ND，数据类型支持`int32`。
- **sinks**（`Tensor`）：必选参数，注意力下沉tensor，数据格式支持ND，数据类型支持`float32`，shape为[N1]。
- **win_size**（`Int`）：必选参数，窗口大小，数据类型支持`int32`，仅支持128。
- **mask**（`Tensor`）：必选参数，计算过程中使用到的掩码，数据类型支持`bool`，生成方式固定，调用get_mask方法，shape为[4 * N1, 4 * block\_size]，其中N1仅支持64，block\_size仅支持128。
- **cu_seqlens_q**（`Tensor`）：必选参数，表示不同Batch中`q`的有效token数，维度为B+1，大小为参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须>=前一个元素的值，数据类型支持`int32`。

## 返回值说明

- **atten\_out**（`Tensor`）：注意力计算结果。数据格式支持ND，数据类型支持`bfloat16`，shape为[T1, N1, D]。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持aclgraph模式。
- 参数q中的D和ori_kv的D值相等为512。
- 参数q、ori_kv的数据类型必须保持一致。
- block_size支持128。

## 调用方法

```
python3 ops/pypto_python/example/test_sliding_window_attention_pypto.py
```