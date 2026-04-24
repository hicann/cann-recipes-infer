## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 推理系列产品</term>   | √  |
|<term>Atlas A3 推理系列产品</term>   | √  |
|<term>Ascend 950PR/Ascend 950DT 推理系列产品</term>   | √  |

## 功能说明

- API功能：Compressor将每4或128个token的 KV cache 压缩成一个，然后每个token与这些压缩的 KV cache进行 DSA 计算。在长序列的情况下，Compressor可以有效地减少计算开销。
- 主要计算过程为：
  
  1. 将输入$X$与$W^{KV}$做Matmul运算得到$kv\_state$，将输入$X$与$W^{Gate}$做Matmul运算后再与$Ape$做Add运算得到$score\_state$，$kv\_state$与$score\_state$根据输入的start_pos完成更新。
  2. 对$kv\_state$和$score\_state$进行数据重排，再对$score\_state$进行softmax运算将softmax结果与$kv\_state$做Mul计算，后进行Reducesum运算。
  3. 根据输入数据norm_weight、rope_sin、rope_cos，进行 RmsNorm 和 ROPE 运算，根据 rotate 决定是否需要额外进行 Hadamard Transform，得到$cmp\_kv$结果输出。

## 函数原型

```
torch.ops.pypto.compressor(
    x,
    kv_state,
    score_state,
    kv_block_table,
    state_block_table,
    sin,
    cos,
    wkv,
    wgate,
    ape,
    weight,
    hadamard,
    start_pos,
    ratio,
    rope_head_dim,
    rotate
) -> (Tensor)
```

## 参数说明


- **x**（`Tensor`）：必选参数，表示原始不经压缩的数据，对应公式中的$X$。不支持非连续，数据格式支持$ND$，数据类型支持`bfloat16`。支持输入shape[B,S,H]。
- **kv\_state**（`Tensor`）：必选参数，表示kv\_state的历史数据，对应公式中的$kv\_state$。不支持非连续，数据格式支持$ND$，数据类型支持`float32`。支持输入shape[block_num,block_size,coff*D]。
- **score\_state**（`Tensor`）：必选参数，表示score\_state中的历史数据, 对应公式中的$score\_state$。不支持非连续，数据格式支持$ND$，数据类型支持`float32`。支持输入shape[block_num,block_size,coff*D]。
- **kv\_block\_table**（`Tensor`）：必选参数，表示kv\_state中的历史数据的page table。不支持非连续，数据格式支持$ND$，数据类型支持`int32`。支持输入shape[B, ceil(max_S/block_size)]。
- **state\_block\_table**（`Tensor`）：必选参数，表示score\_state中的历史数据的page table。不支持非连续，数据格式支持$ND$，数据类型支持`int32`。支持输入shape[B, ceil(max_S/block_size)]。
- **sin**（`Tensor`）：必选参数，表示Rope计算的权重系数。数据类型支持`bfloat16`。支持输入shape[min(T,T//ratio+B),rope_head_dim]。
- **cos**（`Tensor`）：必选参数，表示Rope计算的权重系数。数据类型支持`bfloat16`。支持输入shape[min(T,T//ratio+B),rope_head_dim]。
- **wkv**（`Tensor`）：必选参数，表示KV和压缩权重的权重参数，对应公式中的$W^{KV}$。不支持非连续，数据格式支持$ND$，数据类型支持`bfloat16`。支持输入shape[coff*D,H]。
- **wgate**（`Tensor`）：必选参数，表示KV和压缩权重的权重参数，对应公式中的$W^{Gate}$。不支持非连续，数据格式支持$ND$，数据类型支持`bfloat16`。支持输入shape[coff*D,H]。
- **ape**（`Tensor`）：必选参数，表示输入的positional biases，对应公式中的$Ape$。不支持非连续，数据格式支持$ND$，数据类型支持`float32`。支持输入shape[ratio,coff*D]。
- **weight**（`Tensor`）：必选参数，表示计算RmsNorm时的权重系数。数据类型支持`bfloat16`。支持输入shape[D,]。
- **hadamard**（`Tensor`）：可选参数，表示 Hadamard Transform 的权重矩阵。不支持非连续，数据格式支持$ND$，数据类型支持`bfloat16`。支持输入shape[D, D]。
- **start\_pos**（`Tensor`）：可选参数，表示计算起始位置。不支持非连续，数据格式支持$ND$，数据类型支持`int32`。支持输入shape[B,]。当输入为None时，表示从0开始进行计算。
- **ratio**（`int`）：必选参数，表示数据压缩率。支持4/128。
- **rope\_head\_dim**（`int`）：必选参数，表示rope_cos和rope_sin的hidden层最小单元。目前仅支持64。
- **rotate**（`bool`）：必选参数，表示是否需要额外进行 Hadamard Transform。

## 返回值说明

- **out**（`Tensor`）：必选输出，表示压缩后的数据。不支持非连续，数据格式支持$ND$。数据类型支持`bfloat16`。支持输出shape[min(T, T // ratio + B), D]。不压缩的条目的输出数据值是零。

## 约束说明

- 该接口支持 B 泛化。
- S 支持 1/2/3/4。
- D 支持128/512。
- H 支持4096。
- block_size 支持 128。

## 调用方法

```
python3 ops/pypto_python/example/test_compressor_pypto.py
```
