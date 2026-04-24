# MlaProlog

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 推理系列产品</term>   | √  |
|<term>Atlas A3 推理系列产品</term>   | √  |
|<term>Ascend 950PR/Ascend 950DT 推理系列产品</term>   | √  |

## 功能说明

MLA Prolog 模块将hidden states $x$ 转换为 $Query$和 ${Key-Value}$。

## 计算公式

1. $Query(q)$ 的计算
   Query 的计算，包括两次采样和 RmsNorm（其中第二次 RmsNorm 权重恒为 1），最后对 -1 轴的后 rope\_dim 维度进行 inplace interleaved rope 计算：

$$
c^Q = RmsNorm(x @ wq\_a)
$$

$$
q = RmsNorm(c^Q @ wq\_b)
$$

$$
q[..., -rope\_dim:] = ROPE(q[..., -rope\_dim:])
$$

2. $Key-Value(kv)$ 的计算
   kv 的计算，包括一次下采样和 RmsNorm，最后对 -1 轴的后 rope\_dim 维度进行 inplace interleaved rope 计算：

$$
kv = RmsNorm(x @ wkv)
$$

$$
kv[..., -rope\_dim:] = ROPE(kv[..., -rope\_dim:])
$$

## 函数原型

```
torch.ops.pypto.mla_prolog_quant(
    token_x,
    wq_a,
    wq_b,
    wkv,
    rope_cos,
    rope_sin,
    gamma_cq,
    gamma_ckv,
    wq_b_scale
) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

- **token_x**（`Tensor`）：公式中用于计算Query和Key-Value的输入tensor，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`bfloat16`，shape为[t, h]。
- **wq_a**（`Tensor`）：公式中用于计算Query的下采样权重矩阵$wq_a$，数据格式支持NZ/ND，数据类型支持`bfloat16`，shape为[h, q_lora_rank]。
- **wq_b**（`Tensor`）：公式中用于计算Query的上采样权重矩阵$wq_b$，数据格式支持NZ/ND，数据类型支持`int8`，shape为[q_lora_rank, num_heads*head_dim]。
- **wkv**（`Tensor`）：公式中用于计算Key-Value的下采样权重矩阵$wkv$，数据格式支持NZ/ND，数据类型支持`bfloat16`，shape为[h, head_dim]。
- **rope_cos**（`Tensor`）：用于计算旋转位置编码的余弦参数矩阵，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`bfloat16`，shape为[t, rope_dim]。
- **rope_sin**（`Tensor`）：用于计算旋转位置编码的正弦参数矩阵，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`bfloat16`，shape为[t, rope_dim]。
- **gamma_cq**（`Tensor`）：计算$c^Q$的RmsNorm公式中的$\gamma$参数，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`bfloat16`，shape为[q_lora_rank]。
- **gamma_ckv**（`Tensor`）：计算$c^{KV}$的RmsNorm公式中的$\gamma$参数，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`bfloat16`，shape为[head_dim]。
- **wq_b_scale**（`Tensor`）：用于矩阵乘wq_b后反量化操作的per-channel参数，不支持非连续的 Tensor。数据格式支持ND，数据类型支持`float`，shape为[num_heads*head_dim, 1]。


## 返回值说明

- **q_out**（`Tensor`）：公式中Query的输出tensor（对应公式中的$q$），不支持非连续的 Tensor。数据格式支持ND，数据类型支持`bfloat16`，shape为[t, num_heads, head_dim]。
- **kv_out**（`Tensor`）：公式中Key-Value的输出tensor（对应公式中的$kv$），不支持非连续的 Tensor。数据格式支持ND，数据类型支持`bfloat16`，shape为[t, head_dim]。
- **qr_out**（`Tensor`）：公式中Query做完第一次rmsnorm和quant后的输出tensor（对应公式中的$c^Q$，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`int8`, shape为[t, q_lora_rank]。
- **qr_scale_out**（`Tensor`）：公式中Query做完第一次rmsnorm后的输出tensor（对应公式中的$c^Q$，不支持非连续的 Tensor，数据格式支持ND，数据类型支持`float32`, shape为[t, 1]。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持aclgraph模式。
- head_dim支持512，h支持4096，q_lora_rank支持1024，num_heads支持64，rope_dim支持64。
- t值域范围支持[1, 64k]。
- `950PR/DT`上暂不支持int8量化版本。
- 非量化实现可以参考example。


## 调用方法

```
量化：
python3 ops/pypto_python/example/test_mla_prolog_quant_pypto.py

非量化：
python3 ops/pypto_python/example/test_mla_prolog_pypto.py

```


