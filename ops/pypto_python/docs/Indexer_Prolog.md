# QuantLightningIndexerProlog

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A2 推理系列产品</term>   | √  |
|<term>Atlas A3 推理系列产品</term>   | √  |
|<term>Ascend 950PR/Ascend 950DT 推理系列产品</term>   | √  |

## 功能说明

- API功能：`QuantLightningIndexerProlog`算子旨在完成以下公式描述的Prolog计算，主要为后续LightningIndexer计算提供输入q、weight及q_scale。
- 计算公式：
  
  q, q_scale的计算公式为：
  
  $$
  q\_tmp = \text{qr}@{idx\_wq\_b} \cdot \text{qr\_scale} \cdot \text{idx\_wq\_b\_scale}
  $$
  
  $$
  q\_hadamard = \text{Cat}(\{q\_tmp[:, :nope\_dim], Rope(q\_tmp[:, nope\_dim:])\}, -1)@hadamard
  $$
  
  $$
  q, q\_scale = Quant(q\_hadamard)
  $$
  
  其中，Rope表示旋转位置编码计算，Quant表示量化计算。
Weights的计算公式为：

$$
weights = x@\text{weights\_proj} \cdot {\frac{1}{\sqrt{\text{idx\_nq} \cdot \text{head\_dim}}}}
$$

## 函数原型

```
torch.ops.pypto.quant_lightning_indexer_prolog(
    qr,
    idx_wq_b,
    x,
    weights_proj,
    cos,
    sin,
    hadamard,
    qr_scale,
    idx_wq_b_scale
) -> (Tensor, Tensor, Tensor)
```

## 参数说明


- **qr**（`Tensor`）：必选参数，进行q矩阵计算的左输入，不支持非连续，数据格式支持ND，数据类型支持`int8`。`layout_query`为TND时shape为[t, q_lora_rank]。
- **idx_wq_b**（`Tensor`）：必选参数，进行q矩阵计算的右输入，不支持非连续，数据格式支持ND，数据类型支持`int8`。`layout_query`为TND时shape为[q_lora_rank, idx_nq*head_dim]。
- **x**（`Tensor`）：必选参数，进行weights矩阵计算的左输入，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`。`layout_query`为TND时shape为[t， h]。
- **weights_proj**（`Tensor`）：必选参数，进行weights矩阵计算的右输入，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`。`layout_query`为TND时shape为[h, idx_nq]。
- **cos**（`Tensor`）：必选参数， 用于q的位置编码计算，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`。`layout_query`为TND时shape为[t， rope_dim]。
- **sin**（`Tensor`）：必选参数，用于q的位置编码计算，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`。`layout_query`为TND时shape为[t， rope_dim]。
- **hadamard**（`Tensor`）：必选参数， 进行q的hadamard矩阵计算时的右输入，不支持非连续，数据格式支持ND，数据类型支持`bfloat16`。`layout_query`为TND时shape为[head_dim, head_dim]。
- **qr_scale**（`Tensor`）：必选参数，qr矩阵计算后的反量化系数输入，不支持非连续，数据格式支持ND，数据类型支持`float32`。`layout_query`为TND时shape为[t, 1]。
- **idx_wq_b_scale**（`Tensor`）：必选参数，用于qr矩阵计算后的乘法输入，不支持非连续，数据格式支持ND，数据类型支持`float32`。`layout_query`为TND时shape为[idx_nq * head_dim, 1]。

## 返回值说明

- **q**（`Tensor`）：必选输出，公式中的输出q。数据格式支持ND，数据类型支持`int8`。当layout\_query为TND时shape为[t, idx_nq * head_dim]。
- **weights**（`Tensor`）：必选输出，公式中的输出weights。数据格式支持ND，数据类型支持`float16`。当layout\_query为TND时shape为[t, idx_nq]。
- **q_scale**（`Tensor`）：必选输出，公式中的输出q_scale。数据格式支持ND，数据类型支持`float16`。当layout\_query为TND时shape为[t, idx_nq]。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持aclgraph模式。
- q_lora_rank, idx_nq, head_dim, h, rope_dim仅支持默认值，t支持[1-64k]。
- 所有输入输出数据排布仅支持TND。
- 所有输入输出的数据类型仅支持所列场景，不支持额外类型。

## 调用方法

```
python3 ops/pypto_python/example/test_lightning_indexer_prolog_quant.py

```
