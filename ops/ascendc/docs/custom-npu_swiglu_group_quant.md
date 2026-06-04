# custom.npu_swiglu_group_quant

## 产品支持情况

| 产品 | 是否支持 |
|:--|:--:|
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

`npu_swiglu_group_quant`在SwiGLU激活函数后执行FP8量化，支持Block FP8量化和MX FP8量化两种模式。输入`x`最后一维按左右两半拆分，左半部分记为`A`，右半部分记为`B`。

### 计算公式

$$
Y_{tmp}=SwiGLU(x)=Swish(A) * B
$$

如果传入`clamp_limit`，在SwiGLU计算前对`A`和`B`执行如下截断：

$$
A=Min(A, clamp\_limit)
$$

$$
B=Min(Max(B, -clamp\_limit), clamp\_limit)
$$

如果传入`weight`，量化前按token乘到`Y_tmp`上：

$$
Y_{tmp}=Y_{tmp} * weight
$$

Block FP8量化模式下：

$$
scale\_out=block\_max(abs(Y_{tmp})) / dstTypeScale
$$

$$
Y=Cast(Y_{tmp} / scale\_out)
$$

MX FP8量化模式下，按每32个通道计算scale，并输出`float8_e8m0`格式scale：

$$
scale\_out=mx\_scale(abs(Y_{tmp}), round\_scale)
$$

$$
Y=Cast(Y_{tmp} / scale\_out)
$$

## 函数原型

```python
custom.npu_swiglu_group_quant(
    Tensor x,
    *,
    Tensor? weight=None,
    Tensor? group_index=None,
    ScalarType dst_type,
    int quant_mode=0,
    int block_size=0,
    bool round_scale=False,
    float? clamp_limit=None,
    bool output_origin=False
) -> (Tensor, Tensor, Tensor)
```

## 参数说明

> **说明：**
>
> - `T`表示除最后一维外所有维度合轴后的token数。
> - `D`表示输入`x`最后一维，SwiGlu后输出隐藏维为`H = D / 2`。
> - `G`表示`group_index`的元素个数。

- **x**（`Tensor`）：必选输入。SwiGLU的输入。不支持非连续，数据格式支持ND，数据类型支持`float16`和`bfloat16`。shape支持`[..., D]`，典型shape为`[b, s, D]`或`[T, D]`。

- **\***：代表其之前的参数是位置相关参数，必须按照顺序输入；其之后的参数是关键字参数。

- **weight**（`Tensor`，可选）：可选输入。量化前按token乘到SwiGLU输出上。不支持非连续，数据格式支持ND，数据类型支持`float32`。shape通常为`[T]`或`[T, 1]`，需要与`x`合轴后的token维匹配。

- **group_index**（`Tensor`，可选）：可选输入。表示各group中的token数量，当前按count模式使用。不支持非连续，数据格式支持ND，数据类型支持`int64`。shape通常为`[G]`。传入后，算子只保证前`sum(group_index)`个token对应输出有效。

- **dst_type**（`ScalarType`）：必选属性。量化输出`y`的数据类型，仅支持`torch.float8_e4m3fn`和`torch.float8_e5m2`。

- **quant_mode**（`int`，可选）：量化模式。默认值为`0`。
  - `0`：Block FP8量化，`scale_out`数据类型为`float32`。
  - `1`：MX FP8量化，`scale_out`数据类型为`float8_e8m0`。

- **block_size**（`int`，可选）：量化block大小。默认值为`0`，表示使用当前`quant_mode`对应默认值。
  - `quant_mode=0`时，仅支持`0`或`128`；`0`按`128`处理。
  - `quant_mode=1`时，仅支持`0`或`32`；`0`按`32`处理。

- **round_scale**（`bool`，可选）：是否对scale进行round处理。
  - `quant_mode=0`时仅支持`False`。
  - `quant_mode=1`时仅支持`True`。

- **clamp_limit**（`float`，可选）：SwiGLU计算前对输入进行截断的阈值。不传入时不执行clamp；传入时取值必须大于等于`0.0`，不支持NaN。

- **output_origin**（`bool`，可选）：是否在MX FP8量化模式下写出量化前的SwiGLU结果`y_origin`，默认值为`False`。`quant_mode=0`时该输出不作为有效结果使用。

## 返回值说明

- **y**（`Tensor`）：量化后的输出。数据格式支持ND，数据类型与`dst_type`一致，shape为`[..., H]`。

- **scale_out**（`Tensor`）：量化scale输出。数据格式支持ND。
  - `quant_mode=0`时，数据类型为`float32`，shape为`[..., ceil(H / 128)]`。
  - `quant_mode=1`时，数据类型为`float8_e8m0`，shape为`[..., ceil(ceil(H / 32) / 2), 2]`。

- **y_origin**（`Tensor`）：量化前的SwiGLU结果。数据格式支持ND，数据类型与`x`一致，shape与`y`一致。仅`quant_mode=1`且`output_origin=True`时结果有效；其他场景下该输出会分配但不保证写入有效业务数据。

## 约束说明

- 输入`x`约束：
  - rank必须大于0。
  - 最后一维`D`必须大于等于`256`，且必须能被`256`整除。
  - `D`会被均分为`A`和`B`两部分，因此SwiGLU输出最后一维为`H=D/2`。
  - 数据类型仅支持`float16`和`bfloat16`，格式仅支持ND。

- 属性取值约束：

  | 属性 | 支持取值 |
  |:--|:--|
  | `dst_type` | `torch.float8_e4m3fn`、`torch.float8_e5m2` |
  | `quant_mode` | `0`、`1` |
  | `block_size` | `quant_mode=0`时为`0`或`128`；`quant_mode=1`时为`0`或`32` |
  | `round_scale` | `quant_mode=0`时必须为`False`；`quant_mode=1`时必须为`True` |
  | `clamp_limit` | 不传入，或传入大于等于`0.0`的有限/无穷浮点值；不支持NaN |

- 输出dtype约束：
  - `y`数据类型必须与`dst_type`一致。
  - `scale_out`在`quant_mode=0`时必须为`float32`，在`quant_mode=1`时必须为`float8_e8m0`。
  - `y_origin`数据类型必须与`x`一致。

- `group_index`约束：
  - 仅支持count模式，表示每个group的token数量。
  - 需要保证`sum(group_index) <= T`，否则可能访问超出`x`有效token范围。
  - 传入`group_index`后，输出中超过`sum(group_index)`对应token范围的部分不保证为有效业务数据。

- 其他约束：
  - 该接口支持推理场景下使用。
  - 该接口支持aclgraph入图。
  - 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。

## 调用示例

```python
import torch
import torch_npu
import custom_ops

x = torch.randn(1, 128, 4096, device="npu", dtype=torch.float16)

# Block FP8 quant
y, scale, y_origin = torch.ops._ascend_dsv4.npu_swiglu_group_quant(
    x,
    dst_type=torch.float8_e5m2,
    quant_mode=0,
    block_size=0,
    round_scale=False,
)

# MX FP8 quant
y_mx, scale_mx, y_origin_mx = torch.ops._ascend_dsv4.npu_swiglu_group_quant(
    x,
    dst_type=torch.float8_e4m3fn,
    quant_mode=1,
    block_size=32,
    round_scale=True,
    output_origin=True,
)
```

- 详见[test_npu_swiglu_group_quant.py](../examples/test_npu_swiglu_group_quant.py)
