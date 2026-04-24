# custom-npu_dequant_swiglu_clamp_quant

## 产品支持情况

| 产品      | 是否支持 |
|:----------------------------|:-----------:|
|Atlas A3 推理系列产品|      √     |

## 功能说明

在Swish门控线性单元激活函数前后添加反量化（dequant）和量化（quant）操作，实现输入x的DequantSwigluClampQuant计算。支持标准SwiGLU模式和变体SwiGLU模式（带clamp截断）。

### 计算公式

**swiglu_mode=0（标准模式）:**

$$
dequantOut = Dequant(x, weightScale, activationScale, bias)
$$

$$
swigluOut = SwiGLU(dequantOut) = Swish(A) * B
$$

$$
out = Quant(swigluOut, quantScale, quantOffset)
$$

其中，A表示dequantOut的前半部分，B表示dequantOut的后半部分。当activate_left=True时，A部分做Swish激活；当activate_left=False时，B部分做Swish激活。

**swiglu_mode=1（变体模式）:**

$$
dequantOut = Dequant(x, weightScale, activationScale, bias)
$$

将dequantOut按最后一维切分为前半部分和后半部分，根据activate_left决定激活部分：

- 当activate_left=True时：
    - A（前半部分）：$A = A.clamp(min=None, max=clamp\_limit)$
    - B（后半部分）：$B = B.clamp(min=-clamp\_limit, max=clamp\_limit)$
    - $swigluOut = A * sigmoid(glu\_alpha * A) * (B + glu\_bias)$

- 当activate_left=False时：
    - A（前半部分）：$A = A.clamp(min=-clamp\_limit, max=clamp\_limit)$
    - B（后半部分）：$B = B.clamp(min=None, max=clamp\_limit)$
    - $swigluOut = B * sigmoid(glu\_alpha * B) * (A + glu\_bias)$

$$
out = Quant(swigluOut, quantScale, quantOffset)
$$

## 函数原型

```
custom.npu_dequant_swiglu_clamp_quant(Tensor x, *, Tensor? weight_scale=None, Tensor? activation_scale=None, Tensor? bias=None, Tensor? quant_scale=None, Tensor? quant_offset=None, Tensor? group_index=None, bool activate_left=False, int quant_mode=0, int? dst_type=None, int? round_mode=None, int? activate_dim=None, int swiglu_mode=0, float clamp_limit=7.0, float glu_alpha=1.702, float glu_bias=1.0) -> (Tensor, Tensor)
```

## 参数说明

>**说明：**<br> 
>
>- TokensNum、H参数维度含义：TokensNum表示传输的Token数，取值是自然数，H表示嵌入向量的长度，取值>0。

-   **x**（`Tensor`）：必选参数，输入tensor，待处理的数据。不支持非连续，数据格式支持ND，数据类型支持`int32`、`float16`、`bfloat16`，shape为[TokensNum, H]。

- <strong>*</strong>：代表其之前的参数是位置相关的，必须按照顺序输入，属于必选参数；其之后的参数是键值对赋值，与位置无关，属于可选参数（不传入会使用默认值）。

-   **weight_scale**（`Tensor`，可选）：权重反量化scale。当输入x为`int32`类型时必选。不支持非连续，数据格式支持ND，数据类型支持`float32`，shape为[H]或[groupNum, H]。

-   **activation_scale**（`Tensor`，可选）：激活反量化scale。当输入x为`int32`类型时可选。不支持非连续，数据格式支持ND，数据类型支持`float32`，shape为[TokensNum, 1]。

-   **bias**（`Tensor`，可选）：偏置项。不支持非连续，数据格式支持ND，数据类型支持`float32`、`float16`、`bfloat16`、`int32`，shape为[H]或[groupNum, H]。

-   **quant_scale**（`Tensor`，可选）：量化scale。不支持非连续，数据格式支持ND，数据类型支持`float32`、`float16`，shape为[H/2]或[groupNum, H/2]。

-   **quant_offset**（`Tensor`，可选）：量化offset。不支持非连续，数据格式支持ND，数据类型支持`float32`，shape为[H/2]或[groupNum, H/2]。

-   **group_index**（`Tensor`，可选）：MoE分组索引，用于指定分组处理。不支持非连续，数据格式支持ND，数据类型支持`int64`，shape为[groupNum]，表示每个分组的token数量。

-   **activate_left**（`bool`，可选）：表示是否对输入的前半部分做SwiGLU激活。默认为False，表示对后半部分做激活。

-   **quant_mode**（`int`，可选）：量化模式。0表示静态量化，1表示动态量化。默认为0。

-   **dst_type**（`int`，可选）：输出数据类型编码。默认为1。
    - 1：int8
    - 23：float8_e5m2
    - 24：float8_e4m3fn
    - 296：float4_e2m1
    - 297：float4_e1m2

-   **round_mode**（`int`，可选）：舍入模式编码。默认为0。
    - 0：rint
    - 1：round
    - 2：floor
    - 3：ceil
    - 4：trunc

-   **activate_dim**（`int`，可选）：进行SwiGLU切分的维度。默认为-1（最后一维）。

-   **swiglu_mode**（`int`，可选）：SwiGLU计算模式。默认为0。
    - 0：标准SwiGLU模式（SiLU激活）
    - 1：变体SwiGLU模式（带clamp截断）

-   **clamp_limit**（`float`，可选）：变体SwiGLU模式下的截断阈值。默认为7.0。仅在swiglu_mode=1时生效。

-   **glu_alpha**（`float`，可选）：变体SwiGLU模式下的sigmoid参数。默认为1.702。仅在swiglu_mode=1时生效。

-   **glu_bias**（`float`，可选）：变体SwiGLU模式下的偏置参数。默认为1.0。仅在swiglu_mode=1时生效。

## 返回值说明

-   **y**（`Tensor`）：输出tensor，量化后的结果。不支持非连续，数据格式支持ND，数据类型由dst_type参数决定，shape为[TokensNum, H/2]。

-   **scale**（`Tensor`）：输出tensor，量化scale。不支持非连续，数据格式支持ND，数据类型支持`float32`。当quant_mode=0（静态量化）时输出固定值；当quant_mode=1（动态量化）时输出动态计算的scale，shape为[TokensNum]。

## 约束说明

-  输入约束
    - x的activate_dim维度必须是2的倍数。
    - x的维数必须大于1维。
    - 当输入x为`int32`类型时，weight_scale必须输入。
    - 当输入x为非`int32`类型时，weight_scale不允许输入。
    - 当输入x为非`int32`类型时，activation_scale不允许输入。
    - 当输入x为非`int32`类型时，bias不允许输入。

-  输出约束
    - y的最后一维不超过5120。
    - 当dst_type为float4类型（296或297）时，y的最后一维需要是2的倍数。

-  quant_mode约束
    - quant_mode=0（静态量化）时，quant_scale和quant_offset必须输入。
    - quant_mode=1（动态量化）时，quant_offset不支持输入。

-  该接口支持推理场景下使用。
-  该接口支持aclgraph入图。
-  该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。

## 调用示例

- 详见 [test_npu_dequant_swiglu_clamp_quant.py](../examples/test_npu_dequant_swiglu_clamp_quant.py)