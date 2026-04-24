# custom-npu_swiglu_group_quant

## 产品支持情况

| 产品      | 是否支持 |
|:----------------------------|:-----------:|
|Ascend 950PR/Ascend 950DT|      √     |

## 功能说明

在SwiGlu激活函数后添加量化操作，实现输入x的SwiGluQuant计算。根据quant_mode不同，量化分组大小有所差异：quant_mode=1或3时groupSize固定为128，quant_mode=2时groupSize固定为32。计算过程见计算公式。

### 计算公式

  $$
    Y_{tmp} = SwiGLU(x) = Swish(A)*B 
  $$

  $$
    scale=row\_max(abs(Y_{tmp}))/dstTypeScale
  $$

  $$
    Y = Cast(Mul(Y_{tmp}, Scale))
  $$
     其中，A表示输入x的前半部分，B表示输入x的后半部分。

## 函数原型
```
custom.npu_swiglu_group_quant(Tensor x, *, Tensor? topk_weight=None, Tensor? group_index=None, ScalarType dst_type, int quant_mode=1, int group_size=128, bool round_scale=False, bool ue8m0_scale=False, bool output_origin=False, int group_list_type=0, float clamp_value=0.0) -> (Tensor, Tensor, Tensor)
```

## 参数说明

>**说明：**<br> 
>
>- b（batch size）表示输入样本批量大小、s（sequence length）表示输入样本序列长度、d（head dimension）表示注意力头的维度数、T表示bs合轴后的大小、G表示MoE场景下的分组数量。

-   **x**（`Tensor`）：必选参数，输入tensor，公式中的输入 x 。不支持非连续，数据格式支持ND，数据类型支持`float16`和`bfloat16`，shape为[ b, s, d ]或[ T, d ]（T为bs合轴后的大小）。

- <strong>*</strong> ：代表其之前的参数是位置相关的，必须按照顺序输入，属于必选参数；其之后的参数是键值对赋值，与位置无关，属于可选参数（不传入会使用默认值）。

-   **topk_weight**（`Tensor`，可选）：MoE场景下的topk权重tensor，用于对SwiGLU输出进行加权。数据类型支持`float32`，shape为[ T, 1 ]。默认为None，表示不进行加权操作。

-   **group_index**（`Tensor`，可选）：MoE场景下分组索引tensor，用于指定每个group的token个数。数据类型支持`int64`，shape为[G]，其中G表示分组数量。该Tensor的数值总和代表输入x中的有效Token数。默认值为None，表示非MoE场景，此时所有Token都有效。

-   **dst_type**（`ScalarType`，可选）：量化输出数据类型，支持`torch.float8_e5m2`和`torch.float8_e4m3fn`。默认为`torch.float8_e4m3fn`。

-   **quant_mode**（`int`, 可选）：量化模式，取值范围为1、2、3。
    - 1：PerGroup fp8量化，量化输出scale为float32类型，groupSize固定为128。
    - 2：MX fp8量化，量化输出scale为float8_e8m0类型，groupSize固定为32。
    - 3：PerGroup fp8量化增强模式，groupSize固定为128。scale类型由ue8m0_scale参数决定：ue8m0_scale=False时scale为float32类型（与quant_mode=1功能相同），ue8m0_scale=True时scale为float8_e8m0类型。相比quant_mode=1，该模式还支持round_scale、output_origin等额外参数。

-   **group_size**（`int`, 可选）：量化分组大小。quant_mode=1或3时固定为128；quant_mode=2时固定为32。默认为128。

-   **round_scale**（`bool`, 可选）：是否对scale进行舍入处理。默认为False。仅在quant_mode=3时有效。

-   **ue8m0_scale**（`bool`, 可选）：是否使用float8_e8m0格式输出scale。默认为False，scale为float32类型；设置为True时，scale为float8_e8m0类型。仅在quant_mode=3时有效。

-   **output_origin**（`bool`, 可选）：是否输出SwiGLU计算后的原始fp16/bf16结果。默认为False。仅在quant_mode=3时有效。

-   **group_list_type**（`int`, 可选）：分组列表类型，固定为0，默认为0，表示count模式。在count模式下，group_index的每个元素代表对应分组的元素个数。在quant_mode=1、2、3时均生效。

-   **clamp_value**（`float`, 可选）：SwiGLU激活函数的截断值。当clamp_value>0时，对激活值进行截断：A部分只有最大值约束，截断到不超过clamp_value；B部分截断到[-clamp_value, clamp_value]。默认为0.0，表示不进行截断。


## 返回值说明

-   **y**（`Tensor`）：输出tensor，量化后的value。不支持非连续，数据格式支持ND，数据类型支持`float8_e4m3fn`和`float8_e5m2`，shape为[ b, s, d/2 ]或[ T, d/2 ]。当使用group_index时，前有效Token数行的数据有效，其余为填充数据。

-   **scale**（`Tensor`）：输出tensor，量化后的scale。不支持非连续，数据格式支持ND。当quant_mode=1或quant_mode=3且ue8m0_scale=False时，数据类型为`float32`；当quant_mode=2或quant_mode=3且ue8m0_scale=True时，数据类型为`float8_e8m0`。

-   **yOrigin**（`Tensor`）：输出tensor，SwiGLU计算后的原始结果。仅在quant_mode=3且output_origin=True时有效输出，其他情况下返回空tensor。数据类型与输入x相同（`float16`或`bfloat16`），shape为[ b, s, d/2 ]或[ T, d/2 ]。

## 约束说明

-  shape 字段取值范围约束
    | quant_mode | d（最后一维）约束 |
    |------------|------------------|
    | 1 或 3     | 必须是256的倍数   |
    | 2          | 必须是128的倍数   |

- quant_mode 取值范围为1、2、3。具体说明详见参数说明中的quant_mode字段。
- round_scale、ue8m0_scale、output_origin参数仅在quant_mode=3时有效。
- group_list_type固定为0，表示count模式，在quant_mode=1、2、3时均生效。
- group_index的数值总和必须不超过输入x的第一维大小。
- 该接口支持推理场景下使用。
- 该接口支持aclgraph入图。
- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。

    
## 调用示例

- 详见 [test_npu_swiglu_group_quant.py](../examples/test_npu_swiglu_group_quant.py)