# custom-inplace_partial_rotary_mul

## 产品支持情况

| 产品      | 是否支持 |
|:----------------------------|:-----------:|
|Atlas A3 推理系列产品|      √     |
|Ascend 950PR/Ascend 950DT|      √     |

## 功能说明
inplace_partial_rotary_mul是rotary_mul的融合算子，它在四维输入x的尾轴中partial_slice的部分做部分rope，计算逻辑如下：

$$
o_nope,_1, o_rope, o_nope_2 = x.split([partial_slice[0], partial_slice[1] - partial_slice[0], x.dim(-1) - partial_slice[1]], -1)
$$

$$
o_rope = npu_rotary_mul(o_rope, cos, sin ,rotary_mode)
$$

$$
x = concat([o_nope,_1, o_rope, o_nope_2], -1)
$$

## 函数原型
```
custom.inplace_partial_rotary_mul(Tensor(a!) x, Tensor r1, Tensor r2, *, str rotary_mode, int[2] partial_slice) -> ()
```

## 参数说明

>**说明：**<br> 
>
>- b（batch size）表示输入样本批量大小、s（sequence length）表示输入样本序列长度、n（number of heads）表示注意力头数、d（head dimension）表示注意力头的维度数，slice_size表示待执行rope运算的片段长度。

-   **x**（`Tensor`）：必选参数，输入及输出tensor，待执行rope的tensor。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，`float16`，`float32`，shape为[b, s, n, d]。

-   **r1**（`Tensor`）：必选参数，输入tensor，rope运算中的cos。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，`float16`，`float32`，shape为[b, 1, 1, slice_size]或[b, s, 1, slice_size]。

-   **r2**（`Tensor`）：必选参数，输入tensor。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，`float16`，`float32`，shape为[b, 1, 1, slice_size]或[b, s, 1, slice_size]且必须与`r1`一致。
-   **rotary_mode**（`String`）：必选参数，输入字符串，目前只支持`interleave`。

-   **partial_slice**（`int[2]`）：必选参数，x尾轴的起始坐标和结束坐标，在坐标内的部分会参与rope计算。


## 约束说明
- d轴必须小于等1024，且关于2对齐。
- slice_size必须关于2对齐，且如果低于64会有性能影响。
- x、r1、r2的dtype必须一致。
- 该接口支持推理场景下使用。
- 该接口支持aclgraph入图。
- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
    
## 调用示例

- 详见 [test_inplace_partial_rotary_mul.py](../examples/test_inplace_partial_rotary_mul.py)