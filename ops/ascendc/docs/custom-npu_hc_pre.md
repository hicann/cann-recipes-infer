# custom-npu_hc_pre

## 产品支持情况

| 产品      | 是否支持 |
|:----------------------------|:-----------:|
|Atlas A3 推理系列产品|      √     |
|Ascend 950PR/Ascend 950DT|      √     |

## 功能说明

HcPre 是 mHC 结构前处理部分的融合算子，详细计算过程参考[test_npu_hc_pre.py](../examples/test_npu_hc_pre.py)。

## 函数原型
```
custom.npu_hc_pre(Tensor x, Tensor hc_fn, Tensor hc_scale, Tensor hc_base, *,int hc_mult=4, int hc_sinkhorn_iters=20, float norm_eps=1e-6, float hc_eps=1e-6) -> (Tensor, Tensor, Tensor)
```

## 参数说明

>**说明：**<br> 
>
>- b（batch size）表示输入样本批量大小、s（sequence length）表示输入样本序列长度、hc（head count）表示注意力头数、d（Head dimension）表示注意力头的维度数、T表示bs合轴后的大小。

-   **x**（`Tensor`）：必选参数，输入tensor，mHC 结构的输入数据。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[T, hc_mult, d]或[b, s, hc_mult, d]。

-   **hc_fn**（`Tensor`）：必选参数，输入tensor。不支持非连续，数据格式支持ND，数据类型支持`float32`，shape为[hc_mix, hc_mult * d]。

-   **hc_scale**（`Tensor`）：必选参数，输入tensor。不支持非连续，数据格式支持ND，数据类型支持`float`，shape为[3]。

-   **hc_base**（`Tensor`）：必选参数，输入tensor。不支持非连续，数据格式支持ND，数据类型支持`float`，shape为[hc_mix]。

- <strong>*</strong> ：代表其之前的参数是位置相关的，必须按照顺序输入，属于必选参数；其之后的参数是键值对赋值，与位置无关，属于可选参数（不传入会使用默认值）。

-   **hc_mult**（`int`）：固定为4。

-   **hc_sinkhorn_iters**（`int`, 可选）：取值固定为20。

-   **norm_eps**（`float`, 可选）：InvRms计算过程中的$\epsilon$参数，Host侧参数。仅支持double类型，默认值为1e-06。

-   **hc_eps**（`float`, 可选）：Sinkhorn计算过程中的$\epsilon$参数，Host侧参数。仅支持double类型，默认值为1e-06。


## 返回值说明
-   **y**（`Tensor`）：输出tensor。数据格式支持ND，数据类型支持`bfloat16`，shape为[T, d]或[b, s, d]。

-   **post**（`Tensor`）：输出tensor。数据格式支持ND，数据类型支持`float`，shape为[T, hc_mult]或[b, s, hc_mult]。

-   **comb_frag**（`Tensor`）：输出tensor。数据格式支持ND，数据类型支持`float`，shape为[T, hc_mult, hc_mult]或[b, s, hc_mult, hc_mult]。

## 约束说明
-  shape 字段取值范围约束
    | 字段名       | 取值规则与说明                                                                 |
    |--------------|-------------------------------------------|
    | hc_mult           |  取值固定为: 4                                                |
    | d          |  取值固定为：4096                                                           |
    | hc_mix            |  取值固定为: 24          |

- 该接口支持推理场景下使用。
- 该接口支持aclgraph入图。
- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- 该接口在T/bs小于等于128，且能被16整除时，会使能hc_pre融合算子，性能较高；其他场景会使能hc_pre_inv_rms及hc_pre_sinkhorn小算子拼接，性能较低。
    
## 调用示例

- 详见 [test_npu_hc_pre.py](../examples/test_npu_hc_pre.py)