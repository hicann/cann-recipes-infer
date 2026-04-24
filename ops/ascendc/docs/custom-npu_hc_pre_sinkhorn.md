# custom-npu_hc_pre_sinkhorn

## 产品支持情况

| 产品      | 是否支持 |
|:----------------------------|:-----------:|
|Atlas A3 推理系列产品|      √     |
|Ascend 950PR/Ascend 950DT|      √     |

## 功能说明
hc_pre_sinkhorn 负责 hc_pre 的 sinkhorn 部分的计算处理，详细计算过程参考[test_npu_hc_pre_sinkhorn.py](../examples/test_npu_hc_pre_sinkhorn.py)


## 函数原型
```
custom.npu_hc_pre_sinkhorn(Tensor mixes, Tensor rsqrt, Tensor hc_scale, Tensor hc_base, Tensor x, int hc_mult=4, int hc_sinkhorn_iters=20, float hc_eps=1e-5) -> (Tensor, Tensor, Tensor)
```

## 参数说明

>**说明：**<br> 
>
>- b（batch size）表示输入样本批量大小、s（sequence length）表示输入样本序列长度、hc（head count）表示注意力头数、d（head dimension）表示注意力头的维度数、T表示bs合轴后的大小。

-   **mixes**（`Tensor`）：必选参数，输入tensor。不支持非连续，数据格式支持ND，数据类型支持`float`，shape为[T, hc_mix]或[b, s, hc_mix]。

-   **rsqrt**（`Tensor`）：必选参数，输入tensor。不支持非连续，数据格式支持ND，数据类型支持`float`，shape为[T, 1]或[b, s, 1]。

-   **hc_scale**（`Tensor`）：必选参数，输入tensor。不支持非连续，数据格式支持ND，数据类型支持`float`，shape为[3]。

-   **hc_base**（`Tensor`）：必选参数，输入tensor。不支持非连续，数据格式支持ND，数据类型支持`float`，shape为[hc_mix]。

-   **x**（`Tensor`）：必选参数，输入tensor。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[T, hc_mult, d]或[b, s, hc_mult, d]。

-   **hc_mult**（`int`）：固定为4。

-   **hc_sinkhorn_iters**（`int`, 可选）：取值固定为20。

-   **hc_eps**（`float`, 可选）：计算过程中的$\epsilon$参数，Host侧参数。仅支持double类型，默认值为1e-05。


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
    
## 调用示例

- 详见 [test_npu_hc_pre_sinkhorn.py](../examples/test_npu_hc_pre_sinkhorn.py)