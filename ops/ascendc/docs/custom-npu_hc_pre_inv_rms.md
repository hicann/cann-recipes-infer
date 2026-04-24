# custom-npu_hc_pre_inv_rms

## 产品支持情况

| 产品      | 是否支持 |
|:----------------------------|:-----------:|
|Atlas A3 推理系列产品|      √     |
|Ascend 950PR/Ascend 950DT|      √     |

## 功能说明

HcPre中用于计算InvRms的部分，计算逻辑见InvRms计算公式

### 计算公式

#### InvRms公式

$$
\text{InvRms}(x) = \frac{1}{\text{RMS}(x)}
$$

$$
\text{RMS}(x) = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}
$$


## 函数原型
```
custom.npu_hc_pre_inv_rms(Tensor x, *, float epsilon=1e-20) -> Tensor
```

## 参数说明

>**说明：**<br> 
>
>- b（batch size）表示输入样本批量大小、s（sequence length）表示输入样本序列长度、hc（head count）表示注意力头数、d（head dimension）表示注意力头的维度数、T表示bs合轴后的大小。

-   **x**（`Tensor`）：必选参数，输入Tensor，公式中用于计算的输入tensor。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[T, hc_mult, d]或[b, s, hc_mult, d]。


- <strong>*</strong>：代表其之前的参数是位置相关的，必须按照顺序输入，属于必选参数；其之后的参数是键值对赋值，与位置无关，属于可选参数（不传入会使用默认值）。


-   **epsilon**（`float`, 可选）：计算RmsNorm公式中的$\epsilon$参数，Host侧参数，仅支持double类型，默认值为1e-20。


## 返回值说明
-   **y**（`Tensor`）：计算公式中InvRms的输出tensor。数据格式支持ND，数据类型支持`float`，shape为[T, 1]或[b, s, 1]。

## 约束说明
-  shape 字段取值范围约束
    | 字段名       | 取值规则与说明                                                                 |
    |--------------|-------------------------------------------|
    | hc_mult           |  取值固定为: 4                                                |
    | d          |  取值固定为：4096                                                           |

- 该接口支持推理场景下使用。
- 该接口支持aclgraph入图。
- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
    
## 调用示例

- 详见 [test_npu_hc_pre_inv_rms.py](../examples/test_npu_hc_pre_inv_rms.py)