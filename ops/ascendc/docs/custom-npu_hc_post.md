# custom-npu_hc_post

## 产品支持情况

| 产品      | 是否支持 |
|:----------------------------|:-----------:|
|Atlas A3 推理系列产品|      √     |
|Ascend 950PR/Ascend 950DT|      √     |

## 功能说明
hc_post 是 mHC 结构的后处理部分计算逻辑，计算过程如下：

$$
out = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
$$

## 函数原型
```
custom.npu_hc_post(Tensor x, Tensor residual, Tensor post, Tensor comb) -> Tensor
```

## 参数说明

>**说明：**<br> 
>
>- b（batch size）表示输入样本批量大小、s（sequence length）表示输入样本序列长度、hc（head count）表示注意力头数、d（head dimension）表示注意力头的维度数。

-   **x**（`Tensor`）：必选参数，输入tensor。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，`float16`，`float32`，shape为[b, s, d]或[b * s, d]。

-   **residual**（`Tensor`）：必选参数，输入tensor。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，`float16`，`float32`，shape为[b, s, hc, d]或[b * s, hc, d]。

-   **post**（`Tensor`）：必选参数，输入tensor。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，`float16`，`float32`，shape为[b, s, hc]或[b * s, hc]。

-   **comb**（`Tensor`）：必选参数，输入tensor。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，`float16`，`float32`，shape为[b, s, hc, hc]或[b * s, hc, hc]。

## 返回值说明
-   **y**（`Tensor`）：输出tensor。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，`float16`，`float32`，shape与**residual**的shape一致，为[b, s, hc, d]或[b * s, hc, d]。

## 约束说明
-  shape 字段取值范围约束
    | 字段名       | 取值规则与说明                                                                 |
    |--------------|-------------------------------------------|
    | hc           |  取值固定为: 4                                                |
    | d         |  取值固定为：4096                                                           |

- x、residual和y的dtype要保持一致。
- post和comb的dtype要保持一致。
- 该接口支持推理场景下使用。
- 该接口支持aclgraph入图。
- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
    
## 调用示例

- 详见 [test_npu_hc_post.py](../examples/test_npu_hc_post.py)