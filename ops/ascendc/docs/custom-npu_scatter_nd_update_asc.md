# custom-npu_scatter_nd_update_asc

## 产品支持情况

| 产品      | 是否支持 |
|:----------------------------|:-----------:|
|Atlas A3 推理系列产品|      √     |

## 功能说明

ScatterNdUpdateAsc 对已有的可变张量在指定索引位置进行稀疏更新，详细计算过程参考[test_npu_scatter_nd_update_asc.py](../examples/test_npu_scatter_nd_update_asc.py)。

## 函数原型
```
custom.scatter_nd_update_asc(Tensor(a!) var, Tensor indices, Tensor update) -> ()
```

## 参数说明

>**说明：**<br> 
>
>- b（batch size）表示输入样本批量大小、s（sequence length）表示输入样本单次更新数据量、u（update num）表示更新的次数。

-   **var**（`Tensor`）：必选参数，输入tensor，。不支持非连续，数据格式支持ND，数据类型支持`bfloat16, float16, int8`，shape为[b, s]。

-   **indices**（`Tensor`）：必选参数，输入tensor。不支持非连续，数据格式支持ND，数据类型支持`int32, int64`，shape为[u, 1]。

-   **update**（`Tensor`）：必选参数，输入tensor。不支持非连续，数据格式支持ND，数据类型和var保持一致，shape为[u, s]。


## 返回值说明
无返回值，var做原地操作更新。

## 约束说明
-  shape 字段取值范围约束
    | 字段名       | 取值规则与说明                                                                 |
    |--------------|-------------------------------------------|
    | s          |  取值固定为：1或者128或者512                                                           |
    | u            |  取值小于等于b          |

- 该接口索引不支持越界，负值会被直接跳过，不更新。
- 该接口支持推理场景下使用。
- 该接口支持aclgraph入图。
- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
    
## 调用示例

- 详见 [test_npu_scatter_nd_update_asc.py](../examples/test_npu_scatter_nd_update_asc.py)