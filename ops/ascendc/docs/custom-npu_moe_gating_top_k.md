# custom-npu_moe_gating_top_k

## 产品支持情况

| 产品      | 是否支持 |
|:----------------------------|:-----------:|
|Atlas A3 推理系列产品|      √     |
|Ascend 950PR/Ascend 950DT|      √     |

## 功能说明

- 算子功能：MoE (Mixture of Experts) 计算中，对输入x做Sigmoid、SoftMax或者Softplus计算，对计算结果分组进行排序，对结果取前K个专家或根据输入的词表进行hash。
- 计算公式：

    对输入做Sigmoid或者SoftMax：

    $$
    \begin{aligned}
    &if\ normType == 1:
        normOut=Sigmoid(x) \\
    &else\ if\ normType == 0:
        normOut=SoftMax(x) \\
    &else\ :
        normOut=Softplus(x)
    \end{aligned}
    $$

    如果bias不为空：

    $$
    normOut = normOut + bias
    $$

    对计算结果按照groupCount进行分组，每组按照groupSelectMode取max或topk2的sum值对group进行排序，取前kGroup个组：

    $$
    groupOut, groupId = TopK(ReduceSum(TopK(Split(normOut, groupCount), k=2, dim=-1), dim=-1),k=kGroup)
    $$
    
    如果指定了input_ids和tid2eid，则根据输入的词表进行hash操作；
    否则根据上一步的groupId获取normOut中对应的元素，将数据再做TopK，得到expertIdxOut的结果：

    $$
    y,expertIdxOut=TopK(normOut[groupId, :],k=k)
    $$

    对y按照输入的routedScalingFactor和eps参数进行计算，得到yOut的结果：

    $$
    yOut = y / (ReduceSum(y, dim=-1)+eps)*routedScalingFactor
    $$

## 函数原型
```
custom.npu_moe_gating_top_k(Tensor x, int k, *, Tensor? bias=None, Tensor? input_ids=None, Tensor? tid2eid=None, int k_group=1, int group_count=1, float routed_scaling_factor=1., float eps=9.9999999999999995e-21, int group_select_mode=0, int renorm=0, int norm_type=0, bool out_flag=False) -> (Tensor, Tensor, Tensor)
```

## 参数说明
>**说明：**<br> 
>
>- b（batch size）表示输入样本批量大小、s（sequence length）表示输入样本序列长度、T表示bs合轴后的大小、e表示专家数量、k表示选取Top专家数。

| 参数名 | 类型 | 描述 |
|:---|:---|:---|
| `x` | Tensor | 输入张量，支持2D或3D，shape为 (T, e) 或 (b, s, e)，支持`float16`、`bfloat16`和`float32` |
| `k` | int | 选取的专家数量，取值小于等于e，且必须小于等于64 |
| `bias` | Tensor（可选） | 偏置张量，shape为（e），dtype与x相同， 支持`float16`、`bfloat16`和`float32` |
| `input_ids` | Tensor（可选） | 输入词表，shape为（T）， 仅支持`int64`，取值范围为[0 ,n]，n为tid2eid第一维的大小 |
| `tid2eid` | Tensor（可选） | 词表到专家id的映射关系表，shape为（n，k）， 仅支持`int32`，取值范围为[0 ,e]，e代表专家数 |
| `k_group` | int（可选） | 选取的组数量，默认为1 |
| `group_count` | int（可选） | 总组数，默认为1 |
| `routed_scaling_factor` | float（可选） | 路由缩放因子，默认为1 |
| `eps` | float（可选） | 数值稳定性参数，防止除零，默认为1e-20 |
| `group_select_mode` | int（可选） | 组选择模式：0-使用最大值排序，1-使用top2的和排序 |
| `renorm` | int（可选） | 重归一化标志，仅支持0 |
| `norm_type` | int（可选） | 归一化标志，0-Softmax, 1-Sigmoid, 2-Softplus |
| `out_flag` | bool（可选） | 是否输出归一化结果 |

## 返回值说明
| 返回值 | 类型 | 描述 |
|:---|:---|:---|
| `yOut` | Tensor | 归一化、分组排序和TopK后的结果 |
| `expertIdxOut` | Tensor | 专家索引，数据类型为int32 |
| `out` | Tensor | 归一化结果（当out_flag=True时有效） |

## 约束说明
* renorm仅支持0，表示先进行norm操作，再计算topk。
* group_select_mode取值0和1，0表示使用最大值对group进行排序, 1表示使用topk2的sum值对group排序。
* norm_type取值0和1，0表示使用Softmax函数，1表示使用Sigmoid函数，2表示使用Softplus函数。
* outFlag取值true和false，true表示输出，false表示不输出。
* input_ids和tid2eid都不为空表示hash场景，都为空表示topk场景，不允许只有一个为空。
* k_group和group_count为1时，表示不分组排序。
* bias的dtype要和x相同。
* 该接口支持推理场景下使用。
* 该接口支持aclgraph入图。
* 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
    
## 调用示例

- 详见 [test_npu_moe_gating_top_k.py](../examples/test_npu_moe_gating_top_k.py)