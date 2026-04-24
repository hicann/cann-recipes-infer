# custom-npu_kv_compress_epilog

## 产品支持情况

| 产品      | 是否支持 |
|:----------------------------|:-----------:|
|Ascend 950PR/Ascend 950DT|      √     |

## 功能说明

将输入x split成nope和rope，nope部分进行量化后与rope进行拼接，根据slot_mapping中的下标映射（-1表示无效数据，不进行更新），原地更新到kv_compress_cache对应的行上。


## 函数原型
```
custom.kv_compress_epilog(Tensor(a!) kv_compress_cache, Tensor x, Tensor slot_mapping, *, int quant_group_size = 64, int quant_mode = 2, bool round_scale_flag = True) -> ()
```

## 参数说明

>**说明：**<br> 
>
>- b（batch size）表示输入样本批量大小、s（sequence length）表示输入样本序列长度、hc（head count）表示注意力头数、d（Head dimension）表示注意力头的维度数、T表示bs合轴后的大小。

-   **kv_compress_cache**（`Tensor`）：必选参数，输入tensor，待更新的cache数据。不支持非连续，数据格式支持ND，数据类型支持`float8_e4m3fn`和`float8_e5m2`，shape为[ T, d' ]，d'的取值见约束说明。

-   **x**（`Tensor`）：必选参数，输入tensor，待量化的输入数据。不支持非连续，数据格式支持ND，数据类型支持`bfloat16`，shape为[ T, d ]。

-   **slot_mapping**（`Tensor`）：必选参数，输入tensor，表示下标映射，其值为kv_compress_cache的下标，-1代表无效数据，不进行更新。不支持非连续，数据格式支持ND，数据类型支持`int32`和`int64`，shape为[ T ]。

- <strong>*</strong>：代表其之前的参数是位置相关的，必须按照顺序输入，属于必选参数；其之后的参数是键值对赋值，与位置无关，属于可选参数（不传入会使用默认值）。

-   **quant_group_size**（`int`，可选）：quant_group的大小，取值固定为64。

-   **quant_mode**（`int`，可选）：量化模式，取值固定为2，表示量化输出并拼接到kv_compress_cache中的scale为float8_e8m0类型。

-   **round_scale_flag**（`bool`, 可选）：表示是否计算round_scale，默认为True。


## 返回值说明
kv_compress_cache 做原地更新操作。

## 约束说明
-  shape 字段取值范围约束
    | 字段名       | 取值规则与说明                                                                 |
    |--------------|-------------------------------------------|
    | d'           |  计算公式如下说明                                               |
    | d          |  取值固定为：512                                                          |
    d'计算公式如下：
$$
ySize = d - 64 + 2 * 64 + Ceil((d - 64) / 128) * 1
$$

$$
d' = (128 - ySize \% 128) \% 128 + ySize
$$

- quant_group_size固定为64。
- 该接口支持推理场景下使用。
- 该接口支持aclgraph入图。
- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
    
## 调用示例

- 详见 [test_npu_kv_compress_epilog.py](../examples/test_npu_kv_compress_epilog.py)