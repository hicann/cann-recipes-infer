# custom-npu_indexer_compress_epilog

## 产品支持情况

| 产品      | 是否支持 |
|:----------------------------|:-----------:|
|Ascend 950PR/Ascend 950DT|      √     |

## 功能说明

对输入 x 进行量化，根据 slot_mapping 中的下标映射值，将量化后输出的 value 和 scale 分别更新到 indexer_compress_cache 和 indexer_compress_scale 中的对应位置（slot_mapping 的值为 -1 时，表示无效，不执行更新）。该接口的量化算法实现为每128个数进行一次量化。

### 计算公式

$$
index = torch.where(slot\_mapping != -1)
$$

$$
slot\_mapping = slot\_mapping[index]
$$

$$
x = x[index]
$$

$$
value, scale = DynamicQuant(x)
$$

$$
indexer\_compress\_cache[slot\_mapping] = value
$$

$$
indexer\_compress\_scale[slot\_mapping] = scale
$$

## 函数原型
```
custom.indexer_compress_epilog(Tensor(a!) indexer_compress_cache, Tensor(b!) indexer_compress_scale, Tensor x, Tensor slot_mapping,  *, int quant_mode=1, bool round_scale=True) -> ()
```

## 参数说明

>**说明：**<br> 
>
>- b（batch size）表示输入样本批量大小、s（sequence length）表示输入样本序列长度、d（head dimension）表示注意力头的维度数、T表示bs合轴后的大小。

-   **indexer_compress_cache**（`Tensor`）：必选参数，输入tensor，存放量化的value，x量化后的值更新到此Tensor。不支持非连续，数据格式支持ND，数据类型支持`float8_e4m3fn`和`float8_e5m2`，shape为[ T, d ]。

-   **indexer_compress_scale**（`Tensor`）：必选参数，输入tensor，存放量化的scale，x量化后的scale更新到此Tensor。不支持非连续，数据格式支持ND，数据类型支持`float`和`float8_e8m0`，shape为[ T, scale_factor ]。

-   **x**（`Tensor`）：必选参数，输入tensor，x为待量化的数据。不支持非连续，数据格式支持ND，数据类型支持`float16`和`bfloat16`，shape为[ T, d ]。

-   **slot_mapping**（`Tensor`）：必选参数，输入tensor，存放x中每条数据到indexer_compress_cache 和 indexer_compress_scale 的下标映射，x量化后的 value 和 scale 根据下标进行更新。不支持非连续，数据格式支持ND，数据类型支持`int32`，shape为[ T ]，取值范围为[ -1, T )。

- <strong>*</strong>：代表其之前的参数是位置相关的，必须按照顺序输入，属于必选参数；其之后的参数是键值对赋值，与位置无关，属于可选参数（不传入会使用默认值）。

-   **quant_mode**（`int`，可选）：取值范围0或1，取值为1时量化输出indexer_compress_scale为float32类型；取值为0时量化输出indexer_compress_scale为float8_e8m0类型。两种量化均为每128个数进行一次量化。

-   **round_scale**（`bool`, 可选）：是否开启round计算，仅Mx量化时生效，默认为True。


## 返回值说明
无返回值，indexer_compress_cache 和 indexer_compress_scale 做原地操作更新。

## 约束说明
-  shape 字段取值范围约束
    | 字段名       | 取值规则与说明                                                                 |
    |--------------|-------------------------------------------|
    | d           |  取值固定为: 4096                                                |
    | scale_factor          |  取值为d / 128向上取整，固定为32                                                           |
    | quant_mode            |  取值为0或1， 1为普通量化，0为Mx量化          |
- slot_mapping表示下标映射，取值范围为[ -1, T )
- 该接口支持推理场景下使用。
- 在indexer_compress_scale为`float`时，该接口支持aclgraph入图。
- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
    
## 调用示例

- 详见 [test_npu_indexer_compress_epilog.py](../examples/test_npu_indexer_compress_epilog.py)