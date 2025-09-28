# custom.lightning\_indexer<a name="ZH-CN_TOPIC_0000001979260729"></a>

## 产品支持情况 <a name="zh-cn_topic_0000001832267082_section14441124184110"></a>
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 推理系列产品</term>   | √  |

## 功能说明<a name="zh-cn_topic_0000001832267082_section14441124184110"></a>

-   算子功能：用于高效处理索引数据。
-   计算公式：

    $$
      Indices(query,key,weights)=Topk(broadcast\_vmul(relu(query · key)), weights)
    $$

## 函数原型<a name="zh-cn_topic_0000001832267082_section45077510411"></a>

```
custom.lightning_indexer(query, key, weights) -> Tensor
```

## 参数说明<a name="zh-cn_topic_0000001832267082_section112637109429"></a>

>**说明：**<br> 
>- query、key、weights参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N。

-   **query**（`Tensor`）：必选参数，不支持非连续的Tensor，数据格式支持ND，数据类型支持`float16`。
    
-   **key**（`Tensor`）：必选参数，不支持非连续的Tensor，数据格式支持ND，数据类型支持`float16`。
    
-   **weights**（`Tensor`）：必选参数，不支持非连续的Tensor，数据格式支持ND，数据类型支持`float16`。

## 返回值说明<a name="zh-cn_topic_0000001832267082_section22231435517"></a>

-   **out**（`Tensor`）：公式中的输出，数据类型支持`int32`。数据格式支持ND。

## 约束说明<a name="zh-cn_topic_0000001832267082_section12345537164214"></a>

-   该接口支持推理场景下使用。
-   该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
-   参数key、value的N仅支持1。
-   参数query中的D和key的D值相等为128。

## 调用示例<a name="zh-cn_topic_0000001832267082_section14459801435"></a>

-   详见[test_lightning_indexer.py](../examples/test_lightning_indexer.py)

