# custom.sparse\_flash\_attention<a name="ZH-CN_TOPIC_0000001979260729"></a>

## 产品支持情况 <a name="zh-cn_topic_0000001832267082_section14441124184110"></a>
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 推理系列产品</term>   | √  |

## 功能说明<a name="zh-cn_topic_0000001832267082_section14441124184110"></a>

-   **算子功能**：稀疏注意力Sparse Flash Attention的计算。

-   **计算公式**：

    Self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为$n$的输入样本序列$x$，$x$的每个元素都是一个$d$维向量，可以将每个$d$维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为$n*d$的矩阵。

    Sparse Flash Attention的计算由基于索引的离散token与attention计算融合而成。首先，通过$sparseIndices$索引从$key$中取出$key\_sparsed$，从$value$中取出$value\_sparse$，计算self_attention公式如下：

    $$ 
      Attention(query,key,value)=Softmax(\frac{query · key\_sparse^T}{\sqrt{d}})value\_sparse
    $$
    其中$query$和$key\_sparse^T$的乘积代表输入$x$的注意力，为避免该值变得过大，通常除以$d$的开根号进行缩放，并对每行进行softmax归一化，与$value\_sparse$相乘后得到一个$n*d$的矩阵。


## 函数原型<a name="zh-cn_topic_0000001832267082_section45077510411"></a>

```
custom.sparse_flash_attention(Tensor query, Tensor key, Tensor value, Tensor sparse_indices) -> Tensor
```

## 参数说明<a name="zh-cn_topic_0000001832267082_section112637109429"></a>

>**说明：**<br> 
>- query、key、weights参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N。
>- Q_S和S1表示query shape中的S，K_S和S2表示key shape中的S，Q_N表示num\_query\_heads，K_N表示num\_key\_heads。
-   **query**（`Tensor`）：必选参数，不支持非连续的Tensor，数据格式支持ND,数据类型支持`float16`。 
-   **key**（`Tensor`）：必选参数，不支持非连续的Tensor，数据格式支持ND,数据类型支持`float16`。

-   **value**（`Tensor`）：必选参数，不支持非连续的Tensor，数据格式支持ND,数据类型支持`float16`。
    
-   **sparse\_indices**（`Tensor`）：必选参数，代表离散取kvCache的索引，不支持非连续的Tensor，数据格式支持ND，数据类型支持`int32`。

## 返回值说明<a name="zh-cn_topic_0000001832267082_section22231435517"></a>

-   **out**（`Tensor`）：公式中的输出。数据格式支持ND，数据类型支持`float16`。

## 约束说明<a name="zh-cn_topic_0000001832267082_section12345537164214"></a>

-   该接口支持推理场景下使用。
-   该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。

## 调用示例<a name="zh-cn_topic_0000001832267082_section14459801435"></a>

-   详见[test_sparse_flash_attention.py](../examples/test_sparse_flash_attention.py)