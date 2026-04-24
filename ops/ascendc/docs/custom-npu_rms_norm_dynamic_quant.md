# custom-npu_rms_norm_dynamic_quant<a name="ZH-CN_TOPIC_0000001979260729"></a>

## 产品支持情况 <a name="zh-cn_topic_0000001832267082_section14441124184110"></a>
| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 推理系列产品</term>   | √  |

## 功能说明<a name="zh-cn_topic_0000001832267082_section14441124184110"></a>

`RmsNormDynamicQuant`RmsNorm算子是大模型常用的归一化操作。DynamicQuant算子则是为输入张量进行对称动态量化的算子。RmsNormDynamicQuant算子将RmsNorm归一化输出给到DynamicQuant算子融合起来，减少搬入搬出操作，`RmsNormDynamicQuant`的具体计算公式如下：

  $$
  y = \operatorname{RmsNorm}(x)=\frac{x}{\operatorname{Rms}(\mathbf{x})}\cdot gamma+beta, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  $$
  input =\begin{cases}
    y\cdot smoothScaleOptional & \ \ smoothScaleOptional \\
    y & !\ smoothScaleOptional
    \end{cases}
  $$

  $$
  scaleOut=row\_max(abs(input))/127
  $$

  $$
  yOut=round(input1/scaleOut)
  $$

  公式中的row\_max代表每行求最大值。

## 函数原型<a name="zh-cn_topic_0000001832267082_section45077510411"></a>

```
custom.npu_rms_norm_dynamic_quant(Tensor x, Tensor gamma, *, Tensor? smooth_scale=None, Tensor? beta=None, float epsilon=1e-6) -> (Tensor, Tensor)
```

## 参数说明<a name="zh-cn_topic_0000001832267082_section112637109429"></a>

-   **x**（`Tensor`）：公式中的输入x，必选参数，不支持非连续，数据格式支持ND，数据类型支持`float16、bfloat16`。
    
-   **gamma**（`Tensor`）：公式中的gamma，必选参数，不支持非连续，数据格式支持ND，数据类型支持`float16、bfloat16`，要求是1D的Tensor，数据类型同`x`保持一致，shape同`x`最后一维一致。

- <strong>*</strong>：代表其之前的参数是位置相关的，必须按照顺序输入，属于必选参数；其之后的参数是键值对赋值，与位置无关，属于可选参数（不传入会使用默认值）。
    
-   **smooth_scale**（`Tensor`）：公式中的smoothScaleOptional，可选参数，不支持非连续，数据格式支持ND，数据类型支持`float16、bfloat16`，要求是1D的Tensor。shape和数据类型同`gamma`保持一致。

-   **beta**（`Tensor`）：公式中的beta，表示标准化过程中的偏置项；可选参数，不支持非连续，数据格式支持ND，数据类型支持`float16、bfloat16`，要求是1D的Tensor。shape和数据类型同`gamma`保持一致。。

-   **epsilon**（`float`）：公式中的`epsilon`，表示用于防止除0错误；可选参数，默认值1e-6。

## 返回值说明<a name="zh-cn_topic_0000001832267082_section22231435517"></a>

-   **y**（`Tensor`）：公式中的输出yOut，表示量化后的输出tensor，数据类型支持`int8`。数据格式支持ND，shape需要与输入`x`保持一致。
-   **scale**（`Tensor`）：公式中的输出scaleOut，表示量化scale参数，数据类型支持`float32`。数据格式支持ND,shape需要与输入`x`除了最后一维后的shape一致，或者与`x`除了最后一维的乘积一致。

## 约束说明<a name="zh-cn_topic_0000001832267082_section12345537164214"></a>

-   该接口支持推理场景下使用。
-   该接口支持aclgraph入图。
-   该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。

## 调用示例<a name="zh-cn_topic_0000001832267082_section14459801435"></a>

-   详见[test_npu_rms_norm_dynamic_quant.py](../examples/test_npu_rms_norm_dynamic_quant.py)