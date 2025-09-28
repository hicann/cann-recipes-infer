# 权重在线切分适配指导

权重在线切分是指模型加载时，系统自动将对应的权重值分发到各个Device，无需预先进行离线处理，从而避免了额外的磁盘存储开销，有效提升调试效率并降低了对磁盘空间的占用。

cann-recipes-infer仓提供了权重在线切分功能的相关代码实现，开发者可以参考本文档适配权重在线切分功能。

## 权重在线切分流程总览
权重在线切分功能的模型初始化及权重加载流程如下图所示：
<p align="center">
  <img src="./figures/load_weights.png" width="50%" alt="weight loading process]">
</p>

为了实现权重在线切分功能，每个相关模块在加载权重时需要获得当前device需要加载的权重切片位置。对于涉及权重切分的Linear、MoEGMM等类，在量化与非量化场景中，它们之间的继承与调用关系如下图所示。
<p align="center">
  <img src="./figures/module_construct.png" width="100%" alt="module construct]">
</p>

## 模型适配修改点
### 替换Linear类和Embedding类
矩阵乘计算公式：

$$
Y=XW+B
$$

其中X为input_activations，shape为(M, K)；W为权重，shape为(K, N)；Y为Linear模块中的矩阵乘输出，shape为(M, N)。

下述说明中切分轴使用M、K、N表示。

根据不同的切分策略将模型脚本中定义的Linear类替换成权重在线切分对应的Linear类，当前`module/Linear.py`中定义了四种不同切分策略下的Linear类和适配TP切分的VocabEmbedding类：

- **ReplicatedLinear**：不涉及TP切分的Linear模块。

- **ColumnParallelLinear**：TP切分时需要切权重的N轴时使用的Linear模块。

- **MergedColumnParallelLinear**：当多个权重需要和同一个input_activations做矩阵乘且切权重N轴时使用的Linear模块，当前支持的merge顺序为先对权重slice再concat。例如，DeepSeek模型中dense层MLP模块的up和gate权重。

- **RowParallelLinear**：TP切分时需要切权重的K轴时使用的Linear模块。

- **VocabParallelEmbedding**：VocabEmbedding类对vocab size做切分。

### 替换MoeGMM类

若模型中涉及到了MoE模块，需要参考本节进行适配。

`module/fuse_moe_gmm.py`中定义了`FusedMoEGMM`类，当前支持纯TP或纯EP切分。

### 修改modeling文件

修改modeling文件，在模型类中增加`load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]])`函数。

该函数遍历了权重文件中的所有权重，并将权重文件中的权重name层级与模型param层级进行匹配，匹配成功后调用模块的`weight_loader()`函数加载权重。若该模块未定义`weight_loader()`函数, 则直接调用`default_weight_loader()`函数进行加载。

大部分模块的权重可通过上述方式加载，但如果存在部分权重加载时需要特殊处理或需要额外传参的场景，可通过定义map的方式在该函数中进行匹配，再对匹配到的模块进行需要的权重处理。例如，使用MergedColumnParallelLinear的模块。

### 适配_process_weight_after_loading()函数

当前基类的`modelrunner`通过遍历模型子模块的方式调用子模块的`process_weights_after_loading()`方法，默认情况下，对所有Linear及GroupedMatmul的权重进行转置，并转换为NZ格式。

如果部分特殊Linear类权重不需要转置或转NZ格式，则需要重写`_process_weight_after_loading()`函数，修改对应子模块函数`process_weights_after_loading()`的传参。

量化场景下如果需要改变Linear类中scale或smooth scale的数据类型，需要给参数scales_dtype传一个dict，dict格式应为`{"scale名称": "target数据类型"}`。

### 配置YAML文件
修改对应用例YAML配置文件中的model_config参数，将`enable_online_split_weight`设置为True，设置如下：
```
enable_online_split_weight: True
```