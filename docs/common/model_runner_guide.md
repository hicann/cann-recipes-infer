# ModelRunner使用指导

## 概述
在深度学习和自然语言处理领域，模型的迁移效率和运行性能至关重要，随着大语言模型（LLM）的持续迭代更新，推理加速的优化手段需要在不同模型上使能，以适应多样的应用场景。ModelRunner类是cann-recipes-infer提供的模型运行器，它封装了模型的输入输出处理、性能数据采集、权重在线切分及其他自定义优化策略，从而帮助用户实现高效灵活的推理过程。

开发者可以基于ModelRunner类将系列大语言模型方便地迁移到CANN平台并高效运行，本文以gpt_oss为例，详细介绍如何基于ModelRunner类进行适配，并在CANN平台轻量化实现模型推理和性能优化。

## 支持的产品型号
<term>Atlas A3 系列产品</term>

## 类功能和配置结构介绍
### ModelRunner类功能层级

ModelRunner类是一个用于模型推理的基类，主要负责模型的初始化、推理等相关功能的实现。以下是ModelRunner类的功能层级说明：

1. 初始化与设备设置

   - 初始化配置：通过`__init__`方法读取配置文件，初始化模型名称`model_name` 、数据类型`dtype` 、最大位编码长度`max_position_embeddings` 、输入最大长度`input_max_len` 、最大生成长度`max_new_tokens`等参数。

   - 设备初始化：通过`init_device`方法初始化设备及分布式推理环境。

   - 模型初始化：通过`init_model`方法加载预训练模型或从配置文件初始化模型。
     - `_init_model_with_online_splited_weight`：加载原始权重时[在线切分](online_split_weight_guide.md)。
     - `_init_model_with_offline_splited_weight`：加载离线切分过的权重。

   - Tokenizer初始化：通过`init_tokenizer`方法加载tokenizer，支持自动填充和截断策略。


2. 模型处理

   - 数据类型适配：通过`scale_dtype_adapter`方法调整模型量化scale的数据类型，非量化的模型不涉及。

   - 设备迁移：通过`to_device`方法将模型部署到目标设备。

   - 格式转换：通过`cast_format`方法将需要使能`weight_nz`特性的模型权重进行格式转换，以提升推理性能。

   - 图编译：通过`graph_compile`方法使用`torch.compile`进行图编译，支持使能GE图模式或aclgraph模式，提升模型推理速度。


3. 模型推理

   模型多步推理：通过`model_generate`方法实现模型的多步自回归推理，采集模型的性能数据，输出最终结果。

   - 模型输入准备：通过`model_input_prepare`方法准备模型输入。

   - 模型推理：通过`model_inference`方法执行模型推理。

   - 模型输出处理：通过`model_output_process`方法处理模型输出。


4. 优化与性能分析

   - 量化配置解析：通过`_parse_quant_hf_config`方法从模型配置文件中解析量化方法。

   - 量化验证：通过`_verify_quantization`方法验证量化方法是否支持。

   - 性能分析：通过`define_profiler`方法启用性能分析工具，记录模型运行的性能数据。


### runner_settings配置

`runner_settings`是一个字典结构，用于配置模型运行器的运行参数和模型参数。用户可以根据模型的需求自定义 YAML文件来配置`runner_settings`参数结构，以下是gpt_oss的`runner_settings`结构说明：
1. 模型基本配置
   - model_name：模型的名字，用于标识模型。
   - model_path：模型的存储路径，用于加载预训练模型。
   - exe_mode：模型的执行模式，支持单算子或图模式(eager/ge_graph/acl_graph)。
   - world_size：多卡并行推理中的进程数。

2. 模型相关配置：Model Config
   - enable_profiler：是否启用性能分析器，用于记录模型运行的性能数据(True/False)。
   - enable_online_split_weight：支持在线切分权重功能(True/False)。

3. 数据配置：Data Config
   - input_max_len：输入的最大长度，限制输入文本的长度。
   - max_new_tokens：生成的最大新token数，控制生成文本的长度。
   - batch_size：每个批次处理的样本数量。

4. 并行配置：Parallel Config
   - attn_tp_size:多卡推理中的Attention张量并行数。
   - moe_tp_size:多卡推理中的MoE张量并行数。
   - lmhead_tp_size:多卡推理中的LMHead张量并行数。

## 实现步骤
### 获取并使用 runner_settings 参数
以gpt_oss为例，介绍 runner_settings 配置信息的获取和使用流程：
- 读取yaml文件（如./config/gpt_oss_20b.yaml），将其内容解析为字典类型的`runner_settings` ，用户可以自定义检查或更新`runner_settings`参数，确保配置参数的合理性和正确性。
- 将`runner_settings`配置信息传入[GptOssRunner](#GptOssRunner)实例。

  ```python
  runner_settings = read_yaml() # 读取yaml文件
  preset_prompts = ["Explain what is gpt_oss."] # 设置prompts
  model_runner = GptOssRunner(runner_settings) # 将配置参数传入运行器实例
  model_runner.init_model() # 初始化运行器
  model_runner.model_generate(preset_prompts) # 输入prompts进行推理得到结果
  ```


### 继承ModelRunner类并重写关键方法
在模型迁移过程中，继承ModelRunner类并重写关键方法是实现模型推理逻辑的核心步骤。

#### 继承ModelRunner类
<span id="GptOssRunner">以gpt_oss模型为例，新增继承自ModelRunner类的GptOssRunner类。继承后，GptOssRunner类能够拥有并利用父类提供的基础属性和功能方法，且可以根据模型的特点进行扩展和优化。</span>

```python
class GptOssRunner(ModelRunner): # 定义GptOssRunner类，继承自ModelRunner
    def __init__(self, runner_settings):
        super().__init__(runner_settings) # 调用父类初始化方法，传递runner_settings参数
        # 其他初始化操作
```
#### 重写关键方法

在重写父类ModelRunner的方法时，必须重写`model_generate`、`model_input_prepare`和`model_output_process`这三个方法，用户可以参考[transformers库](https://github.com/huggingface/transformers/blob/main/src/transformers)的源码或cann-recipes-infer仓提供的模型实现，其余方法按照用户需求可以自定义重写。

1. 重写model_generate方法

   `model_generate`方法是模型推理的核心，如下图所示，`model_generate`负责处理输入prompt，准备输入，执行推理循环，直到满足终止条件。具体实现细节可参考[gpt_oss](../models/gpt_oss/runner_gpt_oss.py)中的`model_generate`方法。

   <p align="center">
     <img src="./figures/model_generate.png" width="30%" alt="model_generate process]">
   </p>

   - 处理输入提示词：将tokenizer输入最大长度设置为input_max_len，超过这个长度会做截断处理，小于这个长度会做填充处理，使用tokenizer将输入prompt变成token ID 。
   - 执行推理循环：在循环中调用`get_jump_prepare`方法判断是否终止推理，调用`model_input_prepare`方法准备模型输入，执行模型推理`model_inference`方法，获取的输出结果调用`model_output_process`方法进行处理。
   - 解码生成结果：在循环结束后，将输出的token ID解码成文本。

2. 重写model_input_prepare方法

   `model_input_prepare`方法负责准备模型的输入数据，确保输入数据正确传入模型的推理中。在 GptOssRunner 类中，`model_input_prepare`方法调用模型的[prepare_inputs_for_generation](#prepare_inputs_for_generation)方法生成模型输入。

3. 重写model_output_process方法

   `model_output_process`方法负责处理模型的输出，生成新的token，并更新输入字典以便后续的生成步骤使用。

   > 如果是在线切分即`enable_online_split_weight: True`，还需要重写cast_format方法，将部分权重层进行转置或转换为NZ格式，能够有效调整模型权重，方便后续模型推理。

### 模型类修改

在模型迁移适配过程中，为了确保模型能够与Runner运行器兼容，需要对模型进行一些修改，以下将重点介绍初始化阶段传入`runner_settings`和适配`prepare_inputs_for_generation`方法的操作。

#### 初始化传入runner_settings
`runner_settings`是重要的配置参数，包含了并行配置、数据配置等信息，将`runner_settings`添加为模型类的`__init__`方法入参，能够让模型的层级或模块间方便获取到统一的模型参数配置，使得模型具有更高的可配置性，灵活满足多样的需求。
```python
class GptOssForCausalLM(GptOssPreTrainedModel, GenerationMixin):
    def __init__(self, config, runner_settings):
        super().__init__(config)
        self.runner_settings = runner_settings
        self.model = GptOssModel(config, runner_settings)
        self.lm_head_tp_size = runner_settings.get("parallel_config").get("lmhead_tp_size", 1)
        # 其他初始化操作
```


#### 适配prepare_inputs_for_generation方法
<span id="prepare_inputs_for_generation">`prepare_inputs_for_generation`方法需要处理模型输入参数数据，并将这些参数更新到model_inputs字典中，确保将这些参数在调用`forward`方法时已经正确准备。</span>
