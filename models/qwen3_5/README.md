# Qwen3.5-MoE模型在NPU实现推理

## 概述

Qwen3.5-MoE模型是Qwen3.5系列中的混合专家模型。本样例基于transformers库的Qwen3.5-MoE模型结构完成文本侧推理适配，支持Qwen3.5-35B-A3B模型在NPU上的多卡推理，并提供自定义prompt输入与MMLU评测脚本。


## 支持的产品型号

<term>Atlas A3 系列产品</term>

## 环境准备

1. 安装CANN软件包。

   本样例的编译执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），支持的CANN软件版本为`CANN 9.0.0`。

   请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.3.RC1.alpha002)下载`Ascend-cann-toolkit_${version}_linux-${arch}.run`与`Atlas-A3-cann-kernels_${version}_linux-${arch}.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Debian&Software=cannToolKit)进行安装。

   - `${version}`表示CANN包版本号，如`9.0.0`。
   - `${arch}`表示CPU架构，如`aarch64`、`x86_64`。

2. 安装Ascend Extension for PyTorch（torch_npu）。

   Ascend Extension for PyTorch（torch_npu）为支撑PyTorch框架运行在NPU上的适配插件，本样例支持的Ascend Extension for PyTorch版本为`v26.0.0`，PyTorch版本为`2.8.0`。
   请从[软件包下载地址](https://gitcode.com/Ascend/pytorch/releases/v26.0.0-pytorch2.8.0)下载`torch_npu-2.8.0.post4-cp311-cp311-manylinux_2_28_${arch}.whl`安装包，参考[torch_npu安装文档](https://www.hiascend.com/document/detail/zh/Pytorch/2600/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md)进行安装。

3. 下载项目源码并安装依赖的Python库。

   ```bash
   # 下载项目源码
   git clone https://gitcode.com/cann/cann-recipes-infer.git
   # 安装依赖的python库，仅支持python 3.11
   cd cann-recipes-infer


   pip3 install -r ./models/qwen3_5/requirements.txt
   ```


4. 配置样例运行所需环境信息。

   修改`executor/scripts/set_env.sh`中的如下字段：

   - `IPs`：配置所有节点的IP，按照rank id排序，多个节点的IP通过空格分开，例如：`('xxx.xxx.xxx.xxx' 'xxx.xxx.xxx.xxx')`。
   - `cann_path`：CANN软件包安装路径，例如`/usr/local/Ascend/ascend-toolkit/latest`。

   `set_env.sh`会自动将当前代码仓根目录加入`PYTHONPATH`。HCCL相关配置，如`HCCL_SOCKET_IFNAME`、`HCCL_OP_EXPANSION_MODE`，可参考[集合通信文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/maintenref/envvar/envref_07_0001.html#ZH-CN_TOPIC_0000002449945377__section163522499503)并在`executor/scripts/function.sh`中按实际网卡和集群环境自定义配置。

## 权重准备
请根据所使用的模型类型自行下载原始权重到本地路径，例如`/data/models/Qwen3.5-35B-A3B`。

以Qwen3.5-35B-A3B为例，权重下载地址：[Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/tree/main)。


## 推理执行

### 1. 选择并修改YAML配置

修改YAML文件中`model_path`参数。关于YAML文件中的更多配置说明可参见[YAML参数描述](../../docs/common/inference_config_guide.md)。

在`models/qwen3_5/config`目录下已提供多组YAML样例供您参考，您可以根据模型规模、卡数、并行策略、输入长度和batch size选择对应的YAML文件：

- `qwen3_5_35b_ep8.yaml`：Qwen3.5-35B-A3B，EP8配置。
- `qwen3_5_122b_tp8.yaml`：Qwen3.5-122B-A10B，TP8配置。
- `qwen3_5_397b_tp16.yaml`：Qwen3.5-397B-A17B，TP16配置。

本文以`models/qwen3_5/config/qwen3_5_35b_ep8.yaml`文件为例，修改其中的`model_path`参数，将其设置为[权重准备](#权重准备)阶段准备好的权重文件存储路径，例如`/data/models/Qwen3.5-35B-A3B`。

### 2. 准备输入prompt

默认情况下，样例可使用`dataset/default_prompt.json`中的内置prompt。

如需使用自定义prompt，可准备一个包含`default_prompt.json`的目录，并在YAML中设置`data_config.dataset_path`为该目录路径。`default_prompt.json`内容为JSON数组，示例：

```json
[
  "Summarize what attention does in one sentence.",
  "What is tensor parallelism? Answer in one sentence.",
  "Explain Mixture-of-Experts briefly.",
  "What is the purpose of a KV cache in decoding?"
]
```

样例通过 YAML 文件中的`data_config.input_truncated_len`参数设置最大输入长度。在读取 prompt 时，若其 token 长度超过该阈值，则会自动进行截断。

### 3. 配置启动参数

统一入口脚本位于`executor/scripts/infer.sh`，可通过命令行参数指定模型目录和YAML文件。

8卡执行示例：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
bash executor/scripts/infer.sh --model qwen3_5 --yaml qwen3_5_35b_ep8.yaml
```

16卡执行示例：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
bash executor/scripts/infer.sh --model qwen3_5 --yaml qwen3_5_397b_tp16.yaml
```

### 4. 执行推理脚本

进入仓库根目录，执行统一推理脚本：

```bash
cd cann-recipes-infer
bash executor/scripts/infer.sh --model qwen3_5 --yaml qwen3_5_35b_ep8.yaml
```

> 说明：如果是多机环境，需要在每个节点上执行。脚本会根据YAML中的`world_size`和`executor/scripts/set_env.sh`中的IP列表计算各节点rank信息。

推理输出会打印在rank 0日志中，同时日志会保存到仓库当前工作目录下的`res/${DATE}/${CASE_NAME}/log_*.log`。

如需额外保存输入输出JSON，可在启动前设置：

```bash
export IO_DUMP_PATH=/tmp/qwen3_5_io_dump.json
bash executor/scripts/infer.sh --model qwen3_5 --yaml qwen3_5_35b_ep8.yaml
```

## MMLU评测

`models/qwen3_5/benchmark`目录提供了MMLU评测入口。评测前请准备好[MMLU数据](https://people.eecs.berkeley.edu/~hendrycks/data.tar)。

>  注意事项：框架默认开启thinking模式。如需使用本脚本进行精度测试，需要关闭thinking模式。请在`executor/core/scheduler/scheduler.py`的`tokenize_request`方法中，将`apply_chat_template`调用添加`enable_thinking=False`参数：
>
> ```python
> prompt_text = self.tokenizer.apply_chat_template(
>    prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False
> )
> ```

执行评测：

```bash
cd cann-recipes-infer
python3 models/qwen3_5/benchmark/run_mmlu_benchmark.py \
  --mmlu-dir /data/MMLU/data \
  --yaml-path models/qwen3_5/benchmark/qwen3_5_35b_mmlu_test.yaml \
  --visible-devices 0,1 \
  --output-name mmlu_eval_results.json
```

常用参数：

- `--split`：评测数据划分，可选`test`或`val`，默认`test`。
- `--ntrain`：few-shot样例数量，默认`5`。
- `--max-subjects`：限制评测科目数量，默认`0`表示不限制。
- `--max-prompts`：限制总评测题目数量，默认`0`表示不限制。
- `--max-examples-per-subject`：限制每个科目的题目数量，默认`0`表示不限制。
- `--visible-devices`：评测使用的NPU卡号，默认读取环境变量`ASCEND_RT_VISIBLE_DEVICES`，未设置时使用`0,1`。
- `--output-name`：rank 0保存的评测结果文件名，默认`mmlu_eval_results.json`。

评测结果默认保存到`models/qwen3_5/benchmark/${output-name}`，内容包含总准确率、分科目准确率和逐题预测详情。注意，开启`HCCL_OP_EXPANSION_MODE=AIV`可能会引入不确定计算，导致测评结果出现波动，如有需要请关闭该功能。


## 优化点参考

本样例采用的详细优化点介绍可参见[Qwen3.5-MoE模型推理性能优化实践](../../docs/models/qwen3_5/qwen3_5_moe_optimization.md)。
