# LongCat-Flash模型在NPU上推理

## 概述
本样例基于[LongCat-Flash开源代码](https://huggingface.co/meituan-longcat/LongCat-Flash-Chat/blob/main/modeling_longcat_flash.py)进行迁移，并完成对应的优化适配。

## 支持的产品型号
<term>Atlas A3 系列产品</term>

## 环境准备

1. 安装CANN软件包。

   本样例的编译执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（A3-ops）。支持的CANN软件版本为`CANN 8.5.0`。

   请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0)下载`Ascend-cann-toolkit_${version}_linux-${arch}.run`与`Ascend-cann-A3-ops_${version}_linux-${arch}.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Debian&Software=cannToolKit)进行安装。

    - `${version}`表示CANN包版本号，如8.5.0。
    - `${arch}`表示CPU架构，如aarch64、x86_64。

    > [!NOTE]说明 
    > 当前若想使能Attention-FFN Disaggregation(AFD)优化，需要更新`cann-hcomm`包（CommitID `a43e7e9cf42fa264d465147e710eeb6d262f01f4`），参考[hcomm源码编译安装](https://gitcode.com/cann/hcomm/blob/master/docs/build.md)。

2. 安装Ascend Extension for PyTorch（torch_npu）。

   Ascend Extension for PyTorch（torch_npu）为支撑PyTorch框架运行在NPU上的适配插件。
   请从[软件包下载地址](https://gitcode.com/Ascend/pytorch/tree/v2.6.0)下载`v2.6.0`源码（CommitID `71047a90f6e32af5de6268795b73c4cdeb07c4ea`），参考[源码编译安装](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0005.html)。

3. 下载项目源码并安装依赖的python库。
    ```bash
    # 下载项目源码，以master分支为例
    git clone https://gitcode.com/cann/cann-recipes-infer.git

    # 安装依赖的python库
    cd cann-recipes-infer
    pip3 install -r ./models/longcat-flash/requirements.txt
    ```

4. 配置样例运行所需环境信息。

   修改`executor/scripts/set_env.sh`中的如下字段:
   - `IPs`：配置所有节点的IP，按照rank id排序，多个节点的ip通过空格分开，例如：`('xxx.xxx.xxx.xxx' 'xxx.xxx.xxx.xxx')`。
   - `recipes_path`: 当前代码仓根目录，例如`/home/cann-recipes-infer`。
   - `cann_path`: CANN软件包安装路径，例如`/usr/local/Ascend/ascend-toolkit/latest`。

## 权重准备

请根据所使用的模型类型自行下载原始权重到本地路径，例如`/data/models/origin/`。

LongCat-Flash-Chat模型的原始权重下载地址为：[LongCat-Flash-Chat权重](https://huggingface.co/meituan-longcat/LongCat-Flash-Chat/tree/main)

## 权重转换

本样例支持LongCat-Flash模型量化，基于`models/longcat-flash/utils/convert_model.py`可以完成从Bfloat16到Int8的权重转换。

  >入参介绍：`input_bf16_hf_path`：原始Bfloat16权重路径；`output_hf_path`：转换后输出的权重路径。

  如果权重转换的运行环境为NPU，需要先执行：
  ```shell
  cann_path=/usr/local/Ascend/ascend-toolkit/latest # cann包安装路径
  source ${cann_path}/bin/setenv.bash
  ```

  权重转换执行示例：
  ```shell
  # 转换为W8A8权重
  python models/longcat-flash/utils/convert_model.py --input_bf16_hf_path /data/models/LongCat-Flash-Chat --output_hf_path /data/models/LongCat-Flash-Chat-W8A8
  ```

## 推理执行

1. 配置推理执行需要加载的权重文件以及YAML文件。

   - 修改YAML文件中`model_path`参数。

     在`models/longcat-flash/config`目录下已提供了较优性能的YAML样例供您参考，您可以根据模型类型、集群规模以及量化类型选择对应的YAML文件，本文以`models\longcat-flash\config\longcat_flash_densetp8_ep32_gegraph_mtp.yaml`文件为例，修改其中的`model_path`参数，将其设置为[权重准备](#权重准备)阶段准备好的权重文件存储路径，例如`/data/models/origin/LongCat-Flash-Chat/`。

   - 修改`models/longcat-flash/infer.sh`脚本中`YAML_FILE_NAME`参数。

     将`YAML_FILE_NAME`设置为`config`文件夹下YAML文件名称，例如`longcat_flash_densetp8_ep32_gegraph_mtp.yaml`。

2. 准备输入prompt。

   - 使用内置prompt。

     本样例已在`dataset/default_prompt.json`中内置了输入prompt，若您直接使用内置prompt，本步骤可直接跳过。

     当然，您也可以在`dataset/default_prompt.json`文件中自定义prompt输入。

   - 使用长序列prompt。

     本样例默认使用内置prompt，若您需要使用长序列prompt，需要执行以下操作：

     1. 修改YAML文件中的`dataset`参数，将其修改为`dataset: "LongBench"`，使用LongBench数据集作为长序列prompt。

     2. 若您的机器无法联网，需要您从[huggingface](http://huggingface.co/datasets/zai-org/LongBench/tree/main)手动下载数据集至`dataset/LongBench`目录下，`LongBench`文件夹需手工创建，目录中包含`LongBench.py`和`data`目录，并需要在`LongBench.py`中修改数据集加载路径；若您的机器可正常联网，样例执行过程中会自动在线读取LongBench数据集，您无需手工下载。

     > 说明：
     > - 使用LongBench数据集时，默认执行文本摘要任务，可在`cann-recipes-infer/executor/utils/data_utils.py`的`build_dataset_input`函数里修改默认的system prompt。
     > - 长序列请求执行中若出现`out of memory`问题，可参见附录中的[长序列请求out of memory问题处理](#long_bench_faq)。


3. 执行推理脚本。

   ```shell
   cd models/longcat-flash
   bash infer.sh
   ```
   > 说明：如果是多机环境，需要在每个节点上执行。


## 优化点参考

本样例采用的详细优化点介绍及性能Benchmark可参见[基于Atlas A3训练/推理集群的LongCat-Flash模型推理性能优化实践](../../docs/models/longcat-flash/longcat_flash_optimization.md)。

## Benchmark

基于Atlas A3，本实践使用`config/longcat_flash_densetp8_ep128_gegraph_mtp_eplb_w8a8.yaml`作为运行配置文件，对Longcat-Flash W8A8量化版本进行了性能Benchmark测试。
|Quant Mode| Global Batch Size | Seq Length | Chips | TPOT (ms) | Throughput (tokens/p/s) |
|-------| ----------------- | ---------- | ----- | --------- | ----------------------- |
|W8A8 |    512           | 4608       | 64   | 10.37      |   771.46                 |

> 1. 性能数据基于 MTP2 与 perfect eplb 配置采集。
> 2. 当前CANN软件版本（CANN 8.5.0）下，SuperKernel标记范围内的部分算子尚不支持完全融合。该限制将在后续社区版本中得到解决，以进一步提升模型性能。

## 附录

### 常见问题处理

**长序列请求out of memory问题处理<a id="long_bench_faq"></a>**

1. 长序列请求可能导致device内存out of memory，尤其是在perfill阶段，MoE的Routing分发可能存在极端负载不均，导致个别卡上的grouped_matmul算子占用较大内存。为缓解由此引入的OOM问题，可采用以下方法：

- 为了缓解MoE负载不均带来的峰值内存，我们可进行Chunk MoE推理，即在MoE切Chunk串行推理，降低极端场景下的峰值内存，可通过YAML中的`moe_chunk_max_len`开关设置chunk的大小。当前该开关只针对prefill生效，开启后，由于MoE部分将串行计算各chunk，会对prefill的性能产生相应的影响。

2. 长序列请求的KV Cache内存占用过大，单个rank上存储完整的KV Cache导致device内存out of memory，而针对长序列低时延场景支持的KVP特性能够缓解OOM问题：

- 通过使能KVP特性，可实现将长序列的KV Cache沿S维度交错式存储至多个rank，并在Attention计算过程中利用各个rank上分片后的KV Cache完成全局注意力计算，最终达到缓解OOM的效果。KVP切分的大小可通过YAML中的`kvp_size`开关设置。