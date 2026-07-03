# LongCat-Flash模型在NPU上推理

## 概述
本样例基于[LongCat-Flash开源代码](https://huggingface.co/meituan-longcat/LongCat-Flash-Chat/blob/main/modeling_longcat_flash.py)进行迁移，适配到cann-recipes-infer推理框架，并完成对应优化。

## 支持的产品型号
<term>Atlas A3 系列产品</term>

## 环境准备

1. 安装CANN软件包。

   本样例的编译执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（A3-ops）。

   请从[软件包下载地址](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-release/software/master)下载`Ascend-cann-toolkit_${version}_linux-${arch}.run`与`Ascend-cann-A3-ops_${version}_linux-${arch}.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Debian&Software=cannToolKit)进行安装。

    - `${version}`表示CANN包版本号，如9.0.0等。
    - `${arch}`表示CPU架构，如aarch64、x86_64。

2. 安装Ascend Extension for PyTorch（torch_npu）。

   Ascend Extension for PyTorch（torch_npu）为支撑PyTorch框架运行在NPU上的适配插件。
   请从[软件包下载地址](https://gitcode.com/Ascend/pytorch/releases/v26.0.0-pytorch2.8.0)下载`v2.8.0.post4` whl包，参考[离线安装(Whl)](https://www.hiascend.com/document/detail/zh/Pytorch/2600/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md)。

3. 下载项目源码并安装依赖的python库。
    ```bash
    # 下载项目源码，以master分支为例
    git clone https://gitcode.com/cann/cann-recipes-infer.git

    # 安装依赖的python库
    cd cann-recipes-infer
    pip3 install -r ./models/longcat_flash/requirements.txt
    ```

4. 配置样例运行所需环境信息。

   修改`executor/scripts/set_env.sh`中的如下字段:
   - `IPs`：配置所有节点的IP，按照rank id排序，多个节点的ip通过空格分开，例如：`('xxx.xxx.xxx.xxx' 'xxx.xxx.xxx.xxx')`。
   - `PREFILL_IPS`: 配置Online场景的Prefill节点IP。
   - `DECODE_IPS`: 配置Online场景的Decode节点IP。
   - `cann_path`: CANN软件包安装路径，例如`/usr/local/Ascend/ascend-toolkit/latest`。

## 权重准备

请根据所使用的模型类型自行下载原始权重到本地路径，例如`/data/models/origin/`。

LongCat-Flash-Chat模型的原始权重下载地址为：[LongCat-Flash-Chat权重](https://huggingface.co/meituan-longcat/LongCat-Flash-Chat/tree/main)

## 权重转换

本样例支持LongCat-Flash模型量化，基于`models/longcat_flash/utils/convert_model.py`可以完成从Bfloat16到W8A8Int8的权重转换。

> 入参介绍：`input_bf16_hf_path`：原始Bfloat16权重路径；`output_hf_path`：转换后输出的权重路径。

如果权重转换的运行环境为NPU，需要先执行：

```bash
cann_path=/usr/local/Ascend/ascend-toolkit/latest # cann包安装路径
source ${cann_path}/bin/setenv.bash
```

权重转换执行示例：

```bash
# 转换为W8A8权重
python models/longcat_flash/utils/convert_model.py --input_bf16_hf_path /data/models/LongCat-Flash-Chat --output_hf_path /data/models/LongCat-Flash-Chat-W8A8
```

## 推理执行

1. 配置推理执行需要加载的权重文件以及YAML文件。

   - 修改YAML文件中`model_path`参数。

     在`models/longcat_flash/config`目录下已提供了较优性能的YAML样例供您参考，您可以根据模型类型、集群规模以及量化类型选择对应的YAML文件，本文以`models/longcat_flash/config/longcat_flash_densetp8_ep64_gegraph_mtp2_eplb_afd_w8a8.yaml`文件为例，修改其中的`model_path`参数，将其设置为[权重转换](#权重转换)阶段准备好的权重文件存储路径，例如`/data/models/origin/LongCat-Flash-Chat-W8A8/`。

     - YAML文件中的配置说明可见[YAML参数描述](../../docs/common/inference_config_guide.md)。

     - 除框架统一配置之外，还额外支持以下特性，放置在 YAML 文件 `model_config` 的 `custom_params` 字段下：

     | 参数名 | 位置 | 含义 |
     | --- | --- | --- |
     | `enable_mla_prolog` | `model_config.custom_params` | 启用MLA Prolog算子。 |
     | `enable_prefetch` | `model_config.custom_params` | 启用权重预取。 |
     | `enable_multi_streams` | `model_config.custom_params` | 启用多流优化。 |
     | `enable_superkernel` | `model_config.custom_params` | 启用superkernel优化。 |
     | `enable_afd` | `model_config.custom_params` | 启用Attn FFN分离优化。 |
     | `moe_chunk_max_len` | `model_config.custom_params` | 设置MoE chunk大小，用于缓解长序列场景下的峰值内存压力。 |

   - 配置`executor/scripts/infer.sh`脚本中的参数。

     离线推理模式下，将`--yaml`设置为`config`文件夹下YAML文件名称，例如`longcat_flash_densetp8_ep64_gegraph_mtp2_eplb_afd_w8a8.yaml`。
     在线推理模式下，将`--mode`设置为`online`，`--pd-role`设置为`prefill`或`decode`，可通过`--p-yaml-name`和`--d-yaml-name`指定prefill/decode的YAML文件。

2. 准备输入prompt。

   - 使用内置prompt。

     本样例已在`dataset/default_prompt.json`中内置输入prompt。若您直接使用内置prompt，可将YAML中的`data_config.dataset`设置为`default`。

     当然，您也可以在`dataset/default_prompt.json`中自定义prompt输入。

   - 使用长序列prompt。

     若您需要使用长序列prompt，需要执行以下操作：

     1. 修改YAML文件中的`data_config.dataset`参数，将其修改为`LongBench`，使用LongBench数据集作为长序列prompt。

     2. 若您的机器无法联网，需要您从[huggingface](https://huggingface.co/datasets/zai-org/LongBench/tree/main)手动下载数据集至`dataset/LongBench`目录下，`LongBench`文件夹需手工创建，目录中包含`LongBench.py`和`data`目录，并需要在`LongBench.py`中修改数据集加载路径；若您的机器可正常联网，样例执行过程中会自动在线读取LongBench数据集，您无需手工下载。

     > 说明：
     > - 使用LongBench数据集时，默认执行文本摘要任务，可在`cann-recipes-infer/executor/utils/data_utils.py`的`build_dataset_input`函数里修改默认的system prompt。
     > - 长序列请求执行中若出现`out of memory`问题，可参见附录中的[长序列请求out of memory问题处理](#long_bench_faq)。

3. 执行推理脚本。

   以下命令需在仓库根目录下执行。

   入口脚本位于`executor/scripts/infer.sh`，通过以下参数控制启动：

   | 参数 | 含义 | 取值示例 |
   | --- | --- | --- |
   | `--model` | 模型目录名，对应`models/`下的子目录 | `longcat_flash` |
   | `--mode` | 推理模式 | `offline`（离线推理） / `online`（在线PD分离推理） |
   | `--yaml` | 离线模式：YAML文件名 | `longcat_flash_densetp8_ep64_gegraph_mtp2_eplb_afd_w8a8.yaml` |
   | `--pd-role` | 在线模式：PD部署角色 | `prefill` / `decode` |
   | `--p-yaml-name` | 可选，在线模式：prefill YAML文件名，不传则默认`longcat_flash_pd/prefill.yaml` | `longcat_flash_pd/prefill.yaml` |
   | `--d-yaml-name` | 可选，在线模式：decode YAML文件名，不传则默认`longcat_flash_pd/decode.yaml` | `longcat_flash_pd/decode.yaml` |

   > 在线模式IP等更多配置参见[executor设计文档 §5.1 启动方式](../../docs/design/executor_design.md#51-启动方式)。

   **使用方式一：命令行传参**

   ```bash
   # offline模式
   bash executor/scripts/infer.sh --model longcat_flash --yaml longcat_flash_densetp8_ep64_gegraph_mtp2_eplb_afd_w8a8.yaml

   # online模式
   bash executor/scripts/infer.sh --model longcat_flash --mode online --pd-role prefill

   # online模式（指定prefill/decode YAML）
   bash executor/scripts/infer.sh --model longcat_flash --mode online --pd-role prefill --p-yaml-name longcat_flash_pd/prefill.yaml --d-yaml-name longcat_flash_pd/decode.yaml
   ```

   如需查看参数说明，可以执行`bash executor/scripts/infer.sh --help`。

   **使用方式二：直接修改脚本默认值后执行**

   编辑`executor/scripts/infer.sh`，按照需求修改`MODEL`、`MODE`、`YAML_FILE`、`PD_ROLE`、`P_YAML_NAME`、`D_YAML_NAME`等参数的默认值，例如：

   ```bash
   MODEL=longcat_flash
   MODE=offline
   YAML_FILE=longcat_flash_densetp8_ep64_gegraph_mtp2_eplb_afd_w8a8.yaml
   ```

   保存后直接执行：

   ```bash
   bash executor/scripts/infer.sh
   ```

   > 说明：
   > - 如果是多机环境，需要在每个节点上执行。
   > - 推理日志和结果保存在`models/longcat_flash/res/`路径下。

## 优化点参考

本样例采用的详细优化点介绍及性能Benchmark可参见[基于Atlas A3训练/推理集群的LongCat-Flash模型推理性能优化实践](../../docs/models/longcat_flash/longcat_flash_optimization.md)。

## Benchmark

基于 Atlas A3 环境，本实践对 LongCat-Flash W8A8量化版本进行了性能 Benchmark 测试。
在使能 AFD 特性后，模型的 TPOT 迈入了 10 ms 之内, 并且相比于同样卡数和 global batch size 的不分离场景，拥有更优的 TPOT 和吞吐。

|Enable AFD|Quant Mode| Global Batch Size | Seq Length | Chips | TPOT (ms) | Throughput (tokens/p/s) |
|---|-------| ----------------- | ---------- | ----- | --------- | ----------------------- |
| N |  W8A8 |    512            | 4608       | 64    | 10.37     |   771.46                |
| N |  W8A8 |    256            | 4608       | 32    | 10.64     |   751.88                |
| N |  W8A8 |    256            | 4608       | 64    | 9.95      |   402.01                |
| Y |  W8A8 |    256            | 4608       | 64    | 9.5       |   421.05                |

> 1. 性能数据基于 MTP2 与 perfect eplb 配置采集。
> 2. 当前 CANN 软件版本下，SuperKernel 标记范围内的部分算子尚不支持完全融合。该限制将在后续社区版本中得到解决，以进一步提升模型性能。
> 3. 由于当前 Send/Recv 算子单次通信只支持1:1的发送/接收模式，不支持 M:N 模式，所以 AFD 场景部署时 Attention Instance 的 Node 个数 和 FFN Instance 的 Node 个数是一样，也即 M == N；后续会计划支持 M:N 的部署模式。
---

## 附录

### 常见问题处理

**长序列请求out of memory问题处理<a id="long_bench_faq"></a>**

1. 长序列请求可能导致device内存out of memory，尤其是在prefill阶段，MoE的Routing分发可能存在极端负载不均，导致个别卡上的grouped_matmul算子占用较大内存。为缓解由此引入的OOM问题，可采用以下方法：

   - 为了缓解MoE负载不均带来的峰值内存，我们可进行Chunk MoE推理，即在MoE切Chunk串行推理，降低极端场景下的峰值内存，可通过YAML中的`moe_chunk_max_len`开关设置chunk的大小。当前该开关只针对prefill生效，开启后，由于MoE部分将串行计算各chunk，会对prefill的性能产生相应的影响。

**HCCL_BUFFSIZE不足问题**

如果报错日志中出现关键字`HCCL_BUFFSIZE is too SMALL, ..., NEEDED_HCCL_BUFFSIZE..., HCCL_BUFFSIZE=200MB, ...`，可通过配置环境变量`HCCL_BUFFSIZE`解决，所有rank上的该环境变量需保持一致。HCCL_BUFFSIZE参数介绍可参考[昇腾资料](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/maintenref/envvar/envref_07_0080.html)。
