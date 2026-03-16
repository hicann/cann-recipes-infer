# 在NPU环境上适配HunyuanImage-3.0模型的推理

## 概述
[HunyuanImage-3.0](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/assets/HunyuanImage_3_0.pdf)是腾讯于2025年9月28日正式开源的一个突破性的原生多模态模型，它在自回归框架内统一了多模态理解和生成任务。它的文生图能力实现了与领先的闭源模型相当或更优的性能。本项目旨在提供HunyuanImage-3.0的昇腾适配版本。

## 支持的产品型号
<term>Atlas A2/A3 系列产品</term>

## 环境准备

1. 安装CANN软件包。

   本样例的编译执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），支持的CANN软件版本为`CANN 9.0.0-beta.1`。

   请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0-beta.1)下载`Ascend-cann-toolkit_9.0.0-beta.1_linux-${arch}.run`与`Atlas-A3-ops_9.0.0-beta.1_linux-${arch}.run`（A3环境）或`Ascend-cann-910b-ops_9.0.0-beta.1-${arch}.run`（A2环境）软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=openEuler)进行安装。

    - `${arch}`表示CPU架构，如aarch64、x86_64。

2. 安装Ascend Extension for PyTorch（torch_npu）。

   Ascend Extension for PyTorch（torch_npu）为支撑PyTorch框架运行在NPU上的适配插件，本样例建议使用的PyTorch版本为`2.7.1`，相应的Ascend Extension for PyTorch版本为`v7.3.0.1-pytorch2.7.1`。可以通过`pip install torch_npu==2.7.1`进行安装，或者从[软件包下载地址](https://gitcode.com/Ascend/pytorch/releases/v7.3.0.1-pytorch2.7.1)下载`release v7.3.0.1-pytorch2.7.1`源码，参考[源码编译安装](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/installation_description.md)。

3. 下载项目源码并安装依赖。

    ```bash
    # 下载项目源码，以master分支为例
    git clone https://gitcode.com/cann/cann-recipes-infer.git

    # 本仓库依赖于[HunyuanImage-3.0](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0)的开源仓库代码。进入HunyuanImage-3.0的仓库，下载开源仓库代码：
    git clone https://github.com/Tencent-Hunyuan/HunyuanImage-3.0.git
    cd HunyuanImage-3.0
    git reset --hard 62da220178f4b0b7d83e91665a46a20a3ee4f7cd
    cd ..

    # 将HunyuanImage-3.0仓库的代码以“非覆盖模式”复制到本项目目录下：
    cp -rn HunyuanImage-3.0/* cann-recipes-infer/models/hunyuan-image-3.0/

    # 安装依赖的python库
    cd cann-recipes-infer
    pip install -r ./models/hunyuan-image-3.0/requirements.txt
    ```
    请注意，实测发现`transformers==5.3.0`版本会产生精度问题，推荐使用`5.2.0`及以下版本。

4. 配置样例运行所需环境信息。

   修改`executor/scripts/set_env.sh`中的如下字段:
   - `IPs`：配置所有节点的IP，按照rank id排序，多个节点的ip通过空格分开，例如：`('xxx.xxx.xxx.xxx' 'xxx.xxx.xxx.xxx')`。
   - `cann_path`: CANN软件包安装路径，例如`/usr/local/Ascend/ascend-toolkit/latest`。

   > 说明：HCCL相关配置，如：`HCCL_SOCKET_IFNAME`、`HCCL_OP_EXPANSION_MODE`，可以参考[集合通信文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/maintenref/envvar/envref_07_0001.html#ZH-CN_TOPIC_0000002449945377__section163522499503)并在`executor/scripts/function.sh`中自定义配置。

## 权重准备与转换

| 模型             | 版本                                                                                           |
|------------------|------------------------------------------------------------------------------------------------|
| HunyuanImage-3.0 | [HunyuanImage-3.0](https://huggingface.co/tencent/HunyuanImage-3.0#download-pretrained-models) |

下载HunyuanImage-3.0模型权重到本地路径`ckpts`。

```
hunyuan-image-3.0/
├── hunyuan_image_3/
|   └──...
├── utils/
│   └──...
├── ckpts/
|   └──...
└──...
```

使用`weight_convert.sh` 脚本完成权重转换:

```
python convert_model.py \
    --model-path ../ckpts/HunyuanImage-3.0 \   # 模型原始权重路径
    --output-path ../ckpts/weight_tp8 \        # 模型权重输出路径
    --tp-attn 8 \                              # attn TP 切分份数
    --tp-moe 8 \                               # moe TP 切分份数
    --ep 1 \                                   # EP 切分份数
    --max-shard-size 5.0                       # 权重每块最大限制（G）
```
在`convert_model.py`中使用了多线程技术提高并发读写速度，如果默认的并发数量性能不佳，请根据调测环境的实际情况合理设置`max_workers`的值。

权重转换拉起示例：

```shell
cd utils
bash weight_convert.sh
```

## 推理执行

本样例准备了多卡的推理脚本，最少需要4个Device可正常运行，若需要开启CFG_PARALLEL，则需要8个Device可正常运行。

1、设置环境变量

   通过设置环境变量`export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`指定启用哪几个Device进行推理，更多环境变量相关问题，请参考[CANN社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/maintenref/envvar/envref_07_0028.html)。

   通过设置环境变量`export CFG_PARALLEL=0`，控制是否开启CFG并行功能，0表示关闭，1表示开启。

   通过设置环境变量`export USE_VAE_PARALLEL=1`，控制是否开启VAE并行功能，0表示关闭，1表示开启。

   通过设置环境变量`export CPU_AFFINITY_CONF=1`，控制[自动绑核](https://www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/Frameworkfeatures/docs/zh/framework_feature_guide_pytorch/automatic_core_binding.md)，通过避免CPU核心抢占来降低出现HOST下发瓶颈的概率。

2、执行推理脚本

   除了上述的环境变量外，脚本中使用torchrun拉起推理的命令还有许多入参，其中`moe-tp`与`moe-ep`是一对互斥参数，只能设置其中一个：
   - `moe-tp`或缺省：代表MoE部分使用TP（Tensor Parallel，张量并行）；
   - `moe-ep`：代表MoE部分使用EP（Expert Parallel，专家并行）。
   脚本中其余入参的设置详见`run_image_gen.py`文件中`parse_args`函数中的参数说明。

   在`run_image_gen.py`中，以循环的形式对`model.generate_image()`调用了多次，第一次可以视为warmup，后续的性能为稳态性能，而就精度而言则则每次都是一致的。

   ```shell
   cd models/hunyuan-image-3.0

   # 开启CFG并行，开启VAE并行示例
   bash demo.sh
   ```

   本样例测试结果如下：

   |            Model             |   Environment  |  TP/EP   | CFG Parallel | VAE Parallel |     E2E      |
   | :--------------------------: | :------------: | :------: | :----------: | :----------: | :----------: |
   |       hunyuan-image-3.0      |       A3       |   TP8    |    enable    |    enable    |    10.3s     |
   |       hunyuan-image-3.0      |       A3       |   EP8    |    enable    |    enable    |     9.9s     |

3、性能分析

   如果需要分析profiling，可以将`adaptor_patches/hunyuan_image_e_pipeline.py`中的`enable_prof = False if idx_round > 1 else False`改为`enable_prof = True if idx_round > 1 else False`，并确保`run_image_gen.py`中对`model.generate_image()`调用2次以上，这样在第2次及以后便会自动采集profiling数据，相关配置可以在adaptor_patches/hunyuan_image_e_pipeline.py中配置，参考[torch_npu.profiler接口](https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu-profiler_list.md)。

## 优化点参考

本样例采用的详细优化点介绍可参见[NPU HunyuanImage-3.0模型推理优化实践](../../docs/models/hunyuan-image-3.0/hunyuan_image_3_optimization.md)。