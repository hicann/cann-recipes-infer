# 在昇腾Atlas A3环境上适配HunyuanImage-3.0模型的推理

## 概述
HunyuanImage-3.0 是腾讯于2025年9月28日正式开源的一个突破性的原生多模态模型，它在自回归框架内统一了多模态理解和生成任务。它的文生图能力实现了与领先的闭源模型相当或更优的性能。本项目旨在提供HunyuanImage-3.0的昇腾适配版本。

## 支持的产品型号
<term>Atlas A2/A3 系列产品</term>

## 环境准备

1. 安装CANN软件包。

   本样例的编译执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），支持的CANN软件版本为`CANN 8.5.0.alpha002`。

   请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0.alpha002)下载`Ascend-cann-toolkit_${version}_linux-${arch}.run`与`Atlas-A3-cann-kernels_${version}_linux-${arch}.run`（A3环境）或`Ascend-cann-kernels-910b_8.5.0.alpha002_linux-aarch64.run`（A2环境）软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=openEuler&Software=cannToolKit)进行安装。

    - `${version}`表示CANN包版本号，如8.5.0.alpha002。
    - `${arch}`表示CPU架构，如aarch64、x86_64。

2. 安装Ascend Extension for PyTorch（torch_npu）。

   Ascend Extension for PyTorch（torch_npu）为支撑PyTorch框架运行在NPU上的适配插件，本样例支持的Ascend Extension for PyTorch版本为`v7.2.0.1-pytorch2.7.1`，PyTorch版本为`2.7.1`。
   请从[软件包下载地址](https://gitcode.com/Ascend/pytorch/releases/v7.2.0.1-pytorch2.7.1)下载`release v7.2.0.1-pytorch2.7.1`源码，参考[源码编译安装](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0005.html)。

3. 下载项目源码并安装依赖。

    ```bash
    # 下载项目源码，以master分支为例
    git clone https://gitcode.com/cann/cann-recipes-infer.git

    # 本仓库依赖于[HunyuanImage-3.0](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0)的开源仓库代码。进入HunyuanImage-3.0的仓库，下载开源仓库代码：
    git clone https://github.com/Tencent-Hunyuan/HunyuanImage-3.0.git
    cd HunyuanImage-3.0
    git checkout 62da220178f4b0b7d83e91665a46a20a3ee4f7cd
    cd ..

    # 将HunyuanImage-3.0仓库的代码以**非覆盖模式**复制到本项目目录下：
    cp -rn HunyuanImage-3.0/* cann-recipes-infer/models/hunyuan-image-3.0/

    # 安装依赖的python库
    cd cann-recipes-infer
    pip3 install -r ./models/hunyuan-image-3.0/requirements.txt
    ```

4. 配置样例运行所需环境信息。

   修改`executor/scripts/set_env.sh`中的如下字段:
   - `IPs`：配置所有节点的IP，按照rank id排序，多个节点的ip通过空格分开，例如：`('xxx.xxx.xxx.xxx' 'xxx.xxx.xxx.xxx')`。
   - `recipes_path`: 当前代码仓根目录，例如`/home/cann-recipes-infer`。
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

2、执行推理脚本

   除了上述的环境变量外，脚本中其余入参的设置详见run_image_gen.py文件中`parse_args`函数中的参数说明。

   ```shell
   cd models/hunyuan-image-3.0

   # 开启CFG并行，开启VAE并行示例
   bash demo.sh
   ```

   本样例测试结果如下：

   |            Model             |   Environment  |    TP    | CFG Parallel | VAE Parallel |     E2E      |
   | :--------------------------: | :------------: | :------: | :----------: | :----------: | :----------: |
   |       hunyuan-image-3.0      |       A3       |    8     |    enable    |    enable    |    11.13s    |

## 优化点参考

本样例采用的详细优化点介绍可参见[NPU HunyuanImage-3.0模型推理优化实践](../../docs/models/hunyuan-image-3.0/hunyuan_image_3_optimization.md)。