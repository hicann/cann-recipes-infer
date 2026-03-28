# 在昇腾Atlas A2环境上适配HunyuanVideo模型的推理

HunyuanVideo模型是一款多模态视频生成模型，提供了文生视频功能。本项目旨在提供HunyuanVideo的昇腾适配版本。

本项目基于NPU主要完成了以下优化点，具体内容可至[NPU HunyuanVideo模型推理优化实践](../../docs/models/hunyuan-video/hunyuan_video_optimization.md)查看：

- 支持NPU npu_fused_infer_attention_score融合算子，npu_rms_norm融合算子，npu_rotary_mul融合算子；
- 支持ulysses序列并行；
- 支持ring attention序列并行和通算掩盖；
- 集成step-level Dit-Cache加速方案，支持[FBCache](https://github.com/chengzeyi/ParaAttention)和[TeaCache](https://github.com/ali-vilab/TeaCache)；
- 集成[TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer) Dit-Cache加速方案，支持自定义DiT Cache方案，具备warmup、cutoff、offload功能；
- 支持VAE并行；
- 支持[UAA: Ulysses Anything Attention](https://cache-dit.readthedocs.io/en/latest/user_guide/CONTEXT_PARALLEL/#uaa-ulysses-anything-attention)；
- 支持950PR平台的量化加速特性，涵盖FP8 FA激活量化及MXFP8 A8W8全量化方案。

## 执行样例

本样例支持Atlas A2环境的单卡、多卡推理。

### CANN环境准备

1. 本样例的编译执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），支持的CANN软件版本为`CANN 8.5.0.alpha002`。

请从[CANN软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0)下载`Ascend-cann-toolkit_${version}_linux-${arch}.run`与`Ascend-cann-kernels-${chip_type}_${version}_linux-${arch}.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_quick.html?Mode=PmIns&OS=openEuler&Software=cannToolKit)进行安装。

2. 本样例依赖的torch及torch_npu版本为2.7.1。

请从[Ascend Extension for PyTorch插件](https://gitcode.com/Ascend/pytorch/tree/v2.7.1-7.3.0)下载`v2.7.1-7.3.0`源码，参考[源码编译安装](https://www.hiascend.com/document/detail/zh/Pytorch/720/configandinstg/instg/insg_0005.html)。

### 依赖安装

本仓库依赖于[HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo)的开源仓库代码。

首先进入HunyuanVideo的仓库，下载开源仓库代码：

```shell
git clone https://github.com/Tencent-Hunyuan/HunyuanVideo.git
```

下载本仓库代码：

```shell
git clone https://gitcode.com/cann/cann-recipes-infer.git
```

将HunyuanVideo仓库的代码以**非覆盖模式**复制到本项目目录下：

```shell
cp -rn HunyuanVideo/* cann-recipes-infer/models/hunyuan-video
```

```shell
# 安装Python依赖
pip install -r requirements.txt
```


### 准备模型权重

| 模型      | 版本                                           |
|---------|----------------------------------------------|
| HunyuanVideo | [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo#download-pretrained-models) |

下载HunyuanVideo模型权重到本地路径`ckpts`。

```
HunyuanVideo/
├── hyvideo/
|   └──...
├── scripts/
│   └──...
├── ckpts/
|   └──...
└──...
```

### 快速启动

本样例在scripts文件夹中准备了单卡和多卡的推理脚本。

首先参考[依赖按照](#12-依赖安装)准备环境和代码。

执行测试脚本前，请参考[Ascend社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/softwareinst/instg/instg_quick.html?Mode=PmIns&OS=Debian&Software=cannToolKit)中的CANN安装软件教程，配置环境变量：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh 
```

启用torch_npu环境, 添加PYTHONPATH：

```shell
source scripts/set_env.sh
```

#### 单卡推理: 

原生hunyuanVideo模型在单块Atlas 800I A2上，支持生成视频规格`720*1280*129`。执行以下脚本启用单卡推理，环境变量的详细信息请参考[CANN社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/envvar/envref_07_0001.html)。

```shell
bash scripts/test.sh
```

**Dit-Cache**：本样例集成了多种Dit-Cache方案，包括[FBCache](https://github.com/chengzeyi/ParaAttention)、[TeaCache](https://cvpr.thecvf.com/virtual/2025/poster/33872)，[TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer)加速方案，**当前仅支持单机单卡**。

通过读取配置文件`hyvideo/cache/cache_config.json`初始化Dit-Cache，用户可自定义配置文件，通过传入参数`--cache-config`来自定义配置文件地址。

用户可修改配置文件中的以下几项config文件里的字段，其他配置信息详见[HunyuanVideo优化文档](../../docs/models/hunyuan-video/hunyuan_video_optimization.md)：

`cache_forward`: 选择Dit-Cache方案，设为`FBCache`启用FBCache，设为`TeaCache`启用TeaCache，设为`TaylorSeer`启用TaylorSeer，其他情况不启用Dit-Cache。默认不启用Dit-Cache。

`rel_l1_thresh`: 控制加速比，当Dit-Cache为FBCache时，`rel_l1_thresh=0.1`时DiT模型加速比为2.0；当Dit-Cache为TeaCache时，`rel_l1_thresh=0.1`时DiT模型加速比为1.6，`rel_l1_thresh=0.15`时DiT模型加速比为2.1。请注意，更大的阈值可以获得更高的加速比，但也会带来更高的精度损失。

`offload`: 开关offload功能，减少TaylorSeer对内存的占用，以支持生成序列长度更长的视频。大样例建议使能该特性规避TaylorSeer额外缓存导致的oom问题，默认`offload=True`，开启offload。**`720*1280*129`规格下，开启TaylorSeer offload功能需要保证有400GB以上的内存。**

**注意**：开启DiT-Cache后，会轻微加剧内存占用。如果内存占用过高（e.g. 剩余可用内存不足100MB），开启虚拟内存可能会产生严重的性能下降。此时请删除环境变量`export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True`，并且开启`--cpu-offload`。

**量化**：传入参数`--fa-perblock-fp8` 启用FP8 FA量化，当前量化策略为per-block，q的block粒度为128，kv的block粒度为256；传入参数`--mm-mxfp8`启用MXFP8 A8W8量化，当前策略为per-channel直转。量化功能**支持多卡推理**，**仅支持950PR**。

**性能分析**：本样例支持Ascend PyTorch Profiler接口采集并分析模型性能，在脚本中传入参数`--prof-dit`，启用性能分析，分析文件默认保存在`.prof`路径。具体使用方法请参考CANN社区文档[性能分析](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/devaids/devtools/profiling/atlasprofiling_16_0006.html)。**支持多卡推理**。

#### 多卡推理: 

本样例适配了Ulysses/Ring Attention两种序列并行方法，用于多卡并行推理，减少显存占用，提高推理速率，通过传入参数`--ulysses-degree=<SP number>`或者`--ring-degree=<SP number>`启用序列并行。原生hunyuanVideo模型在8块Atlas 800I A2上，支持生成视频规格`720*1280*649`。多卡推理要求满足以下约束：
1. 混合并行策略约束 `nproc_per_node == ulysses-degree * ring-degree`；
2. 视频规格约束条件`H % 16 % <SP number> == 0 or W % 16 % <SP number> == 0`，其中`H, W, T` 分别是视频帧的高、宽、数量；
3. 序列并行度的约束条件`<head num> % <SP number> == 0`，其中`<head num> = 24`。

执行以下脚本启用多卡序列并行，环境变量的详细信息请参考[CANN社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/envvar/envref_07_0001.html)。

```shell
bash scripts/test_sp.sh
```

**VAE并行**: 在多卡推理时，本样例支持VAE并行，传入参数`--use-vae-parallel`开启VAE并行功能。

**UAA, Ulysses Anythin Attention**: 传入参数`--ulysses-anything`启用UAA。开启后解除对视频规格`H, W`的约束，支持任意尺寸的`H, W`；解除序列并行度的约束条件`<head num> % <SP number> == 0`，支持`<head num> 大于 <SP number>`，其中`<head num> = 24`。**UAA仅支持纯Ulysses，不支持Ulysses+RingAttention混合序列并行策略**。