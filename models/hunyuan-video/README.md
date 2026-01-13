# 在昇腾Atlas A2环境上适配HunyuanVideo模型的推理

HunyuanVideo模型是一款多模态视频生成模型，提供了文生视频功能。本项目旨在提供HunyuanVideo的昇腾适配版本。

本项目基于NPU主要完成了以下优化点，具体内容可至[NPU HunyuanVideo模型推理优化实践](../../docs/models/hunyuan-video/hunyuan_video_optimization.md)查看：

- 支持NPU npu_fused_infer_attention_score融合算子，npu_rms_norm融合算子，npu_rotary_mul融合算子；
- 支持ulysses序列并行；
- 支持ring attention序列并行和通算掩盖；
- 集成step-level Dit-Cache加速方案，支持[FBCache](https://github.com/chengzeyi/ParaAttention)和[TeaCache](https://github.com/ali-vilab/TeaCache)。

## 执行样例

本样例支持Atlas A2环境的单卡、多卡推理。

### CANN环境准备

1. 本样例的编译执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），支持的CANN软件版本为`CANN 8.5.0.alpha002`。

请从[CANN软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0.alpha002)下载`Ascend-cann-toolkit_${version}_linux-${arch}.run`与`Ascend-cann-kernels-${chip_type}_${version}_linux-${arch}.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Debian&Software=cannToolKit)进行安装。

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

### 对Python依赖的补充修改

[xfuser仓](https://github.com/xdit-project/xDiT)和[yunchang仓](https://github.com/feifeibear/long-context-attention)已经对npu进行了适配，访问github仓库将xfuser目录和yunchang目录放到`HunyuanVideo/`目录下。当前支持yunchang`7a52abd669efb35e550680a239e1745b620b2bae`commit之后的版本，xfuser`e559fe8f07c7cfdd02a73ed03c00b8b128de682a`commit之后的版本。如果版本不匹配，请参考[附录](#附录)修改。

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

**单卡推理**: 通过设置环境变量`export ASCEND_RT_VISIBLE_DEVICES=0`指定启用第0卡推理，更多环境变量相关问题，请参考[CANN社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/maintenref/envvar/envref_07_0028.html)。原生hunyuanVideo模型在单块Atlas 800I A2上，支持生成视频规格`780*1280*129`。
```shell
bash scripts/test.sh
```

**多卡推理**: 本样例适配了Ulysses/Ring Attention两种序列并行方法，用于多卡并行推理，减少显存占用，提高推理速率，通过传入参数`--ulysses-degree=<SP number>`或者`--ring-degree=<SP number>`启用序列并行。请满足序列并行约束条件`nproc_per_node == ulysses-degree * ring-degree`，以及视频规格约束条件`H % 16 % <SP number> == 0 or W % 16 % <SP number> == 0`，其中`H, W, T` 分别是视频帧的高、宽、数量。原生hunyuanVideo模型在8块Atlas 800I A2上，支持生成视频规格`780*1280*649`。

执行以下脚本启用多卡序列并行，环境变量的详细信息请参考[CANN社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/maintenref/envvar/envref_07_0001.html)。

```shell
bash scripts/test_sp.sh
```

**Dit-Cache**：本样例集成了step-level Dit-Cache方案，集成了[FBCache](https://github.com/chengzeyi/ParaAttention)和[TeaCache](https://cvpr.thecvf.com/virtual/2025/poster/33872)和FBCache加速方案，当前仅支持单机单卡。

通过读取配置文件`hyvideo/cache/cache_config.json`初始化Dit-Cache，用户可自定义配置文件，通过传入参数`--cache-config`来自定义配置文件地址。

用户可修改配置文件中的以下几项自定义cache方案，其他配置信息详见[HunyuanVideo优化文档](../../docs/models/hunyuan-video/hunyuan_video_optimization.md)：

`cache_forward`: 选择Dit-Cache方案，设为`FBCache`启用FBCache，设为`TeaCache`启用TeaCache，其他情况不启用Dit-Cache。默认不启用Dit-Cache。

`rel_l1_thresh`: 控制加速比，当Dit-Cache为FBCache时，`rel_l1_thresh=0.1`时DiT模型加速比为2.0；当Dit-Cache为TeaCache时，`rel_l1_thresh=0.1`时DiT模型加速比为1.6，`rel_l1_thresh=0.15`时DiT模型加速比为2.1。请注意，更大的阈值可以获得更高的加速比，但也会带来更高的精度损失。


**性能分析**：本样例支持Ascend PyTorch Profiler接口采集并分析模型性能，具体使用方法请参考CANN社区文档[性能分析](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/devaids/devtools/profiling/atlasprofiling_16_0006.html)。

在脚本中传入参数`--prof-dit`，启用性能分析，分析文件默认保存在`.prof`路径。

## 附录

### 侵入式修改外部依赖库

HunyuanVideo使用xdit框架实现序列并行，核心的库是xfuser和yunchang。需要手动消除其内部的cuda版本检查，并适配npu的fa算子。本仓库通过`try: import torch_npu`的方式代替npu检查，走npu方案。

如果使用`xfuser<=0.4.4`和`yunchang<=0.6.3.post1`的版本，需要按照以下方式手动修改代码，具体代码可参考`appendix/xfuser`和`appendix/yunchang`。

在site-packages目录下找到xfuser包，`xfuser/envs.py`的`PackagesEncChecker.initialize()`（L154），改为：

```python
def initialize(self):
    try:
        import torch_npu
        self.packages_info = {
            "has_flash_attn": False,
            "has_long_ctx_attn": self.check_long_ctx_attn(),
            "diffusers_version": self.check_diffusers_version(),
        }
    except ImportError:
        self.packages_info = {
            "has_flash_attn": self.check_flash_attn(),
            "has_long_ctx_attn": self.check_long_ctx_attn(),
            "diffusers_version": self.check_diffusers_version(),
        }
```

在site-packages目录下找到xfuser包，`xfuser/config/config.py`的L11，改为：

```python
try:
    import torch_npu
    from xfuser.envs import TORCH_VERSION, PACKAGES_CHECKER
except ImportError:
    from xfuser.envs import CUDA_VERSION, TORCH_VERSION, PACKAGES_CHECKER
```

在site-packages目录下找到yunchang包，`yunchang/ring/ring_flashinfer_attn.py`的L9，改为：

```python
try:
    import torch_npu
except ImportError:
    torch_cpp_ext._get_cuda_arch_flags()
```