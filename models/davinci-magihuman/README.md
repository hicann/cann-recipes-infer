# 在昇腾Atlas A2/A3环境上适配daVinci-MagiHuman模型的推理
daVinci-MagiHuman 模型是一款多模态视频生成模型，提供了文生视频、图文生视频功能。本项目旨在提供 daVinci-MagiHuman 模型的 Atlas A2/A3 适配版本，为开发者开展相关 NPU 迁移工作提供参考。

本项目基于NPU主要完成以下优化点，具体内容可至[NPU daVinci-MagiHuman模型推理优化实践](https://gitcode.com/cann/cann-recipes-infer/blob/master/docs/models/davinci-magihuman/davinci-magihuman_optimization.md)查看：

- 支持NPU npu_fused_infer_attention_score融合算子；
- 支持NPU npu_rotary_mul融合算子；
- 支持NPU npu_rms_norm融合算子；
- 支持NPU npu_clipped_swiglu融合算子；
- 支持NPU npu_fast_gelu融合算子；
- 支持NPU npu_add_rms_norm融合算子；
- Offload优化。


## 执行样例
本样例支持支持Atlas A2/A3环境的单卡推理。

### 创建conda环境

```shell
conda create -n davinci-magihuman python=3.12
conda activate davinci-magihuman
conda install ffmpeg
```

**在后面的安装过程中都使用`davinci-magihuman`这个环境**。

### CANN环境准备

1. 安装CANN软件包

   本样例的编译执行依赖CANN开发套件包与CANN二进制算子包，支持的CANN软件版本为`CANN 9.0.0`。
  
   请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0)下载`Ascend-cann-toolkit_${version}_linux-${arch}.run`和`Ascend-cann-${soc}-ops_${version}_linux-${arch}.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html?OS=Debian&InstallType=netconda)进行安装。
   
   - `${soc}`表示芯片版本，如910b、A3。
   - `${version}`表示CANN包版本号，如9.0.0。
   - `${arch}`表示CPU架构，如aarch64、x86_64。

2. 安装Ascend Extension for PyTorch（torch_npu）。

   Ascend Extension for PyTorch（torch_npu）为支撑PyTorch框架运行在NPU上的适配插件，本样例支持的Ascend Extension for PyTorch版本为`v26.0.0`，PyTorch版本为`2.10.0`。

   请参考[torch_npu安装文档](https://www.hiascend.com/document/detail/zh/Pytorch/latest/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md)进行安装，需要安装PyTorch框架、torch_npu插件。

    - `${arch}`表示CPU架构，如aarch64、x86_64。



### 依赖安装

本仓库依赖于daVinci-MagiHuman的开源仓库代码。

下载daVinci-MagiHuman开源仓库代码：

```shell
git clone https://github.com/GAIR-NLP/daVinci-MagiHuman.git
cd daVinci-MagiHuman
git checkout 209209b7086eba2020c5439265221495a8357322
cd ..
```


下载本仓库代码：

```shell
git clone https://gitcode.com/cann/cann-recipes-infer.git
```

此刻，当前工作目录下有`daVinci-MagiHuman`、`cann-recipes-infer`这两个目录。`daVinci-MagiHuman/inference`包含了完整的开源代码Python脚本文件，`cann-recipes-infer/models/davinci-magihuman/inference`仅仅包含NPU适配与优化涉及到的Python脚本文件。接下来需要将`daVinci-MagiHuman/inference`合并到`cann-recipes-infer/models/davinci-magihuman/inference`，使得`cann-recipes-infer/models/davinci-magihuman/inference`包含完整的Python脚本文件。将daVinci-MagiHuman仓库的inference目录代码以**非覆盖模式**复制到本项目目录下：

```shell
cp -rn daVinci-MagiHuman/inference cann-recipes-infer/models/davinci-magihuman/
```

此刻，`cann-recipes-infer/models/davinci-magihuman/inference`包含了完整的Python脚本文件，同时也包含了NPU适配与优化的代码。

将daVinci-MagiHuman仓库的assets目录代码复制到本项目目录下：

```shell
cp -r daVinci-MagiHuman/assets cann-recipes-infer/models/davinci-magihuman/
```

将daVinci-MagiHuman仓库的prompts目录代码复制到本项目目录下：

```shell
cp -r daVinci-MagiHuman/prompts cann-recipes-infer/models/davinci-magihuman/
```

将daVinci-MagiHuman仓库的example/assets目录代码复制到本项目目录下：

```shell
mkdir -p cann-recipes-infer/models/davinci-magihuman/example
cp -r daVinci-MagiHuman/example/assets cann-recipes-infer/models/davinci-magihuman/example
```


安装依赖：

```shell
# 安装torchvision，torchaudio
pip install torchvision==0.25.0 torchaudio==2.10.0

# 安装MagiCompiler
git clone https://github.com/SandAI-org/MagiCompiler.git
cd MagiCompiler
pip install -r requirements.txt
pip install .
cd ..

# 安装daVinci-MagiHuman依赖
cd cann-recipes-infer/models/davinci-magihuman
pip install -r requirements.txt
pip install --no-deps -r requirements-nodeps.txt
```


### 准备模型权重

  
| 模型 |下载链接  |
|--|--|
| daVinci-MagiHuman | [daVinci-MagiHuman](https://www.modelscope.cn/models/GAIR/daVinci-MagiHuman/files) |
| Text Encoder | [t5gemma-9b-9b-ul2](https://www.modelscope.cn/models/google/t5gemma-9b-9b-ul2/files) |
| Audio Model | [stable-audio-open-1.0](https://www.modelscope.cn/models/stabilityai/stable-audio-open-1.0/files) |
| VAE | [Wan2.2-TI2V-5B](https://www.modelscope.cn/models/Wan-AI/Wan2.2-TI2V-5B/files) |
  

下载上述4个模型权重到本地路径`models`，下载完成后，本地的模型权重目录结构如下图所示：

```
models/
├── daVinci-MagiHuman/
│   ├── 1080p_sr/
|   |   ├── model-00001-of-00013.safetensors
|   |   ├── ...
|   |   ├── model-00013-of-00013.safetensors
|   |   └── model.safetensors.index.json
│   ├── 540p_sr/
|   |   ├── model-00001-of-00013.safetensors
|   |   ├── ...
|   |   ├── model-00013-of-00013.safetensors
|   |   └── model.safetensors.index.json
│   ├── architecture.png
│   ├── base/
|   |   ├── model-00001-of-00007.safetensors
|   |   ├── ...
|   |   ├── model-00007-of-00007.safetensors
|   |   └── model.safetensors.index.json
│   ├── config.json
│   ├── configuration.json
│   ├── distill/
|   |   ├── model-00001-of-00013.safetensors
|   |   ├── ...
|   |   ├── model-00013-of-00013.safetensors
|   |   └── model.safetensors.index.json
│   ├── README.md
│   └── turbo_vae/
|       ├── checkpoint-340000.ckpt
|       └── TurboV3-Wan22-TinyShallow_7_7.json
│
├── stable-audio-open-1.0/
│   ├── configuration.json
│   ├── fma_dataset_attribution2.csv
│   ├── freesound_dataset_attribution2.csv
│   ├── LICENSE.md
│   ├── model.ckpt
│   ├── model_config.json
│   ├── model_index.json
│   ├── model.safetensors
│   ├── projection_model/
|   |   ├── config.json
|   |   └── diffusion_pytorch_model.safetensors
│   ├── README.md
│   ├── scheduler/
|   |   └── scheduler_config.json
│   ├── stable_audio_light.png
│   ├── text_encoder/
|   |   ├── config.json
|   |   └── model.safetensors
│   ├── tokenizer/
|   |   ├── special_tokens_map.json
|   |   ├── spiece.model
|   |   ├── tokenizer_config.json
|   |   └── tokenizer.json
│   ├── transformer/
|   |   ├── config.json
|   |   └── diffusion_pytorch_model.safetensors
│   ├── vae/
|   |   ├── config.json
|   |   └── diffusion_pytorch_model.safetensors
│   ├── vae_model.ckpt
│   └── vae_model_config.json
│
├── t5gemma-9b-9b-ul2/
│   ├── config.json
│   ├── configuration.json
│   ├── generation_config.json
│   ├── model-00001-of-00009.safetensors
│   ├── ...
│   ├── model-00009-of-00009.safetensors
│   ├── model.safetensors.index.json
│   ├── README.md
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── tokenizer.model
│
└── Wan2.2-TI2V-5B
    ├── assets/
    |   ├── comp_effic.png
    |   ├── logo.png
    |   ├── moe_2.png
    |   ├── moe_arch.png
    |   ├── performance.png
    |   └── vae.png
    ├── config.json
    ├── configuration.json
    ├── diffusion_pytorch_model-00001-of-00003.safetensors
    ├── diffusion_pytorch_model-00002-of-00003.safetensors
    ├── diffusion_pytorch_model-00003-of-00003.safetensors
    ├── diffusion_pytorch_model.safetensors.index.json
    ├── google/
    |   └── umt5-xxl/
    |       ├── special_tokens_map.json
    |       ├── spiece.model
    |       ├── tokenizer_config.json
    |       └── tokenizer.json
    ├── models_t5_umt5-xxl-enc-bf16.pth
    ├── README.md
    └── Wan2.2_VAE.pth
```

### 设置模型权重路径

设置`models/daVinci-MagiHuman/config`目录下`base_config.json`、`distill_config.json`、`sr_540p_config.json`、`sr_1080p_config.json`配置文件里面的模型权重路径，将其中包含`models`的路径替换为本地`models`路径。

### 快速启动

本样例通过 `bash infer.sh ${mode}` 拉起，推理参数集中在 `config/*.yaml`、`config/.json` 维护。

- `${mode}`表示模式，如base、base_ti2v、distill、sr_540p等，共8种模式，详见下表关于模式的说明。


|模式|说明|配置文件|
|----|---|-------|
|base | 使用base模型，文生视频，输出256p|base.yaml, base_config.json|
|base_ti2v | 使用base模型，图文生视频，输出256p|base_ti2v.yaml, base_config.json|
|distill | 使用distill模型，文生视频，输出256p|distill.yaml, distill_config.json|
|distill_ti2v | 使用distill模型，图文生视频，输出256p|distill_ti2v.yaml, distill_config.json|
|sr_540p | 使用distill模型，文生视频，超分辨率到540p|sr_540p.yaml, sr_540p_config.json|
|sr_540p_ti2v | 使用distill模型，图文生视频，超分辨率到540p|sr_540p_ti2v.yaml, sr_540p_config.json|
|sr_1080p | 使用distill模型，文生视频，超分辨率到1080p|sr_1080p.yaml, sr_1080p_config.json|
|sr_1080p_ti2v | 使用distill模型，图文生视频，超分辨率到1080p|sr_1080p_ti2v.yaml, sr_1080p_config.json|

#### 1. 配置 CANN 环境变量

执行推理前先完成 CANN 环境变量配置：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

设置使用哪张卡，下面示例设置为1卡，也可以设置为其他可用的卡：

```shell
export ASCEND_RT_VISIBLE_DEVICES=1
```

> 注意：`PYTHONPATH` 不会也不应全局设置项目根路径，`mm_function.sh` 会在 `torchrun` 子进程中按需注入，避免影响 CANN TBE 编译器的内置 Python 解释器。

#### 2. 更换模型权重路径

根据前述模式说明表格，找到模式对应的`json`配置文件，修改其中包含`models`的模型权重路径。
比如`base`模式，对应的`json`配置文件为`config/base_config.json`。

#### 3. 修改提示词与输入图片

提示词在`example/assets/prompt.txt`文件里面，可以修改该文件或者替换该文件，或者直接修改`infer.sh`里面`PROMPT_FILE`指向的文件路径。
输入图片为`example/assets/image.png`，可以替换该文件或者修改模式对应的`yaml`配置文件里面`image_path`指向的文件路径。
比如`base_ti2v`模式，对应的`yaml`配置文件为`config/base_ti2v.yaml`，修改该文件里面`image_path`指向的文件路径就可以更换输入图片。

#### 4. 开启或者关闭profiling

默认关闭profiling。

开启profiling：

```shell
export ENABLE_PROFILER=1
```

开启profiling后关闭profiling（下面2种方法任选1种）:

```shell
# 方法1
export ENABLE_PROFILER=0
# 方法2
unset ENABLE_PROFILER
```

Profiling输出的目录为`prof/${mode}_prof`。

- `${mode}`表示模式，如base、base_ti2v、distill、sr_540p等，共8种模式，详见前述关于模式的表格。

#### 5. 拉起推理

拉起推理的脚本如下所示，可以根据需要选择其中一种模式。

```shell
# 使用base模型，文生视频，输出256p
bash infer.sh base
# 使用base模型，图文生视频，输出256p
bash infer.sh base_ti2v
# 使用distill模型，文生视频，输出256p
bash infer.sh distill
# 使用distill模型，图文生视频，输出256p
bash infer.sh distill_ti2v
# 使用distill模型，文生视频，超分辨率到540p
bash infer.sh sr_540p
# 使用distill模型，图文生视频，超分辨率到540p
bash infer.sh sr_540p_ti2v
# 使用distill模型，文生视频，超分辨率到1080p
bash infer.sh sr_1080p
# 使用distill模型，图文生视频，超分辨率到1080p
bash infer.sh sr_1080p_ti2v
```


## 性能数据

下面性能数据的耗时单位都为秒。

本样例在昇腾Atlas A3单卡性能数据如下表所示：

| 模式 | Base | SR | Decode | Total |
|--|:--:|:--:|:--:|:--:|
| base | 35.67 | - | 0.98 | 36.65 |
| base_ti2v | 35.84 | - | 0.92 | 36.76 |
| distill | 5.21 | - | 1.01 | 6.22 |
| distill_ti2v | 5.24 | - | 0.94 | 6.18 |
| sr_540p | 4.92 | 24.31 | 3.05 | 32.29 |
| sr_540p_ti2v | 5.06 | 24.49 | 3.01 | 32.57 |
| sr_1080p | 4.91 | 123.69 | 14.49 | 143.09 |
| sr_1080p_ti2v | 5.00 | 123.35 | 14.47 | 142.82 |

开源代码仓在H100 GPU单卡性能数据如下表所示（来源于[论文](https://arxiv.org/abs/2603.21986)中的Table 2）：

| 模式 | Base | SR | Decode | Total |
|--|:--:|:--:|:--:|:--:|
| distill | 1.6 | - | 0.4 | 2.0 |
| sr_540p | 1.6 | 5.1 | 1.3 | 8.0 |
| sr_1080p | 1.6 | 31.0 | 5.8 | 38.4 |


## 附录

### 公共环境变量说明

以下环境变量由 `executor/scripts/mm_function.sh` 在启动时统一设置（作为默认值，可通过 YAML 的 `env_vars` 覆盖）：

- `PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'`：PyTorch 针对昇腾 NPU 的内存分配配置，启用"可扩展内存段"减少 OOM 风险，详见[官方文档](https://www.hiascend.com/document/detail/zh/Pytorch/710/comref/Envvariables/Envir_012.html)；
- `TASK_QUEUE_ENABLE=2`：开启 task_queue 算子下发队列 Level 2 优化，将 workspace 相关任务迁移至二级流水，详见[官方文档](https://www.hiascend.com/document/detail/zh/Pytorch/710/comref/Envvariables/Envir_007.html)；
- `CPU_AFFINITY_CONF=1`：开启粗粒度绑核，将任务绑定在 NPU 业务绑核区间的 CPU 核上，避免不同卡任务之间的线程抢占，详见[官方文档](https://www.hiascend.com/document/detail/zh/Pytorch/710/comref/Envvariables/Envir_033.html)；
- `TOKENIZERS_PARALLELISM=false`：禁用 tokenizers 库内部的并行化处理。

同时会自动设置 HCCL 通信相关的 `HCCL_IF_IP / HCCL_IF_BASE_PORT / HCCL_CONNECT_TIMEOUT / HCCL_EXEC_TIMEOUT`。

### 配置文件说明

每种模式都包含`json`配置文件、`yaml`配置文件，比如`base`模式包含`base_config,json`、`base.yaml`这两个配置文件。下面分别介绍`json`配置文件、`yaml`配置文件。

#### json配置文件说明

`json`配置文件样例如下：

```json
{
  "engine_config": {
    "load": "/path/to/models/daVinci-MagiHuman/distill",
    "distill": true,
    "cp_size": 1
  },
  "evaluation_config": {
    "cfg_number": 1,
    "num_inference_steps": 8,
    "audio_model_path": "/path/to/models/stable-audio-open-1.0",
    "txt_model_path": "/path/to/models/t5gemma-9b-9b-ul2",
    "vae_model_path": "/path/to/models/Wan2.2-TI2V-5B",
    "use_sr_model": true,
    "sr_model_path": "/path/to/models/daVinci-MagiHuman/1080p_sr",
    "sr_num_inference_steps": 5,
    "sr_cfg_number": 1,
    "use_turbo_vae": true,
    "student_config_path": "/path/to/models/daVinci-MagiHuman/turbo_vae/TurboV3-Wan22-TinyShallow_7_7.json",
    "student_ckpt_path": "/path/to/models/daVinci-MagiHuman/turbo_vae/checkpoint-340000.ckpt"
  }
}
```

`engine_config`配置项说明如下：

|参数名|类型|默认值|含义|
|-----|----|-----|----|
| load | str | - | 基础DiT模型路径 |
| distill | bool | false|基础DiT模型是不是`distill`模型，不是的话采用`base`模型 |
| cp_size | int | 1 | CP并行大小，本样例中固定为1 |

`evaluation_config`配置项说明如下：

|参数名|类型|默认值|含义|
|-----|----|-----|----|
| cfg_number | int | 2 | 基础DiT模型的CFG（Classifier-Free Guidance）数量，可以取1或者2 |
| num_inference_steps | int | 32 | 推理时去噪的步骤数量 |
| audio_model_path | str | - | audio模型的路径 |
| txt_model_path | str | - | text模型的路径 |
| vae_model_path | str | - | VAE模型的路径 |
| use_sr_model | bool | false | 是否使用SR（super-resolution）模型，540p/1080p需要使用SR模型 |
| sr_model_path | str | - | SR（super-resolution）模型的路径 |
| sr_num_inference_steps | int | 5 | 在SR（super-resolution）推理时去噪的步骤数量 |
| sr_cfg_number | int | 2 | SR（super-resolution）的DiT模型的CFG（Classifier-Free Guidance）数量，可以取1或者2 |
| use_turbo_vae | bool | true | 解码视频的时候是否使用TurboVAED，不是的话使用Wan2_2_VAE |
| student_config_path | str | - | 学生模型的配置文件路径 |
| student_ckpt_path | str | - | 学生模型的checkpoint文件路径 |


#### yaml配置文件说明

`yaml`配置文件样例如下：

```yaml
model_args:
  config-load-path: "config/sr_1080p_config.json"
  seconds: 5
  br_width: 448
  br_height: 256
  sr_width: 1920
  sr_height: 1088
  output_path: "output_sr_1080p"

model_name: "daVinci-MagiHuman"
world_size: 1
master_port: 29616
entry_script: "sample_video.py"

env_vars:
  TORCH_COMPILE_DISABLE: "1"
  MAGI_COMPILE_COMPILE_MODE: "NONE"
  CPU_OFFLOAD: "true"

dit_cache:
  method: "NoCache"
```

`model_args`配置项说明如下：

|参数名|类型|默认值|含义|
|-----|----|-----|----|
| config-load-path | str | - | `json`配置文件路径 |
| seconds | int | 4 | 输出视频的秒数，视频的默认帧率为25fps |
| br_width | int | 480 | 基础去噪输出的视频帧宽度 |
| br_height | int | 272 | 基础去噪输出的视频帧高度 |
| sr_width | int | - | SR（super-resolution）去噪输出的视频帧宽度 |
| sr_height | int | - | SR（super-resolution）去噪输出的视频帧高度 |
| output_path | str | - | 输出视频文件路径 |

一层配置项说明如下：

|参数名|类型|默认值|含义|
|-----|----|-----|----|
| model_name | str | - | 模型名称 |
| world_size | int | 1 | 并行计算的总卡数，本样例固定为1 |
| master_port | int | - | 进程端口号 |
| entry_script | str | - | 入口的脚本文件 |

`env_vars`配置项说明如下：

|参数名|类型|默认值|含义|
|-----|----|-----|----|
| TORCH_COMPILE_DISABLE | str | "0" | PyTorch框架的环境变量 |
| MAGI_COMPILE_COMPILE_MODE | str | - | MagiCompiler框架的编译模式开关 |
| CPU_OFFLOAD | str | - | 基础DiT模型、SR DiT模型是否需要offload到CPU, 540p/1080p需要设置为"true" |

`dit_cache`配置项说明如下：

|参数名|类型|默认值|含义|
|-----|----|-----|----|
| method | str | - | DiT cache采用的方法，本样例不开启cache |
