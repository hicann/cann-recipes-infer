# 在昇腾Atlas A2/A3环境上适配SANA-Video模型的推理
SANA-Video模型是一个多模态视频生成模型，提供了文生视频功能。本项目旨在提供 SANA-Video 模型的 Atlas A2/A3 适配版本，为开发者开展相关 NPU 迁移工作提供参考。

本项目基于NPU主要完成以下优化点，具体内容可至[NPU SANA-Video模型推理优化实践](../../docs/models/sana-video/SANA-Video_optimization.md)查看：
- 转换1×1 conv2d为matmul计算提升性能；
- 调换时序conv2d的hw轴提升性能；
- 支持NPU npu_rotary_mul融合算子；
- 支持NPU npu_rms_norm融合算子；
- 支持NPU npu_fusion_attention融合算子；

## 执行样例
本样例支持昇腾Atlas A2/A3环境的单机单卡推理和单机多卡DP推理。

> 使用一站式平台的用户可直接跳转 [「一站式平台的快速启动」](#一站式平台的快速启动)章节。

### CANN环境准备
  1.安装CANN软件包
  
  本样例的编译执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），支持的CANN软件版本为`CANN 8.5.0`。
  
  请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0)下载`Ascend-cann-toolkit_{version}_linux-{arch}.run`和`Ascend-cann-{soc}-ops_{version}_linux-{arch}.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=netconda&OS=Debian)进行安装。
  
  2.安装Ascend Extension for PyTorch（torch_npu）
  
  Ascend Extension for PyTorch（torch_npu）为支撑PyTorch框架运行在NPU上的适配插件，本样例支持的Ascend Extension for PyTorch版本为`torch-npu == 7.1.0.post17`，支持的Torch版本为`torch == 2.6.0`，请从[Ascend Extension for PyTorch插件](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0004.html)下载并安装torch与torch_npu安装包。建议在conda环境中安装：
  ```
conda create -n sana python=3.10
conda activate sana
  ```

### 依赖安装
本仓库依赖 SANA-Video 的开源代码。
首先在项目目录拉取SANA-Video源代码：
```
git clone https://github.com/NVlabs/Sana.git
```

进入到源代码目录：
```
cd Sana
```

切换到指定版本：
```
git checkout 08c656c3
```

回到原目录：
```
cd ..
```

拉取本仓库代码：
```
git clone https://gitcode.com/cann/cann-recipes-infer.git
```

将Sana仓库的代码以**非覆盖模式**复制到本项目目录下：
```
cp -rn Sana/* cann-recipes-infer/models/sana-video/
```

说明：`models/sana-video` 现在只保留 NPU patch 层、入口脚本与文档，绝大部分 `diffusion/` 源码来自上游 Sana。推理入口会在导入上游模块前自动执行 `patches.apply_all()`：`patches/__init__.py` 中包含NPU适配 patch，包括 `patch_triton_rms_norm_import()` 和 `patch_wan_rotary_npu()`；`patches/npu_patches.py` 中包含优化相关 patch，并通过 `apply_npu_optimization_patches()` 统一应用。

安装依赖：

```
cd cann-recipes-infer/models/sana-video
pip install -e .
pip uninstall -y opencv-python
pip install opencv-python-headless==4.8.0.76
```

说明：依赖中的`mmengine` 会引入 `opencv-python`，本样例需要 `opencv-python-headless==4.8.0.76`，因此需切换为 headless 版本。

编译安装mmcv库(注意1.x分支才有Registry模块)：
```
git clone -b 1.x https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py install
cd ..
```
## 模型权重
普通 `infer.sh` 默认使用本地离线权重路径，运行前需准备以下文件或目录，并按实际存放位置修改 `config/2b_480p_single.yaml`：

| 权重类型 | YAML 字段 | 默认相对路径 | 说明 |
|----------|-----------|--------------|------|
| DiT / transformer | `model_args.model_path` | `./SANA-Video_2B_480p/checkpoints/SANA_Video_2B_480p.pth` | SANA-Video 主模型权重 |
| VAE | `model_args.vae.vae_pretrained` | `./SANA-Video_2B_480p/vae/Wan2.1_VAE.pth` | WanVAE 权重 |
| 文本编码器 | `model_args.text_encoder.text_encoder_name` | `./gemma-2-2b-it` | 本地 Gemma HuggingFace 目录，需包含 `config.json`、tokenizer 文件和模型权重 |

权重可从 [SANA-Video_2B_480p](https://huggingface.co/Efficient-Large-Model/SANA-Video_2B_480p) 和 [gemma-2-2b-it](https://huggingface.co/Efficient-Large-Model/gemma-2-2b-it) 下载后放置到上述目录。`infer_platform.sh` 面向一站式平台场景，会使用 `config/2b_480p_single_platform.yaml` 和脚本内的 `WEIGHTS_DIR` 自动下载/定位权重。

## 快速启动

本样例通过 `bash infer.sh` 拉起，推理参数集中在 `config/*.yaml` 维护，启动器使用 `accelerate`。

### 1. 配置 CANN 环境变量

执行推理前先完成 CANN 环境变量配置（参考 [Ascend 社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=Debian)）：

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

> 注意：`PYTHONPATH` 不会也不应全局设置项目根路径，`mm_function.sh` 会在 `accelerate` 子进程中按需注入，避免影响 CANN TBE 编译器的内置 Python 解释器。

### 2. 选择推理配置

`config/` 目录下已预置以下 YAML：

| YAML 文件 | 卡数 | 启动器 | 适用场景 |
|-----------|------|--------|----------|
| `2b_480p_single.yaml` | 1 | accelerate（`mixed_precision: bf16`） | SANA-Video 2B 480p 单卡文生视频，本地离线权重 |
| `2b_480p_single_a8w8.yaml` | 1 | accelerate（`mixed_precision: bf16`） | SANA-Video 2B 480p 做mxfp_a8w8单卡文生视频 |
| `2b_480p_single_a4w4.yaml` | 1 | accelerate（`mixed_precision: bf16`） | SANA-Video 2B 480p 做mxfp_a4w4单卡文生视频(部分回退a8w8) |
| `2b_480p_single_platform.yaml` | 1 | python 直接拉起 | 一站式平台专用模板，由 `infer_platform.sh` 读取并生成临时本地权重配置 |

YAML 中的关键字段：

```yaml
launcher: "accelerate"
launcher_args:
  mixed_precision: "bf16"          # 透传 accelerate launch --mixed_precision=bf16
env_vars:
  DISABLE_XFORMERS: "1"            # 屏蔽 xformers，走 NPU 融合算子
  HF_HUB_OFFLINE: "1"              # 普通 infer.sh 默认离线运行
  TRANSFORMERS_OFFLINE: "1"
  HF_DATASETS_OFFLINE: "1"
model_args:
  config: "configs/sana_video_config/Sana_2000M_480px_AdamW_fsdp.yaml"
  model_path: "./checkpoints/SANA_Video_2B_480p.pth"
  vae.vae_pretrained: "./vae/Wan2.1_VAE.pth"
  text_encoder.text_encoder_name: "./gemma-2-2b-it"
  txt_file: "asset/samples/video_prompts_samples.txt"
  cfg_scale: 6
  motion_score: 30
  flow_shift: 8
  work_dir: "output/sana_t2v_video_results"
  model.fp32_attention: "False"    # 用字符串 "False"，argparse 侧按字面量接收
  quant_type: "bf16"               # 8bit量化:"a8w8" 4bit量化:"a4w4" 
```

### 3. 修改推理输入

根据需要修改 YAML 中 `model_args` 下的字段：

| YAML 字段 | 含义 |
|-----------|------|
| `config` | 推理使用的配置文件（上游 Sana 的 YAML） |
| `model_path` | 推理使用的 DiT / transformer 权重文件路径 |
| `vae.vae_pretrained` | 推理使用的 VAE 权重文件路径 |
| `text_encoder.text_encoder_name` | 推理使用的文本编码器，普通离线配置中填写本地 Gemma HuggingFace 目录 |
| `txt_file` | 推理使用的文本提示词文件 |
| `cfg_scale` | 提示词对齐强度 |
| `motion_score` | 运动强度分数 |
| `flow_shift` | 流偏移参数，用于调整扩散模型去噪时间步 |
| `work_dir` | 生成视频输出路径 |
| `model.fp32_attention` | attn 是否使用 fp32 精度，推理时置 `"False"` 以提升性能 |
| `quant_type` | 量化配置可选 bf16 / a8w8 / a4w4 |

透传规则（适用于所有字段）：

| YAML 类型 | 命令行效果 |
|-----------|-----------|
| 字符串/数字 | `--key value` |
| 布尔 `true` | `--key`（flag） |
| 布尔 `false` | 忽略 |
| 列表 | `--key v1 v2 …` |
| 加引号字符串 `"False"` | `--key False`（用于 pyrallis/Hydra 类解析器） |

### 4. 拉起推理

```
bash infer.sh
```

拉起过程会自动：
- 解析 YAML 中的 `world_size`、`master_port`、`entry_script`、`env_vars`、`launcher_args`、`model_args`；
- 设置通用 NPU 优化环境变量（`PYTORCH_NPU_ALLOC_CONF`、`TASK_QUEUE_ENABLE`、`CPU_AFFINITY_CONF`、`TOKENIZERS_PARALLELISM`）及 HCCL 通信配置；
- 在 `res/<YYYYMMDD>/<model_name>/` 下自动创建日志目录，并将推理输出 `tee` 到 `log_<timestamp>.log`；
- 调用 `accelerate launch --num_processes=<world_size> --num_machines=1 --main_process_port=<master_port> --mixed_precision=bf16 inference_video_scripts/inference_sana_video.py <model_args>`。

如需切换到多卡 DP，可将 YAML 中的 `world_size` 调整为目标卡数，`mm_function.sh` 会自动转换为 `accelerate --num_processes`。

## 一站式平台的快速启动

本章节面向使用一站式平台的用户，平台已预置完整的 CANN 环境，按以下步骤即可在单卡上完成 SANA-Video 文生视频推理。

### 1. 安装 Miniconda 并创建 Python 环境

若当前环境未预装 conda，在用户目录下安装 Miniconda（无需 root 权限）：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/etc/profile.d/conda.sh
echo "source $HOME/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc   # 永久生效
```

说明：x86_64 机器将命令中的 `aarch64` 替换为 `x86_64`。

创建并激活 Python 环境：

```bash
conda create -n sana python=3.10 -y --override-channels -c conda-forge
conda activate sana
```

### 2. 修改 `infer_platform.sh` 中的两处路径

打开 `infer_platform.sh`，按本机实际情况修改「User configuration」段的两个路径：

```bash
WEIGHTS_DIR="/mnt/workspace/gitCode/cann/models"                     # 权重与 HF cache 的存放目录
CANN_SET_ENV="/home/developer/Ascend/ascend-toolkit/set_env.sh"      # 平台 CANN 的 set_env.sh
```

### 3. 一键拉起推理

```bash
cd cann-recipes-infer/models/sana-video
bash infer_platform.sh
```

前置条件：已完成 §1（conda 环境已激活）。脚本本身会依次完成：source CANN 环境脚本 → 安装 torch / torch_npu → 合入上游 Sana 源码 → 安装项目依赖与 NPU mmcv → 通过 hf-mirror 下载权重 → 生成指向本地权重的临时 YAML → 调用 python 启动单卡推理。各步骤均幂等，已完成的环节会自动跳过。

说明：`infer_platform.sh` 不直接读取普通离线配置 `2b_480p_single.yaml`，而是读取 `2b_480p_single_platform.yaml` 作为平台模板，再按 `WEIGHTS_DIR` 生成临时 YAML。普通 `infer.sh` 与一站式平台启动互不影响。

生成结果位于 `output/sana_t2v_video_results/vis/*.mp4`，一条 prompt 对应一个 mp4。

## 性能数据

本样例在Atlas A2/A3的推理性能如下表所示：

| 规格| 单步时延(s) | Diffusion sampling总时长(s) |
|--|--| --|
| 480p81f | 2.75 | 132 |

本样例在 Atlas A5 的推理性能如下表所示：
|量化方式 | 规格| 单步时延(s) | Diffusion sampling总时长(s) | 整网耗时(s) |
|--|--|--|--| -- |
|bf16| 480p81f | 1.67 | 83.5 | 90.44 |
|8bit| 480p81f | 1.46 | 73   | 79.38 |
|4bit| 480p81f | 1.43 | 71.5 | 77.92 |