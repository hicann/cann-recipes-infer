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
- 支持TopK，[SVG](https://github.com/svg-project/Sparse-VideoGen)稀疏attention算法；

## 执行样例

本样例支持Atlas A2环境的单卡、多卡推理。

### CANN环境准备

1. 本样例的编译执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），支持的CANN软件版本为`CANN 8.5.0.alpha002`。

    请从[CANN软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0)下载`Ascend-cann-toolkit_${version}_linux-${arch}.run`与`Ascend-cann-kernels-${chip_type}_${version}_linux-${arch}.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_quick.html?Mode=PmIns&OS=openEuler&Software=cannToolKit)进行安装。

2. 本样例依赖的torch及torch_npu版本为2.7.1，**Python==3.11**。

    ```shell
    #安装Pytorch
    pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
    ```

    [Ascend Extension for PyTorch插件](https://gitcode.com/Ascend/pytorch/tree/v2.7.1-7.3.0)的下载和安装，请参考[二进制软件包安装](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md)。

    ```
    # 下载并安装torch_npu插件
    wget https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.7.1/torch_npu-2.7.1.post2-cp311-cp311-manylinux_2_28_aarch64.whl
    pip3 install torch_npu-2.7.1.post2-cp311-cp311-manylinux_2_28_aarch64.whl
    ```

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
# 安装Python依赖，仅支持python 3.11
cd models/hunyuan-video
pip install -r requirements.txt
```


### 准备模型权重

| 模型      | 版本                                           |
|---------|----------------------------------------------|
| HunyuanVideo | [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo#download-pretrained-models) |
| text_encoder | [llava-llama-3-8b](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers) |
| text_encoder_2 | [CLIP](https://huggingface.co/openai/clip-vit-large-patch14) |

权重下载依赖`huggingface_hub[cli]`，安装方式如下：

```shell
python -m pip install "huggingface_hub[cli]"
```

**准备HunyuanVideo模型权重**，默认本地路径在`ckpts`：

```shell
huggingface-cli download tencent/HunyuanVideo --local-dir ./ckpts
```

国内用户可以考虑使用hf-mirror镜像：
```shell
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download tencent/HunyuanVideo --local-dir ./ckpts
```

**准备text_encoder模型权重**:

```shell
cd models/hunyuan-video/ckpts
huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir ./llava-llama-3-8b-v1_1-transformers
```

为了节省npu内存，将llama3中的llm部分切分到`text_encoder`目录：
```shell
cd models/hunyuan-video/
python hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py --input_dir ckpts/llava-llama-3-8b-v1_1-transformers --output_dir ckpts/text_encoder
```

**准备text_encoder2模型权重**:

```shell
cd models/hunyuan-video/ckpts
huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./text_encoder_2
```


```shell
hunyuan-video/
├── hyvideo/
|   └──...
├── scripts/
│   └──...
├── ckpts/
│  ├──README.md
│  ├──hunyuan-video-t2v-720p
│  │  ├──transformers
│  │  │  ├──mp_rank_00_model_states.pt
│  │  │  ├──mp_rank_00_model_states_fp8.pt
│  │  │  ├──mp_rank_00_model_states_fp8_map.pt
├  │  ├──vae
│  ├──text_encoder
│  ├──text_encoder_2
├──...
└──...
```

#### 自定义模型权重路径

用户可选自定义模型权重路径，按照如下方式修改测试脚本：

```shell
ckpts_path=/path/to/ckpts/
export MODEL_BASE=${ckpts_path}
export DIT_CKPT_PATH=${MODEL_BASE}/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt
python sample_video.py \
    --model-base ${MODEL_BASE} \
    --dit-weight ${DIT_CKPT_PATH} \
    ...
// 其他参数不变，参考models/hunyuan-video/scripts/test.sh
```

### 快速启动

本样例通过 `bash infer.sh` 拉起，推理参数集中在 `config/*.yaml` 维护。

#### 1. 配置 CANN 环境变量

执行推理前先完成 CANN 环境变量配置（参考 [Ascend 社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/softwareinst/instg/instg_quick.html?Mode=PmIns&OS=Debian&Software=cannToolKit)）：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

> 注意：`PYTHONPATH` 不会也不应全局设置项目根路径，`mm_function.sh` 会在 `torchrun` 子进程中按需注入，避免影响 CANN TBE 编译器的内置 Python 解释器。

#### 2. 选择推理配置

`config/` 目录下已预置以下 YAML，分别对应不同的推理场景：

| YAML 文件 | 卡数 | Dit-Cache | 适用场景 |
|-----------|------|-----------|----------|
| `single.yaml` | 1 | NoCache | 单卡基线，无加速 |
| `single_fbcache.yaml` | 1 | FBCache | 单卡 + FBCache 加速 |
| `single_teacache.yaml` | 1 | TeaCache | 单卡 + TeaCache 累积 L1 阈值 + warmup |
| `single_taylorseer.yaml` | 1 | TaylorSeer | 单卡 + TaylorSeer，Taylor 展开 + CPU offload |
| `single_fp8.yaml` | 1 | NoCache | 单卡 + FP8 量化权重（950PR） |
| `sp8.yaml` | 8 | NoCache | 8 卡 Ulysses 序列并行 + VAE 并行，原生规格 `720*1280*129` |
| `sp8_fbcache.yaml` | 8 | FBCache | 8 卡 SP + FBCache 加速 |
| `sp8_teacache.yaml` | 8 | TeaCache | 8 卡 SP + TeaCache 加速 |
| `sp8_taylorseer.yaml` | 8 | TaylorSeer | 8 卡 SP + TaylorSeer，Taylor 展开 + CPU offload |

#### 3. 修改模型权重路径与提示词

打开所选 YAML，按需修改 `model_args` 中的 `prompt`、`video-size`、`video-length`、`seed` 等字段；多卡 YAML 通过 `ulysses-degree / ring-degree` 控制序列并行，通过 `use-vae-parallel` 启用 VAE 并行；FP8 YAML 通过 `dit-weight` 指定量化权重路径。

YAML 中 `model_args` 会按以下规则透传给 `sample_video.py`：

| YAML 类型 | 命令行效果 | 示例 |
|-----------|-----------|------|
| 字符串/数字 | `--key value` | `infer-steps: 50` → `--infer-steps 50` |
| 布尔 `true` | `--key`（flag） | `flow-reverse: true` → `--flow-reverse` |
| 布尔 `false` | 忽略 | `use-cpu-offload: false` → 不添加 |
| 列表 | `--key v1 v2 …` | `video-size: [720, 1280]` → `--video-size 720 1280` |

#### 4. 切换要使用的 YAML

`infer.sh` 中仅需指定一行 `YAML_FILE_NAME`：

```bash
# 默认：8 卡多卡推理
export YAML_FILE_NAME=sp8.yaml

# 单卡 + TeaCache
# export YAML_FILE_NAME=single_teacache.yaml
```

#### 5. 拉起推理

```shell
bash infer.sh
```

拉起过程会自动：
- 解析 YAML 中的 `world_size`、`master_port`、`entry_script`、`env_vars`、`dit_cache`、`model_args`；
- 设置通用 NPU 优化环境变量（`PYTORCH_NPU_ALLOC_CONF`、`TASK_QUEUE_ENABLE`、`CPU_AFFINITY_CONF`、`TOKENIZERS_PARALLELISM`）以及 HCCL 通信配置；
- 在 `res/<YYYYMMDD>/<model_name>/` 下自动创建日志目录，并将推理输出 `tee` 到 `log_<timestamp>.log`；
- 调用 `torchrun --nproc_per_node=<world_size>` 启动 `sample_video.py`，并将 YAML 中的 `dit_cache` 段通过 `--cache-config <yaml>` 透传给 Python 侧的 `CacheManager`。

### Dit-Cache 说明

本样例集成了多种 Dit-Cache 方案（[FBCache](https://github.com/chengzeyi/ParaAttention)、[TeaCache](https://cvpr.thecvf.com/virtual/2025/poster/33872)、[TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer)），通过 YAML 的 `dit_cache` 段配置：

```yaml
dit_cache:
  method: "TeaCache"                # NoCache / FBCache / TeaCache / TaylorSeer
  enable_separate_cfg: true
  params:                           # 方法特有参数（覆盖内置默认值）
    rel_l1_thresh: 0.15
    coefficients: [733.226126, -401.131952, 67.5869174, -3.149879, 0.0961237896]
    warmup: 2
```

常用调参：

- **FBCache `rel_l1_thresh`**：L1 差异阈值。`rel_l1_thresh=0.1` 时 DiT 加速比约 2.0；阈值越大越快，精度损失越大。
- **TeaCache `rel_l1_thresh / coefficients / warmup`**：累积 L1 阈值、多项式重缩放系数、前 N 步强制完整计算。`rel_l1_thresh=0.1` 约 1.6 倍加速，`rel_l1_thresh=0.15` 约 2.1 倍加速。
- **TaylorSeer `n_derivatives / skip_interval_steps / warmup / cutoff_steps / offload`**：Taylor 展开阶数、跳算间隔、前后强制完整计算步数、是否 offload 到 CPU。**`720*1280*129` 规格开启 offload 需保证 400GB 以上主机内存。**

> **内存提示**：开启 Dit-Cache 后内存占用会略增。若剩余可用内存过低（如不足 100MB），可在 YAML 的 `env_vars` 中移除/改写 `PYTORCH_NPU_ALLOC_CONF`，并在 `model_args` 中启用 `use-cpu-offload: true`。

### 量化

通过单卡 FP8 YAML（`single_fp8.yaml`）直接启用；也可向任意 YAML 的 `model_args` 添加：

- `fa-perblock-fp8: true`：启用 FP8 FA 量化（per-block，Q block=128，KV block=256）；
- `mm-mxfp8: true`：启用 MXFP8 A8W8 量化（per-channel 直转）。

量化**支持多卡推理**，**仅支持 950PR**。

### 性能分析

**块稀疏 Attention**：传入参数`--sparse-method` 启用块稀疏 Attention，可选的稀疏策略为`TopK`和`SVG`，在`models/hunyuan-video/hyvideo/sparse/sparse_config.yaml`文件中可以调整参数配置。该优化方法基于[blitz_sparse_attention算子](https://gitcode.com/cann/ops-transformer/blob/master/experimental/attention/blitz_sparse_attention/README.md)实现，运行前需要依据参考文档编译算子。当选择稀疏策略为TopK时，需要运行`offline-profiling`程序，可参考[优化文档](https://gitcode.com/cann/cann-recipes-infer/blob/master/docs/models/hunyuan-video/hunyuan_video_optimization.md)中，`TopK`章节实现过程。

**性能分析**：本样例支持Ascend PyTorch Profiler接口采集并分析模型性能，在脚本中传入参数`--prof-dit`，启用性能分析，分析文件默认保存在`.prof`路径。具体使用方法请参考CANN社区文档[性能分析](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/devaids/devtools/profiling/atlasprofiling_16_0006.html)。**支持多卡推理**。

**多卡推理支持**：块稀疏 Attention 支持与 Ulysses / Ring Attention 序列并行组合使用。在脚本中同时启用 `--sparse-method` 和多卡并行配置即可，例如通过 YAML 设置 `world_size`、`ulysses-degree`、`ring-degree` 等参数，并选择 `TopK` 或 `SVG` 稀疏策略。多卡稀疏推理仍需满足下文的并行度、视频规格和 head 数约束。

### 多卡推理约束与并行配置组合

本样例适配了 Ulysses / Ring Attention 两种序列并行方法，原生 HunyuanVideo 在 8 卡 Atlas 800I A2 上支持 `720*1280*129` 及以上规格。多卡推理需满足以下约束：

1. 混合并行策略约束：`world_size == ulysses-degree * ring-degree`（由 YAML 中 `world_size` 与 `ulysses-degree / ring-degree` 共同决定，`mm_function.sh` 会将 `world_size` 映射为 `torchrun --nproc_per_node`）；
2. 视频规格约束：`H % 16 % <SP number> == 0 or W % 16 % <SP number> == 0`；
3. 序列并行度约束：`<head num> % <SP number> == 0`，其中 `<head num> = 24`。

要使用不同的并行配置组合时，复制 `sp8.yaml` 并修改 `world_size / ulysses-degree / ring-degree / video-size / video-length` 即可：

|     video-size        | video-length | ulysses-degree x ring-degree | world_size |
|-----------------------|--------------|------------------------------|------------|
| 1280 720 or 720 1280  | 129          | 8x1, 4x2, 2x4, 1x8           | 8          |
| 1280 720 or 720 1280  | 129          | 1x5                          | 5          |
| 1280 720 or 720 1280  | 129          | 4x1, 2x2, 1x4                | 4          |
| 1280 720 or 720 1280  | 129          | 3x1, 1x3                     | 3          |
| 1280 720 or 720 1280  | 129          | 2x1, 1x2                     | 2          |
| 1104 832 or 832 1104  | 129          | 4x1, 2x2, 1x4                | 4          |
| 1104 832 or 832 1104  | 129          | 3x1, 1x3                     | 3          |
| 1104 832 or 832 1104  | 129          | 2x1, 1x2                     | 2          |
| 960 960               | 129          | 6x1, 3x2, 2x3, 1x6           | 6          |
| 960 960               | 129          | 4x1, 2x2, 1x4                | 4          |
| 960 960               | 129          | 3x1, 1x3                     | 3          |
| 960 960               | 129          | 1x2, 2x1                     | 2          |
| 960 544 or 544 960    | 129          | 6x1, 3x2, 2x3, 1x6           | 6          |
| 960 544 or 544 960    | 129          | 4x1, 2x2, 1x4                | 4          |
| 960 544 or 544 960    | 129          | 3x1, 1x3                     | 3          |
| 960 544 or 544 960    | 129          | 1x2, 2x1                     | 2          |
| 832 624 or 624 832    | 129          | 4x1, 2x2, 1x4                | 4          |
| 832 624 or 624 832    | 129          | 3x1, 1x3                     | 3          |
| 832 624 or 624 832    | 129          | 2x1, 1x2                     | 2          |
| 720 720               | 129          | 1x5                          | 5          |
| 720 720               | 129          | 3x1, 1x3                     | 3          |

**VAE 并行**：在多卡 YAML 中置 `use-vae-parallel: true` 即可启用。

**UAA（Ulysses Anything Attention）**：在 YAML 中置 `ulysses-anything: true` 可启用。开启后解除对视频规格 `H, W` 的整除约束，解除 `<head num> % <SP number> == 0` 的约束，支持 `<head num>` 大于 `<SP number>`（`<head num> = 24`）。**UAA 仅支持纯 Ulysses，不支持 Ulysses + Ring Attention 混合序列并行策略。**

## 附录：公共环境变量说明

以下环境变量由 `executor/scripts/mm_function.sh` 在启动时统一设置（作为默认值，可通过 YAML 的 `env_vars` 覆盖）：

- `PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'`：PyTorch 针对昇腾 NPU 的内存分配配置，启用"可扩展内存段"减少 OOM 风险；
- `TASK_QUEUE_ENABLE=2`：开启 task_queue 算子下发队列 Level 2 优化；
- `CPU_AFFINITY_CONF=1`：开启粗粒度绑核；
- `TOKENIZERS_PARALLELISM=false`：禁用 tokenizers 并行化；
- 自动设置 HCCL 通信相关的 `HCCL_IF_IP / HCCL_IF_BASE_PORT / HCCL_CONNECT_TIMEOUT / HCCL_EXEC_TIMEOUT`。

指定参与推理的设备：`export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` 后再执行 `bash infer.sh`。
