# 在昇腾Atlas A2/A3/950环境上适配Wan2.2-I2V模型的推理
Wan2.2-I2V模型是一款多模态视频生成模型，提供了图生视频功能。本项目旨在提供 Wan2.2-I2V 模型的 Atlas A2/A3/950 适配版本，为开发者开展相关 NPU 迁移工作提供参考。

本项目基于NPU主要完成以下优化点，具体内容可至[NPU Wan2.2-I2V模型推理优化实践](https://gitcode.com/cann/cann-recipes-infer/blob/master/docs/models/wan2.2-i2v/wan2.2-i2v_optimization.md)查看：

- 支持NPU npu_fused_infer_attention_score融合算子；
- 支持NPU npu_rotary_mul融合算子；
- 支持NPU npu_rms_norm融合算子；
- 支持NPU npu_layer_norm_eval融合算子；
- 支持多卡VAE并行；
- 支持CFG并行;
- 支持Ring_Attention序列并行和通算掩盖。


## 执行样例
本样例支持Atlas A2/A3/950环境的单卡/多卡推理。
> 使用CANNLab一站式开发平台的用户可直接跳转[CANNLab一站式开发平台的快速启动](#cannlab一站式开发平台的快速启动)章节。

### CANN环境准备

#### 1. 安装CANN软件包

本样例的编译执行依赖CANN开发套件包与CANN二进制算子包，不同硬件版本要求不一致：
- 昇腾Atlas A2/A3环境：支持的CANN软件版本>=`CANN 8.5.0`
- 昇腾Atlas 950环境：支持的CANN软件版本>=`CANN 9.0.0`

可选择：手动安装或直接使用cann两种方式
**a. cann镜像**
请登录[cann镜像地址](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884)，点击**镜像下载**，根据芯片，cann版本，系统架构选择对应镜像，比如`8.5.0-910b-ubuntu22.04-py3.11` 或者 `9.0.1-950-ubuntu22.04-py3.12`，点击**立即下载**按指导进行docker pull。 完成镜像拉取后，使用如下指令开启镜像：
```
docker run -u root -itd --name cann_recipes_infer_image --ulimit nproc=65535:65535 --ipc=host \
--device=/dev/davinci0 \
--device=/dev/davinci1 \
--device=/dev/davinci2 \
--device=/dev/davinci3 \
--device=/dev/davinci4 \
--device=/dev/davinci5 \
--device=/dev/davinci6 \
--device=/dev/davinci7 \
--device=/dev/davinci_manager --device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v your_develop_path:your_develop_path \
-v /etc/localtime:/etc/localtime \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /etc/ascend_install.info:/etc/ascend_install.info -v /var/log/npu/:/usr/slog \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
-v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/sbin:/usr/local/sbin \
-v /etc/hccn.conf:/etc/hccn.conf -v /root/.pip:/root/.pip -v /etc/hosts:/etc/hosts \
-v /usr/bin/hostname:/usr/bin/hostname \
--net=host \
--shm-size=128g \
--privileged \
swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:${version}-${soc}-${os}-py${python_version}-devel /bin/bash
```
注意：需要将权重路径和源码路径挂载到容器里。以上示例挂载 8 个 NPU 设备，并挂载源码与权重目录。

**b. 手动安装**
请下载Ascend-cann-toolkit_${version}_linux-${arch}.run`和`Ascend-cann-${soc}-ops_${version}_linux-${arch}.run`软件包：
- 昇腾Atlas A2/A3环境：支持的CANN软件版本>=`CANN 8.5.0`, [下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0)
- 昇腾Atlas 950环境：支持的CANN软件版本>=`CANN 9.0.0`,[下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0)

安装步骤请按硬件对应的软件版本参考文档：
- 昇腾Atlas A2/A3环境：[CANN 8.5.0安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=netconda&OS=Debian)
- 昇腾Atlas 950环境：[CANN 9.0.0安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0090.html?OS=Ubuntu&InstallType=localpack)

其中：
- `${soc}`表示芯片版本，如910b、A3、950。
- `${version}`表示CANN包版本号，如8.5.0，9.0.0。
- `${arch}`表示CPU架构，如aarch64、x86_64。
- `${os}`表示操作系统版本，如ubuntu22.04。
- `${python_version}`表示Python版本，如3.11、3.12。


#### 2. 安装Ascend Extension for PyTorch（torch_npu）。

Ascend Extension for PyTorch（torch_npu）为支撑PyTorch框架运行在NPU上的适配插件，按硬件选择相关版本

- A2/A3 ： Ascend Extension for PyTorch版本为`v7.3.0`，PyTorch版本为`2.7.1`。
- 950 ： Ascend Extension for PyTorch版本为`v26.0.0`，PyTorch版本为`2.7.1`。

按照上述版本，根据当前环境的系统架构（x86_64 or AArch64）和Python版本（建议3.12），参考对应的安装文档：
- A2/A3：[torch_npu v7.3.0安装文档](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md)
- 950：[torch_npu v26.0.0安装文档](https://www.hiascend.com/document/detail/zh/Pytorch/2600/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md)

### 依赖安装

本仓库依赖于Wan2.2的开源仓库代码。

首先进入Wan2.2的仓库，下载开源仓库代码：

```
git clone https://github.com/Wan-Video/Wan2.2.git
```



下载本仓库代码：

```
git clone https://gitcode.com/cann/cann-recipes-infer.git
```



将Wan2.2仓库的代码以**非覆盖模式**复制到本项目目录下：


```
cp -rn Wan2.2/* cann-recipes-infer/models/wan2.2-i2v/
```

```
#安装Python依赖
pip install -r requirements.txt
```


### 准备模型权重


| 模型 |版本  |
|--|--|
| Wan2.2-I2V | [BF16](https://www.modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B-BF16/files?version=) |


  下载Wan2.2-I2V-A14B-BF16模型权重到本地路径`ckpts`。
```
Wan2.2-I2V-A14B-BF16/
├── configuration.json
├── gitattributes
├── google/
│   └── umt5-xxl/
│       ├── special_tokens_map.json
│       ├── spiece.model
│       ├── ...
├── high_noise_model/
│   ├── config.json
│   ├── diffusion_pytorch_model-00001-of-00006
│   ├── ...
│   └── diffusion_pytorch_model.index.json
├── low_noise_model/
│   ├── config.json
│   ├── diffusion_pytorch_model-00001-of-00006
│   ├── ...
│   └── diffusion_pytorch_model.index.json
├── models_t5_umt5-xxl-enc-bf16.pth
├── README.md
└── Wan2.1_VAE.pth
```


## 快速启动

本样例通过 `bash infer.sh` 拉起，推理参数集中在 `config/*.yaml` 维护。

### 1. 配置 CANN 环境变量

执行推理前先完成 CANN 环境变量配置：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

> 注意：`PYTHONPATH` 不会也不应全局设置项目根路径，`mm_function.sh` 会在 `torchrun` 子进程中按需注入，避免影响 CANN TBE 编译器的内置 Python 解释器。

### 2. 选择推理配置

`config/` 目录下已预置以下 YAML，分别对应不同的推理场景：

| YAML 文件 | 卡数 | 适用场景 |
|-----------|------|----------|
| `14b_single.yaml` | 1 | 单卡基线；`dit_cache.method` 可选 `NoCache / FBCache / TeaCache`，yaml 内有注释展示各方法参数 |
| `14b_single_platform.yaml` | 1 | CANNLab一站式开发平台专用模板，由 `infer_platform.sh` 读取并生成临时本地权重配置 |
| `14b_cfg2_ulysses4.yaml` | 8 | CFG 并行 × 2 + Ulysses 序列并行 × 4 + VAE 并行，支持 `1280*720*81`、`832*480*61` 等原生规格（`cfg_size / ulysses_size / ring_size` 可按需调整，乘积等于 `world_size` 即可）|

> 多卡推理与 Dit-Cache 互斥，多卡配置里 `dit_cache.method` 固定为 `NoCache`。

### 3. 修改模型权重路径与提示词

打开所选 YAML，修改 `model_args.ckpt_dir` 指向已下载的权重目录。如需更换分辨率、帧数、图片或提示词，直接修改 `size`、`frame_num`、`image`、`prompt` 等字段即可。

YAML 中的 `model_args` 会按以下规则透传给 `generate.py`：

| YAML 类型 | 命令行效果 | 示例 |
|-----------|-----------|------|
| 字符串/数字 | `--key value` | `sample_steps: 40` → `--sample_steps 40` |
| 布尔 `true` | `--key`（flag） | `dit_fsdp: true` → `--dit_fsdp` |
| 布尔 `false` | 忽略 | `dit_fsdp: false` → 不添加 |
| 列表 | `--key v1 v2 …` | `video-size: [720, 1280]` → `--video-size 720 1280` |

> 需要强制将布尔以字面量传给解析器时（如 pyrallis/Hydra），用加引号的字符串：`key: "False"` → `--key False`。

### 4. 切换要使用的 YAML

`infer.sh` 中仅需指定一行 `YAML_FILE_NAME`：

```bash
# 默认：8 卡多卡推理
export YAML_FILE_NAME=14b_cfg2_ulysses4.yaml

# 单卡（dit_cache.method 改为 FBCache / TeaCache 即启用对应缓存）
# export YAML_FILE_NAME=14b_single.yaml
```

### 5. 拉起推理

```shell
bash infer.sh
```

拉起过程会自动：
- 解析 YAML 中的 `world_size`、`master_port`、`entry_script`、`env_vars`、`dit_cache`、`model_args`；
- 设置通用 NPU 优化环境变量（`PYTORCH_NPU_ALLOC_CONF`、`TASK_QUEUE_ENABLE`、`CPU_AFFINITY_CONF`、`TOKENIZERS_PARALLELISM`）以及 HCCL 通信配置；
- 在 `res/<YYYYMMDD>/<model_name>/` 下自动创建日志目录，并将推理输出 `tee` 到 `log_<timestamp>.log`；
- 调用 `torchrun --nproc_per_node=<world_size>` 启动 `generate.py`，并将 YAML 中的 `dit_cache` 段通过 `--cache_config <yaml>` 透传给 Python 侧的 `CacheManager`。

### 6. 并行度约束与 Dit-Cache 调参

- 序列并行约束：`nproc_per_node == cfg_size * ulysses_size * ring_size`，可通过修改 YAML 中 `cfg_size / ulysses_size / ring_size` 调整并行切分方式；
- 指定参与推理的设备：`export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` 后再执行 `bash infer.sh`；
- Dit-Cache 可调参数（位于 YAML 的 `dit_cache.params` 下）：
  - `FBCache.rel_l1_thresh`：L1 差异阈值，越大越快，质量损失越大；
  - `TeaCache.rel_l1_thresh / coefficients / warmup`：累积 L1 阈值、多项式重缩放系数、前 N 步强制完整计算。

### W8A8 量化推理

W8A8 使用两个噪声专家的 Attention/FFN Linear。准备原始 BF16 checkpoint 和量化输出目录，在模型目录执行：

```bash
python -m tools.quantization.weight_convert \
  --source /path/to/Wan2.2-I2V-A14B-BF16 \
  --output /path/to/Wan2.2-I2V-A14B-W8A8 \
  --ignored-layers /path/to/ignored_layers.json
```

`--ignored-layers` 可省略；提供时为保留 BF16 的层名或 glob 模式 JSON 列表，支持专家前缀。权重转换逐 shard 写出，不覆盖原始 checkpoint。输出保留两个专家的 `config.json`、safetensors 及其索引，`quantization_config` 描述静态通道权重与动态 token 激活。

编辑 `config/14b_quantization.yaml`：`ckpt_dir` 指向原始 BF16 checkpoint，供 T5/VAE 和额外 BF16 回退使用；`quant_data_dir` 指向量化输出根目录。随后执行：

```bash
YAML_FILE_NAME=14b_quantization.yaml bash infer.sh
```

推理依据产物 `config.json.quantization_config` 加载 W8A8。`quant_layout` 选择 `nd_view`、`nd_contiguous` 或 `nz`；`quant_ignored_layers` 可额外指定从原始 checkpoint 恢复的 BF16 层。量化推理使用单卡、NoCache 配置。

I2V 的四项量化配置统一来自启动 YAML 的 `model_args`，初始化时写入模型 `cfg`。使用 `infer.sh` 时编辑对应 YAML；直接运行 `generate.py` 时通过 `--cache_config` 选择该 YAML。通用 CLI 参数保留供启动器兼容及其他任务使用，I2V 量化配置以 YAML 为准。

`quant_mode` 的取值和含义保持通用入口的 `0/1/2/3`：0 为普通推理，1 为导出校准数据，2 为导出量化模型，3 为量化推理。I2V 的 1/2 保留基线分支，不参与产物自动加载；本 W8A8 权重转换使用上述离线命令。

未指定 `save_file` 时沿用推理入口按时间生成的文件名。量化策略见 [模型优化文档](../../docs/models/wan2.2-i2v/wan2.2-i2v_optimization.md#w8a8-量化)。

## CANNLab一站式开发平台的快速启动

本章节面向使用CANNLab一站式开发平台的用户，平台已预置完整的 CANN 环境，按以下步骤即可在单卡上完成 Wan2.2-I2V 图生视频推理。

### 1. 安装 Miniconda 并创建 Python 环境

```shell
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate
conda init --all
```

关闭当前终端并重新打开后，执行：

```shell
conda create -n wan22 python=3.11 -y
conda activate wan22
```

### 2. 配置脚本路径

打开 `infer_platform.sh`，按平台实际路径检查并修改以下两项：

```bash
WEIGHTS_DIR="/mnt/workspace/gitCode/cann/models"
CANN_SET_ENV="/home/developer/Ascend/ascend-toolkit/set_env.sh"
```

### 3. 启动推理

```shell
cd cann-recipes-infer/models/wan2.2-i2v
bash infer_platform.sh
```

前置条件：已完成 §1（conda 环境已激活）。脚本本身会依次完成：source CANN 环境脚本 → 安装 torch / torch_npu → 合入上游 Wan2.2 源码 → 安装项目依赖 → 通过 ModelScope 下载权重 → 生成指向本地权重的临时 YAML → 启动单卡推理。各步骤均幂等，已完成的环节会自动跳过。

说明：`infer_platform.sh` 会读取 `config/14b_single_platform.yaml` 作为平台模板，目前默认规格为 61 帧 640*360。普通 `infer.sh` 与CANNLab一站式开发平台启动互不影响。

## 性能数据

本样例在 Atlas A2 和 Atlas 950 上的端到端推理性能如下表所示（Ulysses=8，Ring_size=1，CFG_size=1）：

| 规格 | Atlas 800I A2 - 8卡 / s | Atlas 950 - 8卡 / s |
|--|:--:|:--:|
| 832 x 480 x 61 | 102.54 | 52.24 |
| 1280 x 720 x 81 | 418.01 | 303.17 |

## 附录

### 公共环境变量说明

以下环境变量由 `executor/scripts/mm_function.sh` 在启动时统一设置（作为默认值，可通过 YAML 的 `env_vars` 覆盖）：

- `PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'`：PyTorch 针对昇腾 NPU 的内存分配配置，启用"可扩展内存段"减少 OOM 风险，详见[官方文档](https://www.hiascend.com/document/detail/zh/Pytorch/710/comref/Envvariables/Envir_012.html)；
- `TASK_QUEUE_ENABLE=2`：开启 task_queue 算子下发队列 Level 2 优化，将 workspace 相关任务迁移至二级流水，详见[官方文档](https://www.hiascend.com/document/detail/zh/Pytorch/710/comref/Envvariables/Envir_007.html)；
- `CPU_AFFINITY_CONF=1`：开启粗粒度绑核，将任务绑定在 NPU 业务绑核区间的 CPU 核上，避免不同卡任务之间的线程抢占，详见[官方文档](https://www.hiascend.com/document/detail/zh/Pytorch/710/comref/Envvariables/Envir_033.html)；
- `TOKENIZERS_PARALLELISM=false`：禁用 tokenizers 库内部的并行化处理。

同时会自动设置 HCCL 通信相关的 `HCCL_IF_IP / HCCL_IF_BASE_PORT / HCCL_CONNECT_TIMEOUT / HCCL_EXEC_TIMEOUT`。

### `model_args` 字段与 `generate.py` 参数对应关系

| YAML 字段 | 含义 |
|-----------|------|
| `task` | 任务类型，取值 `i2v-A14B` |
| `ckpt_dir` | 模型权重路径 |
| `size` | 生成视频分辨率，支持 `1280*720`、`832*480` 等 |
| `frame_num` | 生成视频帧数 |
| `sample_steps` | 推理步数 |
| `dit_fsdp` / `t5_fsdp` | DiT 与 T5 使能 FSDP，降低显存占用 |
| `cfg_size` / `ulysses_size` / `ring_size` | CFG / Ulysses / Ring 并行度 |
| `vae_parallel` | 使能多卡 VAE 并行 |
| `image` | 输入图像路径 |
| `prompt` | 文本提示词 |
| `base_seed` | 随机种子 |
| `convert_model_dtype` | 单卡场景将模型权重转换至推理精度，降低显存占用 |

## 单卡 TaylorSeer、缓存 Offload 与 Prefetch

TaylorSeer 适用于 Atlas A3 单卡、BF16 `i2v-A14B`、batch 1 和串行请求。各并行度均设为 1，关闭 FSDP、VAE 并行、量化和图捕获。

按前文准备环境、权重、输入图片和上游 Wan 源码。复现固定五场景结果时，补充的上游 `Wan-Video/Wan2.2` 源码使用提交 `42bf4cfaa384bc21833865abc2f9e6c0e67233dc`，按[依赖安装](#依赖安装)的非覆盖方式复制。

```bash
cd models/wan2.2-i2v
cp config/14b_taylorseer.yaml config/14b_taylorseer_local.yaml
```

编辑本地 YAML 中的 `model_args.ckpt_dir`、`image` 和 `prompt`，然后在模型目录执行：

```bash
WAN_CONFIG=config/14b_taylorseer_local.yaml bash infer.sh
```

`WAN_CONFIG` 支持绝对路径和相对于模型目录的路径。模板默认启用 TaylorSeer 预测，缓存 `offload` 和 `prefetch` 均为 `false`。需要缓存搬运时将 `dit_cache.params.offload` 设为 `true`，同时设 `prefetch: true` 可启用异步预取。回退到 NoCache 时可使用 `14b_single.yaml`，或将本地配置的 `dit_cache.method` 改为 `NoCache` 并删除 `dit_cache.params`。参数范围见[配置参考](config/README.md#taylorseer-parameters)，实现与实测结果见[技术文档](../../docs/models/wan2.2-i2v/wan2.2-i2v_optimization.md#taylorseer)。

### 四档复现

从上述本地 YAML 复制四份配置。四档使用相同的权重、输入图片、prompt、seed、分辨率、帧数和采样步数。复现技术文档的 A3 数据时，保留模板的 `size: "640*360"`、`frame_num: 61`、`sample_steps: 40` 和 `base_seed: 100`。四档均启用 `model_args.offload_model: true` 和 `model_args.reuse_offload_memory: true`，分别控制模型权重迁移和显存池复用。

```bash
cp config/14b_taylorseer_local.yaml config/14b_taylorseer_B0_local.yaml
cp config/14b_taylorseer_local.yaml config/14b_taylorseer_B1_local.yaml
cp config/14b_taylorseer_local.yaml config/14b_taylorseer_B2_local.yaml
cp config/14b_taylorseer_local.yaml config/14b_taylorseer_B3_local.yaml
mkdir -p output/taylorseer/sample-id
```

| 档位 | `dit_cache.method` | `params.offload` | `params.prefetch` |
| --- | --- | --- | --- |
| B0 | `NoCache` | — | — |
| B1 | `TaylorSeer` | `false` | `false` |
| B2 | `TaylorSeer` | `true` | `false` |
| B3 | `TaylorSeer` | `true` | `true` |

B0 删除整个 `dit_cache.params` 映射。B1–B3 在已有的 `dit_cache.params` 中应用下列 A3 场景参数，再按表切换 `offload` 和 `prefetch`。其余参数沿用模板。

```yaml
n_derivatives: 1
skip_interval_steps: 5
warmup: 3
cutoff_steps: 2
cache_group_size: 20
```

在每档已有的 `model_args` 中补充以下字段。此处以 B0 为例，B1–B3 使用各自的档位名。

```yaml
sample_solver: unipc
sample_shift: 5.0
reuse_offload_memory: true
save_file: "output/taylorseer/sample-id/B0.mp4"
```

```bash
WAN_CONFIG=config/14b_taylorseer_B0_local.yaml bash infer.sh
WAN_CONFIG=config/14b_taylorseer_B1_local.yaml bash infer.sh
WAN_CONFIG=config/14b_taylorseer_B2_local.yaml bash infer.sh
WAN_CONFIG=config/14b_taylorseer_B3_local.yaml bash infer.sh
```

每个样本使用独立的输出目录。启动脚本完成常规预热与生成，并将结果写入本档的 `save_file`。
