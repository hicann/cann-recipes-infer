# 在昇腾Atlas A2环境上适配Wan2.2-I2V模型的推理
Wan2.2-I2V模型是一款多模态视频生成模型，提供了图生视频功能。本项目旨在提供 Wan2.2-I2V 模型的 Atlas A2 适配版本，为开发者开展相关 NPU 迁移工作提供参考。

本项目基于NPU主要完成以下优化点，具体内容可至[NPU Wan2.2-I2V模型推理优化实践](https://gitcode.com/weixin_45381022/cann-recipes-infer/blob/master/docs/models/Wan2.2-I2V/Wan2.2-I2V_optimization.md)查看：

- 支持NPU npu_fused_infer_attention_score融合算子；
- 支持NPU npu_rotary_mul融合算子；
- 支持NPU npu_rms_norm融合算子；
- 支持NPU npu_layer_norm_eval融合算子；
- 支持多卡VAE并行；
- 支持CFG并行;
- 支持Ring_Attention序列并行和通算掩盖。


## 执行样例
本样例支持支持Atlas A2环境的多卡推理。

###  CANN环境准备
  1.安装CANN软件包
  
  本样例的编译执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），支持的CANN软件版本为`CANN 8.5.0.alpha002`。
  
  请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0.alpha002)下载`Ascend-cann-toolkit_{version}_linux-{arch}.run`和`Ascend-cann-kernels-{soc}_{version}_linux.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Debian&Software=cannToolKit)进行安装。
  
  2.安装Ascend Extension for PyTorch（torch_npu）
  
  Ascend Extension for PyTorch（torch_npu）为支撑PyTorch框架运行在NPU上的适配插件，本样例支持的Ascend Extension for PyTorch版本为`torch-npu == 2.1.0.post17`，支持的Torch版本为`torch == 2.1.0`，详细内容可见[官方文档](https://pypi.org/project/torch-npu/2.1.0.post17/)。
  
  
  


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

| YAML 文件 | 卡数 | Dit-Cache | 适用场景 |
|-----------|------|-----------|----------|
| `14b_single.yaml` | 1 | NoCache | 单卡基线，无加速 |
| `14b_single_fbcache.yaml` | 1 | FBCache | 单卡 + FBCache 加速（`rel_l1_thresh` 越大越快，质量越低） |
| `14b_single_teacache.yaml` | 1 | TeaCache | 单卡 + TeaCache 累积 L1 阈值加速，带 warmup |
| `14b_cfg2_ulysses4.yaml` | 8 | NoCache | CFG 并行 × 2 + Ulysses 序列并行 × 4 + VAE 并行，支持 `1280*720*81`、`832*480*61` 等原生规格（`cfg_size / ulysses_size / ring_size` 可按需调整，乘积等于 `world_size` 即可）|

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

# 单卡 + FBCache
# export YAML_FILE_NAME=14b_single_fbcache.yaml
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

## 性能数据

本样例的多卡端到端推理性能如下表所示（Ulysses=8，Ring_size=1，CFG_size=1)：

| 规格 | Atlas 800I A2 - 8卡 / s |
|--|:--:|
| 832*480*61 | 102.54 |
| 1280*720*81 | 418.01 |

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
 

