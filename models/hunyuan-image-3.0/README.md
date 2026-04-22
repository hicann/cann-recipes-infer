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

   Ascend Extension for PyTorch（torch_npu）为支撑PyTorch框架运行在NPU上的适配插件，本样例建议使用的Python版本为`3.11`，PyTorch版本为`2.7.1`，请先使用`pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu`安装CPU版本的torch包。

   相应的Ascend Extension for PyTorch版本为`v7.3.1-pytorch2.7.1`。可以通过`pip install torch_npu==2.7.1`进行安装，也可以从[软件包下载地址](https://gitcode.com/Ascend/pytorch/releases/v7.3.1-pytorch2.7.1)下载`release v7.3.1-pytorch2.7.1`的whl包进行安装，或者下载其源码，参考[源码编译安装](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/installation_description.md)。

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

    # 安装依赖的python库（请使用3.11版本的Python）
    cd cann-recipes-infer
    pip install -r ./models/hunyuan-image-3.0/requirements.txt
    ```
    请注意，实测发现`transformers==5.3.0`版本会产生精度问题，推荐使用`5.2.0`及以下版本。

4. 配置 CANN 环境变量。

   ```shell
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

   > 说明：`mm_function.sh` 在拉起时会自动设置 HCCL 基础参数（`HCCL_IF_IP`、`HCCL_IF_BASE_PORT`、`HCCL_CONNECT_TIMEOUT`、`HCCL_EXEC_TIMEOUT`）。若需要追加 `HCCL_SOCKET_IFNAME`、`HCCL_OP_EXPANSION_MODE` 等集群特有参数，可直接写入 `config/ep8_cfg.yaml` 的 `env_vars` 段，会自动透传给 `torchrun` 子进程。详细参数含义可参考[集合通信文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/maintenref/envvar/envref_07_0001.html#ZH-CN_TOPIC_0000002449945377__section163522499503)。

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
    --output-path ../ckpts/weight_ep8 \        # 模型权重输出路径
    --tp-attn 8 \                              # attn TP 切分份数
    --tp-moe 1 \                               # moe TP 切分份数
    --ep 8 \                                   # EP 切分份数
    --max-shard-size 5.0                       # 权重每块最大限制（G）
```
模型的Attention部分目前仅支持TP切分，可以通过`tp-attn`参数来设置并行度。模型的MoE部分支持使用TP切分方案或EP切分方案，分别可以通过`tp-moe`和`ep`两个参数来设置所要切分权重的并行度。这几个并行度需要满足以下约束：
- MoE部分的TP与EP只能选一，不能同时使能；
- Attention部分的TP并行度需要与MoE部分的并行度一致；
- `models/hunyuan-image-3.0/utils/convert_model.py`中对这3个参数的其他相关校验。

此示例中output-path名中weight_ep8是指MoE采用EP8切分方案，如前所述Attention部分目前仅支持TP，所以这个路径名未单独提升先Attention部分的切分方式。路径名可根据个人喜好确定。

在`convert_model.py`中使用了多线程技术提高并发读写速度，如果默认的并发数量性能不佳，请根据调测环境的实际情况在`models/hunyuan-image-3.0/utils/weight_convert.sh`中添加并合理设置`--max_workers`参数的值。

权重转换拉起示例：

```shell
cd models/hunyuan-image-3.0/utils
bash weight_convert.sh
```

## 推理执行

本样例通过 `bash infer.sh` 拉起，推理参数集中在 `config/*.yaml` 维护。本模型最少需要 4 个 Device 可正常运行，若需要开启 `CFG_PARALLEL`，则需要 8 个 Device 可正常运行；预置配置为 EP8 + CFG 并行 + VAE 并行共 16 卡。

### 1. 选择推理配置

`config/` 目录下已预置以下 YAML：

| YAML 文件 | 卡数 | 启动器 | 适用场景 |
|-----------|------|--------|----------|
| `ep8_cfg.yaml` | 16 | torchrun | Attn TP8 + MoE EP8 + CFG 并行 + VAE 并行 |

预置 YAML 关键字段：

```yaml
model_name: "hunyuan-image-3.0"
world_size: 16
master_port: 10086
entry_script: "run_image_gen.py"

env_vars:
  CFG_PARALLEL: "1"                 # 开启 CFG 并行（nproc_per_node 需为原来的 2 倍）
  USE_VAE_PARALLEL: "1"             # 开启 VAE 并行
  CPU_AFFINITY_CONF: "2"            # 自动绑核，缓解 HOST 下发瓶颈

model_args:
  reproduce: true
  model-id: "./ckpts/weight_ep8"    # 指向权重转换后的目录
  prompt: "A cinematic medium shot captures a single Asian woman seated on a chair within a dimly lit room, creating an intimate and theatrical atmosphere."
  attn-impl: "npu"                  # 使能 NPU 上的 Attention 实现
  moe-impl: "npu_grouped_matmul"    # 使能 NPU 上的 MoE 实现
  moe-ep: true                      # MoE 使用 EP；与 moe-tp 互斥
  seed: 42
  diff-infer-steps: 50
  image-size: "1024x1024"
  verbose: 0
```

> 如需切换到 MoE TP 切分方案，将 `moe-ep` 改为 `moe-tp: true`，并保证 `model-id` 指向对应切分后的权重。两者互斥，只能二选一。

### 2. 修改推理输入

根据需要修改 YAML 中 `model_args` 下的字段，YAML → 命令行透传规则：

| YAML 类型 | 命令行效果 | 示例 |
|-----------|-----------|------|
| 字符串/数字 | `--key value` | `seed: 42` → `--seed 42` |
| 布尔 `true` | `--key`（flag） | `reproduce: true` → `--reproduce` |
| 布尔 `false` | 忽略 | `moe-ep: false` → 不添加 |
| 列表 | `--key v1 v2 …` | `video-size: [720, 1280]` → `--video-size 720 1280` |

完整参数说明详见 [HunyuanImage-3.0 官方 Command Line Arguments](https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/tree/62da220178f4b0b7d83e91665a46a20a3ee4f7cd#4%EF%B8%8F%E2%83%A3-command-line-arguments)。与原开源仓库版本相比，NPU 适配在以下参数上做了扩展：

- `--attn-impl`：新增 `npu` 选项，用于使能 NPU 的 Attention 实现；
- `--moe-impl`：新增 `npu_grouped_matmul`，用于使能 NPU 的 MoE 实现；
- `--moe-tp` / `--moe-ep`：一对互斥参数，选其一指定 MoE 的并行方式（TP 或 EP），需与"权重准备与转换"一致。Attention 部分仅支持 TP，与 MoE 的并行度相同。

关于 YAML 中环境变量开关的效果：

- `CFG_PARALLEL=1`：开启 CFG 并行，推理性能提升，但需要将 `world_size` 相对于原 TP 规模翻倍。例如 Attn TP8 时若关闭 CFG 则 `world_size: 8`，开启 CFG 则需要 `world_size: 16`；
- `USE_VAE_PARALLEL=1`：开启 VAE 并行，推理性能提升；
- `CPU_AFFINITY_CONF=2`：开启[自动绑核](https://www.hiascend.com/document/detail/zh/Pytorch/730/ptmoddevg/Frameworkfeatures/docs/zh/framework_feature_guide_pytorch/automatic_core_binding.md)，避免 CPU 核心抢占。

通过 `export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15` 指定参与推理的 Device，数量不能少于 YAML 中的 `world_size`。更多环境变量请参考 [CANN 社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/maintenref/envvar/envref_07_0028.html)。

### 3. 拉起推理

```shell
cd models/hunyuan-image-3.0
bash infer.sh
```

拉起过程会自动：
- 解析 YAML 中的 `world_size`、`master_port`、`entry_script`、`env_vars`、`model_args`；
- 设置通用 NPU 优化环境变量（`PYTORCH_NPU_ALLOC_CONF`、`TASK_QUEUE_ENABLE`、`TOKENIZERS_PARALLELISM`）、YAML 中声明的环境变量（`CFG_PARALLEL`、`USE_VAE_PARALLEL`、`CPU_AFFINITY_CONF` 会覆盖默认值），以及 HCCL 通信配置；
- 在 `res/<YYYYMMDD>/<model_name>/` 下自动创建日志目录，并将推理输出 `tee` 到 `log_<timestamp>.log`；
- 调用 `torchrun --master_port=<master_port> --nproc_per_node=<world_size> run_image_gen.py <model_args>`。

在 `run_image_gen.py` 中，以循环的形式对 `model.generate_image()` 调用了多次，前面几次可以视为 warmup，多轮推理后的性能趋于稳定，而就精度而言则每次都是一致的。

本样例测试结果如下：

|            Model             |   Environment  |   Attn   |   MoE   | CFG Parallel | VAE Parallel |     E2E      |
| :--------------------------: | :------------: | :------: | :-----: | :----------: | :----------: | :----------: |
|       hunyuan-image-3.0      |       A3       |   TP8    |   TP8   |    enable    |    enable    |    10.3s     |
|       hunyuan-image-3.0      |       A3       |   TP8    |   EP8   |    enable    |    enable    |     9.9s     |

### 4. 性能分析

如果需要分析 profiling，可以将 `adaptor_patches/hunyuan_image_e_pipeline.py` 中的 `enable_prof = False if idx_round > 1 else False` 改为 `enable_prof = True if idx_round > 1 else False`，并确保 `run_image_gen.py` 中对 `model.generate_image()` 调用 2 次以上，这样在第 2 次及以后便会自动采集 profiling 数据。相关配置可以在 `models/hunyuan-image-3.0/adaptor_patches/hunyuan_image_3_pipeline.py` 中配置，参考 [torch_npu.profiler 接口](https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu-profiler_list.md)。

## 优化点参考

本样例采用的详细优化点介绍可参见[NPU HunyuanImage-3.0模型推理优化实践](../../docs/models/hunyuan-image-3.0/hunyuan_image_3_optimization.md)。