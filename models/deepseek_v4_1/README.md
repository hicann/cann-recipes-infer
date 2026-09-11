# DeepSeek-V4.1 Inference on NPU
## 概述
DeepSeek团队发布了最新的模型DeepSeek-V4.1-Flash，本实践基于DeepSeek开源代码进行迁移，并在CANN平台上完成性能优化，支持在昇腾`Ascend 950`平台部署。

- 本实践的优化特性参见[DeepSeek-V4.1-Flash CANN优化实践](../../docs/models/deepseek_v4_1/deepseek_v4.1_flash_cann_tech_report.md)。

---

## 硬件要求
产品型号：Ascend 950 系列

操作系统：Linux x86（Ascend 950）

## 环境准备
### Atlas A5 部署

1. 安装 PyTorch 和 Ascend Extension for PyTorch（torch_npu）。

   `torch_npu` 为 PyTorch 在 NPU 上运行提供适配。执行：

   ```shell
   python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu

   mkdir -p /tmp/torch_npu_290_daily
   cd /tmp/torch_npu_290_daily
   wget -O pytorch_v2.9.0_py311.tar.gz \
     "https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/Daily/v2.9.0/20260903.1/pytorch_v2.9.0_py311.tar.gz"
   tar -xzf pytorch_v2.9.0_py311.tar.gz
   python -m pip install --no-deps \
     ./torch_npu-2.9.0.post7.dev20260903-cp311-cp311-manylinux_2_28_x86_64.whl
   ```

2. 下载项目源码并安装 Python 依赖。

   ```shell
   git clone https://gitcode.com/cann/cann-recipes-infer.git
   cd cann-recipes-infer
   python -m pip install -r ./models/deepseek_v4_1/requirements.txt
   ```

3. 安装 CANN 软件包。

   本实践依赖 CANN 开发套件包（toolkit）和与目标硬件匹配的二进制算子包（ops），支持 CANN 9.2.0。请从 [CANN 下载页面](https://www.hiascend.com/cann/download)选择对应架构的 weekly 版本。下载后，按以下命令依次安装 toolkit 和 ops 软件包：

   ```shell
   chmod +x Ascend-cann-toolkit_<version>_linux-<arch>.run
   ./Ascend-cann-toolkit_<version>_linux-<arch>.run --install
   chmod +x Ascend-cann-kernels-<product>_<version>_linux.run
   ./Ascend-cann-kernels-<product>_<version>_linux.run --install
   ```

   多节点部署时，各节点应使用相同的 CANN、驱动和固件版本。

   CANN 安装完成后，根据实际安装路径加载环境变量：

   ```shell
   cann_path=/usr/local/Ascend/cann
   source "${cann_path}/bin/setenv.bash"
   ```

   新建终端后，重新加载 CANN 环境脚本即可。

4. 配置样例运行环境。

   修改 `executor/scripts/set_env.sh` 中的节点 IP 和 CANN 安装路径。其中 `cann_path` 需与实际的 CANN 安装路径保持一致，例如 `/usr/local/Ascend/cann`。

---

## 快速启动

### 下载InfiniteBench数据集
  从[链接](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/blob/main/longbook_qa_eng.jsonl)中下载长序列输入数据集longbook_qa_eng，并上传到各个节点上新建的路径`dataset/InfiniteBench`下。
  ```shell
  mkdir -p dataset/InfiniteBench
  ```
### 下载MMMU数据集
  从[链接](https://huggingface.co/datasets/MMMU/MMMU)中下载多模态理解数据集MMMU，并上传到各个节点上新建的路径`dataset/MMMU`下。

  ```shell
  mkdir -p dataset/MMMU
  ```
### 下载权重

  下载[DeepSeek-V4-1-Flash原始Hybrid FP8-MXFP4权重](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)，并上传到各节点的某个固定的路径下，比如`/data/models/deepseek_v4_1_hybrid_fp8_mxfp4`。

### 转换权重

DeepSeek-v4.1的原始权重中使用了32x32-block的量化，通过broadcast scale可以将其转化为1x32-group的标准MXFP8量化。在各个节点上进入 `models/deepseek_v4_1` 目录，使用`utils/convert_model.py` 脚本完成 Hybrid FP8-MXFP4 到 Hybrid MXFP8-MXFP4 的权重转换。

  >入参介绍：`input_fp8_hf_path`：原始权重路径；`output_hf_path`：转换后输出的权重路径；

  如果权重转换的运行环境为NPU，需要先执行：

  ```shell
  cann_path=/usr/local/Ascend/cann  # cann包安装路径
  source ${cann_path}/bin/setenv.bash
  ```

权重转换拉起示例：
```shell
python utils/convert_model.py --input_fp8_hf_path /data/models/deepseek_v4_1_hybrid_fp8_mxfp4  --output_hf_path /data/models/deepseek_v4_1_hybrid_mxfp8_mxfp4
```

### 修改配置
- 在各个节点上修改公共环境脚本 `cann-recipes-infer/executor/scripts/set_env.sh` 中的如下字段:
   - `IPs`：离线模式配置所有节点的IP，按照rank id排序，多个节点的ip通过空格分开，例如：`('xxx.xxx.xxx.xxx' 'xxx.xxx.xxx.xxx')`。
   - `cann_path`: CANN软件包安装路径，例如`/usr/local/Ascend/cann`。
- `executor/scripts/infer.sh` 会加载公共环境脚本 `executor/scripts/set_env.sh`。

- 在 yaml 配置中，默认采用 `eager` 执行方式。

- 除框架统一配置之外，DeepSeek-V4.1 还额外支持以下特性，放置在 YAML 文件 `model_config` 的 `custom_params` 字段下：

  | 参数名 | 类型 | 默认值 | 含义 |
  | --- | --- | --- | --- |
  | `enable_multi_streams` | bool | `false` | 启用模型内多流并行，主要用于 decode 阶段 MoE shared expert 等模块的并行调度。 |
  | `moe_chunk_max_len` | int | `65536` | MoE token 分发的最大 chunk 长度，用于长序列 prefill 场景规避 OOM。 |
  | `enable_engram_offload` | bool | `True` | 支持对Engram Table做host offload。 |
  | `engram_tp_size` | int | 8 | 支持对Engram Table做TP切分，当前支持`tp_size`=`world_size`。 |
  | `kernel_config` | dict | `{}` | 按算子类型覆盖 kernel 实现。当前涉及 `hc_pre`、`hc_post`、`gate_topk` 和 `compressor`；`hc_pre`/`hc_post` 当前仅有 `native`，`gate_topk` 支持 `native`/`ascendc`，默认为`ascendc`，`compressor` 支持 `native`/`ascendc`，默认为`native`。 |

### 拉起多卡推理
以下命令在仓库根目录执行。统一入口脚本位于 `executor/scripts/infer.sh`，通过以下参数控制启动：

| 参数 | 含义 | 取值示例 |
| --- | --- | --- |
| `--model` | 模型目录名，对应 `models/` 下的子目录 | `deepseek_v4_1` |
| `--mode` | 推理模式 | `offline` / `online` |
| `--yaml` | 离线模式：yaml 文件名，路径相对 `models/deepseek_v4_1/config/` | `deepseek_v4_1_flash_rank_8_8ep.yaml` |

> 在线模式 IP 等更多配置可参考 [executor 设计文档 §5.1 启动方式](../../docs/design/executor_design.md#51-启动方式)。

**使用方式一：命令行传参**
```shell
# offline 模式，Ascend 950
bash executor/scripts/infer.sh --model deepseek_v4_1 --yaml deepseek_v4_1_flash_rank_8_8ep.yaml
```

如需查看参数说明，可执行 `bash executor/scripts/infer.sh --help`。

**使用方式二：直接修改脚本默认值后执行**
编辑 `executor/scripts/infer.sh`，按需修改 `MODEL` / `MODE` / `YAML_FILE` 等参数的默认值，例如：
```shell
MODEL=deepseek_v4_1
MODE=offline
YAML_FILE=deepseek_v4_1_flash_rank_8_8ep.yaml
```
保存后直接执行：
```shell
bash executor/scripts/infer.sh
```

> 如果是多机环境，需要在每个节点上同步执行拉起命令。

> **Note：** 不同平台最小部署单元要求如下

| 平台  | 模型型号             |  推荐量化策略  | 最小部署单元（chips）|
|-------|---------------------|--------------|--------------|
| Ascend 950  | DeepSeek-V4.1 Flash    | Hybrid MXFP8-MXFP4 |8          |
