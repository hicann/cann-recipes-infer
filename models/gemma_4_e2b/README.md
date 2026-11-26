# Gemma-4-E2B 模型在 NPU 上推理

## 概述

Gemma-4-E2B 是 Google Gemma-4 系列模型的 E2B 规模版本。本样例仅适配官方 checkpoint 的纯文本生成路径，不覆盖视觉、音频和其他多模态输入；模型通过统一执行器注册为 `gemma_4_e2b`，完成对应的 NPU 推理适配。

- HuggingFace: https://huggingface.co/google/gemma-4-E2B
- 架构: 纯文本 Decoder，结构配置以官方 checkpoint 的 `config.json` 为准
- 参数规模: E2B，BF16

## 支持的产品型号

<term>Atlas A2 系列产品</term>
<term>Atlas A3 系列产品</term>

## 环境准备

1. 安装 CANN 软件包。

   本样例的编译执行依赖 CANN 开发套件包与 CANN 二进制算子包，支持的 CANN 软件版本为 `CANN 9.0.0`。

   请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0)下载与目标平台匹配的 `Ascend-cann-toolkit_${version}_linux-${arch}.run`、算子包、驱动和固件，并参考 [CANN 安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0090.html?OS=Ubuntu&InstallType=localpack)进行安装。

   - Atlas A2 使用 `Ascend-cann-A2-ops_${version}_linux-${arch}.run`。
   - Atlas A3 使用 `Ascend-cann-A3-ops_${version}_linux-${arch}.run`。

   - `${version}` 表示 CANN 包版本号，如 9.0.0。
   - `${arch}` 表示 CPU 架构，如 aarch64、x86_64。

2. 安装 Ascend Extension for PyTorch（torch_npu）。

   Ascend Extension for PyTorch（torch_npu）为支撑 PyTorch 框架运行在 NPU 上的适配插件。本样例使用 PyTorch `2.8.0`、torch_npu `2.8.0.post4`，Python 版本为 3.11。

   请从[软件包下载地址](https://gitcode.com/Ascend/pytorch/releases/v26.0.0-pytorch2.8.0)下载 `torch_npu-2.8.0.post4-cp311-cp311-manylinux_2_28_${arch}.whl` 安装包，并参考 [torch_npu 安装文档](https://www.hiascend.com/document/detail/zh/Pytorch/2600/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md)进行安装。

   - `${arch}` 表示 CPU 架构，如 aarch64、x86_64。

   下载后在已加载 CANN 环境的 Python 3.11 虚拟环境中安装 wheel：

   ```bash
   pip install /path/to/torch_npu-2.8.0.post4-cp311-cp311-manylinux_2_28_${arch}.whl
   ```

3. 下载项目源码并安装依赖的 Python 库。

   ```bash
   git clone https://gitcode.com/cann/cann-recipes-infer.git
   cd cann-recipes-infer
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r ./models/gemma_4_e2b/requirements.txt
   ```

   `requirements.txt` 中的 PyTorch、Transformers、safetensors、huggingface-hub、datasets 等第三方库从 [Python Package Index](https://pypi.org/) 或环境提供的官方镜像获取。参考端使用独立环境；如果环境不能直接访问 PyPI，可将 pip 源替换为可信镜像。

4. 配置样例运行所需环境信息。

   本样例按单卡、单逻辑设备、BF16、1TP、batch=1 配置。验收环境为 Atlas A2、Ascend 910B3、64 GiB HBM；Atlas A3 使用与平台匹配的 CANN、驱动和固件包。

   修改 `executor/scripts/set_env.sh` 中的 `cann_path`，例如：

   ```bash
   cann_path=/usr/local/Ascend/ascend-toolkit/latest
   ```

   运行前加载对应安装目录下的 `set_env.sh`，并确认驱动、固件与 CANN 版本匹配。

## 权重准备

固定官方权重 revision：`d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f`。配置、权重和 tokenizer 必须来自同一 revision，权重文件不提交 Git。

- HuggingFace: https://huggingface.co/google/gemma-4-E2B/tree/d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f

从 HuggingFace 下载官方原始 checkpoint 到本地目录：

```bash
hf download google/gemma-4-E2B \
  --revision d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f \
  --local-dir /data/models/gemma-4-E2B
```

本任务直接加载官方 BF16 checkpoint，不需要权重转换。下载完成后，将 YAML 中的 `model_config.model_path` 设置为 `/data/models/gemma-4-E2B` 或实际权重目录。

## 推理执行

1. 配置 YAML：将 `model_config.model_path` 改为本地权重路径；其他字段含义见 [YAML 参数说明](../../docs/common/inference_config_guide.md)。

   - Eager 配置：`config/gemma_4_e2b_bf16_1tp.yaml`
   - Decode 图配置：`config/gemma_4_e2b_bf16_1tp_npugraph.yaml`

2. 准备推理输入。默认使用 YAML 中的 `data_config.dataset: "default"`，输入截断长度为 2048，最大生成长度为 128；输入使用官方 base tokenizer 和原始文本。

3. 从仓库根目录执行统一入口：

   ```bash
   bash executor/scripts/infer.sh --model gemma_4_e2b --mode offline \
     --yaml gemma_4_e2b_bf16_1tp.yaml
   ```

   使用 Decode 图模式时，将 YAML 文件替换为 `gemma_4_e2b_bf16_1tp_npugraph.yaml`。
