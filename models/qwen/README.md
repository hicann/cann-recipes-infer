# Qwen Dense 模型在 NPU 上推理

## 概述

`models/qwen` 适配 Qwen2/Qwen3 Dense（非 MoE）系列模型，包含 **Qwen3-8B** 与 **Qwen2.5-7B-Instruct**，共用统一建模代码与启动入口，通过 HuggingFace `config.json` 自动识别模型变体。本文给出 BF16 与 W8A8 量化在 1卡 / 2卡 TP 配置下的部署流程与实测性能基线（Qwen3-8B）。

| 模型 | model_name | 特性 |
|---|---|---|
| Qwen3-8B | `qwen3_8b` | QK-Norm，attention_bias=False，支持 W8A8 INT8 量化 |
| Qwen2.5-7B-Instruct | `qwen25_7b_instruct` | 无 QK-Norm，attention_bias=True |

---

## 模型特性

| 特性 | 说明 |
|---|---|
| 架构 | Qwen3 / Qwen2.5 Dense（非 MoE）|
| Attention | GQA（Qwen3-8B：query head 32 / kv head 8）|
| MLP | gate_proj / up_proj / down_proj 三路 |
| Packed Sequence | Prefill / Decode 阶段均使用 TND 打包序列 |
| Page Attention | 块式 KV Cache 管理 |
| 量化 | Qwen3-8B 支持 compressed-tensors 格式 W8A8 INT8 |

---

## 支持的产品型号

<term>Atlas A2 系列产品</term>
<term>Atlas A3 系列产品</term>

---

## 软件版本

| 软件 | 版本 |
|---|---|
| CANN | 9.0.0+ |
| torch_npu | v26.0.0 |
| PyTorch | 2.8.0 |
| transformers | 5.0.0 |
| compressed-tensors | 0.6.0 |
| Python | 3.11 |

---

## 环境准备

1. 安装 CANN 软件包。

   本样例的编译执行依赖 CANN 开发套件包（cann-toolkit）与 CANN 二进制算子包（cann-kernels），支持的 CANN 软件版本为 `CANN 9.0.0`。

   请从 [软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0) 下载 `Ascend-cann-toolkit_${version}_linux-${arch}.run` 与 `Ascend-cann-A3-ops_${version}_linux-${arch}.run` 软件包，并参考 [CANN 安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0090.html?OS=Ubuntu&InstallType=localpack) 进行安装。
   - `${version}` 表示 CANN 包版本号，如 9.0.0。
   - `${arch}` 表示 CPU 架构，如 aarch64、x86_64。

2. 安装 Ascend Extension for PyTorch（torch_npu）。

   Ascend Extension for PyTorch（torch_npu）为支撑 PyTorch 框架运行在 NPU 上的适配插件，本样例支持的 Ascend Extension for PyTorch 版本为 `v26.0.0`，PyTorch 版本为 `2.8.0`。

   请从 [软件包下载地址](https://gitcode.com/Ascend/pytorch/releases/v26.0.0-pytorch2.8.0) 下载 `torch_npu-2.8.0.post4-cp311-cp311-manylinux_2_28_${arch}.whl` 安装包，参考 [torch_npu 安装文档](https://www.hiascend.com/document/detail/zh/Pytorch/2600/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md) 进行安装。

3. 下载项目源码。

   ```bash
   # 下载项目源码，以 master 分支为例
   git clone https://gitcode.com/cann/cann-recipes-infer.git
   cd cann-recipes-infer
   ```

4. 配置样例运行所需环境信息。

   修改 `executor/scripts/set_env.sh` 脚本中的如下字段：
   - `cann_path`：CANN 软件包安装路径，例如 `/usr/local/Ascend/ascend-toolkit/latest`。

   > 说明：HCCL 相关配置（如 `HCCL_SOCKET_IFNAME`、`HCCL_OP_EXPANSION_MODE`）可参考 [集合通信文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/maintenref/envvar/envref_07_0001.html) 并在 `executor/scripts/function.sh` 中自定义配置。

---

## 权重准备

### BF16 原始权重

从 HuggingFace 下载原始权重到本地路径，例如 `/data/models/origin/Qwen3-8B`：

- [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

下载方式（任选其一）：

```bash
# 通过 huggingface-cli（国内访问慢可设镜像：export HF_ENDPOINT=https://hf-mirror.com）
huggingface-cli download Qwen/Qwen3-8B --local-dir /data/models/origin/Qwen3-8B

# 或通过 git lfs（需先 git lfs install）
git clone https://huggingface.co/Qwen/Qwen3-8B /data/models/origin/Qwen3-8B
```

> 大文件超时可重新执行命令，`huggingface-cli` 自动断点续传。

下载完成后将所选 YAML 中的 `model_path` 字段指向该路径。

### W8A8 INT8 量化权重（Qwen3-8B）

本样例使用昇腾官方大模型量化工具 [AMCT (Ascend Model Compression Toolkit)](https://gitcode.com/cann/amct) 通过命令行直接导出 Qwen3-8B W8A8 INT8 部署权重。流程如下：

1. **获取 AMCT 源码**

   ```bash
   git clone https://gitcode.com/cann/amct.git
   ```

2. **安装 AMCT 依赖**

   ```bash
   pip install -r amct/requirements.txt
   pip install 'setuptools<82'    # AMCT requirements 未锁 setuptools 版本，避免升到 >=82 导致 pkg_resources 缺失
   ```

3. **命令行直接导出 Qwen3-8B W8A8 部署权重**

   在 amct 源码根目录执行（执行前确认已 source CANN 环境，见 [环境准备](#环境准备) §4）：

   ```bash
   cd amct
   PYTHONPATH=. python3 -m amct_pytorch.deploy \
       --model /data/models/origin/Qwen3-8B \
       --model_name qwen3 \
       --granularity block \
       --quant_target attn-linear mlp \
       --quant_dtype int \
       --bit_config amct_pytorch/configs/w8a8.yaml \
       --output_dir /data/models/origin/Qwen3-8B-W8A8
   ```

   > `--quant_target attn-linear mlp` 量化 attention 4 线性层 + MLP gate_up/down，与本仓 W8A8 baseline 测试范围一致；省略 `attn-linear` 仅量化 MLP 会导致 attention 保持 BF16 计算，精度退化（实测输出乱码）。

4. **导出目录的兼容性要求**

   AMCT `deploy` 输出目录已是 HuggingFace 风格的 compressed-tensors 目录结构，加载到本推理路径需满足：

   | 项 | 要求 |
   |---|---|
   | 格式 | [compressed-tensors](https://github.com/neuralmagic/compressed-tensors) HuggingFace 风格目录，包含 `layer_*.safetensors`、`rest_*.safetensors`、更新后的 `model.safetensors.index.json` 与 `config.json`（内嵌 `quantization_config`） |
   | 量化模式 | W8 INT8 per-channel weight + A8 INT8 dynamic activation |
   | 量化范围 | attention 4 线性层（q/k/v/o_proj）+ MLP（gate_proj / up_proj / down_proj）全量化 |
   | ignore 列表 | 使用 HF 风格非融合层名（`q_proj` / `k_proj` / `v_proj` / `gate_proj` / `up_proj`），框架侧 `should_ignore_layer` 已通过 `fused_mapping` 将其映射到融合层（`merged_qkv_proj` / `gate_up_proj`）|

导出后 `--output_dir`（上例 `/data/models/origin/Qwen3-8B-W8A8`）即可被 W8A8 yaml 的 `model_path` 引用，具体见 [推理执行](#推理执行)。

---

## 推理执行

1. 安装 Qwen3 推理依赖。

   ```bash
   pip install -r ./models/qwen/requirements.txt
   # 若执行过 [W8A8 INT8 量化权重](#w8a8-int8-量化权重qwen3-8b) 流程，AMCT 会将 torch_npu 降级，需重装 wheel 恢复：
   pip install ./torch_npu-2.8.0.post4-cp311-cp311-manylinux_2_28_aarch64.whl
   ```

2. 配置推理执行需要加载的权重文件以及 YAML 文件。

   修改 YAML 文件中 `model_path` 参数指向 [权重准备](#权重准备) 阶段下载或生成的权重路径。关于 YAML 文件中的更多配置说明可参见 [InferenceConfig 使用指南](../../docs/common/inference_config_guide.md)。

   `models/qwen/config/` 目录下已提供以下 YAML 样例：

   | YAML | 精度 | 卡数 | 适用场景 |
   |---|---|---|---|
   | `qwen3_8b_1tp.yaml` | BF16 | 1 | Qwen3-8B 单卡部署 |
   | `qwen3_8b_2tp.yaml` | BF16 | 2 | Qwen3-8B 2 卡 TP 并行 |
   | `qwen3_8b_w8a8_1tp.yaml` | W8A8 INT8 | 1 | Qwen3-8B 单卡量化部署 |
   | `qwen3_8b_w8a8_2tp.yaml` | W8A8 INT8 | 2 | Qwen3-8B 2 卡 TP 量化部署 |
   | `qwen25_7b_instruct_1tp.yaml` | BF16 | 1 | Qwen2.5-7B-Instruct 单卡部署 |
   | `qwen25_7b_instruct_2tp.yaml` | BF16 | 2 | Qwen2.5-7B-Instruct 2 卡 TP 并行 |

3. 执行统一推理脚本。

   统一入口脚本位于 `executor/scripts/infer.sh`，通过以下参数控制启动：

   | 参数 | 含义 | 取值示例 |
   |---|---|---|
   | `--model` | 模型目录名，对应 `models/` 下的子目录 | `qwen` |
   | `--mode` | 推理模式 | `offline`（离线推理） |
   | `--yaml` | 离线模式：yaml 文件名 | `qwen3_8b_1tp.yaml` |

   **使用方式一：命令行传参**

   ```shell
   # Qwen3-8B BF16 单卡
   bash executor/scripts/infer.sh --model qwen --yaml qwen3_8b_1tp.yaml
   # Qwen3-8B W8A8 2 卡 TP
   bash executor/scripts/infer.sh --model qwen --yaml qwen3_8b_w8a8_2tp.yaml
   # Qwen2.5-7B-Instruct 单卡
   bash executor/scripts/infer.sh --model qwen --yaml qwen25_7b_instruct_1tp.yaml
   ```

   如需查看参数说明，可以执行 `bash executor/scripts/infer.sh --help`。

   **使用方式二：直接修改脚本默认值后执行**

   编辑 `executor/scripts/infer.sh`，修改 `MODEL` / `MODE` / `YAML_FILE` 等参数的默认值，例如：

   ```shell
   MODEL=qwen
   MODE=offline
   YAML_FILE=qwen3_8b_1tp.yaml
   ```

   保存后直接执行：

   ```shell
   bash executor/scripts/infer.sh
   ```

   > 说明：
   > - 如果是多机环境，需要在每个节点上执行。
   > - 推理日志和结果保存在 `models/qwen/res/<日期>/<case_name>/` 路径下。

---

## Benchmark

### 测试环境

| 项 | 值 |
|---|---|
| 硬件 | Atlas A3 系列（910C，单 die 64 GB HBM）|
| CANN | 9.0.0+ |
| torch_npu | v26.0.0 |
| 模型 | Qwen3-8B（HuggingFace 原始权重）|
| 量化权重 | compressed-tensors W8A8 INT8（attn + mlp，per-channel weight + per-token dynamic activation）|
| 执行模式 | `ge_graph`（max-autotune 图模式）|
| 数据集 | default 内置 prompt |
| 输入截断 | 4096 token |
| 输出长度 | 256 token |
| 数据采集 | 2026-06-01，transformers 5.0.0 + torch_npu 2.8.0.post4 + CANN 9.0.0，NPU idle 同窗口连测 |

### 性能数据（Qwen3-8B）

| 配置 | YAML | Prefill (ms) | Decode (ms/step) | 备注 |
|---|---|---|---|---|
| **BF16 1tp** | `qwen3_8b_1tp.yaml` | 41.12 | **15.48** | 单卡基线 |
| **BF16 2tp** | `qwen3_8b_2tp.yaml` | 51.56 | **10.35** | 2 卡 TP，decode 较 1tp **-33.1%**（KV/权重带宽双卡分担）|
| **W8A8 1tp** | `qwen3_8b_w8a8_1tp.yaml` | 36.00 | **10.46** | 量化收益：decode 较 BF16 1tp **-32.4%** |
| **W8A8 2tp** | `qwen3_8b_w8a8_2tp.yaml` | 58.95 | **7.90** | 量化收益：decode 较 BF16 2tp **-23.7%**；较 W8A8 1tp **-24.5%** |

---

## CANNLab一站式开发平台指南

本章节面向使用 CANNLab 一站式开发平台的用户，平台已预置 CANN，按以下步骤即可在单卡上完成 Qwen3-8B 推理。

- 硬件：Atlas A3 单卡环境（64 GB HBM 可容纳 Qwen3-8B BF16 / W8A8），多卡 TP 请使用标准 A3 Pod。
- CANNLab 平台 CANN 安装路径：`/home/developer/Ascend/cann`
- CANNLab 平台权重存放路径：`/home/developer/models/Qwen3-8B`（按需自定义）

### 1. 环境前置（conda + CANN + torch_npu）

> - 先用 `python --version`（期望 3.11.x）与 `python -c "import torch_npu; print(torch_npu.__version__)"`（期望 2.8.0.post4）核对环境；满足则可分别跳过 §1.1（conda 创建）与 §1.3（torch_npu wheel 安装）。§1.2 CANN 环境为必须步骤。
> - 本章节命令中所有 `aarch64` 在 x86_64 机器上替换为 `x86_64`。

**1.1 安装 Miniforge 并创建 Python 3.11 环境**

若平台 Python 不是 3.11，在用户目录下安装 Miniforge（无需 root 权限）：

```bash
# GitHub 源（超时改清华镜像 https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/Miniforge3-Linux-aarch64.sh）
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O /tmp/miniforge.sh
bash /tmp/miniforge.sh -b -p $HOME/miniforge3
source $HOME/miniforge3/etc/profile.d/conda.sh
echo "source $HOME/miniforge3/etc/profile.d/conda.sh" >> ~/.bashrc   # 永久生效
```

创建并激活 Python 3.11 隔离环境：

```bash
conda create -n qwen python=3.11 -y
conda activate qwen
```

**1.2 配置 CANN 环境变量**

```shell
sed -i 's|^cann_path=.*|cann_path="/home/developer/Ascend/cann"|' executor/scripts/set_env.sh
source executor/scripts/set_env.sh
```

**1.3 安装 torch_npu wheel**

```shell
wget https://gitcode.com/Ascend/pytorch/releases/download/v26.0.0-pytorch2.8.0/torch_npu-2.8.0.post4-cp311-cp311-manylinux_2_28_aarch64.whl
pip install ./torch_npu-2.8.0.post4-cp311-cp311-manylinux_2_28_aarch64.whl
```

### 2. 下载 Qwen3-8B BF16 原始权重到平台路径

平台无法访问 HuggingFace 时使用魔塔 (modelscope) 下载：

```shell
pip install modelscope
modelscope download --model Qwen/Qwen3-8B --local_dir /home/developer/models/Qwen3-8B
```

> 大文件超时可直接重新执行命令，`modelscope` 自动断点续传。

### 3. （可选）生成 W8A8 量化权重

仅运行 BF16 推理时可跳过本节；如需 W8A8 推理，则获取 AMCT 源码后通过命令行直接导出 W8A8 部署权重。

**3.1 获取 AMCT 源码**

```shell
git clone https://gitcode.com/cann/amct.git /home/developer/amct
```

**3.2 安装 AMCT 依赖**

```shell
pip install -r /home/developer/amct/requirements.txt
pip install 'setuptools<82'    # AMCT requirements 未锁 setuptools 版本，避免升到 >=82 导致 pkg_resources 缺失
```

**3.3 导出 W8A8 部署权重**

```shell
cd /home/developer/amct
PYTHONPATH=. python3 -m amct_pytorch.deploy \
    --model /home/developer/models/Qwen3-8B \
    --model_name qwen3 \
    --granularity block \
    --quant_target attn-linear mlp \
    --quant_dtype int \
    --bit_config amct_pytorch/configs/w8a8.yaml \
    --output_dir /home/developer/models/Qwen3-8B-W8A8
```

### 4. 修改 YAML 中的 model_path 指向平台权重路径

```shell
# BF16 单卡
sed -i 's|^  model_path:.*|  model_path: "/home/developer/models/Qwen3-8B"|' \
    models/qwen/config/qwen3_8b_1tp.yaml

# W8A8 单卡
sed -i 's|^  model_path:.*|  model_path: "/home/developer/models/Qwen3-8B-W8A8"|' \
    models/qwen/config/qwen3_8b_w8a8_1tp.yaml
```

### 5. 安装依赖 + 拉起推理

**5.1 安装 Qwen3 推理依赖**

```shell
pip install -r ./models/qwen/requirements.txt
# 若执行过 §3 AMCT 量化，AMCT 会将 torch_npu 降级，需重装 wheel 恢复：
pip install ./torch_npu-2.8.0.post4-cp311-cp311-manylinux_2_28_aarch64.whl
```

**5.2 执行统一推理脚本**

```shell
# 单卡 BF16
bash executor/scripts/infer.sh --model qwen --yaml qwen3_8b_1tp.yaml
# 单卡 W8A8
bash executor/scripts/infer.sh --model qwen --yaml qwen3_8b_w8a8_1tp.yaml
```

推理日志与结果保存在 `models/qwen/res/<日期>/<case_name>/` 路径下。

---

## 优化点参考

本样例采用的详细优化点（TP 切分 / 残差融合 / 融合算子 / 图模式 / W8A8 INT8 量化 fused dispatch / AIV 展开）介绍可参见 [基于 Atlas A2/A3 的 Qwen Dense 模型推理性能优化实践](../../docs/models/qwen/qwen_dense_optimization.md)。
