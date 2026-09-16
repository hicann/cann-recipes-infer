# GLM-5.3-Flash Inference on NPU

**中文** | [English](./README_eng.md)

## 概述

GLM-5.3-Flash 是智谱发布的**混合注意力 + mHC + MoE** 大模型，本实践适配 GLM-5.3-Flash 权重（~306B 总参数，发布格式为 FP8 量化），在 CANN 平台上完成量化与性能优化，面向昇腾 `950DT` 平台 **Hybrid HiF8-MXFP8-MXFP4** 推理部署。

- **混合注意力**：45 层中 34 层为 **KDA**（Kimi Delta Attention，线性注意力，短卷积 + 门控 delta 递推），每 4 层插入 1 层 **MLA + DSA**（共 11 层：吸收式 MLA 全程 NoPE + k-pool 压缩 indexer 的稀疏注意力）。
- **mHC**（Manifold-Constrained Hyper-Connections）：层间隐藏态是 4 条并行流 `[T, 4, D]`，每个子层前用 Sinkhorn 归一化的组合矩阵折叠、子层后再展开。
- **MoE**：288 路由专家 top-8 + 1 共享专家，前 `first_k_dense_replace` 层为稠密 MLP，共享专家侧支持多流与路由专家计算并行。
- **量化方案**：**Hybrid HiF8-MXFP8-MXFP4**，DSA 投影 + dense MLP 走 **HiF8 W8A8**，共享专家走 **MXFP8 W8A8**，路由专家走 **MXFP4 W4A8**。KDA 全部权重、DSA 的 `kv_b_proj`、整个 indexer、mHC 参数、各 norm、`mlp.gate`、embed/lm_head 保持 BF16/FP32。
- **图模式**：decode 默认使用 **npugraph_ex**。

---

## 硬件要求

| 项 | 要求 |
|----|------|
| 产品型号 | 昇腾 `950DT` 系列 |
| 操作系统 | Linux ARM |
| 部署规模 | 默认 `world_size: 8`（见 `config/glm_5_3_flash_ep8.yaml`），可按实际卡数调整；MoE EP = world_size |
| 驱动 / 固件 | 按所在平台的标准要求安装；`npu-smi info` 确认固件与驱动已正确安装 |
| CANN | 安装到固定路径（如 `/usr/local/Ascend/cann`） |
| 自定义算子包 | `opp/vendors/customize`（mHC AscendC 算子）与 `opp/vendors/custom_transformer`（causal_conv1d），见「快速启动 · 2」 |

---

## 快速启动

### 1. 下载源码

```shell
mkdir -p /home/code && cd /home/code
git clone https://gitcode.com/cann/cann-recipes-infer.git
cd cann-recipes-infer
```

### 2. 安装依赖

```shell
pip3 install -r models/glm_5_3/requirements.txt
```

本模型依赖两个算子包：

| 包 | 提供的算子 |
|---|---|
| `custom_ops` | `npu_hc_pre` / `npu_hc_post` |
| `cann_ops_transformer` | `causal_conv1d_fn` / `causal_conv1d_update` |


拉起前 source 对应的 vendor 环境：

```shell
source models/glm_5_3/set_env.sh
```

### 3. 下载权重

下载 [GLM-5.3-Flash 原始权重](https://huggingface.co/zai-org/GLM-5.3-Flash)（FP8）到各节点的固定路径下，例如 `/data/models/GLM-5.3-Flash`。

### 4. 下载数据集

默认配置为 `dataset: "InfiniteBench"`，从[链接](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/blob/main/longbook_qa_eng.jsonl)下载 `longbook_qa_eng.jsonl` 放到 `dataset/InfiniteBench/` 下。

> 目前支持的 `dataset` 取值：`default` / `LongBench` / `InfiniteBench`。

### 5. 转换权重（FP8 → HiF8）

用 `utils/convert_model.py` 把发布的 FP8 权重转成运行时可执行的格式。该脚本同时改写 `config.json` 的 `quantization_config`。

| 入参 | 默认 | 说明 |
|------|------|------|
| `--input_hf_path` | — | **必填**，FP8 HF 权重路径 |
| `--output_hf_path` | — | **必填**，输出路径（不支持原地转换） |

```shell
# Hybrid HiF8-MXFP8-MXFP4
python models/glm_5_3/utils/convert_model.py \
    --input_hf_path /data/models/GLM-5.3-Flash \
    --output_hf_path /data/models/GLM-5.3-Flash-HIF8
```

### 6. 拉起多卡推理

- 修改根目录公共的 `../../executor/scripts/set_env.sh`：配置 `IPs`（各节点 IP，按 rank 排序，空格分隔，第一个为主节点）与 `cann_path`。
- 修改 `config/glm_5_3_flash_ep8.yaml` 中的 `model_path` 为上一步转换后的权重路径。YAML 通用参数说明见 [YAML 参数描述](../../docs/common/inference_config_guide.md)。图模式支持 `eager` 和 `npugraph_ex`，模型暂不支持 MTP 功能。

除框架统一配置外，本模型支持以下可选参数，配置在 `model_config.custom_params` 中：

| 参数名 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `enable_multi_streams` | bool | `True` | 共享专家多流，decode 阶段用侧流计算共享专家，与路由专家的 dispatch / GMM / combine 重叠 |

当前配置采用以下并行参数：

```yaml
parallel_config:
  world_size: 8
  cp_size: 1
  attn_tp_size: 1
  dense_tp_size: 1
  moe_tp_size: 1
  embed_tp_size: 1
  lmhead_tp_size: 1
```

- `moe_tp_size=1`，框架据此派生 `moe_ep_size = world_size`，路由专家按 Expert 维切分。
- `cp_size`、`attn_tp_size`、`dense_tp_size`、`moe_tp_size`、`o_proj_tp_size`、`shared_tp_size` 必须为 1。

配置完成后，在各节点同步执行：

```shell
bash executor/scripts/infer.sh --model glm_5_3 --yaml glm_5_3_flash_ep8.yaml
```

8K 序列下，不同 BS 吞吐如下：

| 每卡BS | Decode每步耗时(ms) | 每卡TPS | 8卡总TPS |
|--------|-------------------|---------|---------|
| 1 | 16.60 | 60.24 | 481.93 |
| 16 | 24.07 | 664.73 | 5317.82 |
| 64 | 35.69 | 1793.22 | 14345.76 |
| 128 | 49.88 | 2566.16 | 20529.27 |
