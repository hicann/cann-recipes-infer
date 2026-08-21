# Ling (BailingMoeV2_5) Inference on NPU

## 概述

Ling（`BailingMoeV2_5`）是一款**混合注意力 + MoE** 大模型。本实践基于其开源实现进行迁移，并在 CANN 平台上完成量化与性能优化，面向昇腾 `950DT` 平台部署。

- **混合注意力**：大多数层采用**线性注意力**（Gated Linear Attention，纯 torch 实现见 `models/simple_gla_torch.py`），每隔 `layer_group_size` 层插入一层 **MLA**（Multi-head Latent Attention，全注意力 + 分页 KV cache）。
- **MoE**：路由专家（GroupMatmul）+ 共享专家；前 `first_k_dense_replace` 层为稠密 MLP。共享专家侧支持多流与主流的 MoE 计算并行。
- **量化方案**：Linear 层权重量化为 **MXFP8（w8a8）**，MoE 的 GroupMatmul 量化为 **MXFP4（w4a8）**；激活为 MXFP8 动态量化，按 token、每 32 个元素一组共享一个 e8m0 scale。`norm`、`lm_head`、`embed_tokens`、MLA 的 `kv_b_proj` 保持 bf16，MoE 的 `router.classifier` 保持 fp32。
- **图执行**：decode 默认使用 **npugraph_ex**。
- **精度和性能**：
	- 在950DT上正常吐字，使用默认样例输出见 **“7. 拉起多卡推理”**
    - 8卡部署，bs32，4k输入序列下，8卡总吞吐1038.29TPS（参考值）。


---

## 硬件要求

| 项 | 要求 |
|----|------|
| 产品型号 | 昇腾 `950DT` 系列 |
| 操作系统 | Linux ARM |
| 部署规模 | 默认 `world_size: 8`（见 `config/bailing_2_5.yaml`），可按实际卡数调整 |
| 驱动 / 固件 | 按所在平台的标准要求安装；`npu-smi info` 确认固件与驱动已正确安装 |
| CANN | 安装到固定路径（如 `/usr/local/Ascend/cann`） |

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
pip3 install -r models/bailing_2_5/requirements.txt
```

### 3. 下载权重

下载 Ling（BailingMoeV2_5）的**原始 bf16 HuggingFace 权重**（[inclusionAI/Ling-2.5-1T](https://huggingface.co/inclusionAI/Ling-2.5-1T/tree/main)），上传到各节点的固定路径下，例如 `/data/models/bailing_2_5`。

> 后续的权重转换默认是**原地（in-place）** 进行的，会逐 shard 重写权重文件。如需保留原始 bf16 权重，请先自行备份，或在转换时用 `--output_hf_path` 指定新目录。

### 4. 下载数据集

从[链接](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/blob/main/longbook_qa_eng.jsonl)下载 `longbook_qa_eng.jsonl`，放到 `dataset/InfiniteBench/` 下：

```shell
mkdir -p dataset/InfiniteBench
```

> 目前支持的 `dataset` 取值：`default` / `LongBench` / `InfiniteBench`。

### 5. 生成 quantization_config

用 `utils/convert_config.py` 在权重目录的 `config.json` 中生成 / 规整 `quantization_config`。若在 NPU 上运行，需先 `source` CANN 环境 ：

```shell
# 样例路径仅供参考，需要根据实际路径配置
cann_path=/usr/local/Ascend/cann
source ${cann_path}/bin/setenv.bash

python models/bailing_2_5/utils/convert_config.py --config /data/models/bailing_2_5/config.json --full
```

> 该步骤只改写 `config.json`，不动权重。默认 `--output` 等于 `--config`（原地覆盖）。

### 6. 转换权重（bf16 → MXFP8/MXFP4）

用 `utils/convert_model.py` 把 bf16 权重量化为 MXFP8/MXFP4。**该步骤必须执行**，未转换的权重在加载时匹配不到参数会被丢弃。

| 入参 | 说明 |
|------|------|
| `--input_hf_path` | **必填**，bf16 HF 权重路径 |
| `--output_hf_path` | 输出路径；缺省则**原地**改写 `--input_hf_path` |
| `--shards` | 可选，只转换指定 shard |
| `--quant-impl` | 量化实现，`mxref`（默认）/ `npu-native` |
| `--device` | 量化设备，`npu` / `cpu` |

```shell
# 样例路径仅供参考，需要根据实际路径配置
# 原地转换（覆盖源权重）
python models/bailing_2_5/utils/convert_model.py --input_hf_path /data/models/bailing_2_5

# 或输出到新目录（保留源权重）
python models/bailing_2_5/utils/convert_model.py --input_hf_path /data/models/bailing_2_5 \
    --output_hf_path /data/models/bailing_2_5_mx
```

### 7. 拉起多卡推理

- 修改根目录公共的 `../../executor/scripts/set_env.sh`：配置 `IPs`（各节点 IP，按 rank 排序，空格分隔，第一个为主节点）与 `cann_path`。
- 修改 `config/bailing_2_5.yaml` 中的 `model_path` 为上一步转换后的权重路径，YAML 通用参数说明见 [YAML 参数描述](../../docs/common/inference_config_guide.md)，图模式支持`eager`和`npugraph_ex`模式，模型暂不支持MTP功能。

除框架统一配置外，本模型支持以下可选参数，配置在 `model_config.custom_params` 中：

| 参数名 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `enable_multi_streams` | bool | `True` | MoE 共享专家多流，decode 阶段使用侧流计算共享专家，与路由专家的分发/计算/聚合重叠 |

当前配置采用以下并行参数：

```yaml
parallel_config:
  world_size: 8
  attn_tp_size: 8
  o_proj_tp_size: 8
  moe_tp_size: 1
  dense_tp_size: 8
  embed_tp_size: 8
  lmhead_tp_size: 8
```

- `o_proj_tp_size` 必须等于 `attn_tp_size`（`check_parallel_settings` 会校验）。
- `moe_tp_size=1`，框架据此派生 `moe_ep_size=8`，路由专家按 Expert 维切分。

配置完成后，在各节点同步执行：

```shell
bash executor/scripts/infer.sh --model bailing_2_5 --yaml bailing_2_5.yaml
```
使用默认样例输出如下
 
 ```shell
It looks like your sentence was cut off. Based on what you’ve written, you’re describing the core idea of the **attention mechanism** (as introduced in "Attention Is All You Need").

Here’s the complete idea, filling in the likely intended ending:

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is **computed as a weighted sum of the values, where the weight assigned to each value is determined by the compatibility (or similarity) of the query with the corresponding key.**
    ...
 ```

