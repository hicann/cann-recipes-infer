# Kimi K3 模型在 NPU 上实现推理

## 概述

Kimi K3 总参数量约 2.8T，采用 Kimi Delta Attention（KDA）与 Gated MLA 交错排布的混合注意力 MoE 架构。本样例完成 Kimi K3 的 NPU 推理适配，并接入 cann-recipes-infer 统一推理框架，当前提供 Atlas A5（Ascend 950DT）4 机 32 卡推理配置。

模型架构、KDA、AttnRes、Stable LatentMoE、并行策略及量化方案详见 [Kimi K3 昇腾 NPU 推理优化实践](../../docs/models/kimi_k3/kimi_k3_inference_guide.md)。

模型主要结构如下：

- Decoder 主干共 93 层，首层使用 Dense FFN，其余 92 层使用 Stable LatentMoE。
- 混合注意力由 69 层 KDA 与 24 层 Gated MLA 组成。
- KDA 包含 96 个注意力头，头维度为 128，ShortConv 卷积核大小为 4。
- Gated MLA 的 Q LoRA rank 为 1536、KV LoRA rank 为 512，QK dimension 为 `128 + 64`，V dimension 为 128。
- AttnRes 以 12 个 Decoder Layer 为一个 block，最多使用 8 个常驻 anchor slots。
- Stable LatentMoE 包含 896 个 Routed Experts，每个 token 激活 16 个；Shared Expert 的等价容量为 2。
- Dense、Shared 与 Routed FFN 均使用 SiTU 激活函数，`beta=4`、`linear_beta=25`。
- Routed Expert 使用 group-32 MXFP4 权重与动态 MXFP8 激活，其余权重以 BF16 为主。

---

## 硬件要求

昇腾 950PR/DT 系列产品

---

## 快速启动

### 下载源码

在各个节点上执行如下命令下载源码：

```bash
mkdir -p /home/code
cd /home/code
git clone https://gitcode.com/cann/cann-recipes-infer.git
cd cann-recipes-infer
```

### 下载数据集

本样例默认使用 [`dataset/default_prompt.json`](../../dataset/default_prompt.json) 中的内置 prompt，对应 YAML 中的 `data_config.dataset: "default"`，无需额外下载数据集。输入长度由 `data_config.input_truncated_len` 控制。

### 准备权重

准备与当前配置匹配的 Kimi K3 checkpoint，并将权重放置到各节点的相同路径。权重目录需包含完整的模型配置、权重索引及 tokenizer 文件。

Routed Expert 的量化方式由 checkpoint `config.json` 中的 `quantization_config` 自动识别。当前实现支持 Kimi K3 checkpoint 使用的 `mxfp4-pack-quantized` 格式，无需在 YAML 中额外设置量化开关。

### 修改配置

修改 [`config/kimi_k3_rank_32_mxfp4_npugraph_ex.yaml`](config/kimi_k3_rank_32_mxfp4_npugraph_ex.yaml)，将 `model_config.model_path` 设置为实际权重路径。YAML 通用参数说明见 [YAML 参数描述](../../docs/common/inference_config_guide.md)。

当前仓内提供的配置如下：

| YAML 文件 | 权重 | `world_size` | 执行模式 |
| --- | --- | ---: | --- |
| `kimi_k3_rank_32_mxfp4_npugraph_ex.yaml` | BF16 + MXFP4 | 32 | Prefill `eager`，Decode `npugraph_ex` |

配置采用以下并行参数：

```yaml
parallel_config:
  world_size: 32
  attn_tp_size: 32
  moe_tp_size: 1
  embed_tp_size: 32
  lmhead_tp_size: 32
  dense_tp_size: 32
  o_proj_tp_size: 32
```

- `moe_tp_size=1`，框架据此派生 `moe_ep_size=32`，Routed Experts 按 Expert 维切分。
- `embed_tp_size` 与 `lmhead_tp_size` 按词表维切分 Embedding 和 LM Head。
- `dense_tp_size` 同时切分首层 Dense FFN 与 Shared Expert；`shared_tp_size` 保持默认值 `1`。
- `o_proj_tp_size` 必须与 `attn_tp_size` 一致；KDA 与 Gated MLA 的输出投影随 Attention TP 切分。
- Prefill 阶段按 token 执行模型内序列并行，Decode 阶段按 request 分片。

除框架统一配置外，Kimi K3 支持以下自定义参数，配置在 `model_config.custom_params` 中：

| 参数名 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `attn_res_mode` | str | `original` | AttnRes 后端：`original` 使用原实现，`two_phase` 使用 PyTorch 两阶段实现，`fused` 使用融合算子。当前 YAML 设置为 `original`。 |
| `enable_multi_streams` | bool | `True` | 启用多流并行。Decode 将 Shared Expert 放入独立流，与 Routed Expert 的分发、计算和聚合重叠，并在两路结果相加前同步；Prefill 保持主流执行。当前 YAML 设置为 `True`。 |
| `moe_chunk_max_len` | int | `65536` | Prefill MoE 路由缓冲的 gathered-token 上限；超过时管线内部循环分块以控制峰值内存。设为 `0` 或负数可禁用分块。 |

当前 YAML 的主要数据与调度参数如下：

```yaml
data_config:
  dataset: "default"
  input_truncated_len: 128

scheduler_config:
  batch_size: 32
  max_new_tokens: 128
```

### 拉起推理（多机）

以下命令适用于已完成多机通信配置的运行环境。请先进入模型目录，再在各节点执行相同的统一推理命令：

```bash
cd /home/code/cann-recipes-infer/models/kimi_k3
bash ../../executor/scripts/infer.sh --model kimi_k3 --mode offline \
    --yaml kimi_k3_rank_32_mxfp4_npugraph_ex.yaml
```

各 rank 的推理输出分别保存在 `models/kimi_k3/res/${DATE}/${CASE_NAME}/log_*.log`；默认启动模式下，rank 0 日志还会同步打印到终端。

---

## 已知限制

- 不支持将同一请求拆分为多次 Prefill。当前KDA 在新请求的 Prefill 阶段初始化 ShortConv state 与 recurrent state，调度层重复切分会覆盖前一分块状态。
- 不支持投机解码，`model_config.next_n` 必须为 `0`。
- 不支持 PD 分离与 Context Parallel。
