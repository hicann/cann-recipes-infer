# GLM-5.3-Flash Inference on NPU

[中文](./README.md) | **English**

## Overview

GLM-5.3-Flash is a **hybrid attention + mHC + MoE** large model released by Zhipu AI. This recipe adapts the GLM-5.3-Flash weights (~306B total parameters, released in FP8 quantized format), performs quantization and performance optimization on the CANN platform, and targets **Hybrid HiF8-MXFP8-MXFP4** inference deployment on the Ascend `950DT` platform.

- **Hybrid attention**: among the 45 layers, 34 are **KDA** layers (Kimi Delta Attention, a linear attention built on short convolution and gated delta recurrence); one **MLA + DSA** layer is inserted every 4 layers (11 layers in total: absorption-style MLA using NoPE throughout, combined with sparse attention driven by a k-pool compressed indexer).
- **mHC** (Manifold-Constrained Hyper-Connections): the hidden state between layers consists of 4 parallel streams `[T, 4, D]`, folded by a Sinkhorn-normalized combination matrix before each sub-layer and unfolded again afterwards.
- **MoE**: 288 routed experts with top-8 routing plus 1 shared expert. The first `first_k_dense_replace` layers are dense MLPs, and the shared-expert branch supports multi-stream execution in parallel with routed-expert computation.
- **Quantization scheme**: **Hybrid HiF8-MXFP8-MXFP4**. The DSA projections and dense MLPs use **HiF8 W8A8**, the shared expert uses **MXFP8 W8A8**, and the routed experts use **MXFP4 W4A8**. All KDA weights, the DSA `kv_b_proj`, the entire indexer, the mHC parameters, all norms, `mlp.gate`, and embed/lm_head remain in BF16/FP32.
- **Graph mode**: decode uses **npugraph_ex** by default.

---

## Hardware Requirements

| Item | Requirement |
|----|------|
| Product model | Ascend `950DT` series |
| Operating system | Linux ARM |
| Deployment scale | `world_size: 8` by default (see `config/glm_5_3_flash_ep8.yaml`); adjust it to the actual number of cards. MoE EP = world_size |
| Driver / firmware | Install according to the standard requirements of your platform; run `npu-smi info` to confirm that the firmware and driver are installed correctly |
| CANN | Install to a fixed path (for example `/usr/local/Ascend/cann`) |
| Custom operator packages | `opp/vendors/customize` (mHC AscendC operators) and `opp/vendors/custom_transformer` (causal_conv1d); see "Quick Start · 2" |

---

## Quick Start

### 1. Download the source code

```shell
mkdir -p /home/code && cd /home/code
git clone https://gitcode.com/cann/cann-recipes-infer.git
cd cann-recipes-infer
```

### 2. Install dependencies

```shell
pip3 install -r models/glm_5_3/requirements.txt
```

This model depends on two operator packages:

| Package | Operators provided |
|---|---|
| `custom_ops` | `npu_hc_pre` / `npu_hc_post` |
| `cann_ops_transformer` | `causal_conv1d_fn` / `causal_conv1d_update` |


Source the corresponding vendor environment before launching:

```shell
source models/glm_5_3/set_env.sh
```

### 3. Download the weights

Download the [original GLM-5.3-Flash weights](https://huggingface.co/zai-org/GLM-5.3-Flash) (FP8) to a fixed path on every node, for example `/data/models/GLM-5.3-Flash`.

### 4. Download the dataset

The default configuration is `dataset: "InfiniteBench"`. Download `longbook_qa_eng.jsonl` from [this link](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/blob/main/longbook_qa_eng.jsonl) and place it under `dataset/InfiniteBench/`.

> Currently supported values of `dataset`: `default` / `LongBench` / `InfiniteBench`.

### 5. Convert the weights (FP8 → HiF8)

Use `utils/convert_model.py` to convert the released FP8 weights into a format executable at runtime. The script also rewrites `quantization_config` in `config.json`.

| Argument | Default | Description |
|------|------|------|
| `--input_hf_path` | — | **Required**, path to the FP8 HF weights |
| `--output_hf_path` | — | **Required**, output path (in-place conversion is not supported) |

```shell
# Hybrid HiF8-MXFP8-MXFP4
python models/glm_5_3/utils/convert_model.py \
    --input_hf_path /data/models/GLM-5.3-Flash \
    --output_hf_path /data/models/GLM-5.3-Flash-HIF8
```

### 6. Launch multi-card inference

- Edit the shared `../../executor/scripts/set_env.sh` in the repository root: configure `IPs` (the IP of each node, ordered by rank and separated by spaces, the first one being the master node) and `cann_path`.
- Set `model_path` in `config/glm_5_3_flash_ep8.yaml` to the converted weight path produced in the previous step. For the general YAML parameters, see the [YAML parameter description](../../docs/common/inference_config_guide.md). Graph mode supports `eager` and `npugraph_ex`; this model does not support the MTP feature yet.

Besides the framework-wide configuration, this model supports the following optional parameter, configured under `model_config.custom_params`:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `enable_multi_streams` | bool | `True` | Shared-expert multi-stream. During decode the shared expert is computed on a side stream, overlapping with the dispatch / GMM / combine of the routed experts |

The current configuration uses the following parallel parameters:

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

- With `moe_tp_size=1`, the framework derives `moe_ep_size = world_size`, and the routed experts are partitioned along the expert dimension.
- `cp_size`, `attn_tp_size`, `dense_tp_size`, `moe_tp_size`, `o_proj_tp_size` and `shared_tp_size` must all be 1.

Once the configuration is complete, run the following command on every node at the same time:

```shell
bash executor/scripts/infer.sh --model glm_5_3 --yaml glm_5_3_flash_ep8.yaml
```

The throughput at a sequence length of 8K under different batch sizes is as follows:

| BS per card | Decode latency per step (ms) | TPS per card | Total TPS (8 cards) |
|--------|-------------------|---------|---------|
| 1 | 16.60 | 60.24 | 481.93 |
| 16 | 24.07 | 664.73 | 5317.82 |
| 64 | 35.69 | 1793.22 | 14345.76 |
| 128 | 49.88 | 2566.16 | 20529.27 |
