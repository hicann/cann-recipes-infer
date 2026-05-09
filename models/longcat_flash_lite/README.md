# LongCat-Flash-Lite模型在NPU上推理

> **注**: 本模型初版适配优化由 NPU 推理优化 Agent Skills 自动完成。

## 概述
LongCat-Flash-Lite是基于MLA + Sparse MoE + N-gram Embedding架构的稀疏专家混合大语言模型。本样例基于LongCat-Flash-Lite开源代码进行迁移，并完成对应的NPU优化适配，包括 Paged Attention 缓存管理、融合算子替换、GE 图模式加速、专家并行（EP）路径与多 batch 支持。

## 支持的产品型号
<term>Atlas A2 系列产品</term>
<term>Atlas A3 系列产品</term>

## 环境准备

1. 安装CANN软件包。

   本样例的编译执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels）。

   请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann)下载对应版本的软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_0093.html?Mode=PmIns&InstallType=netconda&OS=openEuler&Software=cannToolKit)进行安装。

2. 安装Ascend Extension for PyTorch（torch_npu）。

   Ascend Extension for PyTorch（torch_npu）为支撑PyTorch框架运行在NPU上的适配插件。
   请从[软件包下载地址](https://gitcode.com/Ascend/pytorch/releases)下载对应版本的安装包，参考[torch_npu安装文档](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0004.html)进行安装。

3. 下载项目源码并安装依赖的python库。
    ```bash
    # 下载项目源码，以master分支为例
    git clone https://gitcode.com/cann/cann-recipes-infer.git

    # 安装依赖的python库，仅支持python 3.11
    cd cann-recipes-infer
    pip3 install -r ./models/longcat_flash_lite/requirements.txt
    ```

4. 配置样例运行所需环境信息。

   修改`executor/scripts/set_env.sh`中的如下字段:
   - `IPs`：配置所有节点的IP，按照rank id排序，多个节点的ip通过空格分开。
   - `cann_path`: CANN软件包安装路径，例如`/usr/local/Ascend/ascend-toolkit/latest`。

## 权重准备

请自行下载LongCat-Flash-Lite原始权重到本地路径，例如`/path/to/LongCat-Flash-Lite`。

## 数据集

样例默认使用仓内自带的 `dataset/default_prompt.json`（~256 token 的关于 attention 的短句），无需额外下载，适用于短输入场景（input_len≤1024）的 1k yaml。

长输入场景（如 `longcat_flash_lite_8tp_4k1k.yaml` 的 4K input）请将 `dataset/default_prompt.json` 的内容替换为约 4K token 的长 prompt 后再运行。也可将 yaml 的 `data_config.dataset` 改为 `"LongBench"` 或 `"InfiniteBench"` 使用公开 benchmark（需自行下载到本地或通过 HF 拉取）。

## 推理执行

1. 修改YAML文件中`model_path`参数。

   在`models/longcat_flash_lite/config`目录下已提供YAML配置供参考：
   - `longcat_flash_lite_8tp.yaml`：8 卡 TP 推理（ge_graph 模式，BS=1 低延迟）
   - `longcat_flash_lite_8tp_4k1k.yaml`：8 卡 TP，4K 输入 / 1K 输出
   - `longcat_flash_lite_ep8.yaml`：8 卡 EP 推理（ge_graph + MC2，BS=2 平衡）
   - `longcat_flash_lite_ep8_b8.yaml`：8 卡 EP 推理（BS=8 高吞吐）
   - `longcat_flash_lite_1card.yaml`：单卡 eager 基线

   将`model_path`设置为权重文件存储路径。

2. 执行推理（`infer.sh` 接受 `$1=mode`、`$2=yaml`，与仓库其他模型一致）。
    ```bash
    cd models/longcat_flash_lite
    bash infer.sh                                              # 默认 8卡TP (longcat_flash_lite_8tp.yaml)
    bash infer.sh offline longcat_flash_lite_8tp_4k1k.yaml     # 8卡TP, 4K input + 1K output
    bash infer.sh offline longcat_flash_lite_ep8.yaml          # 8卡EP, BS=2
    bash infer.sh offline longcat_flash_lite_ep8_b8.yaml       # 8卡EP, BS=8 高吞吐
    bash infer.sh offline longcat_flash_lite_1card.yaml        # 单卡基线
    ```

## 优化点参考

详见 [optimization_report.md](optimization_report.md)，主要优化包括：
1. Paged Attention 内存管理：block-paged KV cache（block_size=128）；cache 内容形态由 MLA 架构决定（每 token 仅存 `kv_lora_rank + qk_rope_head_dim = 576` 维 latent KV，而非 vanilla MHA 的 ~12288 维）
2. MLA absorb decode 路径：将 `q_b_proj` 投影吸收到 K 方向，避免 decode 时显式还原大维度 K
3. 融合算子全量替换：`rms_norm` / `add_rms_norm` / `moe_gating_top_k` / `init_routing_v2` / `grouped_matmul + swiglu`
4. GE 图模式 Decode 整图，含 NgramEmbedding 入图（消除 ~11 ms host dispatch 开销）
5. A3 启用 `npu_mla_prolog_v3` 融合（q_b_proj → split → RoPE 与 KV cache 写入合一）
6. EP MoE 路径（Prefill AllToAll dispatch/combine，Decode MC2 `dispatch_v2/combine_v2`）与多 batch 支持

## 性能基线

A3 8卡实测最佳性能（input_len=1024）：

| 配置 | BS | Prefill | Decode/token | 吞吐 |
|---|---|---|---|---|
| TP8（低延迟） | 1 | 50.68 ms | 5.94 ms | 167 tok/s |
| EP8（高吞吐） | 8 | 129.38 ms | 9.54 ms | 838 tok/s |

> 详细性能拆解（含逐阶段贡献、A2 / A3 跨硬件对比）参见 [optimization_report.md](optimization_report.md) 与 [baselines/baseline_tp8.md](baselines/baseline_tp8.md)。

## 模型结构

| 参数 | 值 |
|------|-----|
| 架构 | MoE LLM (MLA + Sparse MoE + N-gram Embedding) |
| hidden_size | 3072 |
| num_layers | 14 (dual sub-layer) |
| num_attention_heads | 32 |
| n_routed_experts | 256 |
| zero_expert_num | 128 (identity) |
| moe_topk | 12 |
| vocab_size | 131072 |
| dtype | bfloat16 |

## 并行配置

| 参数 | 值 |
|------|-----|
| world_size | 8 |
| attn_tp_size | 8 |
| moe_tp_size | 8 |
| embed_tp_size | 8 |
| lmhead_tp_size | 8 |
