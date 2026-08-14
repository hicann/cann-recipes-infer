# Kimi K3 离线推理样例

## 概述

本目录提供 Kimi K3 在昇腾 NPU 上的固定批次离线推理实现，不接入统一执行器的 online/offline 调度流程。模型仍复用仓库的算子、量化和基础线性层，但以下运行态数据均由模型目录本地维护：

- `models/modules/attention_data.py` 分配 KDA state cache 与 MLA paged latent cache，并构造每个 Prefill/Decode step 的 attention metadata 字典。
- `models/model_infer.py` 管理 Prefill、普通逐 token Decode，以及 DSpark Proposal/Verify、Rejection Sampling 和跨轮状态。
- `runner_kimi_k3.py` 负责主模型、可选 DSpark 模型与 tokenizer 加载、共享模块以及两个 Decode 图的编译。
- `infer.py` 校验并派生离线配置，执行 warmup、清空本地 cache 后运行正式推理。

模型架构、KDA、AttnRes、Stable LatentMoE、并行策略及量化方案详见 [Kimi K3 昇腾 NPU 推理优化实践](../../docs/models/kimi_k3/kimi_k3_inference_guide.md)。

## 硬件要求

昇腾 950PR/DT 系列产品。

## 环境准备

1. 安装 CANN 软件包。

   本样例的编译执行依赖 CANN 开发套件包（toolkit）与昇腾 950 对应的二进制算子包（ops）。支持9.2.0版本cann包，下载选择[weekly版本](https://www.hiascend.com/cann/download)， 并参考链接中的指导方式完成安装。

   Kimi K3 使用 KDA、MLA 和 MoE 相关昇腾算子，建议所有节点使用相同的 CANN、驱动和固件版本。当前实现面向昇腾 950PR/DT，不支持在其他产品上直接运行。

2. 安装 Ascend Extension for PyTorch（torch_npu）。

   `torch_npu` 为 PyTorch 在 NPU 上运行提供适配。请安装与当前 CANN 和 Python 版本匹配的 `torch_npu`、PyTorch 及其依赖；本目录的依赖版本记录在 `models/kimi_k3/requirements.txt` 中。安装[torch_npu](https://pypi.org/project/torch-npu/2.10.0.post4/)时，按照链接中的指导方法完成安装，建议安装对应python版本为3.12。或者通过命令行完成安装：
    ```bash
   # 推荐2.10.0版本
   pip install torch==2.10.0
   pip install torch_npu==2.10.0.post4
   ```

3. 安装cannbot-dsl算子依赖包

    cannbot-dsl是cann的高性能融合算子包，执行kimi-k3需要安装cannbot-dsl依赖。
    ```bash
   pip install cannbot-dsl
   ```

3. 下载项目源码并安装 Python 依赖。

   ```bash
   # 下载项目源码
   git clone https://gitcode.com/cann/cann-recipes-infer.git
   cd cann-recipes-infer

   # Kimi K3 依赖，仅支持项目 requirements.txt 声明的 Python/依赖版本
   pip3 install -r ./models/kimi_k3/requirements.txt
   ```

4. 配置样例运行环境。

   修改 `models/kimi_k3/set_env.sh` 中的如下字段：

   - `IPs`：配置所有节点的 IP，按 rank id 排序，多个节点的 IP 以空格分隔，例如：`('xxx.xxx.xxx.xxx' 'xxx.xxx.xxx.xxx')`。
   - `cann_path`：CANN 软件包安装路径，例如 `/home/code/Ascend/cann/`。



## 快速启动

以下步骤适用于已完成上述环境准备的昇腾 950PR/DT 多卡离线推理场景。

### 准备权重

从 [ModelScope Kimi-K3 模型页面](https://www.modelscope.cn/models/moonshotai/Kimi-K3/files) 下载模型权重，并将完整 checkpoint 上传到各节点可访问的相同路径。在配置文件中填写该路径：

```yaml
# models/kimi_k3/config/kimi_k3_rank_32_mxfp4_npugraph_ex.yaml or models/kimi_k3/config/kimi_k3_rank_32_mxfp4_npugraph_ex_dspark.yaml (example)
model_path: "/data/models/kimi_k3"
```

当前实现直接加载 Kimi K3 原生 MXFP4 权重，不提供本目录内的权重转换脚本。

### 下载数据集
  从[链接](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/blob/main/longbook_qa_eng.jsonl)中下载长序列输入数据集longbook_qa_eng，并上传到各个节点上新建的路径`dataset/InfiniteBench`下。
  ```shell
  mkdir -p dataset/InfiniteBench
  ```

### 修改配置

默认配置文件为 `config/kimi_k3_rank_32_mxfp4_npugraph_ex.yaml`，当前配置使用 32 卡 Attention TP 与 Expert Parallel：

```yaml
model_name: "kimi_k3"
model_path: "/data/models/kimi_k3"
exe_mode: "npugraph_ex"
world_size: 32

model_config:
  with_ckpt: True
  enable_online_split_weight: True
  enable_profiler: False
  enable_static_kernel: True
  enable_cache_compile: False
  enable_weight_nz: False
  platform_version: "950"
  draft_model_type: "none"
  next_n: 0
  skip_warm_up: True
  prefill_mini_batch_size: 1
  pa_block_size: 128
  custom_params:
    attn_res_mode: "fused"
    enable_multi_streams: True
    enable_flash_kda: True
    enable_fused_recurrent_kda: True
    moe_chunk_max_len: 12800
    enable_mega_moe: True
    enable_moe_bf16_mode: True

data_config:
  dataset: "default"
  input_max_len: 128
  max_new_tokens: 128
  batch_size: 32
  temperature: 1.0

parallel_config:
  attn_tp_size: 32
  moe_tp_size: 1
  embed_tp_size: 16
  lmhead_tp_size: 8
  dense_tp_size: 8
  oproj_tp_size: 32
  cp_size: 1
```

离线实现要求：

- `batch_size` 固定，且必须能被 `attn_tp_size` 整除。
- `attn_tp_size` 为 `1` 或覆盖完整 `world_size`；当前配置使用完整 32 卡 TP。
- `moe_tp_size=1`，由本地入口派生 `moe_ep_size=world_size`。
- `oproj_tp_size=attn_tp_size`、`cp_size=1`。普通路径使用 `draft_model_type=none,next_n=0`。
- `prefill_mini_batch_size=0` 表示整个固定 batch 一次 Prefill；大于 0 时必须能整除 `batch_size_per_rank`。默认配置为 `1`，即 32 个 Prefill mini cycle。
- `skip_warm_up=True` 可跳过启动时的完整 Prefill/Decode warm-up，默认开启。开启后首次正式 Decode 会触发图编译，并且打印的首次推理耗时包含编译和 NPU 算子冷启动开销；设置为 `False` 可恢复完整 warm-up。
- `pa_block_size` 必须是 16 的倍数，以满足 MLA NZ cache 布局。
- Prefill 以 eager 执行；当 `exe_mode` 为 `npugraph_ex` 或 `ge_graph` 时，只编译固定形状 Decode。
- 输入先按 `input_max_len` 为 chat template 预留空间并截断正文，再调用 checkpoint tokenizer 的 `apply_chat_template`；最终 Prefill 长度不会超过 `input_max_len`。
- 请求输出使用模型 config、`generation_config.json` 和 tokenizer 三处 EOS 的并集进行截止，兼容 Kimi K3 的 `<|end_of_msg|>` 与 `[EOS]`。
- `enable_profiler=True` 时，Prefill 采集中间的一个 mini cycle；Decode 跳过前 10 个 step 后采集 10 个 step。两阶段分别写入 `prof/prefill` 和 `prof/decode`，且不依赖 warm-up 是否启用。

每个 Prefill mini cycle 都由完整 Attention TP 通信组参与：prompt token 在 `attn_tp_size` 个 rank 间做 Sequence Parallel，Attention 在 head 维做 TP。KDA state 根据完整 Decode batch 中的 request index 直接写入对应 state row；MLA latent 只由该请求的 Decode owner rank 写入其本地 paged blocks。所有 cycle 共用同一组预分配 cache tensor，完成后直接以完整 batch 切换到 request-DP + Attention TP Decode，不发生 cache 合并、拷贝或重新分配。

参数开关说明：

| 参数名 | 类型 | YAML 值 | 含义 |
| --- | --- | --- | --- |
| `exe_mode` | str | `npugraph_ex` | 执行模式，可选 `eager`、`npugraph_ex`。 |
| `enable_online_split_weight` | bool | `True` | 启动时按 rank 从完整 checkpoint 切分加载权重。 |
| `enable_profiler` | bool | `False` | 是否采集 Prefill/Decode 性能数据。 |
| `enable_static_kernel` | bool | `True` | 启用静态 kernel 优化路径。 |
| `enable_cache_compile` | bool | `False` | 是否启用算子编译缓存。 |
| `enable_weight_nz` | bool | `False` | 是否按 NZ 格式加载权重，需匹配权重格式和平台。 |
| `draft_model_type` | str | `none` | Dspark开关；`none` 为普通 Decode，`dspark` 启用 DSpark 投机推理。 |
| `next_n` | int | `0` | DSpark 投机推理 token 数；`none` 时必须为 `0`。 |
| `skip_warm_up` | bool | `True` | 是否跳过 warm-up；开启后首次正式推理包含图编译开销。 |
| `attn_res_mode` | str | `fused` | AttnRes 后端，可选 `original`、`two_phase`、`fused`。 |
| `enable_multi_streams` | bool | `True` | Decode 使用多流重叠 Shared Expert 与 Routed Expert。 |
| `enable_flash_kda` | bool | `True` | 是否启用 Prefill `flash_kda` 融合算子，关闭为python小算子实现。 |
| `enable_fused_recurrent_kda` | bool | `True` | 是否启用 Decode `fused_recurrent_kda`，关闭为GDR融合算子实现。 |
| `enable_mega_moe` | bool | `True` | 是否启用 MegaMoE 融合算子。 |
| `enable_moe_bf16_mode` | bool | `True` | MoE 路由后处理和 SiTU 是否使用 BF16。 |

`enable_online_split_weight` 仅表示启动时从完整 checkpoint 按 rank 加载权重，不代表支持在线请求调度。

### 拉起多卡推理

拉起推理在每个节点执行：

进入模型目录执行封装脚本：

```shell
cd /home/code/cann-recipes-infer/models/kimi_k3
bash infer.sh
```

启动脚本需要修改`YAML_FILE_NAME`参数，选取`models/kimi_k3/config`目录下的yaml文件，修改示例`export YAML_FILE_NAME=kimi_k3_rank_32_mxfp4_npugraph_ex.yaml`。默认 prompt 来自仓库 `dataset/default_prompt.json`，如果序列长度超过128，必须更换其他数据集。如需验证其他数据集，需要修改yaml文件中`dataset`字段参数，以InfiniteBench为例，下载好的数据集文件在`/data/InfiniteBench`目录下，将yaml文件中的`dataset`字段修改为`"InfiniteBench"`。可以在yaml文件中设置warm-up开关选择是否跳过warm-up阶段；设置 `skip_warm_up=False` 后，会先用同一固定 batch 执行 warm-up，再原地清空本地 cache，最后执行正式推理。

启用 DSpark 时，从 [ModelScope Kimi-K3-DSpark 模型页面](https://www.modelscope.cn/models/skyai/sglang-Kimi-K3-DSpark/files) 下载 DSpark 模型权重，修改 `config/kimi_k3_rank_32_mxfp4_npugraph_ex_dspark.yaml` 中主模型的 `model_path` 和 DSpark 模型的 `draft_model_path`，并在每个节点执行：

```shell
cd /home/code/cann-recipes-infer/models/kimi_k3
YAML_FILE_NAME=kimi_k3_rank_32_mxfp4_npugraph_ex_dspark.yaml（修改models/kimi_k3/infer.sh）
bash infer.sh
```

32P 示例将 KDA Attention 和 MLA `g_proj/o_proj` 保持 TP32，同时使用 Dense/LMHead/DSpark TP8 和 Embed TP16；Decode target hidden、DSpark 层间 hidden 与 LMHead logits 均保持 request-owner 本地化。`attn_res_mode` 默认保持 `original`，也支持 `two_phase` 和 `fused`。主模型和草稿模型的 cache 独立分配，warmup 后两组 cache 均原地清零。正式推理结束后会打印平均 acceptance length、acceptance length 分布，以及每个 draft position 的 acceptance rate。


## 已知限制

- 仅支持固定批次离线生成；Prefill mini batch 是静态多 cycle，不支持在线请求加入、退出或动态调度。
- 不支持 chunked Prefill、MTP、PD 分离和 Context Parallel；投机推理仅支持显式配置的 DSpark。
- 不创建或依赖统一执行器的 `CacheInfo`、`ForwardMetaData` 与 scheduler step 状态。
