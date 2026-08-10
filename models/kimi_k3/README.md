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

## 配置

修改 `config/kimi_k3_rank_32_mxfp4_npugraph_ex.yaml` 中顶层的 `model_path`，使其指向包含模型配置、权重索引、权重和 tokenizer 文件的 Kimi K3 checkpoint。

当前配置使用 32 卡 Attention TP 与 Expert Parallel：

```yaml
world_size: 32
exe_mode: "npugraph_ex"

model_config:
  prefill_mini_batch_size: 1

data_config:
  input_max_len: 128
  max_new_tokens: 128
  batch_size: 32

parallel_config:
  attn_tp_size: 32
  moe_tp_size: 1
  embed_tp_size: 32
  lmhead_tp_size: 32
  dense_tp_size: 32
  oproj_tp_size: 32
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
| 参数名 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `attn_res_mode` | str | `original` | AttnRes 后端：`original` 使用原实现，`two_phase` 使用 PyTorch 两阶段实现，`fused` 使用融合算子。当前 YAML 设置为 `original`。 |
| `enable_multi_streams` | bool | `True` | 启用多流并行。Decode 将 Shared Expert 放入独立流，与 Routed Expert 的分发、计算和聚合重叠，并在两路结果相加前同步；Prefill 保持主流执行。当前 YAML 设置为 `True`。 |
| `moe_chunk_max_len` | int | `65536` | Prefill MoE 路由缓冲的 gathered-token 上限；超过时管线内部循环分块以控制峰值内存。设为 `0` 或负数可禁用分块。 |
| `enable_mega_moe` | bool | `False` | Prefill MoE使能MegaMoe融合算子，开启后，chunkmoe功能将关闭，moe_chunk_max_len不生效。 |
| `enable_moe_bf16_mode` | bool | `True` | 开启后，MoE中的finalise_routing与SiTU将使用BF16精度进行运算，但MoeGating将总是保持FP32的计算精度。 |

`enable_online_split_weight` 仅表示启动时从完整 checkpoint 按 rank 加载权重，不代表支持在线请求调度。

## 启动

按仓库要求配置各节点的通信信息后，在每个节点进入模型目录执行：

```bash
cd /home/code/cann-recipes-infer/models/kimi_k3
bash infer.sh
```

启动脚本默认读取 `config/kimi_k3_rank_32_mxfp4_npugraph_ex.yaml`。默认 prompt 来自仓库 `dataset/default_prompt.json`。程序默认跳过 warm-up 并直接执行正式推理；设置 `model_config.skip_warm_up=False` 后，会先用同一固定 batch 执行 warm-up，再原地清空本地 cache，最后执行正式推理。

启用 DSpark 时，修改 `config/kimi_k3_rank_32_mxfp4_npugraph_ex_dspark.yaml` 的 `model_path` 和 `draft_model_path`，并执行：

```bash
YAML_FILE_NAME=kimi_k3_rank_32_mxfp4_npugraph_ex_dspark.yaml bash infer.sh
```

DSpark 配置要求 `draft_model_type=dspark`、`next_n` 与草稿 checkpoint 的 `block_size` 一致、`dense_tp_size=attn_tp_size`，并使用 16 或 128 的 `pa_block_size`。`attn_res_mode` 默认保持 `original`，也支持 `two_phase` 和 `fused`。主模型和草稿模型的 cache 独立分配，Decode target hidden 保持 request-owner 本地化，warmup 后两组 cache 均原地清零。正式推理结束后会打印平均 acceptance length、acceptance length 分布，以及每个 draft position 的 acceptance rate。

## 已知限制

- 仅支持固定批次离线生成；Prefill mini batch 是静态多 cycle，不支持在线请求加入、退出或动态调度。
- 不支持 chunked Prefill、MTP、PD 分离和 Context Parallel；投机推理仅支持显式配置的 DSpark。
- 不创建或依赖统一执行器的 `CacheInfo`、`ForwardMetaData` 与 scheduler step 状态。
