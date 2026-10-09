# Kimi K3 模型在 NPU 上的推理实现

模型架构、并行策略、量化方案与性能优化详见技术报告：[Kimi K3 昇腾 NPU 推理优化实践](../../docs/models/kimi_k3/kimi_k3_inference_guide.md)。

## 概述

本目录提供 Kimi K3 在昇腾 NPU 上的推理实现与部署配置，支持分块 Prefill、DSpark 投机推理，以及 eager 和 npugraph_ex Decode。

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

3. 安装 CANNBot-DSL 算子依赖包

    CANNBot-DSL 提供 Kimi K3 所需的高性能融合算子，算子编译依赖 `ninja==1.13.0`，该版本已列入 `requirements.txt`。
    ```bash
   pip install cannbot-dsl
   ```

4. 下载项目源码并安装 Python 依赖。

   ```bash
   # 下载项目源码
   git clone https://gitcode.com/cann/cann-recipes-infer.git
   cd cann-recipes-infer

   # Kimi K3 依赖，仅支持项目 requirements.txt 声明的 Python/依赖版本
   pip3 install -r ./models/kimi_k3/requirements.txt
   ```

5. 安装 SiTU 自定义算子（`situ_and_mul_sparse` 与融合算子 `grouped_situ_mx_quant`）。

   MoE 量化专家路径的 SiTU 激活 + MXFP8 动态量化使用 `grouped_situ_mx_quant` 融合算子，与 `situ_and_mul_sparse` 需要一并安装。注意自定义算子包的 `op_api` 层是整体替换，**两个算子必须打进同一个包安装**，分开安装会互相覆盖导致先装的算子失效：

   ```bash
   # 初始化 CANN 环境变量
   source /usr/local/Ascend/cann/set_env.sh

   # 一次编译 SituAndMulSparse + GroupedSituMxQuant 两个算子（分号分隔传给 -n）
   cd ops/ascendc
   bash build.sh -n "situ_and_mul_sparse;grouped_situ_mx_quant" -c "ascend950"
   cd output
   ./CANN-custom_ops-*.run --quiet --install-path="${ASCEND_HOME_PATH}/opp"
   source "${ASCEND_HOME_PATH}/opp/vendors/customize/bin/set_env.bash"

   # 编译并安装 Torch 扩展
   cd ../torch_ops_extension
   bash build_and_install.sh

   # （可选）分别验证两个 SiTU 自定义算子相对原生 PyTorch golden 的精度
   cd ../examples
   python3 test_situ_custom_ops.py
   ```

6. 配置样例运行环境。

   修改 `models/kimi_k3/set_env.sh` 中的如下字段：

   - `IPs`：配置所有节点的 IP，按 rank id 排序，多个节点的 IP 以空格分隔，例如：`('xxx.xxx.xxx.xxx' 'xxx.xxx.xxx.xxx')`。
   - `cann_path`：CANN 软件包安装路径，例如 `/home/code/Ascend/cann/`。



## 快速启动

以下步骤适用于已完成上述环境准备的昇腾 950PR/DT 多卡推理场景。

### 准备权重

从 [ModelScope Kimi-K3 模型页面](https://www.modelscope.cn/models/moonshotai/Kimi-K3/files) 下载模型权重，并将完整 checkpoint 上传到各节点可访问的相同路径。在配置文件中填写该路径：

```yaml
# models/kimi_k3/config/kimi_k3_rank_32P_100k_1batch.yaml
model_path: "/data/models/kimi_k3"
```

默认 GQA 草稿模型可从 [RadixArk/Kimi-K3-DSpark](https://huggingface.co/RadixArk/Kimi-K3-DSpark) 获取。请使用与主模型版本匹配的 DSpark 权重，并在 YAML 中填写 `draft_model_path`。

### 转换权重

Kimi-K3 权重转换会同步生成转换后的 safetensors 权重、权重索引和 `config.json`，无需单独转换 `config.json`。请使用包含 `kimi_k3` tensorwise deploy 适配的 AMCT 代码，并在已安装 AMCT 依赖的环境中，从 AMCT 仓库根目录执行以下命令。

```shell
git clone https://gitcode.com/cann/amct.git
cd amct
```

如果权重转换的运行环境为NPU，需要先执行：

```shell
cann_path=/usr/local/Ascend/cann  # cann包安装路径
source ${cann_path}/bin/setenv.bash
```

Kimi-K3 采用的量化方案需独立创建config，请在 `amct_pytorch/configs/` 路径下创建 `w4a8_kimi_k3.yaml`，并在文件中输入：

```yaml
w_bits: 8
a_bits: 8

attn-linear:
   w_bits: 8
   a_bits: 8
   q_a_proj: { w_bits: 8, a_bits: 8 }
   q_b_proj: { w_bits: 8, a_bits: 8 }
   kv_a_proj_with_mqa: { w_bits: 8, a_bits: 8 }
   o_proj: { w_bits: 8, a_bits: 8 }

moe:
   routed:
      w_bits: 4
      a_bits: 4
      routed_expert_down_proj:
         w_bits: 8
         a_bits: 8
      routed_expert_up_proj:
         w_bits: 8
         a_bits: 8
   shared:
      w_bits: 8
      a_bits: 8
```

>入参介绍：`model`：原始权重路径；`model_name`：AMCT 内部模型适配器名称，Kimi-K3 使用 `kimi_k3`；`device`：权重转换使用的 NPU 设备；`granularity`：转换粒度，Kimi-K3 tensorwise 权重转换使用 `tensor`；`quant_target`：量化目标模块；`quant_dtype`：量化数据类型；`bit_config`：量化位宽配置文件；`output_dir`：转换后输出的权重路径。

权重转换拉起示例：

```shell
python3 amct_pytorch/cli/llm/deploy.py \
   --model /data/models/kimi_k3 \
   --model_name kimi_k3 \
   --device npu:0 \
   --granularity tensor \
   --quant_target moe attn-linear \
   --quant_dtype mxfp \
   --bit_config amct_pytorch/configs/w4a8_kimi_k3.yaml \
   --output_dir /data/models/Kimi-K3-MXFP
```

转换完成后，将推理配置中的 `model_path` 设置为转换后的目录，例如 `/data/models/Kimi-K3-MXFP`。

### 修改配置

`infer.sh` 默认使用 [config/kimi_k3_rank_32P_100k_1batch.yaml](config/kimi_k3_rank_32P_100k_1batch.yaml)。该配置使用 32 卡、Attention TP16/DP2、global batch 32、102400-token 输入上限，启用 DSpark GQA 和 MegaKDA ReplaySSM。注意 `data_config.batch_size=32`，`model_config.prefill_mini_batch_size=1` 表示每个 Attention DP group 每个 Prefill mini cycle 处理 1 个请求。以下内容与该 YAML 一致；模型路径需按部署环境填写：

```yaml
model_name: "kimi_k3"
model_path: "/data/models/kimi_k3"
draft_model_path: "/data/models/kimi_k3_dspark"
exe_mode: "npugraph_ex"
world_size: 32

model_config:
  with_ckpt: True
  enable_online_split_weight: True
  enable_profiler: True
  enable_static_kernel: True
  enable_cache_compile: True
  force_eplb: False
  platform_version: "950"
  draft_model_type: "dspark_gqa"  # Use "dspark_mla" for the vLLM MLA checkpoint.
  dspark_tp_size: 8
  next_n: 7
  skip_warm_up: True
  # Per Attention DP group: 16 requests/group x DP2 = global batch 32.
  prefill_mini_batch_size: 1
  prefill_chunk_size: 8192
  pa_block_size: 128
  custom_params:
    enable_multi_streams: True
    enable_superkernel: True  # Requires npugraph_ex, static kernels and multi-streams.
    enable_dspark_confidence_head: False
    enable_mega_kda: True
    enable_mega_kda_replayssm: True
    moe_chunk_max_len: 12800
    enable_prefill_mega_moe: True

data_config:
  dataset: "default"
  input_max_len: 102400
  max_new_tokens: 256
  # TP16/DP2 assigns sixteen requests to each Attention group (one request per rank).
  batch_size: 32
  temperature: 0.0

parallel_config:
  # Attention/Embedding/OProj use TP16; Dense/LMHead/DSpark stay at TP8 to reduce collectives.
  attn_tp_size: 16
  moe_tp_size: 1
  embed_tp_size: 16
  lmhead_tp_size: 8
  dense_tp_size: 8
  oproj_tp_size: 16
  cp_size: 1
```

当前实现要求：

- `batch_size` 固定；每个 Attention DP group 分到的请求数必须能被 `attn_tp_size` 整除。
- `world_size` 和模型的 KDA/MLA head 数必须能被 `attn_tp_size` 整除。
- `dense_tp_size`、`embed_tp_size`、`lmhead_tp_size` 和 `oproj_tp_size` 必须整除 `attn_tp_size`，避免通信组跨越不同请求的 Attention DP group。
- `moe_tp_size=1`，由本地入口派生 `moe_ep_size=world_size`。
- `oproj_tp_size=attn_tp_size`、`cp_size=1`。当前唯一配置使用 `draft_model_type=dspark_gqa,next_n=7`。
- `prefill_mini_batch_size=0` 表示每个 Attention DP group 不再拆分 Prefill mini batch；大于 0 时必须能整除 `batch_size_per_rank`。当前配置为 `1`，每个 Attention DP group 执行 16 个 Prefill mini cycle。
- `prefill_chunk_size=0` 表示每个请求一次完成 Prefill；大于 0 时按固定 token chunk 续写 KDA、MLA 和 DSpark cache。当前 chunked Prefill 要求同一 Attention DP group 内请求等长。
- `skip_warm_up=True` 可跳过启动时的完整 Prefill/Decode warm-up，默认开启。图模式下，未命中编译缓存时首次 Decode 需要编译，命中时加载缓存；首次推理仍可能包含图捕获和算子冷启动开销。设置为 `False` 可恢复完整 warm-up。
- `skip_prefill=True` 时不执行 Prefill，KDA/MLA（以及启用 DSpark 时的草稿模型）cache 保持全零，使用占位 token 直接进入 Decode。该模式仅用于 Decode 执行和性能测试，生成文本无精度意义；可用 `prefill_stub_token_id` 指定非 EOS 占位 token。
- `pa_block_size` 必须是 16 的倍数，以满足 MLA PA cache 布局。
- MLA Decode 支持 BF16 和 C8/W8A8C8，使用与所选精度路径兼容的 checkpoint；不支持 MLA 投影部分量化或 MLA 层间混用 BF16/MXFP8。量化方案见[技术报告](../../docs/models/kimi_k3/kimi_k3_inference_guide.md#量化策略)。
- 切换权重、输入形状、算子路径或编译选项后，应重新生成对应的编译缓存。
- Decode 支持 `eager` 和 `npugraph_ex`；Prefill 固定 eager。当前 YAML 开启 SuperKernel，切换 eager 时需同时设置 `custom_params.enable_superkernel=False`。
- 默认开启 MegaKDA ReplaySSM。关闭 ReplaySSM 后使用 MegaKDA snapshot；两者均关闭时使用 snapshot fused recurrent。
- ReplaySSM 要求启用 DSpark、`next_n=7`、950 平台、每个 Attention DP group 的 batch 不超过 16、本地 KDA heads 为 6，以及模型的 hidden size 7168、head dim 128、ShortConv kernel 4 和全秩输出 gate；默认 YAML 满足这些约束。
- ReplaySSM 需要同时安装 `ops.mega_recurrent_kda_replayssm` 和 `ops.commit_recurrent_kda_replayssm`。AttnRes 和 ReplaySSM 的归一化 epsilon 沿用模型配置中的 `rms_norm_eps`。
- 输入先按 `input_max_len` 为 chat template 预留空间并截断正文，再调用 checkpoint tokenizer 的 `apply_chat_template`；最终 Prefill 长度不会超过 `input_max_len`。
- 请求输出使用模型 config、`generation_config.json` 和 tokenizer 三处 EOS 的并集进行截止，兼容 Kimi K3 的 `<|end_of_msg|>` 与 `[EOS]`。
- `enable_profiler=True` 时仅采集正式 Decode，包含对应的 DSpark 调用；Prefill 和 warm-up 不采集，开关未配置时默认为 `False`。输出位置见[日志与性能采集](#日志与性能采集)。

当前 YAML 中的主要执行参数：

下表的值均为 YAML 显式填写的值。`exe_mode` 位于顶层；从 `enable_multi_streams` 到 `enable_prefill_mega_moe` 的开关及 `moe_chunk_max_len` 位于 `model_config.custom_params`；其余表内参数位于 `model_config`。配置为 `True` 表示此配置启用，不等于代码固定开启。

| 参数名 | 类型 | YAML 值 | 含义 |
| --- | --- | --- | --- |
| `exe_mode` | str | `npugraph_ex` | Decode 执行模式，支持 `eager`、`npugraph_ex`；Prefill 固定 eager。 |
| `with_ckpt` | bool | `True` | 是否加载 checkpoint 权重。 |
| `enable_online_split_weight` | bool | `True` | 启动时按 rank 从完整 checkpoint 切分加载权重。 |
| `enable_profiler` | bool | `True` | 是否采集 Decode 性能数据；Prefill 和 warm-up 不采集。 |
| `enable_static_kernel` | bool | `True` | 启用静态 kernel 优化路径。 |
| `enable_cache_compile` | bool | `True` | 是否缓存主模型和 DSpark 的图编译结果。 |
| `force_eplb` | bool | `False` | 是否强制使用均衡专家路由。 |
| `draft_model_type` | str | `dspark_gqa` | DSpark 草稿模型类型；当前默认启用 GQA 草稿模型。 |
| `next_n` | int | `7` | DSpark 生成的草稿 token 数；对应 Verify width 8。 |
| `dspark_tp_size` | int | `8` | DSpark MLP/Markov TP；必须整除 Attention TP，当前部署配置使用默认 TP8。 |
| `skip_warm_up` | bool | `True` | 是否跳过 warm-up；开启后首次正式推理可能包含编译、图捕获和冷启动开销。 |
| `prefill_mini_batch_size` | int | `1` | 每个 Attention DP group 每个 Prefill mini cycle 处理的请求数。 |
| `prefill_chunk_size` | int | `8192` | 每个请求单次 Prefill 的 token chunk 上限。 |
| `pa_block_size` | int | `128` | MLA paged cache 的 block size。 |
| `enable_multi_streams` | bool | `True` | 控制 Decode 多流执行；MoE Decode 使用 router/shared/shared_comm 流，MoE Prefill 固定在当前流执行。 |
| `enable_superkernel` | bool | `True` | 控制 MoE Decode 融合 scope、编译选项及缓存目录；开启要求 `npugraph_ex`、静态 kernel、多流和 shared experts。省略此参数时默认 `False`。 |
| `enable_dspark_confidence_head` | bool | `False` | 是否使用 DSpark confidence head；开启要求草稿 checkpoint 提供对应模块。 |
| `enable_mega_kda` | bool | `True` | Decode 是否进入 MegaKDA 路径；与 ReplaySSM 同时关闭时使用 snapshot fused recurrent。 |
| `enable_mega_kda_replayssm` | bool | `True` | MegaKDA 是否使用 ReplaySSM；要求 DSpark、`next_n=7` 及两个 ReplaySSM 算子包。 |
| `enable_prefill_mega_moe` | bool | `True` | Prefill 是否使用 MegaMoE；关闭时使用 split double-routing 路径。 |
| `moe_chunk_max_len` | int | `12800` | Prefill MoE 分块预算；每 rank chunk 上限按此值除以 EP size 计算，当前 EP32 为 400。 |

当前 YAML 未填写、但代码仍支持的可选参数：

| 参数路径 | 未填写时的值 | 含义 |
| --- | --- | --- |
| `model_config.skip_prefill` | `False` | 跳过 Prefill，以全零 cache 执行 Decode，仅用于性能测试。 |
| `model_config.prefill_stub_token_id` | `None` | 跳过 Prefill 时可指定非 EOS 占位 token。 |
| `model_config.enable_weight_nz` | `True` | 控制 DSpark 权重后处理的 NZ 转换。主模型按模块及算子要求处理权重布局，不受这个开关统一控制。 |

`enable_prefill_mega_moe`、`enable_mega_kda`、`enable_mega_kda_replayssm` 未填写时均为 `False`，当前 YAML 显式设为 `True`。关闭 `enable_mega_kda` 时必须同时关闭 `enable_mega_kda_replayssm`。SuperKernel 与多流独立控制，启用 SuperKernel 时需满足表中的配置条件。

`enable_online_split_weight` 表示启动时从完整 checkpoint 按 rank 加载权重，不代表支持在线请求调度。SiTU 扩展为启动必需依赖，安装方法见[环境准备](#环境准备)。算子融合、缓存管理与并行方案见[技术报告](../../docs/models/kimi_k3/kimi_k3_inference_guide.md)。

### 拉起多卡推理

在每个节点执行：

```shell
cd /home/code/cann-recipes-infer/models/kimi_k3
bash infer.sh
```

脚本默认使用 `kimi_k3_rank_32P_100k_1batch.yaml`；选择其他配置时，在启动前设置 `YAML_FILE_NAME`。脚本会将 K3 内置的 100K 请求复制到 `dataset/default_prompt.json`，供 `dataset: "default"` 使用。若改用 `InfiniteBench`，请提前将数据放到各节点的 `dataset/InfiniteBench` 目录下。

默认跳过 warm-up；设置 `skip_warm_up=False` 后，先执行 warm-up，再清空主模型和 DSpark 的 cache，开始正式推理。

**复现最优性能时，请开启 `model_config.enable_cache_compile=True`，使用相同配置完整运行两遍 `bash infer.sh`：第一遍生成编译缓存，第二遍复用缓存，以第二遍的性能结果为准。两次运行之间保留 `compile_cache` 目录。**

### 日志与性能采集

- 日志输出生成结果、主模型 Verify 和 DSpark 的平均耗时，以及实测平均接受长度和接受率。
- `enable_profiler=True` 时，仅采集 Decode：跳过前 10 轮，再采集 5 轮，结果写入运行输出目录下的 `prof/decode`。Prefill 和 warm-up 不采集。

性能结果见[技术报告总结](../../docs/models/kimi_k3/kimi_k3_inference_guide.md#总结)。

## 已知限制

- 当前支持固定批次生成；Prefill mini batch 是静态多 cycle，不支持在线请求加入、退出或动态调度。
- Chunked Prefill 当前仅支持等长请求；不支持 MTP、PD 分离和 Context Parallel，投机推理仅支持显式配置的 DSpark。
