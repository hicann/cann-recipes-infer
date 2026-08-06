# 设计说明：DeepSeek-V4-Flash 单卡 NPU + Kunpeng CPU 混合推理

> 操作步骤见 [dsv4_flash_single_card_inference_guide.md](dsv4_flash_single_card_inference_guide.md)。本文只讲设计原理、当前状态和后续规划。

## 术语

| 术语 | 含义 |
|---|---|
| CPU MoE | MoE 路由专家中 offload 到 CPU 的那部分：权重常驻 CPU 内存，由 Kunpeng CPU 计算，与 NPU 上计算的专家相对 |
| 原生 MXFP4 | 官方发布权重自带的 MXFP4 量化（E2M1 nibble + ue8m0 scale），非本项目再量化产生 |
| 无损 repack | 只调整权重字节排布以适配 kernel，数值逐 bit 不变 |
| 常驻专家（resident） | 常驻 NPU HBM 的那批路由专家 |
| 命中率 H | 一个 token 的 top-k 专家中，命中常驻专家的比例 |
| `cpu_moe_wall` | 单 token 内 CPU MoE 的墙钟耗时 |

## 一、设计原理

### 总体思路

DeepSeek-V4-Flash 为 43 层全 MoE 模型（`first_k_dense_replace=0`），每层 256 个路由专家，每 token 激活 6 个（top-k=6），另有 1 个共享专家。单卡 HBM 放不下完整模型，故按下述方式切分：attention、共享专家、router 及一小部分最热的路由专家放在 Ascend NPU（单卡 64 GB HBM），其余路由专家 offload 到 Kunpeng CPU，由 kt-kernel 读取官方权重转出的 GGUF。

官方发布的权重本身即为 MXFP4。offload 到 CPU 的专家沿用这份原生 MXFP4，转 GGUF 只做 bit 级无损 repack；NPU 侧因硬件只支持 int8，使用转出的 int8（W8A8）权重。两侧量化格式不同是由格式来源和硬件能力决定的，不涉及为提速而选择量化方案。

瓶颈在 CPU offload 一侧，且是内存带宽而非算力。batch size 为 1 时，每 token 的主要开销是把当轮激活的专家权重从 DRAM 读取一遍；GEMV、每份权重只用一次，算术强度远低于 roofline 拐点，落在内存受限区。因此工程重点是压低需搬运至 CPU 的权重字节数：以 MXFP4 将搬运量减半、提升 kernel 的搬运效率、将热专家常驻 NPU 以减少 CPU 侧 miss。NPU 上的算子本 PR 不做改动。

经上述优化，`cpu_moe_wall` 从串行模式的 22–27 ms 降至约 18.3 ms（A3 生产配置、H≈26%），此时 CPU 侧带宽已近饱和（实测 142 GB/s，硬顶约 155 GB/s）。后续提速的主要途径是提高命中率、减少搬运字节，量化分析见 §CPU MoE 的 roofline 与吞吐上限。

### 模型与硬件

模型其余规格：`hidden_size=4096`、`moe_intermediate_size=2048`、`head_dim=512`、`num_attention_heads=64`、`num_key_value_heads=1`；attention 为 MLA + NSA + Lightning Indexer（`index_topk=512`）。MTP（`num_nextn_predict_layers=1`）本项目未使能。

本项目在两套环境上均已验证可运行：

| | 环境一：A3 CANN Lab lite-infer-and-train 镜像 | 环境二：910B + K920 裸机 |
|---|---|---|
| NPU | Ascend A3，64 GB HBM | Atlas 910B，64 GB HBM |
| CANN | 9.0.0（镜像仅提供 CANN，依赖与算子从源码构建） | 8.5.0（镜像自带） |
| CPU | Kunpeng 40 核 / 1 NUMA | Kunpeng-920，4 socket × 48 核 / 8 NUMA / 1.5 TB DRAM |
| 内存带宽 | 单 NUMA 硬顶约 155 GB/s | 每 NUMA 3/4 通道 DDR4-3200，理论约 614 GB/s，8 NUMA 独占实测聚合约 442 GB/s |

两代 CANN 的 NSA compressor 调用约定不同（9.0.0 为公开的 18 参 single-state，8.5.0 为私有的 19 参 split-state），由 `KT_NSA_COMPRESSOR_MODE` 选择，启动脚本按已安装的 CANN 版本自动派生。本文 decode 与 roofline 的定量数据测自环境一；涉及 8 NUMA 的调优数值针对环境二。

CPU 侧有一条构建约束：MXFP4 kernel 只依赖 ARMv8.2-A + NEON dotprod（`asimddp`，即 SDOT）与 FP16（`asimdhp`），不使用 SVE / BF16 / I8MM，编译 march 固定为 `-march=armv8.2-a+fp16+dotprod`。该约束是双向的：

- 对 CPU 的要求低，有 dotprod 即可。Kunpeng-920 无 SVE/BF16/I8MM，执行这些指令会 SIGILL。
- 在具备这些扩展的机器上必须显式关闭。A3 主机 CPU 带 `sve`/`bf16`/`i8mm`，kt-kernel 的 `setup.py` 会依 `/proc/cpuinfo` 自动开启对应 march，导致 gcc-9 编译失败（`+bf16/+i8mm` 与 `-std=gnu++20` 不被识别），且 SVE 开启后 MXFP4 MoE 失效（报 `llamafile not supported`）。构建日志应确认 `DOTPROD=ON` 且 `SVE=OFF / BF16=OFF / I8MM=OFF`。

### 两份权重

NPU 只支持 int8，故需两份权重：

- **NPU 侧 W8A8 safetensors**（int8 + fp32 per-channel scale）：承载 attention、共享专家、router 及常驻 NPU 的热专家，启动时由 `MODEL_PATH` 指向。
- **CPU 侧以官方原生 MXFP4 为转换源**（`expert_dtype:"fp4"`）：转成 GGUF 后供 CPU offload 的专家使用。原生 MXFP4 专家张量为 `layers.{L}.ffn.experts.{i}.w1/w3/w2.weight`（`I8`，K 维 nibble-packed 成 K/2）加 `.scale`（`F8_E8M0`，K/32 分组），每层专家独占一个 safetensors shard。

router gate 与共享专家始终留在 NPU，不 offload：二者直接影响路由与精度，放到 CPU 无收益且引入额外往返。

### 系统架构与数据流

```
单卡：Ascend A3 (64 GB HBM) + Kunpeng CPU (环境一 40 核 / 1 NUMA；环境二 K920 8 NUMA)

input → [NPU: embedding / RoPE / MLA+NSA+Indexer attention]
      → [NPU: MoE router gate → topk_ids, topk_weights (k=6)]
      → ┌──────────────────────────┬──────────────────────────────┐
        │ NPU experts (默认 N=32)   │ CPU experts (默认 224)          │
        │ W8A8 safetensors         │ kt-kernel LLAMAFILE GGUF        │
        │ + shared experts (常驻)   │ 原生 MXFP4                       │
        └──────────────────────────┴──────────────────────────────┘
      → merge → linear + residual → 下一层
```

**NPU 端**：attention 走 SGLang 的 `--attention-backend ascend`（MLA+NSA+Lightning Indexer）；NPU 上的 MoE 用 `fused_experts_npu`（W8A8），承载前 N 个路由专家、共享专家与 router top-k；KV cache 放 HBM。

**CPU 端（kt-kernel）**：backend 为 LLAMAFILE（`kt-kernel/operators/llamafile/moe.hpp`）。该路径对量化类型泛化，buffer 尺寸、激活量化、NUMA 张量并行、加载加速、graph callback 均经 ggml 的 `type_traits`，换用 MXFP4 无需改动主线。线程池按 NUMA 划分，`KT_THREADPOOL_COUNT` 不得超过主机 NUMA 节点数；默认按单 NUMA 的环境一配置（1 池 / 32 线程），8 NUMA 机器的推荐值见使用指南。

MXFP4 的 GEMV kernel 为 `ggml_vec_dot_mxfp4_q8_0`（llama.cpp patch 0002）：`vqtbl1q_s8` 查 E2M1 表，`vdotq_s32` 做 SDOT，再乘 e8m0 scale。其中行内 `__builtin_prefetch(+512B)` 加双 `float32x4` FMA 累加链对 K920 尤为关键——TSV110 的硬件预取器跟不上 MXFP4 这类低密度 load 流，手工软预取将单核 GEMV 从约 0.9 提到约 3.2 GB/s/核，整体约 2.4×。kernel 在做点积前，需要把该次的激活向量量化为 Q8_0（int8 + 每 32 个元素一个 fp16 scale）。这是 ggml 的约定：每种权重类型都指定一个 `vec_dot_type`，即点积另一侧必须使用的格式，MXFP4 对应 Q8_0，因此每次调用都会做一次这样的在线量化。权重侧是无损 repack、累加在 int32 上进行（`vdotq_s32`）再乘 scale 转回 fp32，两者都不引入误差，所以这一步激活量化是 CPU MoE 路径上唯一产生数值误差的环节。

**NPU↔CPU 桥**：`kt-kernel/cpu_backend/ascend_callback_worker.{cpp,h}` 起后台线程做 `aclrtSubscribeReport` + 循环 `aclrtProcessReport`，把 CPU MoE 的 submit/flush 接入 NPU graph 的 host callback。ACL 的 `aclrtLaunchCallback` 不会自动触发，必须有专用 poller 线程 subscribe + process，否则会卡在 sync、NPU 空转。

**SGLang 集成**：核心是 per-layer 的 `KTMoEWrapper`（`…/layers/moe/kt_ep_wrapper.py`，负责 `mask_cpu_expert_routing`、prefill/decode 分化、graph 走 host callback），设备抽象在 `…/utils/kt_accel.py`。triton 与 ascend 版本错配时自动探测并回退到纯 PyTorch 的等价实现（数值等价，无需开关）。集成方式是在 sglang DSv4 基线上加分支/继承，不 fork 整个模型实现，便于后续升级子模块。

### 量化与 nibble 序

MXFP4 是官方发布的量化（训练侧已对齐），转 GGUF 全程 bit 级无损 repack。CPU 用 MXFP4、NPU 用 W8A8 混用不影响正确性，各专家独立近似同一份母权重；离线对账 cosine 0.999939、max_rel 1.12%，其中的数值误差全部来自 kernel 对激活的在线 Q8_0 量化（见 §系统架构与数据流），权重侧无损。量级上每元素 0.53125 字节（17 B / 32 元素），单专家 13.4 MB，43 层全部常驻 DRAM 约 137 GiB。

**MXFP4 的存储格式**：每个权重元素占 4 bit（E2M1 格式），因此两个元素合用一个字节、各占其中半个字节——半个字节即一个 **nibble**。沿 K 维每 32 个连续元素为一组，共享一个 8 bit 的 e8m0 指数 scale。所以一组 = 16 字节 codes（32 个 nibble）+ 1 字节 scale，共 17 字节，这也是"每元素 0.53125 字节"的由来。

**所谓 nibble 序，是指某个字节里的两个 nibble 分别对应哪两个 K 位置。** 官方 ckpt 与上游 GGUF 用的是两种不同约定：

| 排布 | 低 nibble 存 | 高 nibble 存 | 用在 |
|---|---|---|---|
| consecutive | K 位置 `2i` | K 位置 `2i+1` | 官方原生 ckpt |
| half-block | K 位置 `j` | K 位置 `j+16` | 上游 GGUF `block_mxfp4` |

以一组 32 个元素为例：consecutive 排布下，第 0 字节装 K0 与 K1，第 1 字节装 K2 与 K3，依此类推；half-block 排布下，第 0 字节装 K0 与 K16，第 1 字节装 K1 与 K17，相当于把这一组从中间切成两半、各取一个 nibble 拼成一个字节。两种排布的字节总数完全相同，但同一个字节的含义不同。

因此转换必须**逐 32 元素组重新排列 nibble**，不能整段 byte copy；e8m0 scale 是独立字节，原样直存即可。

排布用错不会报任何错，只会让每个权重元素被当成另一个 K 位置的值，表现为输出乱码或精度骤降。这也是转换器与 kernel 必须遵循同一套约定的原因。`verify_mxfp4_layer.py` 做的正是 GGUF 反量化与原生反量化的逐元素 bit-exact 对账，改动这条路径前后都应执行。

## 二、当前状态

### 已实现能力

单卡整网可用，HTTP 服务正常、输出连贯。

**基础能力**

- 编译期 NPU 适配（`main_repo/0001`）
- 单卡整网 wiring：SGLang + CPU MoE offload（`sglang/0002`）
- NPU graph + ACL callback worker 闭合（`main_repo/0001` + `sglang/0002`）
- CPU 权重加载加速：zero-copy mmap + 并行重排，43 层约 47 s（`main_repo/0001`、`0002`）
- graph decode 提速：kt-cpuinfer 24→128 + GEMV 行内预取（`main_repo/0002`）
- CPU MoE 直接读取原生 MXFP4（无损 repack、不重新量化），叠加 kernel 行内预取 2.4×（`main_repo/0002` + `llama_cpp/0002`）
- triton×ascend 自动回退 torch 等价实现（KV / MoE，无需 env）（`sglang/0001`）

**device-offload 主线（默认开启）**

- **动态热专家常驻**（`KT_DYNAMIC_RESIDENT=1`）：按实际路由选取最热的专家常驻 NPU，替代静态前 32，覆盖的激活量约为静态方式的 3 倍。
- **长序列 prefill 流式加载**（`KT_PREFILL_STREAM=1`）：长 prefill 将全部 256 专家流式搬上 NPU 计算，4096 prefill 约 14 s（hybrid 约 137 s，~8×）。
- **depool + AscendC MXFP4 算子**（`KT_MXFP4_DEPOOL=1`）：不再常驻整份 W8A8 池，改由 device 端 AscendC kernel 在线转换（MXFP4 dequant + ND→NZ），显著降低内存占用。
- **GGUF dedup**（`KT_MXFP4_GGUF_DEDUP=1`）：NPU 侧复用 CPU 已 mmap 的 GGUF，省掉一份常驻。
- **CPU↔NPU overlap**（`KT_SIDE_STREAM` / `KT_SHARED_EXPERTS_STREAM`）：CPU MoE 的 host callback 走侧流，与 NPU 上的专家计算重叠。

**硬件兼容**

- **NSA compressor 模式**（`KT_NSA_COMPRESSOR_MODE`）：兼容 CANN 9.0.0（A3，公开 18 参 single-state）与 8.5.0（910B，私有 19 参 split-state）两版算子的调用约定差异（`sglang/0004`），启动脚本按已安装 CANN 版本自动派生。

### 吞吐

A3，graph 模式、单请求、预热后的稳态 decode（inter-token 中位）。配置即默认配置：`KT_CPUINFER=32`、`KT_THREADPOOL_COUNT=1`、`KT_NUM_GPU_EXPERTS=32`，depool + 流式 prefill + 动态常驻全开。

| prompt | decode |
|---|---|
| ≤1k | 21.6–22.6 tok/s |
| 8k / 16k / 32k | 19.8 / 19.3 / 19.2 tok/s |

区间约 19–22.5 tok/s（中位），单 token 快端可达 23–25，随上下文变长小幅下降（KV 增大）。长上下文可用 `KT_HOT_TAIL_TOKENS=64` 再获得 +3%~8%；短 prompt 下该项为负收益，故默认关闭。

prefill（预热后）：130/1k/8k/16k/32k = 15.4 / 15.6 / 16.5 / 17.6 / 20.3 s。流式 prefill 有约 15 s 固定开销（每次请求都需将全套专家权重从 DDR 搬到 HBM），几乎不随长度增长；页缓存未热时 32k 可达约 63 s。

`cpu_moe_wall` 约 18.3 ms/token（A3 生产配置、H≈26%）。复现方法见使用指南 §8.1。

### 精度

GPQA-Diamond（910B 实测，3 轮重复，thinking 关闭）：R1 / R2 / R3 = 69.19% / 72.73% / 73.23%，mean **71.72%**、SD 1.80pp。

GPQA 仅 198 题，`temperature=1` 下单次的二项标准误约 ±3.3pp，应以多轮均值为准；与其它实现对比时同样比较多轮均值。复现方法见使用指南 §8.2。temp=0 下逐位一致，clean-code 前后机制上不改变精度。

### 影响吞吐的因素

| 因素 | 影响 |
|---|---|
| 共享机邻居争抢 DRAM 带宽 | 最常见。decode 内存受限，邻居吃带宽会直接抬高 `cpu_moe_wall`，8-NUMA 取 max 又放大尾延迟 |
| 未预热 | 动态热专家常驻需若干次请求收敛，冷启动的前几次请求结果明显偏低 |
| NPU 卡被占用 | 拉服务前先 `npu-smi info` 选空卡 |
| 线程池与 NUMA 拓扑不匹配 | `KT_THREADPOOL_COUNT` 不得超过 NUMA 节点数；线程过少无法用满带宽，占满全部核心则因资源争抢而下降 |
| 并发请求 | `--max-running-requests > 1` 会撞 NPU 争抢窗口，runtime 可能失稳 |
| 其它 | 上下文变长（attention 变慢）、冷盘首启（page cache 未热）、路由偏斜（top-6 全落 CPU 的层多搬字节） |

### CPU MoE 的 roofline 与吞吐上限

> 本节定量数据测自环境一：单卡 A3 + Kunpeng 40 核 / 1 NUMA，graph 开、side-stream 开、`KT_CPUINFER=32`、`KT_NUM_GPU_EXPERTS=32`，固定 16k prompt、`temp=0`。

decode（bs=1）在 CPU 侧是 GEMV、内存带宽 bound：每个专家权重只用一次，算术强度约 3.8 OP/byte，远低于该机 roofline 拐点（约 33）。CPU 只处理未命中常驻的那部分专家：

```
每 token top-k slots ≈ 264（6×43 层 + 共享）
命中率 H = 26.2%（实测）
落 CPU 的 slots = 264 × (1 − 0.262)  ≈ 195
每 token 从 DRAM 读的权重 = 195 × 13.37 MB  ≈ 2.61 GB
反推带宽 = 2.61 GB / 18.3 ms          ≈ 142 GB/s
```

142 GB/s 贴近该机单 NUMA 带宽硬顶（CPU-MoE 微基准峰值约 137，纯拷贝探测硬顶约 155 GB/s），即 CPU MoE 已带宽近饱和。因此进一步提速不能依靠提高带宽利用率，而要依靠提高命中率 H、减少需搬运至 CPU 的字节数。

命中率的收益可量化。`cpu_moe_wall` 随 miss 比例近似线性缩（≈ 24.8 × (1−H) ms），其中约 4.8 ms 可与 NPU 的 resident GEMM 重叠执行，超出部分暴露在关键路径上。NPU 侧不随 H 变化的临界路径（attention + NSA + resident GEMM）约 36 ms，故 `TPOT(H) ≈ 36.4 + max(0, cpu_moe_wall(H) − 4.8)`：

| 命中率 H | `cpu_moe_wall` | 暴露的 CPU 时间 | TPOT | tok/s |
|---|---|---|---|---|
| **26%（当前）** | 18.3 ms | 13.5 ms | 49.9 ms | **20.1** |
| 50% | 12.4 ms | 7.6 ms | 44.0 ms | 22.7（+13%） |
| 60% | 9.9 ms | 5.1 ms | 41.5 ms | 24.1（+20%） |
| **≈80%** | 4.8 ms | 0 | 36.4 ms | **~27.5（+37%）** |

H≈80% 时 CPU MoE 被完全重叠覆盖，decode 达到 NPU 临界路径决定的上限，继续提高命中率不再带来收益。当前动态常驻的 H 约为 26–31%。上表按 16k 口径计算；H 上升会使更多 slot 由 resident GEMM 承担、抬高临界路径，实际上限更可能在 25–27 tok/s。

side-stream 相对串行模式的净收益实测约 9.4 ms/token，由两部分构成：约 4.8 ms 是与 resident GEMM 重叠执行的 CPU 计算；约 4.6 ms 来自提交方式的变化——串行模式下 host 回调提交在主流上，会逐层阻塞后续 kernel 的下发，改走侧流后主流不再被占用。attention 与 NSA 对 CPU MoE 的输入存在数据依赖，无法并入侧流。

## 三、Roadmap

### 已完成

降低 CPU offload 开销有三类途径：减少需由 CPU 计算的专家数、使 CPU 计算与 NPU 计算重叠、减少 prefill 的权重搬运。三者均已实现并默认开启：静态常驻（前 32 个专家，覆盖约 13% 的激活，可回退）→ 动态常驻（按实际路由选取最热专家，覆盖量约为静态的 3 倍）→ CPU↔NPU 侧流重叠。

流式 prefill 仍有约 15 s 固定开销，来自每次请求都需将全套专家权重从 DDR 搬到 HBM，与 prompt 长度基本无关，是长 prompt TTFT 的主要构成。

### 后续方向

按上节模型，H 从当前 26% 提高到 50–60% 对应 +13~20%（22–24 tok/s），H≈80% 的理论上限对应 +37%（约 27 tok/s）。后续方向分为提高命中率与扩展模型/特性两类：

- **改进热专家常驻策略**：当前动态常驻的 H 约为 26–31%。通过跨请求频率统计、按层差异化、hot-tail 采样等策略使 H 达到 50–60%，是 decode 收益最直接的来源。
- **专家预测 + H2D 预取**：提前一个 token 或一层预测将命中的 expert，利用当前计算时间异步预取权重，使搬运与计算重叠。
- **加大 `KT_NUM_GPU_EXPERTS`**：上调常驻专家数可直接提高 H，达到 80% 上限主要依赖该项；代价是 HBM 占用（32→48 约增加十几 GB），受静态权重 48.3 GB、剩余约 8.8 GB 的约束，需与 KV/context 一起权衡。
- **MTP（多 token 预测）**：模型自带 `num_nextn_predict_layers=1`，本项目未使能。使能后可在 decode 侧摊薄单 token 成本。
- **推广到同类模型**：把这套单卡 CPU-offload 方案用于 dspark / dflash 等模型，复用 MXFP4 CPU MoE + 热专家常驻 + 流式 prefill 的整条链路。

当命中率使 CPU MoE 完全被重叠覆盖后，decode 转为 NPU-bound，继续提速需改动 NPU 侧的 attention / NSA / resident GEMM，属本项目当前范围之外。
