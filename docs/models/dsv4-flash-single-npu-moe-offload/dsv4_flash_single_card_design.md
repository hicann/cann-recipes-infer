# 设计说明：DeepSeek-V4-Flash 单卡 NPU + K920 CPU 混合推理

> 操作步骤见 [dsv4_flash_single_card_inference_guide.md](dsv4_flash_single_card_inference_guide.md)。本文只讲设计原理、当前状态和后续规划。

## 一、设计原理

### 总体思路

DeepSeek-V4-Flash 是 43 层、全 MoE 的模型（`first_k_dense_replace=0`，每层都是 MoE），每层 256 个路由专家、每个 token 激活 6 个（top-k=6），外加 1 个共享专家。完整放进单张 NPU 放不下，于是做了一个混合切分：attention、共享专家、router，以及一小部分最热的路由专家放在 Atlas 910B（64 GB HBM）上跑；其余路由专家 offload 到 Kunpeng-920 CPU（1.5 TB 内存、192 核），用 kt-kernel 直接吃官方权重转出的 GGUF。

关于权重格式先说清楚，免得误会：官方发布的权重本身就是 MXFP4。offload 到 CPU 的专家直接沿用这份原生 MXFP4——kt-kernel 能直接吃，转 GGUF 只是 bit 级无损 repack，不重新量化；放在 NPU 上的部分则因为 NPU 只支持 int8，用的是转出来的 int8（W8A8）权重。所以这里并不存在「为了提速去选某种量化」：CPU 用 MXFP4 是因为它就是原生格式，NPU 用 int8 是被硬件限制。

这套混合推理真正的工作量和瓶颈都在 **CPU offload 这条路上，而且是内存带宽、不是算力**。batch size 为 1 时，每个 token 主要花在把当轮激活到的专家权重从 DRAM 搬一遍——算术强度只有约 0.94 MAC/byte，远低于 roofline 平衡点（约 21），牢牢落在内存受限区，算力是富余的。所以本项目的工程重点是 CPU 侧的带宽利用率：kernel 怎么搬得快、线程/NUMA 怎么摆、热专家怎么常驻在 NPU 上少落 CPU。NPU 上的算子本 PR 不动。生产实测 decode 约 13–16 tok/s。

### 模型与硬件

模型其余规格：`hidden_size=4096`、`moe_intermediate_size=2048`、`head_dim=512`、`num_attention_heads=64`、`num_key_value_heads=1`；attention 为 MLA + NSA + Lightning Indexer（`index_topk=512`）。MTP（`num_nextn_predict_layers=1`）本项目禁用（accept_len 不划算，且 sglang 的 NPU NEXTN 另有坑）。

硬件是单张 Atlas 910B（64 GB HBM，只用一张）+ Kunpeng-920 CPU（4 socket × 48 = 192 物理核，8 NUMA，每 NUMA 约 24 核 / 192 GB，共 1.5 TB DRAM）。CANN 8.5.0，由开源镜像 `lmsysorg/sglang:deepseek-v4-npu-910b` 自带。

内存带宽是全局瓶颈，这里给个量级：K920 每 NUMA 3/4 通道 DDR4-3200，理论 spec 约 614 GB/s，清净独占下实测聚合约 442 GB/s。

CPU 侧有一条硬约束要先记住：**K920 没有 SVE / BF16 / I8MM**，只有 ARMv8.2-A + NEON dotprod（`asimddp`，即 SDOT）和 FP16（`asimdhp`）。任何 SVE/BF16/I8MM 指令（`+sve` march、`__bf16`、`smmla`/`usdot`/`ptrue`）在这块 CPU 上会直接 SIGILL。所以编译 march 固定 `-march=armv8.2-a+fp16+dotprod`，MXFP4 kernel 只用 `vqtbl1q_s8`（查表）和 `vdotq_s32`（SDOT）两条指令。后面想换更激进的量化 kernel，先过这一关。

### 两份权重

因为 NPU 只支持 int8，需要两份权重，缺一不可：

- **NPU 侧用 W8A8 safetensors**（int8 + fp32 per-channel scale），承载 attention、共享专家、router，以及常驻 NPU 的那批热专家；启动时 `MODEL_PATH` 指向它。
- **CPU 侧的转换源是官方原生 MXFP4**（`expert_dtype:"fp4"`，E2M1 nibble + ue8m0 scale），转成 GGUF 后喂给 CPU offload 的专家。原生 MXFP4 专家张量是 `layers.{L}.ffn.experts.{i}.w1/w3/w2.weight`（`I8`，K 维 nibble-packed 成 K/2）加 `.scale`（`F8_E8M0`，K/32 分组），每层专家独占一个 safetensors shard。

注意 router gate 和共享专家始终留在 NPU、绝不 offload——它们直接关系到路由和精度，放到 CPU 上既无收益又会引入额外往返。

### 系统架构与数据流

```
单卡：Atlas 910B (64 GB HBM) + K920 (1.5 TB DRAM, 192 核, 8 NUMA)

input → [NPU: embedding / RoPE / MLA+NSA+Indexer attention]
      → [NPU: MoE router gate → topk_ids, topk_weights (k=6)]
      → ┌──────────────────────────┬──────────────────────────────┐
        │ NPU experts (默认 N=32)   │ CPU experts (默认 224)          │
        │ W8A8 safetensors         │ kt-kernel LLAMAFILE GGUF        │
        │ + shared experts (常驻)   │ 原生 MXFP4                       │
        └──────────────────────────┴──────────────────────────────┘
      → merge → linear + residual → 下一层
```

**NPU 端**：attention 走 SGLang 的 `--attention-backend ascend`（MLA+NSA+Lightning Indexer）；NPU 上的 MoE 用 `fused_experts_npu`（W8A8），承载前 N 个路由专家、共享专家和 router top-k；KV cache 放 HBM。

**CPU 端（kt-kernel）**：backend 是 LLAMAFILE（`kt-kernel/operators/llamafile/moe.hpp`）。这条路对量化类型是泛化的——buffer 尺寸、激活量化、NUMA 张量并行、加载加速、graph callback 全部经 ggml 的 `type_traits` 走，换成 MXFP4 不用改这条主线。线程池是 8 个 NUMA worker pool，默认 `--kt-cpuinfer 128 --kt-threadpool-count 8`（每 NUMA 16 线程，留 8 核/NUMA 给 NPU host 和 scheduler）。128 是甜点：再往上到 192 满核会 thrash 直接崩，太低（如 24）又喂不满带宽。

MXFP4 的 GEMV kernel 是 `ggml_vec_dot_mxfp4_q8_0`（llama.cpp patch 0002）：`vqtbl1q_s8` 查 E2M1 表，`vdotq_s32` 做 SDOT，再乘 e8m0 scale。这里有个对 K920 很关键的优化——**行内 `__builtin_prefetch(+512B)` 加双 `float32x4` FMA 累加链**：TSV110 的硬件预取器跟不上 MXFP4 这种低密度 load 流，手工行内软预取把单核 GEMV 从约 0.9 提到约 3.2 GB/s/核，整个 kernel 约 2.4×。激活在线量化到 Q8_0（`vec_dot_type=Q8_0`）是整条路径**唯一的数值损失源**。

**NPU↔CPU 桥**：`kt-kernel/cpu_backend/ascend_callback_worker.{cpp,h}` 起一个后台线程做 `aclrtSubscribeReport` + 循环 `aclrtProcessReport`，把 CPU MoE 的 submit/flush 接进 NPU graph 的 host callback。这里有个坑：ACL 的 `aclrtLaunchCallback` 不会自动触发，必须有专用 poller 线程去 subscribe + process，否则会卡在 sync、NPU 空转。

**SGLang 集成**：核心是 per-layer 的 `KTMoEWrapper`（`…/layers/moe/kt_ep_wrapper.py`，负责 `mask_cpu_expert_routing`、prefill/decode 分化、graph 走 host callback），设备抽象在 `…/utils/kt_accel.py`。当 triton 与 ascend 版本错配时会自动探测、回退到纯 PyTorch 的等价实现（数值等价，不需要任何开关）。集成方式是在 sglang DSv4 基线上加分支/继承，不 fork 整个模型实现，这样后续升级子模块不容易被破坏。

### 量化：CPU 侧原生 MXFP4

MXFP4 是官方发布的量化（训练侧已经对齐），转成 GGUF 全程是 bit 级无损 repack，不是再量化一次。所以 CPU 专家用 MXFP4、NPU 专家用 W8A8 混用没有问题——各专家独立近似同一份母权重。离线对账 cosine 0.999939、max_rel 1.12%，唯一的损失来自激活在线量化到 Q8。

几个量级数字：每元素 0.53125 字节（17 B / 32 元素），单个专家（gate+up+down）13.4 MB，最坏情况一层 top-6 全落 CPU 是 80 MB，43 层全部常驻 DRAM 约 137 GiB。

转换里有一个核心雷区是 **nibble 序**：官方 ckpt 是 consecutive 排布（byte i 存第 2i / 2i+1 个 nibble），而上游 GGUF 是 half-block 排布（`qs[j]` 存第 j / j+16 个）。转换器必须逐 32-group 重排 nibble，不能直接 byte copy；e8m0 scale 字节则原样直存。转换器和 kernel 必须用同一套约定，错了不会报错、只会数值不对——所以 `verify_mxfp4_layer.py` 的 bit-exact 对账是裁判，改这条路径前后都要跑。

## 二、当前状态

### 跑通了什么

v1.0 开源即带，单卡整网已经 HTTP 200、输出连贯：

- 编译期 NPU 适配（`main_repo/0001`）
- 单卡整网 wiring：SGLang + CPU MoE offload（`sglang/0002`）
- NPU graph + ACL callback worker 闭合（`main_repo/0001` + `sglang/0002`）
- CPU 权重加载加速：zero-copy mmap + 并行重排，43 层约 47s（`main_repo/0001`、`0002`）
- graph decode 提速：kt-cpuinfer 24→128 + GEMV 行内预取（`main_repo/0002`）
- **CPU MoE 直接吃原生 MXFP4**（无损 repack、不重新量化）+ kernel 行内预取 2.4×，decode 约 13–16 tok/s（`main_repo/0002` + `llama_cpp/0002`）
- triton×ascend 自动回退 torch 等价（KV / MoE，无需 env）（`sglang/0001`）
- 静态热专家放置：前 32 个常驻 NPU（`--kt-num-gpu-experts 32`），接住约 13% 的激活（`sglang/0002`）

### 吞吐与影响因素

| 场景 | decode |
|---|---|
| 清净独占（NPU 空卡 + CPU 无邻居争抢，`kt-cpuinfer 128`，graph-on） | 约 16 tok/s |
| 中等争抢（共享机有邻居吃 DRAM 带宽） | 约 13–14 tok/s |
| eager（`--disable-cuda-graph`，仅排障/对照） | 约 1.6 tok/s |

以上是单发、`--max-running-requests 1`、短上下文下的稳态 decode；首 token 含约 2–3.5 min 加载（热 cache 后）。

会让吞吐掉下来的情况，按常见程度：

- **共享机邻居争抢 DRAM 带宽**（最常见）：decode 内存受限，邻居吃带宽会直接抬高 `cpu_moe_wall`，8-NUMA 取 max 又放大尾延迟，表现为 median−min 抖动（独占约 16 → 争抢约 13–14）。
- **NPU 卡被别的容器/session 占用**：拉服务前先 `npu-smi info` 选一张空卡。
- **`KT_CPUINFER` 设错**：192 满核 thrash 会崩，太低喂不满带宽，128 是甜点。
- **并发多发**（`--max-running-requests > 1`）：会撞 NPU 争抢窗口，runtime 失稳甚至崩。
- **上下文变长**：NPU attention 随序列增长变慢，长对话 decode 会逐步降速。
- **冷盘首启**：page cache 没热时加载和首 token 慢，但不影响稳态 decode。
- **路由偏斜**：某个 token 的 top-6 恰好全落 CPU 的层会多搬字节（见下面均值 vs 最坏的差别）。

### Roofline：为什么是带宽、差多少

decode（bs=1）是内存受限问题，每 token 的开销主要是把激活到的专家权重从 DRAM 搬一遍。每 token CPU 要搬的字节大致是：

```
top-6 × (1 − 静态热专家命中率) × 单专家 13.4 MB × 43 层
≈ 6 × (1 − ~13%)            × 13.4 MB        × 43  ≈ ~3.0 GB / token
（交叉验证：每 token 读约 224 个专家 ÷ 9632 常驻 × 137 GiB ≈ 3.2 GB）
```

拿这个字节量对带宽算 roofline，再和实测比：

| 量 | 值 |
|---|---|
| 每 token CPU 字节 | ~3.0 GB |
| DRAM 带宽 | 清净聚合 ~442 GB/s ／ 真 spec 614 GB/s |
| 理想 cpu_moe（roofline） | ~6.8 ms（442）／ ~4.9 ms（614） |
| 实测 `cpu_moe_wall` | min ~17 ms，median ~22–27 ms |
| gap | ~2.5–3.5× |

gap 来自激活在线 Q8 量化 + merge + submit/sync 开销 + NUMA 负载不均（max-of-8 的尾巴）+ 共享机带宽低于 spec。算力这边算术强度 0.94 MAC/byte 远低于平衡点 21，FLOPs 不是墙，所以 CPU 侧的杠杆在带宽利用率、以及让更少的专家落到 CPU（见 roadmap），而不是算力。

## 三、Roadmap

### 主线：device-offload（让更多命中在 NPU 上算）

权重字节由原生 MXFP4 格式定死（CPU 直接吃、不再量化），所以压 `cpu_moe_wall` 的杠杆是两个：减少落到 CPU 的专家数、提高 CPU 侧带宽利用率。这条主线讲前者——让更多命中的专家常驻/计算在 NPU 上，落到 CPU 的就更少。它分三步，是同一条线的不同侧面：

1. **静态热专家常驻（已带 v1.0）**：前 32 个专家用 `--kt-num-gpu-experts 32` 常驻 NPU，接住约 13% 的激活。
2. **动态热专家常驻 / EPLB（在做，机制已验证，未并入 v1.0）**：按 activation 频次取最热的专家常驻。动态 top-K 比静态 prefix-32 大约多接 3× 的激活，精度正确，预计能拿到约 2× 提速。这里踩过一个坑：real-topK 输出乱码的根因是常驻权重 gather 走了 host 的 NZ 池切片（host 切片 format-unaware、字节错乱），改成设备切片后修复。
3. **overlap + 专家预测 / H2D 预取（后续）**：提前一个 token / 一层预测会命中的 expert，用当前 token/层的计算时间异步把它的权重 H2D（host→NPU HBM）预取，让搬运被计算掩盖；同时让 CPU MoE 与 NPU 重叠。命中动态前移到 device，落 CPU 的专家更少、`cpu_moe_wall` 的阻塞更小。

### 其它在做 / 计划

- **长序列 prefill 流式加载**（另一条线，已满配跑通约 8×，未并入 v1.0）：4096 prefill 约 14s（对比约 137s），搬的同样是 MXFP4 GGUF。
- **kernel 预取距离扫描 / NUMA 负载均衡**（后续）：用来缩小 `cpu_moe_wall` 相对内存 roofline 的 2.5–3.5× gap。
