# DeepSeek-V4-Flash 模型在 Ascend NPU + Kunpeng CPU 实现单卡推理

在**单张 Ascend NPU + Kunpeng CPU** 上完成 DeepSeek-V4-Flash 的混合推理：attention、shared 专家、router
与热专家在 **NPU** 上以 **W8A8** 计算并以 graph 模式执行；其余专家 offload 至 **CPU**，由 kt-kernel 以**原生
MXFP4 GGUF** 为输入直接计算（每元素 0.53125 字节，相比 8 bit 量化格式权重搬运量减半）。
A3 默认配置实测 decode **约 19–22.5 tok/s**（中位值，单 token 峰值 23–25）。CPU MoE（`cpu_moe_wall` 约 18.3 ms）
已接近内存带宽饱和，进一步提速取决于**提高热专家命中率、减少需搬运至 CPU 的权重字节数**；按 roofline 估算
仍有约 37% 提升空间，对应上限约 27 tok/s。

**默认配置即当前最优配置**：动态热专家常驻、长序列流式 prefill、depool（AscendC MXFP4 算子在线转换）、
GGUF dedup、CPU↔NPU overlap 均**默认开启**，启动脚本无需额外调参。

**两条硬件路径**（[使用指南](../../../docs/integration/sglang/dsv4-flash-single-npu-moe-offload/dsv4_flash_single_card_inference_guide.md) 分两节）：
- **A3 —— CANN Lab 单卡镜像**：以未定制的 CANN 9.0.0 镜像为起点，由脚本分阶段安装依赖与自定义算子（三方仓均固定版本）。本文性能数据测自 A3。
- **Atlas 910B —— 单节点裸机环境**：使用已集成依赖的镜像（CANN 8.5.0）。

950 支持正在准备中。

精度：GPQA-Diamond 3 轮重复均值 **71.72%**（SD 1.80pp，910B 实测，thinking 关闭），复现方法见[使用指南](../../../docs/integration/sglang/dsv4-flash-single-npu-moe-offload/dsv4_flash_single_card_inference_guide.md) §8.2。
GPQA 单次采样的标准误约 ±3.3pp，故以多轮均值为准。

> ⚠️ **sglang 部分目前不是正式版本**：当前以 patch 形式应用于一个 DSv4 公开基线。待 **sglang 主干完成性能优化后，
> 将改为基于主干**（届时本交付的 sglang 补丁会相应调整甚至废弃）。kt-kernel 与 llama.cpp 的改动相对稳定。

## 硬件要求

| 部件 | 要求 |
|---|---|
| **NPU** | 1× Ascend A3 或 Atlas 910B。静态占用约 48.3 GB 权重 + 3.7 GB KV + 0.3 GB graph，余量约 8.8 GB |
| **CPU** | aarch64，**必需 ARMv8.2-A + NEON dotprod（SDOT）**——MXFP4 kernel 只用 `vqtbl1q_s8` + `vdotq_s32` 两条指令实现。**SVE / BF16 / I8MM 不需要，且构建时必须关闭**：kernel 不使用它们；若启用（尤其 SVE），MXFP4 MoE 会失效（报 `llamafile not supported`）。**A3 环境的 CPU 是 Kunpeng 40 核 / 1 NUMA（单 NUMA 带宽约 155 GB/s）**，对应默认配置（1 池 / 32 线程）。多 NUMA 主机（910B 环境的 K920，192 核 / 8 NUMA）同样已验证可运行，需按拓扑调高 `KT_THREADPOOL_COUNT=8 KT_CPUINFER=128`（`KT_THREADPOOL_COUNT` 不得超过主机 NUMA 数）。CPU MoE 受内存带宽限制，通道 / NUMA 带宽越高，`cpu_moe_wall` 越低 |
| **DDR（内存）** | **≥ 160 GiB 可用，推荐 ≥ 256 GiB**：需将约 138 GiB 的 MXFP4 GGUF 常驻于 page cache。CPU MoE 的瓶颈在内存带宽，容量之外**带宽也直接影响 decode**。两套环境均已验证可运行：**A3 环境为单 NUMA（约 155 GB/s）**；910B 环境的 K920（8 NUMA、1.5 TB、聚合约 442 GB/s）功能一致 |
| **磁盘** | 见下表。**建议预留 ≥ 600 GiB**（转换期 W8A8、原生 MXFP4 源与生成的 GGUF 三者并存，峰值约 560 GiB）；GGUF 转换并校验完成后可删除原生 MXFP4 源，serving 常驻降至约 415 GiB（见[使用指南](../../../docs/integration/sglang/dsv4-flash-single-npu-moe-offload/dsv4_flash_single_card_inference_guide.md) §5） |

权重/产物实际大小（本环境实测）：

| 项 | 大小 | 用途 |
|---|---|---|
| W8A8 safetensors（ModelScope） | **~275 GiB** | NPU 侧，serving 常驻（`MODEL_PATH`）|
| 原生 MXFP4 源（HuggingFace） | **~150 GiB** | 仅用于转换与校验，完成后可删除 |
| MXFP4 GGUF（43 层，转换产物） | **~138 GiB** | CPU 专家，serving 常驻 |

> 下载地址与流程见[使用指南](../../../docs/integration/sglang/dsv4-flash-single-npu-moe-offload/dsv4_flash_single_card_inference_guide.md) §1。

## 交付物

| 内容 | 位置 |
|---|---|
| **代码补丁**（仅三仓源码改动） | `main_repo/` `sglang/` `llama_cpp/` + `apply_all.sh` |
| **使用文档**（端到端步骤） | [dsv4_flash_single_card_inference_guide.md](../../../docs/integration/sglang/dsv4-flash-single-npu-moe-offload/dsv4_flash_single_card_inference_guide.md) |
| **方案文档**（架构/量化/roadmap/已证伪） | [dsv4_flash_single_card_design.md](../../../docs/integration/sglang/dsv4-flash-single-npu-moe-offload/dsv4_flash_single_card_design.md) |
| 独立脚本（权重转换、启动、校验，**不在 patch 内**） | `scripts/` |

> **patch 只含三仓代码改动**；脚本、文档与权重均不包含在 patch 中。
> 端到端步骤见[使用指南](../../../docs/integration/sglang/dsv4-flash-single-npu-moe-offload/dsv4_flash_single_card_inference_guide.md)，架构/量化/进度见[方案文档](../../../docs/integration/sglang/dsv4-flash-single-npu-moe-offload/dsv4_flash_single_card_design.md)，本文不展开。

## pristine 基线

| 仓 | 公开来源 | SHA |
|---|---|---|
| ktransformers-AK | `kvcache-ai/ktransformers`（0.6.2.post1） | `d7b5b49` |
| sglang | `iforgetmyname/sglang`（dsv4_release） | `298193eb3` |
| llama.cpp | `ggerganov/llama.cpp`（tag b3173） | `a94e6ff` |

## 第三方来源与许可

下列第三方开源项目由 **Huawei Technologies Co., Ltd. 于 2026 年以 patch 形式修改**
（补丁见 `main_repo/`、`sglang/`、`llama_cpp/`）。各项目版权归其原作者所有、遵循各自许可证；
本交付仅含相对上述 pristine 基线的改动，原始版权与许可声明均保留。

| 项目 | 上游 | 许可证 | 修改方 | 本交付的修改 |
|---|---|---|---|---|
| ktransformers | `kvcache-ai/ktransformers` @ `d7b5b49` | Apache-2.0 | Huawei (2026) | `main_repo/*.patch`（kt-kernel：Ascend NPU 后端、CPU MoE MXFP4 kernel） |
| sglang | `iforgetmyname/sglang` @ `298193eb3` | Apache-2.0 | Huawei (2026) | `sglang/*.patch`（NPU KV/triton 回退、KT EP CPU MoE offload、流式 prefill + depool、NSA compressor 兼容、打包） |
| llama.cpp | `ggerganov/llama.cpp` @ `a94e6ff`(b3173) | MIT | Huawei (2026) | `llama_cpp/*.patch`（GGUF NumPy2 修复、新增 ggml MXFP4 类型） |

> 上述对 Apache-2.0 项目（ktransformers / sglang）的改动，依 Apache-2.0 §4(b) 标注为
> "Modified by Huawei Technologies Co., Ltd. in 2026"；新增的 ggml MXFP4 类型等对 llama.cpp(MIT)
> 的改动同此署名。本目录新增的脚本/文档为 Huawei Technologies Co., Ltd. 版权，按 Apache-2.0 发布
> （见各文件头与 `LICENSE.txt`）。

**A3 路径的运行时依赖**（由 `scripts/tools/setup_dsv4_env_from_clean_cann.sh` 从源码构建，**均固定版本、
本交付不修改其代码、不再分发其源码**）：

| 项目 | 上游 | 固定版本 | 用途 |
|---|---|---|---|
| ops-transformer | `gitcode.com/cann/ops-transformer` | `dd9f31f34` | NSA/DSA 算子 → `custom_transformer` vendor |
| cann-recipes-infer | `gitcode.com/cann/cann-recipes-infer` | `c5cc95e` | 融合算子 → `customize` vendor + `custom_ops` 绑定 |
| sgl-kernel-npu | `github.com/sgl-project/sgl-kernel-npu` | tag `2026.6.2` | NPU 融合算子（sglang 依赖） |

## 快速开始

端到端步骤（获取镜像与权重 → 启动容器 → clone 三仓至上述 SHA 并设置 third_party → 应用补丁 → 编译 → 转换 GGUF → 启动服务 → 连贯性验收）详见 [`../../../docs/integration/sglang/dsv4-flash-single-npu-moe-offload/dsv4_flash_single_card_inference_guide.md`](../../../docs/integration/sglang/dsv4-flash-single-npu-moe-offload/dsv4_flash_single_card_inference_guide.md)。

