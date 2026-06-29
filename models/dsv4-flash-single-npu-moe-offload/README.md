# DeepSeek-V4-Flash 模型在 Ascend NPU + K920 CPU 实现单卡推理

在**单张 Atlas 910B + Kunpeng-920 CPU** 上跑 DeepSeek-V4-Flash 的混合推理：attention / shared / router /
热专家走 **NPU W8A8**，其余专家 offload 到 **CPU**（kt-kernel 吃**原生 MXFP4 GGUF**，搬运字节减半）+ NPU graph。
生产实测 decode **~13–16 tok/s**。瓶颈是 CPU MoE 内存带宽，roadmap 围绕「让 NPU 多接热专家 + CPU↔NPU 重叠」。

950支持正在准备，当前验证基于910B验证，A3验证进行中

当前还未做完备的精度验证，后续会补上，当前decode输出正常回复，基本和https://github.com/sgl-project/sglang/issues/23598的版本能对齐。后续sglang正式版本发布后，会完成正式版本的精度验证。

> ⚠️ **sglang 这部分目前不是正式版本**：当前以 patch 形式打在一个 DSv4 公开基线上。待 **sglang 主干正式支持
> 该路径后，会改为基于主干**（届时本交付的 sglang 补丁会相应调整甚至废弃）。kt-kernel / llama.cpp 改动相对稳定。

## 硬件要求

| 部件 | 要求 |
|---|---|
| **NPU** | 1× Atlas 910B（64 GB HBM）或 A3。运行占 HBM ~16–20 GB（常驻 expert）+ attention + KV |
| **CPU** | aarch64，ARMv8.2-A + NEON dotprod（SDOT）；**不需要** SVE/BF16/I8MM。核越多越好（decode 内存带宽受限，默认 128 线程跨 8 NUMA）。验证于 Kunpeng-920（192 核 / 8 NUMA） |
| **DDR（内存）** | **≥ 160 GiB 可用，推荐 ≥ 256 GiB**：要把 ~138 GiB 的 MXFP4 GGUF 常驻 page cache。decode 是内存带宽瓶颈 → **多通道高带宽（DDR4-3200+ / 多 NUMA）直接决定吞吐**，不只是容量。验证于 1.5 TB（8 NUMA） |
| **磁盘** | 见下表。**建议预留 ≥ 600 GiB**（转换期 W8A8 + 原生 MXFP4 源 + 生成 GGUF 三者并存峰值 ~560 GiB）；GGUF 转完并校验后删原生 MXFP4 源，serving 常驻降到 ~415 GiB（见 `../../docs/models/dsv4-flash-single-npu-moe-offload/dsv4_flash_single_card_inference_guide.md` §5） |

权重/产物实际大小（本环境实测）：

| 项 | 大小 | 用途 |
|---|---|---|
| W8A8 safetensors（ModelScope） | **~275 GiB** | NPU 侧，serving 常驻（`MODEL_PATH`）|
| 原生 MXFP4 源（HuggingFace） | **~150 GiB** | 仅转换/校验用，转完可删 |
| MXFP4 GGUF（43 层，转换产物） | **~138 GiB** | CPU 专家，serving 常驻 |

> 下载地址与流程见 `../../docs/models/dsv4-flash-single-npu-moe-offload/dsv4_flash_single_card_inference_guide.md` §1。

## 交付物

| 内容 | 位置 |
|---|---|
| **代码补丁**（仅三仓源码改动） | `main_repo/` `sglang/` `llama_cpp/` + `apply_all.sh` |
| **使用文档**（端到端步骤） | `../../docs/models/dsv4-flash-single-npu-moe-offload/dsv4_flash_single_card_inference_guide.md` |
| **方案文档**（架构/量化/roadmap/已证伪） | `../../docs/models/dsv4-flash-single-npu-moe-offload/dsv4_flash_single_card_design.md` |
| 独立脚本（转权重/拉起/校验，**不在 patch 内**） | `scripts/` |

> **patch 只含三仓代码改动**；脚本、文档、权重都不进 patch。背景/方案/进度细节看 USAGE 与 DESIGN，本文不展开。

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
| sglang | `iforgetmyname/sglang` @ `298193eb3` | Apache-2.0 | Huawei (2026) | `sglang/*.patch`（NPU KV/triton 回退、KT EP CPU MoE offload、打包） |
| llama.cpp | `ggerganov/llama.cpp` @ `a94e6ff`(b3173) | MIT | Huawei (2026) | `llama_cpp/*.patch`（GGUF NumPy2 修复、新增 ggml MXFP4 类型） |

> 上述对 Apache-2.0 项目（ktransformers / sglang）的改动，依 Apache-2.0 §4(b) 标注为
> "Modified by Huawei Technologies Co., Ltd. in 2026"；新增的 ggml MXFP4 类型等对 llama.cpp(MIT)
> 的改动同此署名。本目录新增的脚本/文档为 Huawei Technologies Co., Ltd. 版权，按 Apache-2.0 发布
> （见各文件头与 `LICENSE.txt`）。

## 快速开始

端到端步骤（拉镜像/权重 → 起容器 → clone 三仓到上述 SHA + 设 third_party → 打补丁 → 编译 → 转 GGUF → 拉起 → 连贯性验收）详见 [`../../docs/models/dsv4-flash-single-npu-moe-offload/dsv4_flash_single_card_inference_guide.md`](../../docs/models/dsv4-flash-single-npu-moe-offload/dsv4_flash_single_card_inference_guide.md)。

