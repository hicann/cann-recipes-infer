# sglang runtime 补丁（10 份，按顺序应用）

基线：`sgl-project/sglang` 主线 `5aab054ec8ce6b6100fbfb7aafe67d632a7df3aa`。
合计 33 个文件 / +4536 / −105。全部应用后 `write-tree` 应得
`f34d08b7f3a447034f2f83cdc69a236b605142b9`。

| # | 补丁 | 改动 | 内容 |
|---|---|---|---|
| 1 | `0001-npu-triton-pdl-shim.patch` | 2 文件 / +68 | 给 triton-ascend 补一个 PDL 兼容垫片。主线 29 个内核模块引用 `tl.extra.cuda`，而 Triton 算 cache_key 时连死 constexpr 分支里的属性链也解析，不打这个垫片，服务在捕获解码图时崩溃 |
| 2 | `0002-npu-platform-gate-and-deps.patch` | 3 文件 / +9 −2 | 非 CUDA 平台关掉需要 DeepGEMM 的 FP8 W_o GEMM；删掉公开源上不存在的 `memfabric-zbal==1.2.0`（它会让 51 个依赖一个都装不上） |
| 3 | `0003-npu-accelerator-helpers.patch` | 1 文件 / +62 | CUDA / 昇腾通用的流与事件助手 |
| 4 | `0004-npu-swiglu-clamp.patch` | 2 文件 / +172 −5 | swiglu 截断的昇腾实现；模型定义中的截断在 NPU、CPU offload 和共享专家路径始终一致，不再提供偏离模型语义的环境开关 |
| 5 | `0005-npu-resident-expert-placement.patch` | 4 文件 / +263 −2 | 常驻专家放置（`prefix` / `frequency` 两种策略）与两个对应的服务器参数 |
| 6 | `0006-npu-streaming-prefill.patch` | 1 文件 / +1521 | 流式预填充：整层专家从主机内存流进一个复用的显存槽，MoE 全在 NPU 上算；固定已验证的 pinned pool、32 个复制线程和 blocked convert kernel，不为固定行为新增环境开关 |
| 7 | `0007-npu-kt-moe-dispatcher.patch` | 3 文件 / +582 −44 | 把 kt-kernel 的 CPU 混合专家接到昇腾调度器上 |
| 8 | `0008-npu-dsa-kpool-bf16.patch` | 6 文件 / +1302 −11 | DSA 稀疏索引改走 bf16，绕开 A3 不支持的 fp8 |
| 9 | `0009-npu-attention-backends.patch` | 5 文件 / +293 −16 | 昇腾注意力后端（KDA、MLA、混合线性注意力）的适配与注册 |
| 10 | `0010-npu-shared-expert-fusion-and-router-fp32.patch` | 6 文件 / +264 −25 | 放开共享专家融合，并把非 CUDA 分支的路由器 logits 提到 fp32 |

顺序不能打乱：7 依赖 3/5/6，6 依赖 4，8 依赖 1。

## 两种应用方式

```bash
# 一、保留提交历史（推荐）
git am /path/to/sglang-patches/*.patch

# 二、只要改动，不要历史
for p in /path/to/sglang-patches/0*.patch; do git apply --3way "$p"; done
```

两种都实测过，结果树相同，零冲突。
