# MiniMax-M2.5 MXFP4 for vLLM-Ascend

## 项目简介

本项目提供 MiniMax-M2.5 在昇腾 NPU 上基于 vLLM 的 MXFP4/W4A4 量化推理部署方案，包含针对 `vllm` 和 `vllm-ascend` 的补丁及一键启动脚本。

主要功能：

- **`vllm-ascend` 量化与 MoE 适配**（`0001` 补丁）：
  - 在 NPU 平台能力列表中补齐 Quark 量化方法识别
  - 增强 `quant_description` 读取健壮性，避免缺失 key 导致加载失败
  - 为 MoE MLP 增加 `w2_input_fn` 钩子，支持 w2 输入激活 MXFP4 QDQ
  - 新增 KV-cache MXFP4 QDQ + 块内 Hadamard 旋转（`patch_minimax_m2_kvcache_mxfp4.py`）
- **`vllm` MiniMax M2.5 MXFP4/W4A4 全链路适配**（`0002` 补丁）：
  - QK RMSNorm 张量并行切分修复（KV head < TP size 场景）
  - NPU 端 MXFP4 FP4-E2M1 软件反量化及 E8M0 block scaling 实现（纯 PyTorch，替代 CUDA-only 的 Quark 内核）
  - NPU 端激活 QDQ 仿真（FP4 E2M1 codebook + power-of-two block scale）
  - Quark MoE apply 函数签名适配昇腾 `fused_experts` 调用路径
  - MoE 权重加载时预反量化，避免图模式捕获阶段的动态反量化
  - MoE routing 回退对齐 layer 级 router 语义（grouped-topk / correction bias）
- **一键启动脚本**：支持 MXFP4 W4A4 + KV-cache 场景的快速部署，可通过环境变量灵活配置

### 量化方案

| 组件 | 量化策略 |
|------|----------|
| Linear / MoE 权重 | MXFP4（group_size=32 沿 hidden_dim），加载时反量化为 BF16 |
| 激活 | 在线 MXFP4 QDQ（group_size=32 沿 hidden_dim） |
| KV cache | 每 token MXFP4 QDQ（group_size=32 沿 head_dim），前 `VLLM_KV_MXFP4_ANCHOR` 个 token 保留 BF16 |
| Q/K Hadamard 旋转 | 块对角 H_32 旋转（在每个 32-dim MXFP4 group 内部） |

### 量化权重获取

本样例使用的 MXFP4 量化权重由 [AMCT](https://gitcode.com/cann/amct/tree/master/amct_pytorch/experimental/quantization) 生成，量化算法为 **OSPlus SmoothQuant**。具体样例与流程见 PR：[feat: 新增 MiniMax-M2.7 OSPlus SmoothQuant 量化样例 #227](https://gitcode.com/cann/amct/pull/227)。

AMCT 侧三阶段流程概要：

1. **stage1**：基于校准数据做 OSPlus 阈值搜索，落盘逐层 scale
2. **stage2**：将 scale 融合进 BF16 权重并导出融合后检查点
3. **stage3**：对融合后 BF16 做 RTN MXFP4 转换，导出 Quark 风格可部署 HuggingFace 检查点

本推理样例加载的是 stage3 产物。样例目录位于 `amct_pytorch/experimental/quantization/MiniMax-M2.7/`（PR 合入后）。MiniMax-M2.5 与 MiniMax-M2.7 结构完全一致，本样例的推理加载方法通用，将 `MODEL_PATH` 指向对应 MXFP4 权重目录即可无缝切换。

### 量化精度

评测使用本推理样例，并开启 KV cache MXFP4 量化（token-wise、group size 32；前 32 个 token 保持 BF16）。评测受采样随机性影响可能小幅波动：

| 方法 | HumanEval+ | GSM8K | MATH500 | LongBench v2 | GPQA Diamond |
|------|------------|-------|---------|--------------|--------------|
| OSPlus SmoothQuant | 89.02 | 95.20 | 91.00 | 51.29 | 82.58 |
| M2.7 baseline（BF16） | 91.60 | 95.55 | 91.26 | 54.87 | 88.88 |

## 目录结构

```text
minimax_m2.5_mxfp4/
├── README.md
├── set_env.sh
├── run_vllm.sh
├── run_vllm_w4a4.sh
└── patch_vllm/
    ├── apply.sh
    ├── 0001-vllm-ascend-patch-for-mxfp4.patch
    └── 0002-vllm-patch-for-mxfp4.patch
```

## 硬件要求

| 项目 | 要求 |
|------|------|
| 昇腾设备 | Atlas A3（Ascend 910_93） |
| NPU 卡数 | 16 张 |
| 磁盘 | 需容纳 MiniMax-M2.5 MXFP4 量化权重 |

## 前置条件

执行前请确认本机已准备：

1. 昇腾 CANN / torch / torch_npu 运行环境
2. 本地 `vllm` 源码目录
3. 本地 `vllm-ascend` 源码目录
4. MiniMax-M2.5 MXFP4 权重目录

### 推荐镜像

推荐直接使用以下 Docker 镜像作为基础环境（镜像已包含兼容版本的 `vllm` 和 `vllm-ascend` 源码）：

```bash
docker pull quay.io/ascend/vllm-ascend:v0.18.0rc1-a3
```

### 补丁基线版本

补丁与 `vllm` / `vllm-ascend` 的源码版本严格对应，建议使用推荐镜像中自带的版本。使用其他版本可能导致 `git apply` 应用失败或运行时行为不一致。

| 补丁文件 | 目标仓库 | 仓库地址 | 版本 | 基线 commit | 容器内路径 |
|---------|---------|---------|------|------------|-----------|
| `0001-vllm-ascend-patch-for-mxfp4.patch` | vllm-ascend | https://github.com/vllm-project/vllm-ascend | `0.18.0rc1` | `99e1ea0fe685e93f53ee5adfe4b41cdd42fb809f` | `/vllm-workspace/vllm-ascend` |
| `0002-vllm-patch-for-mxfp4.patch` | vllm | https://github.com/vllm-project/vllm | `0.18.0` | `bcf2be96120005e9aea171927f85055a6a5c0cf6` | `/vllm-workspace/vllm` |

### 创建容器

```bash
docker run -it -d --net=host --shm-size=512g \
    --privileged \
    --name minimax-m25-mxfp4 \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/sbin:/usr/local/sbin:ro \
    -v /path/to/model:/model \
    quay.io/ascend/vllm-ascend:v0.18.0rc1-a3 /bin/bash
```

将 `/path/to/model` 替换为宿主机上 MiniMax-M2.5 MXFP4 权重的实际路径。

### 默认目录约定

以下目录可通过环境变量覆盖：

```bash
export VLLM_DIR=/vllm-workspace/vllm
export VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend
export MODEL_PATH=/model/MiniMax-M2.5-MXFP4
```

## 使用方式

### 1. 初始化环境

```bash
cd /path/to/cann-recipes-infer/integration/vllm/minimax_m2.5_mxfp4
source set_env.sh
```

`set_env.sh` 会自动配置 CANN toolkit、ATB 库路径、`LD_LIBRARY_PATH` 及运行时环境变量。如果 CANN toolkit 不在默认路径，可提前设置 `ASCEND_TOOLKIT_HOME`：

```bash
export ASCEND_TOOLKIT_HOME=/path/to/ascend-toolkit/latest
source set_env.sh
```

### 2. 应用补丁

```bash
bash patch_vllm/apply.sh
```

脚本会：

- 对部分 CRLF 源文件做换行规范化（保证补丁可干净应用）
- 先对 `vllm` / `vllm-ascend` 做 dry-run 检查
- 再用 `git apply` 幂等应用补丁（已应用则自动跳过）
- 可选安装 `amd-quark`

如需关闭 `amd-quark` 安装：

```bash
INSTALL_AMD_QUARK=0 bash patch_vllm/apply.sh
```

### 3. 启动 MiniMax-M2.5 MXFP4 W4A4 服务

```bash
bash run_vllm_w4a4.sh
```

常用覆盖参数示例：

```bash
MODEL_PATH=/path/to/MiniMax-M2.5 \
TP_SIZE=16 \
PORT=8000 \
MAX_NUM_SEQS=32 \
MAX_NUM_BATCHED_TOKENS=32768 \
ENABLE_EXPERT_PARALLEL=1 \
VLLM_MXFP4_ACT_QDQ_SCALE_FACTOR=6.0 \
VLLM_KV_MXFP4_ANCHOR=32 \
RUN_IN_BACKGROUND=1 \
bash run_vllm_w4a4.sh
```

`run_vllm.sh` 是 `run_vllm_w4a4.sh` 的兼容入口，默认行为一致。

上述示例中与调度、MoE 和 MXFP4 量化相关的参数说明：

- `MAX_NUM_SEQS`：对应 `--max-num-seqs`，调度器同时驻留的最大序列数（并发请求上限）。增大可提升吞吐，但会增加 KV cache 与显存占用。
- `MAX_NUM_BATCHED_TOKENS`：对应 `--max-num-batched-tokens`，单次调度步中所有序列合计的最大 token 数（prefill + decode）。增大有利于长 prompt 吞吐，过大会更容易 OOM。
- `ENABLE_EXPERT_PARALLEL`：是否开启 MoE Expert Parallel。设为 `1` 时传入 `--enable-expert-parallel`，将不同专家分到不同卡上；与 `TP_SIZE=16` 一起用于 16 卡 MoE 部署。
- `VLLM_MXFP4_ACT_QDQ_SCALE_FACTOR`：激活 MXFP4 QDQ 的 block scale 除数。实现为 `scale = 2^round(log2(max_abs / scale_factor))`。MXFP4 E2M1 codebook 最大值为 `6.0`，默认除以 `6.0` 使量化后数值落入 codebook 动态范围。增大该值会使 scale 更小、更容易饱和；减小则 scale 更大，量化误差通常增加。
- `VLLM_KV_MXFP4_ANCHOR`：每个 sequence 前 N 个 token 的 KV 保持 BF16、不做 MXFP4 QDQ，用于保护 prompt 开头（如 system / instruction）精度。设为 `0` 则全部量化。该变量在模块 import 时读取，修改后需重启服务。

## 关键环境变量

- `MODEL_PATH`：MiniMax-M2.5 MXFP4 权重目录
- `SERVED_MODEL_NAME`：服务暴露模型名，默认 `MiniMax-M2.5`
- `TP_SIZE`：张量并行大小，默认 `16`
- `PORT`：服务端口，默认 `8000`
- `MAX_NUM_SEQS`：调度器同时驻留的最大序列数（并发请求上限），对应 `--max-num-seqs`，默认 `32`
- `MAX_NUM_BATCHED_TOKENS`：单次调度步中所有序列合计的最大 token 数，对应 `--max-num-batched-tokens`，默认 `32768`
- `ENABLE_EXPERT_PARALLEL`：是否开启 MoE Expert Parallel（`1` 传入 `--enable-expert-parallel`），默认 `1`
- `VLLM_ASCEND_ENABLE_FLASHCOMM1`：是否开启 FlashComm1，默认 `1`
- `VLLM_MXFP4_SKIP_ACT_QDQ`：是否跳过激活 QDQ 仿真，默认 `0`
- `VLLM_MXFP4_ACT_QDQ_SCALE_FACTOR`：激活 MXFP4 QDQ 的 block scale 除数（`scale = 2^round(log2(max_abs / scale_factor))`），默认 `6.0`（对齐 E2M1 codebook 最大值）
- `VLLM_KV_MXFP4_SKIP`：设为 `1` 关闭 KV-cache 量化与旋转（debug），默认 `0`
- `VLLM_KV_MXFP4_SKIP_QDQ`：设为 `1` 只旋转、不 QDQ KV，默认 `0`
- `VLLM_KV_MXFP4_SKIP_ROTATE`：设为 `1` 只 QDQ、不旋转 Q/K，默认 `0`
- `VLLM_KV_MXFP4_ANCHOR`：每个 sequence 前 N 个 token 的 KV 保持 BF16 不量化，默认 `32`；修改后需重启服务
- `VLLM_KV_MXFP4_GROUP_SIZE`：KV-cache MXFP4 group size（沿 head_dim），默认 `32`
- `ENABLE_TOOL_REASONING`：是否开启 MiniMax tool/reasoning parser，默认 `1`
- `ENFORCE_EAGER`：是否强制 eager 模式（跳过图编译），默认 `0`
- `COMPILATION_CONFIG`：编译配置 JSON，默认 `{"cudagraph_mode":"FULL_DECODE_ONLY"}`
- `LOG_DIR`：日志输出目录，默认 `/data/logs`
- `RANK`：当前节点编号，用于多机场景的日志文件命名，默认 `0`
- `RUN_IN_BACKGROUND`：是否后台启动，默认 `0`

## 验证方式

启动成功后可执行：

```bash
curl -sf http://127.0.0.1:8000/v1/models
```

或发送一个简单对话请求：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMax-M2.5",
    "messages": [{"role": "user", "content": "介绍一下中国的上下五千年"}],
    "max_tokens": 256
  }'
```

可用 `VLLM_KV_MXFP4_SKIP_QDQ=1 VLLM_KV_MXFP4_ANCHOR=0` 关闭 KV QDQ、仅旋转 Q/K，用于验证 Hadamard 旋转的数值不变性（输出应与 baseline 接近，仅存在 BF16 舍入误差）。

## 故障排查

- `git apply` 失败：通常表示本地 `vllm` 或 `vllm-ascend` 版本与补丁基线不一致，请先确认源码版本或手动处理冲突
- `amd-quark` 安装失败：确认 Python 环境和网络可用，必要时先手动安装
- 服务启动后报 Quark / MXFP4 相关错误：优先确认补丁是否全部应用成功，以及 `MODEL_PATH` 是否为对应的 MXFP4/W4A4 权重
- 多卡或多进程异常：优先确认 `ASCEND_RT_VISIBLE_DEVICES`、`TP_SIZE`、EP 配置是否匹配当前机器资源
