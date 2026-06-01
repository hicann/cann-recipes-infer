# MiniMax-M2.5 MXFP4 for vLLM-Ascend

## 项目简介

本项目提供 MiniMax-M2.5 在昇腾 NPU 上基于 vLLM 的 MXFP4/W4A4 量化推理部署方案，包含针对 `vllm` 和 `vllm-ascend` 的补丁及一键启动脚本。

主要功能：

- **`vllm-ascend` 量化能力增强**（`0001` 补丁）：在 NPU 平台能力列表中补齐 Quark 量化方法识别，增强 `quant_description` 读取健壮性，避免缺失 key 导致加载失败
- **`vllm` MiniMax M2.5 MXFP4/W4A4 全链路适配**（`0002` 补丁，含 7 个 commit）：
  - QK RMSNorm 张量并行切分修复（KV head < TP size 场景）
  - NPU 端 MXFP4 FP4-E2M1 软件反量化及 E8M0 block scaling 实现，替代 CUDA-only 的 Quark 内核
  - NPU 端激活 QDQ 仿真（FP4 E2M1 codebook + power-of-two block scale）
  - Quark MoE apply 函数签名适配昇腾 `fused_experts` 调用路径
  - MoE 权重加载时预反量化，避免图模式捕获阶段的动态反量化
  - MoE routing 回退对齐 layer 级 router 语义（grouped-topk / correction bias）
- **ARM 内存序修复**：`shm_broadcast.py` 中添加 `memory_fence()` 保障弱内存序平台的正确性
- **一键启动脚本**：支持 MXFP4 W4A4 场景的快速部署，可通过环境变量灵活配置

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
docker pull quay.io/ascend/vllm-ascend:v0.14.0rc1-a3
```

### 补丁基线版本

补丁与 `vllm` / `vllm-ascend` 的源码版本严格对应，建议使用推荐镜像中自带的版本。使用其他版本可能导致 `git am` 应用失败或运行时行为不一致。

| 补丁文件 | 目标仓库 | 仓库地址 | 基线 commit | 容器内路径 |
|---------|---------|---------|------------|-----------|
| `0001-vllm-ascend-patch-for-mxfp4.patch` | vllm-ascend | https://github.com/vllm-project/vllm-ascend | `52d4acfa51fb868823d1070b81cbd2d97e9e4696` | `/vllm-workspace/vllm-ascend` |
| `0002-vllm-patch-for-mxfp4.patch` | vllm | https://github.com/vllm-project/vllm | `d7de043d55d1dd629554467e23874097e1c48993` | `/vllm-workspace/vllm` |

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
    quay.io/ascend/vllm-ascend:v0.14.0rc1-a3 /bin/bash
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
cd /project/to/cann-recipes-infer/contrib/minimax_m2.5_mxfp4
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

- 先对 `vllm` / `vllm-ascend` 做 dry-run 检查
- 再用 `git am --3way` 应用补丁
- 对 `vllm` 额外补上 `shm_broadcast.py` 的 ARM 内存序修复
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
RUN_IN_BACKGROUND=1 \
bash run_vllm_w4a4.sh
```

`run_vllm.sh` 是 `run_vllm_w4a4.sh` 的兼容入口，默认行为一致。

## 关键环境变量

- `MODEL_PATH`：MiniMax-M2.5 MXFP4 权重目录
- `SERVED_MODEL_NAME`：服务暴露模型名，默认 `MiniMax-M2.5`
- `TP_SIZE`：张量并行大小，默认 `16`
- `PORT`：服务端口，默认 `8000`
- `MAX_NUM_SEQS`：最大并发请求数，默认 `32`
- `MAX_NUM_BATCHED_TOKENS`：最大 batch token 数，默认 `32768`
- `ENABLE_EXPERT_PARALLEL`：是否开启 EP，默认 `1`
- `VLLM_ASCEND_ENABLE_FLASHCOMM1`：是否开启 FlashComm1，默认 `1`
- `VLLM_MXFP4_SKIP_ACT_QDQ`：是否跳过激活 QDQ 仿真，默认 `0`
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

## 故障排查

- `git am` 失败：通常表示本地 `vllm` 或 `vllm-ascend` 版本与补丁基线不一致，请先确认源码版本或手动处理冲突
- `amd-quark` 安装失败：确认 Python 环境和网络可用，必要时先手动安装
- 服务启动后报 Quark / MXFP4 相关错误：优先确认补丁是否全部应用成功，以及 `MODEL_PATH` 是否为对应的 MXFP4/W4A4 权重
- 多卡或多进程异常：优先确认 `ASCEND_RT_VISIBLE_DEVICES`、`TP_SIZE`、EP 配置是否匹配当前机器资源
