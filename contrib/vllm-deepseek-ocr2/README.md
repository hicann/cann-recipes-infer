# DeepSeek-OCR-2

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![Ascend](https://img.shields.io/badge/Ascend-A2-orange.svg)](https://www.hiascend.com/)
[![vLLM](https://img.shields.io/badge/vLLM-0.8.5-purple.svg)](https://github.com/vllm-project/vllm)

## 项目简介

基于 vLLM-Ascend 的 DeepSeek-OCR-2 模型昇腾 NPU 推理适配方案，支持高精度文档 OCR 识别与 Markdown 格式输出，此适配任务由智子芯元 KernelCAT 智能体工具自动化完成。

## 功能特性

- ✅ NPU 原生 MOE 算子支持
- ✅ 非侵入式适配方案
- ✅ 模块化设计，易于维护和升级
- ✅ 支持单图、PDF 文档和批量评估
- ✅ 结构化 Markdown 输出

## 快速开始指南

### 硬件要求

| 项目 | 要求 |
|------|------|
| 昇腾设备 | Atlas 800I/T A2 |
| 内存 | ≥ 32GB |
| 磁盘 | ≥ 50GB (模型存储) |

### 基础环境

本项目基于 vllm-ascend v0.8.5rc1 开发，可以使用以下镜像：

```bash
docker pull quay.io/ascend/vllm-ascend:v0.8.5rc1
```

### 创建容器

```bash
docker run -it -d --net=host --shm-size=512g \
    --privileged \
    --name ds-ocr-2 \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/sbin:/usr/local/sbin:ro \
    -v /data/model_weight:/data \
    quay.io/ascend/vllm-ascend:v0.8.5rc1 /bin/bash
```

### 安装依赖

```bash
pip install einops addict easydict triton-ascend PyMuPDF img2pdf -i https://mirrors.huaweicloud.com/repository/pypi/simple
```

### 项目部署

将本项目包放在容器内 `/workspace` 目录下
## 快速开始
### 模型下载
```bash
pip install modelscope -i https://mirrors.huaweicloud.com/repository/pypi/simple
# 下载模型
modelscope download --model deepseek-ai/DeepSeek-OCR-2 --local-dir /data/models/DeepSeek-OCR-2
```
参数说明：
- --model: 模型名称
- --local-dir: 模型存储路径

### 配置文件修改

修改配置文件 `config.py`，设置正确的模型路径和输入数据路径：

```bash
vi /workspace/vllm_deepseek_ocr2/config.py

# 配置模型、输入和输出路径
MODEL_PATH = './models/DeepSeek-OCR-2'
INPUT_PATH = './data/ocrtest.jpg'
OUTPUT_PATH = './output'
```

### 运行推理

```bash
# 快速验证命令
source /workspace/vllm_deepseek_ocr2/set_env_reliable.sh && export ASCEND_RT_VISIBLE_DEVICES=0 && cd /workspace/vllm_deepseek_ocr2 && python3 run_inference_sync.py
```

## 脚本使用说明

### 图片流式输出

```bash
source /workspace/vllm_deepseek_ocr2/set_env_reliable.sh
python /workspace/vllm_deepseek_ocr2/run_dpsk_ocr2_image.py
```

### PDF 批量处理

```bash
source /workspace/vllm_deepseek_ocr2/set_env_reliable.sh
python /workspace/vllm_deepseek_ocr2/run_dpsk_ocr2_pdf.py
```

### 图片批量处理

```bash
source /workspace/set_env_reliable.sh
python /workspace/vllm_deepseek_ocr2/run_dpsk_ocr2_eval_batch.py
```

> 注意：使用批量处理脚本时，`config.py` 中输入图片路径应为图片文件夹路径

## 性能测试（单卡）
## 测试脚本使用

```bash
# 脚本使用前需在`config.py`中配置正确的模型路径和输入数据路径
vi /workspace/vllm_deepseek_ocr2/config.py
source /workspace/vllm_deepseek_ocr2/set_env_reliable.sh
python3 /workspace/vllm_deepseek_ocr2/benchmark.py --concurrent 256 --gpu-mem 0.9
```
参数说明：
- --concurrent: 并发数量
- --max-tokens: 最大生成长度

### 性能数据

| 并发数 | 输出吞吐 (tokens/s) | 总吞吐 (tokens/s) |
|--------|---------------------|-------------------|
| 1      | 40.50               | 96.78             |
| 4      | 106.50              | 292.68            |
| 8      | 212.52              | 584.02            |
| 32     | 413.68              | 1136.81           |
| 64     | 486.62              | 1337.26           |
| 100    | 550.45              | 1512.68           |


## 技术适配

### NPU MOE 算子适配

- **新增文件**: `deepseek_ocr2_npu.py`
- **问题**: 原版 vLLM 的 MOE 层依赖 CUDA 专有的 `torch.ops._moe_C` 操作
- **解决方案**:
  - 使用 vllm-ascend 提供的 Ascend 原生 MOE 实现
  - 替换 `fused_moe` 为 `select_experts + fused_experts`
  - 避免 CUDA 依赖，实现 NPU 原生加速
- **效果**: 模型可完全在 NPU 上运行，无需 CPU 回退

### 注意力机制适配

- **文件**: `deepencoderv2/sam_vary_sdpa.py`
- **移除依赖**: 注释掉 flash_attn 的导入
- **原因**: FlashAttention 是 CUDA 专用
- **解决方案**: 使用 `scaled_dot_product_attention` 实现注意力计算


## 文件结构

### 关键文件

```
vllm_deepseek_ocr2/
 ├─ 🔴 deepseek_ocr2_npu.py      [核心] NPU MOE补丁
 ├─ 🔴 config.py                 [修改] NPU配置参数
 ├─ 🔴 set_env_reliable_v2.sh    [新增] 环境初始化
 ├─ ⚪ deepseek_ocr2.py          [原样] 模型定义
 ├─ ✅ run_inference_sync.py     [新增] 同步推理
 ├─ ✅ run_dpsk_ocr2_image.py    [修改] 单图推理
 ├─ ✅ run_dpsk_ocr2_pdf.py      [修改] PDF推理
 ├─ ✅ run_dpsk_ocr2_eval_batch.py [修改] 批量评估
 ├─ deepencoderv2/
 │   └─ 🟡 sam_vary_sdpa.py      [修改] 注释flash_attn
 │   └─ ⚪ qwen2_d2e.py          [原样] 模型定义
 │   └─ ⚪ build_linear.py       [原样] 线性层构建
 ├─ process/                     [原样] 数据处理
 └─ 📚 文档 (1个)
      └─ README.md
```

### 图例说明

- 🔴 = NPU适配相关 (新增或修改)
- ⚪ = 原始代码 (未修改)
- 🟡 = 轻微修改
- ✅ = 新增文档

## 故障排除

| 问题 | 解决方案 |
|------|--------|
| `libhccl.so` 未找到 | `source /workspace/set_env_reliable.sh` |
| 指定 NPU 设备 | `export ASCEND_RT_VISIBLE_DEVICES=0` |

## KernelCAT内测申请
KernelCAT限时免费内测中，欢迎体验：https://kernelcat.autokernel.cn

## 项目参考

- [DeepSeek AI](https://github.com/deepseek-ai/DeepSeek-OCR-2) - DeepSeek-OCR-2 模型
- [vLLM](https://github.com/vllm-project/vllm) - 高效 LLM 推理框架
- [vLLM-Ascend](https://gitee.com/ascend/vllm-ascend) - vLLM 昇腾适配
- [Meta SAM](https://github.com/facebookresearch/segment-anything) - 视觉编码器

## 许可证

本项目采用 [MIT License](LICENSE) 开源许可。

本项目包含以下第三方代码：
- SAM (Meta) - Apache License 2.0
- DeepSeek-VL2 (DeepSeek AI) - MIT License
- vLLM - Apache License 2.0
