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

## 硬件要求

| 项目 | 要求 |
|------|------|
| 昇腾设备 | Atlas 800I/T A2 |
| 内存 | ≥ 32GB |
| 磁盘 | ≥ 50GB (模型存储) |

## 快速开始
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

### 模型下载
```bash
pip install modelscope -i https://mirrors.huaweicloud.com/repository/pypi/simple
# 下载模型
modelscope download --model deepseek-ai/DeepSeek-OCR-2 --local_dir /data/models/DeepSeek-OCR-2
```
参数说明：
- --model: 模型名称
- --local-dir: 模型存储路径

### 项目部署
将本项目包下载后放在容器内 `/workspace` 目录下或通过git clone进行拉取。
```bash
cd /workspace
git clone https://gitcode.com/cann/cann-recipes-infer.git
cd cann-recipes-infer/contrib/vllm-deepseek-ocr-2
```
执行转换脚本
```bash
./convert_to_npu.sh
```

脚本会自动：
1. 安装 Python 依赖包（einops, addict, easydict, triton-ascend, PyMuPDF, img2pdf）
2. 克隆 DeepSeek-OCR-2 源码
3. 应用 NPU 适配补丁
4. 输出到 `deepseek_ocr2_npu/` 目录

### 配置文件修改

```bash
cd deepseek_ocr2_npu
# 初始化环境变量
source set_env.sh

# 编辑 config.py 修改以下参数：
vi config.py
# - MODEL_PATH: 模型路径（如 /data/models/DeepSeek-OCR-2）
# - INPUT_PATH: 输入文件路径
# - OUTPUT_PATH: 输出文件路径
```

### 运行推理

```bash
# 图片流式输出
python run_dpsk_ocr2_image.py

# PDF 处理
python run_dpsk_ocr2_pdf.py

# 图片批量处理
python run_dpsk_ocr2_eval_batch.py
```
> 注意：使用批量处理脚本时，`config.py` 中输入图片路径应为图片文件夹路径


## 性能测试（单卡）

```bash
python benchmark.py --image /path/to/image.jpg --concurrent 1,8,16 --warmup 2 --rounds 3
```

**参数说明**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--image` | 图片文件或目录 | 必填 |
| `--concurrent` | 并发数列表 | 1,8,16 |
| `--warmup` | 预热轮数 | 2 |
| `--rounds` | 测试轮数 | 5 |
| `--max-tokens` | 最大输出 token | 8192 |
| `--gpu-mem` | 显存利用率 | 0.85 |
| `--output` | 结果输出文件 | benchmark_results.txt |

## 性能数据

| 并发数 | 输出吞吐 (tokens/s) | 总吞吐 (tokens/s) |
|--------|---------------------|-------------------|
| 1      | 40.50               | 96.78             |
| 4      | 106.50              | 292.68            |
| 8      | 212.52              | 584.02            |
| 32     | 413.68              | 1136.81           |
| 64     | 486.62              | 1337.26           |
| 100    | 550.45              | 1512.68           |


## 适配内容

- **MOE 算子**: 使用 vllm-ascend 的 `fused_experts` 替换 CUDA 实现
- **注意力机制**: 注释 `flash_attn`，使用 SDPA
- **NPU 配置**: `ENFORCE_EAGER=True`、`gpu_memory_utilization=0.85`

## 项目结构

```
vllm-deepseek-ocr-2/
├── convert_to_npu.sh           # 一键转换脚本
├── README.md
├── LICENSE
└── npu_patch/
    ├── deepseek_ocr2_npu.py    # NPU MOE 补丁
    └── set_env.sh              # 环境初始化
```

## 故障排除

| 问题 | 解决方案 |
|------|--------|
| 指定 NPU 设备 | `export ASCEND_RT_VISIBLE_DEVICES=0` |

## KernelCAT内测申请
KernelCAT限时免费内测中，欢迎体验：https://kerminal.cn

## 项目参考

- [DeepSeek AI](https://github.com/deepseek-ai/DeepSeek-OCR-2) - DeepSeek-OCR-2 模型
- [vLLM](https://github.com/vllm-project/vllm) - 高效 LLM 推理框架
- [vLLM-Ascend](https://github.com/vllm-project/vllm-ascend) - vLLM 昇腾适配
- [Meta SAM](https://github.com/facebookresearch/segment-anything) - 视觉编码器

## 许可证

本项目采用 [MIT License](LICENSE) 开源许可。

本项目包含以下第三方代码：
- SAM (Meta) - Apache License 2.0
- DeepSeek-VL2 (DeepSeek AI) - MIT License
- vLLM - Apache License 2.0