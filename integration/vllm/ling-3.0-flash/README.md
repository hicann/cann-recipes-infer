# 百灵模型 ling-3.0-flash
[![Python](https://img.shields.io/badge/Python-3.11-green.svg)](https://python.org)
[![Ascend](https://img.shields.io/badge/Ascend-A3-orange.svg)](https://www.hiascend.com/)
[![Ascend](https://img.shields.io/badge/Ascend-A2-orange.svg)](https://www.hiascend.com/)
[![vLLM](https://img.shields.io/badge/vLLM-0.20.2-purple.svg)](https://github.com/vllm-project/vllm)

## 项目简介
百灵模型 ling-3.0-flash 采用 KDA 线性 attention，叠加逐通道门控衰减、Delta Rule 状态更新优化，实现复杂度从 O(L²) 降至 O(L)，解决超长文本推理问题。本项目基于 PyPTO 编程框架实现 KDA 核心算子，包括 prefill 阶段的 chunk-kda 算子（块内保留精度，跨块近似递推，并行加速 prefill 计算）及 decode 阶段的 fused-recurrent 算子（token 少时串行执行以保精度）。
本项目提供百灵模型 ling-3.0-flash 在昇腾 NPU 上基于 vLLM 的推理部署方案，包含针对 `vllm-ascend` 的补丁及一键启动脚本。

## 功能特性
- ✅ PyPTO chunk_kda 算子支持
- ✅ PyPTO fused_recurrent_kda 算子支持
- ✅ 非侵入式vllm/vllm-ascend适配方案

## 版本要求 
| 项目 | 要求 |
|------|------|
| 昇腾设备 | Atlas A2/A3 |
| hdk | 25.5.x |

## 快速开始
### 推荐镜像

推荐直接使用以下 Docker 镜像作为基础环境（镜像已包含兼容版本的 `vllm-ascend` 源码）：

```bash
# Atlas A2: 
#官方镜像源
export IMAGE=quay.io/ascend/vllm-ascend:nightly-releases-v0.20.2rc-openeuler
#国内镜像源
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:nightly-releases-v0.20.2rc-openeuler

# Atlas A3: 
#官方镜像源
export IMAGE=quay.io/ascend/vllm-ascend:nightly-releases-v0.20.2rc-a3-openeuler
#国内镜像源
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:nightly-releases-v0.20.2rc-a3-openeuler
```

### 模型下载
1、下载pto-isa包并安装
```bash
#/data作为挂载路径，在后续创建容器时，挂载到容器内部
cd /data
# pto-isa包下载命令，后续拉取镜像命令默认拉取arm64
# arm64
wget --no-check-certificate https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann/cann-pto-isa_linux-aarch64_9.1.0_20260723142436.run
# x86_64
wget --no-check-certificate https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann/cann-pto-isa_linux-x86_64_9.1.0_20260723142436.run
```

2、下载pypto-whl包
```bash
# arm64
wget --no-check-certificate https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann/pypto-0.2.1-20260723142436-cp311-cp311-linux_aarch64.whl
# x86_64
wget --no-check-certificate https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann/pypto-0.2.1-20260723142436-cp311-cp311-linux_x86_64.whl
```

### 创建容器

```bash
1、拉取镜像
#官方镜像源
export IMAGE=quay.io/ascend/vllm-ascend:nightly-releases-v0.20.2rc-openeuler
#国内镜像源
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:nightly-releases-v0.20.2rc-openeuler

# Atlas A3: 
#官方镜像源
export IMAGE=quay.io/ascend/vllm-ascend:nightly-releases-v0.20.2rc-a3-openeuler
#国内镜像源
export IMAGE=m.daocloud.io/quay.io/ascend/vllm-ascend:nightly-releases-v0.20.2rc-a3-openeuler

docker pull ${IMAGE}
2、创建容器
docker run -it --rm \
    --name bailing_v3_pypto \
    --device=/dev/davinci0     --device=/dev/davinci1 \
    --device=/dev/davinci2     --device=/dev/davinci3 \
    --device=/dev/davinci4     --device=/dev/davinci5 \
    --device=/dev/davinci6     --device=/dev/davinci7 \
    --device=/dev/davinci8     --device=/dev/davinci9 \
    --device=/dev/davinci10    --device=/dev/davinci11 \
    --device=/dev/davinci12    --device=/dev/davinci13 \
    --device=/dev/davinci14    --device=/dev/davinci15 \
    --device=/dev/davinci_manager --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /data:/data \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /root/.cache:/root/.cache \
    --privileged \
    -it ${IMAGE} /bin/bash
```
参数说明：
- -v /data:/data: 宿主机模型数据/脚本等路径挂载

### 启动容器
```bash
docker exec -it bailing_v3_pypto /bin/bash
```

### 环境部署
本样例的编译执行依赖 CANN 开发套件包（cann-toolkit）与 CANN 二进制算子包（cann-kernels），支持的 CANN 软件版本为 `CANN 9.0.0`，在上述镜像中已默认具备；
pto-isa、pypto-whl、vllm框架patch依赖适配方法如下（其中pto-isa、pypto-whl在CANN 9.1.0以后默认支持）, pto-isa包为9.1.0版本，与CANN 9.0.0版本兼容：
1、安装pto-isa包
```bash
cd /data
# 安装命令
bash cann-pto-isa_linux-*.run --full
```

2、安装pypto-whl包
```bash
pip uninstall pypto && pip install pypto-0.2.1-20260723142436-cp311-cp311-linux_aarch64.whl
或
pip uninstall pypto && pip install pypto-0.2.1-20260723142436-cp311-cp311-linux_x86_64.whl
```

### 项目部署
在npu上部署ling-3.0-flash，执行脚本，程序会自动探测当前Python环境中的vllm与vllm-ascend安装路径并完成注入
```bash
git clone https://gitcode.com/cann/cann-recipes-infer.git
cd cann-recipes-infer/integration/vllm/ling-3.0-flash/npu_patch
bash patch_bailing_v3.sh --monkey
```

### 运行推理

1、启动模型
```bash
# 指定模型路径(路径为已下载的模型权重)
export MODEL_PATH="/data/ling_flash_v3"

# 模型启动脚本所在路径为/data/cann-recipes-infer/integration/vllm/ling-3.0-flash/
cd /data/cann-recipes-infer/integration/vllm/ling-3.0-flash/ && bash bailing_v3_start.sh && tail -f vllm.log
```

2、对话测试
```bash
bash bailing_v3_chat.sh
```

3、输出示例
```bash
Question 1：
"messages": [
      {"role": "system", "content": "你是一个乐于助人的助手。"},
      {"role": "user", "content": "请告诉我中国的首都是北京吗？"}
    ],
Answer：
message":{"role":"assistant","content":"是的，中国的首都是北京。","refusal":null,"annotations":null,"audio":null,"function_call":null,"reasoning":null}

Question 2：
messages": [
      {"role": "system", "content": "you are a helpful assistant"},
      {"role": "user", "content": "tell me what is the result of 3+3?"}
    ],
Answer：
"message":{"role":"assistant","content":"The result of 3 + 3 is **6**.","refusal":null,"annotations":null,"audio":null,"function_call":null,"reasoning":null}
```

## 性能测试

```bash
# 指定模型路径
export MODEL_PATH="/data/ling_flash_v3"

bash bailing_v3_perf_verify.sh && tail -f bench.log
```

## 项目结构

```
ling-3.0-flash/
├── README.md
├── bailing_v3_chat.sh                          # 对话测试脚本
├── bailing_v3_perf_verify.sh                   # 性能验证脚本
├── bailing_v3_start.sh                         # 一键启动脚本
└── npu_patch/
    ├── bailing_v3_monkey_patch.py              # monkey patch 入口
    ├── bailing_v3_vllm_ascend_cpp.patch        # vllm-ascend C++ 补丁
    ├── patch_bailing_v3.sh                     # 补丁适配脚本
    ├── bailing_v3_patches/                     # Python 补丁框架
    │   ├── __init__.py
    │   ├── new_files_adapter.py                # 新文件拷贝逻辑
    │   ├── patch_core.py                       # 补丁核心工具
    │   ├── vllm_ascend_patches.py              # vllm-ascend 差异应用
    │   ├── vllm_patches.py                     # vllm 差异应用
    │   ├── vllm_ascend_diffs/                  # vllm-ascend 差异文件 (28 个)
    │   └── vllm_diffs/                         # vllm 差异文件 (14 个)
    └── Incremental_adapter_files/              # 增量适配文件
        ├── vllm/
        │   └── model_executor/
        │       └── models/
        │           ├── bailing_moe_v3.py       # BailingMoeV3 模型实现
        │           └── bailing_moe_v3_mtp.py   # BailingMoeV3 MTP 模型
        └── vllm_ascend/
            └── ops/
                ├── bailing_moe_v3_kda.py       # KDA 算子入口
                ├── pypto/
                │   └── kda/
                │       ├── chunk_kda_impl.py       # chunk-kda 算子 (prefill)
                │       └── fused_recurrent_kda_impl.py  # fused-recurrent 算子 (decode)
                └── triton/
                    ├── fla/
                    │   └── op_kda.py           # FLA KDA 算子
                    └── kda/
                        ├── __init__.py
                        ├── chunk_delta_h.py    # chunk delta-h 计算
                        ├── cumsum.py           # 累加和
                        ├── fused_recurrent_kda.py  # recurrent KDA
                        ├── kda.py              # KDA 核心
                        ├── l2norm.py           # L2 归一化
                        ├── solve_tril.py       # 三角求解
                        └── utils.py            # 工具函数
```

## 项目参考
- [vLLM](https://github.com/vllm-project/vllm) - 高效 LLM 推理框架
- [vLLM-Ascend](https://github.com/vllm-project/vllm-ascend) - vLLM 昇腾适配

## License
本项目基于 Apache License 2.0 开源协议。部分代码改编自 [vLLM](https://github.com/vllm-project/vllm) 和 [flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention) 项目，具体版权信息见各源文件头部声明。
