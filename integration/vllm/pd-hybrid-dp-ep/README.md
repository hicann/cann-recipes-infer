# 基于 vLLM 支持 A2 四机环境下进行 PD 混合（PD Hybrid）模式的 MoE 模型 DP+EP 推理

## 背景

针对 DeepSeek-R1 等 MoE 模型的高性能推理需求，根据不同规模的部署形态，需要采取差异化的优化策略。在 A2 四机 32 卡部署规模场景中，经评估 PD 混合（PD Hybrid）优于 PD 分离。因此，本项目基于 vLLM 进行能力拓展并提交了 patch，主要实现了 A2 四机环境下 PD 混合（PD Hybrid）模式的 MoE 模型 DP+EP 推理支持。

为了支持在 A2 四机环境下进行 PD 混合模式的 MoE 模型 DP+EP 推理部署，提交对 dummy_run 的执行逻辑进行了改进，新增了 prefill 与 decode 阶段的区分逻辑，以便匹配各阶段所需的通信算子。改动基于 OmniInfer v0.5.0 分支进行开发。

## 简述

- 配置：
  -  四机 32 卡 910B，运行 DeepSeek-R1 等 MoE 模型。
- 提交内容：
  -   含运行流程、性能数据、模型、可复现镜像环境的文档（模型和镜像存储于外部服务）。
- 功能增量：
  -   增加对 PD 混合部署的支持；
  -   增加对 Qwen3 系列 MoE 模型的支持。
- 性能效果（DeepSeek-R1）：
  -   全局并发 1024、输入长度 128、输出长度 1024 时，用户视角单卡 TPS 达到 303.67；
  -   全局并发 1024、输入长度 1024、输出长度 1024 时，用户视角单卡 TPS 达到 204.16。

## 功能改动

### 1. PD 混合调度策略

提交实现了 prefill / decode 混合调度策略，并确保兼容原本的 PD 分离部署，所有替换逻辑都仅在设置环境变量 OMNI_PD_HYBRID 时生效，默认情况下不会影响现有功能。

可选的选项有：

- 启用 PD Hybrid (OMNI_PD_HYBRID=1)
  - 适用于单 API Server 场景。
  - 通过全局阶段协调机制自动调度 prefill 与 decode 请求，提高多进程并行执行效率和资源利用率。

- 禁用 PD Hybrid (OMNI_PD_HYBRID=0，默认)
  - 适用于手动扩展多个 API Server 的场景。
  - 保持现有执行逻辑不变，无兼容性影响。

### 2. 引擎客户端选择 (async_llm.py)

当启用 OMNI_PD_HYBRID 时，API Server 使用 DPAsyncMPClient 以支持多进程异步请求分发：

- OmniInfer 原本依靠推理引擎外的 Nginx 将请求路由至各进程，与部署方式较为耦合。为了适配 PD 混合调度策略，新增了推理引擎内单 API server 控制多进程的逻辑。即当启用 OMNI_PD_HYBRID 时，采用 DPAsyncMPClient 引擎客户端替换 core_client。
  
### 3. 全局阶段协调 (core.py)

- 新增 _get_global_phase_hint() 方法，通过 DP all-reduce 在进程间同步 prefill / decode 状态，确保相互进行 MoE 通信的进程同属 prefill 或 decode。

- 将 DPEngineCoreProc.run_busy_loop() 替换为新的调度逻辑，用于根据当前执行阶段动态分配计算与 dummy 执行：
  - 有 prefill 请求的进程执行实际计算，其他进程执行 dummy prefill。
  - 有 decode 请求未完成的进程执行实际计算，其他进程执行 dummy decode。

- 扩展 execute_dummy_batch() 与 _dummy_run()，支持通过 phase 参数指定执行阶段，便于精确控制推理行为，在请求数不整除进程数时避免通信不一致。
  
### 4. 分离调度 (scheduler.py)

使用 prefill / decode 分离调度策略：当有新的 prefill 任务时，优先仅执行 prefill 阶段，不再与 decode 混合，避免通信阶段冲突。
  
### 5. Qwen3-MoE 支持

为支持 Qwen3-MoE 在多机 DP 模式下的推理，将原本不支持多机的 all_gather_v 算子替换为通过 padding + two-stage all_gather 的同步方式，实现 hidden 在 DP 维度上的对齐与同步。脚本中更改模型路径，将 graph_model_compile_config level 设置为 0 即可运行。

## 部署流程

### 运行平台

A2 4 机 32 卡

### 模型文件

1. qingcheng-ai/DeepSeek-R1-QC-INT8
```
https://www.modelscope.cn/models/qingcheng-ai/DeepSeek-R1-QC-INT8
```

2. Qwen/Qwen3-235B-A22B
```
https://huggingface.co/Qwen/Qwen3-235B-A22B
```

### 环境准备

#### 方法一：使用 OmniInfer 官方 Docker 镜像 + 原版代码 + 补丁进行安装

1. 准备 omniinfer 及 vllm 代码

下载 omniinfer release_v0.5.0 版本代码、vllm v0.9.0 版本代码并应用相关补丁
```
git clone --branch release_v0.5.0 --single-branch --depth 1 https://gitee.com/omniai/omniinfer.git
cd omniinfer
git apply omni_v0.5.0.patch

cd infer_engines
git clone --branch v0.9.0 --single-branch --depth 1 https://gitee.com/mirrors/vllm.git
chmod a+x bash_install_code.sh
./bash_install_code.sh
```

2. 启动容器

以 swr.cn-east-4.myhuaweicloud.com/omni/omni_infer-a2-arm:release_v0.5.0 作为基础镜像，并根据实际环境调整脚本中的 omniinfer 挂载配置。
```
#!/bin/bash

IMAGE="swr.cn-east-4.myhuaweicloud.com/omni/omni_infer-a2-arm:release_v0.5.0"

DEV_SETS=(
    --device=/dev/davinci_manager
    --device=/dev/hisi_hdc
    --device=/dev/devmm_svm
    -v /usr/local/dcmi:/usr/local/dcmi
    -v /usr/local/sbin:/usr/local/sbin
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver
    -v /etc/hccn.conf:/etc/hccn.conf
    -v /etc/ascend_install.info:/etc/ascend_install.info
    -v /usr/bin/hccn_tool:/usr/bin/hccn_tool
    -v /tmp:/tmp
)

USER_SETS=(
    -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime
    -v /xx/omniinfer:/workspace/omniinfer
)

docker run -it -d --shm-size=500g \
    --privileged \
    --ipc host \
    --network host \
    --env TERM="$TERM" \
    --entrypoint=bash \
    "${DEV_SETS[@]}" \
    "${USER_SETS[@]}" \
    $IMAGE
```

3. 重新安装

在容器中运行 reinstall_vllm_omni.sh 脚本，重装 omniinfer 和 vllm，覆盖镜像中原安装的版本。
```
cd /workspace/omniinfer
bash tools/scripts/reinstall_vllm_omni.sh
```

4. 启动服务

运行 run_ds_w8a8_a2_hybrid.sh 脚本用于快速启动：
- 修改脚本开头的机器 hostname、master_ip 以及部分 socket 网卡配置，在 A2 四机上分别执行。
```
bash tools/scripts/run_ds_w8a8_a2_hybrid.sh
```

#### 方法二：使用打包好代码的 Docker 镜像进行安装

1. 拉取镜像
```
docker pull qingcheng-ai-cn-beijing.cr.volces.com/public/omni_infer-a2-arm-qc:release_v0.5.0_dev
```

2. 启动容器（启动命令同方法一）

3. 启动服务

在容器内配置环境变量并运行 run_ds_w8a8_a2_hybrid.sh 脚本，并在 A2 四机上分别执行。
```
export MACHINE1_HOSTNAME=<machine1_hostname>
export MACHINE2_HOSTNAME=<machine2_hostname>
export MACHINE3_HOSTNAME=<machine3_hostname>
export MACHINE4_HOSTNAME=<machine4_hostname>
export MACHINE1_IP=<machine1_ip>
export LOCAL_IP=<local_ip>
export MODEL_PATH=<model_path>
export COMPILE_LEVEL=<compile_level> # 1 when running DeepSeek-R1-QC-INT8, 0 when running Qwen3-235B-A22B

bash omniinfer/tools/scripts/run_ds_w8a8_a2_hybrid.sh
```

## 测试验证

精度及 dummy_run 功能验证：
```
curl -X POST "http://127.0.0.1:8000/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/data1/models/Deepseek-R1-ascend-int8-osp",
    "temperature": 0,
    "max_tokens": 256,
    "prompt": "宫保鸡丁怎么做？"
  }'
```

性能测试：
```
python3 ../benchmarks/benchmark_serving.py \
    --backend vllm \
    --model $MODEL_PATH \
    --dataset-name random \
    --trust-remote-code \
    --ignore-eos \
    --random-input-len "$INPUT_LEN" \
    --random-output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_REQS"
```

global batch size 为 1024、输入长度为 128、输出长度为 1024 ，单卡吞吐率（TPS）达到 303.67。
```
============ Serving Benchmark Result ============
Successful requests:                     1024
Benchmark duration (s):                  107.91
Total input tokens:                      130048
Total generated tokens:                  1048576
Request throughput (req/s):              9.49
Output token throughput (tok/s):         9717.49
Total Token throughput (tok/s):          10922.68
---------------Time to First Token----------------
Mean TTFT (ms):                          8393.18
Median TTFT (ms):                        8213.30
P99 TTFT (ms):                           9580.35
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          96.92
Median TPOT (ms):                        97.16
P99 TPOT (ms):                           97.28
---------------Inter-token Latency----------------
Mean ITL (ms):                           109.55
Median ITL (ms):                         100.76
P99 ITL (ms):                            255.84
==================================================
```

global batch size 为 1024、输入长度为 1024、输出长度为 1024 ，单卡吞吐率（TPS）达到 204.16。
```
============ Serving Benchmark Result ============
Successful requests:                     1024
Benchmark duration (s):                  160.50
Total input tokens:                      1047552
Total generated tokens:                  1048576
Request throughput (req/s):              6.38
Output token throughput (tok/s):         6533.26
Total Token throughput (tok/s):          13060.14
---------------Time to First Token----------------
Mean TTFT (ms):                          31617.38
Median TTFT (ms):                        28835.88
P99 TTFT (ms):                           51890.32
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          125.56
Median TPOT (ms):                        128.27
P99 TPOT (ms):                           145.59
---------------Inter-token Latency----------------
Mean ITL (ms):                           134.36
Median ITL (ms):                         102.23
P99 ITL (ms):                            288.79
==================================================
```
