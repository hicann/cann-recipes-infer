# Qwen3-MoE模型基于sglang框架在NPU实现低时延推理

## 概述
Qwen3-MoE模型是2025年开源的大语言模型，本样例基于sglang开源框架[qwen3_moe.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen3_moe.py)，完成Qwen3-235B-A22B模型在sglang框架上的适配优化。

## 支持的产品型号
<term>Atlas A3 系列产品</term>

## 环境准备

1. 拉取镜像
   ```shell
   docker pull quay.io/ascend/sglang:v0.5.7-cann8.3.rc2-a3
   ```
2. 创建容器
   ```shell
   # 请设置容器名称，例如 your_docker_name，镜像名称同上一步
   container_name=your_docker_name
   image_name=quay.io/ascend/sglang:main-cann8.3.rc2-a3

   # 执行docker run命令创建容器，可通过-v按需挂载宿主机目录至容器
   docker run -itd --shm-size=500g \
   --name ${container_name} \
   --net=host \
   --privileged=true \
   -u root \
   -w /data \
   --device=/dev/davinci_manager \
   --device=/dev/hisi_hdc \
   --device=/dev/devmm_svm \
   --entrypoint=bash \
   -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
   -v /usr/local/dcmi:/usr/local/dcmi \
   -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
   -v /etc/ascend_install.info:/etc/ascend_install.info \
   -v /usr/local/sbin/:/usr/local/sbin/ \
   -v /etc/hccn.conf:/etc/hccn.conf \
   -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
   -v /data/:/data/ \
   -v /tmp:/tmp \
   -v /etc/localtime:/etc/localtime \
   -v /var/log/npu/slog/slogd:/var/log/npu/slog/slogd \
   -v /dev/shm:/dev/shm \
   ${image_name}

   # 执行docker exec命令进入容器
   docker exec -it ${container_name} bash
   ```

## 权重准备
1. 下载[Qwen3-235B-A22B权重](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507/tree/main)
2. 权重int8量化
   1) 安装modelslim，可参考[安装指南](https://gitcode.com/Ascend/msit/blob/master/msmodelslim/docs/%E5%AE%89%E8%A3%85%E6%8C%87%E5%8D%97.md)
   2) 执行量化
      ```shell
      cd models/qwen3-moe-sglang/utils/
      # MODEL_PATH为原始权重下载后的存储路径，SAVE_PATH为权重量化后的存储路径
      msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu --model_type Qwen3-235B --config_path qwen3_moe_w8a8.yaml
      ```
   3) 删除权重中的offset
      ```shell
      python drop_offset.py --model_path ${SAVE_PATH}
      ```

## 代码准备
1. 下载本样例所在代码仓，以master分支为例
   ```shell
   git clone https://gitcode.com/cann/cann-recipes-infer.git
   ```

2. 获取sglang主仓源码，并应用patch
   ```shell
   git clone https://github.com/sgl-project/sglang.git
   cd sglang
   git reset --hard 2aec8b6e1b588f025ad5e25e2682a44b41a6cdbd

   # 将修改本仓中的修改patch应用到sglang代码中
   git am ../cann-recipes-infer/models/qwen3-moe-sglang/patches/*.diff
   ```

## 推理执行
1. 修改服务拉起脚本中推理执行需要相关配置。

   |部署方式|修改脚本|
   |-------|--------|
   |pd混部| `infer.sh`|
   |pd分离|`infer_prefill.sh`， `infer_decode.sh`， `infer_router.sh`|

   | 修改点 | 修改描述 | 涉及修改脚本 |
   |-------|--------|--------|
   |`IPs`| 各部署节点ip |`infer.sh`，`infer_prefill.sh`， `infer_decode.sh`|
   |`IFNAMES`| 各部署节点网卡 |`infer.sh`，`infer_prefill.sh`， `infer_decode.sh`|
   |`MODEL_PATH`| 模型权重存储路径 |`infer.sh`，`infer_prefill.sh`， `infer_decode.sh`|
   |`PYTHONPATH`|PYTHONPATH中增加打完patch后的sglang路径 |`infer.sh`，`infer_prefill.sh`， `infer_decode.sh`，`infer_router.sh`|
   |`Prefill_Master_Server_IP`| Prefill主节点ip |`infer_decode.sh`，`infer_router.sh`|
   |`Decode_Master_Server_IP`| Decode主节点ip |`infer_router.sh`|

2. 拉起服务
   ```shell
   # pd混部
   bash infer.sh
   ```
   ```shell
   # pd分离
   # prefill节点执行
   bash infer_prefill.sh

   # decode节点执行
   bash infer_decode.sh

   # router拉起，需要在prefill和decode节点服务拉起后执行
   bash infer_router.sh
   ```

## 测试方法
### 单请求精度验证
* 普通长度的序列可以通过`curl`的方式直接发送验证：
```sh
curl --location 'http://127.0.0.1:30002/generate' -H 'Content-Type: application/json' --data '{"text": ["1 + 1 = ?"], "sampling_params": { "temperature": 0, "max_new_tokens": 15}}'
```

### 基于数据集的精度验证
1. 下载数据集
   ```
   cd /data/
   git clone https://github.com/openai/grade-school-math.git
   ```
2. 通过以下命令可以执行few_shot_gsm8k进行精度验证，结果大于0.9即为精度正常：
   ```sh
   cd python/sglang/test
   python3 few_shot_gsm8k.py --parallel 64 --num-questions 200 --num-shots 5 --port 30002 --temperature 0 --data-path "/data/grade-school-math/grade-school-math/data/test.jsonl"
   ```

### 基于数据集的性能压测
1. 通过`bench_serving.sh`脚本，可以通过bench_serving指定B/S发送请求。
* 可先通过 https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json?download=true 下载数据集，并在脚本里指定DatasetJsonPath为对应json文件路径。
2. 如需采集profiling，拉起服务时配置环境变量`export ENABLE_PROFILER=True`。

## Benchmark
基于Atlas A3，本实践使用下述部署方式，使能优化点superkernel，专家强制负载均衡和ge图模式cache（优化点使能方式详见[sglang新增入参说明](#sglang新增入参说明)），对Qwen3MoE W8A8量化版本进行了性能Benchmark测试。
   | 基础模型             | 机器型号     | GBS | dp_size | tp_size | ep_size | max_prompt_length | max_response_length | 纯模型时延（ms） |
   |---------------------|-------------|-----|----------|---------| --------|-------------------|--------------------|-----------|
   | Qwen3-235B-A22B    | Atlas A3 32卡 | 256 | 16        | 4 | 64 |  5120             | 500      |  30   |

## 附录

### 新增环境变量说明
```shell
# 采集ge图模式执行时的profiling文件，采集到的profiling文件位于执行脚本同目录的`prof/`文件下
export ENABLE_PROFILER=True
```

### sglang新增入参说明
<a name="sglang新增入参说明"></a>
|入参|说明|
|-------|--------|
|--enable-superkernel| qwen3-moe模型使能superkernel特性 |
|--perfect-eplb| qwen3-moe模型使能专家强制负载均衡|
|--enable-cache-compile| 使能ge图模式缓存功能，缓存文件位于执行脚本同目录的`compile_cache/`文件下，如果执行的模型结构或部署方式有调整，需要手动删除`compile_cache/`文件|

### 文件说明
|文件路径|说明|
|-------|--------|
|[0001-feat-basic-modification-for-qwen3-moe.diff](patches/0001-feat-basic-modification-for-qwen3-moe.diff)|Qwen3MoE模型基础修改，修改内容包含deepep模块替换，atb算子替换，Qwen3MoE MTP模型适配，MTP场景下GQA attention分支适配，dp+tp混合并行场景bug修复，w8a8场景bug修复，模型中matmul类算子使能nz|
|[0002-feat-support-qwen3-moe-forced-eplb.diff](patches/0002-feat-support-qwen3-moe-forced-eplb.diff)|Qwen3MoE模型适配专家强制均衡|
|[0003-feat-npu-profiler.diff](patches/0003-feat-npu-profiler.diff)|适配npu profiler工具|
|[0004-feat-npu-support-main-model-ge-graph.diff](patches/0004-feat-npu-support-main-model-ge-graph.diff)|修改NPUGraphRunner，适配主模型ge图模式特性|
|[0005-feat-mtp1-support-ge-graph.diff](patches/0005-feat-mtp1-support-ge-graph.diff)|修改EAGLEDraftExtendNpuGraphRunner，适配MTP时第一个投的ge图模式特性|
|[0006-feat-mtp2-support-ge-graph.diff](patches/0006-feat-mtp2-support-ge-graph.diff)|改EAGLEDraftNpuGraphRunner，适配MTP时除第一个投外其他投的ge图模式特性|
|[0007-feat-modify-mtp-lm-head-load-method.diff](patches/0007-feat-modify-mtp-lm-head-load-method.diff)|适配EAGLE场景下Qwen3MoE模型可通过判断mtp权重是否含shared head权重决定是否和主模型共享一个lm head|
|[0008-fix-GQA-FIA-pd-disaggregation-bug.diff](patches/0008-fix-GQA-FIA-pd-disaggregation-bug.diff)|修复pd分离场景下使用fia分支时kv items len计算错误导致的精度问题|
|[0009-feat-support-superkernel.diff](patches/0009-feat-support-superkernel.diff)|Qwen3MoE模型使能superkernel特性|







