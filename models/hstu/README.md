# HSTU模型的NPU适配与优化
## HSTU模型介绍
Meta论文《Actions Speak Louder Than Words》提出生成式推荐（GR），把推荐视为生成建模问题，并引入HSTU架构以应对高基数、非平稳的数据流，同时统一支持检索与排序。基于HSTU的超大模型（1.5万亿参数）在线A/B指标提升12.4%，已在Meta多平台部署。

在本示例中，我们介绍HSTU的NPU推理实现。

## 支持的产品型号

本样例支持Atlas A2系列产品的单卡、多卡推理。

## 环境准备

### 镜像准备

本样例基于 RecSDK-Torch 环境运行，并提供 Docker 镜像，便于轻松部署与快速完成环境配置。

1. 首先下载对应[rec_sdk-torch](https://www.hiascend.com/developer/ascendhub/detail/9faeb4847b3e419f81b78a4d0ed574b5) Docker镜像；

2. 使用下面命令拉起一个镜像容器：

```bash
docker run -u root -itd --name rec_gr --ulimit nproc=65535:65535 --ipc=host \
    --device=/dev/davinci0     --device=/dev/davinci1 \
    --device=/dev/davinci2     --device=/dev/davinci3 \
    --device=/dev/davinci4     --device=/dev/davinci5 \
    --device=/dev/davinci6     --device=/dev/davinci7 \
    --device=/dev/davinci_manager --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /home/:/home \
    -v /data:/data \
    -v /etc/localtime:/etc/localtime \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /etc/ascend_install.info:/etc/ascend_install.info -v /var/log/npu/:/usr/slog \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
    -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/sbin:/usr/local/sbin \
    -v /etc/hccn.conf:/etc/hccn.conf -v /root/.pip:/root/.pip -v /etc/hosts:/etc/hosts \
    -v /usr/bin/hostname:/usr/bin/hostname \
    --net=host \
    --shm-size=128g \
    --privileged \
    $REPOSITORY:TAG \ # 通过docker images命令查看REPOSITORY和TAG；如swr.cn-south-1.myhuaweicloud.com/ascendhub/rec_sdk-torch:openeuler2203-arm
    /bin/bash
```
其中`-v /data:/data`用于映射代码文件、权重文件和数据集文件。

运行了docker容器后，使用命令`docker exec it rec_gr bash`进入容器内部，设置以下环境变量：

```bash
export PATH=/usr/local/python3.11.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/python3.11.0/lib:$LD_LIBRARY_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 代码准备
本仓库依赖于recsys-examples的开源仓代码。

1. 克隆cann-recipes推理仓：

```bash
mkdir -p /data/code
cd /data/code
git clone https://gitcode.com/cann/cann-recipes-infer.git
```

2. 克隆recsys-examples仓获取所需文件：

```bash
git clone --branch v25.11 --depth 1 https://github.com/NVIDIA/recsys-examples.git
# 非覆盖模式替换文件
cp -an recsys-examples/examples/hstu/* cann-recipes-infer/models/hstu/
```

3. 应用patch修改：

```bash
cd cann-recipes-infer/models/hstu
git apply hstu_cann.patch
```

### 相关依赖安装

Docker容器内需要安装nnal加速库，以及RecSDK算子组件，安装方式如下。

1. nnal加速库安装

本样例KV Cache管理功能依赖NPU算子`_npu_reshape_and_cache`，使用该算子需要安装nnal加速库。
从昇腾官方社区下载[Ascend-cann-nnal_8.2.RC1_linux-aarch64.run](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.2.RC1)包，并在docker环境中使用`bash Ascend-cann-nnal_8.2.RC1_linux-aarch64.run --install`命令安装，安装完成后设置环境变量：

```bash
source /usr/local/Ascend/nnal/atb/set_env.sh
```

2. RecSDK算子库安装

从[RecSDK](https://gitcode.com/Ascend/RecSDK)库下载适配代码：

```bash
cd cann-recipes-infer/models/hstu/
git clone https://gitcode.com/Ascend/RecSDK.git
```

部分算子依赖外部组件，编译前请将组件 [json-3.9.1.tar.gz](https://github.com/nlohmann/json/archive/v3.9.1.tar.gz) 下载并重命名为v3.9.1.tar.gz后放置于`RecSDK/cust_op/ascendc_op/build/scripts/onnx_plugin`目录。

同时，`in_linear_silu`算子依赖CATLASS源码，验证过的版本是[catlass v1.3.0](https://raw.gitcode.com/cann/catlass/archive/refs/heads/v1.3.0.zip)，点击链接下载压缩包并解压。

设置如下环境变量：
```bash
export CATLASS_HOME=<catlass_home>
```

随后执行`build_install_ops.sh`脚本一键完成所需算子的编译以及适配层安装：

```bash
chmod +x build_install_ops.sh
bash build_install_ops.sh A2 ./RecSDK
```

3. 其他库安装

最后执行下面命令安装和导入相关库：

```bash
pip uninstall torchrec
pip install torchrec==1.1.0
pip install rich einops
export LIB_FBGEMM_NPU_API_SO_PATH=/usr/local/python3.11.0/lib/python3.11/site-packages/libfbgemm_npu_api.so
```

### 模型推理
1. 下载并处理数据集

使用下面命令可以下载和处理数据集：

```bash
python3 ./preprocessor.py --dataset_name "kuairand-1k" --inference
```

2. 执行推理脚本

使用以下命令进行推理：

```bash
# eval 模式
python3 ./inference/inference_gr_ranking.py --gin_config_file ./inference/configs/kuairand_1k_inference_ranking.gin --mode eval
# eval 模式 profiling 采集
python3 ./inference/inference_gr_ranking.py --gin_config_file ./inference/configs/kuairand_1k_inference_ranking.gin --mode eval --enable_profiler
# simulate 模式
python3 ./inference/inference_gr_ranking.py --gin_config_file ./inference/configs/kuairand_1k_inference_ranking.gin --mode simulate
```

分布式推理（支持单机多卡推理，以两卡为例）：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:6000 --nnodes=1 --nproc-per-node=2 ./inference/inference_gr_ranking.py --gin_config_file ./inference/configs/kuairand_1k_inference_ranking.gin --embed_tp_size 2 --mode eval
```

## 优化点介绍

我们基于recsys-examples库，针对NPU做了如下的适配：
1. TensorRT-LLM库KV Cache管理impl方法的NPU适配替换；
2. 使用torch小算子替换Triton和CUDA算子；

并基于上述适配使能了下面的优化点：
1. KVCache多级缓存、onload和offload流程适配与优化；
2. 高性能融合算子替换（`hstu_paged`）；
3. ACLGraph使能；
4. KV Cache layout优化，引入`_npu_reshape_and_cache`算子；
5. 分布式推理功能，在Embedding Table占用显存大的情况下，支持更大的dense参数模型推理。