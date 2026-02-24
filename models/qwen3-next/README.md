# Qwen3-next SGLang优化实践样例

## 概述

Qwen3-next模型是2025.9推出的混合注意力开源大语言模型，本样例针对Qwen3-next模型，基于[SGLang开源框架](https://github.com/sgl-project/sglang)，完成模型推理部署的优化适配。

本项目基于NPU主要完成了以下优化特性，具体内容介绍可参见[基于SGLang&A3集群的Qwen3-next模型推理部署优化实践](../../docs/models/qwen3-next/qwen3_next_optimization.md)。

- 支持MTP1部署
- 支持AscendC融合算子npu_recurrent_gated_delta_rule
- 支持PTO 融合算子
- 支持线性Attention/GatedAttention的SP（Sequence Parallel）序列并行部署策略
- 支持W8A8C8量化

## 硬件要求
产品型号：Atlas A3 系列

## 基于Docker构建环境
1. 创建SGLang镜像。

   ```bash
   # 镜像下载
   docker pull quay.io/ascend/sglang:main-cann8.3.rc2-a3
   
   # 执行以下脚本创建容器，请传入容器名称，如 your_docker_name
   docker run -u root -itd --name your_docker_name --ulimit nproc=65535:65535 --ipc=host --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 --device=/dev/davinci8 --device=/dev/davinci9 --device=/dev/davinci10 --device=/dev/davinci11 --device=/dev/davinci12 --device=/dev/davinci13 --device=/dev/davinci14 --device=/dev/davinci15 --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc -v /etc/localtime:/etc/localtime -v /home/:/home/ -v /data/:/data/ -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /etc/ascend_install.info:/etc/ascend_install.info -v /var/log/npu:/usr/slog -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi -v /sys/fs/cgroup:/sys/fs/cgroup:ro -v /usr/local/dcmi:/usr/local/dcmi -v /etc/hccn.conf:/etc/hccn.conf -v /root/.pip/pip.conf:/root/.pip/pip.conf -v /etc/hosts:/etc/hosts --net=host --shm-size=128g --privileged quay.io/ascend/sglang:main-cann8.3.rc2-a3 /bin/bash
   
   # 执行docker exec命令进入容器
   docker exec -it -u root your_docker_name bash
  
   ```
   
2. 在容器中安装CANN软件包与Ascend Extension for PyTorch软件包。
   
   为使能GDN融合算子，需要先将镜像中的CANN版本更新到8.5.0。
   - **清理旧版CANN**
     ```bash
     cd /usr/local/Ascend
     rm -rf 8.5.0/ ascend-toolkit/ latest/ nnal/
     ```
   - **安装CANN：8.5.0**

      请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann)下载如下软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=Debian)进行安装。

      - 开发套件包：`Ascend-cann-toolkit_${version}_linux-${arch}.run`
      - 二进制算子包：`Ascend-cann-A3-ops_${version}_linux-${arch}.run`
      - NNAL加速包：`Ascend-cann-nnal_${version}_linux-${arch}.run`

      软件包文件名中 `${version}` 表示CANN包版本号，`${arch}` 表示CPU架构（如aarch64、x86_64）。
   
      > 请参考[版本兼容性说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/releasenote/releasenote_0000.html)确认HDK版本。为兼容pd分离特性，推荐版本为25.2.x。
      
   - **创建latest软连接**
        ```bash
        ln -s /usr/local/Ascend/cann /usr/local/Ascend/latest
        ```

   - **安装Ascend Extension for PyTorch：7.3.0**

      Ascend Extension for PyTorch（torch_npu）为支撑PyTorch框架运行在NPU上的适配插件，本样例支持的Ascend Extension for PyTorch版本为`7.3.0`，PyTorch版本为`2.6.0`。

      请参考[Ascend Extension for PyTorch安装文档](https://www.hiascend.com/document/detail/zh/Pytorch/730/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md)安装相应版本的torch_npu插件。

3. 下载项目源码并安装依赖的python库。
    ```bash
    # 下载项目源码，以master分支为例
    git clone https://gitcode.com/cann/cann-recipes-infer.git
    ```
    
    
4. 下载依赖的开源框架代码，加载patch。

   为了让使用者和开发者直观了解我们基于开源代码做的修改，本样例中只包含patch代码，其他框架代码需要拉取。

   返回cann-recipes-infer项目代码上级目录，即执行git clone命令时所在目录，并执行如下命令，需注意，请确保环境能够正常连通网络。
   ```bash
   # 返回cann-recipes-infer项目代码上级目录
   # 下载sglang源码
   git clone https://github.com/sgl-project/sglang.git -b v0.5.6
   ```
   patches目录下有四个对应不同特性更改的patch组，对应关系如下：
    - stage1：包含基础功能适配和MoE多流优化。
    - stage2：包含W8A8C8动态量化。
    - stage3：包含GDN和MTP适配。
    - stage4：包含CP并行。

   按顺序加载patches目录下的四个patch组：

   ```bash
   cd sglang
   
   bash ../cann-recipes-infer/models/qwen3-next/apply_patches.sh   
   ```

   安装sglang：
   
	```bash
	pip install -e python --no-deps --no-build-isolation
	```

    
5. 安装最新Triton-Ascend

	Triton-Ascend是CANN生态对Triton的支持库，请参考[Triton-Ascend安装指南](https://ascend.github.io/triton-ascend/sources/getting-started/installation.html)安装最新Triton-Ascend：
    ```bash
    pip install triton-ascend --force-reinstall
    ```
    
6. 安装其他依赖
    ```bash
    pip install torchvision==0.21.0
    pip install torchao==0.9.0
    ```

## 模型权重准备
本样例使用的Qwen3-next模型权重准备方法如下：
```bash
# 从魔搭社区下载Qwen3-Next完整BF16权重至指定目录，例如 your_bf16_weights
pip install modelscope
modelscope download --model Qwen/Qwen3-Next-80B-A3B-Instruct --local_dir your_bf16_weights

# 将BF16权重转换为W8A8权重（可选）
python python/sglang/srt/utils/convert_model_qwen3_next.py \
   --input_bf16_hf_path your_bf16_weights \
   --output_hf_path your_w8a8_weights \

# 将BF16权重转换为W8A8C8权重（可选）
# 下载quant_param到指定路径，比如 your_param_path
wget --no-check-certificate -P your_param_path https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-quantization/Qwen3-next/attn_c8_scale.zip
# 解压quant_param到指定路径，比如 your_param_path
apt update
apt install unzip
unzip -o  your_param_path/attn_c8_scale.zip -d your_param_path
# 转换权重
python python/sglang/srt/utils/convert_model_qwen3_next.py \
   --input_bf16_hf_path your_bf16_weights \
   --output_hf_path your_w8a8c8_weights \
   --c8 --quant_param_path your_param_path/attn_c8_scale \
```
## 推理执行

在本样例代码根目录下启动Qwen3-next的推理执行。
### 前置环境配置
修改`set_env.sh`中推理执行需要相关配置。

   | 修改点 | 修改描述 |
|-------|--------|
   |`SOCKET_IFNAME`| 各部署节点网卡 |
   |`MODEL_PATH`| 模型权重存储路径 |
   |`PYTHONPATH`|PYTHONPATH中增加打完patch后的sglang路径 |
   |`IP_NODE_P0`| Prefill主节点ip |
   |`IP_NODE_D0`| Decode主节点ip |

### PD混部命令
使用`infer.sh`启动SGLang服务，以下是对相关服务拉起参数的说明：

| 配置项                        | 说明                                              |
|----------------------------|-------------------------------------------------|
   | `--tp 16`                      | TP并行数（若开启DP，实际TP并行数为此配置参数/DP配置参数）               |
   | `--enable-dp-attention --dp-size 8`                  | DP并行数配置，若需禁用DP请删除这两项配置                          |
   | `--moe-a2a-backend deepep --deepep-mode auto`               | 启用deepep，若需禁用deepep请删除这两项配置                     |
   | `--cuda-graph-bs 16`               | 计算图batch size，使用单算子模式请替换为`--disable-cuda-graph` |


### PD分离命令
PD分离部署需要分别启动Prefill+Decode+Router节点，
1. 修改各个节点`set_env.sh`中的参数

```sh
#修改prefill节点和decode节点的主节点ip
IP_NODE_P0=x.x.x.x   # 修改为prefill主节点ip
IP_NODE_D0=y.y.y.y   # 修改为decode主节点ip

# prefill节点和decode节点分别设置各自的nnodes和node_rank,比如 2prefill + 2decode
# p0
export nnodes=2
export node_rank=0
# p1
export nnodes=2
export node_rank=1
# d0
export nnodes=2
export node_rank=0
# d1
export nnodes=2
export node_rank=1
```

2. 在Prefill节点运行`infer_prefill.sh`

3. 在Decode节点运行`infer_decode.sh`

4. 在Prefill节点启动Router：
```sh
source set_env.sh
python3 -m sglang_router.launch_router --decode http://${d0}:30001 --prefill http://${p0}:30001 --pd-disaggregation --mini-lb --host 0.0.0.0 --port 30002
```

### 新增特性使能

| 特性                       | 使能方式说明                                                                                                                                                                                                                                                                                                                        |
|--------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ACLGraph                 | 服务拉起参数中配置`--cuda-graph-bs`                                                                                                                                                                                                                                                                                                    |
| MoE多流                    | `set_env.sh`中增加以下环境变量配置：`export ENABLE_NPU_DEEPEP_MOE_MULTI_STREAM=1 `                                                                                                                                                                                                                                                        |
| GDN（Gated Delta Net）融合算子 | `set_env.sh`中增加以下环境变量配置：`export ENABLE_ASCENDC_FUSION_GDN="true"`                                                                                                                                                                                                                                                             |
   | MTP(多Token预测)            | 1. `set_env.sh`中增加以下环境变量配置：`export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1` `export SGLANG_ENABLE_SPEC_V2=1`<br/>2. 增加以下服务拉起的配置项`--speculative-algorithm NEXTN --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 \`<br/>3. 确保`set_env.sh`中使能GDN融合算子`export ENABLE_ASCENDC_FUSION_GDN="true" ` |
   | A8W8量化                   | 1. `set_env.sh`中修改deepep环境变量配置，将`export SGLANG_DEEPEP_BF16_DISPATCH=1`替换为`export DEEP_NORMAL_MODE_USE_INT8_QUANT=1`<br/>2. 增加以下服务拉起的配置项`--quantization w8a8_int8`<br/>3. `set_env.sh`中配置A8W8权重路径                                                                                                                              |
   | A8W8C8量化                 | 1. `set_env.sh`中修改deepep环境变量配置，将`export SGLANG_DEEPEP_BF16_DISPATCH=1`替换为`export DEEP_NORMAL_MODE_USE_INT8_QUANT=1`<br/>2. `set_env.sh`中增加以下环境变量配置：`export ASCEND_USE_C8=1`<br/>3. 增加以下服务拉起的配置项`--quantization w8a8_int8`<br/>4. `set_env.sh`中配置A8W8C8权重路径                                                                    |
   | 序列并行                 | 1. 本样例支持PD分离场景下Prefill节点Gated Attention的CP和TP混合并行与CP的负载均衡，启用方式：在`infer_prefill.sh`中将DP配置`--enable-dp-attention --dp-size 8`替换为`--cp-size 8`<br/>2. `set_env.sh`中增加以下环境变量配置开启CP负载均衡：`export CP_USE_ZIGZAG=1`，负载均衡的优化面向长序列单batch场景。                                                                                             |

### 规格约束说明：
1. PD分离场景P和D部署策略需相同（SGLang框架暂未支持MHA(Multi-Head Attention)PD不同策略部署）；
2. MTP暂不支持和图模式同时开启；
3. MTP暂不支持和C8同时开启；
4. 暂不支持PD分离下开启MTP；
5. CP仅支持PD分离场景开启，启用时P和D部署策略可不同（P可增加cp_size配置），需cp_size<tp_size，需要PD卡数相同（SGLang框架kv传输逻辑限制），CP不支持与DP同时开启（CP复用了DP通信域）；
6. 请确保chunk prefill size <= 71680（triton算子的规格约束）；
7. GDN切分头数存在限制，tp_size需不大于32。

## 测试方法
### 单请求精度验证
* 普通长度的序列可以通过`curl`的方式直接发送验证：
```sh
curl --location 'http://127.0.0.1:30002/generate' -H 'Content-Type: application/json' --data '{"text": ["1 + 1 = ?"], "sampling_params": { "temperature": 0, "max_new_tokens": 15}}'
```
* sglang框架图模式存在精度问题，请在单算子模式下验证精度。
* 若服务拉起配置了`--skip-server-warmup`，请在验证精度前发送dp_size个请求，保证每个dp_rank都预热到。
* 长序列可以通过以下`send_long_text.py`构造发送，请将脚本内的`TXT_PATH`配置为要发送的文本。

### 基于数据集的精度验证
通过以下命令可以执行few_shot_gsm8k进行精度验证，结果大于0.9即为精度正常：
```sh
cd python/sglang/test
python3 few_shot_gsm8k.py --parallel 16 --num-questions 100 --num-shots 5 --port 30002 --temperature 0
```

### 指定B/S的随机请求验证
通过`bench_serving.sh`脚本，可以通过bench_serving指定B/S发送请求。
* 可先通过 https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json?download=true 下载数据集，并在脚本里指定DatasetJsonPath为对应json文件路径。

# Benchmark
基于Atlas A3，本实践使用下述部署方式，使能优化点ACLGraph、A8W8C8量化、MoE多流、GDN融合算子，对Qwen3Next本进行了性能Benchmark测试。  

| 基础模型             | 机器型号     | graph_bs | dp_size | tp_size | ep_size | max_prompt_length | num_prompts | 纯模型decode时延（ms） |
   |---------------------|-------------|-----|----------|---------| --------|-------------------|--------------------|-----------|
   | Qwen3-Next-80B-A3B-Instruct-A8W8C8    | Atlas A3 16die | 16 | 8        | 2 | 16 |  256k             | 16      |  20.6   |
# 性能采集
可通过以下方式，加载profiling相关的patch修改，使用torch_npu.profiler采集性能数据。
## 加载patch
在sglang目录下加载PROFILE.patch:
```bash
git apply ../cann-recipes-infer/models/qwen3-next/patches/PROFILE.patch
```
## profiler配置
1. 通过设置环境变量`PROFILER_MODE`为`[all, decode, prefill]`中的指定值选择profiler的范围。

    以decode为例：
    ```bash
    export PROFILER_MODE='decode'
   ```
2. 配置`SGLANG_TORCH_PROFILER_DIR`指定profiler的保存路径：
    ```bash
    export SGLANG_TORCH_PROFILER_DIR='/home/sglang/prof/'
   ```
3. 修改`python/sglang/srt/model_executor/model_runner.py`中的schedule，控制采集范围。

    torch_npu.profiler的接口与torch原生profiler类似，具体使用方法可参考[昇腾社区](https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000851.html)。
    默认配置下，采集step4-step14的profiling数据。
## 运行采集
1. 使用要采集的配置拉起服务，若需精准控制采集的step，请确保配置了`--skip-server-warmup`参数跳过服务启动时的warmup。
2. 可使用[指定B/S的随机请求验证](#指定bs的随机请求验证)发送指定请求，若需精准控制采集的step，在使用bench_serving时添加`--warmup-requests 0`可跳过warmup请求。
3. 采集结束后，profiling数据保存在`SGLANG_TORCH_PROFILER_DIR`指定的目录下。
## 查看Trace图
可以通过`ASCEND_PROFILER_OUTPUT`目录下的`trace_view.json`文件查看算子运行的trace图。

推荐使用昇腾官方可视化工具MindStudio Insight查看trace图，具体下载和使用方式请参考[MindStudio Insight工具](https://www.hiascend.com/document/detail/zh/mindstudio/830/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html)。
