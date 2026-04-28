# DeepSeek-V4 Inference on NPU
## 概述
DeepSeek团队发布了最新的模型DeepSeek-V4，本实践基于DeepSeek开源代码进行迁移，并在CANN平台上完成性能优化，支持在昇腾`Atlas A3 Pod`平台和`950PR/DT`平台部署。

- 本实践的优化特性和性能Benchmark可参见[NPU DeepSeek-V4推理优化实践](../../docs/models/deepseek-v4/deepseek_v4_inference_guide.md)。

---

## 硬件要求
产品型号：Atlas A3 Pod 系列

操作系统：Linux ARM

镜像版本：cann9.0_pt2.8.0_ds_aarch_image:v1.0

驱动版本：Ascend HDK 25.5.1
> npu-smi info 检查Ascend NPU固件和驱动是否正确安装。如果已安装，通过命令`npu-smi info`确认版本是否为`25.5.1`。如果未安装或者版本不是`25.5.1`，请先下载[固件和驱动包](https://www.hiascend.com/hardware/firmware-drivers/community?product=7&model=33&cann=9.0.0-beta.2&driver=Ascend+HDK+25.5.1)，并根据[指导](https://hiascend.com/document/redirect/CannCommunityInstSoftware)自行安装。


## 快速启动

### 下载源码

  在各个节点上执行如下命令下载 cann-recipes-infer 源码。
  ```shell
  mkdir -p /home/code; cd /home/code/
  git clone https://gitcode.com/cann/cann-recipes-infer.git
  cd cann-recipes-infer
  ```
### 下载数据集
  从[链接](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/blob/main/longbook_qa_eng.jsonl)中下载长序列输入数据集longbook_qa_eng，并上传到各个节点上新建的路径`dataset/InfiniteBench`下。
  ```shell
  mkdir -p dataset/InfiniteBench
  ```

### 下载权重

  下载[DeepSeek-V4-Flash原始Hybrid FP8-MXFP4权重](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)或[DeepSeek-V4-Pro原始Hybrid FP8-MXFP4权重](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)，并上传到各节点的某个固定的路径下，比如`/data/models/deepseek_v4_hybrid_fp8_mxfp4`。

### 获取 docker 镜像

  从[ARM镜像地址](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-quantization/DeepSeek/cann9.0_pt2.8.0_ds_aarch_image_v1.0.tar)中下载 docker 镜像，然后上传到A3服务器的每个节点上，并通过命令导入镜像 `docker load -i cann9.0_pt2.8.0_ds_aarch_image_v1.0.tar`。

### 拉起 docker 容器

  在各个节点上通过如下脚本拉起容器，默认容器名为 cann_recipes_infer。注意：需要将权重路径和源码路径挂载到容器里。
  ```
  # A3 容器拉起脚本
  docker run -u root -itd --name cann_recipes_infer --ulimit nproc=65535:65535 --ipc=host \
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
      cann9.0_pt2.8.0_ds_aarch_image:v1.0 /bin/bash
  ```
  在各个节点上通过如下命令进入容器：
  ```
  docker attach cann_recipes_infer
  cd /home/code/cann-recipes-infer/models/deepseek-v4
  ```

### 转换权重中的config.json
  使用原生`Hybrid FP8-MXFP4版本权重`执行推理时需要执行这一步骤，其他场景跳过该步骤。需要进入容器并在各个节点上使用`utils/convert_config.py` 脚本完成权重路径下的config.json转换。

  **注意：** 该步骤不会对权重进行任何处理，仅将新生成的config.json覆盖原始config.json，如需保留原始config.json，请自行备份

  如果权重config.json转换的运行环境为NPU，需要先执行：

  ```shell
  cann_path=/usr/local/Ascend/cann  # cann包安装路径
  source ${cann_path}/bin/setenv.bash
  ```

  >入参介绍：`input_fp8_hf_path`：原始权重路径；

  拉起示例：

  ```shell
  python utils/convert_config.py --input_fp8_hf_path /data/models/deepseek_v4
  ```

### 转换权重

 原生`Hybrid FP8-MXFP4权重`执行推理时可跳过这一步骤，若需要使用 INT8 或 Hybrid MXFP8-MXFP4 权重执行推理，需要进入容器并在各个节点上使用`utils/convert_model.py` 脚本完成 Hybrid FP8-MXFP4 到 INT8/Hybrid MXFP8-MXFP4 权重转换。

  >入参介绍：`input_fp8_hf_path`：原始权重路径；`output_hf_path`：转换后输出的权重路径；`quant_mode`：量化模式

  如果权重转换的运行环境为NPU，需要先执行：

  ```shell
  cann_path=/usr/local/Ascend/cann  # cann包安装路径
  source ${cann_path}/bin/setenv.bash
  ```

  权重转换拉起示例：

  ```shell
  # 转换为W8A8-INT8权重，适用于Atlas A3 Pod系列
  python utils/convert_model.py --input_fp8_hf_path /data/models/deepseek_v4  --output_hf_path /data/models/deepseek_v4_int8_w8a8 --quant_type w8a8-int

  # 转换为Hybrid MXFP8-MXFP4权重，适用于950PR/DT系列
  python utils/convert_model.py --input_fp8_hf_path /data/models/deepseek_v4  --output_hf_path /data/models/deepseek_v4_hybrid_mxfp8_mxfp4 --quant_type w4a8-mx

  ```

### 修改代码
- 在各个节点上修改`cann-recipes-infer/models/deepseek-v4/set_env.sh`中的如下字段:
   - `IPs`：配置所有节点的IP，按照rank id排序，多个节点的ip通过空格分开，例如：`('xxx.xxx.xxx.xxx' 'xxx.xxx.xxx.xxx')`。
   - `cann_path`: CANN软件包安装路径，例如`/usr/local/Ascend/cann`。
- 在Atlas A3 Pod各个节点上修改 `config/ci_a3` 路径下需要执行的yaml文件中的model_path真实路径；在950PR/DT各个节点上修改 `config/ci_950` 路径下需要执行的yaml文件中的model_path路径。关于yaml文件中的更多配置说明可参见[YAML参数描述](./config/README.md)。

- 在 yaml 配置中，默认采用npugraph_ex执行方式。这一后端是 NPU 平台全新推出的高性能图计算组件，其基于 CANN 的 AclGraph（对标 CUDAGraph）底层能力，深度融合了一系列 NPU 架构的亲和调度和图优化技术。从落地层面来看，npugraph_ex具备以下显著优势：可快速接入 PyTorch 生态、能无缝集成到 SGLang、vLLM 等主流推理框架中，同时保障极致的运行性能。

- 在各个节点上修改 infer.sh 文件中的YAML_FILE_NAME，指定为上一步需要执行的yaml文件名。

  ```
  # A3 用例
  export YAML_FILE_NAME=ci_a3/deepseek_v4.yaml
  # 950PR/DT  用例
  export YAML_FILE_NAME=ci_950/deepseek_v4.yaml
  ```

  > **Note**: 在A3环境下，INT8 W8A8场景支持 4~64卡部署。可分别在config下的yaml文件中修改world_size (chips * 2) 配置。
### 拉起多卡推理
  在各个节点上同步执行如下命令即可拉起多卡推理任务。
  ```shell
  bash infer.sh
  ```

> **Note：** 不同平台最小部署单元要求如下

| 平台  | 模型型号             | 最小部署单元（chips）|
|-------|---------------------|--------------|
| 950PR/DT  | DeepSeek-V4 Flash    | 4          |
| 950PR/DT  | Deepseek-V4 Pro      | 16         |
| Atlas A3  | DeepSeek-V4 Flash    | 4          |
| Atlas A3  | Deepseek-V4 Pro      | 适配支持中  |
