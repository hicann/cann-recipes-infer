# DeepSeek-V3.2-Exp Inference on NPU
## 概述
DeepSeek团队发布了最新的模型DeepSeek-V3.2-Exp，在各项指标上都达到了SOTA水平。本样例基于Deepseek开源代码进行迁移，并在CANN平台上完成对应的优化适配，可在华为 Atlas A3 集群上运行起来。

- 本样例的并行策略和性能优化点详细介绍可参见[NPU DeepSeek-V3.2-Exp推理优化实践](../../docs/models/deepseek-v3.2-exp/deepseek_v3.2_exp_inference_guide.md)。

---

## 硬件要求
产品型号：Atlas A3 系列

操作系统：Linux ARM

镜像版本：cann8.3.rc1.alpha002_pt2.5.1_dsv3.2_aarch_image:v0.1

驱动版本：Ascend HDK 25.2.0
> npu-smi info 检查Ascend NPU固件和驱动是否正确安装。如果已安装，通过命令`npu-smi info`确认版本是否为 25.2.0。如果未安装或者版本不是 25.2.0，请先下载[固件和驱动包](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743/software/264360782?idAbsPath=fixnode01|23710424|251366513|254884019|261408772|252764743)，然后根据[指导](https://hiascend.com/document/redirect/CannCommunityInstSoftware)自行安装。


## 快速启动


### 下载源码
  
  在各个节点上执行如下命令下载 cann-recipes-infer 源码。
  ```shell
  mkdir -p /home/code; cd /home/code/
  git clone git@gitcode.com:cann/cann-recipes-infer.git
  cd cann-recipes-infer
  ```
### 下载数据集
  从[链接](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/blob/main/longbook_qa_eng.jsonl)中下载长序列输入数据集longbook_qa_eng，并上传到各个节点上新建的路径 dataset/InfiniteBench下。
  ```shell
  mkdir -p dataset/InfiniteBench
  ```

### 下载权重

  下载[DeepSeek-V3.2-Exp原始fp8权重](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/tree/main)，并上传到Atlas A3各节点某个固定的路径下，比如`/data/models/DeepSeek-V3.2-Exp-fp8`。

### 获取 docker 镜像
  从[ARM镜像地址](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/cann8.3.rc1.alpha002/pt2.5.1/aarch/ascendc/cann8.3.rc1.alpha002_pt2.5.1_dsv3.2_aarch_image.tar)中下载 docker 镜像，然后上传到A3服务器的每个节点上，并通过命令导入镜像 `docker load -i cann8.3.rc1.alpha002_pt2.5.1_dsv3.2_aarch_image.tar`。

### 拉起 docker 容器

  在各个节点上通过如下脚本拉起容器，默认容器名为 cann_recipes_infer。注意：需要将权重路径和源码路径挂载到容器里。
  ```
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
      cann8.3.rc1.alpha002_pt2.5.1_dsv3.2_aarch_image:v0.1 /bin/bash
  ```
  在各个节点上通过如下命令进入容器：
  ```
  docker attach cann_recipes_infer
  cd /home/code/cann-recipes-infer
  ```

### 转换权重

  在各个节点上使用`convert_model.py` 脚本完成FP8到bfloat16权重转换。脚本输入参数*input_fp8_hf_path*为原始fp8权重路径，*output_hf_path*为转换后的bfloat16权重路径。

  ```
  # 转换为bf16权重
  cd models/deepseek-v3.2-exp
  python utils/convert_model.py --input_fp8_hf_path /data/models/DeepSeek-V3.2-Exp-fp8 --output_hf_path /data/models/DeepSeek-V3.2-Exp-bf16
  ```

### 修改代码
- 在各个节点上修改 set_env.sh 文件中的IPs。
  ```shell
  export IPs=('xxx.xxx.xxx.xxx' 'xxx.xxx.xxx.xxx') # 所有节点的IP，确保第1个IP是master，多个节点的ip通过空格分开
  ```

- 在各个节点上修改 config/ 路径下需要执行的yaml文件中的model_path路径。关于YAML文件中的更多配置说明可参见[YAML参数描述](./config/README.md)。
  ```
  model_path: "/data/models/DeepSeek-V3.2-Exp-bf16/"
  ```

- 在各个节点上修改 infer.sh 文件中的YAML_FILE_NAME，指定为上一步需要执行的yaml文件名。默认的yaml路径为32卡推理。
  ```
  export YAML_FILE_NAME=deepseek_v3.2_exp_rank_64_64ep_prefill.yaml
  ```
 
  > **Note**: 本样例Prefill支持32-128卡，Decode支持32-128卡，可分别在config下的deepseek_v3.2_exp_rank_64_64ep_prefill.yaml和deepseek_v3.2_exp_rank_128_128ep_decode.yaml文件中修改world_size配置。
    
### 拉起多卡推理
  在各个节点上同步执行如下命令即可拉起多卡推理任务。
  ```shell
  cd models/deepseek-v3.2-exp/
  bash infer.sh
  ```

## 附录
### FAQ
- **HCCL_BUFFSIZE不足问题**：如果报错日志中出现关键字"HCCL_BUFFSIZE is too SMALL, ..., NEEDED_HCCL_BUFFSIZE..., HCCL_BUFFSIZE=200MB, ..."，可通过配置环境变量 `export HCCL_BUFFSIZE=实际需要的大小` 解决。HCCL_BUFFSIZE参数介绍可[参考昇腾资料](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/maintenref/envvar/envref_07_0080.html)中的详细描述。