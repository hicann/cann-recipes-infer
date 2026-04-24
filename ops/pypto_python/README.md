## 概述

此项目是基于昇腾硬件Atlas A2/Atlas A3/Ascend 950PR/Ascend 950DT的融合算子库。当前项目包含的算子说明详见同级docs目录。

## 目录结构

```text
├── pypto_python                                                        # pypto算子代码目录
│    │   ├── docs                                                       # 自定义算子文档
│    │   ├── impl                                                       # 自定义算子计算流代码目录
│    │   │   ├── compressed_flash_attention_pypto.py                    # compressed_flash_attention算子kernel实现
│    │   │   ├── compressor_pypto.py                                    # compressor算子kernel实现
│    │   │   ├── hc_pre_pypto.py                                        # hc_pre算子kernel实现
│    │   │   ├── lightning_indexer_prolog_quant_pypto.py                # lightning_indexer_prolog_quant算子kernel实现
│    │   │   ├── mla_prolog_pypto.py                                    # mla_prolog算子kernel实现
│    │   │   ├── mla_prolog_quant_pypto.py                              # mla_prolog_quant算子kernel实现
│    │   │   ├── sliding_window_attention_pypto.py                      # sliding_window_attention算子kernel实现
│    │   │   ├── sparse_compress_flash_attention_pypto.py               # sparse_compress_flash_attention算子kernel实现
│    │   ├── example                                                    # 自定义算子测试代码目录
│    │   │   ├── test_compressed_flash_attention_pypto.py               # compressed_flash_attention算子测试样例
│    │   │   ├── test_compressor_pypto.py                               # compressor算子测试样例
│    │   │   ├── test_hc_pre_pypto.py                                   # hc_pre算子测试样例
│    │   │   ├── test_lightning_indexer_prolog_quant.py                 # lightning_indexer_prolog_quant算子测试样例
│    │   │   ├── test_mla_prolog_pypto.py                               # mla_prolog算子测试样例
│    │   │   ├── test_mla_prolog_quant_pypto.py                         # mla_prolog_quant算子测试样例
│    │   │   ├── test_sliding_window_attention_pypto.py                 # sliding_window_attention算子测试样例
│    │   │   ├── test_sparse_compressed_flash_attention_pypto.py        # sparse_compressed_flash_attention算子测试样例
```
PyPto自定义算子开发资料：[PyPto文档](https://pypto.gitcode.com/)

## 环境准备<a name="1"></a>
###  硬件要求

  | 产品型号      | 操作系统   | 镜像版本 | 驱动版本 |
  |---------------|-----------|---------|---------|
  | Atlas A2/A3 系列 | Linux ARM |   cann9.0.pt2.8.0_ds_pypto_aarch_image:v0.2     | 25.5.0        |
  | Ascend 950PR/DT 系列 | Linux ARM |   待后续发布     | 待后续发布        |
> npu-smi info 检查Ascend NPU固件和驱动是否正确安装。如果已安装，通过命令`npu-smi info`确认版本是否为25.5.0。如果未安装或者版本不是25.5.0，请先下载[固件和驱动包](https://www.hiascend.com/hardware/firmware-drivers/community?product=7&model=33&cann=All&driver=Ascend+HDK+25.5.0)，然后根据[指导](https://hiascend.com/document/redirect/CannCommunityInstSoftware)自行安装。

### 下载源码

  可以选择在宿主机或者容器内下载源码，如果在容器内下载，应在主机挂载在容器的目录下下载；在宿主机内下载则无此约束。 执行如下命令即可下载 [cann-recipes-infer 源码](https://gitcode.com/cann/cann-recipes-infer)。
  ```shell
  mkdir -p /home/code; cd /home/code/
  git clone git@gitcode.com:cann/cann-recipes-infer.git
  ```

### 获取 docker 镜像

从[ARM镜像地址](https://cann-ai.obs.cn-north-4.myhuaweicloud.com/cann-quantization/DeepSeek/cann9.0.pt2.8.0_ds_pypto_aarch_image.tar)中下载 docker 镜像，然后上传到需要A3服务器每个节点上，并通过命令导入镜像`docker load -i cann9.0.pt2.8.0_ds_pypto_aarch_image.tar`

### 拉起 docker 容器

  容器拉起脚本如下，默认容器名为 cann_recipes_infer_pypto：
  ```
  docker run -u root -itd --name cann_recipes_infer_pypto --ulimit nproc=65535:65535 --ipc=host \
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
      cann9.0.pt2.8.0_ds_pypto_aarch_image:v0.2 /bin/bash
  ```
  进入容器：
  ```
  docker attach cann_recipes_infer_pypto
  ```

### 设置环境变量

  ```bash
  source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
  ```

## 编译执行

### PyPTO算子工程编译安装：

PyPTO已切换为Python代码实现（**镜像中已安装PyPTO的whl包**）。

若涉及PyPTO框架源码修改，可下载PyPto开源仓 https://gitcode.com/cann/pypto ，参考[ReadMe](https://gitcode.com/cann/pypto/blob/master/README.md) “快速入门” 编译PyPTO软件包并安装。


### 示例算子执行：

在pypto_python/examples目录通过执行以下脚本执行示例算子
```shell
cd /home/code/cann-recipes-infer/ops/pypto_python/example
python3 test_hc_pre_pypto.py
```

## DeepSeek-V4 整网集成样例执行
算子已支持集成到DeepSeek-V4整网，样例执行过程如下：

### 权重和数据集准备
DeepSeek-V4模型和数据集准备，请参考[模型权重和数据集准备](../../models/deepseek-v4/README.md)中相关章节

### 代码修改适配
网络执行前需对配置做一些调整，参考[修改代码](../../models/deepseek-v4/README.md)章节进行适配

### 修改网络配置和环境配置
当前网络脚本中，在各个节点上修改models/deepseek-v4/config/ 路径下需要执行的yaml文件中model_config配置项，配置过程如下：
- 增加 enable_pypto: True配置将pypto算子集成到网络中
- 修改 enable_limit_core: False配置将limit_core配置关闭
```
model_config:
    enable_limit_core: False
    enable_pypto: True
```

### 推理执行
参考[拉起多卡推理](../../models/deepseek-v4/README.md)章节。

执行结束后，出现`model run success`，则表示推理执行成功。
