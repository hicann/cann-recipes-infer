## 概述
此项目是基于昇腾硬件Atlas A3的融合算子库，当前项目中包括[DeepseekIndexerAttention](./docs/deepseek_indexer_attention.md) 和[LightningIndexer](./docs/lightning_indexer.md)两个算子。

## 目录结构

```text
├── src                                                                 # 自定义算子工程目录
│    │   ├── deepseek_indexer_attention                                 # DeepseekIndexerAttention算子代码
│    │   │      ├── op_host                                             # ophost目录
│    │   │      │     ├── deepseek_indexer_attention_def.cpp            # DeepseekIndexerAttention算子原型注册代码
│    │   │      │     ├── deepseek_indexer_attention_proto.cpp          # DeepseekIndexerAttention算子InferShape、InferDataType代码
│    │   │      │     ├── deepseek_indexer_attention_tiling.cpp         # DeepseekIndexerAttention tiling代码
│    │   │      ├── op_kernel                                           # op_kernel代码目录
│    │   │      │     ├── deepseek_indexer_attention.cpp                # DeepseekIndexerAttention算子kernel源码
│    │   │      │     ├── deepseek_indexer_attention_impl.cpp           # DeepseekIndexerAttention算子ut测试用例
│    │   │      │     ├── dynamic_mla.cpp                               # MLA prolog算子kernel源码
│    │   │      │     ├── gather_after_prolog.cpp                       # gather算子kernel源码
│    │   │      │     ├── lightning_indexer_prolog.cpp                  # lightning indexer prolog算子kernel源码
│    │   │      │     ├── lightning_indexer_topk.cpp                    # LightningIndexer算子kernel源码
│    │   │      │     ├── selected_attention.cpp                        # selected attention算子kernel源码
│    │   ├── lightning_indexer_pto                                      # lightning indexer算子代码
│    │   │      ├── op_host                                             # ophost目录
│    │   │      │     ├── lightning_indexer_def.cpp                     # LightningIndexer算子原型注册代码
│    │   │      │     ├── lightning_indexer_proto.cpp                   # LightningIndexer算子InferShape、InferDataType代码
│    │   │      │     ├── lightning_indexer_tiling.cpp                  # lightningIndexer tiling代码
│    │   │      ├── op_kernel                                           # op_kernel代码目录
│    │   │      │     ├── lightning_indexer_topk.cpp                    # LightningIndexer算子kernel源码
│    │   │      │     ├── lightning_indexer_impl.cpp                    # LightningIndexer算子ut测试用例
├── torch_ops_extension                                                 # 自定义算子注册代码目录
│    │   ├── custom_ops                                                 # custom_ops目录
│    │   │      ├── converter                                           # 自定义算子包python侧converter代码目录
│    │   │      │     ├── npu_lightning_indexer_pto.py                  # LightningIndexer算子converter注册
│    │   │      │     ├── npu_sparse_attention_pto.py                   # DeepseekIndexerAttention算子converter注册
│    │   │      ├── csrc                                                # 自定义算子适配层c++代码目录
│    │   │      │     ├── npu_lightning_indexer_pto.cpp                 # LightningIndexer算子适配代码以及实现注册
│    │   │      │     ├── npu_sparse_attention_pto.cpp                  # DeepseekIndexerAttention算子适配代码以及实现注册

```

## 环境准备
###  硬件要求
产品型号：Atlas A3 系列

操作系统：Linux ARM

镜像版本：cann8.3.rc1.alpha002_pt2.5.1_dsv3.2_pypto_aarch_image:v0.1

驱动版本：Ascend HDK 25.2.0
> npu-smi info 检查Ascend NPU固件和驱动是否正确安装。如果已安装，通过命令`npu-smi info`确认版本是否为25.2.0。如果未安装或者版本不是25.2.0，请先下载[固件和驱动包](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743/software/264360782?idAbsPath=fixnode01|23710424|251366513|254884019|261408772|252764743)，然后根据[指导](https://hiascend.com/document/redirect/CannCommunityInstSoftware)自行安装。
### 下载源码
  可以选择在宿主机或者容器内下载源码，如果在容器内下载，应在主机挂载在容器的目录下下载；在宿主机内下载则无此约束。 执行如下命令即可下载 cann-recipes-infer 源码。
  ```
  mkdir -p /home/code; cd /home/code/
  git clone git@gitcode.com:cann/cann-recipes-infer.git
  ```

### 获取 docker 镜像
从[ARM镜像地址](https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/cann8.3.rc1.alpha002/pt2.5.1/aarch/pypto/cann8.3.rc1.alpha002_pt2.5.1_dsv3.2_pypto_aarch_image.tar)中下载 docker 镜像，然后上传到需要A3服务器每个节点上，并通过命令导入镜像`docker load -i cann8.3.rc1.alpha002_pt2.5.1_dsv3.2_pypto_aarch_image.tar`

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
      cann8.3.rc1.alpha002_pt2.5.1_dsv3.2_pypto_aarch_image:v0.1 /bin/bash
  ```
  进入容器：
  ```
  docker attach cann_recipes_infer_pypto
  ```

### 设置环境变量
  ```bash
  source /usr/local/Ascend/latest/bin/setenv.bash
  ```

### 依赖安装
 - python >= 3.7.0
  - gcc >= 7.3.0
  - cmake >= 3.16.0
  - JSON for Modern C++（建议版本 [v3.11.3](https://github.com/nlohmann/json/releases/tag/v3.11.3)）

   - 如下以[JSON for Modern C++源码](https://github.com/nlohmann/json/releases/tag/v3.11.3)编译安装为例，安装命令如下：

     ```bash
     mkdir temp && cd temp                # 在 JSON for Modern C++ 源码根目录下创建临时目录并进入
     cmake .. -D_GLIBCXX_USE_CXX11_ABI=0 -DJSON_MultipleHeaders=ON -DJSON_BuildTests=OFF
     make
     make install                         # root用户安装
      ```

## 编译执行
### 算子工程编译安装：
进入pypto目录通过执行build.sh脚本编译自定义算子工程，代码发生修改后需要重新执行该步骤
```shell
cd /home/code/cann-recipes-infer/ops/pypto
bash ./build.sh
```
编译完成后，在run_pkg目录下生成customize_ops_linux.XXXX.run自定义算子包，安装run包
```shell
cd run_pkg
./customize_ops_linux.XXXX.run
```

### whl包编译安装：
在pypto/torch_ops_extension目录通过执行build_and_install.sh脚本编译安装whl包，代码发生修改后需要重新执行该步骤
```shell
cd torch_ops_extension
bash ./build_and_install.sh
```

### sample算子执行：
在pypto/examples目录通过执行run.sh脚本执行示例算子
```shell
cd examples
python3 test_deepseek_indexer_attention.py
python3 test_lightning_indexer_pto.py 
```

## DeepSeek-V3.2-Exp 整网集成样例执行
lightning_indexer_pto算子已支持集成到DeepSeek-V3.2-Exp整网，样例执行过程如下：
### 权重和数据集准备
DeepSeek-V3.2-Exp模型和数据集准备，请参考[模型权重和数据集准备](../../models/deepseek-v3.2-exp/README.md)中相关章节
### 代码修改适配
网络执行前需对配置做一些调整，参考[修改代码](../../models/deepseek-v3.2-exp/README.md)章节进行适配
### 修改网络配置和环境配置
当前网络脚本中，在各个节点上修改models/deepseek-v3.2-exp/config/ 路径下需要执行的yaml文件中model_config配置项，配置过程如下：
- 增加 enable_pypto: True配置将pypto的lighning_indexer算子集成到网络中
```
model_config:
    enable_pypto: True
```
修改`models/deepseek-v3.2-exp/set_env.sh`中CANN_PATH配置:
``` shell
CANN_PATH=/usr/local/Ascend/latest
export ASCEND_HOME_PATH=$CANN_PATH
```

### 推理执行
参考[推理执行](../../models/deepseek-v3.2-exp/README.md)中推理执行章节。

执行结束后，出现`model run success`，则表示推理执行成功；

