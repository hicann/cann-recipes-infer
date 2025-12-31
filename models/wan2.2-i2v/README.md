# 在昇腾Atlas A2环境上适配Wan2.2-I2V模型的推理
Wan2.2-I2V模型是一款多模态视频生成模型，提供了图生视频功能。本项目旨在提供 Wan2.2-I2V 模型的 Atlas A2 适配版本，为开发者开展相关 NPU 迁移工作提供参考。

本项目基于NPU主要完成以下优化点，具体内容可至[NPU Wan2.2-I2V模型推理优化实践](https://gitcode.com/weixin_45381022/cann-recipes-infer/blob/master/docs/models/Wan2.2-I2V/Wan2.2-I2V_optimization.md)查看：

- 支持NPU npu_fused_infer_attention_score融合算子；
- 支持NPU npu_rotary_mul融合算子；
- 支持NPU npu_rms_norm融合算子；
- 支持NPU npu_layer_norm_eval融合算子；
- 支持多卡VAE并行；
- 支持CFG并行;
- 支持Ring_Attention序列并行和通算掩盖。


## 执行样例
本样例支持支持Atlas A2环境的多卡推理。

###  CANN环境准备
  1.安装CANN软件包
  
  本样例的编译执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），支持的CANN软件版本为`CANN 8.5.0.alpha002`。
  
  请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0.alpha002)下载`Ascend-cann-toolkit_{version}_linux-{arch}.run`和`Ascend-cann-kernels-{soc}_{version}_linux.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/softwareinst/instg/instg_0001.html?Mode=PmIns&OS=Debian&Software=cannToolKit)进行安装。
  
  2.安装Ascend Extension for PyTorch（torch_npu）
  
  Ascend Extension for PyTorch（torch_npu）为支撑PyTorch框架运行在NPU上的适配插件，本样例支持的Ascend Extension for PyTorch版本为`torch-npu == 2.1.0.post17`，支持的Torch版本为`torch == 2.1.0`，详细内容可见[官方文档](https://pypi.org/project/torch-npu/2.1.0.post17/)。
  
  
  


### 依赖安装



本仓库依赖于Wan2.2的开源仓库代码。

首先进入Wan2.2的仓库，下载开源仓库代码：

```
git clone https://github.com/Wan-Video/Wan2.2.git
```



下载本仓库代码：

```
git clone https://gitcode.com/cann/cann-recipes-infer.git
```



将Wan2.2仓库的代码以**非覆盖模式**复制到本项目目录下：


```
cp -rn Wan2.2 cann-recipes-infer/models/Wan2.2-I2V
```

```
#安装Python依赖
pip install -r requirements.txt
```


### 准备模型权重

  
| 模型 |版本  |
|--|--|
| Wan2.2-I2V | [BF16](https://www.modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B-BF16/files?version=) |
  

  下载Wan2.2-I2V-A14B-BF16模型权重到本地路径`ckpts`。
```
Wan2.2-I2V-A14B-BF16/
├── configuration.json
├── gitattributes
├── google/
│   └── umt5-xxl/
│       ├── special_tokens_map.json
│       ├── spiece.model
│       ├── ...  
├── high_noise_model/
│   ├── config.json
│   ├── diffusion_pytorch_model-00001-of-00006
│   ├── ...  
│   └── diffusion_pytorch_model.index.json
├── low_noise_model/
│   ├── config.json
│   ├── diffusion_pytorch_model-00001-of-00006
│   ├── ...  
│   └── diffusion_pytorch_model.index.json
├── models_t5_umt5-xxl-enc-bf16.pth
├── README.md
└── Wan2.1_VAE.pth
```
  

## 快速启动
  
  
  本样例准备了多卡的推理脚本，首先请按照前述的内容准备好执行环境和代码。
  
  提前启用torch_npu环境：
```
  source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash 
```
 
 再通过执行`bash run.sh`命令进行多卡推理。
 
 可通过`--size`参数和`--frame_num`参数修改生成的视频规格，原模型的720P、480P等规格均支持；需要注意的是序列并行的约束条件，即`nproc_per_node == cfg_size * ulysses_size * ring_size`。
 
 启动脚本的更多细节可见**附录**。
 
 ## 性能数据
 
 本样例的多卡端到端推理性能如下表所示（Ulysses=8，Ring_size=1，CFG_size=1)：
 
 
| 规格|Atlas 800I A2 - 8卡 / s |
|--|:--:|
| 832*480 *61 | 102.54 | 
| 1280*720 *81 | 418.01 | 

 
 

 
 
 
 ## 附录
**` run.sh`脚本详细说明：**
```
export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false
source set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh

torchrun --nproc_per_node=8 generate.py \
--task i2v-A14B \
--ckpt_dir ${model_base} \
--size 1280*720 \
--frame_num 81 \
--sample_steps 40 \
--dit_fsdp \
--t5_fsdp \
--cfg_size 1 \
--ring_size 8 \
--ulysses_size 1 \
--vae_parallel \
--image examples/i2v_input.JPG \
--prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
--base_seed 100
```
**参数说明：**
- `task`: 任务类型
- `ckpt_dir`: 模型的权重路径
- `size`: 生成视频的分辨率，支持(1280,720)、(832,480)等分辨率
- `frame_num`: 生成视频的帧数
- `sample_steps`: 推理步数
- `dit_fsdp`: dit使能fsdp, 用以降低显存占用
- `t5_fsdp`: t5使能fsdp, 用以降低显存占用
- `cfg_size`: cfg并行数
- `ring_size`: ring并行数
- `ulysses_size`: ulysses并行数
- `vae_parallel`: 使能vae并行策略
- `prompt`: 文本提示词
- `base_seed`: 随机种子


**环境变量说明：**

 - `PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'`：PyTorch针对昇腾NPU的内存分配配置，'True'启用'可扩展内存段'功能，减少因内存碎片或预分配不足导致的OOM问题，详情可见[官方文档](https://www.hiascend.com/document/detail/zh/Pytorch/710/comref/Envvariables/Envir_012.html)；
-  `TASK_QUEUE_ENABLE=2`：开启task_queue算子下发队列Level 2优化，将workspace相关任务迁移至二级流水，掩盖效果更好，性能收益更大，详情可见[官方文档](https://www.hiascend.com/document/detail/zh/Pytorch/710/comref/Envvariables/Envir_007.html)；
-  `CPU_AFFINITY_CONF=1`：表示开启粗粒度绑核，将所有任务绑定在NPU业务绑核区间的所有CPU核上，避免不同卡任务之间的线程抢占，详情可见[官方文档](https://www.hiascend.com/document/detail/zh/Pytorch/710/comref/Envvariables/Envir_033.html)；
-  `TOKENIZERS_PARALLELISM=false`：禁用tokenizers库内部的并行化处理；
- `source /usr/local/Ascend/ascend-toolkit/set_env.sh`：[配置CANN环境变量](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/softwareinst/instg/instg_quick.html?Mode=PmIns&OS=Debian&Software=cannToolKit);
- `source set_env.sh`: 动态计算脚本所在的项目根目录，并将其添加到 Python 的模块搜索路径中，确保 Python 能找到项目内的自定义模块。
 

