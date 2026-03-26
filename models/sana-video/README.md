# 在昇腾Atlas A2环境上适配SANA-Video模型的推理
SANA-Video模型是一个多模态视频生成模型，提供了文生视频功能。本项目旨在提供 SANA-Video 模型的 Atlas A2 适配版本，为开发者开展相关 NPU 迁移工作提供参考。

本项目基于NPU主要完成以下优化点，具体内容可至[NPU SANA-Video模型推理优化实践](../../docs/models/sana-video/SANA-Video_optimization.md)查看：
- 转换1×1 conv2d为matmul计算提升性能；
- 调换时序conv2d的hw轴提升性能；
- 支持NPU npu_rotary_mul融合算子；
- 支持NPU npu_rms_norm融合算子；
- 支持NPU npu_fusion_attention融合算子；

## 执行样例
本样例支持昇腾Atlas A2环境的单机单卡推理和单机多卡DP推理。

### CANN环境准备
  1.安装CANN软件包
  
  本样例的编译执行依赖CANN开发套件包（cann-toolkit）与CANN二进制算子包（cann-kernels），支持的CANN软件版本为`CANN 8.5.0`。
  
  请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.0)下载`Ascend-cann-toolkit_{version}_linux-{arch}.run`和`Ascend-cann-{soc}-ops_{version}_linux-{arch}.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=netconda&OS=Debian)进行安装。
  
  2.安装Ascend Extension for PyTorch（torch_npu）
  
  Ascend Extension for PyTorch（torch_npu）为支撑PyTorch框架运行在NPU上的适配插件，本样例支持的Ascend Extension for PyTorch版本为`torch-npu == 7.1.0.post17`，支持的Torch版本为`torch == 2.6.0`，请从[Ascend Extension for PyTorch插件](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0004.html)下载并安装torch与torch_npu安装包。建议在conda环境中安装：
  ```
conda create -n sana python=3.10
conda activate sana
  ```

### 依赖安装
本仓库依赖SANA-Video的开源代码代码。
首先在项目目录拉取SANA-Video源代码：
```
git clone https://github.com/NVlabs/Sana.git
```

进入到源代码目录：
```
cd Sana
```

切换到指定版本：
```
git checkout 08c656c3
```

回到原目录：
```
cd ..
```

拉取本仓库代码：
```
git clone https://gitcode.com/cann/cann-recipes-infer.git
```

将Sana仓库的代码以**非覆盖模式**复制到本项目目录下：
```
cp -rn Sana/* cann-recipes-infer/models/sana-video/
```

安装依赖：

```
cd cann-recipes-infer/models/sana-video
pip install -e .
```

编译安装mmcv库(注意1.x分支才有Registry模块)：
```
git clone -b 1.x https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 FORCE_NPU=1 python setup.py install
cd ..
```
## 模型权重
执行脚本会自动下载模型权重到`/root/.cache/huggingdace/hub/`目录下。
若下载失败，可手动下载[SANA-Video_2B_480p](https://huggingface.co/Efficient-Large-Model/SANA-Video_2B_480p)模型到该目录下。


## 快速启动

本样例准备了单卡环境下的训练和推理脚本。执行脚本前，参考[Ascend社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=Debian)中的CANN安装软件教程配置环境变量：
```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
执行推理脚本：
```
bash inference_video_scripts/inference_sana_video.sh \
      --np 1 \
      --config configs/sana_video_config/Sana_2000M_480px_AdamW_fsdp.yaml \
      --model_path hf://Efficient-Large-Model/SANA-Video_2B_480p/checkpoints/SANA_Video_2B_480p.pth \
      --txt_file=asset/samples/video_prompts_samples.txt \
      --cfg_scale 6 \
      --motion_score 30 \
      --flow_shift 8 \
      --work_dir output/sana_t2v_video_results \
      --model.fp32_attention False
```
参数说明：
- `np`: 推理使用卡数
- `config`: 推理使用的配置文件
- `model_path`: 推理使用的权重文件路径
- `txt_file`: 推理使用的文本提示词文件
- `cfg_scale`: 提示词对齐强度
- `motion_score`: 运动强度分数
- `flow_shift`: 流偏移参数，用于调整扩散模型去噪时间步
- `work_dir`: 生成视频输出路径
- `model.fp32_attention`: 控制attn是否使用fp32精度，推理时可设为False提升性能

## 性能数据

本样例在Atlas A2的推理性能如下表所示：

| 规格| 单步时延(s) | 端到端时延(s) |
|--|--| --|
| 480p81f | 2.79 | 154.2 |