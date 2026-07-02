<h1 align="center">
deepseek_v3_ascend310p
</h1>

## 1. 概述

### 1.1. 介绍

deepseek_v3_ascend310p专注于在Ascend 310P平台上实现DeepSeekV3模型（r1 671B版本）的高效推理。本项目提供了完整的推理代码及详细的部署指南，旨在帮助开发者快速部署和优化模型，充分发挥Ascend 310P的计算能力，提升推理性能。

### 1.2. 项目结构

```text
mlguider_llm_ascend/
├── cpp/                               # C++核心推理引擎源码
|   ├── 3rdparty/          		       # 第三方库
│   └── mlguider_llm_ascend/           # 第三方库
│       ├── common/                    # 公共工具模块
│       ├── config/                    # 配置管理
│       ├── core/                      # 核心计算图与算子实现
│       ├── data_processor/            # 数据预处理/后处理
│       ├── runtime/                   # 运行时管理
│       ├── CMakeLists.txt             # C++模块构建配置
│       └── main_inference.cpp         # 推理主程序入口
|
|
├── example/deepseek-v3-quant/         # DeepSeek-V3量化模型示例
│   ├── convert_checkpoint_rtn.py      # RTN量化权重转换
│   ├── fp8_cast_int8_quantize_RTN.py  # FP8转INT8量化工具
│   ├── inference_run_config_tp8_ep32_dp1.yaml           # 标准推理配置
│   ├── inference_run_config_tp8_ep32_dp1_msprof.yaml    # 性能分析配置
│   ├── inference_run_config_tp8_ep32_dp1_prefill.yaml   # 预填充阶段配置
│   ├── inferencetp8ep32dp1.sh                       # 基础推理启动脚本
│   ├── inferencetp8ep32dp1_msprof.sh                # 带性能分析的推理脚本
│   ├── inferencetp8ep32dp1_example.sh              # 完整示例脚本
│   └── inferencetp8ep32dp1_example_prefill.sh      # 预填充示例脚本
```

## 2. 环境配置

### 2.1. 硬件环境

**NPU**: Ascend 310IDUO*20

**CPU**: X86架构

### 2.2. 软件环境

### 2.2.1. CANN安装

1. 安装toolkit[必须下载指定CANN包]

下载toolkit包：https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20250325_newest/Ascend-cann-toolkit_8.1.RC1_linux-x86_64.run

```Bash
chmod +x Ascend-cann-toolkit_8.1.RC1_linux-x86_64.run
bash Ascend-cann-toolkit_8.1.RC1_linux-x86_64.run --full
```

2. 安装对应的aclnn-kernel包

下载kernel包：https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/CANN/20250418_newest/Ascend-cann-kernels-310p_8.1.RC1_linux-x86_64.run

```Bash
chmod +x Ascend-cann-kernels-310p_8.1.RC1_linux-x86_64.run
bash Ascend-cann-kernels-310p_8.1.RC1_linux-x86_64.run --install --install-for-all --quiet
```

3. 激活CANN脚本

```Bash
source /path/to/Ascend/ascend-toolkit/set_env.sh
```

4. 安装自定义算子库

   [下载](https://download-public.tos-cn-beijing.volces.com/vendors.zip)自定义算子vendor包，解压后将其放置在`/path_to_ascend_toolkit/latest/opp/`目录下。

### 2.2.2. 安装定制HCCL

请注意：**必须严格安装**上述指定的CANN-toolkit，否则将因版本不匹配导致异常。

1. [下载](https://download-public.tos-cn-beijing.volces.com/hccl.zip)hccl补丁及定制HCCL通信库，并解压。

2. **安装补丁**。由于hccl中控制拓扑与建链等相关模块尚未完全开源，需安装相应补丁方可启用20卡通信功能。

```Bash
cd path_to_hccl/hccl/hccl-patch

cp hccl_common.h /path/to/Ascend/ascend-toolkit/8.1.RC1/x86_64-linux/include/experiment/hccl/
cp x86/libhccl* /path/to/Ascend/ascend-toolkit/8.1.RC1/x86_64-linux/lib64/
```

3. 安装HCCL库

```Bash
bash path_to_hccl/hccl/CANN-hccl_alg-7.7.t13.0.b055-linux.x86_64.run
```

### 2.2.3. 安装模型运行所需环境

1. **安装配置openmpi**

   1. 下载并解压Open MPI软件包。

   2.  参见[Open MPI-4.1.7](https://www.open-mpi.org/software/ompi/v4.1/)下载软件包并解压。

   3. ```Bash
      wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.7.tar.gz
      tar -zxvf openmpi-4.1.7.tar.gz
      ```

   4. 编译并安装Open MPI软件包。

   5. ```Bash
      cd openmpi-4.1.7
      ./configure --disable-fortran --enable-ipv6 --prefix=${HOME}/.local/openmpi
      make && make install
      ```

   6.  详情参见[MPI安装与配置](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha001/devaids/devtools/hccltool/HCCLpertest_16_0002.html)。

   7. 配置openmpi环境变量。

   8.  安装完成后，需要配置环境变量，将openmpi的bin目录添加到PATH中，例如：

   9. ```Bash
      export PATH=/path/to/openmpi/bin:$PATH
      ```

2. **安装ABSL**

   1. 克隆absl代码

   2. ```Bash
       git clone https://github.com/abseil/abseil-cpp.git
      ```

   3. 进入 abseil-cpp 目录并创建 build 目录

   4. ```Bash
         cd abseil-cpp
         mkdir build
         cd build
      ```

   5. 使用 CMake 构建并安装 absl

   6. ```Bash
         # 需要根据情况修改安装目录
         cmake .. -DCMAKE_INSTALL_PREFIX=${HOME}/.local/
         make -j
         make install
      ```


3. 下载第三方库（json库&cpp-httplib）
```Bash
   wget https://download-public.tos-cn-beijing.volces.com/3rdparty.zip
   unzip 3rdparty.zip
   cp -r 3rdparty contrib/ascend310PDSV3/src/cpp/
```

4. 安装python依赖库

```Bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
```

## 3. 安装/部署

1. 运行该编译安装脚本

```Bash
cd cpp
./buils.sh
```

## 4. 快速开始

### 4.1. 模型权重转换

0. **下载权重**

从HF上下载权重：[huggingface.co](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324)

1. **模型量化（FP8->Int8）**

```Bash
cd example/deepseek-v3-quant
python fp8_cast_int8_quantize_RTN.py \
--input-fp8-hf-path $INPUT_FP8_MODEL_PATH \
--output-int8-hf-path $OUTPUT_INT8_MODEL_PATH
```

将下列四个文件从`$INPUT_FP8_MODEL_PATH`拷贝至`$OUTPUT_INT8_MODEL_PATH`目录中。

1. config.json
2. configuration_deepseek.py
3. tokenizer_config.json
4. tokenizer.json

2. **模型转化**

```Bash
python convert_checkpoint_rtn.py \
--model_name_or_path $OUTPUT_INT8_MODEL_PATH \
--save_path $OUTPUT_MODEL_PATH \
--world_size 32 \
--tp_size 8 \
--ep_size 32

# NOTE:
# tp_size for MLA
# ep_size for MOE
# world_size == ep_size == 32
# tp_size = [1,2,4,8]
```

### 4.2. 模型推理

1. 运行推理脚本：

```Bash
bash example/deepseek-v3-quant/inferencetp8ep32dp1_example.sh
```

推理脚本解释：

```Bash
#!/bin/bash
export TM_LOG_LEVEL=DEBUG
# 获取当前脚本的绝对路径
SCRIPT_DIR=$(dirname "$(realpath "$0")")
# 删除历史outputIds.txt，每次推理会生成新的outputIds.txt
rm -rf $SCRIPT_DIR/../../outputIds.txt
# 构建 launch.py 的完整路径
LAUNCH_PY="$SCRIPT_DIR/../../launch_inference.py"
# 启动推理，单次推理2-Batch
python "$LAUNCH_PY"
--config "$SCRIPT_DIR/inference_run_config_tp8_ep32_dp1.yaml"
--input-text \
"Hello" \
"我希望昨天是明天，那么今天就是星期五。今天是星期几？"
```

## 5. 结果

### 5.1. 性能

**Decode阶段**

- **数据集**：采用**HumanEval**数据集，从中选取多样化的Prompt组成测试Batch。
- **Batch Size**梯度测试：[1, 2, 4] 共5个梯度。
- 上下文长度动态范围：
  - 基础长度：2048 tokens。
  - 最大长度：min(Single_Batch Prompts长度) + 2048 + 512（保留512 tokens作为生成缓冲）。
- **指标**：TPOT——平均值（Mean）、中位数P50（典型情况）、P90（长尾性能）。

| BatchSize | TPOT AVG(ms) | TPOT 50%(ms) | TPOT 90%(ms) | AVG Decode TPS |
| --------- | ------------ | ------------ | ------------ | -------------- |
| 1         | 124.129      | 123.478      | 124.724      | 8.06           |
| 2         | 137.175      | 136.600      | 138.390      | 7.29           |
| 4         | 161.050      | 160.524      | 162.618      | 6.21           |



### 5.2. 精度

采用常见数据集[MMLU](https://people.eecs.berkeley.edu/~hendrycks/data.tar)**和**[HumanEval](https://github.com/openai/human-eval)，[MGSM](https://huggingface.co/datasets/juletxara/mgsm)，[DROP](https://huggingface.co/datasets/ucinlp/drop)数据集进行精度测试。

| 数据集          | Ours    | BaseLine |
| --------------- | ------- | -------- |
| MMLU            | 84.82 % | 85.09 %  |
| MMLU-stem       | 85.63 % | 86.30 %  |
| MMLU-Humanities | 77.85 % | 78.30 %  |
| MMLU-other      | 88.61 % | 88.67 %  |
| MMLU-social     | 90.84 % | 90.64 %  |
| HumanEval       | 70.73 % | 93.29 %  |
| MGSM            | 92.11 % | 92.07 %  |
| DROP(3-shot)    | 80.39 % | 81.05 %  |

**NOTE**：

HumanEval基准测试的精度出现22.56%的显著下降，其主要原因在于模型推理过程中因权重精度转换引入了生成偏差。具体分析如下：

1. **问题根源**

原始模型基于BF16精度存储，在PyTorch CPU环境下进行精度对比实验时发现：当BF16权重转换为FP16后，模型在推理特定token时会产生异常输出，表现为生成内容中随机出现如“吃点”等不符合语义的中文字符。

2. **对评估基准的差异化影响**

- **HumanEval**：该基准依赖**实际执行模型生成的代码**以验证功能正确性。异常字符会直接破坏代码语法结构或逻辑，导致执行失败，因此对精度影响显著。
- **MMLU**：该任务仅需模型**输出选择题的正确答案标签**（如选项“A”或“B”）。即使生成内容中掺杂无关字符，只要正确选项能被准确识别，对最终判断的影响相对有限。

3. **结论**

​	权重从BF16至FP16的精度转换引发了模型输出层的分布漂移，从而在生成过程中引入结构性噪声。这种噪声对强依赖生成完整性和一致性的任务（如代码生成）产生严重负面影响，而对仅需关键信息提取的任务（如选项判断）影响较小。

## 6. LICENSE
[LICENSE](LICENSE)
