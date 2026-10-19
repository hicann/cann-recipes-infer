# 一、基于 EvalScope 的 Online 推理 Benchmark 测评

## 1. 概述

本文档用于验证 **cann-recipes-infer** 仓库 Online（PD 分离）模式的在线推理能力及端到端可用性。

Benchmark 测评目标：基于 **EvalScope** 框架对在线推理服务进行 Benchmark 基准测试，验证其推理精度达到预期标准。

---

## 2. 测评准备

本次测评采用 **EvalScope**，针对部署完成的 Online 推理服务进行精度验证。

### 测试模型

- 支持 Online 模式的模型

### Benchmark 数据集

本次测评包含以下公开 Benchmark：

| Benchmark | 能力类型        |
| --------- | --------------- |
| HumanEval | Python 代码生成 |
| MATH-500  | 数学推理        |
| MMLU      | 综合知识理解    |
| LongBench | 长上下文理解    |

### 测评框架

- EvalScope

### 支持的产品型号

- Atlas A3系列产品

### 环境准备

1. 安装CANN软件包。

   本样例的编译执行依赖CANN开发套件包与CANN二进制算子包，支持的CANN软件版本为`CANN 9.0.0`。

   请从[软件包下载地址](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0)下载`Ascend-cann-toolkit_${version}_linux-${arch}.run`与`Ascend-cann-A3-ops_${version}_linux-${arch}.run`软件包，并参考[CANN安装文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0090.html?OS=Ubuntu&InstallType=localpack)进行安装。

   - `${version}`表示CANN包版本号，如9.0.0。
   - `${arch}`表示CPU架构，如aarch64、x86_64。
2. 安装Ascend Extension for PyTorch（torch_npu）。

   Ascend Extension for PyTorch（torch_npu）为支撑PyTorch框架运行在NPU上的适配插件，本样例支持的Ascend Extension for PyTorch版本为`v26.0.0`，PyTorch版本为`2.8.0`。

   请从[软件包下载地址](https://gitcode.com/Ascend/pytorch/releases/v26.0.0-pytorch2.8.0)下载`torch_npu-2.8.0.post4-cp311-cp311-manylinux_2_28_${arch}.whl`安装包，并参考[torch_npu安装文档](https://www.hiascend.com/document/detail/zh/Pytorch/2600/configandinstg/instg/docs/zh/installation_guide/installation_via_binary_package.md)进行安装。

   - `${arch}`表示CPU架构，如aarch64、x86_64。
3. 下载项目源码并安装依赖的 Python 库。

   ```bash
   # 下载项目源码，以master分支为例
   git clone https://gitcode.com/cann/cann-recipes-infer.git

   # 安装依赖的python库，仅支持python 3.11
   cd cann-recipes-infer
   pip3 install -r ./models/qwen3_moe/requirements.txt
   ```
4. 配置样例运行所需环境信息。

   修改`executor/scripts/set_env.sh`中的如下字段：

   - `IPs`：配置所有节点的IP，按照rank id排序，多个节点的 IP 通过空格分开，例如：`('xxx.xxx.xxx.xxx' 'xxx.xxx.xxx.xxx')`。
   - `cann_path`: CANN软件包安装路径，例如`/usr/local/Ascend/ascend-toolkit/latest`。

   > 说明：HCCL相关配置，如：`HCCL_SOCKET_IFNAME`、`HCCL_OP_EXPANSION_MODE`，可以参考[集合通信文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/maintenref/envvar/envref_07_0001.html)并在`executor/scripts/function.sh`中自定义配置。
   >
5. 安装 EvalScope。

   安装：

   ```bash
   pip install evalscope==1.8.0
   ```

   验证：

   ```bash
   evalscope --version
   ```

   **可选组件**

   | 功能          | 安装命令                                    | 说明                 |
   | ------------- |-----------------------------------------| -------------------- |
   | Visualization | `pip install evalscope[service]==1.8.0` | Benchmark 结果可视化 |
   | Sandbox       | `pip install evalscope[sandbox]==1.8.0` | HumanEval 必需       |


   > **说明：**
   > - 本样例中，默认使用 `evalscope==1.8.0`，其中 `evalscope[service]` 为可选组件。
   > - HumanEval 涉及代码执行，必须安装 Sandbox 环境。
   >

---

# 二、Online 推理服务部署

本样例对 Qwen3-235B-A22B 和 Qwen3-8B 进行benchmark测评。两个模型的推理部署和测评流程基本一致，本文后续关于推理部署和测评流程的介绍，均以Qwen3-235B-A22B为例。

## 1. 配置模型 YAML

Online 推理所需的配置文件位于：

```text
models/qwen3_moe/config/qwen3_moe_pd/
```

首次使用时，请将 YAML 文件中的 `model_path` 修改为模型权重所在路径，例如：

```yaml
model_path: /data/models/Qwen3-235B-A22B
```

更多 YAML 参数说明请参考：[YAML 参数说明](../docs/common/inference_config_guide.md)。

### 默认配置

对于以下 Benchmark，可直接使用 `models/qwen3_moe/config/qwen3_moe_pd` 目录下提供的默认配置：

* HumanEval
* MATH-500
* MMLU

无需额外修改其他参数。

### LongBench 配置

LongBench 包含大量长上下文任务，输入长度可达到数万甚至超过百万 Token。为了支持长上下文推理，需要调整 Prefill 与 Decode 的调度参数。

#### Prefill 配置

修改 `prefill.yaml` 中的相关配置：

```yaml
model_config:
  model_name: "qwen3_moe"
  model_path: "/data/models/Qwen3-235B-A22B"
  exe_mode: "eager"
  enable_profiler: False
  with_ckpt: True
  enable_cache_compile: False

parallel_config:
  world_size: 16
  attn_tp_size: 16
  moe_tp_size: 16
  embed_tp_size: 16
  lmhead_tp_size: 16

scheduler_config:
  batch_size: 4
  max_prefill_tokens: 102400
  max_new_tokens: 16
  mem_fraction_static: 0.57
```

#### Decode 配置

修改 `decode.yaml` 中的相关配置：

```yaml
model_config:
  model_name: "qwen3_moe"
  model_path: "/data/models/Qwen3-235B-A22B"
  exe_mode: "ge_graph"
  enable_profiler: False
  with_ckpt: True
  enable_cache_compile: False

parallel_config:
  world_size: 16
  attn_tp_size: 16
  moe_tp_size: 16
  embed_tp_size: 16
  lmhead_tp_size: 16

scheduler_config:
  batch_size: 8
  max_prefill_tokens: 102400
  max_new_tokens: 8192
  mem_fraction_static: 0.85
```

> **说明**
>
> Qwen3-235B-A22B 默认支持的最大上下文长度（`max_position_embeddings`）为 **40960**。
>
> 为支持 LongBench 的长上下文评测，需要将模型目录下 `config.json` 中的 `max_position_embeddings` 修改为 **131072**，否则超长输入会因超过模型上下文长度限制而无法完成推理。

---

## 2. 启动 Online 服务

### Prefill 节点

```bash
bash executor/scripts/infer.sh \
    --model qwen3_moe \
    --mode online \
    --pd-role prefill
```

### Decode 节点

```bash
bash executor/scripts/infer.sh \
    --model qwen3_moe \
    --mode online \
    --pd-role decode
```

---

## 3. 服务验证

Online 服务启动后，Router 默认部署在 **Prefill 节点**。

支持 OpenAI Compatible API：

- `/v1/completions`
- `/v1/chat/completions`

示例：

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
    "model":"default",
    "messages":[
        {
            "role":"user",
            "content":"hello"
        }
    ],
    "max_tokens":10
}'
```

若接口能够正常返回推理结果，则说明 Online 服务已部署成功，可继续进行 Benchmark 测评。

---

# 三、EvalScope Benchmark 测评

## 1. Benchmark 测试

---
> **说明**：本样例未额外设置采样参数，生成时均使用 infer 仓的默认采样参数配置。
### 1.1 HumanEval / MATH-500 / MMLU 评测（EvalScope）

使用 EvalScope 对 HumanEval、MATH-500 和 MMLU 数据集进行评测：

```bash
# 参数说明：
# datasets：选择测试的数据集，本样例可选值：humaneval、math_500、mmlu
# sandbox：是否启用 Sandbox，仅 HumanEval 测试时需要设置为 true
evalscope eval \
    --model Qwen/Qwen3-235B-A22B \
    --api-url http://localhost:8000/v1 \
    --datasets humaneval \
    --sandbox '{"enabled": true}' \
    --eval-batch-size 32 \
    --generation-config '{"max_tokens":65535}' \
    --limit 10  # 正式评测时请删除该参数
```

---

### 1.2 LongBench 评测（EvalScope）

LongBench 为长上下文测评任务，根据 LongBench 论文及其代码仓中的 [LongBench/pred.py](https://github.com/THUDM/LongBench/blob/main/pred.py) 实现，需要对输入进行中间截断处理。

因此，本示例参考 [EvalScope 自定义模型评测](https://evalscope.readthedocs.io/zh-cn/latest/advanced_guides/custom_model.html) 方式，对 LongBench 输入进行中间截断预处理后，再调用模型完成评测。

相关脚本位于：

```text
benchmark/evalscope_scripts/eval_online.py
```

在 `benchmark/evalscope_scripts/` 目录下执行：

```bash
python eval_online.py \
    --datasets longbench_v2 \
    --model-name Qwen/Qwen3-235B-A22B \
    --base-url http://localhost:8000/v1/chat/completions \
    --model-path /data/models/Qwen3-235B-A22B \
    --max-prefill-tokens 102400 \
    --max-tokens 8192 \
    --limit 10 # 正式评测时请删除该参数
```

该脚本也可通过 `--datasets` 选择其他 EvalScope 数据集，例如 `math_500`、`mmlu`；每次运行选择一个数据集，默认是 `longbench_v2`。如需使用本地数据集，可通过 `--dataset-local-path` 指定对应数据集适配器支持的本地路径。

### 1.3 投机推理接受率统计与自动保存

`eval_online.py` 支持在评测结束后汇总投机推理统计。服务端需启用投机推理（`speculative_config.num_speculative_tokens > 0`），且 `/v1/chat/completions` 响应的 `usage` 中包含 `spec_num_accepted_tokens`、`spec_num_draft_tokens` 和 `spec_num_forward_ct`。

例如，在 `benchmark/evalscope_scripts/` 目录下运行 MATH-500 评测，将 `--model-path` 替换为实际部署的 DeepSeek-V4 主模型路径：

```bash
python eval_online.py \
    --datasets math_500 \
    --model-name deepseek_V4 \
    --base-url http://localhost:8000/v1/chat/completions \
    --model-path /data/models/deepseek_v4_model \
    --max-tokens 8192 \
    --batch-size 2 \
    --limit 10 # 正式评测时请删除该参数
```

脚本根据本次运行收到的响应先累加原始计数，再计算总体指标，不直接平均各请求的接受率：

| 报告字段 | 含义 |
| --- | --- |
| `num_requests_total` | 本次脚本调用中完成响应解析并进入统计的请求数 |
| `num_requests_with_spec` | 响应 `usage` 包含 `spec_num_accepted_tokens` 的请求数 |
| `spec_num_accepted_tokens` | 累计接受的 draft token 数 |
| `spec_num_draft_tokens` | 累计参与验证的有效 draft token 数 |
| `spec_num_forward_ct` | 累计投机验证轮数 |
| `acceptance_rate` | 累计接受 token 数 / 累计 draft token 数；分母为零时为 `0.0` |
| `mean_accept_length` | 每轮投机验证平均产生的 token 数，包括接受的 draft token 和主模型补充的一个 token |

评测结束后，若响应包含投机统计字段，脚本会打印统计摘要，并自动保存到本次 EvalScope 输出目录下的 `spec_stats.json`；无投机统计字段时不打印、不生成文件。

---

## 2. Benchmark 结果

| Benchmark                                                                              | 能力类型        | Qwen3-235B-A22B 得分 | Qwen3-8B 得分   |
| -------------------------------------------------------------------------------------- | --------------- | -------------------- | --------------- |
| [HumanEval](https://evalscope.readthedocs.io/zh-cn/latest/benchmarks/humaneval.html)    | Python 代码生成 | **92.24**      | **85.37** |
| [MATH-500](https://evalscope.readthedocs.io/zh-cn/latest/benchmarks/math_500.html)      | 数学推理        | **88.67**      | **82.40** |
| [MMLU](https://evalscope.readthedocs.io/zh-cn/latest/benchmarks/mmlu.html)              | 综合知识理解    | **87.37**      | **77.56** |
| [LongBench](https://evalscope.readthedocs.io/zh-cn/latest/benchmarks/longbench_v2.html) | 长上下文理解    | **44.26**      | **35.98** |
