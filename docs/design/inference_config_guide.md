# InferenceConfig 类使用指南

## 1. 概述

`InferenceConfig` 是推理框架中统一的配置入口类，用于替代原有的 `runner_settings` 字典。它采用了结构化的 `dataclass` 设计，提供了类型安全检查、嵌套配置管理以及自动化的验证逻辑，使得模型推理的参数配置更加清晰、易维护。

## 2. 配置类结构

`InferenceConfig` 聚合了多个维度的子配置类，每个类负责特定的功能领域，主要通过yaml文件配置，可参考各模型config目录下的示例yaml进行配置：

### 2.1 DataConfig (数据配置)
管理输入数据相关的参数。
- `dataset`: 数据集名称（默认 "default"）。
- `dataset_path`: 数据集路径（默认 ""），非"default"数据集时必须指定。

### 2.2 ModelConfig (模型配置)
管理模型运行时的特有行为。
- `model_name`: 模型名称（默认 "model"）。
- `model_path`: 模型权重路径（默认 ""）。
- `output_path`: 保存输出、日志、分析数据、图缓存等的目录（不指定时默认和所执行的yaml配置统一目录）。
- `dtype`: 计算使用的数据类型（默认 "bfloat16"）。
- `with_ckpt`: 是否加载权重检查点（默认 True）。
- `next_n`: 投机采样步骤数，即 MTP 模块数量（默认 0）。
- `exe_mode`: 执行模式，可选 eager, ge_graph, acl_graph（默认 "eager"）。
- `enable_cache_compile`: 是否启用缓存编译（默认 False）。
- `micro_batch_mode`: 微批次模式（默认 0）。
- `perfect_eplb`: 是否为 MoE 模型开启完美的专家负载均衡（默认 False）。
- `enable_profiler`: 是否启用性能分析器（默认 False）。

### 2.3 ParallelConfig (并行配置)
定义分布式推理时的并行维度和 Rank 信息。
- `world_size`: 总进程数（默认 1）。
- `global_rank`: **不支持yaml配置**。全局 Rank ID，可在脚本中获取环境变量后传入。
- `local_rank`: **不支持yaml配置**。节点内 Rank ID，可在脚本中获取环境变量后传入。
- `attn_tp_size`: Attention 层TP并行数（默认 1）。
- `attn_dp_size`: **不支持配置**。Attention 层DP并行数，推导方式：`world_size // attn_tp_size`。
- `embed_tp_size`: Embedding 层TP并行数（默认 1）。
- `embed_dp_size`: **不支持配置**。Embedding 层DP并行数，推导方式：`world_size // embed_tp_size`。
- `moe_tp_size`: MoE 层TP并行数，只有MoE模型支持（默认 1）。
- `moe_ep_size`: **不支持配置**。MoE 层专家并行度，推导方式：`world_size // moe_tp_size`。
- `lmhead_tp_size`: LM Head 层TP并行数（默认 1）。
- `dense_tp_size`: dense层TP并行数（默认 1）。
- `o_proj_tp_size`: attention output 映射层TP并行数（默认 1）。
- `cp_size`: context并行度（默认 1）。
- `kvp_size`: KV sequence并行度（默认 1）。

### 2.4 SchedulerConfig (调度配置)
控制请求调度器的策略。
- `batch_size`: 全局总 Batch Size（默认 1）。
- `input_max_len`: Scheduler最大输入序列长度（默认 1024），如果prompt超过这一长度会被截断。
- `max_new_tokens`: 最大生成 token 数（默认 32）。
- `batch_size_per_dp_rank`: **不支持配置**。每个 Rank 的 Batch Size。推导方式：`batch_size // attn_dp_size`。

## 3. 基本用法

### 3.1 从字典创建
通常与 YAML 解析器配合使用：

```python
import yaml
from executor.core.config import InferenceConfig

# 读取 YAML 配置文件
with open("config.yaml", "r") as f:
    yaml_data = yaml.safe_load(f)

# 构造配置对象，需要传入 rank 信息
config = InferenceConfig.from_dict(
    yaml_data, 
    global_rank=0, 
    local_rank=0
)

# 访问配置（注意：所有配置均需通过子配置对象访问）
print(config.model_config.model_name)
print(config.model_config.exe_mode)
print(config.parallel_config.attn_tp_size)
print(config.scheduler_config.max_new_tokens)
print(config.scheduler_config.batch_size)
```
