# cann-recipes-infer — NPU 推理优化

cann-recipes-infer 是基于 CANN 平台的 LLM/多模态模型离线推理优化样例库，面向昇腾 Atlas A2/A3 硬件。支持 TP/EP/DP 并行策略组合，提供 `ge_graph` 和 `eager` 两种执行模式。核心特性包括多流并行、融合算子、FBCache/TeaCache 等加速算法、W8A8/W4A16/W8A8C8 量化等。

---

## 代码库结构

```
executor/       # 执行器框架：ModelRunner、模型加载、推理脚本
models/         # 各模型实现（modeling_*.py, runner_*.py, config/）
module/         # 共享基础模块：Linear、MoE GMM、量化、序列并行
ops/            # 自定义算子：AscendC、PyPTO、TileLang
accelerator/    # 加速组件
docs/           # 设计文档、模型文档
dataset/        # 数据集和默认 prompt
contrib/        # 社区贡献模型
scripts/        # 工具脚本
```

---

## 参考模型速查

| 模型特性 | 参考模型 |
|---------|---------|
| 大语言模型（普通） | deepseek-r1, gpt-oss |
| MoE 架构 | deepseek-v3.2-exp, qwen3-moe |
| 长序列（256K+） | kimi-k2-thinking, longcat-flash |
| 视频生成 | hunyuan-video, wan2.2-i2v |
| 图像生成 | hunyuan-image-3.0 |

---

## 硬件平台映射

`npu-smi info` 输出的芯片型号与 Atlas 系列的对应关系：

| Atlas 系列 | 芯片型号 | 单卡 HBM |
|-----------|---------|----------|
| Atlas A2 | Ascend 910B | 32/64 GB |
| Atlas A3 | Ascend 910C | 64 GB |
| Atlas A5 | Ascend 910D | — |

---

## 推荐环境

CANN 8.5.0 + PyTorch 2.8.0 + torch_npu 2.8.0

## 常用命令

```bash
# 环境设置
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
pip3 install -r models/{model_name}/requirements.txt

# 推理执行
cd models/{model_name} && bash infer.sh

# 权重转换（未启用 enable_online_split_weight 时）
bash utils/weight_convert.sh --input_fp8_hf_path /path/to/origin \
    --output_hf_path /path/to/output --quant_mode w8a8

# 性能分析：YAML 中设 enable_profiler: True，结果在 prof/ 目录

# CI 测试
bash executor/scripts/test_all_case.sh

# 多机部署：修改 executor/scripts/set_env.sh 中的 IPs 和 cann_path，各节点执行 infer.sh
```

---

## Skill 路由

| 场景 | Skill |
|------|-------|
| 模型部署基线 | model-infer-migrator |
| 端到端模型优化 | model-infer-optimize（编排入口） |
| KVCache 静态化 / FA 替换 | model-infer-kvcache |
| 融合算子分析与替换 | model-infer-fusion |
| 图模式适配 | model-infer-graph-mode |
| KVCache/FA 精度问题 | model-infer-precision-debug |
| 并行策略分析 | model-infer-parallel-analysis |
| 并行策略实施 | model-infer-parallel-impl |
| NPU 运行时错误诊断 | model-infer-runtime-debug |

---

## 行为约束

- **先理解再行动**：分析或修改模型代码前，先读懂当前实现和模型架构，参考对应 skill 的分析流程，不要基于猜测行动
- **失败时回到 skill**：修复失败后不盲目重试，重新读取对应 skill 的排查流程，按步骤定位根因再动手
- **调用而非重建**：需要 skill 覆盖的工作流，调用对应 skill 按步骤执行，不要凭记忆重建步骤
- **及时持久化**：长任务中关键结论、设计决策、调试发现要及时写入文件（如 progress.md），上下文压缩会丢失未保存的信息
- **前置条件检查**：执行任务前，确认所需信息（模型路径、权重路径、部署配置等）是否可从 YAML 配置或 progress.md 中获取，缺失时向用户确认，不盲目搜索或猜测。单独使用 model-infer 系列 skill 时，若模型尚未完成框架适配或基线采集，应建议用户先通过 model-infer-optimize 完成阶段 0（模型分析与基线建立）
- **工作目录规范**：运行推理、采集基线等操作必须先 `cd` 到模型目录（`models/{model_name}/`）再执行，不要从仓库根目录直接运行，避免日志和输出文件生成在错误位置

---

## 注意事项

- **LICENSE**：新增模型请确保 LICENSE 合规，建议 Apache 2.0 或 MIT
- **代码规范**：提交前执行 `pre-commit run --all-files`
- **OOM 缓解**：减 batch_size → enable_prefill_multi_cycle → moe_chunk_max_len（MoE）→ kvp_size（长序列）→ 换更高压缩量化
