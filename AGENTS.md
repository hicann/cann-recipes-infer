# cann-recipes-infer — NPU 推理优化

cann-recipes-infer 是基于 CANN 平台的 LLM/多模态模型离线推理优化样例库，面向昇腾 Atlas A2/A3 硬件。支持 TP/EP/DP 并行策略组合，提供 `eager`、`npugraph_ex` 和 `ge_graph` 三种执行模式，新接入图模式优先评估 `npugraph_ex`。核心特性包括多流并行、融合算子、FBCache/TeaCache 等加速算法、W8A8/W4A16/W8A8C8 量化等。

---

## Agent 使用说明

Agent skill 与 subagent 源码位于 `.agents/`。首次克隆后根据使用的客户端运行对应命令，默认安装知识库查询技能并生成客户端视图：

- Codex：`bash scripts/init-agent.sh --codex`
- Claude Code：`bash scripts/init-agent.sh --claude`
- OpenCode：`bash scripts/init-agent.sh --opencode`

复杂模型优化任务可指定工作流与角色分工：

```text
调用 model-infer-optimize，对 models/{model_name} 做端到端 NPU 推理优化。
按 model-infer-analyzer / model-infer-implementer / model-infer-reviewer 三个 agent 执行，并通过 progress.md 传递状态。
```

---

## 知识库

cannbot-knowledge 提供 CANN、模型部署与推理优化、算子、图模式和 Runtime 的技术知识与工程经验。Agent 可根据任务需要，通过已安装的 `knowledge-query` skill 查询参考实现、接口约束、优化方法和故障排查经验。

- 查询入口、知识根定位和参数用法遵循 `knowledge-query` skill，使用项目已配置的知识仓。
- 根据当前问题选择相关卡片，阅读全文，按需追踪关联知识和来源。
- 采用结论前核对目标框架、平台、版本及适用条件，并结合当前代码和实际运行结果判断。
- 启用 `contributor` 安装模式后，可通过 `knowledge-ingest` 沉淀已验证、可复用的工程经验，附上适用条件和验证证据；安装选项见 [.agents/README.md](.agents/README.md#安装)。

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
integration/custom/        # 社区贡献模型
scripts/        # 工具脚本
```

---

## 参考模型速查

当前模型存在两种方式，**统一流程**（推荐优先采用，参考 [统一流程方案](docs/design/executor_design.md)）与**ModelRunner**（逐渐废弃）。

| 流程 | 类别 | 模型 |
|---|---|---|
| **统一流程** | Dense LLM | `gpt_oss`, `qwen25_7b_instruct`, `qwen3_8b`, `deepseek_v3_2_exp`, `gemma_4` |
| | MoE LLM | `deepseek_r1`, `glm_5`, `kimi_k2_thinking`, `qwen3_moe`, `qwen3.5`, `hy3`, `step3p7_flash`, `longcat_flash`, `longcat_flash_lite`, `deepseek_v4` |
| **ModelRunner** | LLM (Dense / MoE) | `pangu_7b` |
| | 视频生成 | `hunyuan-video`, `wan2.2-i2v`, `sana-video` |
| | 图像生成 | `hunyuan-image-3.0` |
| | 推荐 / 其他 | `hstu` |

---

## 硬件平台映射

Atlas 系列、产品族与运行时可探测标识的对应关系（`asys info -r=status` 与 `torch.npu.get_device_name()` 输出的是 SoC 典型标识列）：

| Atlas 系列 | 产品族（Chip Name） | SoC 典型标识 | 单 die HBM | 每物理卡 die 数 | 每整机物理卡数 |
|-----------|---------|---------|----------|---------------|---------------|
| Atlas A2 | Ascend 910B 系列 | `910B` | 32 / 64 GB | 1 | 8 |
| Atlas A3 | Ascend 910C 系列 | `910_93` | 64 GB | 2 | 8 |
| Atlas A5 | Ascend 950 系列（PR/DT） | - | 112 / 128 GB | 1 | 8 |

> **口径**：skill 中描述的"卡"常用部署拓扑口径（= die = rank，一个进程一份 HBM），不等同于物理口径。A3 是特例，一物理卡 = 2 die。
>
> 单 die HBM 列为常见 SKU 容量，实际可能含其他规格变体。

---

## 推荐环境

CANN 9.0.0 + PyTorch 2.8.0 + torch_npu 2.8.0 + transformers 5.0.0

## 常用命令

```bash
# 环境设置
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
pip3 install -r models/{model_name}/requirements.txt

# 推理执行（框架部署，在仓库根目录执行）
bash executor/scripts/infer.sh --model {model_name} --yaml {yaml_name}.yaml

# 推理执行（独立部署，模型目录自带 infer.sh）
cd models/{model_name} && bash infer.sh

# 离线权重转换（未启用 enable_online_split_weight 时，按目标模型脚本；下例为 FP8 → W8A8）
bash models/{model_name}/utils/weight_convert.sh --input_fp8_hf_path /path/to/origin \
    --output_hf_path /path/to/output --quant_mode w8a8

# 性能采集：YAML 中设 model_config.enable_profiler: True，结果在本次运行的 <res_path>/prof/ 下

# CI 测试
bash executor/scripts/test_all_case.sh

# 多机部署：修改 executor/scripts/set_env.sh 中的 IPs 和 cann_path，各节点执行对应部署模式的拉起命令
```

---

## Skill 路由

| 场景 | Skill |
|------|-------|
| 模型部署基线 | model-infer-migrator |
| 端到端模型优化 | model-infer-optimize |
| KVCache 静态化 / FA 替换 | model-infer-kvcache |
| 融合算子分析与替换 | model-infer-fusion |
| 量化适配改造 | model-infer-quantization |
| 图模式适配 | model-infer-graph-mode |
| 模型推理精度问题（含 KVCache/FA） | model-infer-precision-debug |
| 并行策略分析 | model-infer-parallel-analysis |
| 并行策略实施 | model-infer-parallel-impl |
| NPU 运行时错误诊断 | model-infer-runtime-debug |
| 多流并行优化 | model-infer-multi-stream |
| 权重预取适配 | model-infer-prefetch |
| SuperKernel 自动调优 | [superkernel-auto-tune](https://gitcode.com/cann/graph-autofusion/blob/master/.claude/skills/superkernel-auto-tune/SKILL.md)（graph-autofusion 外部工作流；按[复用指南](https://gitcode.com/cann/graph-autofusion/blob/master/docs/zh/skill-reuse-guide.md)准备完整技能资源后执行） |
| 性能数据采集 | model-infer-profiling |
| 性能数据拆解 | model-infer-perf-breakdown |
| baseline 之上的进阶性能调优 | model-infer-sota-approach |

---

## 行为约束

- **先理解再行动**：分析或修改模型代码前，先读懂当前实现和模型架构，参考对应 skill 的分析流程，不要基于猜测行动
- **按证据排障**：修复失败后结合日志与代码定位原因，按需查阅对应 skill 或知识库的排查经验
- **复用 skill**：任务有对应 skill 时，读取相关指引，按本次分析、实施或验证范围使用
- **及时持久化**：长任务中关键结论、设计决策、调试发现要及时写入文件（如 progress.md），上下文压缩会丢失未保存的信息
- **核实事实配置**：硬件型号、卡数、die 数、显存容量、模型与权重路径等，从任务信息、配置、已有记录或运行环境核实，记录到 progress.md；未决事项交调用方确认
- **前置条件检查**：按本次任务核对模型、权重与部署配置；涉及运行验证且缺少可运行模型或基线时，先通过 model-infer-migrator 或 model-infer-optimize 阶段 0 补齐
- **工作目录规范**：框架部署在仓库根目录执行 `bash executor/scripts/infer.sh --model {model_name} --yaml {yaml_name}.yaml`，脚本按绝对路径把日志与产物写到 `models/{model_name}/` 下，进程工作目录仍是调用处；独立部署先 `cd` 到模型目录（`models/{model_name}/`）再执行 `bash infer.sh`，避免日志和输出文件生成在错误位置
- **证据优先**：结论依据代码、需求、文档或运行结果，说明证据来源，区分已验证事实与待验证判断

---

## 注意事项

- **LICENSE**：新增模型请确保 LICENSE 合规，建议 Apache 2.0 或 MIT
- **代码规范**：提交前执行 `pre-commit run --all-files`
- **OOM 定位**：通过 model-infer-runtime-debug 区分权重、KVCache、激活与临时 buffer 占用，再按目标负载选择调整方案
