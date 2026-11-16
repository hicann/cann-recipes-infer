# scenario subagent 操作细则

供 scenario subagent 使用。目标是**构造可复现的推理输入、验证精度基线并确定可自动验证的判据**，不实施性能优化。下文以本仓 `models/<model>/` 的标准结构为例，入口以目标模型实际实现为准。

## 0. 工作目录

框架部署在仓库根执行拉起命令，脚本按绝对路径写日志与产物，进程工作目录仍是调用处；独立部署先 `cd models/<model>/` 再跑，避免日志和产物落到错误位置。

## 1. 确认推理入口与场景

- 入口按部署模式定：框架部署在仓库根执行 `bash executor/scripts/infer.sh --model {model_name} --yaml {yaml_name}.yaml`；独立部署执行模型目录自带的 `bash infer.sh`。框架部署会 source `executor/scripts/set_env.sh` 与 `function.sh`，取 `config/` 下的 YAML 后调 `launch` 启动；仓内既有的独立部署模型多数也 source `executor/scripts/` 下的 `set_env.sh` 与 `function.sh` 或 `mm_function.sh`，调 `launch` 或 `mm_launch`，具体以该模型 `infer.sh` 为准。
- 根据已确认的 workload 核对 YAML、权重与实际推理路径，确定阶段、卡数、并行和量化。框架部署用 `--yaml` 选择配置，独立部署沿用模型入口的配置方式。
- 多机场景：节点 IP 和 cann_path 在 `executor/scripts/set_env.sh`，每个节点都要执行上述拉起命令。

## 2. 构造可复现输入

infer 仓输入由提示词来源（`data_config.dataset` 与 `dataset_path`）和形状设置（`scheduler_config.batch_size` / `max_new_tokens`、`data_config.input_truncated_len`）决定。默认 dataset 路径下，`default` 使用 `dataset/default_prompt.json`。不同数据集的预处理分支不同，应记录实际送入模型的 token 长度与 batch。

复现要点，逐项固定并记录：

- **固定输入**：沿用目标场景的数据集、样本或构造方式，记录原文或可复现的来源。
- **固定形状**：记录 batch、输入长度与生成长度配置。infer 仓文本路径中，`input_truncated_len` 也会在 Scheduler 的 tokenize 阶段限制 default 输入；以实际 token 长度核对目标负载。
- **固定解码**：沿用已确认的解码设置，记录 temperature、top_p、top_k 与 seed。infer 仓 `temperature=0` 走贪心路径，top_p / top_k 不参与；采样路径由 `data_config.seed` 提供逐请求随机种子。精度对照另用贪心时单独记录，性能测量仍使用目标场景设置。
- **阶段一致**：记录目标 Prefill / Decode 或其他推理阶段，使后续 profiling 窗口覆盖该阶段。

## 3. 跑通基线并采样输出

```bash
# 框架部署（仓库根执行）
bash executor/scripts/infer.sh --model <model> --yaml <scenario>.yaml

# 独立部署（模型目录执行，先把 infer.sh 的 YAML_FILE_NAME 指向目标场景）
cd models/<model> && bash infer.sh
```

记录正式 generate 阶段的输出，排除 warmup 阶段。框架部署由 `executor/offline/offline_inference.py` 调用 `executor/core/engine/execution_engine.py` 的 `warm_up`，在正式推理前执行预热；独立部署检查对应 `infer.py` 中的 `warm_up=True` 调用。首次运行确认无运行错误且输出正常，保存生成文本 / token 作为基线样本。

## 4. 建立精度 / 功能判定口径

沿用已确认的参照、指标与达标线；缺失时按模态提出可复现的判定方法，交主 agent 确认后记录，后续 round 沿用：

- **LLM**：确定性解码可逐 token 对比，存在差异时结合约定的数值或任务指标判定；核对相对基线新增的文本退化、异常空输出与提前结束。
- **MoE / 量化模型**：记录参考路径与允许的误差范围，区分量化本身的误差和接入后新增偏差。
- **图像 / 视频生成**：固定 seed 和 step 数，比对输出图 / 帧的关键指标或可视一致；记录采样器和 step。

提供可自动判定的规则（逐 token 一致 / 误差阈值 / 指标范围）。

## 5. 保存场景记录

保存到指定输出目录的 `scenario.md`，至少包含：

- 模型 / case、代码目录、推理入口、选用的 YAML（标明 phase / 卡数 / 并行 / 量化）
- 输入：prompt 原文或来源、`input_truncated_len`、`batch_size`、`max_new_tokens`、解码参数
- 基线运行命令（可直接复制重跑）
- 关键环境：`set_env.sh` 的 cann_path / IPs（多机）、torch_npu 版本、卡数
- 完整基线输出或 token 产物路径，场景记录只放摘要
- 判定口径（自动判定规则与阈值）
- 目标性能指标、测量命令与预热范围、基线结果及波动
- 阻塞项（如有）

## 6. 阻塞处理

- 无法运行时返回阻塞原因及缺失信息（权重路径、卡数、依赖等），基线记录以实际运行结果为准。
- 模型尚未完成框架适配或缺少可运行 baseline 时，建议先通过 `model-infer-migrator` 或 `model-infer-optimize` 阶段 0 建立基线，再继续本流程。

## 返回内容

场景记录文件路径 / 输入样本（或构造脚本）路径 / 基线推理命令 / 精度口径摘要 / 性能基线摘要与证据路径 / 阻塞项（如有）。
