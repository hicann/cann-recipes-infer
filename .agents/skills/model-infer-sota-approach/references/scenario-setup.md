# scenario subagent 操作细则

供 scenario subagent 使用。目标是**建立可复现的推理输入 + 跑通精度基线 + 定一把可机判的判定口径**，不做任何性能优化、不改优化代码。下面以本仓 `models/<model>/` 的标准结构为准；个别模型入口命名略有差异，按实际为准。

## 0. 工作目录纪律

框架部署在仓库根执行拉起命令，脚本按绝对路径写日志与产物，进程工作目录仍是调用处；独立部署先 `cd models/<model>/` 再跑，避免日志和产物落到错误位置。

## 1. 锁定推理入口与场景

- 入口按部署模式定：框架部署在仓库根执行 `bash executor/scripts/infer.sh --model {model_name} --yaml {yaml_name}.yaml`；独立部署执行模型目录自带的 `bash infer.sh`。框架部署会 source `executor/scripts/set_env.sh` 与 `function.sh`，取 `config/` 下的 YAML 后调 `launch` 启动；仓内既有的独立部署模型多数也 source `executor/scripts/` 下的 `set_env.sh` 与 `function.sh` 或 `mm_function.sh`，调 `launch` 或 `mm_launch`，具体以该模型 `infer.sh` 为准。
- **选场景 = 选 YAML**。`config/` 下的 YAML 文件名编码了关键场景维度，例如 `decode_r1_rank_16_16ep_a8w8.yaml` = decode 阶段 / 16 卡 / 16 EP / W8A8，`prefill_r1_rank_32_32dp_32ep_a8w8.yaml` = prefill / 32 卡 / DP+EP。按要优化的 workload 选对应 YAML，框架部署改拉起命令的 `--yaml`，独立部署改模型目录 `infer.sh` 里指向 yaml 的变量。
- 多机场景：节点 IP 和 cann_path 在 `executor/scripts/set_env.sh`，每个节点都要执行上述拉起命令。

## 2. 构造可复现输入

输入由两部分决定：提示词来源（`data_config.dataset` 与 `dataset_path`，取 `default` 时读 `dataset/default_prompt.json`）+ YAML 里的形状设置（`scheduler_config.batch_size` / `max_new_tokens`，以及 `data_config.input_truncated_len`）。框架部署由 `executor/offline/infer.py` 的 `generate_prompt` 取提示词，再由同文件的 `preprocess_prompts_for_scheduler` 按形状复制并截断 batch；独立部署按各自 `infer.py` 的实现。

复现要点，逐项固定并记录：

- **固定提示词**：用 `dataset/default_prompt.json`，或写死一个自定义 prompt，把原文记进场景记录。
- **固定形状**：`scheduler_config.batch_size`、`scheduler_config.max_new_tokens` 写进 YAML 并记录。`data_config.input_truncated_len` 在 `dataset` 取 LongBench 这类需截断的数据集时经 `preprocess_prompts_for_scheduler` 生效；取 `default` 时走另一条分支，输入长度由 prompt 本身决定。
- **固定解码**：把 `data_config.temperature` 设为 0 走贪心解码，此时 `top_p` / `top_k` 不参与，保证逐 token 可复现。框架部署不设 torch 全局种子，逐请求随机性由 `data_config.seed` 经 `executor/core/scheduler/scheduler.py` 传给 per-request generator，贪心路径不取该 generator；独立部署在各自 `infer.py` 里用 `torch.manual_seed` 固定，按该文件实际设置记录。
- **阶段一致**：prefill 场景用 prefill YAML、decode 场景用 decode YAML，不要混；这一步定的阶段要和后续 profiling 采集的阶段一致。

## 3. 跑通基线并采样输出

```bash
# 框架部署（仓库根执行）
bash executor/scripts/infer.sh --model <model> --yaml <scenario>.yaml

# 独立部署（模型目录执行，先把 infer.sh 的 YAML_FILE_NAME 指向目标场景）
cd models/<model> && bash infer.sh
```

取正式 generate 段的输出，warmup 段不计入。框架部署的 warmup 由 `executor/core/engine/execution_engine.py` 的 `warm_up` 在正式推理前单独跑一轮，`executor/offline/offline_inference.py` 调用它；独立部署看各自 `infer.py` 是否先以 `warm_up=True` 跑一轮。第一次先确认无 crash、吐字正常，再把生成的文本 / token 采样存下，作为基线输出样本。

## 4. 建立精度 / 功能判定口径

按模态选一把**可机判**的尺子，写进场景记录，后续所有 round 都用同一把：

- **LLM（贪心可复现）**：以基线输出的 token ids / 文本做逐字对比；附最低可用性门槛——可读、不重复、非全零、不提前 EOS。
- **MoE / 量化模型**：同上；量化基线允许与浮点有界误差，记录可接受的误差范围。
- **图像 / 视频生成**：固定 seed 和 step 数，比对输出图 / 帧的关键指标或可视一致；记录采样器和 step。

不要只写"看起来对"——要给出能自动判定的规则（逐 token 一致 / 误差阈值 / 指标范围）。

## 5. 落盘场景记录

落到 `optimization-analysis/<case>/scenario.md`，至少包含：

- 模型 / case、代码目录、推理入口、选用的 YAML（标明 phase / 卡数 / 并行 / 量化）
- 输入：prompt 原文或来源、`input_truncated_len`、`batch_size`、`max_new_tokens`、解码参数
- 基线运行命令（可直接复制重跑）
- 关键环境：`set_env.sh` 的 cann_path / IPs（多机）、torch_npu 版本、卡数
- 基线输出样本（截断，够复核即可）
- 判定口径（机判规则 + 阈值）
- 阻塞项（如有）

## 6. 阻塞处理

- 跑不通时只回报阻塞原因 + 需要补的信息（权重路径、卡数、缺依赖等），不要绕过或伪造基线。
- 若模型还没适配进框架、或没有可运行 baseline，属于本流程的前置条件未满足：建议先走 `model-infer-migrator` 或 `model-infer-optimize` 阶段 0 建立 baseline，再回到本流程。

## 只回传

场景记录文件路径 / 输入样本（或构造脚本）路径 / 基线推理命令 / 精度口径摘要 / 阻塞项（如有）。
