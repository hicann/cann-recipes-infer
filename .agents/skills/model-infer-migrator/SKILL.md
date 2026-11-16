---
name: model-infer-migrator
description: 基于 PyTorch 框架的昇腾 NPU 模型推理适配与部署基线技能。支持两种部署模式：框架部署模式（接入 cann-recipes-infer 的 executor/core/）和独立部署模式（自管 Runner 不依赖框架）。从 HF 链接或本地代码适配为可运行的标准模型目录，并采集性能基线。触发场景：新模型适配到昇腾 NPU 推理框架、已有模型的部署基线采集、模型迁移和初始跑通验证。
---

# 模型适配与部署基线技能

把 HF 模型或本地代码适配到 infer 仓的两种部署模式之一，产出符合规范的模型目录和标准化性能基线。

本技能聚焦**单卡基础适配 + 部署基线采集**（eager 模式、PyTorch 标准算子），不涉及并行化、KV 模式优化、融合算子、图模式等性能改造。这些由后续 skill（parallel-impl / kvcache / fusion / graph-mode）接手。

---

## 部署模式判定

**框架部署模式（推荐）**：模型声明契约接入 `executor/core/`，调度 / KV 管理 / 批组装 / 采样 / PD 分离 / MTP / profiler 等公共流程由框架负责。模型通过 `support_models.py` 注册。

**独立部署模式**：模型不接 `executor/core/`，自管 Runner / KV cache / 推理脚本。适用仓外项目复用 skill 或框架契约不适用的特殊算法。

优先采用任务中已明确的部署模式。未指定时，已注册到 `executor/core/support_models.py` 的模型走框架部署，自管 Runner 和推理循环且不接入 `executor/core/` 的模型走独立部署。**新模型或仍无法判定时，默认优先框架部署。**

---

## 共通工作流

已有部署分析或适配方案时，从任务说明或 `progress.md` 读取并沿用已确认结论，只补查缺失或与当前实现不一致的信息。

### Step 1: 环境检测 + 部署模式判定

确认 NPU 可用（`npu-smi info`）、torch 和 torch_npu 版本匹配。NPU 环境不可用时停止试跑并报告原因。

### Step 2: 场景判断

已有匹配当前状态的成功运行记录时，直接进入 Step 4。框架部署缺少模型注册或 YAML、独立部署缺少启动脚本时，进入 Step 3 补齐部署文件。其余情况按 Step 4 的入口试跑。成功或单卡 OOM 时按该步交付，其他模型错误进入 Step 3 诊断修复。

### Step 3: 补齐或修复部署代码

跳到对应大节执行：
- 框架部署模式 → 「## 框架部署模式」适配要点
- 独立部署模式 → 「## 独立部署模式」适配要点

### Step 4: 试运行 + 基线采集

已有日志（包括 Step 2 的试跑日志）对应的代码、配置、权重和环境与本次基线一致，且数据完整时，可直接提取；否则按部署模式拉起：
- **框架部署**：`bash executor/scripts/infer.sh --model {model_name} --yaml {yaml_name}.yaml`
- **独立部署**：模型目录内执行 `bash infer.sh`

按结果分流：

- **跑通**：用 `dataset: "default"`（读 `dataset/default_prompt.json`）采集基线
  ```bash
  python3 .agents/skills/model-infer-migrator/scripts/collect_baseline.py \
      --log-file {log_path} \
      --output models/{model_name}/agentic/baseline/baseline_metadata.json \
      --yaml-file {yaml_path} \
      --rank 0
  ```
  在仓库根执行。输出吐字正常（可读、不重复、长度合理），baseline_metadata.json 中的 output_text 需记录完整输出。migrator 阶段是单卡，只产 `log_0.log`，故读 rank 0；多卡场景改读 rank 1（rank 0 经 tee 可能截断）。

- **OOM（显存不足）**：本阶段输出代码骨架并标记"需多卡"，由 optimize 编排进入并行化阶段；并行 + kvcache 等优化跑通后由编排层补采基线。产物：多卡占位 YAML（待 parallel-impl 接入）+ 空 `agentic/baseline/` 目录 + README 标注 OOM 预期与下游 skill 任务；单卡 YAML 跑不通时不必保留。

- **其他错误**：可查询 cannbot-knowledge 中的依赖兼容、权重加载和推理启动故障排查经验。NPU 运行时错误 → model-infer-runtime-debug；精度异常 → model-infer-precision-debug。

日志路径：
- 框架部署模式：`models/{model_name}/res/{date}/{model_name}_{yaml_stem}/log_{rank}.log`（由 launch 按 `WORK_DIR/RES_PATH` 创建，多卡读 rank 1）
- 独立部署模式：`infer.py` 输出到 stdout / 自定义日志路径

---

## 产物文件结构

两模式的 YAML schema 同为 4 段式，公共文件命名一致；启动入口与 Runner 按部署模式区分：

```
{output_dir}/{model_name}/
├── config/{model_name}.yaml
├── models/
│   ├── configuration_{model_name}.py
│   └── modeling_{model_name}.py
├── infer.py                      # 独立部署自带；框架离线启动优先使用已有的模型入口，缺省用 executor/offline/infer.py
├── infer.sh                      # 仅独立部署模式自带；框架部署模式统一走 executor/scripts/infer.sh
├── runner_{model_name}.py        # 独立模式独有
├── requirements.txt
├── README.md
└── agentic/baseline/baseline_metadata.json
```

> 注意：`models/{model_name}/models/` 下**不要放 `__init__.py`**。

---

## 框架部署模式

接入 `executor/core/` 框架，模型走 `InferenceConfig + CommManager + cache_unit` 契约，由统一启动脚本拉起。本阶段为最简部署基线，KV 先用 Legacy 模式跑通；PA 等进一步改造由 kvcache 优化阶段处理。

### 适配要点

模型类与 Attention 骨架见 `references/framework_templates.md`。可查询 cannbot-knowledge 中的模型接入、权重加载和部署样例，并结合当前 `executor/core/support_models.py` 注册表和模型源码选择参考实现。

1. **复制 HF 实现到 `models/{model_name}/models/`**：vendoring HF modeling 和 configuration（**禁止 `from transformers import 模型类`**）。删除训练专用代码，保留 PyTorch 标准算子。modeling 类基类改为 `nn.Module`。
   - **命名规则**：目录 / 文件 / 模块名用下划线
   - **HF 依赖剪裁**（通用替换）：
     - `Cache` / `DynamicCache` → 删，用 Legacy `self.k_cache` / `v_cache`
     - `create_causal_mask` → 删，用 `forward_metadata.attention_mask`
     - `PreTrainedModel` / `GradientCheckpointingLayer` / `ALL_ATTENTION_FUNCTIONS` → 删，改为 `nn.Module`
     - 纯函数（RMSNorm / rotate_half / RoPE 工具）→ 保留

2. **改造 ForCausalLM 构造签名**为 `(config, infer_config, comm_manager, prefix)`。读取参数从 `runner_settings.get(...)` 改为 `infer_config.parallel_config.attn_tp_size` 等点访问；通信组从 `comm_manager.get_group(...)` 取。

3. **改造 forward 签名**为 `(input_ids, position_ids, forward_metadata, **kwargs)`。框架只透传这几个 key，`slot_mapping` / `block_table` 从 `forward_metadata` 取再下发给子层。input_ids 是 packed `[TotalTokens]`，返回 logits 的布局见模板。

4. **KV 路径选择**（按场景二选一，互斥；框架按是否存在 `get_cache_info()` 分发）：
   - **路径 A — Legacy + HF SDPA**（默认推荐 — 单卡 + 等长 batch baseline，最简）
     - Attention 声明 `self.cache_unit = (num_kv_heads_per_rank * head_dim,)`，框架按 `(batch, seq, *cache_unit)` 预分配 `k_cache` / `v_cache`；模型 forward 内用标准 indexing 或 `torch_npu.scatter_update_` 写 cache
     - 顶层 forward 入口对 packed 1D `input_ids` 做 BSND reshape 回 `[B, S]`（仅等长 batch 适用）
     - 不实现 `get_cache_info()` / 不声明 `cache_entries`
   - **路径 B — 最简 Paged + HF SDPA**（多卡 / 变长 batch / 用户明确指定 Paged 起步时使用）
     - 实现 `get_cache_info()` + `cache_entries`（`FullAttention`，按实际 KV head 维度，不引入 MLA latent 压缩）
     - 写 cache 用 `npu_scatter_nd_update_` 接 `forward_metadata.slot_mapping[attn_type]`；attention 读 cache 后用 HF 原版 SDPA / 标准 PyTorch（**不**上 FA 算子）
     - 顶层 forward 直接走 packed 1D，无需 BSND 形态适配
   - **共通约定**：
     - 两路径互斥，选定后整套代码（cache 写入、attention 读法、forward 输入形态）保持一致
     - reference 已注册模型的 `get_cache_info()` 是 model-infer-kvcache skill 升级后的产物（含 MLA absorb / FA / nope+rope 双 entry），**不要整段照搬**到 migrator 阶段
     - 模型自有 cache（非 KVCache，如 N-gram / hash buffer / speculative buffer）由 model class 自管，不上报 `cache_entries`
     - FA 算子接入 / MLA absorb / 复杂 attn_type（SlidingWindow / 混合）/ latent 压缩 由 model-infer-kvcache skill 接手

5. **实现权重契约**：`load_weights(weights)` 接 HF checkpoint stream；`process_weights_after_loading()` 触发 `quant_method.process_weights_after_loading`（MoE / 量化模型必须实现）。

6. **注册到 `executor/core/support_models.py`**：
   ```python
   "{model-key}": [
       ("models.{model_name}.models.modeling_{model_name}", "{ModelName}ForCausalLM"),
       ("models.{model_name}.models.configuration_{model_name}", "{ModelName}Config"),
   ],
   ```
   条目是 `(模块路径, 类名)` 字符串对列表，由 `load_model_classes` 惰性加载。含 MTP 头时把 MTP 类作为第二项插在中间。

7. **写 YAML、requirements、README**：YAML 模板与命名规范见 `references/yaml_template.md`，README 模板见 `references/readme_template.md`。框架部署模型**不建模型目录下 `infer.sh`**，拉起统一用 `bash executor/scripts/infer.sh --model {model_name} --yaml {yaml_name}.yaml`。`requirements.txt` 优先对齐仓内已注册同架构模型（以当前注册表和实现为准），缺失的再按 HF 原仓补；避免引入与既有模型冲突的版本约束。

> migrator 输出框架契约骨架 + Legacy KV + HF 原版 attention 的 `world_size=1` 单卡代码。FA 算子改造、Paged 改造、复杂架构 KV、多卡部署 / 权重切分由 model-infer-kvcache / model-infer-parallel-impl skill 接手；本阶段不做。

---

## 独立部署模式

不依赖 cann-recipes-infer 框架，模型自管 Runner / 推理脚本。仍可复用 `module/linear.py` 的 ParallelLinear（独立加载）和 `executor/model_loader/weight_utils.py` 的权重加载工具。

### 适配要点

1. **复制 HF 实现到 `models/{model_name}/models/`**：vendoring HF modeling 和 configuration，删除训练专用代码。

2. **改造 modeling forward 接口**为自定义参数（不用 ForwardMetaData）：
   ```python
   def forward(self, input_ids, position_ids=None,
               past_key_values=None, kv_len=None,
               attention_mask=None, is_prefill=True)
   ```
   Attention 层接收 `past_key_value` + `kv_len`，用 `torch_npu.scatter_update_` 更新 KV。decode 分支 `attention_mask=None`。

3. **写自定义 Runner 骨架**（`runner_{model_name}.py`，不继承 ModelRunner）：
   - `__init__(yaml_file_path)`: 解析 yaml；加载 tokenizer / config / model；预创建 mask；预计算 RoPE
   - `init_static_kvcache`: 最简占位（每层一对 (k, v) tensor，BSH layout，简单 zeros 分配）
   - `model_generate(prompts, warm_up=False)`: tokenize batch → prefill → decode loop → detokenize；日志格式 `{model_name} inference time cost of {stage} is X ms` 与仓库对齐

4. **写 infer.py**：实例化 Runner（传 yaml 路径），先 `model_generate(prompts, warm_up=True)` 预热，再 `warm_up=False` 正式跑。

5. **写 infer.sh、requirements、README**：模板见 `references/standalone_templates.md`。

> migrator 输出 Runner 骨架 + 最简 KV 占位 + HF 原版 attention 的单卡代码。FA 算子接入、PA 启用、SlidingWindow / MLA absorb 等优化由 model-infer-kvcache skill 接手；多卡 `dist.*` + ParallelLinear 替换由 model-infer-parallel-impl skill 接手。

---

## 权重管理（共通）

通过 YAML 的 `model_config.model_path` 指定权重路径。

> migrator 阶段单卡跑通即可；并行权重切分、在线/离线切分由 model-infer-parallel-impl skill 处理。

---

## baseline_metadata.json 格式（共通）

```json
{
  "timestamp": "<ISO8601>",
  "yaml_path": "config/{model_name}.yaml",
  "environment": {
    "npu_model": "<npu_model>",
    "num_cards": <int>,
    "cann_version": "<x.y.z>",
    "pytorch_version": "<x.y.z>",
    "torch_npu_version": "<x.y.z>",
    "exe_mode": "<eager|ge_graph|npugraph_ex>"
  },
  "model_config": {
    "model_name": "<model_name>",
    "model_source": "<hf_url_or_local_path>"
  },
  "performance": {
    "prefill_ms": <float>,
    "decode_avg_ms": <float>,
    "output_text": "<sample_output>"
  }
}
```

> `yaml_path` 指向运行时使用的 YAML，`parallel_config` / `scheduler_config` 等并行/调度配置不重复嵌入 JSON。`decode_avg_ms` 须剔除首步冷启（编译落地开销，约 30-50ms 偏高）。分析推算（非 runtime 实测）的性能字段加 `_estimated` 后缀以示区分。

---

## 结束条件

- **单卡跑通**：推理输出正常，已生成实测的 `agentic/baseline/baseline_metadata.json`。
- **单卡 OOM**：交付可继续并行适配的模型骨架，标记“需多卡”，基线待并行化后补采。

---

## 参考文档索引

| 文档 | 路径 | 适用模式 |
|------|------|---|
| YAML 配置模板与命名规范 | `references/yaml_template.md` | 共通 |
| README 模板 | `references/readme_template.md` | 共通 |
| 框架部署模型类与 Attention 模板 | `references/framework_templates.md` | 框架部署 |
| 独立部署 Runner 与 infer.py/infer.sh 模板 | `references/standalone_templates.md` | 独立部署 |
| 基线采集脚本 | `scripts/collect_baseline.py` | 共通 |
| InferenceConfig 字段语义（仓库框架） | `docs/common/inference_config_guide.md` | 框架部署 |
| 框架架构总览（仓库框架设计） | `docs/design/executor_design.md` | 框架部署 |
| KV cache 设计（仓库框架的 paged 实现） | `docs/design/kv_cache_design.md` | 框架部署 |
| 在线推理 / PD 分离（仓库框架机制） | `docs/design/online_inference_design.md` | 框架部署 |
| MTP 投机采样（仓库框架流程） | `docs/design/mtp_design.md`、`docs/common/mtp_model_guide.md` | 框架部署 |
| 在线切分权重（仓库框架流程） | `docs/common/online_split_weight_guide.md` | 框架部署 |
