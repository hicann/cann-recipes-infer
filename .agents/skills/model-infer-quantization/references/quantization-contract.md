# 量化产物接入契约与运行机制

本卡描述 infer 仓 `compressed-tensors` 主线接入侧的事实：产物最小交付物、配置字段、张量语义、运行对象映射，以及本仓量化运行机制。

## 1. 最小交付物

| 文件 | 要求 |
|------|------|
| `config.json` | 包含 `quantization_config` 或等价 `compression_config` |
| `model.safetensors.index.json`（如有） | 能索引量化张量和保留张量 |
| `*.safetensors` | 包含量化权重、scale、bias 等运行所需张量 |
| `deploy_quantization.md` | 写清量化对象、张量语义、回退策略和下游加载要求 |

量化范围或张量语义无法确定、权重不完整时，不进入改造，输出产物契约补充诉求。

## 2. `config.json` 字段要求

必须能读出：

- `quant_method: compressed-tensors`
- `config_groups`
- `targets`
- `ignore`（可省略，运行时默认为空列表）
- 权重量化格式、激活量化格式、dtype
- 可选 `kv_cache_scheme`

禁止 infer 侧因报错擅自修改 `targets`、`ignore`、量化 dtype 或张量语义后宣称成功。

## 3. 张量语义要求

从权重文件或索引定位张量，结合部署说明核对以下内容：

- 权重张量，例如 `qweight`
- scale 张量，例如 `weight_scale`
- 可选 bias / zero point / smooth scale
- 张量名到 infer 模型参数名前缀的映射关系
- 未量化模块的保留权重

张量语义不清时，不能猜测落代码；输出补充诉求。

**ignore 完备性启发式**：若产物中存在既不在 `ignore`、又没有对应 `weight_scale` 的 `Linear` 候选模块，说明 `ignore` 可能漏列 implicit BF16 模块，或 scale 丢失。命中此启发式时按 SKILL 第六步 §6.6 判定处理，不擅自决策。

## 4. 运行对象映射

优先把量化目标映射到 infer 已有运行对象：

| 量化目标 | infer 运行对象 |
|----------|----------------|
| Dense Linear | `Linear` / `ReplicatedLinear` / `RowParallelLinear` |
| Dense MLP gate/up/down | 可量化 Linear；必要时 post-load 恢复 `gate/up` 融合 |
| Attention q/k/v/o | 可量化 Linear；必要时 post-load 恢复 `q/k/v` 融合 |
| MoE Expert gate/up/down | `MoEGMM` / `FusedMoEGMM`；产物映射与运行时能力均支持时，可选 per-expert Linear |
| KV Cache | `kv_cache_scheme` / cache scale / cache runtime |
| `ignore` 模块 | `UnquantizedLinearMethod` 浮点回退，不参与量化收益统计 |

无法映射时，按正文 §4.3 区分运行时能力缺口与产物契约缺失。

## 5. infer 仓框架接入

新模型由 `executor/core/model_worker/model_worker.py` 调用 `get_quant_config` 解析配置，并向模型透传 `hf_config.quant_config`。接入已有模型时核对其实际加载入口，少数早期模型在 runner 中完成这一步。

量化层凭 prefix 和 target 选择 quant method，注册权重与 scale，再按分片加载。框架通过 `_process_weights_after_loading` 触发模型的 `process_weights_after_loading`，将权重和 scale 整理成 forward 所需的布局及 dtype。
