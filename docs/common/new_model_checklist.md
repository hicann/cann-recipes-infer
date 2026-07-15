# 新模型合入 Checklist

本文提供一份新 LLM 模型合入本仓库的 Checklist，分为两个部分：

- **一、功能与精度**：验证模型跑通离线推理、在线推理，并通过数据集精度验证，涉及 Packed Sequence、KV Cache 管理、npugraph_ex 图模式优化等关键事项。
- **二、规则约束**：从版本、环境变量、YAML、文档、量化、公共代码等方面约束模型的接入方式，确保符合仓库规范。

新模型基于框架开发后，在进行合入前需要按清单完成并验证以上两个部分的关键事项。

# 一、功能与精度

## 1. 基础接入

- [ ] 检查环境版本，验证模型在对应 README 中声明的环境下功能和精度正常。
- [ ] 参考[框架设计文档](../design/executor_design.md#6-框架对模型的接口契约)，在 `models/<model_name>/` 下提供模型实现、配置类、README文档和 YAML 示例，参考[配置参数说明](inference_config_guide.md)和[示例](../../models/deepseek_r1/config/decode_r1_rank_16_16ep_a8w8.yaml)，YAML 至少覆盖：
  - `model_config.model_name`
  - `model_config.model_path`
  - `model_config.exe_mode`
  - `data_config.input_truncated_len`
  - `scheduler_config.batch_size`
  - `scheduler_config.max_new_tokens`
  - `scheduler_config.max_prefill_tokens`
  - `parallel_config.world_size`
  - 根据切分方式配置 `parallel_config.attn_tp_size`、`moe_tp_size`、`embed_tp_size`、`lmhead_tp_size` 等。

## 2. KV Cache 管理

- [ ] 参考 [KV Cache 管理设计文档](../design/kv_cache_design.md)，选择 `KVCacheManager` 管理（推荐使用）或模型自行维护 Cache 的方式，并完成对应 Cache 管理适配。

## 3. Packed Sequence

- [ ] 验证在多 batch 输入（batch_size_per_dp_rank > 1）且序列不等长的场景下，模型功能和吐字精度正常。

## 4. npugraph_ex 图模式优化

- [ ] 模型需要支持 npugraph_ex 图模式优化，至少有一个配置 YAML 启用了 `model_config.exe_mode: npugraph_ex`。
- [ ] `npugraph_ex` 在 warmup 阶段的首次编译功能正常。
- [ ] 正式推理时，decode 阶段能直接复用 warmup 编译的图执行，不能出现重编译的现象。
- [ ] 使能 MTP 时，主模型和 MTP 模型都能正常编译。

## 5. Online 推理（推荐）

具体设计思路可参考[online推理设计文档](../design/online_inference_design.md)和[online推理流程](../design/executor_design.md#5-在线推理流程pd-分离)。

- [ ] online 必须依赖**框架托管的 Cache 管理**，且必须在 offline 模式下跑通，再适配 online 功能。
- [ ] 在 `models/<model_name>/config/` 目录下提供 online 默认启动配置：`<model_name>_pd/prefill.yaml` 和 `<model_name>_pd/decode.yaml`。
- [ ] 跑 online 单角色服务接口，验证服务拉起功能正常。
- [ ] 跑 online prefill/decode/router 全链路，验证发送请求功能精度正常。

## 6. 数据集评分（推荐）

框架提供了数据集评分功能用于验证新模型在 **online 服务**形态下的端到端精度。当前框架提供 OpenAI 兼容接口，可以用 evalscope 进行精度测评，具体执行细节见[请求方式](../design/executor_design.md#55-请求方式)。

- [ ] 跑数据集精度测评，例如 GSM8K 数据集，得到正常的评分。

# 二、规则约束

## 1. 版本统一

- [ ] 统一执行版本，复用仓库多数 LLM 模型的环境依赖，Python/CANN/Torch 版本可参考 [DeepSeek-R1](../../models/deepseek_r1/README.md#环境准备)，transformers 等第三方库版本可参考 [requirements.txt](../../models/deepseek_r1/requirements.txt)，非必要不新增第三方库。

## 2. 环境变量

- [ ] 复用框架已有环境变量，如日志级别统一使用 `CANN_RECIPES_LOG_LEVEL`，不自行定义等价开关。
- [ ] 非必要不新增环境变量；确需新增时，在模型 `README.md` 中说明其含义、取值范围与默认行为。

## 3. 文档规范

- [ ] 用户指导文档放在 `models/<model_name>/README.md`，技术优化点放在 `docs/models/<model_name>/` 下的技术文档。
- [ ] `README.md` 站在用户视角还原完整使用流程（环境准备、权重转换、推理执行等），确保步骤无缺失或错误，可被其他开发者复现。
- [ ] 影响用户界面的修改（尤其是量化、版本、参数相关指导）必须同步刷新文档。

## 4. YAML 配置与自定义字段

- [ ] 模型特有配置统一收敛到 `model_config.custom_params` 字典中，仅对本模型生效，不得改动 `data_config`、`parallel_config`、`scheduler_config` 等现有的公共配置字段。
- [ ] `model_config.custom_params` 中的每个字段均需在模型 `README.md` 中解释含义与取值。
- [ ] YAML 数量尽量精简，避免因单一开关差异复制出多份近似 YAML，例如，MTP 与非 MTP 可共用同一份 YAML，不同量化类型（如 W8A8、W8A8C8）也可共用。
- [ ] YAML 中的 `model_config.model_path` 等路径使用占位符（如 `your_model_path`），不硬编码本地真实路径或敏感信息。

## 5. 量化控制

- [ ] 量化方式统一由模型权重 `config.json` 的 `quantization_config`（框架自动解析）决定，不通过 YAML 字段或额外开关控制。
- [ ] 支持的量化类型（如 W8A8、W4A16、W8A8C8）及对应的权重转换方式，需在 `README.md` 中说明并提供权重转换脚本。

## 6. 命名规范

- [ ] 模型名称统一使用下划线命名，与 `models/<model_name>/` 模型目录、`docs/models/<model_name>/` 文档目录及 `model_config.model_name` 保持一致（如 `gpt_oss`、`deepseek_r1`）。

## 7. 公共框架代码修改

- [ ] 非必要不要修改框架公共代码（即非 `models/` 目录代码），若涉及公共代码的修改，须确认能够兼容库上所有已接入模型，避免破坏公共链路。
- [ ] 若新增配置参数，需要分别在代码和[参数文档](./inference_config_guide.md)中补充注释，说明语义信息和默认值；若新增接口或修改公共流程时，需要在代码中补充注释说明。

## 8. 代码风格与提交

- [ ] 库代码统一使用 `logging.getLogger(__name__)`，不在库代码中调用 `logging.basicConfig`，避免覆盖根日志配置。

## 9. License 与外部代码引用

- [ ] 引用外部代码需正确标注 license，注明来源与许可证类型；新增模型的 LICENSE 建议使用 Apache 2.0 或 MIT，并按实际情况标注版权信息。
