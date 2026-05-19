# 新模型合入 Checklist

本文提供一份新 LLM 模型合入本仓库的 Checklist。新模型基于框架开发后，在进行合入前需要按清单完成并验证若干关键事项，例如 Packed Sequence、KV Cache 管理、npugraph_ex 图模式优化等，最终让模型顺利跑通离线推理、在线推理，并通过数据集精度验证。

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

- [ ] 跑数据集精度测评，例如 GMS8K 数据集，得到正常的评分。
