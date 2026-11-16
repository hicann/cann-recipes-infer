---
name: model-infer-parallel-impl
description: 基于 PyTorch 框架的昇腾 NPU 模型推理并行切分实施技能。根据已确认的 parallel_config，实施模型代码的并行化改造，包括并行线性层替换、MoE 并行模式适配、通信组创建、Embedding/LMHead 并行、YAML 配置生成和权重转换。支持 infer 仓框架部署与独立部署两种模式。触发场景：model-infer-parallel-analysis 完成后需要实施改造、现有模型需要支持新的并行配置。
---

# 模型并行切分实施

按已确认的 parallel_config 实施模型并行化改造，覆盖并行层替换、通信组接入、YAML 配置生成、权重处理与验证。

本文及下属文档中的“框架部署”指接入 `cann-recipes-infer`（infer 仓库）的 `executor/core` 框架；相关框架接口与约束均以 infer 仓库为范围。

---

## 部署模式判定

**框架部署模式**：模型类构造接受 `infer_config + comm_manager`，在 `__init__` 调 `init_parallel_comm_group()` 按需 `comm_manager.register_group()` 注册通信组（框架自动判断 HCCL 物理组复用），启动通过 `executor/scripts/function.sh::launch`。

**独立部署模式**：Runner 自管推理循环（含进程组初始化、通信组建立、ForwardMetaData 构造、prefill / decode 调度），通过 `torchrun` 启动。

优先沿用任务中已明确的部署模式。未指定时，已注册到 `executor/core/support_models.py` 的模型走 infer 仓框架部署；自管 Runner 和推理循环且不接入 `executor/core/` 的模型走独立部署。新模型或仍无法判定时，默认优先 infer 仓框架部署。

判定后跳转：
- 框架部署模式 → 「## 框架部署模式」
- 独立部署模式 → 「## 独立部署模式」

> 两模式共通流程（配置参数速查 / YAML 配置生成 / 权重处理 / 验证）见后续公共章节。

---

## 实施输入与边界

从任务说明、已有配置或分析记录获取已确认的 parallel_config、部署模式，以及已有的模块拆解和适配前提。沿用适用结论，只补查缺失或与当前实现不一致的信息。

```yaml
parallel_config:
  world_size: {value}
  attn_tp_size: {value}       # attn_dp_size = world_size // attn_tp_size
  dense_tp_size: {value}
  moe_tp_size: {value}        # moe_ep_size = world_size // moe_tp_size
  embed_tp_size: {value}
  lmhead_tp_size: {value}
  o_proj_tp_size: {value}     # MLA 模型需要，非 MLA 可省略
```

按已确认方案落实各模块及非标准大参数模块的切分、通信边界、配置和权重处理。若需改变并行度、部署模式或性能目标，说明原因与建议，返回方案确认环节。

当前模板覆盖 TP/EP/DP，未覆盖 CP 接入步骤；方案含 CP 时，按已确认的适配前提单独完成代码接入与验证。

---

## 配置参数 → 实施步骤速查

根据 parallel_config 中各参数值，确定需要执行哪些步骤：

| 参数条件 | 需要的通信组 | 需要的代码改造 |
|---------|------------|--------------|
| `attn_tp_size > 1` | `attn_tp_group` | Attention QKV/O 替换为 ParallelLinear |
| `attn_tp_size = 1` 且 `world_size > 1` | 无 Attention TP 通信；按调用需要注册单 rank 逻辑组 | 保留并行层的权重加载与量化接口 |
| `dense_tp_size > 1` | `dense_tp_group` | Dense FFN Gate/Up/Down 替换 |
| `moe_tp_size > 1` 且 `moe_ep_size = 1` | `moe_tp_group` | MoE 专家 FFN 做 TP 切分 |
| `moe_tp_size > 1` 且 `moe_ep_size > 1` | `moe_tp_group` + `moe_ep_group`，按路由实现配置 | TP×EP 混合切分，需适配跨 EP 分发与聚合 |
| `moe_tp_size = 1` 且 `moe_ep_size > 1` | `moe_ep_group`（prefill double-routing AllToAll）+ `moe_ep_group_mc2`（decode MC2 dispatch/combine，须 `allow_physical_reuse=False` 占独立物理组） | MoE EP dispatch/combine 实现 |
| `o_proj_tp_size ≠ attn_tp_size` | `oproj_tp_group`（独立组） | O_proj 使用独立通信组 |
| `embed_tp_size > 1` | `embed_tp_group` | Embedding 替换为 VocabParallelEmbedding |
| `lmhead_tp_size > 1` | `lmhead_tp_group` | LMHead 替换为 ColumnParallelLinear |
| 相邻模块 TP 度不同 | — | 模块间插入 AllGather/ReduceScatter 重排 |

> `tp_size = 1` 不需要 TP 通信，但仍可复用并行层的权重加载与量化接口。若代码调用 `get_group/get_rank`，需注册对应逻辑组；普通单 rank 组由 CommManager 表示为 `None`、组内 rank 为 0。

---

## 框架部署模式

接入 `executor/core/` 框架，模型类构造接受 `comm_manager`，启动走 `executor/scripts/function.sh::launch`。

### 实施流程

```
第一步：核对适配点 + 选择参考实现
    ↓
第二步：通信组创建
    ↓
第三步：逐模块并行层替换
    ↓
第四步：Embedding / LMHead 并行
    ↓
后续 YAML / 权重 / 验证 走公共章节
```

---

### 第一步：核对适配点 + 选择参考实现

选择参考实现时，优先查询 cannbot-knowledge 中同推理部署框架的并行适配样例，关注通信组布局、模块边界重排和 MoE 路由。

读取所选参考实现的关键代码，核对模型结构、并行布局、硬件和精度格式；通信组接入见第二步。

---

### 第二步：通信组创建

模型初始化时，在子模块首次取用通信组或组内 rank 前完成注册。

#### 2.1 通信组创建与获取

CommManager 提供 `register_group()` API，模型在 `__init__` 调用 `init_parallel_comm_group()` **按需注册**自己用到的所有通信组，框架按 (`group_num`, `group_size`, `group_stride`) 形状自动判断 HCCL 物理通信组复用，需要独立物理组（MC2 fullmesh_v2、multi-stream 副本组）时显式 `allow_physical_reuse=False`。

```python
from executor.core.config import InferenceConfig, CommManager

class XxxForCausalLM(nn.Module):
    def __init__(self, config, infer_config: InferenceConfig,
                 comm_manager: CommManager = None, prefix: str = ""):
        super().__init__()
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.world_size = infer_config.parallel_config.world_size
        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.moe_ep_size = infer_config.parallel_config.moe_ep_size
        # ... 其他 *_size
        self.init_parallel_comm_group()  # 在子模块构造前完成注册
```

`init_parallel_comm_group` 内部调 `self.comm_manager.register_group(name, group_num, group_size, ...)` 注册三类组：

- **基础 TP 组**（按本模型启用的切分注册）：`attn_tp_group` / `embed_tp_group` / `lmhead_tp_group`
- **模块组**（按实际取用注册）：`moe_tp_group` / `moe_ep_group`（prefill `double-routing AllToAll` 用） / `dense_tp_group` / `oproj_tp_group`；其中 `moe_ep_group` 需传 `group_stride` 和 `return_name=True`
- **特殊独立物理组**（必须 `allow_physical_reuse=False`）：`moe_ep_group_mc2`（decode MC2 `dispatch_v2 / combine_v2` 用；参考实现按 `comm_alg="fullmesh_v2"` 注册时置 `allow_physical_reuse=False`）、`*_stream1` 副本组（multi-stream micro-batch）等；MC2 组的 `hccl_buffer_size` 通过 `calc_moe_hccl_buffer_size` 算出

模板代码、参数表与上游 `qwen3_moe` / `deepseek_r1` 参考实现见 `references/framework_moe_parallel.md` 的"通信组注意事项"段。

**取通信组**：模型内部用 `comm_manager.get_group(name)` 取 ProcessGroup，`comm_manager.get_group_name(name)` 取 HCCL name（NPU dispatch / combine 算子要求），`comm_manager.get_rank(name)` 取组内 rank。

#### 2.2 模块级差异化并行的通信组

当 attn_tp ≠ dense_tp ≠ moe_tp 时，各模块从 `comm_manager` 取不同的通信组（`attn_tp_group` / `dense_tp_group` / `moe_ep_group` 等）。模型在 `init_parallel_comm_group()` 中注册实际取用的逻辑组，infer 仓框架判断是否创建或复用物理组。

#### 2.3 DP 大小自动计算

```
attn_dp_size  = world_size // attn_tp_size
moe_ep_size   = world_size // moe_tp_size
embed_dp_size = world_size // embed_tp_size
```

---

### 第三步：逐模块并行层替换

> 并行层替换同时支持了权重在线切分：ParallelLinear 内置 `weight_loader()`，按 `tp_rank` 自动加载切片。权重处理详见公共章节「## 权重处理」。

**infer 仓接口约定**：上述线性层与 Embedding 使用对应并行包装，即使 `tp_size=1` 也保留统一 weight_loader 和量化接口；特殊投影按目标模块语义适配。

#### 3.1 Attention 层

标准 Q/K/V 投影可用 `QKVParallelLinear`，O 用 `RowParallelLinear`，使用 `attn_tp_group`。MLA 按实际低秩投影结构切分，O_proj 可能使用独立的 `oproj_tp_group`。

> **forward 接口**：模型顶层接收 `forward_metadata`，从中取 `slot_mapping` / `block_table` 下发到 attention 层，见 `executor/core/engine/execution_engine.py::_build_model_inputs`。

> **paged 模式联动**：改变 Attention TP 时，按实际 KV 切分或复制布局更新 `cache_entries.num_head`。`QKVParallelLinear` 的标准头切分/复制路径使用 `max(num_kv_heads // attn_tp_size, 1)`；MLA absorb 的 nope / rope 缓存各为一个 latent head。

> **TP+DP 下的 token 切分路径**：采用 layer 间 `[T/tp, H]` 布局时，attention 层用 `all_gather_into_tensor` + `reduce_scatter_tensor` 配对 `attn_tp_group`（不用 `all_reduce`）：layer 间 hidden 维持 `[T/tp, H]` 切分以省 tp_size 倍激活内存，attention 入口 AllGather 还原全 T，出口 ReduceScatter 切回。通信原语要求各 rank 等长，因此 model 顶层 forward 必须做：
>
> 1. packed T pad 到 `attn_tp_size` 倍数
> 2. 按 rank 切 input_ids `[rank * T/tp : (rank+1) * T/tp]`（建议 position_ids / cos/sin 不切，每 rank 保留 T_real，简化 attention 内部对齐）
> 3. attention 内部 QKV 前 AllGather，O 投影后 ReduceScatter
>
>    备注：framework metadata（`slot_mapping` / `actual_seq_lengths_kv`）按 T_real 构造，cache write 和 FA 必须 T_real。AllGather 出 `[T_padded, H]` 后用 `prompt_tokens` slice 回 T_real 再算，O 投影后 cat-zeros 补回 `[T_padded, H]` 给 ReduceScatter。
> 4. Prefill 末尾选末 token：AllGather hidden → `index_select(cu_seq_q - 1)` → lm_head（原 BSND 路径是本地取末位 → lm_head → gather logits，TND 顺序倒置）
>
> 参考 `models/qwen3_moe/models/modeling_qwen3_moe.py::Qwen3MoeModel.forward` 和 `Qwen3MoeAttention.forward`。

#### 3.2 Dense FFN 层

门控 FFN 的 Gate/Up → `MergedColumnParallelLinear`，Down → `RowParallelLinear`，均使用 `dense_tp_group`，保留原模型的激活语义。

#### 3.3 MoE 层（有 MoE 时必须处理）

- `moe_tp_size > 1` 且 `moe_ep_size = 1`：专家 FFN 做 TP 切分 + AllReduce
- `moe_tp_size > 1` 且 `moe_ep_size > 1`：按目标实现适配 TP×EP 的路由、通信和权重切分；纯 TP 模板不覆盖跨 EP 分发与聚合
- `moe_tp_size = 1` 且 `moe_ep_size > 1`：全 EP 主路径中 Prefill 用 re_routing + AllToAll，Decode 用 MC2 dispatch/combine；MC2 不适配时按参考文档选择保持语义的回退路径

> MoE 并行与融合算子紧密耦合，详见 `references/framework_moe_parallel.md`（含完整代码和算子说明）。

> MoE 模型必填属性：config 暴露 `num_experts_per_tok` / `n_routed_experts`（hccl_utils 计算 MC2 buffer 用），ForCausalLM 实例暴露 `self.num_experts` / `self.num_experts_per_tok`（EPLB hook 校验）。HF 上游若用其他命名（如 `moe_topk`）加 `@property` alias。

#### 3.4 模块间数据重排

按边界两侧的 token 分片与部分和布局选择 AllGather、AllReduce 或 ReduceScatter；相邻模块 TP 度不同时，结合通信组核对布局转换。

> 上述 1-4 项的代码示例（Attention / Dense FFN / Embed / LMHead / 数据重排）见 `references/framework_code_examples.md`。

---

### 第四步：Embedding / LMHead 并行

- Embedding → `VocabParallelEmbedding`，按词表维度切，`tp_rank=comm_manager.get_rank("embed_tp_group")`
- LMHead → `ColumnParallelLinear`，`tp_rank=comm_manager.get_rank("lmhead_tp_group")`

完整代码示例见 `references/framework_code_examples.md`「Embedding / LMHead」节。

---

## 独立部署模式

Runner 自管 `dist.init_process_group` + ParallelContext 构造，启动通过 `torchrun`。复用 infer 仓并行层，按独立 Runner 的上下文接口和数据布局适配 modeling。

完整 ParallelContext 骨架、Runner 改造、modeling 改造、启动模板见 `references/standalone_parallel.md`。

### 复用与自管边界

按功能边界划分：

- **自管**：进程组初始化、通信组配置数据结构（`ParallelContext`，对外接口对齐 CommManager）、通信组建立工厂、推理循环（`Runner.model_generate`，命名与签名与仓库 framework 模式 `ModelRunner.model_generate` 对齐，独立部署自行实现，不继承框架基类）、ForwardMetaData 构造、YAML 解析、启动脚本（torchrun）、入口（infer.py）。模板将自管逻辑放在 `runner_{model_name}.py`，已有项目可沿用其文件组织。
- **复用**：仓内通用算子层与工具层 —— `module/linear.py`（ParallelLinear）、`module/fuse_moe_gmm.py`（FusedMoEGMM）、`module/quantization/*`、`executor/utils/hccl_utils.py`（init_comm_group / get_group_name）、`executor/utils/forward_metadata.py`、`executor/model_loader/*`。这些与 `torch_npu` / `numpy` 同性质，import 不引入控制流约束。

modeling 通过 `parallel_ctx` 获取通信组和 rank，接口对齐 `get_group(name)` / `get_group_name(name)` / `get_rank(name)`。并行层与 MoE 算子链可复用，通信拓扑、metadata 与张量布局需与独立 Runner 对齐。运行时配置（如 `scheduler_config.batch_size`）通过 `parallel_ctx.runtime_cfg["..."]` 访问。

### 实施流程

```
第一步：Runner 改造（dist 初始化 + ParallelContext + 注入 modeling）
    ↓
第二步：modeling 代码改造（构造签名 + 并行层替换；并行层替换引用框架部署第三/四步）
    ↓
第三步：启动脚本（torchrun + infer.sh + infer.py）
    ↓
后续 YAML / 权重 / 验证 走公共章节
```

---

### 第一步：Runner 改造

按 `references/standalone_parallel.md` 接入 `ParallelContext` + `build_parallel_context`，`__init__` 内做 dist 初始化、构造 ParallelContext、把 `parallel_ctx` 注入 modeling。

```python
class {ModelName}Runner:
    def __init__(self, yaml_file_path):
        with open(yaml_file_path) as f:
            cfg = yaml.safe_load(f)

        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.npu.set_device(local_rank)
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)

        self.parallel_ctx = build_parallel_context(cfg)

        self.model = {ModelName}ForCausalLM(
            config=..., parallel_ctx=self.parallel_ctx, prefix="").to(f"npu:{local_rank}")
        self._load_weights(...)
        self._init_kvcache()
```

完整 `ParallelContext` dataclass 与 `build_parallel_context` 工厂见 `references/standalone_parallel.md` §1。

---

### 第二步：modeling 代码改造

构造签名从 `(config, prefix)` 改为 `(config, parallel_ctx, prefix)`，属性访问通过 `parallel_ctx.get_group(name)` / `get_group_name(name)` / `get_rank(name)`。

并行层替换、Embedding/LMHead 和数据重排参考 infer 仓框架部署「第三步」「第四步」与 `references/framework_code_examples.md`。MoE 链路参考 `references/framework_moe_parallel.md`；独立模板的 MC2 通信组接入限制见 `references/standalone_parallel.md`「MoE EP 接入」。

---

### 第三步：启动脚本

`infer.sh` 用 `torchrun` 拉起：单机以 `world_size` 设置 `--nproc_per_node`，多机按每节点进程数和节点数匹配全局 `world_size`；HCCL 环境沿用已有配置或由脚本设置。完整 `infer.sh` / `infer.py` 模板见 `references/standalone_parallel.md` §4。

---

## YAML 配置生成

YAML schema 完整字段、命名规范、粒度判定、保留策略见 model-infer-migrator skill `references/yaml_template.md`（两模式共用），本段只补多卡场景填值。

### 多卡填值

根据已确认方案填入 `parallel_config` 段。默认一套配置兼顾 Prefill/Decode，明确要求 PD 分离时分别落实两阶段配置：

- `world_size` 写在 `parallel_config` 下，与实际拉起卡数一致（启动脚本与业务代码都从此处读）
- `attn_tp_size` / `dense_tp_size` / `moe_tp_size` / `embed_tp_size` / `lmhead_tp_size` / `o_proj_tp_size`：按已确认值填
- `moe_ep_size` / `attn_dp_size`：框架自动推导（`world_size // *_tp_size`），不需手填
- `custom_params`：多卡特性开关（`enable_multi_streams` / `moe_chunk_max_len` 等）
- `scheduler_config.batch_size`：全局 batch，TP/DP 路径按 `attn_dp_size` 推导每 rank batch（infer 仓框架自动；独立部署 Runner 自管）；CP Prefill 使用 `cp_prefill_dp_size`，不可沿用 Decode 的分配方式
- 已有 Paged 缓存沿用其配置，并核对 TP 变化后的每卡 KV 头数；新增缓存路径由 model-infer-kvcache 处理

配置文件按 `yaml_template.md` 的命名与粒度规则组织，仅在结构性差异需要时拆分。

---

## 权重处理

并行层替换后，需要确保权重能正确加载到各卡。两模式都直接 import 仓内 `module/linear.py`、`module/fuse_moe_gmm.py` 等，weight_loader 行为完全一致。

需要权重适配参考时，查询 cannbot-knowledge 中同框架、目标切分与精度格式下的加载、转换及后处理样例。

### 在线权重切分（推荐）

框架/Runner 在运行时通过各模块的 `weight_loader()` 自动按 rank 加载对应切片（infer 仓框架默认行为，无需 YAML 开关）。所有 rank 读同一份完整 checkpoint，各自只保留本卡需要的部分。

**TP 权重加载**：`ColumnParallelLinear` / `RowParallelLinear` 的 `weight_loader` 按 `tp_rank` 取对应列/行切片。

**EP 权重加载**（MoE 模型）：`FusedMoEGMM` 的 `weight_loader` 按 `ep_rank` 过滤专家——只保留 `[ep_rank * experts_per_rank, (ep_rank+1) * experts_per_rank)` 范围内的专家权重，丢弃其他。`load_weights()` 中需通过 `make_expert_params_mapping(num_experts=...)` 生成全局专家映射，逐个传入 `weight_loader(expert_id=...)`。

适配要点：
- 模型类须实现 `load_weights()` 方法，遍历权重文件并匹配到各模块的 `weight_loader()`
- MoE 模型必须调用 `make_expert_params_mapping` 生成专家权重映射
- `MergedColumnParallelLinear`（如 gate+up 合并）需要特殊的 weight_loader 处理 slice 顺序
- 模型类须实现 `process_weights_after_loading()`，加载完成后遍历子模块触发权重后处理（量化 scale 处理、NZ 格式转换等）：

```python
def process_weights_after_loading(self):
    for _, module in self.named_modules():
        qm = getattr(module, "quant_method", None)
        if qm is not None and hasattr(qm, "process_weights_after_loading"):
            qm.process_weights_after_loading(module)
```

> 上述模板仅覆盖子模块的通用权重后处理；MLA 或量化模型还需核对目标投影结构、权重布局及量化格式，补充模型特有的后处理。已量化的 MoE 模型改变 EP 切分时，核对本 rank 加载的 smooth scale 是否覆盖消费算子所需的专家范围，见[量化结构卡](../model-infer-quantization/references/quantization-structure-cards.md)。

### 离线权重转换（备选）

若模型未实现 online split，或需要预切权重用于离线部署：

优先复用与目标权重布局、精度格式和 parallel_config 匹配的转换脚本；没有适用脚本时，参考 infer 仓库中适用的权重转换实现适配或新增。模型专用脚本可放在 `models/{model_name}/utils/` 下，执行前确认其参数和支持的权重格式。

> 并行配置变更后，在线切分从完整 checkpoint 重新加载，离线预切需使用匹配的产物或重新转换。若加载时报 shape 不匹配，在线路径先查本 rank 的加载切片，离线路径先查产物的切分配置。

---

## 验证

### 配置校验

infer 仓的 `ParallelConfig._validate()` 已校验各并行度能否整除 `world_size`；自管配置的 Runner 需自行校验。模型还需按实际切分方式检查头数和专家数：

```python
assert num_attention_heads % attn_tp_size == 0
# QKVParallelLinear supports KV-head sharding and replication.
if attn_tp_size >= num_key_value_heads:
    assert attn_tp_size % num_key_value_heads == 0
else:
    assert num_key_value_heads % attn_tp_size == 0
assert num_experts % ep_size == 0               # MoE
```

对要求 `embed_tp_size >= attn_tp_size` 且整除的既有重排路径保留相应校验；这是该实现的数据布局约束，不作为所有模型的通用限制。

### 功能验证

1. 确认推理实际加载的是修改后的代码（检查模型注册表、import 路径、日志中的模块路径等，确保运行时代码路径与修改路径一致）
2. 确认配置指向目标 YAML：框架部署由拉起命令的 `--yaml` 指定，独立部署改模型目录 `infer.sh` 里指向 yaml 的变量（skill 模板为 `YAML`，仓内 legacy 模型多为 `YAML_FILE_NAME`）
3. 按部署模式拉起推理入口（框架部署 `bash executor/scripts/infer.sh --model {model_name} --yaml {yaml_name}.yaml`，独立部署 `bash infer.sh`）
4. 检查 Prefill + Decode 推理成功（无 crash）
5. 按 TP/DP/EP 布局核对各 rank 的分片形状、collective 配对及聚合后的输出
6. 加载真实权重后与基线对比输出；精度异常交由 model-infer-precision-debug 定位

按交付配置汇总所用权重、Prefill/Decode 验证及基线对比结果，并说明未完成的适配或验证项。需要性能校准时，通过 model-infer-profiling 采集数据供 parallel-analysis 使用。

### 权重加载验证

加载权重验证时，在 YAML 中设置 `with_ckpt: True` + `model_path`，确认各 rank 加载的权重与当前切分匹配、后处理已执行；使用离线转换产物时核对其目录结构和加载路径。

---

## infer 仓库框架源码入口

- `module/linear.py`：并行 Linear / Embedding 及 TP 权重加载。
- `module/fuse_moe_gmm.py`：MoE 专家计算、权重映射与 TP/EP 切分。
- `executor/core/config/comm_manager.py`：`register_group`、通信组复用及 group / rank 获取。
