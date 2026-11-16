# 框架部署模型模板

本文件提供框架部署模式的 ForCausalLM 与 Attention 骨架及必要接口约束。

YAML 模板与命名规范见共用文件 `yaml_template.md`；README 模板见 `readme_template.md`。

接口契约见 [docs/design/executor_design.md §6](../../../../docs/design/executor_design.md)，ForwardMetaData 字段定义见 `executor/utils/forward_metadata.py`。框架接口语义与启动机制可查询 cannbot-knowledge 中的 recipes 框架说明。

替换 `{model_name}`（小写下划线）和 `{ModelName}`（驼峰）后使用。

---

## 1. 模型类契约

> **入参布局**：input_ids 由框架按 packed 布局拼装为一维 `[TotalTokens]`，prefill 各 prompt 串接，decode 每请求 1 token。返回 logits `[batch_size, 1, vocab_size]`：prefill 由模型内按 `actual_seq_lengths_cu_q - 1` 取每请求末位后再过 lm_head。

### 1.1 ForCausalLM 类骨架

```python
import torch
import torch.nn as nn
from typing import Iterable, Optional, Tuple

from executor.core.config import InferenceConfig, CommManager
from executor.utils.forward_metadata import ForwardMetaData

from .configuration_{model_name} import {ModelName}Config


class {ModelName}ForCausalLM(nn.Module):
    def __init__(
        self,
        config: {ModelName}Config,
        infer_config: InferenceConfig,
        comm_manager: CommManager = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager

        # 读取并行配置
        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.lmhead_tp_size = infer_config.parallel_config.lmhead_tp_size
        # ... MoE 模型还有 moe_tp_size / moe_ep_size

        # 构造子模块
        self.model = {ModelName}Model(config, infer_config, comm_manager, prefix="model")
        self.lm_head = ...  # ColumnParallelLinear

    def forward(
        self,
        input_ids: torch.LongTensor,        # packed 一维 [TotalTokens]
        position_ids: Optional[torch.LongTensor] = None,
        forward_metadata: ForwardMetaData = None,
        **kwargs,                           # 框架只透传 input_ids / position_ids / forward_metadata / batch
    ) -> torch.Tensor:
        # Prefill：每请求取末位 token（cu_seq_q - 1 索引）→ lm_head → [batch_size, 1, vocab_size]
        # Decode：input_ids 本就是每请求 1 token → lm_head → [batch_size, 1, vocab_size]
        ...
        return logits

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """从 HF checkpoint stream 加载权重，返回已加载参数名集合。"""
        params_dict = dict(self.named_parameters())
        loaded = set()
        for name, tensor in weights:
            # 按名称匹配 param，调用 weight_loader 或 default_weight_loader
            ...
        return loaded

    def process_weights_after_loading(self):
        """加载权重后触发量化/MoE 后处理（NZ 转换、scale 处理等）。框架自动调用。"""
        for _, module in self.named_modules():
            qm = getattr(module, "quant_method", None)
            if qm is not None and hasattr(qm, "process_weights_after_loading"):
                qm.process_weights_after_loading(module)

    def check_model_settings(self):
        """模型配置一致性校验（如 attn_tp_size 整除约束、num_heads 切分可行性等）。
        框架在加载完模型后自动调用；校验失败应 raise RuntimeError。可选实现，缺方法时框架跳过。"""
        pass
```

> Paged 模式接入（`get_cache_info()` 实现 + `cache_entries` 声明）由 model-infer-kvcache skill 改造时添加，migrator 阶段不写。

### 1.2 Attention 层（KV 接入）

按 SKILL.md 步骤 4 选定路径，二选一。

#### 路径 A — Legacy 模式（默认推荐，单卡 + 等长 batch baseline）

```python
class {ModelName}Attention(nn.Module):
    def __init__(self, config, infer_config, comm_manager, layer_idx, prefix=""):
        super().__init__()
        # ... QKV / O 投影、RoPE 等 ...

        # Legacy 模式：声明 cache_unit，框架按 (batch, seq, *cache_unit) 自动预分配
        self.k_cache = torch.Tensor([])
        self.v_cache = torch.Tensor([])
        self.cache_unit = (self.num_kv_heads_per_rank * self.head_dim,)

    def forward(self, hidden_states, position_embeddings, forward_metadata: ForwardMetaData, **kwargs):
        is_prefill = forward_metadata.is_prefill
        kv_len = forward_metadata.kv_len  # Prefill cache offsets; decode positions within each request.
        # decode 分支必须把 atten_mask 设为 None
        atten_mask = forward_metadata.attention_mask if is_prefill else None

        # 计算 Q / K / V，写 KV 到 self.k_cache / self.v_cache
        # 用 torch_npu.scatter_update_(self.k_cache, kv_len, key_states, -2) 写入
        # 然后 SDPA / 标准 PyTorch attention 计算
        ...
```

> 顶层 forward 需对 packed 1D `input_ids` 做 BSND reshape 回 `[B, S]`（仅等长 batch 适用）。`ForCausalLM` 不实现 `get_cache_info()`。

#### 路径 B — 最简 Paged 模式（多卡 / 变长 batch / 用户指定 Paged 起步）

```python
class {ModelName}Attention(nn.Module):
    def __init__(self, config, infer_config, comm_manager, layer_idx, prefix=""):
        super().__init__()
        # ... QKV / O 投影、RoPE 等 ...

        self.attn_type = "FullAttention"
        self.block_size = infer_config.scheduler_config.block_size
        self.k_cache = torch.Tensor([])   # 占位，框架通过 tensor_setter 注入
        self.v_cache = torch.Tensor([])
        cache_dtype = config.torch_dtype or torch.bfloat16
        self.cache_entries = [
            CacheEntry(
                cache_name="k_cache", attn_type=self.attn_type,
                dim=self.head_dim, num_head=self.num_kv_heads_per_rank,
                dtype=cache_dtype, needs_block=True,
                block_size=self.block_size,
                tensor_setter=lambda t, layer=self: setattr(layer, "k_cache", t),
            ),
            CacheEntry(
                cache_name="v_cache", attn_type=self.attn_type,
                dim=self.head_dim, num_head=self.num_kv_heads_per_rank,
                dtype=cache_dtype, needs_block=True,
                block_size=self.block_size,
                tensor_setter=lambda t, layer=self: setattr(layer, "v_cache", t),
            ),
        ]

    def forward(self, hidden_states, position_embeddings, forward_metadata: ForwardMetaData, **kwargs):
        slot_mapping = forward_metadata.slot_mapping[self.attn_type]  # Cache slot for each token.
        block_table = forward_metadata.block_table[self.attn_type]  # Cache block indices for each request.
        # ... QKV 投影 / RoPE → key_states / value_states (packed [TotalTokens, ...]) ...

        # 写 cache：按 slot_mapping 写入
        torch_npu.npu_scatter_nd_update_(
            self.k_cache.view(-1, self.num_kv_heads_per_rank, self.head_dim),
            slot_mapping.view(-1, 1),
            key_states.view(-1, self.num_kv_heads_per_rank, self.head_dim),
        )
        torch_npu.npu_scatter_nd_update_(
            self.v_cache.view(-1, self.num_kv_heads_per_rank, self.head_dim),
            slot_mapping.view(-1, 1),
            value_states.view(-1, self.num_kv_heads_per_rank, self.head_dim),
        )

        # Attention：HF 原版 SDPA / 标准 PyTorch，不上 FA
        # - Prefill：直接用刚算的 key_states / value_states + causal mask
        # - Decode：用 block_table 从 cache gather 累计 KV，再走 SDPA
        # Split variable-length requests by cumulative query lengths in forward_metadata.actual_seq_lengths_cu_q.
        ...
```

> `ForCausalLM` 必须实现 `get_cache_info()`（5 行模板，遍历 `self.model.layers` 收集 `cache_entries`）；`__init__` 保存 `self.block_size = infer_config.scheduler_config.block_size`。完整字段表与 `get_cache_info()` 实现见 model-infer-kvcache skill 的 framework_kv_reference 1.1 节与 3.1 节。FA 算子接入 / MLA absorb / 复杂 attn_type 由 model-infer-kvcache skill 接手。
