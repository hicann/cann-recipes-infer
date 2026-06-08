# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Step-3.7-Flash text backbone configuration (model_type=step3p5)."""

from typing import Any, Optional, Sequence

from transformers.configuration_utils import PretrainedConfig


def _normalize_per_layer_values(
    values: Optional[Sequence[Any]],
    num_hidden_layers: int,
) -> Optional[list]:
    """Pad/trim per-layer config lists to exactly ``num_hidden_layers`` entries.

    Step checkpoints keep MTP/spec layer entries after the decoder layers; this
    backbone only builds ``num_hidden_layers`` decoder layers, so per-layer
    fields must be trimmed to that count.
    """
    if values is None:
        return None
    normalized = list(values)
    if not normalized:
        return normalized
    if len(normalized) < num_hidden_layers:
        normalized.extend([normalized[-1]] *
                          (num_hidden_layers - len(normalized)))
    return normalized[:num_hidden_layers]


class Step3p7TextConfig(PretrainedConfig):
    """Configuration for the Step-3.7-Flash text MoE decoder.

    This class is registered as the unified-flow config class. The on-disk
    ``config.json`` is the *top-level* multimodal config (``model_type=step3p7``)
    that nests the text fields under ``text_config``. To stay robust whether we
    are handed the nested text dict or the full top-level config, ``__init__``
    detects a ``text_config`` kwarg and merges those fields up before parsing.
    """

    model_type = "step3p5"
    architectures = ["Step3p5ForCausalLM"]
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 11264,
        num_attention_heads: int = 64,
        num_attention_groups: int = 8,
        num_hidden_layers: int = 45,
        max_seq_len: int = 262144,
        vocab_size: int = 128896,
        rms_norm_eps: float = 1e-5,
        moe_intermediate_size: int = 1280,
        moe_num_experts: int = 288,
        moe_top_k: int = 8,
        rope_theta: Any = 10000,
        rope_scaling: Optional[dict] = None,
        max_position_embeddings: int = 262144,
        share_expert_dim: int = 1280,
        head_dim: int = 128,
        norm_expert_weight: bool = True,
        layer_types: Optional[list] = None,
        sliding_window: Optional[int] = None,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: Any = None,
        attention_dropout: float = 0.0,
        use_qk_norm: bool = False,
        use_head_wise_attn_gate: bool = False,
        use_moe_router_bias: bool = False,
        moe_router_activation: str = "softmax",
        moe_router_scaling_factor: float = 1.0,
        need_fp32_gate: bool = False,
        attention_other_setting: Optional[dict] = None,
        swiglu_limits: Optional[list] = None,
        swiglu_limits_shared: Optional[list] = None,
        use_rope_layers: Optional[list] = None,
        yarn_only_types: Optional[list] = None,
        num_nextn_predict_layers: int = 0,
        moe_layers_enum: Any = None,
        moe_every_n_layer: int = 1,
        moe_layer_offset: int = 0,
        **kwargs,
    ) -> None:
        # When fed the top-level step3p7 config.json, the text fields arrive as
        # a nested `text_config` dict. Merge them up so a single flat config
        # exposes everything the modeling code expects.
        text_config = kwargs.pop("text_config", None)
        if isinstance(text_config, dict):
            # Promote shared rope_scaling from the top level if the nested dict
            # omits it (mirrors the original multimodal loader behaviour).
            shared_rope_scaling = kwargs.get("rope_scaling")
            merged = dict(text_config)
            if shared_rope_scaling is not None and "rope_scaling" not in merged:
                merged["rope_scaling"] = shared_rope_scaling
            # Re-dispatch through __init__ with the flattened fields. Drop the
            # vision-only / top-level keys that this text config does not model.
            for drop_key in ("vision_config", "understand_projector_stride",
                             "projector_bias", "image_token_id",
                             "architectures", "auto_map", "model_type"):
                kwargs.pop(drop_key, None)
            self.__init__(**{**merged, **kwargs})
            return

        torch_dtype = kwargs.get("torch_dtype")
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        # Alias commonly used by framework helpers (GQA KV-head count).
        self.num_key_value_heads = num_attention_groups
        self.num_hidden_layers = num_hidden_layers
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        # Framework-standard aliases consumed by executor helpers
        # (calc_moe_hccl_buffer_size / get_global_routed_expert_num, etc.).
        self.num_experts = moe_num_experts
        self.num_experts_per_tok = moe_top_k
        self.rope_theta = rope_theta
        self.rope_scaling = dict(rope_scaling) if isinstance(rope_scaling, dict) else rope_scaling
        self.max_position_embeddings = max_position_embeddings
        self.share_expert_dim = share_expert_dim
        self.head_dim = head_dim
        self.norm_expert_weight = norm_expert_weight
        self.layer_types = _normalize_per_layer_values(layer_types, num_hidden_layers)
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout
        self.use_qk_norm = use_qk_norm
        self.use_head_wise_attn_gate = use_head_wise_attn_gate
        self.use_moe_router_bias = use_moe_router_bias
        self.moe_router_activation = moe_router_activation
        self.moe_router_scaling_factor = moe_router_scaling_factor
        self.need_fp32_gate = need_fp32_gate
        self.attention_other_setting = attention_other_setting
        self.swiglu_limits = _normalize_per_layer_values(swiglu_limits, num_hidden_layers)
        self.swiglu_limits_shared = _normalize_per_layer_values(swiglu_limits_shared, num_hidden_layers)
        self.use_rope_layers = use_rope_layers
        self.yarn_only_types = yarn_only_types
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.moe_every_n_layer = moe_every_n_layer
        self.moe_layer_offset = moe_layer_offset

        # Normalize moe_layers_enum to a set of integer layer indices.
        self.moe_layers_enum = moe_layers_enum
        self.moe_layers_idx = self._parse_moe_layers(moe_layers_enum, num_hidden_layers)

        # Per-layer rope_theta / partial_rotary_factor lists (Step3p7 specific).
        # rope_theta may be a scalar or a per-layer list in config.json.
        self.partial_rotary_factors = _normalize_per_layer_values(
            kwargs.pop("partial_rotary_factors", None), num_hidden_layers)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        if torch_dtype is not None:
            self.torch_dtype = torch_dtype

    @staticmethod
    def _parse_moe_layers(moe_layers_enum, num_hidden_layers):
        if moe_layers_enum is None:
            return list(range(1, num_hidden_layers))
        if isinstance(moe_layers_enum, str):
            return [int(i) for i in moe_layers_enum.split(",") if i.strip()]
        return [int(i) for i in moe_layers_enum]


__all__ = ["Step3p7TextConfig"]
