# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Copyright 2026 The Moonshot AI Team. All rights reserved.
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
"""Kimi K3 configuration adapters for the text-only inference path."""

from __future__ import annotations

from copy import deepcopy
from typing import Optional

from transformers.configuration_utils import PretrainedConfig


def _get_custom_params(runner_settings) -> dict:
    if runner_settings is None:
        return {}
    model_config = getattr(runner_settings, "model_config", None)
    if model_config is None and isinstance(runner_settings, dict):
        model_config = runner_settings.get("model_config", {})
    custom_params = getattr(model_config, "custom_params", None)
    if custom_params is None and isinstance(model_config, dict):
        custom_params = model_config.get("custom_params", {})
    return custom_params or {}


# The Kimi K3 release spells its quantization contract as
# ``format: "mxfp4-pack-quantized"`` with ``input_activations: null``. That
# names exactly the scheme the framework already knows as ``float-quantized``
# with an explicit dynamic MXFP8 activation block -- compare the LongCat and
# Qwen3 MXFP4 releases, whose MoEGMM group is 4-bit float group-32 weights
# against 8-bit float group-32 dynamic activations, which is what this model
# computes. The shared compressed-tensors parser only accepts CompressionFormat
# values and asserts that a missing input_activations implies integer weights,
# so translate the vendor spelling here instead of teaching that parser a
# vendor alias.
_MX_PACK_QUANTIZATION_FORMATS = ("mxfp4-pack-quantized",)
_CANONICAL_MX_FORMAT = "float-quantized"
_MXFP8_DYNAMIC_ACTIVATIONS = {
    "actorder": None,
    "block_structure": None,
    "dynamic": True,
    "group_size": 32,
    "num_bits": 8,
    "observer": "minmax",
    "observer_kwargs": {},
    "strategy": "group",
    "symmetric": True,
    "type": "float",
}


def normalize_mx_pack_quantization(quant_config: Optional[dict]) -> Optional[dict]:
    """Restate an MX pack quantization config in the framework's own terms."""
    if not isinstance(quant_config, dict):
        return quant_config
    if quant_config.get("format") not in _MX_PACK_QUANTIZATION_FORMATS:
        return quant_config
    normalized = deepcopy(quant_config)
    normalized["format"] = _CANONICAL_MX_FORMAT
    for group in normalized.get("config_groups", {}).values():
        if group.get("input_activations") is None:
            group["input_activations"] = deepcopy(_MXFP8_DYNAMIC_ACTIVATIONS)
    return normalized


class KimiLinearConfig(PretrainedConfig):
    model_type = "kimi_linear"
    keys_to_ignore_at_inference = ["past_key_values"]

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        # Transformers keeps unknown ``from_pretrained`` kwargs outside the
        # constructor and applies them only after ``cls(**config_dict)``. Model
        # custom parameters must take effect before validation, so inject
        # runner_settings into the constructor dictionary explicitly.
        runner_settings = kwargs.pop("runner_settings", None)
        config_dict = deepcopy(config_dict)
        if runner_settings is not None:
            config_dict["runner_settings"] = runner_settings
        return super().from_dict(config_dict, **kwargs)

    def __init__(
        self,
        model_type: str = "kimi_linear",
        vocab_size: int = 163840,
        hidden_size: int = 4096,
        head_dim: Optional[int] = None,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = None,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int | list[int] = 2,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,
        tie_word_embeddings: bool = False,
        moe_intermediate_size: Optional[int] = None,
        moe_renormalize: bool = True,
        moe_router_activation_func: str = "sigmoid",
        num_experts: Optional[int] = None,
        num_experts_per_token: Optional[int] = None,
        num_shared_experts: int = 0,
        routed_scaling_factor: float = 1.0,
        first_k_dense_replace: int = 0,
        moe_layer_freq: int = 1,
        use_grouped_topk: bool = True,
        num_expert_group: int = 1,
        topk_group: int = 1,
        q_lora_rank: Optional[int] = None,
        kv_lora_rank: Optional[int] = None,
        qk_nope_head_dim: Optional[int] = None,
        qk_rope_head_dim: Optional[int] = None,
        v_head_dim: Optional[int] = None,
        mla_use_nope: bool = False,
        mla_use_output_gate: bool = False,
        num_nextn_predict_layers: int = 0,
        linear_attn_config: Optional[dict] = None,
        attn_res_block_size: Optional[int] = None,
        latent_moe_use_norm: bool = False,
        activation_situ_beta: Optional[float] = None,
        activation_situ_linear_beta: Optional[float] = None,
        max_position_embeddings: int = 4096,
        routed_expert_hidden_size: Optional[int] = None,
        topk_method: str = "noaux_tc",
        enable_attn_res_two_phase: bool = False,
        **kwargs,
    ) -> None:
        runner_settings = kwargs.pop("runner_settings", None)
        nested_text_config = kwargs.pop("text_config", None)
        if isinstance(nested_text_config, dict):
            # ``from_pretrained`` received the multimodal root config.  The
            # nested text section is the architecture truth.
            self.__init__(**{**nested_text_config, "runner_settings": runner_settings})
            return

        values = dict(
            model_type=model_type,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            head_dim=head_dim if head_dim is not None else hidden_size // num_attention_heads,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads or num_attention_heads,
            hidden_act=hidden_act,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            mla_use_nope=mla_use_nope,
            mla_use_output_gate=mla_use_output_gate,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            moe_renormalize=moe_renormalize,
            num_shared_experts=num_shared_experts,
            routed_scaling_factor=routed_scaling_factor,
            moe_router_activation_func=moe_router_activation_func,
            moe_intermediate_size=moe_intermediate_size,
            first_k_dense_replace=first_k_dense_replace,
            moe_layer_freq=moe_layer_freq,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            num_nextn_predict_layers=num_nextn_predict_layers,
            linear_attn_config=deepcopy(linear_attn_config),
            attn_res_block_size=attn_res_block_size,
            latent_moe_use_norm=latent_moe_use_norm,
            activation_situ_beta=activation_situ_beta,
            activation_situ_linear_beta=activation_situ_linear_beta,
            max_position_embeddings=max_position_embeddings,
            routed_expert_hidden_size=routed_expert_hidden_size,
            topk_method=topk_method,
            enable_attn_res_two_phase=enable_attn_res_two_phase,
        )

        custom_params = _get_custom_params(runner_settings)
        if "enable_attn_res_two_phase" in custom_params:
            values["enable_attn_res_two_phase"] = bool(
                custom_params["enable_attn_res_two_phase"]
            )

        for key in ("quantization_config", "compression_config"):
            if key in kwargs:
                kwargs[key] = normalize_mx_pack_quantization(kwargs[key])

        for name, value in values.items():
            setattr(self, name, value)
        self.num_experts_per_tok = self.num_experts_per_token

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self._validate_architecture()

    @property
    def is_mla(self) -> bool:
        return any((
            self.q_lora_rank is not None,
            self.kv_lora_rank is not None,
            self.qk_nope_head_dim is not None,
            self.qk_rope_head_dim is not None,
            self.v_head_dim is not None,
            self.mla_use_nope,
        ))

    @property
    def is_moe(self) -> bool:
        return self.num_experts is not None

    @property
    def is_linear_attn(self) -> bool:
        return bool(self.linear_attn_config and self.linear_attn_config.get("kda_layers"))

    def is_kda_layer(self, layer_idx: int) -> bool:
        return bool(
            self.linear_attn_config
            and layer_idx + 1 in self.linear_attn_config["kda_layers"]
        )

    def _validate_architecture(self) -> None:
        if self.moe_router_activation_func not in ("softmax", "sigmoid"):
            raise ValueError("moe_router_activation_func must be softmax or sigmoid")
        # ``PretrainedConfig.to_diff_dict`` constructs a default instance for
        # serialization.  The generic KimiLinear defaults legitimately have
        # no hybrid layer table, so only validate it when supplied.
        if not self.linear_attn_config:
            return
        kda = {int(i) for i in self.linear_attn_config.get("kda_layers", [])}
        full = {int(i) for i in self.linear_attn_config.get("full_attn_layers", [])}
        expected = set(range(1, self.num_hidden_layers + 1))
        if kda & full or kda | full != expected:
            raise ValueError(
                "one-based KDA/full-attention lists must be disjoint and cover 1..num_hidden_layers"
            )
        if self.num_experts is not None:
            if not self.num_experts_per_token or self.num_experts_per_token > self.num_experts:
                raise ValueError("invalid num_experts_per_token")
            if self.num_experts % self.num_expert_group:
                raise ValueError("num_experts must be divisible by num_expert_group")


class KimiK3VisionConfig(PretrainedConfig):
    model_type = "kimi_k3_vision"

    def __init__(self, runner_settings=None, **kwargs) -> None:
        for name, value in kwargs.items():
            setattr(self, name, value)
        super().__init__()


class KimiK3Config(PretrainedConfig):
    model_type = "kimi_k3"
    sub_configs = {"text_config": KimiLinearConfig, "vision_config": KimiK3VisionConfig}

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        runner_settings = kwargs.pop("runner_settings", None)
        config_dict = deepcopy(config_dict)
        if runner_settings is not None:
            config_dict["runner_settings"] = runner_settings
        return super().from_dict(config_dict, **kwargs)

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        runner_settings=None,
        **kwargs,
    ) -> None:
        self.text_config = (
            KimiLinearConfig(**text_config, runner_settings=runner_settings)
            if isinstance(text_config, dict)
            else text_config
        )
        self.vision_config = (
            KimiK3VisionConfig(**vision_config, runner_settings=runner_settings)
            if isinstance(vision_config, dict)
            else vision_config
        )
        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id
        if self.text_config is not None:
            # Generic executor code expects these language attributes at the
            # root even though the checkpoint stores them under text_config.
            self.vocab_size = self.text_config.vocab_size
            self.num_hidden_layers = self.text_config.num_hidden_layers
        if self.text_config is not None and getattr(self.text_config, "quantization_config", None):
            self.quantization_config = self.text_config.quantization_config
        super().__init__(**kwargs)


# Convenience alias used by generic tooling; the checkpoint's canonical name
# remains ``KimiLinearConfig``.
KimiK3TextConfig = KimiLinearConfig

__all__ = ["KimiLinearConfig", "KimiK3TextConfig", "KimiK3VisionConfig", "KimiK3Config"]
