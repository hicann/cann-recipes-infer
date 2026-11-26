# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Adapted from HuggingFace Transformers v5.5.0 Gemma4 (Apache-2.0).
# Copyright 2026 the HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration for the single-device, text-only Gemma-4-E2B adapter."""
import math
from transformers.configuration_utils import PretrainedConfig


class Gemma4E2BConfig(PretrainedConfig):
    model_type = "gemma4_text"
    base_config_key = "text_config"

    def __init__(
        self,
        vocab_size=262144,
        hidden_size=1536,
        intermediate_size=6144,
        num_hidden_layers=35,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=256,
        global_head_dim=512,
        sliding_window=512,
        hidden_size_per_layer_input=256,
        vocab_size_per_layer_input=262144,
        num_kv_shared_layers=20,
        use_double_wide_mlp=True,
        hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
        layer_types=None,
        rope_parameters=None,
        final_logit_softcapping=30.0,
        max_position_embeddings=131072,
        tie_word_embeddings=True,
        attention_bias=False,
        enable_moe_block=False,
        attention_k_eq_v=False,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        **kwargs,
    ):
        values = locals().copy()
        for name in ("self", "kwargs", "values", "__class__"):
            values.pop(name, None)
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        for name, value in values.items():
            setattr(self, name, value)
        if type(num_hidden_layers) is not int or num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be a positive integer")
        self.layer_types = (
            list(layer_types)
            if layer_types is not None
            else [
                "full_attention" if (i + 1) % 5 == 0 else "sliding_attention"
                for i in range(num_hidden_layers)
            ]
        )
        self.rope_parameters = (
            rope_parameters
            if rope_parameters is not None
            else {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {
                    "rope_type": "proportional",
                    "rope_theta": 1000000.0,
                    "partial_rotary_factor": 0.25,
                },
            }
        )
        self.validate_e2b()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        return cls.from_dict(config.get("text_config", config), **kwargs)

    def validate_e2b(self):
        sizes = (
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.global_head_dim,
            self.sliding_window,
            self.hidden_size_per_layer_input,
            self.vocab_size_per_layer_input,
        )
        if any(type(n) is not int or n <= 0 for n in sizes):
            raise ValueError("E2B dimensions must be positive integers")
        if self.enable_moe_block or self.attention_k_eq_v:
            raise ValueError("E2B adapter supports dense MLP and separate K/V only")
        if self.hidden_activation != "gelu_pytorch_tanh":
            raise ValueError("E2B adapter requires gelu_pytorch_tanh")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Q head count must be divisible by KV head count")
        if len(self.layer_types) != self.num_hidden_layers or any(
            t not in {"sliding_attention", "full_attention"} for t in self.layer_types
        ):
            raise ValueError("layer_types must describe every logical layer")
        if (
            type(self.num_kv_shared_layers) is not int
            or not 0 <= self.num_kv_shared_layers < self.num_hidden_layers
        ):
            raise ValueError(
                "num_kv_shared_layers must leave at least one source layer"
            )
        if (
            isinstance(self.rms_norm_eps, bool)
            or not isinstance(self.rms_norm_eps, (int, float))
            or not math.isfinite(self.rms_norm_eps)
            or self.rms_norm_eps <= 0
        ):
            raise ValueError("rms_norm_eps must be finite and positive")
        if self.final_logit_softcapping is not None and (
            isinstance(self.final_logit_softcapping, bool)
            or not isinstance(self.final_logit_softcapping, (int, float))
            or not math.isfinite(self.final_logit_softcapping)
            or self.final_logit_softcapping <= 0
        ):
            raise ValueError("final_logit_softcapping must be finite and positive")
        if getattr(self, "use_bidirectional_attention", False):
            raise ValueError("E2B offline adapter supports causal text attention only")
        self.kv_source_indices = []
        start = self.num_hidden_layers - self.num_kv_shared_layers
        for i, kind in enumerate(self.layer_types):
            candidates = [j for j in range(start) if self.layer_types[j] == kind]
            if i >= start and not candidates:
                raise ValueError(f"no non-shared KV source for {kind}")
            self.kv_source_indices.append(i if i < start else candidates[-1])
        if not isinstance(self.rope_parameters, dict):
            raise ValueError(
                "rope_parameters must map both attention types to settings"
            )
        head_dims = (
            ("sliding_attention", self.head_dim),
            ("full_attention", self.global_head_dim),
        )
        for kind, dim in head_dims:
            params = self.rope_parameters.get(kind)
            if not isinstance(params, dict):
                raise ValueError(f"rope_parameters must contain settings for {kind}")
            expected = "default" if kind == "sliding_attention" else "proportional"
            if params.get("rope_type", expected) != expected:
                raise ValueError(f"unsupported {kind} RoPE type")
            theta = params.get("rope_theta")
            if (
                isinstance(theta, bool)
                or not isinstance(theta, (int, float))
                or not math.isfinite(theta)
                or theta <= 0
            ):
                raise ValueError(f"{kind} rope_theta must be finite and positive")
            factor = params.get("partial_rotary_factor", 0.25)
            if kind == "full_attention" and (
                isinstance(factor, bool)
                or not isinstance(factor, (int, float))
                or not math.isfinite(factor)
                or factor <= 0
            ):
                raise ValueError(
                    "full_attention partial_rotary_factor must be finite and positive"
                )
            rotary = dim if kind == "sliding_attention" else int(dim * factor)
            if dim % 2 or rotary <= 0 or rotary > dim or rotary % 2:
                raise ValueError(
                    "RoPE requires even, positive rotary dimensions within the head"
                )
