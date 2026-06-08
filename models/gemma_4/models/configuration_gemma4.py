# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/models/gemma4/configuration_gemma4.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Copyright 2024-2026 LatenceAI. All rights reserved.
# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""Gemma4 text model configuration (text-only, no vision/audio)."""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Gemma4TextConfig(PretrainedConfig):
    """
    Configuration for the Gemma4 text decoder (language model only).
    Adapted from HuggingFace Gemma4TextConfig, simplified for this repository's framework.
    """

    model_type = "gemma4_text"

    def __init__(
        self,
        vocab_size=262144,
        hidden_size=2816,
        intermediate_size=2112,
        num_hidden_layers=30,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=256,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        sliding_window=1024,
        layer_types=None,
        final_logit_softcapping=None,
        num_global_key_value_heads=None,
        global_head_dim=512,
        attention_k_eq_v=False,
        enable_moe_block=False,
        num_experts=None,
        top_k_experts=None,
        moe_intermediate_size=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_activation = hidden_activation
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.final_logit_softcapping = final_logit_softcapping
        self.num_global_key_value_heads = num_global_key_value_heads
        self.global_head_dim = global_head_dim
        self.attention_k_eq_v = attention_k_eq_v
        self.enable_moe_block = enable_moe_block
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.moe_intermediate_size = moe_intermediate_size

        # Build rope_parameters with defaults if not provided
        if rope_parameters is None:
            self.rope_parameters = {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {
                    "rope_type": "proportional",
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                },
            }
        else:
            self.rope_parameters = rope_parameters

        # Build layer_types with default 5:1 pattern if not provided
        if layer_types is None:
            sliding_window_pattern = 6
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = layer_types

        # Ensure last layer is full_attention
        if self.layer_types and self.layer_types[-1] != "full_attention":
            self.layer_types[-1] = "full_attention"

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def num_experts_per_tok(self):
        """HF-standard alias for top_k_experts; required by executor MoE EP utilities."""
        return self.top_k_experts
