# coding=utf-8
# Adapted from transformers/models/hy_v3/configuration_hy_v3.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Copyright 2026 Tencent HunYuan Team and The HuggingFace Inc. team. All rights reserved.
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

"""HYV3 model configuration"""

import os
import json
from typing import Optional


class HYV3Config:
    r"""
    Configuration class for HYV3 model.

    Args:
        vocab_size (`int`, *optional*, defaults to 120832):
            Vocabulary size.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the hidden states.
        intermediate_size (`int`, *optional*, defaults to 13312):
            Dimensionality of the Dense FFN layer.
        num_hidden_layers (`int`, *optional*, defaults to 80):
            Number of hidden layers in the Transformer.
        num_attention_heads (`int`, *optional*, defaults to 64):
            Number of attention heads.
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key-value attention heads (GQA).
        head_dim (`int`, *optional*, defaults to 128):
            Dimensionality of each attention head.
        hidden_act (`str`, *optional*, defaults to "silu"):
            Activation function.
        max_position_embeddings (`int`, *optional*, defaults to 262144):
            Maximum sequence length.
        rms_norm_eps (`float`, *optional*, defaults to 1e-5):
            Epsilon for RMS normalization.
        num_experts (`int`, *optional*, defaults to 192):
            Number of MoE experts.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of experts to route each token to.
        num_shared_experts (`int`, *optional*, defaults to 1):
            Number of shared experts.
        moe_intermediate_size (`int`, *optional*, defaults to 1536):
            Intermediate size of MoE experts.
        router_scaling_factor (`float`, *optional*, defaults to 2.826):
            Scaling factor applied to top-k expert weights.
        enable_moe_fp32_combine (`bool`, *optional*, defaults to False):
            Whether to combine shared expert outputs in fp32.
        rope_parameters (`dict`, *optional*):
            RoPE configuration parameters.
    """

    model_type = "hy_v3"
    default_theta = 11_158_840.0
    attribute_map = {
        "num_local_experts": "num_experts",
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 120832,
        hidden_size: int = 4096,
        intermediate_size: int = 13312,
        num_hidden_layers: int = 80,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 262144,
        initializer_range: float = 0.006,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int = 120002,
        bos_token_id: int = 120000,
        eos_token_id: int = 120025,
        tie_word_embeddings: bool = False,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        num_experts: int = 192,
        num_experts_per_tok: int = 8,
        num_shared_experts: int = 1,
        moe_intermediate_size: int = 1536,
        router_scaling_factor: float = 2.826,
        enable_moe_fp32_combine: bool = False,
        mlp_layer_types: Optional[list] = None,
        output_router_logits: bool = False,
        rope_parameters: Optional[dict] = None,
        num_nextn_predict_layers: int = 1,
        first_k_dense_replace: int = 1,
        qk_norm: bool = True,
        route_norm: bool = True,
        expert_hidden_dim: int = 1536,
        enable_attention_fp32_softmax: bool = False,
        enable_lm_head_fp32: bool = True,
        sep_token_id: int = 120007,
        eod_token_id: int = 120026,
        use_grouped_mm: bool = False,
        moe_router_enable_expert_bias: bool = True,
        moe_router_use_sigmoid: bool = True,
        torch_dtype: str = "bfloat16",
        quant_config: Optional[dict] = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        # Some checkpoints (e.g. the fp8 export) store eos_token_id as a list;
        # the tokenizer and the engine's stop check expect a scalar id.
        self.eos_token_id = (
            eos_token_id[0] if isinstance(eos_token_id, (list, tuple)) else eos_token_id
        )
        self.tie_word_embeddings = tie_word_embeddings
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.router_scaling_factor = router_scaling_factor
        self.enable_moe_fp32_combine = enable_moe_fp32_combine
        self.output_router_logits = output_router_logits
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.first_k_dense_replace = first_k_dense_replace
        self.qk_norm = qk_norm
        self.route_norm = route_norm
        self.expert_hidden_dim = expert_hidden_dim
        self.enable_attention_fp32_softmax = enable_attention_fp32_softmax
        self.enable_lm_head_fp32 = enable_lm_head_fp32
        self.sep_token_id = sep_token_id
        self.eod_token_id = eod_token_id
        self.use_grouped_mm = use_grouped_mm
        self.moe_router_enable_expert_bias = moe_router_enable_expert_bias
        self.moe_router_use_sigmoid = moe_router_use_sigmoid
        self.torch_dtype = torch_dtype
        self.quant_config = quant_config

        # Number of local experts
        self.num_local_experts = self.num_experts

        # Build mlp_layer_types if not provided
        if mlp_layer_types is None:
            self.mlp_layer_types = (
                ["dense"] + ["sparse"] * (self.num_hidden_layers - 1)
            )
        else:
            self.mlp_layer_types = mlp_layer_types

        # Default rope_parameters if not provided
        if rope_parameters is None:
            self.rope_parameters = {
                "rope_type": "default",
                "rope_theta": self.default_theta,
            }
        else:
            self.rope_parameters = rope_parameters

        self.rope_theta = self.rope_parameters.get("rope_theta", self.default_theta)

        # Keys to ignore on load
        self._keys_to_ignore_on_load_unexpected = [r"model\.layers\.80.*"]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load config from a pretrained model directory."""
        config_path = pretrained_model_name_or_path
        if os.path.isdir(pretrained_model_name_or_path):
            config_path = os.path.join(pretrained_model_name_or_path, "config.json")

        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        # Filter out kwargs that are not config fields
        for key in [
            "torch_dtype", "low_cpu_mem_usage", "ignore_mismatched_sizes",
            "runner_settings", "quant_config", "comm_manager",
        ]:
            config_dict.pop(key, None)

        # Merge explicit kwargs
        for key in ["torch_dtype"]:
            if key in kwargs:
                config_dict[key] = kwargs.pop(key)

        # Preserve compressed-tensors / fp8 quant metadata so the worker's
        # _verify_quantization detects quant_method and binds the quant methods.
        quantization_config = config_dict.pop("quantization_config", None)
        config = cls(**config_dict)
        config.quantization_config = quantization_config
        return config
