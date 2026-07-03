# coding=utf-8
# Adapted from
# https://huggingface.co/FreedomIntelligence/openPangu-Embedded-7B/blob/main/configuration_openpangu_dense.py
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


class PanguEmbeddedConfig(PretrainedConfig):
    r"""
    Configuration for the Pangu Embedded 7B model.

    This adapter intentionally accepts multiple alias names that appear in
    different config.json formats (e.g. GPT-style or LLaMA-style fields).

    Args:
        vocab_size (`int`, *optional*):
            Vocabulary size of the model.
        hidden_size (`int`, *optional*):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            Number of key/value heads for grouped query attention.
        head_dim (`int`, *optional*):
            Dimension of each attention head. Defaults to `hidden_size // num_attention_heads` when omitted.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values attentions.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in attention projections.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
    """

    model_type = "PanguEmbedded"
    _auto_class = "AutoConfig"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=153376,
        hidden_size=4096,
        intermediate_size=12800,
        num_hidden_layers=34,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=None,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=16000000.0,
        rope_scaling=None,
        attention_bias=True,
        attention_dropout=0.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=45892,
        **kwargs,
    ):
        vocab_size = _first_not_none(vocab_size, kwargs.pop("n_vocab", None))
        hidden_size = _first_not_none(hidden_size, kwargs.pop("n_embd", None))
        intermediate_size = _first_not_none(intermediate_size, kwargs.pop("n_inner", None))
        num_hidden_layers = _first_not_none(
            num_hidden_layers,
            kwargs.pop("n_layer", None),
            kwargs.pop("num_layers", None),
        )
        num_attention_heads = _first_not_none(
            num_attention_heads,
            kwargs.pop("n_head", None),
            kwargs.pop("num_heads", None),
        )
        num_key_value_heads = _first_not_none(
            num_key_value_heads,
            kwargs.pop("n_kv_head", None),
            kwargs.pop("num_kv_heads", None),
        )
        head_dim = _first_not_none(head_dim, kwargs.pop("head_dim", None))
        max_position_embeddings = _first_not_none(
            max_position_embeddings,
            kwargs.pop("n_positions", None),
            kwargs.pop("max_seq_len", None),
        )
        rope_theta = _first_not_none(rope_theta, kwargs.pop("rotary_base", None))
        rms_norm_eps = _first_not_none(rms_norm_eps, kwargs.pop("layer_norm_eps", None))
        hidden_act = _first_not_none(hidden_act, kwargs.pop("activation_function", None))

        # Check for missing required fields
        required_fields = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "max_position_embeddings": max_position_embeddings,
        }
        missing = [name for name, value in required_fields.items() if value is None]
        if missing:
            raise ValueError(
                "PanguEmbeddedConfig requires the following fields to be set: "
                f"{', '.join(missing)}. Please load from the official config.json or "
                "pass them explicitly."
            )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.quantization_config = kwargs.get('quantization_config', {})
        self.quant_config = kwargs.get('quant_config', None)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["PanguEmbeddedConfig"]
