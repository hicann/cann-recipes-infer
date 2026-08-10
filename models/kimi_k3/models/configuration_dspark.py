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

from copy import deepcopy

from transformers.configuration_utils import PretrainedConfig


class KimiK3DSparkConfig(PretrainedConfig):
    """Normalize the published Qwen3 DFlash config for this executor."""

    model_type = "qwen3"

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        kwargs.pop("runner_settings", None)
        config_dict = deepcopy(config_dict)
        return super().from_dict(config_dict, **kwargs)

    def __init__(
        self,
        hidden_size: int = 7168,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 5,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 16,
        head_dim: int = 64,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 1048576,
        rope_theta: float = 10000.0,
        rope_parameters=None,
        rope_scaling=None,
        vocab_size: int = 163840,
        num_target_layers: int = 93,
        target_hidden_size=None,
        target_layer_ids=None,
        dflash_config=None,
        markov_rank: int = 256,
        markov_head_type: str = "vanilla",
        block_size: int = 7,
        enable_confidence_head: bool = True,
        confidence_head_with_markov: bool = True,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        sliding_window=None,
        use_sliding_window: bool = False,
        quantization_config=None,
        compression_config=None,
        **kwargs,
    ) -> None:
        dflash_config = deepcopy(dflash_config) or {}
        resolved_target_ids = list(
            target_layer_ids
            or dflash_config.get("target_layer_ids")
            or [7, 23, 51, 67, 83]
        )
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.qk_rope_head_dim = self.head_dim
        self.hidden_act = hidden_act
        self.rms_norm_eps = float(rms_norm_eps)
        self.max_position_embeddings = int(max_position_embeddings)
        self.rope_theta = float(rope_theta)
        self.rope_parameters = deepcopy(rope_parameters)
        self.rope_scaling = deepcopy(rope_scaling)
        self.vocab_size = int(vocab_size)
        self.draft_vocab_size = self.vocab_size
        self.target_hidden_size = int(target_hidden_size or hidden_size)
        # The checkpoint's num_target_layers=93 describes the target depth.
        self.target_num_hidden_layers = int(num_target_layers)
        self.target_layer_ids = resolved_target_ids
        self.num_target_layers = len(resolved_target_ids)
        self.dflash_config = dflash_config
        self.mask_token_id = int(dflash_config.get("mask_token_id", 163824))
        self.markov_rank = int(markov_rank)
        self.markov_head_type = str(markov_head_type)
        self.block_size = int(block_size)
        self.enable_confidence_head = bool(enable_confidence_head)
        self.confidence_head_with_markov = bool(confidence_head_with_markov)
        self.attention_bias = bool(attention_bias)
        self.attention_dropout = float(attention_dropout)
        self.sliding_window = sliding_window
        self.use_sliding_window = bool(use_sliding_window)
        self.quantization_config = deepcopy(quantization_config)
        self.compression_config = deepcopy(compression_config)
        super().__init__(**kwargs)


__all__ = ["KimiK3DSparkConfig"]
