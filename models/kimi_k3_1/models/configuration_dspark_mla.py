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
    """Normalize the Kimi K3 MLA DSpark checkpoint configuration."""

    model_type = "k3_dspark"

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
        num_key_value_heads: int = 64,
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        mla_use_nope: bool = False,
        mla_use_output_gate: bool = False,
        mla_use_qk_norm: bool = False,
        dspark_bonus_anchor: bool = False,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-5,
        max_position_embeddings: int = 1048576,
        rope_theta: float = 10000.0,
        rope_parameters=None,
        rope_scaling=None,
        vocab_size: int = 163840,
        draft_vocab_size=None,
        num_target_layers: int = 5,
        target_num_hidden_layers=None,
        target_hidden_size=None,
        target_layer_ids=None,
        dflash_config=None,
        mask_token_id=None,
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
        configured_target_ids = (
            target_layer_ids
            if target_layer_ids is not None
            else dflash_config.get("target_layer_ids")
        )
        resolved_target_ids = list(
            configured_target_ids
            if configured_target_ids is not None
            else [7, 23, 51, 67, 83]
        )
        if int(num_target_layers) != len(resolved_target_ids):
            raise ValueError(
                "num_target_layers must equal the number of target_layer_ids"
            )
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.q_lora_rank = int(q_lora_rank)
        self.kv_lora_rank = int(kv_lora_rank)
        self.qk_nope_head_dim = int(qk_nope_head_dim)
        self.qk_rope_head_dim = int(qk_rope_head_dim)
        self.v_head_dim = int(v_head_dim)
        self.head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.mla_use_nope = bool(mla_use_nope)
        self.mla_use_output_gate = bool(mla_use_output_gate)
        self.mla_use_qk_norm = bool(mla_use_qk_norm)
        self.dspark_bonus_anchor = bool(dspark_bonus_anchor)
        self.hidden_act = hidden_act
        self.rms_norm_eps = float(rms_norm_eps)
        self.max_position_embeddings = int(max_position_embeddings)
        self.rope_theta = float(rope_theta)
        self.rope_parameters = deepcopy(rope_parameters)
        self.rope_scaling = deepcopy(rope_scaling)
        self.vocab_size = int(vocab_size)
        self.draft_vocab_size = int(draft_vocab_size or vocab_size)
        self.target_hidden_size = int(target_hidden_size or hidden_size)
        # New K3 checkpoints distinguish the number of captured target states
        # from the target model's total depth.  Keep a fallback for older
        # configs that only carried ``num_target_layers``.
        self.target_num_hidden_layers = int(
            target_num_hidden_layers
            if target_num_hidden_layers is not None
            else num_target_layers
        )
        self.target_layer_ids = resolved_target_ids
        self.num_target_layers = len(resolved_target_ids)
        self.dflash_config = dflash_config
        self.mask_token_id = int(
            mask_token_id
            if mask_token_id is not None
            else dflash_config.get("mask_token_id", 163837)
        )
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
