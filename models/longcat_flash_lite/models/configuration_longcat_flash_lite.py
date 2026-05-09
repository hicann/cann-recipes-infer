# coding=utf-8
# Adapted from
# https://huggingface.co/meituan-longcat/LongCat-Flash-Chat-FP8/blob/main/configuration_longcat_flash.py
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/deepseek_v3/configuration_deepseek_v3.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Copyright 2025 bzantium and the HuggingFace Inc. team. All rights reserved.
# Copyright 2024 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
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
from transformers.modeling_rope_utils import rope_config_validation


class LongcatFlashConfig(PretrainedConfig):

    model_type = "longcat_flash"

    def _build_rope_parameters(self, rope_theta, rope_scaling):
        params = {"rope_theta": rope_theta}
        if rope_scaling:
            params.update(rope_scaling)
        params.setdefault("rope_type", "default")
        return params

    def __init__(
        self,
        vocab_size=131072,
        hidden_size=7168,
        ffn_hidden_size=18432,
        expert_ffn_hidden_size=2048,
        num_layers=61,
        num_nextn_predict_layers=1,
        num_attention_heads=128,
        num_key_value_heads=None,
        n_routed_experts=256,
        routed_scaling_factor=1,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=128,
        mla_scale_q_lora=True,
        mla_scale_kv_lora=True,
        moe_topk=8,
        norm_topk_prob=False,
        hidden_act="silu",
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        attention_method='MLA',
        initializer_range=0.006,
        router_bias=False,
        zero_expert_num=None,
        zero_expert_type=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.expert_ffn_hidden_size = expert_ffn_hidden_size
        self.num_layers = num_layers
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.num_attention_heads = num_attention_heads
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.moe_topk = moe_topk
        self.norm_topk_prob = norm_topk_prob
        self.mla_scale_q_lora = mla_scale_q_lora
        self.mla_scale_kv_lora = mla_scale_kv_lora
        self.attention_method = attention_method
        self.initializer_range = initializer_range
        self.router_bias = router_bias
        self.zero_expert_num = zero_expert_num
        self.zero_expert_type = zero_expert_type

        if self.attention_method == "MLA":
            self.head_dim = qk_rope_head_dim
        else:
            raise ValueError('attention_method should be one of ["MLA"]')

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_parameters = self._build_rope_parameters(rope_theta, rope_scaling)
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        rope_config_validation(self)

        # Prevent PretrainedConfig from overwriting computed attributes via kwargs
        kwargs.pop("qk_head_dim", None)
        kwargs.pop("head_dim", None)
        kwargs.pop("num_hidden_layers", None)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def num_hidden_layers(self):
        return self.num_layers

    @property
    def num_experts_per_tok(self):
        """HF-standard alias for moe_topk; required by executor MoE EP utilities."""
        return self.moe_topk

    @property
    def num_experts(self):
        """HF-standard alias for n_routed_experts; required by parallel_config validation."""
        return self.n_routed_experts


class LongcatFlashNgramConfig(LongcatFlashConfig):

    model_type = "longcat_flash_ngram"

    def __init__(
        self,
        # Defaults match LongCat-Flash-Lite config.json. Larger variants
        # override every field via from_pretrained, so changing these
        # defaults does not affect real deployments.
        vocab_size=131072,
        hidden_size=3072,
        num_layers=14,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=327680,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=5000000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        ffn_hidden_size=6144,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        head_dim=64,
        v_head_dim=128,
        qk_head_dim=None,
        moe_topk=12,
        n_routed_experts=256,
        zero_expert_num=128,
        expert_ffn_hidden_size=1024,
        routed_scaling_factor=6.0,
        emb_neighbor_num=4,
        emb_split_num=4,
        ngram_vocab_size_ratio=78,
        **kwargs,
    ):
        self.emb_neighbor_num = emb_neighbor_num
        self.emb_split_num = emb_split_num
        self.ngram_vocab_size_ratio = ngram_vocab_size_ratio

        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            attention_dropout=attention_dropout,
            ffn_hidden_size=ffn_hidden_size,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            moe_topk=moe_topk,
            n_routed_experts=n_routed_experts,
            zero_expert_num=zero_expert_num,
            expert_ffn_hidden_size=expert_ffn_hidden_size,
            routed_scaling_factor=routed_scaling_factor,
            **kwargs,
        )


__all__ = ["LongcatFlashConfig", "LongcatFlashNgramConfig"]
