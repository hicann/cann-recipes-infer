# coding=utf-8
# Adapted from
# https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/configuration_glm5_next.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Copyright 2026 The GLM & ZhipuAI team and HuggingFace Inc. team. All rights reserved.
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

from transformers.configuration_utils import PreTrainedConfig

_TEXT_DEFAULTS = {
    "vocab_size": 154880,
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "moe_intermediate_size": 2048,
    "num_hidden_layers": 45,
    "num_attention_heads": 64,
    "num_key_value_heads": 64,
    "n_shared_experts": 1,
    "n_routed_experts": 288,
    "routed_scaling_factor": 2.5,
    "kv_lora_rank": 512,
    "q_lora_rank": 1536,
    "qk_rope_head_dim": 0,
    "qk_nope_head_dim": 256,
    "v_head_dim": 256,
    "n_group": 1,
    "topk_group": 1,
    "num_experts_per_tok": 8,
    "norm_topk_prob": True,
    "hidden_act": "silu",
    "max_position_embeddings": 1048576,
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-5,
    "use_cache": True,
    "pad_token_id": 154820,
    "bos_token_id": None,
    "eos_token_id": None,
    "tie_word_embeddings": False,
    # hybrid layer schedule
    "layer_types": None,
    "mlp_layer_types": None,
    "first_k_dense_replace": 3,
    "moe_layer_freq": 1,
    # KDA linear attention
    "linear_attn_config": None,
    # DSA indexer (with k-pool compression)
    "index_topk": 2048,
    "index_head_dim": 128,
    "index_n_heads": 32,
    "index_kpool": 4,
    "index_kpool_compress": True,
    "index_kpool_always_select_tail": True,
    "indexer_types": None,
    "index_share_for_mtp_iteration": True,
    "indexer_rope_interleave": True,
    # mHC
    "mhc": True,
    "hc_mult": 4,
    "hc_eps": 1e-6,
    "hc_sinkhorn_iters": 20,
    # misc
    "mla_use_nope": True,
    "swiglu_limit": 10.0,
    "moe_router_dtype": "float32",
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "num_nextn_predict_layers": 1,
    "attention_bias": False,
    "attention_dropout": 0.0,
}


class Glm53Config(PreTrainedConfig):
    r"""Configuration for GLM-5.3-Flash (upstream model_type `glm5_next`).

    The released config.json is multimodal: the text fields live in a nested
    `text_config`. This class flattens it to the top level so the rest of the
    recipe reads `config.hidden_size` directly; `vision_config` is dropped and
    `model.visual.*` is skipped at load time (text-only recipe).

    See README.md for the architecture; the fields below carry the defaults.
    """

    model_type = "glm5_next"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_local_experts": "n_routed_experts",
    }

    def __init__(self, **kwargs):
        text_config = kwargs.pop("text_config", None)
        kwargs.pop("vision_config", None)  # vision tower unsupported in this recipe

        fields = dict(_TEXT_DEFAULTS)
        for key in list(kwargs.keys()):
            if key in fields:
                fields[key] = kwargs.pop(key)
        if isinstance(text_config, dict):
            for key, value in text_config.items():
                if key == "model_type":
                    continue
                if key in fields:
                    fields[key] = value
                else:
                    kwargs[key] = value

        # Model dimensions
        self.vocab_size = fields["vocab_size"]
        self.hidden_size = fields["hidden_size"]
        self.intermediate_size = fields["intermediate_size"]
        self.moe_intermediate_size = fields["moe_intermediate_size"]
        self.num_hidden_layers = fields["num_hidden_layers"]
        self.max_position_embeddings = fields["max_position_embeddings"]

        # Attention dimensions (MLA, NoPE)
        self.num_attention_heads = fields["num_attention_heads"]
        self.num_key_value_heads = fields["num_key_value_heads"]
        self.kv_lora_rank = fields["kv_lora_rank"]
        self.q_lora_rank = fields["q_lora_rank"]
        self.qk_rope_head_dim = fields["qk_rope_head_dim"]
        self.qk_nope_head_dim = fields["qk_nope_head_dim"]
        self.v_head_dim = fields["v_head_dim"]
        self.mla_use_nope = fields["mla_use_nope"]

        # MoE parameters
        self.n_shared_experts = fields["n_shared_experts"]
        self.n_routed_experts = fields["n_routed_experts"]
        self.routed_scaling_factor = fields["routed_scaling_factor"]
        self.n_group = fields["n_group"]
        self.topk_group = fields["topk_group"]
        self.num_experts_per_tok = fields["num_experts_per_tok"]
        self.norm_topk_prob = fields["norm_topk_prob"]
        self.first_k_dense_replace = fields["first_k_dense_replace"]
        self.moe_layer_freq = fields["moe_layer_freq"] or 1
        self.scoring_func = fields["scoring_func"]
        self.topk_method = fields["topk_method"]
        self.moe_router_dtype = fields["moe_router_dtype"]
        self.num_nextn_predict_layers = fields["num_nextn_predict_layers"]
        self.swiglu_limit = fields["swiglu_limit"]

        # Hybrid layer schedule. Default mirrors the HF reference:
        # layer i is KDA unless i % 4 == 3 (DSA).
        self.layer_types = fields["layer_types"]
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention" if idx % 4 != 3 else "full_attention"
                for idx in range(self.num_hidden_layers)
            ]
        # The checkpoint spells DSA layers "deepseek_sparse_attention"; normalize
        # to "full_attention" because transformers' strict PreTrainedConfig
        # validator (validate_layer_type) only accepts ALLOWED_LAYER_TYPES.
        self.layer_types = [
            "full_attention" if t == "deepseek_sparse_attention" else t
            for t in self.layer_types
        ]

        self.mlp_layer_types = fields["mlp_layer_types"]
        if self.mlp_layer_types is None:
            n_dense = min(self.first_k_dense_replace, self.num_hidden_layers)
            self.mlp_layer_types = (
                ["dense"] * n_dense + ["sparse"] * (self.num_hidden_layers - n_dense))

        # KDA linear attention parameters (nested dict in the checkpoint config)
        linear_attn_config = fields["linear_attn_config"] or {}
        self.linear_attn_config = linear_attn_config
        self.linear_num_heads = linear_attn_config.get("num_heads", 64)
        self.linear_head_dim = linear_attn_config.get("head_dim", 128)
        self.linear_conv_kernel_dim = linear_attn_config.get("short_conv_kernel_size", 4)
        self.linear_lower_bound = linear_attn_config.get("gate_lower_bound", -5.0)
        if linear_attn_config.get("safe_gate", True) and self.linear_lower_bound is None:
            self.linear_lower_bound = -5.0

        # DSA indexer (with k-pool compression)
        self.index_topk = fields["index_topk"]
        self.index_head_dim = fields["index_head_dim"]
        self.index_n_heads = fields["index_n_heads"]
        self.index_kpool = fields["index_kpool"]
        self.index_kpool_compress = fields["index_kpool_compress"]
        self.index_kpool_always_select_tail = fields["index_kpool_always_select_tail"]
        self.index_share_for_mtp_iteration = fields["index_share_for_mtp_iteration"]
        self.indexer_rope_interleave = fields["indexer_rope_interleave"]
        # GLM-5.3-Flash ships all-"full" indexer_types (no cross-layer IndexShare),
        # but keep the schedule plumbing for future checkpoints.
        self.indexer_types = fields["indexer_types"]
        if self.indexer_types is None:
            self.indexer_types = ["full"] * self.num_hidden_layers

        # mHC
        self.mhc = fields["mhc"]
        self.hc_mult = fields["hc_mult"]
        self.hc_eps = fields["hc_eps"]
        self.hc_sinkhorn_iters = fields["hc_sinkhorn_iters"]

        # General
        self.hidden_act = fields["hidden_act"]
        self.initializer_range = fields["initializer_range"]
        self.rms_norm_eps = fields["rms_norm_eps"]
        self.use_cache = fields["use_cache"]
        self.attention_bias = fields["attention_bias"]
        self.attention_dropout = fields["attention_dropout"]
        self.quant_config = kwargs.pop("quant_config", None)

        super().__init__(
            pad_token_id=fields["pad_token_id"],
            bos_token_id=fields["bos_token_id"],
            eos_token_id=fields["eos_token_id"],
            tie_word_embeddings=fields["tie_word_embeddings"],
            **kwargs,
        )

        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.head_dim = self.qk_rope_head_dim

        self._validate()

    def _validate(self):
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types length {len(self.layer_types)} != num_hidden_layers "
                f"{self.num_hidden_layers}")
        if len(self.mlp_layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"mlp_layer_types length {len(self.mlp_layer_types)} != num_hidden_layers "
                f"{self.num_hidden_layers}")
        if self.qk_rope_head_dim != 0:
            raise ValueError(
                f"GLM-5.3 DSA layers are NoPE; expected qk_rope_head_dim == 0, "
                f"got {self.qk_rope_head_dim}")
        if self.index_kpool < 1:
            raise ValueError(f"index_kpool must be positive, got {self.index_kpool}")
        if self.index_topk % self.index_kpool != 0:
            raise ValueError(
                f"index_topk ({self.index_topk}) must be divisible by index_kpool "
                f"({self.index_kpool})")
        if self.q_lora_rank is None:
            raise ValueError("GLM-5.3 DSA attention requires q_lora_rank")

    # ---- layer schedule helpers ---------------------------------------
    def is_kda_layer(self, layer_idx: int) -> bool:
        """The MTP layer (index == num_hidden_layers) is always a DSA layer."""
        if layer_idx >= self.num_hidden_layers:
            return False
        return self.layer_types[layer_idx] == "linear_attention"

    def is_moe_layer(self, layer_idx: int) -> bool:
        if layer_idx >= self.num_hidden_layers:  # MTP layer is MoE
            return True
        return (
            self.n_routed_experts is not None
            and self.mlp_layer_types[layer_idx] == "sparse"
        )


__all__ = ["Glm53Config"]
