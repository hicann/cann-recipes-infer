# coding=utf-8
# Qwen3.5-MoE model configuration.
# Adapted from
# https://github.com/huggingface/transformers/blob/v5.8.0/src/transformers/models/qwen3_5_moe/configuration_qwen3_5_moe.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
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
"""Qwen3.5-MoE model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import RopeParameters
from transformers.utils import logging


logger = logging.get_logger(__name__)


class Qwen3_5MoeTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3_5MoeTextModel`].
    It is used to instantiate a Qwen3.5-MoE text model according to the specified arguments.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs.

    Args:
        vocab_size (`int`, *optional*, defaults to 248320):
            Vocabulary size of the Qwen3.5-MoE model.
        hidden_size (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        num_hidden_layers (`int`, *optional*, defaults to 40):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer.
        num_key_value_heads (`int`, *optional*, defaults to 2):
            Number of key_value heads for Grouped Query Attention.
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation for initializing weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values attentions.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_parameters (`RopeParameters`, *optional*):
            Configuration parameters for RoPE embeddings.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use bias in query, key, value and output projection layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for attention probabilities.
        head_dim (`int`, *optional*, defaults to 256):
            Dimension of each attention head.
        linear_conv_kernel_dim (`int`, *optional*, defaults to 4):
            Kernel size of the convolution used in linear attention layers.
        linear_key_head_dim (`int`, *optional*, defaults to 128):
            Dimension of each key head in linear attention.
        linear_value_head_dim (`int`, *optional*, defaults to 128):
            Dimension of each value head in linear attention.
        linear_num_key_heads (`int`, *optional*, defaults to 16):
            Number of key heads used in linear attention layers.
        linear_num_value_heads (`int`, *optional*, defaults to 32):
            Number of value heads used in linear attention layers.
        moe_intermediate_size (`int`, *optional*, defaults to 512):
            Intermediate size of the routed expert.
        shared_expert_intermediate_size (`int`, *optional*, defaults to 512):
            Intermediate size of the shared expert.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of selected experts per token.
        num_experts (`int`, *optional*, defaults to 256):
            Number of routed experts.
        output_router_logits (`bool`, *optional*, defaults to `False`):
            Whether to output router logits.
        router_aux_loss_coef (`float`, *optional*, defaults to 0.001):
            The aux loss factor for the total loss.
        layer_types (`list[str]`, *optional*):
            List of layer types ("full_attention" or "linear_attention") for each layer.
        full_attention_interval (`int`, *optional*, defaults to 4):
            Interval pattern for full attention layers when layer_types is not specified.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*):
            Beginning of stream token id.
        eos_token_id (`int` or `list[int]`, *optional*):
            End of stream token id.
    """

    model_type = "qwen3_5_moe_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_expert.gate_proj": "colwise",
        "layers.*.mlp.shared_expert.up_proj": "colwise",
        "layers.*.mlp.shared_expert.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=248320,
        hidden_size=2048,
        num_hidden_layers=40,
        num_attention_heads=16,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        head_dim=256,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts_per_tok=8,
        num_experts=256,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        layer_types=None,
        full_attention_interval=4,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_parameters = rope_parameters
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.head_dim = head_dim

        # Linear attention specific parameters
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads

        # MoE specific parameters
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

        # Layer types configuration
        if layer_types is None:
            self.layer_types = [
                "linear_attention" if bool((i + 1) % full_attention_interval) else "full_attention"
                for i in range(num_hidden_layers)
            ]
        else:
            self.layer_types = layer_types

        # Set partial_rotary_factor for backward compatibility
        kwargs.setdefault("partial_rotary_factor", 0.25)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )


class Qwen3_5MoeVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of the vision component
    in Qwen3.5-MoE multimodal model.

    Args:
        depth (`int`, *optional*, defaults to 27):
            Number of layers in the vision encoder.
        hidden_size (`int`, *optional*, defaults to 1152):
            Dimension of the hidden representations.
        hidden_act (`str`, *optional*, defaults to `"gelu_pytorch_tanh"`):
            The non-linear activation function.
        intermediate_size (`int`, *optional*, defaults to 4304):
            Dimension of the intermediate representations.
        num_heads (`int`, *optional*, defaults to 16):
            Number of attention heads.
        in_channels (`int`, *optional*, defaults to 3):
            Number of input channels.
        patch_size (`int` or `list[int]` or `tuple[int, int]`, *optional*, defaults to 16):
            Patch size for image tokenization.
        spatial_merge_size (`int`, *optional*, defaults to 2):
            Spatial merge size for vision features.
        temporal_patch_size (`int` or `list[int]` or `tuple[int, int]`, *optional*, defaults to 2):
            Temporal patch size for video processing.
        out_hidden_size (`int`, *optional*, defaults to 3584):
            Output hidden size of the vision model.
        num_position_embeddings (`int`, *optional*, defaults to 2304):
            Maximum sequence length for vision.
        initializer_range (`float`, *optional*, defaults to 0.02):
            Standard deviation for weight initialization.
    """

    model_type = "qwen3_5_moe"

    def __init__(
        self,
        depth=27,
        hidden_size=1152,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4304,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=3584,
        num_position_embeddings=2304,
        initializer_range=0.02,
        **kwargs,
    ):
        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range

        super().__init__(**kwargs)


class Qwen3_5MoeConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3_5MoeForConditionalGeneration`].
    It is used to instantiate a Qwen3.5-MoE multimodal model according to the specified arguments.

    Args:
        text_config (`dict` or `Qwen3_5MoeTextConfig`, *optional*):
            Configuration for the text model component.
        vision_config (`dict` or `Qwen3_5MoeVisionConfig`, *optional*):
            Configuration for the vision model component.
        image_token_id (`int`, *optional*, defaults to 248056):
            Token id for image tokens.
        video_token_id (`int`, *optional*, defaults to 248057):
            Token id for video tokens.
        vision_start_token_id (`int`, *optional*, defaults to 248053):
            Token id marking start of vision content.
        vision_end_token_id (`int`, *optional*, defaults to 248054):
            Token id marking end of vision content.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie word embeddings.
    """

    model_type = "qwen3_5_moe"
    sub_configs = {"vision_config": Qwen3_5MoeVisionConfig, "text_config": Qwen3_5MoeTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=248056,
        video_token_id=248057,
        vision_start_token_id=248053,
        vision_end_token_id=248054,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()
        else:
            self.text_config = text_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.tie_word_embeddings = tie_word_embeddings

        super().__init__(**kwargs)


__all__ = ["Qwen3_5MoeConfig", "Qwen3_5MoeTextConfig", "Qwen3_5MoeVisionConfig"]