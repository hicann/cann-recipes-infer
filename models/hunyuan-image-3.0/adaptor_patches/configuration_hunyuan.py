# Adapted from
# https://github.com/Tencent-Hunyuan/HunyuanImage-3.0,
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026.
# Copyright (C) 2025 THL A29 Limited, a Tencent company. All rights reserved.
#
# This code is based on Tencent-Hunyuan's HunyuanImage-3.0 library and the
# HunyuanImage-3.0 implementations in this library. It has been modified from
# its original forms to accommodate minor architectural differences compared
# to HunyuanImage-3.0 used by Tencent-Hunyuan team that trained the model.
# ================================================================================
#
# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ================================================================================

from typing import List, Union
from hunyuan_image_3.configuration_hunyuan import HunyuanImage3Config


def init(
        self,
        vocab_size=290943,
        hidden_size=4096,
        intermediate_size: int = 11008,
        moe_intermediate_size: Union[int, List] = None,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        attention_head_dim=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        eod_token_id=3,
        im_start_id=4,
        im_end_id=5,
        text_start_id=6,
        text_end_id=7,
        image_token_id=8,
        video_start_id=9,
        video_end_id=10,
        im_newline_id=11,
        mask_init_id=12,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        mlp_bias=False,
        attention_dropout=0.0,
        use_qk_norm=False,
        use_rotary_pos_emb=True,
        use_cla=False,
        cla_share_factor=1,
        norm_type="hf_rms",
        num_experts: Union[int, List] = 1,
        use_mixed_mlp_moe=False,
        num_shared_expert: Union[int, List] = 1,
        moe_topk: Union[int, List] = 1,
        capacity_factor: int = 1.0,
        moe_drop_tokens=False,
        moe_random_routing_dropped_token=False,
        use_mla=False,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=128,
        moe_layer_num_skipped=0,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        group_limited_greedy=False,
        n_group=None,
        topk_group=None,
        add_classification_head=False,
        class_num=0,
        pool_type="last",
        pad_id=-1,
        # Added
        moe_impl="eager",
        vae_downsample_factor=(16, 16),     # (h, w)
        img_proj_type="unet",
        patch_size=1,
        patch_embed_hidden_dim=1024,
        image_base_size=1024,
        vae=None,
        vit=None,
        vit_processor=None,
        vit_aligner=None,
        **kwargs,
):
    self.vocab_size = vocab_size
    self.max_position_embeddings = max_position_embeddings
    self.hidden_size = hidden_size
    self.intermediate_size = intermediate_size
    self.moe_intermediate_size = moe_intermediate_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.moe_impl = moe_impl
    self.num_experts = num_experts
    self.use_mixed_mlp_moe = use_mixed_mlp_moe
    self.num_shared_expert = num_shared_expert
    self.moe_topk = moe_topk
    self.capacity_factor = capacity_factor
    self.moe_drop_tokens = moe_drop_tokens
    self.moe_random_routing_dropped_token = moe_random_routing_dropped_token
    self.moe_tp = kwargs.get('moe_tp', False)

    if attention_head_dim is not None:
        self.attention_head_dim = attention_head_dim
    else:
        self.attention_head_dim = self.hidden_size // num_attention_heads

    # for backward compatibility
    if num_key_value_heads is None:
        num_key_value_heads = num_attention_heads

    self.num_key_value_heads = num_key_value_heads
    self.hidden_act = hidden_act
    self.initializer_range = initializer_range
    self.rms_norm_eps = rms_norm_eps
    self.pretraining_tp = pretraining_tp
    self.use_cache = use_cache
    self.rope_theta = rope_theta
    self.rope_scaling = rope_scaling
    self.attention_bias = attention_bias
    self.mlp_bias = mlp_bias
    self.attention_dropout = attention_dropout
    self.use_qk_norm = use_qk_norm
    self.use_rotary_pos_emb = use_rotary_pos_emb
    self.use_cla = use_cla
    self.cla_share_factor = cla_share_factor
    self.norm_type = norm_type
    # MLA args
    self.use_mla = use_mla
    self.kv_lora_rank = kv_lora_rank
    self.q_lora_rank = q_lora_rank
    self.qk_rope_head_dim = qk_rope_head_dim
    self.qk_nope_head_dim = qk_nope_head_dim
    self.v_head_dim = v_head_dim

    # DeepSeek related args
    self.moe_layer_num_skipped = moe_layer_num_skipped
    self.norm_topk_prob = norm_topk_prob
    self.routed_scaling_factor = routed_scaling_factor
    self.group_limited_greedy = group_limited_greedy
    self.n_group = n_group
    self.topk_group = topk_group
    self.add_classification_head = add_classification_head
    self.class_num = class_num
    self.pool_type = pool_type
    self.pad_id = pad_id

    if self.class_num is not None:
        self.dense_list = [self.hidden_size, self.class_num]

    # ViT args
    self.vit = vit
    self.vit_processor = vit_processor
    self.vit_aligner = vit_aligner

    # Image Gen args
    self.vae = vae
    self.vae_downsample_factor = vae_downsample_factor
    self.img_proj_type = img_proj_type
    self.patch_size = patch_size
    self.patch_embed_hidden_dim = patch_embed_hidden_dim
    self.image_base_size = image_base_size

    # token id
    self.eod_token_id = eod_token_id
    self.im_start_id = im_start_id
    self.im_end_id = im_end_id
    self.text_start_id = text_start_id
    self.text_end_id = text_end_id
    self.image_token_id = image_token_id
    self.video_start_id = video_start_id
    self.video_end_id = video_end_id
    self.im_newline_id = im_newline_id
    self.mask_init_id = mask_init_id

    self.attn_implementation = "npu"

    super(HunyuanImage3Config, self).__init__(
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        tie_word_embeddings=tie_word_embeddings,
        **kwargs,
    )

HunyuanImage3Config.__init__ = init