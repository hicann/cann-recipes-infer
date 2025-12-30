# Adapted from  
# https://github.com/Tencent-Hunyuan/HunyuanImage-3.0,
# Copyright (c) Huawei Technologies Co., Ltd. 2025.
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

import math
import os
import warnings
from typing import List, Union, Optional, Tuple

import torch
import torch.distributed as dist

from einops import rearrange
from torch import nn
import torch_npu
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast

try:
    import flashinfer
except Exception as e:
    flashinfer = None

from hunyuan_image_3.autoencoder_kl_3d import AutoencoderKLConv3D
from hunyuan_image_3.configuration_hunyuan import HunyuanImage3Config
from hunyuan_image_3.image_processor import HunyuanImage3ImageProcessor
from hunyuan_image_3.siglip2 import Siglip2VisionTransformer, LightProjector
from hunyuan_image_3.tokenizer_wrapper import TokenizerWrapper, TokenizerEncodeOutput
from hunyuan_image_3.hunyuan import (
    build_batch_2d_rope,
    default,
    real_batched_index_select,
    to_device
)
from hunyuan_image_3.hunyuan import (
    Hunyuan_ATTENTION_CLASSES,
    Hunyuan_INPUTS_DOCSTRING,
    HunyuanImage3DecoderLayer,
    HunyuanImage3ForCausalMM,
    HunyuanImage3SDPAAttention,
    HunyuanImage3Model,
    HunyuanMLP,
    HunyuanMoE,
    HunyuanRMSNorm,
    HunyuanStaticCache,
    HunyuanTopKGate,
    TimestepEmbedder,
    UNetDown,
    UNetUp
)
from executor.utils import init_comm_group, get_default_group
from module.linear import (
    RowParallelLinear,
    QKVParallelLinear
)
from module.fuse_moe_gmm import FusedMoEGMM

local_rank = int(os.environ['LOCAL_RANK'])


# =======================================================
#     Helper Functions
# =======================================================


def cfg_split_model_inputs(x: TokenizerEncodeOutput, select=0):
    x.tokens = x.tokens[select:select + 1] if x.tokens is not None else None
    x.timestep_scatter_index = \
        x.timestep_scatter_index[select:select + 1] if x.timestep_scatter_index is not None else None
    x.guidance_scatter_index = \
        x.guidance_scatter_index[select:select + 1] if x.guidance_scatter_index is not None else None
    x.text_slices = x.text_slices[select:select + 1] if x.text_slices is not None else None
    x.gen_image_slices = x.gen_image_slices[select:select + 1] if x.gen_image_slices is not None else None
    x.joint_image_slices = x.joint_image_slices[select:select + 1] if x.joint_image_slices is not None else None
    x.cond_vae_image_slices = \
        x.cond_vae_image_slices[select:select + 1] if x.cond_vae_image_slices is not None else None
    x.cond_vit_image_slices = \
        x.cond_vit_image_slices[select:select + 1] if x.cond_vit_image_slices is not None else None
    x.text_mask = x.text_mask[select:select + 1] if x.text_mask is not None else None
    x.gen_image_mask = x.gen_image_mask[select:select + 1] if x.gen_image_mask is not None else None
    x.cond_vae_image_mask = x.cond_vae_image_mask[select:select + 1] if x.cond_vae_image_mask is not None else None
    x.cond_vit_image_mask = x.cond_vit_image_mask[select:select + 1] if x.cond_vit_image_mask is not None else None
    x.real_pos = x.real_pos[select:select + 1] if x.real_pos is not None else None
    x.all_image_slices = x.all_image_slices[select:select + 1] if x.all_image_slices is not None else None
    x.cond_timestep_scatter_index = \
        x.cond_timestep_scatter_index[select:select + 1] if x.cond_timestep_scatter_index is not None else None
    x.gen_timestep_scatter_index = \
        x.gen_timestep_scatter_index[select:select + 1] if x.gen_timestep_scatter_index is not None else None


def npu_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    if position_ids is not None:
        cos = cos[position_ids]
        sin = sin[position_ids]

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    return torch_npu.npu_rotary_mul(q, cos, sin), torch_npu.npu_rotary_mul(k, cos, sin)


def rms_norm_forward(self, hidden_states, *args):
    if len(args) == 0: # only hidden_states exists
        result = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
        return result
    elif len(args) == 1 and args[0] is None: # residual is None
        result = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
        residual = hidden_states
        return (result, residual)
    elif len(args) == 1: # residual is not None
        residual = args[0]
        y, _, x = torch_npu.npu_add_rms_norm(residual, hidden_states, self.weight, self.variance_epsilon)
        return (y, x)
    else:
        raise NotImplementedError(
            f"insupportable HunyuanRMSNorm for input_args len as (include hid): {len(args) + 1}"
        )


def top_k_gate_forward(self, hidden_states, topk_impl='default'):
    bsz, seq_len, hidden_size = hidden_states.shape
    hidden_states = hidden_states.reshape(-1, hidden_size)
    with torch.npu.amp.autocast(enabled=False):
        if self.wg.weight.dtype == torch.float32:
            hidden_states = hidden_states.float()
        logits = self.wg(hidden_states)
    if topk_impl == 'default':
        gate_output = topkgating(logits, self.moe_topk, group_limited_greedy=self.group_limited_greedy,
                                    n_group=self.n_group, topk_group=self.topk_group,
                                    norm_topk_prob=self.norm_topk_prob,
                                    routed_scaling_factor=self.routed_scaling_factor,
                                    capacity_factor=self.config.capacity_factor,
                                    drop_tokens=self.drop_tokens)
    elif topk_impl == 'easy':
        gate_output = self.easy_topk(logits, self.moe_topk)
    else:
        raise ValueError(f"Unsupported topk_impl: {topk_impl}")

    return gate_output


def moe_init(self, config: HunyuanImage3Config, layer_idx: Optional[int] = None, **kwargs):
    super(HunyuanMoE, self).__init__()
    self.config = config
    self.layer_idx = layer_idx
    self.moe_topk = config.moe_topk[0]
    self.num_experts = config.num_experts if isinstance(config.num_experts, int) else config.num_experts[layer_idx]
    if config.use_mixed_mlp_moe:
        self.shared_mlp = HunyuanMLP(config, layer_idx=layer_idx, is_shared_mlp=True)
    self.gate = HunyuanTopKGate(config, layer_idx=layer_idx)
    self._moe_impl = config.moe_impl
    self.hccl_comm_dict = kwargs.get("hccl_comm_dict")
    self.moe_tp_size = self.hccl_comm_dict.get("moe_tp_size")
    self.moe_tp_group = self.hccl_comm_dict["moe_tp_group"]
    self.moe_ep_size = 1
    self.experts_per_rank = config.num_experts // self.moe_ep_size
    if self._moe_impl == "npu_grouped_matmul":
        self.experts = FusedMoEGMM(
            num_experts=self.num_experts,
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.intermediate_size,
            bias=False,
            quant_config=None,
            tp_size=self.moe_tp_size,
            tp_rank=dist.get_rank(self.moe_tp_group) if self.moe_tp_size > 1 else 0,
            ep_size=self.moe_ep_size,
            ep_rank=dist.get_rank(self.moe_tp_group) if self.moe_ep_size > 1 else 0,
            prefix="HunyuanMoE.experts",
        )
        self.reset_weight = True
    else:
        self.experts = nn.ModuleList(
            [HunyuanMLP(config, layer_idx=layer_idx, is_shared_mlp=False, is_moe=True) for _ in range(self.num_experts)]
        )

    # For FlashInfer
    self.moe_weight = None
    self.moe_weight_2 = None
    self._weights_initialized = False


def moe_forward(self, hidden_states):
    torch.npu.set_device(hidden_states.device.index)
    bsz, seq_len, hidden_size = hidden_states.shape
    shared_mlp_input = hidden_states
    reshaped_input = hidden_states.reshape(-1, hidden_size) # [bsz*seq_len, hidden_size]

    if self._moe_impl == "npu_grouped_matmul":
        if self.reset_weight:
            self.reset_weight = False
            self.experts.quant_method.process_weights_after_loading(layer=self.experts)
        topk_weight, expert_index = self.gate(hidden_states, topk_impl='easy')
        routing_args = {
            "expert_idx": expert_index.to(torch.int32),
            "active_num": bsz * seq_len * self.moe_topk,
            "expert_num": self.num_experts,
            "expert_tokens_num_type": 1,  # 0: cumsum mode(not supported now); 1: count mode
            "expert_tokens_num_flag": True,
            "active_expert_range": [0, self.num_experts],
            "quant_mode": -1
        }

        expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = torch_npu.npu_moe_init_routing_v2(
            reshaped_input, **routing_args
        )

        moe_args = {"group_list_type": 1}

        hidden_states_ordered_by_experts = self.experts(expanded_x, tokens_per_expert, **moe_args)

        hidden_states = torch_npu.npu_moe_finalize_routing(
            hidden_states_ordered_by_experts,
            skip1=None, skip2=None,
            bias=None,
            scales=topk_weight.to(hidden_states_ordered_by_experts.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None, drop_pad_mode=2
        )
        if self.moe_tp_size > 1:
            dist_moe = dist.all_reduce(hidden_states, group=self.moe_tp_group, async_op=True)
    elif self._moe_impl == "flashinfer":
        # Get expert weights
        if not self._weights_initialized:
            self._initialize_weights_on_device(hidden_states.device)
        topk_weight, topk_index = self.gate(hidden_states, topk_impl='easy')

        combined_output = torch.zeros_like(reshaped_input)
        _ = flashinfer.fused_moe.cutlass_fused_moe(     # noqa
            reshaped_input.contiguous(),
            topk_index.to(torch.int).contiguous(),
            topk_weight.to(torch.float).contiguous(),
            self.moe_weight,
            self.moe_weight_2,
            torch.bfloat16,
            output=combined_output,
            quant_scales=None,
        )
        combined_output = combined_output.reshape(bsz, seq_len, hidden_size)
    else:
        # Original implementation - fallback for compatibility
        l_moe, combine_weights, dispatch_mask, exp_counts = self.gate(hidden_states, topk_impl='default')
        dispatched_input = torch.einsum("sec,sm->ecm", dispatch_mask.type_as(hidden_states), reshaped_input)
        chunks = dispatched_input.chunk(self.num_experts, dim=0)
        expert_outputs = []
        for chunk, expert in zip(chunks, self.experts):
            expert_outputs.append(expert(chunk))

        expert_output = torch.cat(expert_outputs, dim=0)
        combined_output = torch.einsum("sec,ecm->sm", combine_weights.type_as(hidden_states), expert_output)
        combined_output = combined_output.reshape(bsz, seq_len, hidden_size)

    if self.config.use_mixed_mlp_moe:
        hidden_states_mlp = self.shared_mlp(shared_mlp_input)

    if self._moe_impl == "npu_grouped_matmul":
        if self.moe_tp_size > 1:
            dist_moe.wait()
        combined_output = hidden_states.view(bsz, seq_len, hidden_size)

    if self.config.use_mixed_mlp_moe:
        output = hidden_states_mlp + combined_output    # noqa
    else:
        output = combined_output

    return output


def spda_attention_init(self, config: HunyuanImage3Config, layer_idx: int, **kwargs):
    super(HunyuanImage3SDPAAttention, self).__init__()
    self.config = config
    self.layer_idx = layer_idx
    self.attention_type = 'self'
    self.attention_dropout = config.attention_dropout
    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    # self.head_dim = self.hidden_size // self.num_heads
    self.head_dim = config.attention_head_dim
    self.num_key_value_heads = config.num_key_value_heads if config.num_key_value_heads else self.num_heads
    self.num_key_value_groups = self.num_heads // self.num_key_value_heads
    self.max_position_embeddings = config.max_position_embeddings
    self.rope_theta = config.rope_theta
    self.is_causal = True
    self.use_qk_norm = config.use_qk_norm
    self.use_rotary_pos_emb = config.use_rotary_pos_emb
    self.hidden_size_q = self.head_dim * self.num_heads
    self.hidden_size_kv = self.head_dim * self.num_key_value_heads

    # define layers
    self.hccl_comm_dict = kwargs.get("hccl_comm_dict")
    self.attn_tp_size = self.hccl_comm_dict.get("attn_tp_size")
    self.attn_tp_group = self.hccl_comm_dict.get("attn_tp_group")
    if self.num_key_value_heads < self.attn_tp_size:
        self.num_q_heads_per_group = self.num_heads // self.attn_tp_size
        self.num_local_kv_heads = 1
        hidden_size_kv = 2 * self.hidden_size_kv // self.num_key_value_heads
    else:
        self.num_q_heads_per_group = self.num_key_value_groups
        self.num_local_kv_heads = self.num_key_value_heads // self.attn_tp_size
        hidden_size_kv = 2 * self.hidden_size_kv // self.attn_tp_size

    self.qkv_proj = nn.Linear(
        self.hidden_size,
        self.hidden_size_q // self.attn_tp_size + hidden_size_kv,
        bias=config.attention_bias
    )
    self.o_proj = nn.Linear(self.hidden_size_q // self.attn_tp_size, self.hidden_size, bias=config.attention_bias)

    if self.use_qk_norm:
        self.query_layernorm = HunyuanRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.key_layernorm = HunyuanRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    if self.use_rotary_pos_emb:
        self._init_rope()


def npu_fas(q, k, v, attn_mask_npu, causal=False):
    b, n, s, d = q.shape
    scale = 1.0 / math.sqrt(d)
    if not causal:
        attn_out = torch_npu.npu_fusion_attention(
            q, k, v,
            head_num=n,
            input_layout="BNSD",
            scale=scale
        )[0]
    else:
        if attn_mask_npu is None:
            attn_mask_npu = torch.triu(torch.ones([2048, 2048]), diagonal=1).bool().to(q.device)
        attn_out = torch_npu.npu_fusion_attention(
            q, k, v, 
            pse=None,
            atten_mask=attn_mask_npu,
            sparse_mode=2,
            head_num=n,
            input_layout="BNSD",
            scale=scale
        )[0]
    return attn_out


class HunyuanImage3NpuFIA(HunyuanImage3SDPAAttention):
    def __init__(self, config: HunyuanImage3Config, layer_idx: int, **kwargs):
        super().__init__(config, layer_idx, **kwargs)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: Optional[bool] = False,
        custom_pos_emb: Optional[Tuple[torch.FloatTensor]] = None,
        attn_mask_npu: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        if output_attentions:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
            )
        bsz, q_len, _ = hidden_states.size()
        qkv_states = self.qkv_proj(hidden_states)
        qkv_states = qkv_states.reshape(bsz,
                                        q_len,
                                        self.num_local_kv_heads,
                                        self.num_q_heads_per_group + 2,
                                        self.head_dim)
        query_states, key_states, value_states = torch.split(qkv_states, [self.num_q_heads_per_group, 1, 1], dim=3)

        query_states = query_states.reshape(bsz,
                                            q_len,
                                            self.num_heads // self.attn_tp_size,
                                            self.head_dim).transpose(1, 2)
        key_states = key_states.reshape(bsz, q_len, self.num_local_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.reshape(bsz, q_len, self.num_local_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_rotary_pos_emb:
            cos, sin = custom_pos_emb
            query_states, key_states = npu_apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

        if self.use_qk_norm:
            query_states = self.query_layernorm(query_states)
            key_states = self.key_layernorm(key_states)

        if past_key_value is not None:
            cache_kwargs = {"cache_position": position_ids}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        q_fa = query_states.contiguous()
        k_fa = key_states.contiguous()
        v_fa = value_states.contiguous()

        b, n, s, d = q_fa.shape

        mode = kwargs.get("mode", "gen_text")

        if mode == "gen_text":
            if attention_mask is None:
                attn_output = npu_fas(q_fa, k_fa, v_fa, attn_mask_npu, causal=False)
            else:
                attn_output = npu_fas(q_fa, k_fa, v_fa, attn_mask_npu, causal=True)
        else:
            gen_timestep_scatter_index: Optional[torch.Tensor] = kwargs.get("gen_timestep_scatter_index", None)
            assert gen_timestep_scatter_index is not None, \
                "When gen_image, `gen_timestep_scatter_index` must be provided."
            timestep_index = gen_timestep_scatter_index[0, 0].item()
            first_step = kwargs.get("first_step", None)
            if first_step is None:
                raise ValueError("When gen_image, `first_step` must be provided.")
            if first_step:
                casual_len = timestep_index + 1
                text_query_states = q_fa[:, :, :casual_len, :]
                text_key_states = k_fa[:, :, :casual_len, :]
                text_value_states = v_fa[:, :, :casual_len, :]
                text_attn_output = npu_fas(
                    text_query_states, text_key_states, text_value_states, attn_mask_npu, causal=True)
                image_query_states = q_fa[:, :, casual_len:, :]
                image_attn_output = npu_fas(image_query_states, k_fa, v_fa, attn_mask_npu, causal=False)
                attn_output = torch.cat((text_attn_output, image_attn_output), dim=2)
            else:
                casual_len = timestep_index + 1
                timestep_query_states = q_fa[:, :, 0:1, :]
                timestep_key_states = k_fa[:, :, :casual_len, :]
                timestep_value_states = v_fa[:, :, :casual_len, :]
                timestep_attn_output = npu_fas(
                    timestep_query_states, timestep_key_states, timestep_value_states, attn_mask_npu, causal=True)
                image_query_states = q_fa[:, :, 1:, :]
                image_attn_output = npu_fas(image_query_states, k_fa, v_fa, attn_mask_npu, causal=False)
                attn_output = torch.cat((timestep_attn_output, image_attn_output), dim=2)
        
        attn_output = rearrange(attn_output, 'b n s d -> b s (n d)')
        attn_output = self.o_proj(attn_output)
        if self.attn_tp_size > 1:
            dist.all_reduce(attn_output, op=torch.distributed.ReduceOp.SUM, group=self.attn_tp_group)

        return attn_output, None, past_key_value


def decoder_layer_init(self, config: HunyuanImage3Config, layer_idx: int, **kwargs):
    super(HunyuanImage3DecoderLayer, self).__init__()
    self.hidden_size = config.hidden_size
    self.layer_idx = layer_idx

    attn_impl = config.attn_implementation
    if attn_impl in Hunyuan_ATTENTION_CLASSES:
        self.self_attn = Hunyuan_ATTENTION_CLASSES[attn_impl](config=config, layer_idx=layer_idx, **kwargs)
    else:
        raise ValueError(f"Unsupported attention implementation: {attn_impl}")

    if ((isinstance(config.num_experts, int) and config.num_experts > 1) or (
            isinstance(config.num_experts, list) and max(
            config.num_experts) > 1)) and layer_idx >= config.moe_layer_num_skipped:
        self.mlp = HunyuanMoE(config, layer_idx=layer_idx, **kwargs)
    else:
        self.mlp = HunyuanMLP(config, layer_idx=layer_idx, is_shared_mlp=False, is_moe=False)
    if config.norm_type == 'hf_rms' or config.norm_type == 'rms':
        self.input_layernorm = HunyuanRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = HunyuanRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    elif config.norm_type == 'fused' or config.norm_type == 'torch_nn':
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
    else:
        assert False, "other norm_type are not supported"


def decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    past_residual: Optional[torch.Tensor] = None,
    custom_pos_emb: Optional[Tuple[torch.FloatTensor]] = None,
    **kwargs,
) -> Tuple[torch.FloatTensor]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*):
            attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
            query_sequence_length, key_sequence_length)` if default attention is used.
        position_ids (`torch.LongTensor`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        custom_pos_emb (`Tuple[torch.FloatTensor]`, *optional*): custom position embedding for rotary
            position embedding
    """
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use "
            "`attention_mask` instead.`"
        )

    hidden_states, residual = self.input_layernorm(hidden_states, past_residual)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        custom_pos_emb=custom_pos_emb,
        **kwargs,
    )
    # Fully Connected
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)

    outputs = (residual, hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def model_init(self, config: HunyuanImage3Config, **kwargs):
    super(HunyuanImage3Model, self).__init__(config)
    self.padding_idx = config.pad_token_id
    self.vocab_size = config.vocab_size
    self.add_classification_head = config.add_classification_head
    self.hccl_comm_dict = kwargs.get("hccl_comm_dict", None)
    self.wte = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
    self.layers = nn.ModuleList(
        [HunyuanImage3DecoderLayer(config, layer_idx, **kwargs) for layer_idx in range(config.num_hidden_layers)]
    )
    if not config.add_classification_head:
        self.ln_f = HunyuanRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    # Initialize weights and apply final processing
    self.post_init()

    self.shared_tensor = None
    self.attn_mask_npu = torch.triu(
        torch.ones([2048, 2048], dtype=torch.bool), diagonal=1).to(torch.device(f"npu:{local_rank}")
    )


def model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    custom_pos_emb: Optional[Tuple[torch.FloatTensor]] = None,
    mode: str = "gen_text",
    first_step: Optional[bool] = None,
    gen_timestep_scatter_index: Optional[torch.Tensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.wte(input_ids)

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    residual = None
    for layer_idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            past_residual=residual,
            output_attentions=output_attentions,
            use_cache=use_cache,
            custom_pos_emb=custom_pos_emb,
            attn_mask_npu=self.attn_mask_npu,
            mode=mode,
            first_step=first_step,
            gen_timestep_scatter_index=gen_timestep_scatter_index,
        )

        residual = layer_outputs[0]
        hidden_states = layer_outputs[1]

        if use_cache:
            next_decoder_cache = layer_outputs[-1]

        if output_attentions:
            all_self_attns += (layer_outputs[2],)

    hidden_states = hidden_states + residual
    if not self.add_classification_head:
        # Do ln_f outside of the model for compatibility with image generation.
        pass
        # hidden_states = self.ln_f(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = next_decoder_cache
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

model_forward.__doc__ = Hunyuan_INPUTS_DOCSTRING + (model_forward.__doc__ or '')


def get_parallel_settings(self):
    world_size = dist.get_world_size()
    self.cfg_parallel_size = 2 if os.environ.get("CFG_PARALLEL") == "1" else 1
    self.attn_tp_size = world_size // self.cfg_parallel_size
    self.attn_dp_size = 1
    self.moe_ep_size = 1
    self.moe_tp_size = world_size // self.cfg_parallel_size
    self.moe_dp_size = 1


def init_parallel_comm_group(self):
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    attn_tp_group = init_comm_group(
        global_rank=global_rank,
        group_num=world_size // self.attn_tp_size,
        world_size=world_size,
        group_stride=1,
        group_name="attn_tp_group"
    )

    if self.moe_tp_size == self.attn_tp_size:
        moe_tp_group = attn_tp_group
    else:
        moe_tp_group = init_comm_group(
            global_rank=global_rank,
            group_num=world_size // self.moe_tp_size,
            world_size=world_size,
            group_stride=1,
            group_name="moe_tp_group"
        )
    
    moe_ep_group, moe_ep_group_name = init_comm_group(
            global_rank=global_rank,
            group_num=world_size // self.moe_ep_size,
            world_size=world_size,
            group_stride=1,
            group_name="moe_ep_group",
            return_name=True
    )

    cfg_parallel_group = init_comm_group(
            global_rank=global_rank,
            group_num=world_size // self.cfg_parallel_size,
            world_size=world_size,
            group_stride=world_size // self.cfg_parallel_size,
            group_name="cfg_parallel_group"
    )

    hccl_comm_dict = {
        "default_pg": get_default_group(),
        "attn_tp_group": attn_tp_group,
        "moe_tp_group": moe_tp_group,
        "moe_ep_group": moe_ep_group,
        "moe_ep_group_name": moe_ep_group_name,
        "attn_tp_size": self.attn_tp_size,
        "moe_tp_size": self.moe_tp_size,
        "moe_ep_size": 1,
        "cfg_parallel_group": cfg_parallel_group,
        "cfg_parallel_size": self.cfg_parallel_size,
    }
    return hccl_comm_dict


def causalmm_init(self, config: HunyuanImage3Config):
    super(HunyuanImage3ForCausalMM, self).__init__(config)
    self.config = config
    self._tkwrapper: Optional[TokenizerWrapper] = None

    # Initialize image preprocessor (for conditional images)
    self.image_processor = HunyuanImage3ImageProcessor(config)

    # vae and gen_image pipeline
    self.vae = AutoencoderKLConv3D.from_config(config.vae)
    self._pipeline = None

    # vit
    self.vision_model = Siglip2VisionTransformer(config.vit)
    self.vision_aligner = LightProjector(config.vit_aligner)

    # image generation related
    self.timestep_emb = TimestepEmbedder(hidden_size=config.hidden_size)
    if config.img_proj_type == "unet":
        self.patch_embed = UNetDown(
            patch_size=config.patch_size,
            emb_channels=config.hidden_size,
            in_channels=config.vae["latent_channels"],
            hidden_channels=config.patch_embed_hidden_dim,
            out_channels=config.hidden_size,
        )
        self.time_embed = TimestepEmbedder(hidden_size=config.hidden_size)

        self.final_layer = UNetUp(
            patch_size=config.patch_size,
            emb_channels=config.hidden_size,
            in_channels=config.hidden_size,
            hidden_channels=config.patch_embed_hidden_dim,
            out_channels=config.vae["latent_channels"],
            out_norm=True,
        )
        self.time_embed_2 = TimestepEmbedder(hidden_size=config.hidden_size)
    else:
        raise ValueError(f"Unknown img_proj_type {config.img_proj_type}")

    self.get_parallel_settings()
    kwargs = {}
    default_pg = get_default_group()
    if default_pg is not None:
        if dist.get_world_size() > 1:
            self.hccl_comm_dict = self.init_parallel_comm_group()
            kwargs.update({"hccl_comm_dict": self.hccl_comm_dict})
    
    # transformer backbone
    self.model = HunyuanImage3Model(config, **kwargs)

    self.pad_id = config.pad_id
    self.vocab_size = config.vocab_size

    # linear head
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    # Initialize weights and apply final processing
    self.post_init()


@staticmethod
def causalmm_get_pos_emb(custom_pos_emb, position_ids):
    cos, sin = custom_pos_emb
    cos = real_batched_index_select(cos, dim=1, idx=position_ids).to(torch.bfloat16)
    sin = real_batched_index_select(sin, dim=1, idx=position_ids).to(torch.bfloat16)
    return cos, sin


def causalmm_prepare_model_inputs(
    self,
    prompt=None,
    mode="gen_text",
    system_prompt=None,
    cot_text=None,
    image_size="auto",
    message_list=None,
    device=None,
    max_new_tokens=None,
    **kwargs,
):
    # 1. Sanity check
    self.check_inputs(prompt, message_list)
    device = default(device, self.device)

    # 2. Format inputs
    batch_message_list = message_list
    batch_prompt = prompt
    batch_cot_text = cot_text
    batch_system_prompt = system_prompt
    batch_gen_image_info = None
    batch_cond_image_info = None

    #   -- 2.1 message_list
    if batch_message_list is not None:
        if isinstance(batch_message_list[0], dict):
            batch_message_list = [batch_message_list]
        batch_size = len(batch_message_list)

        batch_gen_image_info = [
            [message['content'] for message in message_list_ if message['type'] == 'gen_image']
            for message_list_ in batch_message_list
        ]
        # At most one gen_image is allowed for each message_list
        batch_gen_image_info = [info[-1] if len(info) > 0 else None for info in batch_gen_image_info]
        # Multiple cond images are allowed.
        batch_cond_image_info = [
            [message['content'] for message in message_list_ if message['type'] == 'joint_image']
            for message_list_ in batch_message_list
        ]

    #   -- 2.2 Prompt, cot text, system prompt
    else:
        if isinstance(batch_prompt, str):
            batch_prompt = [batch_prompt]
        batch_size = len(batch_prompt)

        if batch_cot_text is not None:
            if isinstance(batch_cot_text, str):
                batch_cot_text = [batch_cot_text]
            else:
                assert isinstance(batch_cot_text, list) and len(batch_cot_text) == batch_size, \
                    "`cot_text` should be a string or a list of strings with the same length as `prompt`."

        if batch_system_prompt is not None:
            if isinstance(batch_system_prompt, str):
                batch_system_prompt = [batch_system_prompt]
            else:
                assert isinstance(batch_system_prompt, list) and len(batch_system_prompt) == batch_size, \
                    "`system_prompts` should be a string or a list of strings with the same length as `prompt`."

        if mode == "gen_image":
            batch_gen_image_info = [self.image_processor.build_image_info(image_size) for _ in range(batch_size)]

    #   -- 2.3 seed
    seeds = self.prepare_seed(seed=kwargs.get('seed'), batch_size=batch_size)
    generator = [torch.Generator(self.device).manual_seed(seed) for seed in seeds]

    # 3. apply chat template
    cfg_factor = {"gen_text": 1, "gen_image": 2}
    bot_task = kwargs.pop("bot_task", "auto")
    # If `drop_think` enabled, always drop <think> parts in the context.
    drop_think = kwargs.get('drop_think', self.generation_config.drop_think)
    # Apply batched prompt or batched message_list to build input sequence with associated info.
    out = self._tkwrapper.apply_chat_template(
        batch_prompt=batch_prompt,
        batch_message_list=batch_message_list,
        mode=mode,
        batch_gen_image_info=batch_gen_image_info,
        batch_cond_image_info=batch_cond_image_info,
        batch_system_prompt=batch_system_prompt,
        batch_cot_text=batch_cot_text,
        max_length=kwargs.get('max_length'),
        bot_task=bot_task,
        image_base_size=self.config.image_base_size,
        sequence_template=self.generation_config.sequence_template,
        cfg_factor=cfg_factor[mode],
        drop_think=drop_think,
    )
    output, sections = out['output'], out['sections']

    if self.hccl_comm_dict["cfg_parallel_size"] > 1:
        # Some data still has batchsize 2 in CFG parallel and needs to be selected according to CFG parallel RANK
        idx_select = torch.distributed.get_rank(self.hccl_comm_dict.get("cfg_parallel_group"))
        cfg_split_model_inputs(output, idx_select)
        sections = sections[idx_select:idx_select + 1]

    # 4. Encode conditional images
    if batch_cond_image_info is not None and len(batch_cond_image_info[0]) > 0:
        cond_vae_images, cond_timestep, cond_vit_images = self._encode_cond_image(
            batch_cond_image_info, cfg_factor[mode]
        )
        # Pack vit kwargs. Siglip2-so requires spatial_shapes and attention_mask for inference.
        vit_kwargs = {"spatial_shapes": [], "attention_mask": []}
        for cond_image_info in batch_cond_image_info:
            vit_kwargs["spatial_shapes"].append(
                torch.stack([item.vision_encoder_kwargs["spatial_shapes"] for item in cond_image_info]))
            vit_kwargs["attention_mask"].append(
                torch.stack([item.vision_encoder_kwargs["pixel_attention_mask"] for item in cond_image_info]))
        if cfg_factor[mode] > 1:
            vit_kwargs["spatial_shapes"] = vit_kwargs["spatial_shapes"] * cfg_factor[mode]
            vit_kwargs["attention_mask"] = vit_kwargs["attention_mask"] * cfg_factor[mode]
    else:
        cond_vae_images, cond_timestep, cond_vit_images = None, None, None
        vit_kwargs = None

    # 5. Build position embeddings
    rope_image_info = self.build_batch_rope_image_info(output, sections)
    if mode == "gen_text":
        seq_len = self.generation_config.max_length
    else:
        seq_len = output.tokens.shape[1]
    cos, sin = build_batch_2d_rope(
        image_infos=rope_image_info,
        seq_len=seq_len,
        n_elem=self.config.attention_head_dim,
        device=device,
        base=self.config.rope_theta,
    )

    # 6. Build kv cache
    if bot_task == "img_ratio":
        max_new_tokens = 1
    if mode == "gen_image":
        # Image generation will not extend sequence length, using token length as max_cache_len is enough.
        max_cache_len = output.tokens.shape[1]
    else:
        max_cache_len = output.tokens.shape[1] + default(max_new_tokens, self.generation_config.max_length)
    cache = HunyuanStaticCache(
        config=self.config,
        batch_size=batch_size * cfg_factor[mode],
        max_cache_len=max_cache_len,
        dtype=torch.bfloat16,
        dynamic=mode == "gen_text",
    )

    # 7. Build position ids
    batch_input_pos = torch.arange(
        0, output.tokens.shape[1], dtype=torch.long, device=device)[None].expand(
        batch_size * cfg_factor[mode], -1)  # use expand to share indices to save memory

    # 8. Build model input kwargs
    tkw = self._tkwrapper
    if image_size == "auto":
        extra_auto_stops = [tkw.special_token_map[f"<img_ratio_{i}>"] for i in range(33)]
    else:
        extra_auto_stops = [tkw.boi_token_id]
    stop_token_id = dict(
        auto=[tkw.eos_token_id] + extra_auto_stops,
        image=[tkw.eos_token_id],
        recaption=[tkw.end_recaption_token_id, tkw.end_answer_token_id, tkw.eos_token_id],
        think=[tkw.end_recaption_token_id, tkw.end_answer_token_id, tkw.eos_token_id],
        img_ratio=extra_auto_stops,
    )
    model_input_kwargs = dict(
        input_ids=output.tokens.to(device),
        position_ids=batch_input_pos,
        past_key_values=cache,
        custom_pos_emb=(cos, sin),
        mode=mode,
        image_mask=to_device(output.gen_image_mask, device),
        gen_timestep_scatter_index=to_device(output.gen_timestep_scatter_index, device),
        cond_vae_images=to_device(cond_vae_images, device),
        cond_timestep=to_device(cond_timestep, device),
        cond_vae_image_mask=to_device(output.cond_vae_image_mask, device),
        cond_vit_images=to_device(cond_vit_images, device),
        cond_vit_image_mask=to_device(output.cond_vit_image_mask, device),
        vit_kwargs={
            k: to_device(v, self.device) for k, v in vit_kwargs.items()
        } if vit_kwargs is not None else None,
        cond_timestep_scatter_index=to_device(output.cond_timestep_scatter_index, device),
        # for inner usage
        tokenizer_output=output,
        batch_gen_image_info=batch_gen_image_info,
        generator=generator,
        # generation config
        eos_token_id=stop_token_id[bot_task],
        max_new_tokens=max_new_tokens,
    )

    if self.hccl_comm_dict["cfg_parallel_size"] > 1:
        # There ard some other data needs to be selected for CFG parallel
        idx_select = torch.distributed.get_rank(self.hccl_comm_dict.get("cfg_parallel_group"))
        model_input_kwargs["position_ids"] = model_input_kwargs["position_ids"][idx_select:idx_select + 1]
        model_input_kwargs["eos_token_id"] = model_input_kwargs["eos_token_id"][idx_select:idx_select + 1]

    return model_input_kwargs


Hunyuan_ATTENTION_CLASSES["npu"] = HunyuanImage3NpuFIA
HunyuanRMSNorm.forward = rms_norm_forward
HunyuanMoE.__init__ = moe_init
HunyuanMoE.forward = moe_forward
HunyuanImage3SDPAAttention.__init__ = spda_attention_init
HunyuanImage3DecoderLayer.__init__ = decoder_layer_init
HunyuanImage3DecoderLayer.forward = decoder_layer_forward
HunyuanImage3Model.__init__ = model_init
HunyuanImage3Model.forward = model_forward
HunyuanImage3ForCausalMM.get_parallel_settings = get_parallel_settings
HunyuanImage3ForCausalMM.init_parallel_comm_group = init_parallel_comm_group
HunyuanImage3ForCausalMM.__init__ = causalmm_init
HunyuanImage3ForCausalMM.get_pos_emb = causalmm_get_pos_emb
HunyuanImage3ForCausalMM.prepare_model_inputs = causalmm_prepare_model_inputs