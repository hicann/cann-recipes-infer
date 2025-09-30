# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.53.0/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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

""" PyTorch Qwen3_MOE model."""
import os
import math
import warnings
from typing import List, Optional, Tuple, Union, Iterable

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.distributed.distributed_c10d import _world
import torch_npu
import torchair

import torch.distributed as dist

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from module.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding
    )
from module.fuse_moe_gmm import FusedMoEGMM
from executor.utils import init_comm_group, get_default_group
from executor.model_loader.weight_utils import default_weight_loader
from .configuration_qwen3_moe import Qwen3MoeConfig

logger = logging.get_logger(__name__)
torchair.patch_for_hcom()


class Qwen3MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3MoeRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def ln(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def ln_npu(self, hidden_states):
        result = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
        return result

    def forward(self, hidden_states, *args):
        if len(args) == 0: # only hidden_states exists
            result = self.ln_npu(hidden_states)
            return result
        elif len(args) == 1 and args[0] is None: # residual is None
            result = self.ln_npu(hidden_states)
            residual = hidden_states
            return (result, residual)
        elif len(args) == 1: # residual is not None
            residual = args[0]
            y, _, x = torch_npu.npu_add_rms_norm(residual, hidden_states, self.weight, self.variance_epsilon)
            return (y, x)
        else:
            raise NotImplementedError(
                f"insupportable Qwen3MoeRMSNorm for input_args len as (include hid): {len(args) + 1}"
            )



ALL_LAYERNORM_LAYERS.append(Qwen3MoeRMSNorm)


class Qwen3MoeRotaryEmbedding(nn.Module):
    def __init__(self, config, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.config = config
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def forward(self, x, seq_len, kv_len, max_seq_len=None):
        # x shape is [bs, num_attention_heads, seq_len, head_size]
        if max_seq_len is None:
            self._set_cos_sin_cache(seq_len=kv_len, device=x.device, dtype=x.dtype)
        elif max_seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=max_seq_len, device=x.device, dtype=x.dtype)

        batch_size, _, _ = x.size()
        if seq_len == 1:
            # BD -> BNSD
            cos = torch.index_select(self.cos_cached, dim=0, index=kv_len).unsqueeze(1).unsqueeze(1)
            sin = torch.index_select(self.sin_cached, dim=0, index=kv_len).unsqueeze(1).unsqueeze(1)
        else:
            # SD -> BSND
            cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1, 1)
            sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, 1, 1)

        return (
            cos.to(dtype=x.dtype),
            sin.to(dtype=x.dtype),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# Copied from transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding with Llama->Qwen3Moe
class Qwen3MoeLinearScalingRotaryEmbedding(Qwen3MoeRotaryEmbedding):
    """Qwen3MoeRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(
        self,
        config,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(config, dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# Copied from transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding with Llama->Qwen3Moe
class Qwen3MoeDynamicNTKScalingRotaryEmbedding(Qwen3MoeRotaryEmbedding):
    """
    Qwen3MoeRotaryEmbedding extended with Dynamic NTK scaling.
    Credits to the Reddit users /u/bloc97 and /u/emozilla
    """

    def __init__(
        self,
        config,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(config, dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(
    num_rotations, dim, base=10000, max_position_embeddings=2048
):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


# Find dim range bounds based on rotations
def yarn_find_correction_range(
    low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    low = math.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min_val, max_val, dim):
    if min_val == max_val:
        max_val += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


class Qwen3MoeYarnRotaryEmbedding(Qwen3MoeRotaryEmbedding):

    def __init__(
        self,
        config,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(config, dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (
            self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(
            device=device, dtype=torch.float32
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", (emb.cos() * _mscale).to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", (emb.sin() * _mscale).to(dtype), persistent=False
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim) # BSND->BNSD
    sin = sin[position_ids].unsqueeze(unsqueeze_dim) # BSND->BNSD

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def npu_apply_rotary_pos_emb(q, k, cos, sin, position_ids, layer_idx):
    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    q = q.transpose(1, 2) # BSND

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    k = k.transpose(1, 2) # BSND

    q_embed, k_embed = torch_npu.npu_apply_rotary_pos_emb(q, k, cos, sin, layout="BSH")

    q_embed = q_embed.transpose(1, 2)
    k_embed = k_embed.transpose(1, 2)
    return q_embed, k_embed


class MoEGate(nn.Module):
    def __init__(self, config, runner_settings):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.batch_size = runner_settings.get("data_config").get("batch_size", 1)
        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.num_experts, self.gating_dim))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        pass

    def one_hot(self, tensor):
        index = torch.arange(0, self.n_group, dtype=tensor.dtype, device=tensor.device)
        return (
            tensor.view([*tensor.shape, 1]) == index.view([1] * tensor.ndim + [self.n_group])
        ).to(torch.float32)

    def forward_group_limited_greedy(self, logits):
        bs_seq, h = logits.shape
        scores = logits.softmax(dim=-1, dtype=torch.float32)
        group_scores = (
            scores.view(bs_seq, self.n_group, -1).max(dim=-1).values
        ) # [n, n_group]

        group_idx = torch.topk(
            group_scores, k=self.topk_group, dim=-1
        )[1]
        group_mask = self.one_hot(group_idx)
        group_mask = torch.sum(group_mask, dim=-1)
        score_mask = (
            group_mask.unsqueeze(-1).expand(
                bs_seq, self.n_group, self.num_experts // self.n_group
            ).reshape(bs_seq, -1)
        )
        tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)
        topk_weight, topk_idx = torch.topk(
            tmp_scores, k=self.top_k, dim=-1
        )
        row_idx = None
        return topk_idx, topk_weight, row_idx

    def forward_greedy(self, logits):
        topk_weight, topk_idx, row_idx = torch_npu.npu_moe_gating_top_k_softmax(logits, None, k=self.top_k)
        return topk_idx, topk_weight, row_idx

    def forward(self, hidden_states):
        bsz, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        logits = F.linear(
            hidden_states, self.weight, None
        )
        topk_idx, topk_weight, row_idx = self.forward_greedy(logits)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        aux_loss = None
        return topk_idx, topk_weight, aux_loss, row_idx


class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, loss):
        if loss.numel() != 1:
            raise ValueError(f"Expected loss to be a single value, but got {loss.numel()}")
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config, runner_settings, prefix: str = "", **kwargs):

        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_size
        self.batch_size_decode = runner_settings.get("data_config").get("batch_size", 1)
        self.local_rank = int(os.getenv("LOCAL_RANK", "1"))
        self.input_len = runner_settings.get("data_config").get("input_max_len", 1024)
        self.batch_size_prefill = 1
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hccl_comm_dict = kwargs.get("hccl_comm_dict", None)
        self.max_position_embeddings = runner_settings.get("data_config").get("max_position_embeddings", 4096)
        self.moe_tp_size = runner_settings.get("parallel_config").get("moe_tp_size")
        self.moe_ep_size = runner_settings.get("parallel_config").get("moe_ep_size")
        self.moe_intermediate_size = config.moe_intermediate_size
        self.intermediate_size_per_rank = self.moe_intermediate_size // self.moe_tp_size
        self.experts_per_rank = config.num_experts // self.moe_ep_size
        self.perfect_eplb = runner_settings.get("model_config").get("perfect_eplb", False)
        self.experts = FusedMoEGMM(
            num_experts=self.num_experts,
            hidden_size=self.hidden_dim,
            intermediate_size=self.moe_intermediate_size,
            bias=False,
            quant_config=None,
            tp_size=self.moe_tp_size,
            tp_rank=dist.get_rank(self.hccl_comm_dict["moe_tp_group"]) if self.moe_tp_size > 1 else 0,
            ep_size=self.moe_ep_size,
            ep_rank=dist.get_rank(self.hccl_comm_dict["moe_ep_group"]) if self.moe_ep_size > 1 else 0,
            prefix=f"{prefix}.experts",
        )
        self.gate = MoEGate(config, runner_settings)
        self.row_idx_decode_len = self.batch_size_decode * self.top_k
        self.row_idx_decode = torch.arange(
            0, self.row_idx_decode_len,
            dtype=torch.int32).view(self.top_k, -1).permute(1, 0).contiguous().npu()

    def set_mc2_kwargs(self):
        global_rank = dist.get_rank()
        moe_ep_group_name = self.hccl_comm_dict.get("moe_ep_group_name", None)
        self.dispatch_kwargs = {
                "x_active_mask": None,
                "moe_expert_num": self.num_experts,
                "global_bs": 0,
                "scales": None,
                "group_ep": moe_ep_group_name,
                "ep_world_size": self.moe_ep_size,
                "ep_rank_id": global_rank // self.moe_tp_size,
                "group_tp": moe_ep_group_name,
                "tp_world_size": self.moe_tp_size,
                "tp_rank_id": global_rank % self.moe_tp_size,
                "expert_shard_type": 0,
                "shared_expert_num": 0,
                "shared_expert_rank_num": 0,
                "quant_mode": 0
            }
        self.combine_kwargs = {
                "x_active_mask": None,
                "moe_expert_num": self.num_experts,
                "global_bs": 0,
                "group_ep": moe_ep_group_name,
                "ep_world_size": self.moe_ep_size,
                "ep_rank_id": global_rank // self.moe_tp_size,
                "group_tp": moe_ep_group_name,
                "tp_world_size": self.moe_tp_size,
                "tp_rank_id": global_rank % self.moe_tp_size,
                "expert_shard_type": 0,
                "shared_expert_num": 0,
                "shared_expert_rank_num": 0,
                "comm_quant_mode": 0
            }

    def forward(self, hidden_states, is_prefill=False, cur_topk_list=None):
        topk_idx, topk_weight, _, row_idx = self.gate(hidden_states)
        if self.perfect_eplb:
            topk_idx = cur_topk_list
        topk_idx = topk_idx.to(torch.int32)
        if self.moe_tp_size > 1:
            # MoE TP scene
            return self.moe_infer_tp(hidden_states, topk_idx, topk_weight, row_idx)
        else:
            # MoE EP scene
            if is_prefill:
                return self.moe_infer_double_routing(hidden_states, topk_idx, topk_weight)
            else:
                return self.moe_infer_dispatch_combine(hidden_states, topk_idx, topk_weight)


    def moe_infer_tp(self, hidden_states, topk_idx, topk_weight, row_idx):
        batch_size = hidden_states.shape[0]

        hidden_states_ordered_by_experts, routing_weights, expanded_row_idx, expert_idx \
            = self.moe_infer_fusion(hidden_states, topk_idx, topk_weight, row_idx)
        hidden_states = torch_npu.npu_moe_finalize_routing(
            hidden_states_ordered_by_experts,
            skip1=None, skip2=None,
            bias=None,
            scales=routing_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=expert_idx
        )
        y = hidden_states.view(batch_size, -1, self.hidden_dim)
        return y

    def moe_infer_fusion(self, x, topk_ids, topk_weight, row_idx):
        batch_size, sequence_length, h = x.shape
        hidden_states = x.view(-1, h)

        routing_weights = topk_weight.to(x.dtype)
        expert_idx = topk_ids.int()
        if row_idx is None:
            if sequence_length == 1:
                row_idx = self.row_idx_decode
            else:
                row_idx_prefill_len = self.batch_size_prefill * sequence_length * self.top_k
                row_idx_prefill = torch.arange(
                    0, row_idx_prefill_len, dtype=torch.int32,
                    device=topk_weight.device).view(self.top_k, -1).permute(1, 0).contiguous()
                row_idx = row_idx_prefill

        active_num = batch_size * sequence_length
        expanded_x, expanded_row_idx, expanded_expert_idx = torch_npu.npu_moe_init_routing(
            hidden_states,
            row_idx=row_idx,
            expert_idx=expert_idx,
            active_num=batch_size * sequence_length
        )
        expert_tokens = torch_npu.npu_moe_compute_expert_tokens(expanded_expert_idx, self.num_experts)
        expert_tokens = expert_tokens.to(torch.int64)

        hidden_states_ordered_by_experts = self.experts(expanded_x, expert_tokens)
        dist.all_reduce(hidden_states_ordered_by_experts, group=self.hccl_comm_dict.get("moe_tp_group", None))
        return hidden_states_ordered_by_experts, routing_weights, expanded_row_idx, expert_idx

    def dispatch_double_routing(self, tokens_per_expert, expanded_x):
        moe_ep_group = self.hccl_comm_dict.get("moe_ep_group", None)
        tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
        # (total_experts,)->(total_ranks*n_routed_experts_per_rank)
        dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert, group=moe_ep_group)
        # combine tensors, do reduceSum and D2H togather
        combine_tokens = torch.stack([tokens_per_expert_group, tokens_per_expert], dim=0)
        # view: EP, E // EP
        # sum: EP, 每个rank
        combine_tokens = combine_tokens.view(2, self.moe_ep_size, -1).sum(2)
        all_tokens = combine_tokens[0].sum()
        combine_tokens_cpu = combine_tokens.cpu().tolist()
        input_splits = combine_tokens_cpu[1]
        output_splits = combine_tokens_cpu[0]
        gathered_tokens = expanded_x.new_empty(all_tokens.item(), expanded_x.shape[1])
        dist.all_to_all_single(gathered_tokens, expanded_x, output_splits, input_splits, group=moe_ep_group)
        return tokens_per_expert_group, gathered_tokens, input_splits, output_splits

    def moe_infer_double_routing(self, hidden_states, topk_ids, topk_weight):
        batch_size, sequence_length, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        bs_qlen = hidden_states.shape[0]
        expanded_x, expanded_row_idx, tokens_per_expert, _ = torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            expert_idx=topk_ids,
            active_num=topk_ids.shape[0] * topk_ids.shape[1],
            scale=None,  # non-quant
            expert_num=self.num_experts,
            expert_tokens_num_type=1,  # 0: cumsum mode(not supported now); 1: count mode
            expert_tokens_num_flag=True, active_expert_range=[0, self.num_experts],
            quant_mode=-1  # -1: non-quant; 1: dynamic quant; 0: static quant(not supported now)
        )
        moe_ep_group = self.hccl_comm_dict.get("moe_ep_group", None)
        tokens_per_expert_group, gathered_tokens, input_splits, output_splits =\
            self.dispatch_double_routing(tokens_per_expert, expanded_x)

        # reroute
        hidden_states_ordered_by_experts, _, gathered_ids_unsort, tokens_per_local_expert = \
            torch_npu.npu_moe_re_routing(gathered_tokens, tokens_per_expert_group.view(self.moe_ep_size, -1))

        # compute experts
        gmm_args = {
            "x": hidden_states_ordered_by_experts,
            "expert_tokens": tokens_per_local_expert,
            "group_list_type": 1,
        }
        hidden_states_ordered_by_experts = self.experts(**gmm_args)
        # finalize-rerouting
        new_x = torch.index_select(hidden_states_ordered_by_experts, 0, gathered_ids_unsort.float().argsort().int())
        gathered_tokens = new_x.new_empty(*expanded_x.shape)
        dist.all_to_all_single(gathered_tokens, new_x, input_splits, output_splits, group=moe_ep_group)

        # finalize-routing
        hidden_states = torch_npu.npu_moe_finalize_routing(
            gathered_tokens, skip1=None, skip2=None, bias=None,
            scales=topk_weight.to(gathered_tokens.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None, drop_pad_mode=2
        )
        hidden_states = hidden_states.view(bs_qlen, self.hidden_dim)
        return hidden_states.view(batch_size, -1, h)

    def moe_infer_dispatch_combine(self, x, topk_ids, topk_weight):
        """
        support ep for decode stage
        """
        batch_size, sequence_length, h = x.shape
        hidden_states = x.view(-1, h)
        self.set_mc2_kwargs()

        # moe dispatch
        dispatch_args = {
            "x": hidden_states,
            "expert_ids": topk_ids, # [n*topk]
            **self.dispatch_kwargs
        }
        output = torch_npu.npu_moe_distribute_dispatch_v2(**dispatch_args)
        expand_x, dynamic_scale, expand_idx, expert_token_num, ep_recv_counts, tp_recv_counts = output[:6]

        # compute experts
        gmm_args = {
            "x": expand_x,
            "expert_tokens": expert_token_num,
            "group_list_type": 1,
        }

        hidden_states_ordered_by_experts = self.experts(**gmm_args)

        # moe combine
        combine_args = {
            "expand_x": hidden_states_ordered_by_experts,
            "expert_ids": topk_ids,
            "assist_info_for_combine": expand_idx,
            "expert_scales": topk_weight.to(torch.float32), # [n*topk]
            "ep_send_counts": ep_recv_counts,
            "tp_send_counts": tp_recv_counts,
            **self.combine_kwargs
        }
        hidden_states = torch_npu.npu_moe_distribute_combine_v2(**combine_args)

        hidden_states = hidden_states.view(batch_size, sequence_length, self.hidden_dim)
        return hidden_states


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _init_rope(self):
    if self.config.rope_scaling is None:
        self.rotary_emb = Qwen3MoeRotaryEmbedding(
            self.config,
            self.config.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.config.rope_theta,
        )
    else:
        scaling_type = self.config.rope_scaling["type"]
        scaling_factor = self.config.rope_scaling["factor"]
        if scaling_type == "linear":
            self.rotary_emb = Qwen3MoeLinearScalingRotaryEmbedding(
                self.config,
                self.config.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=self.config.rope_theta,
            )
        elif scaling_type == "dynamic":
            self.rotary_emb = Qwen3MoeDynamicNTKScalingRotaryEmbedding(
                self.config,
                self.config.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=self.config.rope_theta,
            )
        elif scaling_type == "yarn":
            rope_keys = [
                    "original_max_position_embeddings",
                    "beta_fast",
                    "beta_slow",
                    "mscale",
                    "mscale_all_dim",
                ]
            kwargs = {}
            for key in rope_keys:
                if key in self.config.rope_scaling:
                    kwargs[key] = self.config.rope_scaling[key]
            self.rotary_emb = Qwen3MoeYarnRotaryEmbedding(
                self.config,
                self.config.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=self.config.rope_theta,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")


class Qwen3MoeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, runner_settings, layer_idx: Optional[int] = None, prefix: str = "", **kwargs):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attn_tp_size = runner_settings.get("parallel_config").get("attn_tp_size")
        self.moe_tp_size = runner_settings.get("parallel_config").get("moe_tp_size")
        self.moe_ep_size = runner_settings.get("parallel_config").get("moe_ep_size")

        self.num_heads = config.num_attention_heads
        self.num_heads_per_rank = self.num_heads // self.attn_tp_size
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_heads_per_rank = max(self.num_key_value_heads // self.attn_tp_size, 1)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attn_intermediate_size = self.head_dim * self.num_heads
        self.attn_intermediate_size_per_rank = self.attn_intermediate_size // self.attn_tp_size
        self.hccl_comm_dict = kwargs.get("hccl_comm_dict", None)
        self.merged_qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_key_value_heads,
            bias=False,
            skip_bias_add=False,
            tp_size=self.attn_tp_size,
            tp_rank=dist.get_rank() if self.attn_tp_size > 1 else 0,
            quant_config=None,
            prefix=f"{prefix}.merged_qkv_proj",
            return_bias=False
        )
        self.q_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.batch_size = runner_settings.get("data_config").get("batch_size", 1)
        self.input_len = runner_settings.get("data_config").get("input_max_len", 1024)

        self.o_proj = RowParallelLinear(self.attn_intermediate_size,
                                        config.hidden_size,
                                        tp_size=self.attn_tp_size,
                                        tp_rank=dist.get_rank(self.hccl_comm_dict["attn_tp_group"])
                                        if self.attn_tp_size > 1 else 0,
                                        bias=False,
                                        input_is_parallel=True,
                                        prefix=f"{prefix}.o_proj")
        self.scale_fa = 1 / (self.head_dim ** 0.5)

    def exec_qkv(
        self,
        qkv: torch.Tensor,
        kv_len: torch.IntTensor = None,
        actual_seq_lengths_kv: list = None,
        cos_sin: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        **kwargs,
    ):
        bsz, q_len, _ = qkv.size()

        query_states, key_states, value_states = qkv.split((self.num_heads_per_rank * self.head_dim, \
                                                            self.num_key_value_heads_per_rank * self.head_dim, \
                                                            self.num_key_value_heads_per_rank * self.head_dim), dim=2)

        query_shape = (bsz, q_len, self.num_heads_per_rank, self.head_dim)
        key_value_shape = (bsz, q_len, self.num_key_value_heads_per_rank, self.head_dim)

        query_states = self.q_norm(query_states.view(query_shape).contiguous())
        key_states = self.k_norm(key_states.view(key_value_shape).contiguous())

        cos, sin = cos_sin
        query_states, key_states = torch_npu.npu_apply_rotary_pos_emb(query_states, key_states, cos, sin, layout='BSH')
        query_states = query_states.view(bsz, q_len, -1)
        key_states = key_states.view(bsz, q_len, -1)

        past_key, past_value = None, None
        if past_key_value is not None:
            past_key = past_key_value[self.layer_idx][0]
            past_value = past_key_value[self.layer_idx][1]
            torch_npu.scatter_update_(past_key, kv_len, key_states, -2)
            torch_npu.scatter_update_(past_value, kv_len, value_states, -2)

        if q_len == 1:
            past_key_states, past_value_states = past_key_value[self.layer_idx]

            attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                query_states, past_key_states, past_value_states,
                num_heads=self.num_heads_per_rank,
                num_key_value_heads=self.num_key_value_heads_per_rank,
                input_layout="BSH",
                atten_mask=attention_mask,
                scale=self.scale_fa,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                antiquant_mode=0, antiquant_scale=None
            )
        else:
            attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                query_states, key_states, value_states,
                num_heads=self.num_heads_per_rank,
                num_key_value_heads=self.num_key_value_heads_per_rank,
                input_layout="BSH",
                atten_mask=attention_mask,
                sparse_mode=2,
                scale=self.scale_fa,
                next_tokens=0
            )
        attn_output = attn_output.reshape(bsz, q_len, self.attn_intermediate_size_per_rank)
        attn_output = self.o_proj(attn_output)

        if self.attn_tp_size > 1:
            # attention_TP + moe_TP, 当前只支持纯TP或ADP + EP两种场景
            dist.all_reduce(attn_output, group=self.hccl_comm_dict.get("attn_tp_group", None))

        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_len: torch.IntTensor = None,
        actual_seq_lengths_kv: list = None,
        cos_sin: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37."
                " Please make sure use `attention_mask` instead."
            )
        bsz, q_len, _ = hidden_states.size()
        qkv = self.merged_qkv_proj(hidden_states)
        output = self.exec_qkv(
            qkv=qkv,
            kv_len=kv_len,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            cos_sin=cos_sin,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value
        )
        return output

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )


ATTENTION_CLASSES = {
    "eager": Qwen3MoeAttention,
}


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, runner_settings, layer_idx: int, prefix: str = "", **kwargs):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.attn_tp_size = runner_settings.get("parallel_config").get("attn_tp_size")
        self.moe_tp_size = runner_settings.get("parallel_config").get("moe_tp_size")
        self.moe_ep_size = runner_settings.get("parallel_config").get("moe_ep_size", 1)

        self.self_attn = Qwen3MoeAttention(
            config=config, runner_settings=runner_settings, layer_idx=layer_idx, prefix=f"{prefix}.self_attn", **kwargs
        )

        self.layer_idx = layer_idx

        self.mlp = (
            Qwen3MoeSparseMoeBlock(config, runner_settings, prefix=f"{prefix}.mlp", **kwargs)
        )
        self.input_layernorm = Qwen3MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3MoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.hccl_comm_dict = kwargs.get("hccl_comm_dict", None)
        self.batch_size = runner_settings.get("data_config").get("batch_size", 1)
        self.input_len = runner_settings.get("data_config").get("input_max_len", 1024)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_len: torch.IntTensor,
        actual_seq_lengths_kv: list,
        cos_sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        is_prefill: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_residual: Optional[torch.Tensor] = None,
        cur_topk_list: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:

        hidden_states, residual = self.input_layernorm(hidden_states, past_residual)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            kv_len=kv_len,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            cos_sin=cos_sin,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states, is_prefill, cur_topk_list=cur_topk_list)
        outputs = (residual, hidden_states)
        return outputs


QWEN3MOE_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen3MoeConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen3Moe Model outputting raw hidden-states without any specific head on top.",
    QWEN3MOE_START_DOCSTRING,
)
class Qwen3MoePreTrainedModel(PreTrainedModel):
    config_class = Qwen3MoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3MoeDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True

    def _init_weights(self, module):
        pass


QWEN3MOE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Qwen3Moe Model outputting raw hidden-states without any specific head on top.",
    QWEN3MOE_START_DOCSTRING,
)
class Qwen3MoeModel(Qwen3MoePreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen3MoeDecoderLayer`]

    Args:
        config: Qwen3MoeConfig
    """
    def __init__(self, config: Qwen3MoeConfig, runner_settings, prefix: str = "", **kwargs):
        super().__init__(config)
        self.config = config
        self.max_position_embeddings = runner_settings.get("data_config").get("max_position_embeddings", 4096)
        self.rank_id = int(os.getenv("LOCAL_RANK", "0"))
        self.embed_tp_size = runner_settings.get("parallel_config").get("embed_tp_size")
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.vocab_size_per_rank = self.vocab_size // self.embed_tp_size
        self.hccl_comm_dict = kwargs.get("hccl_comm_dict", None)
        self.moe_tp_size = runner_settings.get("parallel_config").get("moe_tp_size")
        self.moe_ep_size = runner_settings.get("parallel_config").get("moe_ep_size")
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            self.padding_idx,
            torch.bfloat16,
            tp_size=self.embed_tp_size,
            tp_rank=dist.get_rank(self.hccl_comm_dict["embed_tp_group"]) if self.embed_tp_size > 1 else 0)
        self.layers = nn.ModuleList(
            [
                Qwen3MoeDecoderLayer(config, runner_settings, layer_idx, prefix=f"model.layers.{layer_idx}", **kwargs)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
        _init_rope(self)
        self.hccl_comm_dict = kwargs.get("hccl_comm_dict", None)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(QWEN3MOE_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor,
        kv_len: torch.IntTensor = None,
        actual_seq_lengths_kv: list = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        is_prefill: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cur_topk_list: Optional[torch.Tensor] = None,
    ):

        batch_size, seq_length = input_ids.shape
        past_key_values_length = past_key_values[0][0].size()[-2]

        if position_ids is None:
            device = input_ids.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if self.embed_tp_size > 1:
            new_input_ids = input_ids - self.rank_id * self.vocab_size_per_rank
            mask = (new_input_ids >= 0) & (new_input_ids < self.vocab_size_per_rank) # (bs, qlen)
            new_input_ids_per_rank = new_input_ids * mask
            inputs_embeds = self.embed_tokens(new_input_ids_per_rank) * mask.unsqueeze(-1)
            dist.all_reduce(inputs_embeds, group=self.hccl_comm_dict.get("embed_tp_group", None))
        else:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        cos_sin = self.rotary_emb(hidden_states, seq_length, kv_len, self.max_position_embeddings)
        residual = None

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                kv_len,
                actual_seq_lengths_kv,
                cos_sin=cos_sin,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                past_residual=residual,
                is_prefill=is_prefill,
                cur_topk_list=cur_topk_list
            )
            residual, hidden_states = layer_outputs

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3MoeForCausalLM(Qwen3MoePreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, runner_settings, prefix: str = ""):
        super().__init__(config)
        self.runner_settings = runner_settings
        self.world_size = runner_settings.get("world_size")
        self.num_hidden_layers = config.num_hidden_layers
        self.perfect_eplb = self.runner_settings.get("model_config").get("perfect_eplb", False)
        self.config = config
        self.input_max_len = runner_settings.get("data_config").get("input_max_len", 1024)
        self.lmhead_tp_size = runner_settings.get("parallel_config").get("lmhead_tp_size")
        self.max_position_embeddings = runner_settings.get("data_config").get("max_position_embeddings", 4096)
        self.get_parallel_settings()
        kwargs = {}
        default_pg = get_default_group()
        if default_pg is not None:
            if dist.get_world_size() > 1:
                self.hccl_comm_dict = self.init_parallel_comm_group()
                kwargs.update({"hccl_comm_dict": self.hccl_comm_dict})

        self.model = Qwen3MoeModel(config, runner_settings, prefix, **kwargs)
        self.vocab_size_per_rank = config.vocab_size // self.lmhead_tp_size
        self.experts_per_rank = config.num_experts // self.moe_ep_size
        self.lm_head = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            bias=False,
            tp_size=self.lmhead_tp_size,
            tp_rank=dist.get_rank(self.hccl_comm_dict.get("lmhead_tp_group")) if self.lmhead_tp_size > 1 else 0
            )
        # Initialize weights and apply final processing
        self.post_init()

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_parallel_settings(self):
        self.embed_tp_size = self.runner_settings.get("parallel_config").get("embed_tp_size", 1)
        self.attn_dp_size = self.runner_settings.get("parallel_config").get("attn_dp_size", 1)
        self.attn_tp_size = self.runner_settings.get("parallel_config").get("attn_tp_size", 1)
        self.moe_ep_size = self.runner_settings.get("parallel_config").get("moe_ep_size", 1)
        self.moe_tp_size = self.runner_settings.get("parallel_config").get("moe_tp_size", 1)
        self.lmhead_tp_size = self.runner_settings.get("parallel_config").get("lmhead_tp_size", self.embed_tp_size)
        self.moe_dp_size = self.runner_settings.get("parallel_config").get("moe_dp_size", 1)
        self.embed_dp_size = self.runner_settings.get("parallel_config").get("embed_dp_size", 1)

    def init_parallel_comm_group(self):
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()

        attn_tp_group = init_comm_group(
            global_rank=global_rank, group_num=self.attn_dp_size, world_size=world_size,
            group_stride=1, group_name="attn_tp_group")

        if self.embed_tp_size == self.attn_tp_size:
            embed_tp_group = attn_tp_group
        else:
            embed_tp_group = init_comm_group(
                global_rank=global_rank, group_num=self.embed_dp_size, world_size=world_size,
                group_stride=1, group_name="embed_tp_group")

        if self.lmhead_tp_size == self.embed_tp_size:
            lmhead_tp_group = embed_tp_group
        else:
            lmhead_tp_group = init_comm_group(
                global_rank=global_rank, group_num=world_size // self.lmhead_tp_size, world_size=world_size,
                group_stride=1, group_name="lmhead_tp_group")

        if self.moe_tp_size == self.attn_tp_size:
            moe_tp_group = attn_tp_group
        else:
            moe_tp_group = init_comm_group(
                global_rank=global_rank, group_num=self.moe_dp_size, world_size=world_size,
                group_stride=1, group_name="moe_tp_group")

        moe_ep_group, moe_ep_group_name = init_comm_group(
            global_rank=global_rank, group_num=self.moe_tp_size, world_size=world_size,
            group_stride=self.moe_tp_size, group_name="moe_ep_group", return_name=True)

        hccl_comm_dict = {
                "default_pg": get_default_group(),
                "attn_tp_group": attn_tp_group, "embed_tp_group": embed_tp_group,
                "moe_tp_group": moe_tp_group, "moe_ep_group": moe_ep_group,
                "moe_ep_group_name": moe_ep_group_name,
                "lmhead_tp_group": lmhead_tp_group,
            }
        return hccl_comm_dict

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        kv_len: torch.IntTensor = None,
        actual_seq_lengths_kv: list = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        is_prefill: Optional[bool] = False,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cur_topk_list: Optional[torch.Tensor] = None,
    ):
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            kv_len=kv_len,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            is_prefill=is_prefill,
            cur_topk_list=cur_topk_list
        )

        hidden_states = outputs

        if hidden_states.size()[1] > 1:
            gather_index, _ = torch.max(position_ids, dim=-1)
            gather_index = gather_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, hidden_states.shape[-1])
            hidden_states = torch.gather(hidden_states, 1, gather_index)

        logits = self.lm_head(hidden_states)
        if self.lmhead_tp_size > 1:
            new_logits = [logits.clone().detach() for _ in range(self.lmhead_tp_size)]
            dist.all_gather(new_logits, logits, group=self.hccl_comm_dict.get("lmhead_tp_group", None))
            logits = torch.concat(new_logits, dim=-1)
        logits = logits.float()
        return logits

    def prefill(
        self,
        **kwargs
    ):
        logits = self.forward(
            **kwargs
        )
        return logits

    def decode(
        self,
        **kwargs
    ):
        logits = self.forward(
            **kwargs
        )
        return logits

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        is_prefill=None,
        kv_len=None,
        share_mask_tril=None,
        **kwargs
    ):
        batch_size, seq_len = input_ids.size()
        if past_key_values is None:
            past_key_values = self.init_cache(input_ids)
        if is_prefill:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            attention_mask = share_mask_tril
            kv_len = torch.zeros((position_ids.size()[0]), dtype=torch.int32, device=input_ids.device)
            actual_seq_lengths_kv = None
        else:
            attention_mask = None
            position_ids = kv_len.unsqueeze(1)
            actual_seq_lengths_kv = (kv_len + 1).detach().cpu().numpy().tolist()

        model_inputs = {}
        model_inputs.update(
            {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                "kv_len": kv_len,
                "actual_seq_lengths_kv": actual_seq_lengths_kv,
                "is_prefill": is_prefill
            }
        )
        return model_inputs

    def gen_cur_topk_idx(
        self,
        is_prefill,
        batch_size,
        seq_len
    ):
        if not self.perfect_eplb:
            return None
        global_rank = dist.get_rank()
        if is_prefill:
            tokens_per_rank_prefill = (batch_size * seq_len + self.attn_tp_size - 1) // self.attn_tp_size \
            if self.moe_ep_size != 1 else batch_size * seq_len * self.attn_dp_size
            step_prefill = tokens_per_rank_prefill * self.config.num_experts_per_tok
            cur_topk_list_prefill = [
                (i + global_rank) % self.config.num_experts for i in range(step_prefill)]
            cur_topk_list = torch.Tensor(cur_topk_list_prefill).int().view(tokens_per_rank_prefill, -1).npu()
        else:
            if self.moe_tp_size > 1:
                expanded_tokens = batch_size * self.config.num_experts_per_tok * seq_len
                cur_topk_list_decode = []
                for offset in range(self.moe_ep_size):
                    expert_start = offset * self.experts_per_rank
                    expert_end = expert_start + expanded_tokens
                    cur_topk_list_decode = cur_topk_list_decode + [i for i in range(expert_start, expert_end)]
                cur_topk_list = torch.Tensor(cur_topk_list_decode).int().view(batch_size * seq_len, -1).npu()
            else:
                expanded_tokens = batch_size * self.config.num_experts_per_tok * seq_len
                step_gap = self.config.num_experts // self.moe_ep_size if expanded_tokens < self.config.num_experts \
                     else 1
                cur_topk_list_decode = [
                    ((i + global_rank // step_gap * step_gap) * step_gap + 
                    global_rank % step_gap) % self.config.num_experts for i in range(expanded_tokens)
                ]
                cur_topk_list = torch.Tensor(cur_topk_list_decode).int().view(batch_size * seq_len, -1).npu()
        return cur_topk_list

    # Adapted from vllm.model_executor.models.qwen3moe.Qwen3MoeModel.load_weights
    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("merged_qkv_proj", "q_proj", "q"),
            ("merged_qkv_proj", "k_proj", "k"),
            ("merged_qkv_proj", "v_proj", "v"),
        ]

        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = ()

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = FusedMoEGMM.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts)
        for name, loaded_weight in weights:
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)

                # Skip loading extra parameters for GPTQ/modelopt models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True

                    # Do not modify `name` since the loop may continue here
                    # Instead, create a new variable
                    name_mapped = name.replace(weight_name, param_name)

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name_mapped.endswith(".bias") and name_mapped not in params_dict:
                        continue

                    param = params_dict[name_mapped]
                    # We should ask the weight loader to return success or not
                    # here since otherwise we may skip experts with other
                    # available replicas.
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                loaded_weight,
                                name_mapped,
                                shard_id=shard_id,
                                expert_id=expert_id,
                                )
                    name = name_mapped
                    break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def init_cache(
        self,
        input_ids,
    ):
        batch_size, seq_len = input_ids.size()

        cache_seq_len = self.max_position_embeddings

        past_key_values = ()
        cache_shape = (
                        batch_size,
                        cache_seq_len,
                        self.config.head_dim * max(self.config.num_key_value_heads // self.attn_tp_size, 1)
                      )
        dtype = self.config.torch_dtype

        for _ in range(self.config.num_hidden_layers):
            key_cache = torch.zeros(cache_shape, dtype=dtype, device=input_ids.device)
            value_cache = torch.zeros(cache_shape, dtype=dtype, device=input_ids.device)
            past_key_values += ((key_cache, value_cache),)

        return past_key_values
