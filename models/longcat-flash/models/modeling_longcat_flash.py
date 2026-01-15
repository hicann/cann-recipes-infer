# coding=utf-8
# Adapted from
# https://huggingface.co/meituan-longcat/LongCat-Flash-Chat/blob/main/modeling_longcat_flash.py
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Copyright (c) 2025 Meituan
#
# MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Callable, Iterable, Optional, Tuple, List, Dict, Set
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch_npu
import torchair as tng

from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import auto_docstring, can_return_tuple, logging
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from executor.utils import npu_stream_switch, limit_core_num, npu_wait_tensor, superkernel_scope, npu_prefetch
from module.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding
    )
from module.fuse_moe_gmm import FusedMoEGMM
from executor.model_loader.weight_utils import default_weight_loader
from executor.utils import (
    override, get_init_attn_mask,
    init_comm_group, get_default_group, get_decode_mask)

from .configuration_longcat_flash import LongcatFlashConfig

logger = logging.get_logger(__name__)


@use_kernel_forward_from_hub("RMSNorm")
class LongcatFlashRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LongcatFlashRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def rms_norm(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)  # main diff with Llama

    def forward(self, hidden_states, *args):
        if len(args) == 0: # only hidden_states exists
            result = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
            return result
        elif len(args) == 1 and args[0] is None: # residual is None
            result = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
            residual = hidden_states
            return (result, residual)
        elif len(args) == 1: # residual is not None
            residual = args[0]
            result, _, r = torch_npu.npu_add_rms_norm(residual, hidden_states, self.weight, self.variance_epsilon)
            return (result, r)
        else:
            raise NotImplementedError(
                f"insupportable LongcatFlashRMSNorm for input_args len as (include hid): {len(args) + 1}"
            )

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LongcatFlashRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

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

    def forward(self, x, kv_len, max_seq_len=None, is_prefill=True, attn_tp_size=1):
        # x shape is [bs, num_attention_heads, seq_len, head_size]
        if max_seq_len is None:
            self._set_cos_sin_cache(seq_len=kv_len, device=x.device, dtype=x.dtype)
        elif max_seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=max_seq_len, device=x.device, dtype=x.dtype)

        batch_size, seq_len, _ = x.size()
        if is_prefill:
            # SD -> TND
            cos = []
            sin = []
            for _ in range(batch_size * attn_tp_size):
                cos.append(self.cos_cached[:seq_len].unsqueeze(1))
                sin.append(self.sin_cached[:seq_len].unsqueeze(1))
            cos = torch.cat(cos, dim=0)
            sin = torch.cat(sin, dim=0)
        else:
            # BD -> BNSD
            cos = torch.index_select(self.cos_cached, dim=0, index=kv_len.view(-1)).unsqueeze(1).unsqueeze(1)
            sin = torch.index_select(self.sin_cached, dim=0, index=kv_len.view(-1)).unsqueeze(1).unsqueeze(1)

        return (
            cos.to(dtype=x.dtype),
            sin.to(dtype=x.dtype),
        )


def _init_rope(self):
    self.rotary_emb = LongcatFlashRotaryEmbedding(
        self.config.qk_rope_head_dim,
        max_position_embeddings=self.config.max_position_embeddings,
        base=self.config.rope_theta,
    )


class LongcatFlashMLP(nn.Module):
    def __init__(self, config, runner_settings, prefix, hidden_size=None, intermediate_size=None, **kwargs):
        super().__init__()
        self.runner_settings = runner_settings
        self.mm_quant_mode = (
            config.quant_config.mm_quant_mode
            if config.quant_config is not None
            else "w16a16")
        self.dense_tp_size = self.runner_settings.get("parallel_config").get("dense_tp_size", 1)
        self.kvp_size = self.runner_settings.get("parallel_config").get("kvp_size", 1)
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.ffn_hidden_size if intermediate_size is None else intermediate_size
        self.hccl_comm_dict = kwargs.get("hccl_comm_dict", None)
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[self.intermediate_size] * 2,
            bias=False,
            tp_size=self.dense_tp_size,
            tp_rank=dist.get_rank(self.hccl_comm_dict["dense_tp_group"]) if self.dense_tp_size > 1 else 0,
            quant_config=config.quant_config,
            prefix=f"{prefix}.gate_up_proj"
            )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            config.hidden_size,
            bias=False,
            tp_size=self.dense_tp_size,
            tp_rank=dist.get_rank(self.hccl_comm_dict["dense_tp_group"]) if self.dense_tp_size > 1 else 0,
            quant_config=config.quant_config,
            prefix=f"{prefix}.down_proj")
        dtype_bit = 1 if self.mm_quant_mode == "w8a8" else 2
        self.up_gate_prefetch_size = self.hidden_size * self.intermediate_size * 2 * dtype_bit // self.dense_tp_size
        self.down_prefetch_size = self.hidden_size * self.intermediate_size * dtype_bit // self.dense_tp_size
        if self.mm_quant_mode == "w8a8":
            self.mlp_forward = self.forward_w8a8
        else:
            self.mlp_forward = self.forward_normal

    def forward(self, x, enable_prefetch=False, o_proj=None):
        if o_proj is not None:
            npu_prefetch(enable_prefetch, self.gate_up_proj.weight.data, o_proj, \
                         self.up_gate_prefetch_size, 0)
        # dense tp
        if self.dense_tp_size > 1 and self.kvp_size == 1:
            bsz, q_len, _ = x.size()
            x_output = torch.empty([bsz * q_len * self.dense_tp_size, self.hidden_size], \
                                   dtype=x.dtype, device="npu")
            dist.all_gather_into_tensor(x_output, x, group=self.hccl_comm_dict.get("dense_tp_group", None))
            x = x_output.view(-1, q_len, self.hidden_size)

        down_proj, dsq = self.mlp_forward(x, enable_prefetch)

        if self.dense_tp_size > 1 and self.kvp_size == 1:
            mlp_res = down_proj.new_empty(bsz, q_len, down_proj.shape[-1])
            dist.reduce_scatter_tensor(mlp_res, down_proj, group=self.hccl_comm_dict.get("dense_tp_group", None))
        elif self.dense_tp_size > 1 and self.kvp_size > 1:
            dist.all_reduce(down_proj, group=self.hccl_comm_dict.get("dense_tp_group", None))
            mlp_res = down_proj
        else:
            mlp_res = down_proj

        return mlp_res, down_proj, dsq

    def forward_normal(self, x, enable_prefetch):
        merged_x = self.gate_up_proj(x)
        intermediate_hidden_states = torch_npu.npu_swiglu(merged_x)
        return self.down_proj(intermediate_hidden_states), merged_x

    def forward_w8a8(self, x, enable_prefetch):
        npu_prefetch(enable_prefetch, self.down_proj.weight.data, x, self.down_prefetch_size, 0)
        merged_x, pertoken_scale = self.gate_up_proj(x, out_dtype=torch.int32)
        intermediate_hidden_states, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
            merged_x, weight_scale=self.gate_up_proj.weight_scale,
            quant_scale=self.down_proj.smooth_scales,
            quant_mode=1, activate_left=True,
            activation_scale=pertoken_scale
        )
        return self.down_proj(intermediate_hidden_states, pertoken_scale), merged_x


class LongcatFlashTopkRouter(nn.Module):
    def __init__(self, config, prefix):
        super().__init__()
        self.config = config
        self.top_k = config.moe_topk
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.router_bias = config.router_bias
        self.num_experts = (
            config.n_routed_experts
            if config.zero_expert_num is None
            else config.n_routed_experts + self.config.zero_expert_num
        )

        self.classifier = ReplicatedLinear(self.config.hidden_size,
                                     self.num_experts,
                                     bias=self.router_bias,
                                     quant_config=None,
                                     params_dtype=torch.float32,
                                     prefix=f"{prefix}.classifier")
        # register_buffer not in named_parameters()
        self.e_score_correction_bias = nn.Parameter(
                torch.empty((self.num_experts), dtype=torch.float32)
            )

    @torch.no_grad()
    def get_topk_indices(self, scores):
        scores_for_choice = scores.view(-1, self.num_experts) + self.e_score_correction_bias.unsqueeze(0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return topk_indices

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states.type(torch.float32), self.classifier.weight, None)
        topk_weights, topk_indices, _ = torch_npu.npu_moe_gating_top_k(
                router_logits,
                k=self.top_k,
                bias=self.e_score_correction_bias.float(),
                renorm=0,  # 0: softmax->topk; 1: topk->softmax
                norm_type=0,  # 0: softmax; 1: sigmoid
                routed_scaling_factor=self.routed_scaling_factor,
                eps=float(1e-20)
            )
        return topk_indices.to(torch.int32), topk_weights, None


class LongcatFlashMoE(nn.Module):
    """
    moe module.
    """

    def __init__(self, config, runner_settings, prefix, **kwargs):
        super().__init__()
        self.config = config
        self.runner_settings = runner_settings
        self.gmm_quant_mode = (
            config.quant_config.gmm_quant_mode
            if config.quant_config is not None
            else "w16a16")
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.expert_ffn_hidden_size
        self.zero_expert_num = config.zero_expert_num
        self.zero_expert_type = config.zero_expert_type
        self.moe_tp_size = self.runner_settings.get("parallel_config").get("moe_tp_size", 1)
        self.moe_ep_size = self.runner_settings.get("parallel_config").get("moe_ep_size", 1)
        self.moe_chunk_max_len = self.runner_settings.get("model_config").get("moe_chunk_max_len", 65536)
        self.enable_multi_stream = self.runner_settings.get("model_config").get("enable_multi_stream", 0)

        self.hccl_comm_dict = kwargs.get("hccl_comm_dict", None)
        self.moe_ep_group = self.hccl_comm_dict.get("moe_ep_group", None)

        self.n_routed_experts = config.n_routed_experts
        self.num_zero_experts = config.zero_expert_num
        self.n_routed_experts_per_rank = self.n_routed_experts // self.moe_ep_size
        self.router = LongcatFlashTopkRouter(config, f"{prefix}.router")
        self.perfect_eplb = runner_settings.get("model_config").get("perfect_eplb", False)
        self.experts = FusedMoEGMM(
            num_experts=config.n_routed_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            bias=False,
            quant_config=config.quant_config,
            tp_size=self.moe_tp_size,
            tp_rank=dist.get_rank(self.hccl_comm_dict["moe_tp_group"]) if self.moe_tp_size > 1 else 0,
            ep_size=self.moe_ep_size,
            ep_rank=dist.get_rank(self.hccl_comm_dict["moe_ep_group"]) if self.moe_ep_size > 1 else 0,
            prefix=f"{prefix}.experts",
        )

    def dispatch_double_routing(self, tokens_per_expert, expanded_x, pertoken_scale):
        tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
        # (total_experts,)->(total_ranks*n_routed_experts_per_rank)
        dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert, group=self.moe_ep_group)
        # combine tensors, do reduceSum and D2H togather
        combine_tokens = torch.stack([tokens_per_expert_group, tokens_per_expert], dim=0)
        # view: EP, E // EP
        # sum: EP, per rank
        combine_tokens = combine_tokens.view(2, self.moe_ep_size, -1).sum(2)
        all_tokens = combine_tokens[0].sum()
        combine_tokens_cpu = combine_tokens.cpu().tolist()
        # alltoall input splits, total number of tokens routed from current rank to other ranks
        input_splits = combine_tokens_cpu[1]
        # alltoall output splits, number of tokens received by current rank from each other ranks
        output_splits = combine_tokens_cpu[0]
        # alltoall output, flattened into 1D, total number of tokens routed to current rank from other ranks
        gathered_tokens = expanded_x.new_empty(all_tokens.item(), expanded_x.shape[1])
        dist.all_to_all_single(gathered_tokens, expanded_x, output_splits, input_splits, group=self.moe_ep_group)

        gathered_pertoken_scale = None if pertoken_scale is None else\
                            pertoken_scale.new_empty(gathered_tokens.shape[0])
        if "a8" in self.gmm_quant_mode:
            dist.all_to_all_single(gathered_pertoken_scale,\
                                   pertoken_scale, output_splits, input_splits, group=self.moe_ep_group)
        return tokens_per_expert_group, gathered_tokens, gathered_pertoken_scale, input_splits, output_splits

    def forward_expert(self, gathered_tokens, tokens_per_expert_group, gathered_pertoken_scale):
        # reroute
        hidden_states_ordered_by_experts, gathered_pertoken_scale, gathered_ids_unsort, tokens_per_local_expert = \
                torch_npu.npu_moe_re_routing(gathered_tokens, tokens_per_expert_group.view(self.moe_ep_size, -1),
                per_token_scales=gathered_pertoken_scale)

        tokens_sum_router = tokens_per_local_expert.sum()
        # compute experts
        gmm_args = {
            "x": hidden_states_ordered_by_experts,
            "expert_tokens": tokens_per_local_expert,
            "group_list_type": 1,
        }
        if "a8" in self.gmm_quant_mode:
            gmm_args.update({"pertoken_scale": gathered_pertoken_scale[:tokens_sum_router]})

        hidden_states_ordered_by_experts = self.experts(**gmm_args)
        # finalize-rerouting
        new_x = torch.index_select(hidden_states_ordered_by_experts, 0, gathered_ids_unsort.float().argsort().int())
        return new_x

    def forward_combine_double_routing(self, new_x, expanded_x0, input_splits, output_splits):
        gathered_tokens = new_x.new_empty(*expanded_x0.shape)
        dist.all_to_all_single(gathered_tokens, new_x, input_splits, output_splits, group=self.moe_ep_group)
        return gathered_tokens

    def moe_infer_double_routing(self, x, topk_ids, topk_weight):
        batch_size, sequence_length, h = x.shape
        x = x.view(-1, h)
        hidden_states_list = []
        for hidden_states, topk_ids, topk_weight in zip(
                *self._split_tensors(batch_size * sequence_length, x, topk_ids, topk_weight)):
            bs_qlen = hidden_states.shape[0]
            expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = torch_npu.npu_moe_init_routing_v2(
                hidden_states,
                expert_idx=topk_ids,
                active_num=topk_ids.shape[0] * topk_ids.shape[1],
                scale=self.experts.smooth_scale_1 if "a8" in self.gmm_quant_mode else None,
                expert_num=self.n_routed_experts + self.num_zero_experts, # n_routed_experts + zero_expert_num
                expert_tokens_num_type=1,  # 0: cumsum mode(not supported now); 1: count mode
                expert_tokens_num_flag=True,
                active_expert_range=[0, self.n_routed_experts],
                quant_mode=1 if "a8" in self.gmm_quant_mode else -1
                # -1: non-quant; 1: dynamic quant; 0: static quant(not supported now)
            )
            # Only token assigned to routed experts need to be dispath
            clip_idx = sum(tokens_per_expert)
            expanded_x0, expanded_x1 = expanded_x[:clip_idx], expanded_x[clip_idx:]
            pertoken_scale0, pertoken_scale1 = pertoken_scale[:clip_idx], pertoken_scale[clip_idx:]

            tokens_per_expert_group, gathered_tokens, gathered_pertoken_scale, input_splits, output_splits =\
                self.dispatch_double_routing(tokens_per_expert, expanded_x0, pertoken_scale0)

            new_x = self.forward_expert(gathered_tokens, tokens_per_expert_group, gathered_pertoken_scale)

            gathered_tokens = self.forward_combine_double_routing(new_x, expanded_x0, input_splits, output_splits)
            gathered_tokens = torch.cat([gathered_tokens, expanded_x1], dim=0)

            zero_expert_mask = topk_ids < self.n_routed_experts
            zero_expert_weight = topk_weight.clone()
            zero_expert_weight[zero_expert_mask] = 0

            # finalize-routing
            routed_expert_mask = topk_ids >= self.n_routed_experts
            topk_weight[routed_expert_mask] = 0
            hidden_states = torch_npu.npu_moe_finalize_routing(
                gathered_tokens, skip1=None, skip2=None, bias=None,
                scales=topk_weight.to(gathered_tokens.dtype),
                expanded_src_to_dst_row=expanded_row_idx,
                export_for_source_row=None, drop_pad_mode=2
            )

            hidden_states += hidden_states * zero_expert_weight.sum(dim=1, keepdim=True).to(hidden_states.dtype)
            hidden_states = hidden_states.view(bs_qlen, self.hidden_size)
            hidden_states_list.append(hidden_states)

        hidden_states = torch.cat(hidden_states_list, dim=0) if len(hidden_states_list) > 1 else hidden_states_list[0]
        return hidden_states.view(batch_size, -1, h)

    def set_mc2_kwargs(self):
        global_rank = dist.get_rank()
        moe_ep_group_name = self.hccl_comm_dict.get("moe_ep_group_name", None)
        self.dispatch_kwargs = {
                "x_active_mask": None,
                "expert_shard_type": 0,
                "shared_expert_rank_num": 0,
                "moe_expert_num": self.n_routed_experts,
                "copy_expert_num": self.num_zero_experts,
                "global_bs": 0,
                "scales": self.experts.smooth_scale_1 if "a8" in self.gmm_quant_mode else None,
                "quant_mode": 2 if "a8" in self.gmm_quant_mode else 0,
                "group_ep": moe_ep_group_name,
                "ep_world_size": self.moe_ep_size,
                "ep_rank_id": global_rank // self.moe_tp_size,
                "group_tp": moe_ep_group_name,
                "tp_world_size": self.moe_tp_size,
                "tp_rank_id": global_rank % self.moe_tp_size,
            }
        self.combine_kwargs = {
                "x_active_mask": None,
                "expert_shard_type": 0,
                "shared_expert_rank_num": 0,
                "moe_expert_num": self.n_routed_experts,
                "copy_expert_num": self.num_zero_experts,
                "global_bs": 0,
                "group_ep": moe_ep_group_name,
                "ep_world_size": self.moe_ep_size,
                "ep_rank_id": global_rank // self.moe_tp_size,
                "group_tp": moe_ep_group_name,
                "tp_world_size": self.moe_tp_size,
                "tp_rank_id": global_rank % self.moe_tp_size
            }

    def moe_infer_dispatch_combine(self, x, topk_ids, topk_weight):
        """
        tp+ep mix strategy, for decode stage
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
        if "a8" in self.gmm_quant_mode:
            gmm_args.update({"pertoken_scale": dynamic_scale})

        hidden_states_ordered_by_experts = self.experts(**gmm_args)

        combine_args = {
            "expand_x": hidden_states_ordered_by_experts,
            "expert_ids": topk_ids,
            "ori_x": hidden_states,
            "assist_info_for_combine": expand_idx,
            "expert_scales": topk_weight.to(torch.float32), # [n*topk]
            "ep_send_counts": ep_recv_counts,
            "tp_send_counts": tp_recv_counts,
            **self.combine_kwargs
        }
        hidden_states = torch_npu.npu_moe_distribute_combine_v2(**combine_args)

        hidden_states = hidden_states.view(batch_size, sequence_length, self.hidden_size)
        return hidden_states

    def _split_tensors(self, bs_qlen, x, topk_ids, topk_weight):
        if bs_qlen > self.moe_chunk_max_len:  # need to chunk moe seq_len dim to avoid OOM
            num_chunks = (bs_qlen + self.moe_chunk_max_len - 1) // self.moe_chunk_max_len
            x_list = x.chunk(num_chunks, dim=0)
            topk_ids_list = topk_ids.chunk(num_chunks, dim=0)
            topk_weight_list = topk_weight.chunk(num_chunks, dim=0)
        else:
            x_list = [x]
            topk_ids_list = [topk_ids]
            topk_weight_list = [topk_weight]
        return x_list, topk_ids_list, topk_weight_list

    def forward(self, hidden_states, is_prefill, cur_topk_list=None):
        topk_indices, topk_weights, _ = self.router(hidden_states)
        if self.perfect_eplb:
            topk_indices = cur_topk_list
        topk_indices = topk_indices.to(torch.int32)

        if is_prefill:
            return self.moe_infer_double_routing(hidden_states, topk_indices, topk_weights)
        else:
            return self.moe_infer_dispatch_combine(hidden_states, topk_indices, topk_weights)


class LongcatFlashAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LongcatFlashConfig, runner_settings: Dict,
                 layer_idx: Optional[int] = None, prefix: Optional[str] = "", **kwargs):
        super().__init__()
        self.config = config
        self.runner_settings = runner_settings
        self.batch_size = self.runner_settings.get("data_config").get("batch_size", 16)
        self.attn_tp_size = self.runner_settings.get("parallel_config").get("attn_tp_size", 1)
        self.attn_dp_size = self.runner_settings.get("parallel_config").get("attn_dp_size", 1)
        self.o_proj_tp_size = self.runner_settings.get("parallel_config").get("o_proj_tp_size", 1)
        self.moe_tp_size = self.runner_settings.get("parallel_config").get("moe_tp_size", 1)
        self.moe_ep_size = self.runner_settings.get("parallel_config").get("moe_ep_size", 1)
        self.kvp_size = self.runner_settings.get("parallel_config").get("kvp_size", 1)
        self.batch_size_per_rank = self.runner_settings.get("data_config").get("batch_size_per_rank", 1)
        self.layer_idx = layer_idx
        # mtp layer is the last layer, with an index of 0
        if layer_idx == config.num_hidden_layers * 2:
            self.layer_idx = 0
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_heads_per_rank = self.num_heads // self.attn_tp_size
        self.num_key_value_heads_per_rank = 1
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_head_dim

        self.is_causal = True
        self.hccl_comm_dict = kwargs.get("hccl_comm_dict", None)
        self.kvp_group = self.hccl_comm_dict.get("kvp_group", None)
        self.kvp_rank = dist.get_rank(self.kvp_group) if self.kvp_group is not None else 0

        if self.q_lora_rank is None:
            self.q_proj = ColumnParallelLinear(self.hidden_size,
                                               self.num_heads * self.qk_head_dim,
                                               bias=False,
                                               quant_config=config.quant_config,
                                               tp_size=self.attn_tp_size,
                                               tp_rank=dist.get_rank(self.hccl_comm_dict["attn_tp_group"])
                                               if self.attn_tp_size > 1 else 0,
                                               prefix=f"{prefix}.q_proj")
        else:
            self.q_a_proj = ReplicatedLinear(self.hidden_size,
                                             self.q_lora_rank,
                                             bias=False,
                                             quant_config=config.quant_config,
                                             prefix=f"{prefix}.q_a_proj")
            self.q_a_layernorm = LongcatFlashRMSNorm(config.q_lora_rank)
            self.q_b_proj = ColumnParallelLinear(config.q_lora_rank,
                                                 self.num_heads * self.qk_head_dim,
                                                 bias=False,
                                                 quant_config=config.quant_config,
                                                 tp_size=self.attn_tp_size,
                                                 tp_rank=dist.get_rank(self.hccl_comm_dict["attn_tp_group"])
                                                 if self.attn_tp_size > 1 else 0,
                                                 prefix=f"{prefix}.q_b_proj")

        self.kv_a_proj_with_mqa = ReplicatedLinear(
                    self.hidden_size,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                    bias=config.attention_bias,
                    quant_config=config.quant_config,
                    prefix=f"{prefix}.kv_a_proj_with_mqa")
        self.kv_a_layernorm = LongcatFlashRMSNorm(self.kv_lora_rank)

        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=config.quant_config,
            tp_size=self.attn_tp_size,
            tp_rank=dist.get_rank(self.hccl_comm_dict["attn_tp_group"]) if self.attn_tp_size > 1 else 0,
            prefix=f"{prefix}.kv_b_proj")

        kv_b_proj_weight = self.kv_b_proj.weight.T
        expected_shape = (
                self.kv_lora_rank,
                self.num_heads_per_rank * (self.qk_nope_head_dim + self.v_head_dim)
            )
        if kv_b_proj_weight.shape != expected_shape:
            raise RuntimeError(f"{kv_b_proj_weight.shape} != {expected_shape}")
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_heads_per_rank,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        self.kv_b_proj_w_k_data, self.kv_b_proj_w_v_data = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        self.kv_b_proj_w_k_data = self.kv_b_proj_w_k_data.permute(1, 2, 0)
        self.kv_b_proj_w_v_data = self.kv_b_proj_w_v_data.transpose(0, 1)
        self.o_proj = RowParallelLinear(self.num_heads * self.v_head_dim,
                                        self.hidden_size,
                                        tp_size=self.o_proj_tp_size,
                                        tp_rank=dist.get_rank(self.hccl_comm_dict["o_proj_tp_group"])
                                        if self.o_proj_tp_size > 1 else 0,
                                        bias=False,
                                        input_is_parallel=True,
                                        quant_config=config.quant_config,
                                        prefix=f"{prefix}.o_proj")

        if config.mla_scale_q_lora:
            self.mla_scale_q_lora = (self.hidden_size / self.q_lora_rank) ** 0.5
        if config.mla_scale_kv_lora:
            self.mla_scale_kv_lora = (self.hidden_size / self.kv_lora_rank) ** 0.5
        self.softmax_scale = self.qk_head_dim ** (-0.5)

        max_length = self.runner_settings.get("model_config").get("pa_max_length", 2048)
        self.block_size = self.runner_settings.get("model_config").get("pa_block_size", 128)
        cache_len = max_length // self.block_size
        self.block_table = torch.arange(0, self.batch_size_per_rank * cache_len * self.attn_tp_size
                                        ).reshape(self.batch_size_per_rank * self.attn_tp_size, -1)
        self.block_table = self.block_table.to(dtype=torch.int32, device="npu")
        self.enable_mla_prolog = runner_settings.get("model_config").get(
            "enable_mla_prolog", False
        )
        self.enable_mla_prolog = (
            self.enable_mla_prolog
            and self.q_lora_rank is not None
        )
        self.kv_scale = None
        self.enable_gegraph = runner_settings.get("exe_mode", "eager") == "ge_graph"
        self.fa_ops = torch.ops.npu
        if self.enable_gegraph:
            self.fa_ops = tng.ops
        self.attn_tp_group = self.hccl_comm_dict.get("attn_tp_group", None)

    def o_proj_forward(
        self,
        attn_output: torch.Tensor = None,
    ):
        bsz, q_len, _ = attn_output.shape
        bsz = (bsz + self.attn_tp_size - 1) // self.attn_tp_size
        if self.kvp_size == 1 and self.o_proj_tp_size > 1 and self.attn_tp_size == 1:
            attn_output = attn_output.view(bsz * q_len, self.o_proj_tp_size, -1).transpose(1, 0).contiguous().view(-1)
            all2all_output = torch.empty_like(attn_output)
            # after all2all: (o_proj_tp_size * bs*q_len * num_heads // o_proj_tp_size * v_head_dim)
            dist.all_to_all_single(all2all_output, attn_output,
                                   group=self.hccl_comm_dict.get("o_proj_tp_group", None))
            attn_output = all2all_output

        # after view: (o_proj_tp_size * bs*q_len, num_heads // self.o_proj.tp_size * v_head_dim)
        attn_output = self.o_proj(attn_output.view(-1, self.num_heads // self.o_proj.tp_size * self.v_head_dim))
        o_proj = attn_output
        if self.kvp_size == 1 and self.o_proj_tp_size > 1:
            reduce_scatter_output = torch.empty((attn_output.size()[0] // self.o_proj_tp_size, attn_output.size()[1]),
                                                dtype=attn_output.dtype, device=attn_output.device)
            dist.reduce_scatter_tensor(reduce_scatter_output, attn_output,
                                       group=self.hccl_comm_dict.get("o_proj_tp_group", None))
            attn_output = reduce_scatter_output
        if self.kvp_size > 1:
            dist.all_reduce(attn_output, group=self.kvp_group)

        return attn_output.view(bsz, q_len, -1), o_proj

    def forward_page_attention_normal(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        kv_len: torch.IntTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        actual_seq_lengths_kv: list = None,
        is_prefill: bool = True,
        slot_mapping: Optional[torch.Tensor] = None
    ):
        if self.attn_tp_size > 1:
            bsz, q_len, _ = hidden_states.size()
            x_output = torch.empty([bsz * q_len * self.attn_tp_size, self.hidden_size], \
                                   dtype=hidden_states.dtype, device="npu")
            dist.all_gather_into_tensor(x_output, hidden_states, group=self.hccl_comm_dict.get("attn_tp_group", None))
            hidden_states = x_output.view(-1, q_len, self.hidden_size)
        bsz, q_len, _ = hidden_states.size()
        cos, sin = position_embeddings

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q_hidden_states = self.q_a_layernorm(self.q_a_proj(hidden_states))
            q = self.q_b_proj(q_hidden_states)

        latent_cache = self.kv_a_proj_with_mqa(hidden_states)

        # (B, S, N, D)
        q = q.view(bsz, -1, self.num_heads_per_rank, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        if self.mla_scale_q_lora is not None:
            q_nope = q_nope * self.mla_scale_q_lora
            q_pe = q_pe * self.mla_scale_q_lora

        q_pe = q_pe.transpose(1, 2)
        cos = cos.view(bsz, 1, -1, self.qk_rope_head_dim)
        sin = sin.view(bsz, 1, -1, self.qk_rope_head_dim)
        q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin)# (B, N, S, D)
        q_pe = q_pe.view(bsz, self.num_heads_per_rank, -1, self.qk_rope_head_dim).transpose(1, 2)
        # (B, S, N, D)
        query_states = [q_nope, q_pe]


        latent_cache = latent_cache.view(-1, 1, 1, self.kv_lora_rank + self.qk_rope_head_dim)  # (B,N,S,D)
        nope_cache = past_key_value[self.layer_idx][0]
        rope_cache = past_key_value[self.layer_idx][1]
        block_num, block_size, key_head_num, cache_dim = nope_cache.size()
        # prefill stage needs to view shapes as the following.
        cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)
        sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)
        _, _, k_rope, k_nope = torch_npu.npu_kv_rmsnorm_rope_cache(
            latent_cache,
            self.kv_a_layernorm.weight,
            cos,
            sin,
            slot_mapping.view(-1),
            rope_cache,
            nope_cache,
            epsilon=self.kv_a_layernorm.variance_epsilon,
            cache_mode="PA_NZ",
            is_output_kv=True
        )

        if self.mla_scale_kv_lora is not None:
            k_nope = k_nope * self.mla_scale_kv_lora

        k_nope_out = torch.matmul(k_nope.view(1, -1, self.kv_lora_rank), self.kv_b_proj_w_k.permute(0, 2, 1))
        v_out = torch.matmul(k_nope.view(1, -1, self.kv_lora_rank), self.kv_b_proj_w_v)

        # NTD foramt, repeat in N
        k_rope = k_rope.view(1, -1, self.qk_rope_head_dim).repeat(self.num_heads_per_rank, 1, 1)

        rank = self.kvp_rank
        heads = self.num_heads_per_rank // self.kvp_size
        indices = slice(rank * heads, (rank + 1) * heads)

        attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            query_states[0].flatten(0, 1).permute(1, 0, 2)[indices], k_nope_out[indices], v_out[indices],
            query_rope=query_states[1].flatten(0, 1).permute(1, 0, 2)[indices], key_rope=k_rope[indices],
            num_heads=heads,
            num_key_value_heads=heads,
            input_layout="NTD_TND",
            atten_mask=attention_mask, sparse_mode=3,
            actual_seq_lengths=actual_seq_lengths_kv,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            scale=self.softmax_scale,
            antiquant_mode=0, antiquant_scale=None,
            next_tokens=0
        )
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output, _ = self.o_proj_forward(attn_output)
        return attn_output

    def forward_page_attention_absorb(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        kv_len: torch.IntTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        actual_seq_lengths_kv: list = None,
        is_prefill: bool = False,
        slot_mapping: Optional[torch.Tensor] = None
    ):
        query_states, k_nope, k_rope = self.prepare_qkv(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            kv_len=kv_len,
            slot_mapping=slot_mapping,
            past_key_value=past_key_value,
        )

        attn_output, _ = self.fused_infer_attention_score(
            query_states=query_states,
            k_nope=k_nope,
            k_rope=k_rope,
            attention_mask=attention_mask,
            actual_seq_lengths_kv=actual_seq_lengths_kv
        )

        return attn_output

    def prepare_qkv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        kv_len: torch.IntTensor = None,
        slot_mapping: torch.IntTensor = None,
        past_key_value: Optional[Cache] = None,
    ):
        if self.attn_tp_size > 1:
            bsz, q_len, _ = hidden_states.size()
            x_output = torch.empty([bsz * q_len * self.attn_tp_size, self.hidden_size], \
                                   dtype=hidden_states.dtype, device="npu")
            dist.all_gather_into_tensor(x_output, hidden_states, group=self.hccl_comm_dict.get("attn_tp_group", None))
            hidden_states = x_output.view(-1, q_len, self.hidden_size)
        input_kwargs = {
            "hidden_states": hidden_states,
            "position_embeddings": position_embeddings,
            "kv_len": kv_len,
            "slot_mapping": slot_mapping,
            "past_key_value": past_key_value,
        }
        if self.enable_mla_prolog:
            fn = self.mla_prolog
        else:
            fn = self.prepare_qkv_absorb
        return fn(**input_kwargs)

    def prepare_qkv_absorb(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        kv_len: torch.IntTensor = None,
        slot_mapping: torch.IntTensor = None,
        past_key_value: Optional[Cache] = None,
    ):
        bsz, q_len, _ = hidden_states.size()
        cos, sin = position_embeddings

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))

        latent_cache = self.kv_a_proj_with_mqa(hidden_states)

        q = q.view(bsz, q_len, self.num_heads_per_rank, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        if self.mla_scale_q_lora is not None:
            q_nope = q_nope * self.mla_scale_q_lora
            q_pe = q_pe * self.mla_scale_q_lora

        q_nope = q_nope.view(-1, self.num_heads_per_rank, self.qk_nope_head_dim)
        if self.kv_b_proj_w_k.shape[0] * self.kv_b_proj_w_k.shape[1] <= 65535:  # 65535: max value of uint16
            q_nope = torch_npu.npu_transpose_batchmatmul(q_nope, self.kv_b_proj_w_k, bias=None, scale=None,
                                                        perm_x1=(1, 0, 2), perm_x2=(0, 1, 2), perm_y=(1, 0, 2)
                                                        )  # (b*s, n, d)
            q_nope = q_nope.view(bsz, q_len, self.num_heads_per_rank, self.kv_lora_rank)
        else:
            q_nope = (
                torch.matmul(q_nope.transpose(0, 1), self.kv_b_proj_w_k)
                .transpose(0, 1)
                .view(bsz, q_len, self.num_heads_per_rank, self.kv_lora_rank)
            )
        q_pe = q_pe.transpose(1, 2)
        cos = cos.view(bsz, 1, -1, self.qk_rope_head_dim)
        sin = sin.view(bsz, 1, -1, self.qk_rope_head_dim)
        q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin)  # rope requires (b, n, s, d)
        q_pe = q_pe.view(bsz, self.num_heads_per_rank, -1, self.qk_rope_head_dim).transpose(1, 2) # (b, s, n, d)
        query_states = [q_nope, q_pe]  # (b, s, n, D)

        latent_cache = latent_cache.view(
            bsz * q_len, 1, 1, self.kv_lora_rank + self.qk_rope_head_dim
        )  # (b*s, n, 1, d)
        cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)  # (b*s, n, 1, d)
        sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)  # (b*s, n, 1, d)
        nope_cache = past_key_value[self.layer_idx][0]
        rope_cache = past_key_value[self.layer_idx][1]
        block_num, block_size, _, _ = nope_cache.size()

        k_rope, k_nope, _, _ = torch_npu.npu_kv_rmsnorm_rope_cache(
            latent_cache,
            self.kv_a_layernorm.weight,
            cos,
            sin,
            slot_mapping.view(-1),
            rope_cache,
            nope_cache,
            epsilon=self.kv_a_layernorm.variance_epsilon,
            cache_mode="PA_NZ"
        )

        # apply kv_lora scaling
        if self.mla_scale_kv_lora is not None:
            k_nope = k_nope * self.mla_scale_kv_lora

        # adapter nz
        kv_cache_nz_dim = 16  # bf16 dtype is 16 for nz format, avoid dynamic shape in high torch version
        k_nope = k_nope.view(block_num, 1, self.kv_lora_rank // kv_cache_nz_dim,
                             block_size, kv_cache_nz_dim)
        k_rope = k_rope.view(block_num, 1, self.qk_rope_head_dim // kv_cache_nz_dim,
                             block_size, kv_cache_nz_dim)

        return query_states, k_nope, k_rope

    def mla_prolog(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        kv_len: torch.IntTensor = None,
        slot_mapping: torch.IntTensor = None,
        past_key_value: Optional[Cache] = None,
    ):
        bsz, q_len, _ = hidden_states.size()
        cos, sin = position_embeddings
        cos = cos.view(bsz, 1, -1, self.qk_rope_head_dim)
        sin = sin.view(bsz, 1, -1, self.qk_rope_head_dim)
        nope_cache = past_key_value[self.layer_idx][0]
        rope_cache = past_key_value[self.layer_idx][1]
        block_num, block_size, key_head_num, cache_dim = nope_cache.size()

        q_nope, q_pe, dequant_scale_q_nope, _, _ = torch.ops.npu.npu_mla_prolog_v3(
            token_x=hidden_states,
            weight_dq=self.q_a_proj.weight, weight_uq_qr=self.q_b_proj.weight,
            weight_uk=self.kv_b_proj_w_k, weight_dkv_kr=self.kv_a_proj_with_mqa.weight,
            rmsnorm_gamma_cq=self.q_a_layernorm.weight,
            rmsnorm_gamma_ckv=self.kv_a_layernorm.weight,
            rope_sin=sin.squeeze(1), rope_cos=cos.squeeze(1),
            cache_index=slot_mapping,
            kv_cache=nope_cache,
            kr_cache=rope_cache,
            rmsnorm_epsilon_cq=self.q_a_layernorm.variance_epsilon,
            rmsnorm_epsilon_ckv=self.kv_a_layernorm.variance_epsilon,
            cache_mode="PA_NZ",
            qc_qr_scale=self.mla_scale_q_lora,
            kc_scale=self.mla_scale_kv_lora
        )

        # adapter nz
        kv_cache_nz_dim = 16  # bf16 dtype is 16 for nz format, avoid dynamic shape in high torch version
        k_nope = nope_cache.view(block_num, 1, self.kv_lora_rank // (kv_cache_nz_dim),
                             block_size, kv_cache_nz_dim)
        k_rope = rope_cache.view(block_num, 1, self.qk_rope_head_dim // kv_cache_nz_dim,
                             block_size, kv_cache_nz_dim)

        query_states = [q_nope, q_pe]
        return query_states, k_nope, k_rope

    def fused_infer_attention_score(
        self,
        query_states: torch.Tensor,
        k_nope: torch.Tensor,
        k_rope: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        actual_seq_lengths_kv: list = None,
    ):
        # query_states here is a list of [q_nope, q_pe] with shape (b, s, n, D)
        bsz, q_len, _, _ = query_states[0].size()

        if q_len > 1: # mtp
            sparse_mode = 3
        else:
            sparse_mode = 0
            attention_mask = None

        attn_partial, lse_partial = self.fa_ops.npu_fused_infer_attention_score(
            query_states[0], k_nope, k_nope,
            query_rope=query_states[1], key_rope=k_rope,
            num_heads=self.num_heads_per_rank,
            num_key_value_heads=self.num_key_value_heads_per_rank,
            input_layout="BSND_NBSD",
            block_table=self.block_table,
            block_size=self.block_size,
            atten_mask=attention_mask,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            scale=self.softmax_scale,
            antiquant_mode=0, antiquant_scale=None,
            sparse_mode=sparse_mode,
            softmax_lse_flag=self.kvp_size > 1,
        )
        attn_partial = attn_partial.view(self.num_heads_per_rank, -1, self.kv_lora_rank)
        attn_partial = (
            torch.matmul(attn_partial, self.kv_b_proj_w_v)
            .reshape(self.num_heads_per_rank, bsz, q_len, -1)
        )
        if self.kvp_size > 1:
            def _all_to_all_along_headdim(partial: torch.Tensor, group: dist.ProcessGroup):
                scattered = partial.new_empty(self.kvp_size, self.num_heads_per_rank // self.kvp_size,
                                              *partial.shape[1:])
                dist.all_to_all_single(scattered.view(-1), partial.view(-1),
                                       group=group)
                return scattered

            # AllToAll of attn_partial(NBSD) along N-axis.
            attn_scatter = _all_to_all_along_headdim(attn_partial, self.kvp_group)

            # Transpose(BNSD->NBSD) and AllToAll of lse_partial along N-axis.
            # This AllToAll can't be merged with the previous one, due to the
            # dtype difference between attn and lse.
            # BNSD -> NBSD
            lse_partial = lse_partial.transpose(0, 1).contiguous()                    
            lse_scatter = _all_to_all_along_headdim(lse_partial, self.kvp_group)            
            attn_output = (attn_scatter * lse_scatter.softmax(dim=0)).sum(dim=0).to(torch.bfloat16)
        else:
            attn_output = attn_partial
        # (N, B, S, D) -> (B, S, N, D) -> (B, S, N * D)
        attn_output = attn_output.permute(1, 2, 0, 3).reshape(bsz, q_len, -1)
        attn_output, o_proj = self.o_proj_forward(attn_output)
        return attn_output, o_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_len: torch.IntTensor = None,
        actual_seq_lengths_kv: list = None,
        position_embeddings: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        is_prefill: bool = True,
        output_attentions: bool = False,
        slot_mapping: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_kwargs = {
            "hidden_states": hidden_states,
            "position_embeddings": position_embeddings,
            "kv_len": kv_len,
            "position_ids": position_ids,
            "past_key_value": past_key_value,
            "actual_seq_lengths_kv": actual_seq_lengths_kv,
            "attention_mask": attention_mask,
            "is_prefill": is_prefill,
            "slot_mapping": slot_mapping
        }
        if is_prefill:
            fn = self.forward_page_attention_normal
        else:
            fn = self.forward_page_attention_absorb
        return fn(**input_kwargs)


class LongcatFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LongcatFlashConfig, runner_settings: Dict, layer_idx: int, prefix: str, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.runner_settings = runner_settings
        self.hidden_size = config.hidden_size
        self.mlp = LongcatFlashMoE(config, self.runner_settings, prefix=f"{prefix}.mlp", **kwargs)
        self.enable_multi_stream = self.runner_settings.get("model_config").get("enable_multi_stream", 0)
        self.enable_superkernel = self.runner_settings.get("model_config").get("enable_superkernel", False)
        self.enable_prefetch = self.runner_settings.get("model_config").get("enable_prefetch", False)
        if self.enable_multi_stream == 2: # takes effects only when enable_multi_stream > 0
            self.aic_num1 = "12"
            self.aiv_num1 = "24"
            self.aic_num2 = "12"
            self.aiv_num2 = "24"
        else:
            self.aic_num1 = "8"
            self.aiv_num1 = "16"
            self.aic_num2 = "16"
            self.aiv_num2 = "32"

        self_attn = []
        mlps = []
        input_layernorm = []
        post_attention_layernorm = []
        for i in range(2):
            self_attn.append(
                LongcatFlashAttention(
                    config=config,
                    runner_settings=self.runner_settings,
                    layer_idx=layer_idx * 2 + i,
                    prefix=f"{prefix}.self_attn.{i}",
                    **kwargs
                )
            )
            mlps.append(LongcatFlashMLP(config, runner_settings, f"{prefix}.mlps.{i}", **kwargs))
            input_layernorm.append(LongcatFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps))
            post_attention_layernorm.append(LongcatFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps))

        self.self_attn = nn.ModuleList(self_attn)
        self.mlps = nn.ModuleList(mlps)
        self.input_layernorm = nn.ModuleList(input_layernorm)
        self.post_attention_layernorm = nn.ModuleList(post_attention_layernorm)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_len: torch.IntTensor,
        actual_seq_lengths_kv: list,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        is_prefill: Optional[bool] = False,
        slot_mapping: Optional[torch.Tensor] = None,
        past_residual: Optional[torch.Tensor] = None,
        cur_topk_list: Optional[torch.Tensor] = None,
        next_layer: Optional['LongcatFlashDecoderLayer'] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if (self.enable_multi_stream > 0) and not is_prefill:
            return self.multi_stream_forward(
                hidden_states,
                kv_len,
                actual_seq_lengths_kv,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                is_prefill=is_prefill,
                slot_mapping=slot_mapping,
                past_residual=past_residual,
                cur_topk_list=cur_topk_list,
                next_layer=next_layer,
                **kwargs,
            )
        residual = past_residual
        for i in range(2):
            hidden_states, residual = self.input_layernorm[i](hidden_states, residual)

            hidden_states = self.self_attn[i](
                hidden_states=hidden_states,
                kv_len=kv_len,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                is_prefill=is_prefill,
                slot_mapping=slot_mapping
            )

            hidden_states, residual = self.post_attention_layernorm[i](hidden_states, residual)

            if i == 0:
                # shortcut output (MoE output)
                shortcut_mlp_output = self.mlp(hidden_states, is_prefill, cur_topk_list=cur_topk_list)

            hidden_states, _, _ = self.mlps[i](hidden_states)
            if i == 1:
                hidden_states = hidden_states + shortcut_mlp_output

        outputs = (residual, hidden_states)
        return outputs

    def multi_stream_forward(
        self,
        hidden_states: torch.Tensor,
        kv_len: torch.IntTensor,
        actual_seq_lengths_kv: list,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        is_prefill: Optional[bool] = False,
        slot_mapping: Optional[torch.Tensor] = None,
        past_residual: Optional[torch.Tensor] = None,
        cur_topk_list: Optional[torch.Tensor] = None,
        next_layer: Optional['LongcatFlashDecoderLayer'] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = past_residual

        with superkernel_scope(self.enable_superkernel, f"scope_{self.layer_idx}_part1", ""):
            hidden_states, residual = self.input_layernorm[0](hidden_states, residual)
            query_states, k_nope, k_rope = self.self_attn[0].prepare_qkv(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                kv_len=kv_len,
                slot_mapping=slot_mapping,
                past_key_value=past_key_value,
            )

            hidden_states, o_proj = self.self_attn[0].fused_infer_attention_score(
                query_states=query_states,
                k_nope=k_nope,
                k_rope=k_rope,
                attention_mask=attention_mask,
                actual_seq_lengths_kv=actual_seq_lengths_kv
            )

            npu_prefetch(self.enable_prefetch, self.mlp.router.classifier.weight.data, o_proj, 18 * 1024 * 1024, 0)
            hidden_states_norm, residual = self.post_attention_layernorm[0](hidden_states, residual)

        # shortcut output (MoE output)
        with npu_stream_switch(True, "1"):
            with limit_core_num(True, self.aic_num1, self.aiv_num1):
                with superkernel_scope(self.enable_superkernel, f"scope_{self.layer_idx}_part2_moe", ""):
                    shortcut_mlp_output = self.mlp(hidden_states_norm, is_prefill, cur_topk_list=cur_topk_list)

        with limit_core_num(True, self.aic_num2, self.aiv_num2):
            with superkernel_scope(self.enable_superkernel, f"scope_{self.layer_idx}_part2_main", ""):
                hidden_states, _, dsq = self.mlps[0](hidden_states_norm, self.enable_prefetch, o_proj)

                #self.q_a_proj.weight, self.q_b_proj.weight, self.kv_a_proj_with_mqa.weight
                npu_prefetch(self.enable_prefetch, self.self_attn[1].q_a_proj.weight.data, \
                             dsq, 18 * 1024 * 1024, 0)
                npu_prefetch(self.enable_prefetch, self.self_attn[1].q_b_proj.weight.data, \
                             dsq, 36 * 1024 * 1024, 0)
                npu_prefetch(self.enable_prefetch, self.self_attn[1].kv_a_proj_with_mqa.weight.data, \
                             dsq, 7 * 1024 * 1024, 0)

                hidden_states, residual = self.input_layernorm[1](hidden_states, residual)

                query_states, k_nope, k_rope = self.self_attn[1].prepare_qkv(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    kv_len=kv_len,
                    slot_mapping=slot_mapping,
                    past_key_value=past_key_value,
                )

                hidden_states, o_proj = self.self_attn[1].fused_infer_attention_score(
                    query_states=query_states,
                    k_nope=k_nope,
                    k_rope=k_rope,
                    attention_mask=attention_mask,
                    actual_seq_lengths_kv=actual_seq_lengths_kv
                )
                hidden_states, residual = self.post_attention_layernorm[1](hidden_states, residual)
                hidden_states, down_proj, _ = self.mlps[1](hidden_states, self.enable_prefetch, o_proj)
                if next_layer is not None:
                    npu_prefetch(self.enable_prefetch, next_layer.self_attn[0].q_a_proj.weight.data, \
                                 down_proj, 18 * 1024 * 1024, 0)
                    npu_prefetch(self.enable_prefetch, next_layer.self_attn[0].q_b_proj.weight.data, \
                                 down_proj, 36 * 1024 * 1024, 0)
                    npu_prefetch(self.enable_prefetch, next_layer.self_attn[0].kv_a_proj_with_mqa.weight.data,
                                 down_proj, 7 * 1024 * 1024, 0)

        hidden_states = hidden_states + shortcut_mlp_output
        outputs = (residual, hidden_states)
        return outputs



class LongcatFlashPreTrainedModel(PreTrainedModel):
    config: LongcatFlashConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LongcatFlashDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": LongcatFlashDecoderLayer,
        "attentions": LongcatFlashAttention,
    }


class LongcatFlashModel(LongcatFlashPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"model\.mtp.*"]

    def __init__(self, config: LongcatFlashConfig, runner_settings: Dict, **kwargs):
        super().__init__(config)
        self.config = config
        self.global_rank = dist.get_rank()
        self.runner_settings = runner_settings
        self.embed_tp_size = self.runner_settings.get("parallel_config").get("embed_tp_size", 1)
        self.attn_tp_size = self.runner_settings.get("parallel_config").get("attn_tp_size", 1)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.vocab_size_per_rank = self.vocab_size // self.embed_tp_size

        self.kv_len_offset = kwargs.get("kv_len_offset", None)
        self.pa_max_length = self.runner_settings.get("model_config").get("pa_max_length", 2048)

        self.hccl_comm_dict = kwargs.get("hccl_comm_dict", None)
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            self.padding_idx,
            torch.bfloat16,
            tp_size=self.embed_tp_size,
            tp_rank=dist.get_rank(self.hccl_comm_dict["embed_tp_group"]) if self.embed_tp_size > 1 else 0)
        self.layers = nn.ModuleList(
            [
                LongcatFlashDecoderLayer(config, self.runner_settings, layer_idx, \
                                         prefix=f"model.layers.{layer_idx}", **kwargs)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LongcatFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        _init_rope(self)
        self.gradient_checkpointing = False

        kvp_group = self.hccl_comm_dict["kvp_group"]
        self.kvp_size = dist.get_world_size(kvp_group) if kvp_group is not None else 1
        self.kvp_rank = dist.get_rank(kvp_group) if kvp_group is not None else 0

        # Initialize weights and apply final processing
        self.post_init()

    def get_slot_mapping(self, input_ids, kv_len, is_prefill, device):
        batch_size, seq_len = input_ids.size()
        if not is_prefill:
            slot_mapping = kv_len.view(batch_size * self.attn_tp_size, -1)
        else:
            slot_mapping = (
                torch.arange(seq_len, dtype=kv_len.dtype, device=device)
                .unsqueeze(0)
                .expand(batch_size * self.attn_tp_size, -1)
                )
        mask = slot_mapping % self.kvp_size != self.kvp_rank
        return torch.where(mask, -1, 
                            slot_mapping // self.kvp_size + self.kv_len_offset[:batch_size * self.attn_tp_size])

    def prepare_inputs_for_layer(self, input_ids, kv_len, position_ids, actual_seq_lengths_kv, is_prefill):
        batch_size, seq_length = input_ids.shape

        if self.embed_tp_size > 1 and self.kvp_size == 1:
            embed_tp_group = self.hccl_comm_dict.get("embed_tp_group", None)
            all_input_ids = input_ids.new_empty(batch_size * self.embed_tp_size, seq_length)
            dist.all_gather_into_tensor(all_input_ids, input_ids, group=embed_tp_group)

            # Map the token IDs in all_input_ids to the vocab shard range assigned to the current rank
            new_input_ids = all_input_ids - (self.global_rank % self.embed_tp_size) * self.vocab_size_per_rank
            # Mark which tokens belong to the current rank's vocab shard
            mask = (new_input_ids >= 0) & (new_input_ids < self.vocab_size_per_rank) # (bs, qlen)
            # Set out-of-bounds parts to 0, keeping valid IDs within [0, vocab_size_per_rank)
            new_input_ids_per_rank = new_input_ids * mask
            inputs_embeds = self.embed_tokens(new_input_ids_per_rank) * mask.unsqueeze(-1)

            inputs_embeds_attn = inputs_embeds.new_empty(batch_size, seq_length, inputs_embeds.shape[-1])
            dist.reduce_scatter_tensor(inputs_embeds_attn, inputs_embeds, group=embed_tp_group)
            inputs_embeds = inputs_embeds_attn
            
        elif self.embed_tp_size > 1 and self.kvp_size > 1:
            embed_tp_group = self.hccl_comm_dict.get("embed_tp_group", None)            
            # Map the token IDs in input_ids to the vocab shard range assigned to the current rank
            new_input_ids = input_ids - (self.global_rank % self.embed_tp_size) * self.vocab_size_per_rank
            # Mark which tokens belong to the current rank's vocab shard
            mask = (new_input_ids >= 0) & (new_input_ids < self.vocab_size_per_rank) # (bs, qlen)
            # Set out-of-bounds parts to 0, keeping valid IDs within [0, vocab_size_per_rank)
            new_input_ids_per_rank = new_input_ids * mask
            inputs_embeds = self.embed_tokens(new_input_ids_per_rank) * mask.unsqueeze(-1)
            dist.all_reduce(inputs_embeds, group=embed_tp_group)
        else:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, kv_len, self.config.max_position_embeddings, \
                                              is_prefill=is_prefill, attn_tp_size=self.attn_tp_size)

        residual = None
        slot_mapping = self.get_slot_mapping(input_ids, kv_len, is_prefill, position_ids.device)
        return hidden_states, residual, kv_len, position_embeddings, slot_mapping, actual_seq_lengths_kv

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_len: Optional[torch.IntTensor] = None,
        actual_seq_lengths_kv: Optional[list] = None,
        is_prefill: Optional[bool] = False,
        cur_topk_list: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        hidden_states, residual, kv_len, position_embeddings, slot_mapping, actual_seq_lengths_kv =\
            self.prepare_inputs_for_layer(input_ids, kv_len, position_ids, actual_seq_lengths_kv, is_prefill)
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states, residual = decoder_layer(
                hidden_states,
                kv_len,
                actual_seq_lengths_kv,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                is_prefill=is_prefill,
                slot_mapping=slot_mapping,
                past_residual=residual,
                cur_topk_list=cur_topk_list,
                next_layer=self.layers[i + 1] if i < self.config.num_hidden_layers - 1 else None,
                **kwargs,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class LongcatFlashForCausalLM(LongcatFlashPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _keys_to_ignore_on_load_unexpected = [r"model\.mtp.*"]

    def __init__(self, config, runner_settings, is_mtp=False):
        super().__init__(config)
        self.config = config
        self.runner_settings = runner_settings
        self.top_k = config.moe_topk
        self.perfect_eplb = self.runner_settings.get("model_config").get("perfect_eplb", False)
        self.is_mtp = is_mtp
        self.lmhead_tp_size = self.runner_settings.get("parallel_config").get("lmhead_tp_size", 1)
        self.moe_ep_size = self.runner_settings.get("parallel_config").get("moe_ep_size", 1)

        self.num_experts = (
            config.n_routed_experts
            if config.zero_expert_num is None
            else config.n_routed_experts + config.zero_expert_num
        )
        self.experts_per_rank = self.num_experts // self.moe_ep_size
        self.get_parallel_settings()
        kwargs = {}
        default_pg = get_default_group()
        if default_pg is not None:
            if dist.get_world_size() > 1:
                self.hccl_comm_dict = self.init_parallel_comm_group()
                kwargs.update({"hccl_comm_dict": self.hccl_comm_dict})
                
        kvp_group = self.hccl_comm_dict["kvp_group"]
        self.kvp_size = dist.get_world_size(kvp_group) if kvp_group is not None else 1
        self.kvp_rank = dist.get_rank(kvp_group) if kvp_group is not None else 0
        batch_size_per_rank = self.runner_settings.get("data_config").get("batch_size_per_rank", 1)
        self.pa_max_length = self.runner_settings.get("model_config").get("pa_max_length", 2048)
        self.block_size = self.runner_settings.get("model_config").get("pa_block_size", 128)
        self.cache_len = self.pa_max_length // self.block_size
        self.kv_cache_num_block = self.cache_len * batch_size_per_rank * self.attn_tp_size
        self.kv_len_offset = torch.arange(
            0,
            batch_size_per_rank * self.pa_max_length * self.attn_tp_size,
            self.pa_max_length,
            dtype=torch.int64,
            device="npu",
        ).view(-1, 1)
        kwargs.update({"kv_len_offset": self.kv_len_offset})
        self.model = LongcatFlashModel(config, runner_settings, **kwargs)
        self.vocab_size = config.vocab_size
        self.lm_head = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            bias=False,
            tp_size=self.lmhead_tp_size,
            tp_rank=dist.get_rank(self.hccl_comm_dict.get("lmhead_tp_group")) if self.lmhead_tp_size > 1 else 0
        )
        # Initialize weights and apply final processing
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def prefill(
        self,
        **kwargs
    ):
        logits, prev_hidden_states = self.forward(
            is_prefill=True,
            **kwargs
        )
        return logits, prev_hidden_states

    def decode(
        self,
        **kwargs
    ):
        logits, prev_hidden_states = self.forward(
            is_prefill=False,
            **kwargs
        )
        return logits, prev_hidden_states

    def mtp_compile_decode(
        self,
        **kwargs
    ):
        logits, prev_hidden_states = self.forward(
            is_prefill=False,
            **kwargs
        )
        return logits, prev_hidden_states

    def get_parallel_settings(self):
        self.embed_tp_size = self.runner_settings.get("parallel_config").get("embed_tp_size", 1)
        self.embed_dp_size = self.runner_settings.get("parallel_config").get("embed_dp_size", 1)
        self.attn_dp_size = self.runner_settings.get("parallel_config").get("attn_dp_size", 1)
        self.attn_tp_size = self.runner_settings.get("parallel_config").get("attn_tp_size", 1)
        self.o_proj_tp_size = self.runner_settings.get("parallel_config").get("o_proj_tp_size", 1)
        self.moe_ep_size = self.runner_settings.get("parallel_config").get("moe_ep_size", 1)
        self.moe_tp_size = self.runner_settings.get("parallel_config").get("moe_tp_size", 1)
        self.lmhead_tp_size = self.runner_settings.get("parallel_config").get("lmhead_tp_size", self.embed_tp_size)
        self.moe_dp_size = self.runner_settings.get("parallel_config").get("moe_dp_size", 1)
        self.dense_tp_size = self.runner_settings.get("parallel_config").get("dense_tp_size", 1)
        self.kvp_size = self.runner_settings.get("parallel_config").get("kvp_size", 1)

    def init_parallel_comm_group(self):
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()

        attn_tp_group = init_comm_group(
            global_rank=global_rank, group_num=self.attn_dp_size, world_size=world_size,
            group_stride=1, group_name="attn_tp_group")

        o_proj_tp_group = init_comm_group(
            global_rank=global_rank, group_num=world_size // self.o_proj_tp_size, world_size=world_size,
            group_stride=1, group_name="o_proj_tp_group")

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

        if self.dense_tp_size == self.attn_tp_size:
            dense_tp_group = attn_tp_group
        else:
            dense_tp_group = init_comm_group(
                global_rank=global_rank, group_num=world_size // self.dense_tp_size, world_size=world_size,
                group_stride=1, group_name="dense_tp_group")

        if self.moe_tp_size == self.attn_tp_size:
            moe_tp_group = attn_tp_group
        else:
            moe_tp_group = init_comm_group(
                global_rank=global_rank, group_num=self.moe_dp_size, world_size=world_size,
                group_stride=1, group_name="moe_tp_group")

        moe_ep_group, moe_ep_group_name = init_comm_group(
            global_rank=global_rank, group_num=self.moe_tp_size, world_size=world_size,
            group_stride=self.moe_tp_size, group_name="moe_ep_group", return_name=True)

        kvp_group = init_comm_group(
            global_rank=global_rank, group_num=world_size // self.kvp_size, world_size=world_size,
            group_name="kvp_group")

        hccl_comm_dict = {
                "default_pg": get_default_group(),
                "attn_tp_group": attn_tp_group, "embed_tp_group": embed_tp_group,
                "o_proj_tp_group": o_proj_tp_group,
                "moe_tp_group": moe_tp_group, "moe_ep_group": moe_ep_group,
                "moe_ep_group_name": moe_ep_group_name,
                "lmhead_tp_group": lmhead_tp_group,
                "dense_tp_group": dense_tp_group,
                "kvp_group": kvp_group,
            }
        return hccl_comm_dict

    def forward_lm_head(self, outputs, position_ids, is_prefill=False):
        bs, q_len, hidden_size = outputs.shape
        if is_prefill:
            gather_index, _ = torch.max(position_ids, dim=-1)
            gather_index = gather_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, outputs.shape[-1])
            outputs = torch.gather(outputs, 1, gather_index)
            q_len = 1 # prefill takes th last token
        else: # combine bs and q_len axes for lm_head
            outputs = outputs.view(bs * q_len, 1, hidden_size)

        if (self.attn_dp_size == 1) or (self.lmhead_tp_size == 1):
            hidden_states = outputs
        else:
            # allgather: (bs / attn_dp, hidden_size) -> (bs, hidden_size)
            hidden_states = torch.zeros_like(outputs).repeat(self.lmhead_tp_size, 1, 1)
            dist.all_gather_into_tensor(hidden_states, outputs, group=self.hccl_comm_dict.get("lmhead_tp_group", None))

        logits = self.lm_head(hidden_states) # (lmhead_tp_size * bs / attn_dp, 1, vocab_size / lmhead_tp_size)
        if self.lmhead_tp_size > 1: # -> (bs / attn_dp, 1, vocab_size)
            if self.attn_dp_size == 1:
                new_logits = torch.zeros_like(logits).repeat(self.lmhead_tp_size, 1, 1)
                dist.all_gather_into_tensor(new_logits, logits, group=self.hccl_comm_dict.get("lmhead_tp_group", None))
            else:
                new_logits = torch.zeros_like(logits).view(-1)
                dist.all_to_all_single(new_logits, logits.view(-1), \
                        group=self.hccl_comm_dict.get("lmhead_tp_group", None))

            # transpose: (lmhead_tp_size * bs / attn_dp, vocab_size / lmhead_tp_size) -> (bs / attn_dp, vocab_size)
            new_logits = new_logits.reshape(
                self.lmhead_tp_size, bs * q_len, logits.shape[1], -1).permute(1, 2, 0, 3)
            logits = new_logits.reshape(bs * q_len, logits.shape[1], self.config.vocab_size)
        logits = logits.reshape(bs, q_len, -1).float()
        return logits


    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        actual_seq_lengths_kv: Optional[list] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_len: Optional[torch.IntTensor] = None,
        is_prefill: Optional[bool] = False,
        cur_topk_list: Optional[torch.Tensor] = None,
        prev_hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        r"""
        Example:
        ```python
        >>> from transformers import AutoTokenizer, LongcatFlashForCausalLM
        >>> model = LongcatFlashForCausalLM.from_pretrained("meta-longcat_flash/LongcatFlash-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-longcat_flash/LongcatFlash-2-7b-hf")
        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")
        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            kv_len=kv_len,
            is_prefill=is_prefill,
            cur_topk_list=cur_topk_list,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            **kwargs,
        )

        prev_hidden_states = outputs.last_hidden_state
        hidden_states = outputs.last_hidden_state
        logits = self.forward_lm_head(hidden_states, position_ids, is_prefill=is_prefill)

        return logits, prev_hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoEGMM.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts)

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if self.config.architectures[0] == 'LongcatFlashForCausalLM' and self.config.num_nextn_predict_layers > 0:
                mtp_prefix = "model.mtp"
                if name.startswith(mtp_prefix):
                    continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue

                if (("mlp.experts." in name) and name not in params_dict):
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def init_cache(
        self,
        input_ids
    ):
        batch_size, seq_len = input_ids.size()
        cache_seq_len = self.config.max_position_embeddings
        dtype = torch.bfloat16 if self.config.torch_dtype is None else self.config.torch_dtype

        past_key_values = ()

        num_hidden_layers = 1 if self.is_mtp else self.config.num_hidden_layers
        num_hidden_layers = num_hidden_layers * 2
        cache_nope_shape = (
                        self.kv_cache_num_block,
                        self.block_size,
                        1,
                        self.config.kv_lora_rank
                    )

        cache_rope_shape = (
                        self.kv_cache_num_block,
                        self.block_size,
                        1,
                        self.config.qk_rope_head_dim
                    )

        for _ in range(num_hidden_layers):
            cache_nope = torch.zeros(cache_nope_shape, dtype=dtype, device=input_ids.device)
            cache_rope = torch.zeros(cache_rope_shape, dtype=dtype, device=input_ids.device)
            past_key_values += ((cache_nope, cache_rope),)

        return past_key_values

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        is_prefill=None,
        kv_len=None,
        share_mask_tril=None,
        input_lens=None,
        prev_hidden_states=None,
        **kwargs
    ):
        batch_size, seq_len = input_ids.size()
        if is_prefill:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            attention_mask = share_mask_tril

            # Obtain the actual length of the request
            kv_len = torch.max(position_ids, axis=1)[0] + 1
            kv_len = self._repeat_batch(kv_len, self.attn_tp_size)
            kv_len_withpad = torch.tensor(
                [seq_len for _ in range(batch_size * self.attn_tp_size)], device=kv_len.device, dtype=kv_len.dtype)
            actual_seq_lengths_kv = torch.cumsum(kv_len_withpad, dim=0).tolist()

        else:
            if kv_len.shape[0] == batch_size:
                kv_len = self._repeat_batch(kv_len, self.attn_tp_size)
            if seq_len > 1: # fa requires sparse mode 3 and 2048 * 2048 mask for mtp
                attention_mask = share_mask_tril
                last_kv = torch.max(kv_len, axis=1)[0]
                actual_seq_lengths_kv = last_kv + 1
            else:
                attention_mask = None
                actual_seq_lengths_kv = kv_len + 1

            actual_seq_lengths_kv = (
                actual_seq_lengths_kv // self.kvp_size +
                (actual_seq_lengths_kv % self.kvp_size > self.kvp_rank)
            )

            if self.runner_settings.get("exe_mode") != "ge_graph":
                # dynamo use fa_tensor
                actual_seq_lengths_kv = actual_seq_lengths_kv.cpu().detach().tolist()
            position_ids = kv_len.view(-1, 1)

        model_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "kv_len": kv_len,
            "actual_seq_lengths_kv": actual_seq_lengths_kv,
            "prev_hidden_states": prev_hidden_states,
        }
        return model_inputs

    def gen_cur_topk_idx(
        self,
        is_prefill,
        batch_size,
        seq_len
    ):
        if not self.perfect_eplb:
            return None
        # if use perfect_eplb
        global_rank = dist.get_rank()
        if is_prefill:
            if self.moe_ep_size != 1:
                tokens_per_rank_prefill = batch_size * seq_len
            else:
                tokens_per_rank_prefill = batch_size * seq_len * self.attn_dp_size
            step_prefill = tokens_per_rank_prefill * self.top_k
            cur_topk_list_prefill = [
                (i + global_rank) % self.num_experts for i in range(step_prefill)]
            cur_topk_list = torch.Tensor(cur_topk_list_prefill).int().view(tokens_per_rank_prefill, -1).npu()
        else:
            if self.moe_tp_size > 1:
                tokens_per_rank_decode = batch_size * self.top_k * seq_len
                cur_topk_list_decode = []
                for offset in range(self.moe_ep_size):
                    expert_start = offset * self.experts_per_rank
                    expert_end = expert_start + tokens_per_rank_decode
                    cur_topk_list_decode = cur_topk_list_decode + [i for i in range(expert_start, expert_end)]
                cur_topk_list = torch.Tensor(cur_topk_list_decode).int().view(batch_size * seq_len, -1).npu()
            else:
                expanded_tokens = batch_size * self.top_k * seq_len  # Total tokens to be allocated to experts
                step_gap = self.num_experts // self.moe_ep_size # Number of experts per rank
                expanded_offset = expanded_tokens * global_rank + global_rank # Token count offset
                cur_topk_list_decode = []
                # Allocate experts using round-robin algorithm
                for idx in range(expanded_tokens):
                    col = (expanded_offset + idx) % self.moe_ep_size  # Column index
                    row = (expanded_offset + idx) // self.moe_ep_size % step_gap  # Row index
                    expert_idx = row + col * step_gap  # Final expert index
                    cur_topk_list_decode.append(expert_idx)
                cur_topk_list = torch.Tensor(cur_topk_list_decode).int().view(batch_size * seq_len, -1).npu()
        return cur_topk_list

    @staticmethod
    def _repeat_batch(tensor, repeat_num):
        if repeat_num == 1:
            return tensor
        return tensor.repeat(repeat_num, *[1] * (tensor.dim() - 1))


class LongcatFlashMTPDecoderLayer(nn.Module):
    def __init__(self, config: LongcatFlashConfig, runner_settings: Dict, layer_idx: int, prefix: str, **kwargs):
        super().__init__()
        self.self_attn = LongcatFlashAttention(config=config, runner_settings=runner_settings, \
                                               layer_idx=layer_idx, prefix=f"{prefix}.self_attn", **kwargs)
        self.mlp = LongcatFlashMLP(config, runner_settings, f"{prefix}.mlp", **kwargs)

        self.input_layernorm = LongcatFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps) # eps=1e-5
        self.post_attention_layernorm = LongcatFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_len: torch.IntTensor,
        actual_seq_lengths_kv: list,
        cos_sin: torch.Tensor,
        past_residual: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        is_prefill: Optional[bool] = False,
        slot_mapping: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            position_embeddings=cos_sin,
            slot_mapping=slot_mapping,
            is_prefill=is_prefill,
            kv_len=kv_len,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _, _ = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class LongcatFlashMTPLayer(nn.Module):
    def __init__(self, config: LongcatFlashConfig, runner_settings: Dict, layer_idx: int, **kwargs):
        super().__init__()
        self.runner_settings = runner_settings
        self.attn_tp_size = self.runner_settings.get("parallel_config").get("attn_tp_size", 1)
        self.mtp = nn.Module()
        self.mtp.layers = nn.ModuleList(
            [
                LongcatFlashMTPDecoderLayer(config, self.runner_settings, layer_idx, \
                                            prefix=f"model.mtp.layers.{i}", **kwargs)
                for i in range(config.num_nextn_predict_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_len: torch.IntTensor,
        actual_seq_lengths_kv: list,
        cos_sin: torch.Tensor,
        past_residual: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        is_prefill: Optional[bool] = False,
        slot_mapping: Optional[torch.Tensor] = None,
        mtp_layer_idx: Optional[int] = 0,
        input_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.mtp.layers[mtp_layer_idx](
            hidden_states,
            kv_len,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            cos_sin=cos_sin,
            past_residual=past_residual,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            is_prefill=is_prefill,
            slot_mapping=slot_mapping,
            mtp_layer_idx=mtp_layer_idx,
            input_ids=input_ids
        )


class LongcatFlashModelMTP(LongcatFlashForCausalLM):
    def __init__(self, config: LongcatFlashConfig, runner_settings: Dict, **kwargs):
        super().__init__(config, runner_settings, is_mtp=True)
        self.global_rank = dist.get_rank()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.vocab_size_per_rank = self.vocab_size // self.embed_tp_size
        self.ignore_share_weight = False
        self.pa_max_length = self.runner_settings.get("model_config").get("pa_max_length", 2048)

        mtp_kwargs = {}
        mtp_kwargs.update({"kv_len_offset": self.kv_len_offset})
        if hasattr(self, "hccl_comm_dict"):
            mtp_kwargs.update({"hccl_comm_dict": self.hccl_comm_dict})
        self.mtp_layer_idx = config.num_hidden_layers * 2 # MTP is the last layer
        self.model = LongcatFlashMTPLayer(config, self.runner_settings, self.mtp_layer_idx, **mtp_kwargs)

        # no reuse
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            self.padding_idx,
            torch.bfloat16,
            tp_size=self.embed_tp_size,
            tp_rank=dist.get_rank(self.hccl_comm_dict["embed_tp_group"]) if self.embed_tp_size > 1 else 0)

        # reuse embed_tokens, lm_head, rotary_emb from main model
        self.lm_head = None
        self.rotary_emb = None

        self.norm = LongcatFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.enorm = LongcatFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = LongcatFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # prev_hidden_states and input_hidden_state feature fusion
        self.eh_proj = ReplicatedLinear(2 * config.hidden_size, config.hidden_size, bias=False)

    def get_slot_mapping(self, input_ids, kv_len, is_prefill, device):
        batch_size, seq_len = input_ids.size()            
        if not is_prefill:
            kv_len = kv_len.view(batch_size * self.attn_tp_size, -1) + \
                self.kv_len_offset[:batch_size * self.attn_tp_size]
            return kv_len
        else:
            all_tensors = []        
            for i in range(batch_size * self.attn_tp_size):
                new_index = torch.arange(self.pa_max_length * i, seq_len + self.pa_max_length * i,
                                        dtype=kv_len.dtype, device=device)
                all_tensors.append(new_index)
            return torch.cat(all_tensors)

    @override
    def forward(
        self,
        input_ids: torch.LongTensor,
        kv_len: torch.IntTensor = None,
        actual_seq_lengths_kv: list = None,
        prev_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        is_prefill: Optional[bool] = False,
        cur_topk_list: Optional[torch.Tensor] = None
    ):
        batch_size, seq_length = input_ids.shape
        if self.embed_tp_size > 1:
            embed_tp_group = self.hccl_comm_dict.get("embed_tp_group", None)
            all_input_ids = input_ids.new_empty(batch_size * self.embed_tp_size, seq_length)
            dist.all_gather_into_tensor(all_input_ids, input_ids.contiguous(), group=embed_tp_group)

            new_input_ids = all_input_ids - (self.global_rank % self.embed_tp_size) * self.vocab_size_per_rank
            mask = (new_input_ids >= 0) & (new_input_ids < self.vocab_size_per_rank) # (bs, qlen)
            new_input_ids_per_rank = new_input_ids * mask
            inputs_embeds = self.embed_tokens(new_input_ids_per_rank) * mask.unsqueeze(-1)

            inputs_embeds_attn = inputs_embeds.new_empty(batch_size, seq_length, inputs_embeds.shape[-1])
            dist.reduce_scatter_tensor(inputs_embeds_attn, inputs_embeds, group=embed_tp_group)
            inputs_embeds = inputs_embeds_attn
        else:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        hidden_states = self.enorm(hidden_states)
        prev_hidden_states = self.hnorm(prev_hidden_states)
        hidden_states_eh = torch.cat([hidden_states, prev_hidden_states], dim=-1)
        hidden_states = self.eh_proj(hidden_states_eh)

        cos_sin = self.rotary_emb(hidden_states, kv_len, self.config.max_position_embeddings, \
                                  is_prefill=is_prefill, attn_tp_size=self.attn_tp_size)
        residual = None
        slot_mapping = self.get_slot_mapping(input_ids, kv_len, is_prefill, position_ids.device)

        hidden_states = self.model(
            hidden_states,
            kv_len,
            actual_seq_lengths_kv,
            input_ids=input_ids,
            cos_sin=cos_sin,
            past_residual=residual,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            is_prefill=is_prefill,
            cur_topk_list=cur_topk_list,
            slot_mapping=slot_mapping
        )

        prev_hidden_states = self.norm(hidden_states)

        outputs = prev_hidden_states
        logits = self.forward_lm_head(outputs=outputs, position_ids=position_ids, is_prefill=is_prefill)

        return logits, prev_hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        mlp_params_mapping, mtp_unique_weight_mapping = self._load_weight_map()

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            is_main = is_main_weight(self.config, name)
            if is_main:
                continue

            unique_weight_load = False
            for (param_name, weight_name) in mtp_unique_weight_mapping:
                if weight_name not in name:
                    continue
                param = params_dict[param_name + ".weight"]
                weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                weight_loader(param, loaded_weight)
                unique_weight_load = True
                loaded_params.add(param_name + ".weight")
            if unique_weight_load:
                continue

            for (param_name, weight_name, shard_id) in mlp_params_mapping:
                # Skip non-stacked layers and experts
                if weight_name not in name:
                    continue
                # no moe but dense
                if (("mlp.experts." in name) and name not in params_dict):
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                if "down_proj" in name:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # self_attn and norm
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def _load_weight_map(self):
        mlp_params_mapping = [
            # (param_name, shard_name, shard_id), reduce module in module
            ("mlp.gate_up_proj", "transformer_layer.mlp.gate_proj", 0),
            ("mlp.gate_up_proj", "transformer_layer.mlp.up_proj", 1),
            ("mlp.down_proj", "transformer_layer.mlp.down_proj", 0),
        ]

        mtp_unique_weight_mapping = [
            # (param_name, weight_name)
            ("embed_tokens", "mtp.embed_tokens"),
            ("enorm", "enorm"),
            ("hnorm", "hnorm"),
            ("norm", "mtp.norm"),
            ("eh_proj", "eh_proj")
        ]

        return mlp_params_mapping, mtp_unique_weight_mapping


def is_main_weight(config, weight_name: str) -> Optional[int]:
    if hasattr(config,
               "num_nextn_predict_layers") and (config.num_nextn_predict_layers
                                                > 0):
        if "model.mtp" in weight_name:
            return False
    return True



__all__ = ["LongcatFlashPreTrainedModel", "LongcatFlashModel", "LongcatFlashForCausalLM"]

