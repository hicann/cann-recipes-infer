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

from typing import Iterable, Optional, Tuple, Dict, Set
from operator import attrgetter
import torch
import npugraph_ex
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch_npu
import torchair as tng

from transformers.integrations import use_kernel_forward_from_hub
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import logging

from executor.utils import align_up, calc_moe_hccl_buffer_size, limit_core_num, superkernel_scope, npu_prefetch
from executor.utils.stream_utils import (npu_stream_switch, npu_stream_switch_gegraph, record_event,
                                         record_stream, wait_event)
from module.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding
    )
from module.fuse_moe_gmm import FusedMoEGMM
from executor.model_loader.weight_utils import default_weight_loader
from executor.utils.forward_metadata import ForwardMetaData
from executor.core.config import InferenceConfig, CommManager
from executor.core.kv_cache.cache_info import CacheEntry, LayerCacheInfo, ModelCacheInfo
from module.quantization import QuantizeMethodBase
from module.quantization.compressed_tensors.compressed_tensors_moe_gmm import CompressedTensorW8A8Int8MoEGMMMethod

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

    def forward(self, x, kv_len, max_seq_len=None, position_ids=None):
        if max_seq_len is None:
            self._set_cos_sin_cache(seq_len=kv_len, device=x.device, dtype=x.dtype)
        elif max_seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=max_seq_len, device=x.device, dtype=x.dtype)

        position_ids = position_ids.to(dtype=torch.long, device=x.device)
        cos = self.cos_cached[position_ids].unsqueeze(1)
        sin = self.sin_cached[position_ids].unsqueeze(1)

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


def _trim_attn_sp_padding(tensor, valid_tokens: int, local_num_tokens: int):
    if valid_tokens >= local_num_tokens:
        return tensor
    if valid_tokens <= 0:
        return torch.zeros_like(tensor)
    return torch.cat([tensor[:valid_tokens], torch.zeros_like(tensor[valid_tokens:])], dim=0)


class LongcatFlashMLP(nn.Module):
    def __init__(self, config, infer_config: InferenceConfig, comm_manager: CommManager, prefix, hidden_size=None,
                 intermediate_size=None, **kwargs):
        super().__init__()
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.mm_quant_mode = (
            config.quant_config.mm_quant_mode
            if config.quant_config is not None
            else "w16a16")
        self.dense_tp_size = self.infer_config.parallel_config.dense_tp_size
        self.attn_tp_size = self.infer_config.parallel_config.attn_tp_size
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.ffn_hidden_size if intermediate_size is None else intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[self.intermediate_size] * 2,
            bias=False,
            tp_size=self.dense_tp_size,
            tp_rank=self.comm_manager.get_rank("dense_tp_group") if self.dense_tp_size > 1 else 0,
            quant_config=config.quant_config,
            prefix=f"{prefix}.gate_up_proj"
            )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            config.hidden_size,
            bias=False,
            tp_size=self.dense_tp_size,
            tp_rank=self.comm_manager.get_rank("dense_tp_group") if self.dense_tp_size > 1 else 0,
            quant_config=config.quant_config,
            prefix=f"{prefix}.down_proj")
        dtype_bit = 1 if self.mm_quant_mode == "w8a8int8" else 2  # int8: 1bit, bf16: 2bit
        self.up_gate_prefetch_size = self.hidden_size * self.intermediate_size * 2 * dtype_bit // self.dense_tp_size
        self.down_prefetch_size = self.hidden_size * self.intermediate_size * dtype_bit // self.dense_tp_size
        if self.mm_quant_mode == "w8a8int8":
            self.mlp_forward = self.forward_w8a8int8
        else:
            self.mlp_forward = self.forward_normal
        self.npugraph_prefetch_stream = None

    def forward(self, x, is_prefill=False):
        return self._forward_impl(x, is_prefill=is_prefill)

    def forward_prefetch(self, x, o_proj, is_prefill=False, record_gateup_event=False, record_down_event=False):
        route_prefetch = self.npugraph_prefetch_stream is not None
        with npu_stream_switch(route_prefetch, self.npugraph_prefetch_stream):
            npu_prefetch(True, self.gate_up_proj.weight.data, o_proj, self.up_gate_prefetch_size, 0)
        return self._forward_impl(x, is_prefill=is_prefill, enable_down_prefetch=True,
                                  record_gateup_event=record_gateup_event, record_down_event=record_down_event)

    def _forward_impl(self, x, is_prefill=False, enable_down_prefetch=False,
                      record_gateup_event=False, record_down_event=False):
        route_prefetch = enable_down_prefetch and self.npugraph_prefetch_stream is not None
        use_dense_all_reduce = self.dense_tp_size > 1 and self.attn_tp_size > 1 and not is_prefill
        if self.dense_tp_size > 1 and not use_dense_all_reduce:
            num_tokens = x.shape[0]
            x_output = torch.empty([num_tokens * self.dense_tp_size, self.hidden_size],
                                    dtype=x.dtype, device=x.device)
            dist.all_gather_into_tensor(x_output, x.contiguous(),
                                        group=self.comm_manager.get_group("dense_tp_group"))
            x = x_output
        x_event = None
        if route_prefetch and self.mm_quant_mode == "w8a8int8":
            x_event = torch.npu.current_stream().record_event()

        down_proj, dsq, gateup_event, down_event = self.mlp_forward(
            x, enable_down_prefetch, x_event,
            record_gateup_event, record_down_event)

        if use_dense_all_reduce:
            dist.all_reduce(down_proj, group=self.comm_manager.get_group("dense_tp_group"))
            mlp_res = down_proj
        elif self.dense_tp_size > 1:
            mlp_res = down_proj.new_empty(num_tokens, down_proj.shape[-1])
            dist.reduce_scatter_tensor(mlp_res, down_proj, group=self.comm_manager.get_group("dense_tp_group"))
        else:
            mlp_res = down_proj

        if record_gateup_event or record_down_event:
            return mlp_res, down_proj, dsq, gateup_event, down_event
        return mlp_res, down_proj, dsq

    def forward_normal(self, x, enable_prefetch, x_event=None,
                       record_gateup_event=False, record_down_event=False):
        route_prefetch = enable_prefetch and self.npugraph_prefetch_stream is not None
        merged_x = self.gate_up_proj(x)
        gateup_event = None
        if route_prefetch and record_gateup_event:
            gateup_event = torch.npu.current_stream().record_event()
        intermediate_hidden_states = torch_npu.npu_swiglu(merged_x)
        down_out = self.down_proj(intermediate_hidden_states)
        down_event = None
        if route_prefetch and record_down_event:
            down_event = torch.npu.current_stream().record_event()
        return down_out, merged_x, gateup_event, down_event

    def forward_w8a8int8(self, x, enable_prefetch, x_event=None,
                         record_gateup_event=False, record_down_event=False):
        route_prefetch = enable_prefetch and self.npugraph_prefetch_stream is not None
        with npu_stream_switch(route_prefetch, self.npugraph_prefetch_stream):
            if route_prefetch and x_event is not None:
                self.npugraph_prefetch_stream.wait_event(x_event)
            npu_prefetch(enable_prefetch, self.down_proj.weight.data, x, self.down_prefetch_size, 0)
        merged_x, pertoken_scale = self.gate_up_proj(x, out_dtype=torch.int32)
        gateup_event = None
        if route_prefetch and record_gateup_event:
            gateup_event = torch.npu.current_stream().record_event()
        intermediate_hidden_states, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
            merged_x, weight_scale=self.gate_up_proj.weight_scale,
            quant_scale=self.down_proj.smooth_scales,
            quant_mode=1, activate_left=True,
            activation_scale=pertoken_scale
        )
        down_out = self.down_proj(intermediate_hidden_states, pertoken_scale)
        down_event = None
        if route_prefetch and record_down_event:
            down_event = torch.npu.current_stream().record_event()
        return down_out, merged_x, gateup_event, down_event


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
        self.prefetch_size = self.config.hidden_size * self.num_experts * 4 # 4: float32 weight
        # register_buffer not in named_parameters()
        self.e_score_correction_bias = nn.Parameter(
                torch.empty((self.num_experts), dtype=torch.float32)
            )

    @torch.no_grad()
    def get_topk_indices(self, scores):
        scores_for_choice = scores.view(-1, self.num_experts) + self.e_score_correction_bias.unsqueeze(0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return topk_indices

    def forward(self, hidden_states, prefetch_stream=None, prefetch_weight=None,
                prefetch_size=0, enable_prefetch=False):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = self.classifier(hidden_states.type(torch.float32))
        route_prefetch = enable_prefetch and prefetch_stream is not None and prefetch_weight is not None
        if route_prefetch:
            router_event = torch.npu.current_stream().record_event()
            with npu_stream_switch(True, prefetch_stream):
                prefetch_stream.wait_event(router_event)
                npu_prefetch(enable_prefetch, prefetch_weight, router_logits, prefetch_size, 0)
        topk_weights, topk_indices, _ = torch_npu.npu_moe_gating_top_k(
                router_logits,
                k=self.top_k,
                bias=self.e_score_correction_bias.float(),
                renorm=0,  # 0: softmax->topk; 1: topk->softmax
                norm_type=0,  # 0: softmax; 1: sigmoid
                routed_scaling_factor=self.routed_scaling_factor,
                eps=float(1e-20)
            )
        return topk_indices.to(torch.int32), topk_weights, router_logits


class LongcatFlashMoE(nn.Module):
    """
    moe module.
    """

    def __init__(self, config, infer_config: InferenceConfig, comm_manager: CommManager, layer_idx, prefix, **kwargs):
        super().__init__()
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        custom_params = self.infer_config.model_config.custom_params
        self.gmm_quant_mode = (
            config.quant_config.gmm_quant_mode
            if config.quant_config is not None
            else "w16a16")
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.expert_ffn_hidden_size
        self.zero_expert_num = config.zero_expert_num
        self.zero_expert_type = config.zero_expert_type
        self.moe_tp_size = self.infer_config.parallel_config.moe_tp_size
        self.moe_ep_size = self.infer_config.parallel_config.moe_ep_size
        self.moe_chunk_max_len = custom_params.get("moe_chunk_max_len", 65536)
        self.enable_multi_streams = custom_params.get("enable_multi_streams", 0)

        self.moe_ep_group = self.comm_manager.get_group("moe_ep_group") if self.moe_ep_size > 1 else None

        self.n_routed_experts = config.n_routed_experts
        self.num_zero_experts = config.zero_expert_num
        self.num_experts = self.n_routed_experts + self.num_zero_experts
        self.experts_per_rank = self.num_experts // self.moe_ep_size
        self.global_rank = dist.get_rank()
        self.n_routed_experts_per_rank = self.n_routed_experts // self.moe_ep_size
        self.router = LongcatFlashTopkRouter(config, f"{prefix}.router")
        self.force_eplb = self.infer_config.model_config.force_eplb
        self.experts = FusedMoEGMM(
            num_experts=config.n_routed_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            bias=False,
            quant_config=config.quant_config,
            tp_size=self.moe_tp_size,
            tp_rank=self.comm_manager.get_rank("moe_tp_group") if self.moe_tp_size > 1 else 0,
            ep_size=self.moe_ep_size,
            ep_rank=self.comm_manager.get_rank("moe_ep_group") if self.moe_ep_size > 1 else 0,
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
        bs_qlen, h = x.shape
        x = x.view(-1, h)
        hidden_states_list = []
        for hidden_states, topk_ids, topk_weight in zip(
                *self._split_tensors(bs_qlen, x, topk_ids, topk_weight)):
            bs_qlen = hidden_states.shape[0]
            moe_input = hidden_states
            routed_mask = topk_ids < self.n_routed_experts
            routed_topk_ids = topk_ids * routed_mask.to(topk_ids.dtype)
            routed_topk_weight = topk_weight * routed_mask.to(topk_weight.dtype)
            identity_weight = (topk_weight * (~routed_mask).to(topk_weight.dtype)).sum(dim=1, keepdim=True)
            expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = torch_npu.npu_moe_init_routing_v2(
                hidden_states,
                expert_idx=routed_topk_ids,
                active_num=routed_topk_ids.shape[0] * routed_topk_ids.shape[1],
                scale=self.experts.smooth_scale_1 if "a8" in self.gmm_quant_mode else None,
                expert_num=self.n_routed_experts,
                expert_tokens_num_type=1,  # 0: cumsum mode(not supported now); 1: count mode
                expert_tokens_num_flag=True,
                active_expert_range=[0, self.n_routed_experts],
                quant_mode=1 if "a8" in self.gmm_quant_mode else -1
                # -1: non-quant; 1: dynamic quant; 0: static quant(not supported now)
            )

            tokens_per_expert_group, gathered_tokens, gathered_pertoken_scale, input_splits, output_splits =\
                self.dispatch_double_routing(tokens_per_expert, expanded_x, pertoken_scale)

            new_x = self.forward_expert(gathered_tokens, tokens_per_expert_group, gathered_pertoken_scale)

            gathered_tokens = self.forward_combine_double_routing(new_x, expanded_x, input_splits, output_splits)
            hidden_states = torch_npu.npu_moe_finalize_routing(
                gathered_tokens, skip1=None, skip2=None, bias=None,
                scales=routed_topk_weight.to(gathered_tokens.dtype),
                expanded_src_to_dst_row=expanded_row_idx,
                export_for_source_row=None, drop_pad_mode=2
            )

            hidden_states = hidden_states + moe_input * identity_weight.to(hidden_states.dtype)
            hidden_states = hidden_states.view(bs_qlen, self.hidden_size)
            hidden_states_list.append(hidden_states)

        hidden_states = torch.cat(hidden_states_list, dim=0) if len(hidden_states_list) > 1 else hidden_states_list[0]
        return hidden_states.view(-1, h)

    def set_mc2_kwargs(self):
        global_rank = dist.get_rank()
        mc2_group_name = self.comm_manager.get_group_name("moe_ep_group_mc2")
        self.dispatch_kwargs = {
                "x_active_mask": None,
                "expert_shard_type": 0,
                "shared_expert_rank_num": 0,
                "moe_expert_num": self.n_routed_experts,
                "copy_expert_num": self.num_zero_experts,
                "global_bs": 0,
                "scales": self.experts.smooth_scale_1 if "a8" in self.gmm_quant_mode else None,
                "quant_mode": 2 if "a8" in self.gmm_quant_mode else 0,
                "group_ep": mc2_group_name,
                "ep_world_size": self.moe_ep_size,
                "ep_rank_id": global_rank // self.moe_tp_size,
                "group_tp": mc2_group_name,
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
                "group_ep": mc2_group_name,
                "ep_world_size": self.moe_ep_size,
                "ep_rank_id": global_rank // self.moe_tp_size,
                "group_tp": mc2_group_name,
                "tp_world_size": self.moe_tp_size,
                "tp_rank_id": global_rank % self.moe_tp_size
            }

    def moe_infer_dispatch_combine(self, x, topk_ids, topk_weight, record_gmm2_event=False):
        """
        tp+ep mix strategy, for decode stage
        """
        bs_qlen, h = x.shape
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
        gmm2_event = None
        if record_gmm2_event:
            gmm2_event = torch.npu.current_stream().record_event()

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

        hidden_states = hidden_states.view(bs_qlen, self.hidden_size)
        if record_gmm2_event:
            return hidden_states, hidden_states_ordered_by_experts, gmm2_event
        return hidden_states, hidden_states_ordered_by_experts

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
        if self.force_eplb:
            topk_indices = cur_topk_list
        topk_indices = topk_indices.to(torch.int32)

        if is_prefill:
            return self.moe_infer_double_routing(hidden_states, topk_indices, topk_weights)
        else:
            return self.moe_infer_dispatch_combine(hidden_states, topk_indices, topk_weights)[0]


class LongcatFlashAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LongcatFlashConfig, infer_config: InferenceConfig, comm_manager: CommManager,
                 layer_idx: Optional[int] = None, prefix: Optional[str] = "", **kwargs):
        super().__init__()
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.attn_type = "FullAttention"
        custom_params = self.infer_config.model_config.custom_params
        self.attn_tp_size = self.infer_config.parallel_config.attn_tp_size
        self.o_proj_tp_size = self.infer_config.parallel_config.o_proj_tp_size
        self.moe_tp_size = self.infer_config.parallel_config.moe_tp_size
        self.moe_ep_size = self.infer_config.parallel_config.moe_ep_size
        self.next_n = self.infer_config.model_config.next_n
        self.enable_afd = custom_params.get("enable_afd", False)
        self.use_attn_sp = (
            self.attn_tp_size > 1
            and self.moe_ep_size > 1
        )
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

        if self.q_lora_rank is None:
            self.q_proj = ColumnParallelLinear(self.hidden_size,
                                               self.num_heads * self.qk_head_dim,
                                               bias=False,
                                               quant_config=config.quant_config,
                                               tp_size=self.attn_tp_size,
                                               tp_rank=self.comm_manager.get_rank("attn_tp_group")
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
                                                 tp_rank=self.comm_manager.get_rank("attn_tp_group")
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
            tp_rank=self.comm_manager.get_rank("attn_tp_group") if self.attn_tp_size > 1 else 0,
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
                                        tp_rank=self.comm_manager.get_rank("o_proj_tp_group")
                                        if self.o_proj_tp_size > 1 else 0,
                                        bias=False,
                                        input_is_parallel=True,
                                        quant_config=config.quant_config,
                                        prefix=f"{prefix}.o_proj")

        self.mla_scale_q_lora = None
        self.mla_scale_kv_lora = None
        if config.mla_scale_q_lora:
            self.mla_scale_q_lora = (self.hidden_size / self.q_lora_rank) ** 0.5
        if config.mla_scale_kv_lora:
            self.mla_scale_kv_lora = (self.hidden_size / self.kv_lora_rank) ** 0.5
        self.softmax_scale = self.qk_head_dim ** (-0.5)

        self.block_size = self.infer_config.scheduler_config.block_size
        cache_dtype = getattr(config, "torch_dtype", None) or torch.bfloat16
        self.cache_nope = torch.Tensor([])
        self.cache_rope = torch.Tensor([])
        self.cache_entries = [
            CacheEntry(
                cache_name="cache_nope",
                attn_type=self.attn_type,
                dim=self.kv_lora_rank,
                num_head=1,
                dtype=cache_dtype,
                block_size=self.block_size,
                needs_block=True,
                tensor_setter=lambda tensor, layer=self: setattr(layer, "cache_nope", tensor),
            ),
            CacheEntry(
                cache_name="cache_rope",
                attn_type=self.attn_type,
                dim=self.qk_rope_head_dim,
                num_head=1,
                dtype=cache_dtype,
                block_size=self.block_size,
                needs_block=True,
                tensor_setter=lambda tensor, layer=self: setattr(layer, "cache_rope", tensor),
            ),
        ]
        self.kv_scale = None
        self.enable_gegraph = self.infer_config.model_config.exe_mode == "ge_graph"
        self.enable_mla_prolog = custom_params.get("enable_mla_prolog", False)
        self.enable_mla_prolog = (
            self.enable_mla_prolog
            and self.q_lora_rank is not None
        )
        self.fa_ops = torch.ops.npu
        if self.enable_gegraph:
            self.fa_ops = tng.ops
        self.attn_tp_group = self.comm_manager.get_group("attn_tp_group") if self.attn_tp_size > 1 else None

    def fold_mla_kv_lora_scale_into_norm(self):
        if self.mla_scale_kv_lora is None or getattr(self, "mla_kv_lora_scale_folded", False):
            return
        with torch.no_grad():
            self.kv_a_layernorm.weight.mul_(self.mla_scale_kv_lora)
        self.mla_scale_kv_lora = None
        self.mla_kv_lora_scale_folded = True

    def _prepare_attn_sp_inputs(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ):
        if not self.use_attn_sp:
            return hidden_states, position_embeddings, hidden_states.shape[0]

        sp_padded_num_tokens = hidden_states.shape[0] * self.attn_tp_size
        new_hidden_states = hidden_states.new_empty(
            (sp_padded_num_tokens, hidden_states.shape[-1])
        )
        dist.all_gather_into_tensor(
            new_hidden_states,
            hidden_states.contiguous(),
            group=self.attn_tp_group,
        )
        cos, sin = position_embeddings
        seq_length_unpad = cos.shape[0]
        new_hidden_states = new_hidden_states[:seq_length_unpad]
        position_embeddings = (cos[:seq_length_unpad], sin[:seq_length_unpad])
        return new_hidden_states, position_embeddings, sp_padded_num_tokens

    def o_proj_forward(
        self,
        attn_output: torch.Tensor,
        record_prefetch_event=False,
        sp_padded_num_tokens: Optional[int] = None,
    ):
        num_tokens = attn_output.shape[0]
        if self.o_proj_tp_size > 1 and self.attn_tp_size == 1:
            attn_output = attn_output.view(num_tokens, self.o_proj_tp_size, -1).transpose(1, 0).contiguous().view(-1)
            all2all_output = torch.empty_like(attn_output)
            dist.all_to_all_single(all2all_output, attn_output,
                                   group=self.comm_manager.get_group("o_proj_tp_group"))
            attn_output = all2all_output

        attn_output = self.o_proj(attn_output.view(-1, self.num_heads // self.o_proj_tp_size * self.v_head_dim))
        o_proj = attn_output
        prefetch_event = None
        if record_prefetch_event:
            prefetch_event = torch.npu.current_stream().record_event()
        if self.o_proj_tp_size > 1:
            if self.use_attn_sp and sp_padded_num_tokens is not None:
                padding_size = sp_padded_num_tokens - attn_output.shape[0]
                if padding_size > 0:
                    attn_output = F.pad(attn_output, (0, 0, 0, padding_size), value=0)
                reduce_scatter_output = torch.empty(
                    (sp_padded_num_tokens // self.o_proj_tp_size, attn_output.size()[1]),
                    dtype=attn_output.dtype, device=attn_output.device
                )
                dist.reduce_scatter_tensor(reduce_scatter_output, attn_output,
                                           group=self.comm_manager.get_group("o_proj_tp_group"))
                attn_output = reduce_scatter_output
            elif self.attn_tp_size > 1:
                dist.all_reduce(attn_output, group=self.attn_tp_group)
            else:
                reduce_scatter_output = torch.empty(
                    (attn_output.size()[0] // self.o_proj_tp_size, attn_output.size()[1]),
                    dtype=attn_output.dtype, device=attn_output.device
                )
                dist.reduce_scatter_tensor(reduce_scatter_output, attn_output,
                                           group=self.comm_manager.get_group("o_proj_tp_group"))
                attn_output = reduce_scatter_output

        if record_prefetch_event:
            return attn_output.view(attn_output.shape[0], -1), o_proj, prefetch_event
        return attn_output.view(attn_output.shape[0], -1), o_proj

    def forward_page_attention_normal(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        kv_len: torch.IntTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        actual_seq_lengths_kv: list = None,
        actual_seq_lengths_q: list = None,
        slot_mapping: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
    ):
        sp_padded_num_tokens = None
        if self.use_attn_sp:
            hidden_states, position_embeddings, sp_padded_num_tokens = self._prepare_attn_sp_inputs(
                hidden_states,
                position_embeddings,
            )
        num_tokens = hidden_states.shape[0]
        cos, sin = position_embeddings

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q_hidden_states = self.q_a_layernorm(self.q_a_proj(hidden_states))
            q = self.q_b_proj(q_hidden_states)

        latent_cache = self.kv_a_proj_with_mqa(hidden_states)
        num_q_tokens = q.shape[0]
        q = q.view(1, -1, self.num_heads_per_rank, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        if self.mla_scale_q_lora is not None:
            q_nope = q_nope * self.mla_scale_q_lora
            q_pe = q_pe * self.mla_scale_q_lora

        q_pe = q_pe.transpose(1, 2)
        cos = cos.view(1, 1, -1, self.qk_rope_head_dim)
        sin = sin.view(1, 1, -1, self.qk_rope_head_dim)
        q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin)
        q_pe = q_pe.view(1, self.num_heads_per_rank, -1, self.qk_rope_head_dim).transpose(1, 2)
        query_states = [q_nope, q_pe]

        latent_cache = latent_cache.view(-1, 1, 1, self.kv_lora_rank + self.qk_rope_head_dim)
        nope_cache = self.cache_nope
        rope_cache = self.cache_rope
        if not self.cache_nope.numel() or not self.cache_rope.numel():
            raise ValueError("kv cache is not initialized properly.")
        cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)
        sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)

        k_rope, k_nope = torch_npu.npu_kv_rmsnorm_rope_cache_v2(
            latent_cache,
            self.kv_a_layernorm.weight,
            cos,
            sin,
            slot_mapping.view(-1),
            rope_cache,
            nope_cache,
            epsilon=self.kv_a_layernorm.variance_epsilon,
            cache_mode="PA_NZ",
            is_output_kv=True,
        )

        if self.mla_scale_kv_lora is not None:
            k_nope = k_nope * self.mla_scale_kv_lora

        k_nope_out = torch.matmul(k_nope.view(1, -1, self.kv_lora_rank), self.kv_b_proj_w_k.permute(0, 2, 1))
        v_out = torch.matmul(k_nope.view(1, -1, self.kv_lora_rank), self.kv_b_proj_w_v)
        k_rope = k_rope.view(1, -1, self.qk_rope_head_dim).repeat(self.num_heads_per_rank, 1, 1)

        heads = self.num_heads_per_rank
        attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            query_states[0].flatten(0, 1).permute(1, 0, 2),
            k_nope_out,
            v_out,
            query_rope=query_states[1].flatten(0, 1).permute(1, 0, 2),
            key_rope=k_rope,
            num_heads=heads,
            num_key_value_heads=heads,
            input_layout="NTD_TND",
            atten_mask=attention_mask,
            sparse_mode=3,
            actual_seq_lengths=actual_seq_lengths_kv,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            scale=self.softmax_scale,
            antiquant_mode=0,
            antiquant_scale=None,
            next_tokens=0,
        )
        attn_output, _ = self.o_proj_forward(
            attn_output.reshape(num_q_tokens, -1),
            sp_padded_num_tokens=sp_padded_num_tokens,
        )
        return attn_output

    def prepare_qkv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        slot_mapping: Optional[torch.Tensor] = None,
    ):
        num_tokens = hidden_states.shape[0]

        input_kwargs = {
            "hidden_states": hidden_states,
            "position_embeddings": position_embeddings,
            "slot_mapping": slot_mapping,
        }
        if self.enable_mla_prolog:
            fn = self.mla_prolog
        else:
            fn = self.prepare_qkv_absorb
        query_states, k_nope, k_rope = fn(**input_kwargs)
        return query_states, k_nope, k_rope, num_tokens

    def mla_prolog(self, hidden_states: torch.Tensor, position_embeddings: torch.Tensor = None,
                   slot_mapping: torch.IntTensor = None):
        cos, sin = position_embeddings

        cos = cos.view(-1, self.qk_rope_head_dim)
        sin = sin.view(-1, self.qk_rope_head_dim)
        nope_cache = self.cache_nope
        rope_cache = self.cache_rope
        if not self.cache_nope.numel() or not self.cache_rope.numel():
            raise ValueError("kv cache is not initialized properly.")
        block_num, block_size, _, _ = nope_cache.size()

        q_nope, q_pe, _, _, _ = torch.ops.npu.npu_mla_prolog_v3(
            token_x=hidden_states,
            weight_dq=self.q_a_proj.weight,
            weight_uq_qr=self.q_b_proj.weight,
            weight_uk=self.kv_b_proj_w_k,
            weight_dkv_kr=self.kv_a_proj_with_mqa.weight,
            rmsnorm_gamma_cq=self.q_a_layernorm.weight,
            rmsnorm_gamma_ckv=self.kv_a_layernorm.weight,
            rope_sin=sin,
            rope_cos=cos,
            cache_index=slot_mapping,
            kv_cache=nope_cache,
            kr_cache=rope_cache,
            rmsnorm_epsilon_cq=self.q_a_layernorm.variance_epsilon,
            rmsnorm_epsilon_ckv=self.kv_a_layernorm.variance_epsilon,
            cache_mode="PA_NZ",
            qc_qr_scale=self.mla_scale_q_lora,
        )

        kv_cache_nz_dim = 16
        k_nope = nope_cache.view(block_num, 1, self.kv_lora_rank // kv_cache_nz_dim, block_size, kv_cache_nz_dim)
        k_rope = rope_cache.view(block_num, 1, self.qk_rope_head_dim // kv_cache_nz_dim, block_size, kv_cache_nz_dim)
        if self.mla_scale_kv_lora is not None:
            k_nope = k_nope * self.mla_scale_kv_lora

        query_states = [q_nope, q_pe]
        return query_states, k_nope, k_rope

    def prepare_qkv_absorb(self, hidden_states: torch.Tensor, position_embeddings: torch.Tensor = None,
                           slot_mapping: torch.IntTensor = None):
        num_tokens = hidden_states.shape[0]
        cos, sin = position_embeddings

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))

        latent_cache = self.kv_a_proj_with_mqa(hidden_states)
        q = q.view(num_tokens, self.num_heads_per_rank, self.qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        if self.mla_scale_q_lora is not None:
            q_nope = q_nope * self.mla_scale_q_lora
            q_pe = q_pe * self.mla_scale_q_lora

        if self.kv_b_proj_w_k.shape[0] * self.kv_b_proj_w_k.shape[1] <= 65535:
            q_nope = torch_npu.npu_transpose_batchmatmul(
                q_nope,
                self.kv_b_proj_w_k,
                bias=None,
                scale=None,
                perm_x1=(1, 0, 2),
                perm_x2=(0, 1, 2),
                perm_y=(1, 0, 2),
            )
            q_nope = q_nope.view(num_tokens, self.num_heads_per_rank, self.kv_lora_rank)
        else:
            q_nope = (
                torch.matmul(q_nope.transpose(0, 1), self.kv_b_proj_w_k)
                .transpose(0, 1)
                .view(num_tokens, self.num_heads_per_rank, self.kv_lora_rank)
            )

        q_pe = q_pe.unsqueeze(0).transpose(1, 2)
        cos = cos.view(1, 1, -1, self.qk_rope_head_dim)
        sin = sin.view(1, 1, -1, self.qk_rope_head_dim)
        q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin)
        q_pe = q_pe.view(1, self.num_heads_per_rank, -1, self.qk_rope_head_dim).transpose(1, 2)
        q_pe = q_pe.view(num_tokens, self.num_heads_per_rank, self.qk_rope_head_dim)

        latent_cache = latent_cache.view(num_tokens, 1, 1, self.kv_lora_rank + self.qk_rope_head_dim)
        cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)
        sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)
        nope_cache = self.cache_nope
        rope_cache = self.cache_rope
        if not self.cache_nope.numel() or not self.cache_rope.numel():
            raise ValueError("kv cache is not initialized properly.")
        block_num, block_size, _, _ = nope_cache.size()

        torch_npu.npu_kv_rmsnorm_rope_cache_v2(
            latent_cache,
            self.kv_a_layernorm.weight,
            cos,
            sin,
            slot_mapping.view(-1),
            rope_cache,
            nope_cache,
            epsilon=self.kv_a_layernorm.variance_epsilon,
            cache_mode="PA_NZ",
        )

        if self.mla_scale_kv_lora is not None:
            nope_cache = nope_cache * self.mla_scale_kv_lora

        kv_cache_nz_dim = 16
        k_nope = nope_cache.view(block_num, 1, self.kv_lora_rank // kv_cache_nz_dim, block_size, kv_cache_nz_dim)
        k_rope = rope_cache.view(block_num, 1, self.qk_rope_head_dim // kv_cache_nz_dim, block_size, kv_cache_nz_dim)

        query_states = [q_nope, q_pe]
        return query_states, k_nope, k_rope

    def fused_infer_attention_score(
        self,
        query_states: torch.Tensor,
        k_nope: torch.Tensor,
        k_rope: torch.Tensor,
        kv_len: torch.IntTensor = None,
        num_tokens: int = None,
        attention_mask: Optional[torch.Tensor] = None,
        actual_seq_lengths_kv: list = None,
        actual_seq_lengths_q: list = None,
        block_table: Optional[torch.Tensor] = None,
        record_prefetch_event=False,
    ):
        q_len = num_tokens // kv_len.shape[0]
        if q_len > 1:
            sparse_mode = 3
        else:
            sparse_mode = 0
            attention_mask = None

        attn_output, _ = self.fa_ops.npu_fused_infer_attention_score_v2(
            query_states[0], k_nope, k_nope,
            query_rope=query_states[1], key_rope=k_rope,
            atten_mask=attention_mask,
            actual_seq_kvlen=actual_seq_lengths_kv,
            actual_seq_qlen=actual_seq_lengths_q,
            block_table=block_table,
            num_query_heads=self.num_heads_per_rank,
            num_key_value_heads=self.num_key_value_heads_per_rank,
            softmax_scale=self.softmax_scale,
            input_layout="TND_NTD",
            sparse_mode=sparse_mode,
            block_size=self.block_size,
            query_quant_mode=0,
            key_quant_mode=0,
            value_quant_mode=0,
            return_softmax_lse=False,
        )
        attn_output = torch_npu.npu_transpose_batchmatmul(
            attn_output,
            self.kv_b_proj_w_v,
            bias=None,
            scale=None,
            perm_x1=(0, 1, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
        )
        if record_prefetch_event:
            attn_output, o_proj, prefetch_event = self.o_proj_forward(
                attn_output.view(num_tokens, -1), record_prefetch_event=True)
            return attn_output, o_proj, prefetch_event
        attn_output, o_proj = self.o_proj_forward(attn_output.view(num_tokens, -1))
        return attn_output, o_proj

    def forward_page_attention_absorb(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        kv_len: torch.IntTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        actual_seq_lengths_kv: list = None,
        actual_seq_lengths_q: list = None,
        slot_mapping: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        return_o_proj: bool = False,
        record_prefetch_event=False,
    ):
        query_states, k_nope, k_rope, num_tokens = self.prepare_qkv(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            slot_mapping=slot_mapping,
        )
        attn_ret = self.fused_infer_attention_score(
            query_states=query_states,
            k_nope=k_nope,
            k_rope=k_rope,
            kv_len=kv_len,
            num_tokens=num_tokens,
            attention_mask=attention_mask,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            actual_seq_lengths_q=actual_seq_lengths_q,
            block_table=block_table,
            record_prefetch_event=record_prefetch_event,
        )
        if record_prefetch_event:
            attn_output, o_proj, prefetch_event = attn_ret
        else:
            attn_output, o_proj = attn_ret
        if return_o_proj:
            if record_prefetch_event:
                return attn_output, o_proj, prefetch_event
            return attn_output, o_proj
        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_len: torch.IntTensor = None,
        actual_seq_lengths_kv: list = None,
        actual_seq_lengths_q: list = None,
        position_embeddings: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        is_prefill: bool = True,
        output_attentions: bool = False,
        slot_mapping: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(slot_mapping, dict):
            slot_mapping = slot_mapping[self.attn_type]
        if isinstance(block_table, dict):
            block_table = block_table[self.attn_type]
        input_kwargs = {
            "hidden_states": hidden_states,
            "position_embeddings": position_embeddings,
            "kv_len": kv_len,
            "position_ids": position_ids,
            "actual_seq_lengths_kv": actual_seq_lengths_kv,
            "actual_seq_lengths_q": actual_seq_lengths_q,
            "attention_mask": attention_mask,
            "slot_mapping": slot_mapping,
            "block_table": block_table,
        }
        if is_prefill:
            fn = self.forward_page_attention_normal
        else:
            fn = self.forward_page_attention_absorb
        return fn(**input_kwargs)


class LongcatFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LongcatFlashConfig, infer_config: InferenceConfig, comm_manager: CommManager,
                 layer_idx: int, prefix: str, *, npugraph_moe_stream: Optional[torch.npu.Stream] = None,
                 npugraph_prefetch_stream: Optional[torch.npu.Stream] = None, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        custom_params = self.infer_config.model_config.custom_params
        self.npugraph_moe_stream = npugraph_moe_stream
        self.hidden_size = config.hidden_size
        self.enable_afd = custom_params.get("enable_afd", False)
        if not self.enable_afd:
            self.mlp = LongcatFlashMoE(
                config, self.infer_config, self.comm_manager, layer_idx, prefix=f"{prefix}.mlp", **kwargs)
        self.enable_multi_streams = custom_params.get("enable_multi_streams", 0)
        self.enable_superkernel = custom_params.get("enable_superkernel", False)
        self.enable_prefetch = custom_params.get("enable_prefetch", False)
        self.enable_npugraph_ex = self.infer_config.model_config.exe_mode == "npugraph_ex"
        self.enable_npugraph_moe_events = self.enable_npugraph_ex and self.enable_multi_streams and not self.enable_afd
        self.npugraph_moe_events = []
        if self.enable_npugraph_moe_events:
            self.npugraph_moe_events = [torch.npu.Event(), torch.npu.Event()]
            if self.npugraph_moe_stream is None:
                self.npugraph_moe_stream = torch.npu.Stream()
        self.npugraph_prefetch_stream = npugraph_prefetch_stream
        self.ffn_world_size = (
            self.infer_config.parallel_config.world_size // 2
            if self.enable_afd
            else 0
        )
        self.global_rank = dist.get_rank()
        # ensure recv/send comm tags do not overlap. Attn send tag value should equal to FFN recv tag.
        self.send_tag = layer_idx
        self.recv_tag = layer_idx + config.num_hidden_layers
        if self.enable_multi_streams == 2: # takes effects only when enable_multi_streams > 0
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
                    infer_config=self.infer_config,
                    comm_manager=self.comm_manager,
                    layer_idx=layer_idx * 2 + i,
                    prefix=f"{prefix}.self_attn.{i}",
                    **kwargs
                )
            )
            mlps.append(LongcatFlashMLP(config, self.infer_config, self.comm_manager, f"{prefix}.mlps.{i}", **kwargs))
            input_layernorm.append(LongcatFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps))
            post_attention_layernorm.append(LongcatFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps))

        self.self_attn = nn.ModuleList(self_attn)
        self.mlps = nn.ModuleList(mlps)
        self.input_layernorm = nn.ModuleList(input_layernorm)
        self.post_attention_layernorm = nn.ModuleList(post_attention_layernorm)

        if self.npugraph_prefetch_stream is not None:
            for mlp in self.mlps:
                mlp.npugraph_prefetch_stream = self.npugraph_prefetch_stream

    def _wait_prefetch_event(self, prefetch_event):
        if prefetch_event is None or self.npugraph_prefetch_stream is None:
            return
        with npu_stream_switch(True, self.npugraph_prefetch_stream):
            self.npugraph_prefetch_stream.wait_event(prefetch_event)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_len: torch.IntTensor,
        actual_seq_lengths_kv: list,
        actual_seq_lengths_q: list = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        is_prefill: Optional[bool] = False,
        slot_mapping: Optional[torch.Tensor] = None,
        past_residual: Optional[torch.Tensor] = None,
        cur_topk_list: Optional[torch.Tensor] = None,
        next_layer: Optional['LongcatFlashDecoderLayer'] = None,
        block_table: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if (self.enable_multi_streams > 0) and not is_prefill:
            return self.multi_stream_forward(
                hidden_states,
                kv_len,
                actual_seq_lengths_kv,
                actual_seq_lengths_q=actual_seq_lengths_q,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                is_prefill=is_prefill,
                slot_mapping=slot_mapping,
                past_residual=past_residual,
                cur_topk_list=cur_topk_list,
                next_layer=next_layer,
                block_table=block_table,
                **kwargs,
            )
        residual = past_residual
        for i in range(2):
            hidden_states, residual = self.input_layernorm[i](hidden_states, residual)

            hidden_states = self.self_attn[i](
                hidden_states=hidden_states,
                kv_len=kv_len,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                actual_seq_lengths_q=actual_seq_lengths_q,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                is_prefill=is_prefill,
                slot_mapping=slot_mapping,
                block_table=block_table,
            )

            hidden_states, residual = self.post_attention_layernorm[i](hidden_states, residual)

            if i == 0:
                if self.enable_afd:
                    defer_afd_recv = self.enable_npugraph_ex and not is_prefill
                    dist.send(hidden_states, dst=(self.global_rank - self.ffn_world_size), tag=self.send_tag)
                    shortcut_mlp_output = torch.empty_like(hidden_states)
                    if not defer_afd_recv:
                        dist.recv(shortcut_mlp_output, src=(self.global_rank - self.ffn_world_size), tag=self.recv_tag)
                else:
                    # shortcut output (MoE output)
                    shortcut_mlp_output = self.mlp(hidden_states, is_prefill, cur_topk_list=cur_topk_list)

            hidden_states, _, _ = self.mlps[i](hidden_states, is_prefill=is_prefill)
            if i == 1:
                if self.enable_afd and defer_afd_recv:
                    dist.recv(shortcut_mlp_output, src=(self.global_rank - self.ffn_world_size), tag=self.recv_tag)
                hidden_states = hidden_states + shortcut_mlp_output

        outputs = (residual, hidden_states)
        return outputs

    def multi_stream_forward(
        self,
        hidden_states: torch.Tensor,
        kv_len: torch.IntTensor,
        actual_seq_lengths_kv: list,
        actual_seq_lengths_q: list = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        is_prefill: Optional[bool] = False,
        slot_mapping: Optional[torch.Tensor] = None,
        past_residual: Optional[torch.Tensor] = None,
        cur_topk_list: Optional[torch.Tensor] = None,
        next_layer: Optional['LongcatFlashDecoderLayer'] = None,
        block_table: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = past_residual
        route_prefetch = self.npugraph_prefetch_stream is not None
        with superkernel_scope(self.enable_superkernel, f"scope_{self.layer_idx}_part1", ""):
            hidden_states, residual = self.input_layernorm[0](hidden_states, residual)
            attn_ret = self.self_attn[0].forward_page_attention_absorb(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                kv_len=kv_len,
                position_ids=position_ids,
                attention_mask=attention_mask,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                actual_seq_lengths_q=actual_seq_lengths_q,
                slot_mapping=slot_mapping,
                block_table=block_table,
                return_o_proj=True,
                record_prefetch_event=route_prefetch,
            )
            if route_prefetch:
                hidden_states, o_proj, pf_ev0 = attn_ret
            else:
                hidden_states, o_proj = attn_ret
                pf_ev0 = None
            hidden_states_norm, residual = self.post_attention_layernorm[0](hidden_states, residual)

        if not self.enable_afd:
            with npu_stream_switch(route_prefetch, self.npugraph_prefetch_stream):
                if route_prefetch and pf_ev0 is not None:
                    self.npugraph_prefetch_stream.wait_event(pf_ev0)
                npu_prefetch(self.enable_prefetch, self.mlp.router.classifier.weight.data,
                             o_proj, 18 * 1024 * 1024, 0)
            use_npugraph_event = self.enable_npugraph_moe_events
            if use_npugraph_event:
                record_stream(use_npugraph_event, hidden_states_norm, self.npugraph_moe_stream)
                record_event(use_npugraph_event, self.npugraph_moe_events, 0)
                stream_ctx = npu_stream_switch(True, self.npugraph_moe_stream)
            else:
                stream_ctx = npu_stream_switch_gegraph(True, "1")
            with stream_ctx:
                if use_npugraph_event:
                    wait_event(use_npugraph_event, self.npugraph_moe_events, 0)
                with limit_core_num(True, self.aic_num1, self.aiv_num1):
                    with superkernel_scope(self.enable_superkernel, f"scope_{self.layer_idx}_part2_moe", ""):
                        shortcut_mlp_output = self.mlp(hidden_states_norm, is_prefill, cur_topk_list=cur_topk_list)
                if use_npugraph_event:
                    record_event(use_npugraph_event, self.npugraph_moe_events, 1)
        else:
            defer_afd_recv = self.enable_npugraph_ex and not is_prefill
            dist.send(hidden_states_norm, dst=(self.global_rank - self.ffn_world_size), tag=self.send_tag)
            shortcut_mlp_output = torch.empty_like(hidden_states)
            if not defer_afd_recv:
                dist.recv(shortcut_mlp_output, src=(self.global_rank - self.ffn_world_size), tag=self.recv_tag)

        with limit_core_num(not self.enable_afd, self.aic_num2, self.aiv_num2):
            with superkernel_scope(self.enable_superkernel, f"scope_{self.layer_idx}_part2_main", ""):

                if route_prefetch and self.enable_prefetch:
                    self._wait_prefetch_event(pf_ev0)
                    hidden_states, _, dsq, pf_ev1, _ = self.mlps[0].forward_prefetch(
                        hidden_states_norm, o_proj, is_prefill=is_prefill, record_gateup_event=True
                    )
                elif self.enable_prefetch:
                    hidden_states, _, dsq = self.mlps[0].forward_prefetch(
                        hidden_states_norm, o_proj, is_prefill=is_prefill
                        )
                    pf_ev1 = None
                else:
                    hidden_states, _, dsq = self.mlps[0](hidden_states_norm, is_prefill=is_prefill)
                    pf_ev1 = None

                with npu_stream_switch(route_prefetch, self.npugraph_prefetch_stream):
                    if route_prefetch and pf_ev1 is not None:
                        self.npugraph_prefetch_stream.wait_event(pf_ev1)
                    npu_prefetch(self.enable_prefetch, self.self_attn[1].q_a_proj.weight.data, dsq, 18 * 1024 * 1024, 0)
                    npu_prefetch(self.enable_prefetch, self.self_attn[1].q_b_proj.weight.data, dsq, 36 * 1024 * 1024, 0)
                    npu_prefetch(
                        self.enable_prefetch, self.self_attn[1].kv_a_proj_with_mqa.weight.data, dsq,
                        7 * 1024 * 1024, 0)

                hidden_states, residual = self.input_layernorm[1](hidden_states, residual)
                attn_ret = self.self_attn[1].forward_page_attention_absorb(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    kv_len=kv_len,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    actual_seq_lengths_kv=actual_seq_lengths_kv,
                    actual_seq_lengths_q=actual_seq_lengths_q,
                    slot_mapping=slot_mapping,
                    block_table=block_table,
                    return_o_proj=True,
                    record_prefetch_event=route_prefetch,
                )
                if route_prefetch:
                    hidden_states, o_proj, pf_ev2 = attn_ret
                else:
                    hidden_states, o_proj = attn_ret
                    pf_ev2 = None
                hidden_states, residual = self.post_attention_layernorm[1](hidden_states, residual)
                if route_prefetch and self.enable_prefetch:
                    self._wait_prefetch_event(pf_ev2)
                    hidden_states, down_proj, _, _, pf_ev3 = self.mlps[1].forward_prefetch(
                        hidden_states, o_proj, is_prefill=is_prefill, record_down_event=True
                    )
                elif self.enable_prefetch:
                    hidden_states, down_proj, _ = self.mlps[1].forward_prefetch(
                        hidden_states, o_proj, is_prefill=is_prefill
                    )
                    pf_ev3 = None
                else:
                    hidden_states, down_proj, _ = self.mlps[1](hidden_states, is_prefill=is_prefill)
                    pf_ev3 = None
                if next_layer is not None:
                    with npu_stream_switch(route_prefetch, self.npugraph_prefetch_stream):
                        if route_prefetch and pf_ev3 is not None:
                            self.npugraph_prefetch_stream.wait_event(pf_ev3)
                        npu_prefetch(self.enable_prefetch, next_layer.self_attn[0].q_a_proj.weight.data, down_proj,
                                     18 * 1024 * 1024, 0)
                        npu_prefetch(self.enable_prefetch, next_layer.self_attn[0].q_b_proj.weight.data, down_proj,
                                     36 * 1024 * 1024, 0)
                        npu_prefetch(
                            self.enable_prefetch, next_layer.self_attn[0].kv_a_proj_with_mqa.weight.data, down_proj,
                            7 * 1024 * 1024, 0)

        if self.enable_npugraph_moe_events:
            wait_event(self.enable_npugraph_moe_events, self.npugraph_moe_events, 1)
        if self.enable_afd and defer_afd_recv:
            dist.recv(shortcut_mlp_output, src=(self.global_rank - self.ffn_world_size), tag=self.recv_tag)
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

    def __init__(self, config: LongcatFlashConfig, infer_config: InferenceConfig, comm_manager: CommManager, **kwargs):
        super().__init__(config)
        self.config = config
        self.global_rank = dist.get_rank()
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        custom_params = self.infer_config.model_config.custom_params
        enable_multi_streams = custom_params.get("enable_multi_streams", 0)
        enable_afd = custom_params.get("enable_afd", False)
        enable_npugraph_ex = self.infer_config.model_config.exe_mode == "npugraph_ex"
        enable_prefetch = custom_params.get("enable_prefetch", False)
        self.npugraph_moe_stream = None
        self.npugraph_prefetch_stream = None
        if enable_npugraph_ex and enable_multi_streams and not enable_afd:
            self.npugraph_moe_stream = torch.npu.Stream()
        if enable_npugraph_ex and enable_multi_streams and enable_prefetch:
            self.npugraph_prefetch_stream = torch.npu.Stream()
        self.ffn_world_size = (
            self.infer_config.parallel_config.world_size // 2
            if enable_afd
            else 0
        )
        self.enable_afd = enable_afd
        self.embed_tp_size = self.infer_config.parallel_config.embed_tp_size
        self.attn_tp_size = self.infer_config.parallel_config.attn_tp_size
        self.moe_ep_size = self.infer_config.parallel_config.moe_ep_size
        self.use_attn_sp = (
            self.attn_tp_size > 1
            and self.moe_ep_size > 1
        )

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.vocab_size_per_rank = self.vocab_size // self.embed_tp_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            self.padding_idx,
            torch.bfloat16,
            tp_size=self.embed_tp_size,
            tp_rank=self.comm_manager.get_rank("embed_tp_group") if self.embed_tp_size > 1 else 0)
        self.layers = nn.ModuleList(
            [
                LongcatFlashDecoderLayer(config, self.infer_config, self.comm_manager, layer_idx, \
                                         prefix=f"model.layers.{layer_idx}",
                                         npugraph_moe_stream=self.npugraph_moe_stream,
                                         npugraph_prefetch_stream=self.npugraph_prefetch_stream,
                                         **kwargs)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LongcatFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        _init_rope(self)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def calc_input_embeddings(self, input_ids, is_prefill=False):
        num_tokens = input_ids.shape[0]
        sp_prefill = is_prefill and self.use_attn_sp
        seq_length_unpad = num_tokens
        if sp_prefill:
            padding_size = align_up(num_tokens, self.attn_tp_size) - num_tokens
            if padding_size > 0:
                input_ids = F.pad(input_ids, (0, padding_size), value=0)
            num_tokens = input_ids.shape[0]
            attn_tp_rank = self.comm_manager.get_rank("attn_tp_group")
            attn_local_num_tokens = num_tokens // self.attn_tp_size
            local_start = attn_tp_rank * attn_local_num_tokens
            valid_tokens = max(0, min(seq_length_unpad - local_start, attn_local_num_tokens))

        if self.embed_tp_size > 1:
            embed_tp_group = self.comm_manager.get_group("embed_tp_group")
            local_num_tokens = num_tokens
            if self.embed_tp_size == self.attn_tp_size or (sp_prefill and self.embed_tp_size < self.attn_tp_size):
                embed_input_ids = input_ids
            else:
                if sp_prefill and self.embed_tp_size % self.attn_tp_size != 0:
                    raise ValueError(
                        f"{self.embed_tp_size=} must be divisible by {self.attn_tp_size=} "
                        "when attention SP is enabled."
                    )
                if is_prefill:
                    max_num_tokens = torch.tensor([local_num_tokens], dtype=torch.long, device=input_ids.device)
                    dist.all_reduce(max_num_tokens, op=dist.ReduceOp.MAX, group=embed_tp_group)
                    max_num_tokens = int(max_num_tokens.item())
                    padded_input_ids = input_ids
                    if local_num_tokens < max_num_tokens:
                        padded_input_ids = F.pad(input_ids, (0, max_num_tokens - local_num_tokens), value=0)
                    embed_input_ids = input_ids.new_empty(max_num_tokens * self.embed_tp_size)
                    dist.all_gather_into_tensor(embed_input_ids, padded_input_ids, group=embed_tp_group)
                else:
                    embed_input_ids = input_ids.new_empty(local_num_tokens * self.embed_tp_size)
                    dist.all_gather_into_tensor(embed_input_ids, input_ids, group=embed_tp_group)
                if sp_prefill:
                    allgather_ratio = self.embed_tp_size // self.attn_tp_size
                    embed_input_ids = embed_input_ids.view(
                        self.embed_tp_size,
                        max_num_tokens,
                    )[::self.attn_tp_size].contiguous().view(-1)

            vocab_rank = (self.global_rank - self.ffn_world_size) % self.embed_tp_size
            new_input_ids = embed_input_ids - vocab_rank * self.vocab_size_per_rank
            mask = (new_input_ids >= 0) & (new_input_ids < self.vocab_size_per_rank)
            new_input_ids_per_rank = new_input_ids * mask
            inputs_embeds = self.embed_tokens(new_input_ids_per_rank) * mask.unsqueeze(-1)
            if sp_prefill:
                if self.embed_tp_size == self.attn_tp_size:
                    inputs_embeds_attn = inputs_embeds.new_empty(attn_local_num_tokens, inputs_embeds.shape[-1])
                    dist.reduce_scatter_tensor(inputs_embeds_attn, inputs_embeds, group=embed_tp_group)
                    inputs_embeds = inputs_embeds_attn
                else:
                    dist.all_reduce(inputs_embeds, group=embed_tp_group)
                    if self.embed_tp_size > self.attn_tp_size:
                        local_dp_rank = self.comm_manager.get_rank("embed_tp_group") // self.attn_tp_size
                        inputs_embeds = inputs_embeds.view(
                            allgather_ratio,
                            max_num_tokens,
                            inputs_embeds.shape[-1],
                        )[local_dp_rank][:num_tokens]
                    elif self.attn_tp_size % self.embed_tp_size != 0:
                        raise ValueError(
                            f"{self.attn_tp_size=} must be divisible by {self.embed_tp_size=} "
                            "when attention SP is enabled."
                        )
                    inputs_embeds = inputs_embeds[
                        local_start:local_start + attn_local_num_tokens
                    ].contiguous()
            elif self.embed_tp_size == self.attn_tp_size:
                dist.all_reduce(inputs_embeds, group=embed_tp_group)
            elif is_prefill:
                inputs_embeds_attn = inputs_embeds.new_empty(max_num_tokens, inputs_embeds.shape[-1])
                dist.reduce_scatter_tensor(inputs_embeds_attn, inputs_embeds, group=embed_tp_group)
                inputs_embeds = inputs_embeds_attn[:local_num_tokens]
            else:
                inputs_embeds_attn = inputs_embeds.new_empty(local_num_tokens, inputs_embeds.shape[-1])
                dist.reduce_scatter_tensor(inputs_embeds_attn, inputs_embeds, group=embed_tp_group)
                inputs_embeds = inputs_embeds_attn
        else:
            inputs_embeds = self.embed_tokens(input_ids)
            if sp_prefill:
                inputs_embeds = inputs_embeds[
                    local_start:local_start + attn_local_num_tokens
                ].contiguous()
        if sp_prefill:
            return _trim_attn_sp_padding(inputs_embeds, valid_tokens, attn_local_num_tokens)
        return inputs_embeds

    def prepare_inputs_for_layer_from_metadata(
        self,
        input_ids,
        position_ids,
        forward_metadata: ForwardMetaData,
    ):
        is_prefill = forward_metadata.is_prefill
        hidden_states = self.calc_input_embeddings(input_ids, is_prefill)
        if is_prefill:
            kv_len = forward_metadata.kv_len
            actual_seq_lengths_kv = forward_metadata.actual_seq_lengths_cu_kv
        else:
            batch_size = forward_metadata.kv_len.shape[0]
            kv_len = position_ids.view(batch_size, -1)
            actual_seq_lengths_kv = forward_metadata.actual_seq_lengths_kv
        position_embeddings = self.rotary_emb(
            hidden_states,
            kv_len,
            self.config.max_position_embeddings,
            position_ids=position_ids,
        )
        return hidden_states, kv_len, position_embeddings, actual_seq_lengths_kv

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        forward_metadata: ForwardMetaData,
        cur_topk_list: Optional[torch.Tensor] = None,
    ):
        hidden_states, kv_len, position_embeddings, actual_seq_lengths_kv = \
            self.prepare_inputs_for_layer_from_metadata(input_ids, position_ids, forward_metadata)
        residual = None
        actual_seq_lengths_q = forward_metadata.actual_seq_lengths_cu_q
        is_prefill = forward_metadata.is_prefill
        is_npugraph_ex_decode = (
            not is_prefill and self.infer_config.model_config.exe_mode == "npugraph_ex"
        )
        attn_actual_seq_lengths_kv = (
            forward_metadata.actual_seq_lengths_list_kv
            if is_npugraph_ex_decode
            else actual_seq_lengths_kv
        )
        attn_actual_seq_lengths_q = (
            forward_metadata.actual_seq_lengths_cu_list_q
            if is_npugraph_ex_decode
            else actual_seq_lengths_q
        )
        slot_mapping = forward_metadata.slot_mapping
        if isinstance(slot_mapping, dict):
            slot_mapping = slot_mapping.get("FullAttention")
        block_table = forward_metadata.block_table
        if isinstance(block_table, dict):
            block_table = block_table.get("FullAttention")

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states, residual = decoder_layer(
                hidden_states,
                kv_len,
                attn_actual_seq_lengths_kv,
                actual_seq_lengths_q=attn_actual_seq_lengths_q,
                position_embeddings=position_embeddings,
                attention_mask=forward_metadata.attention_mask,
                position_ids=position_ids,
                is_prefill=is_prefill,
                slot_mapping=slot_mapping,
                past_residual=residual,
                cur_topk_list=cur_topk_list,
                block_table=block_table,
                next_layer=self.layers[i + 1] if i < self.config.num_hidden_layers - 1 else None,
            )

        if self.npugraph_prefetch_stream is not None and not is_prefill:
            with npu_stream_switch(True, self.npugraph_prefetch_stream):
                prefetch_done_event = torch.npu.current_stream().record_event()
            torch.npu.current_stream().wait_event(prefetch_done_event)

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


def _calc_longcat_moe_mc2_hccl_buffer_size(
    infer_config: InferenceConfig,
    config: LongcatFlashConfig,
    world_size: int,
):
    model_config = infer_config.model_config
    platform_version = model_config.platform_version
    if hasattr(platform_version, "value"):
        platform_version = platform_version.value
    runner_settings = {
        "world_size": world_size,
        "data_config": {
            "batch_size": infer_config.scheduler_config.batch_size_per_dp_rank * world_size,
        },
        "model_config": {
            "next_n": model_config.next_n,
            "platform_version": platform_version,
        },
        "parallel_config": {
            "moe_ep_size": infer_config.parallel_config.moe_ep_size,
        },
    }
    return calc_moe_hccl_buffer_size(runner_settings, config, is_full_mesh_v2=False)


class LongcatFlashForCausalLM(LongcatFlashPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _keys_to_ignore_on_load_unexpected = [r"model\.mtp.*"]

    def __init__(self, config, infer_config: InferenceConfig, comm_manager: CommManager = None, prefix: str = "",
                 is_mtp=False):
        super().__init__(config)
        if not isinstance(comm_manager, CommManager):
            raise ValueError("LongcatFlashForCausalLM requires a CommManager in the new framework.")
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        custom_params = self.infer_config.model_config.custom_params
        self.top_k = config.moe_topk
        self.num_experts_per_tok = config.moe_topk
        self.force_eplb = self.infer_config.model_config.force_eplb
        self.enable_afd = custom_params.get("enable_afd", False)
        self.is_mtp = is_mtp
        self.lmhead_tp_size = self.infer_config.parallel_config.lmhead_tp_size
        self.moe_ep_size = self.infer_config.parallel_config.moe_ep_size
        self.enable_online_split_weight = True
        self.enable_weight_nz = self.infer_config.model_config.enable_weight_nz

        self.num_experts = (
            config.n_routed_experts
            if config.zero_expert_num is None
            else config.n_routed_experts + config.zero_expert_num
        )
        self.experts_per_rank = self.num_experts // self.moe_ep_size
        self.get_parallel_settings()
        self.init_parallel_comm_group(self.comm_manager)

        self.block_size = self.infer_config.scheduler_config.block_size
        self.model = LongcatFlashModel(config, self.infer_config, self.comm_manager)
        self.vocab_size = config.vocab_size
        self.lm_head = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            bias=False,
            tp_size=self.lmhead_tp_size,
            tp_rank=self.comm_manager.get_rank("lmhead_tp_group") if self.lmhead_tp_size > 1 else 0
        )
        # Initialize weights and apply final processing
        self.post_init()

    def check_model_settings(self):
        custom_params = self.infer_config.model_config.custom_params
        exe_mode = self.infer_config.model_config.exe_mode
        enable_cache_compile = self.infer_config.model_config.enable_cache_compile
        enable_multi_streams = custom_params.get("enable_multi_streams", 0)
        enable_superkernel = custom_params.get("enable_superkernel", False)
        enable_prefetch = custom_params.get("enable_prefetch", False)
        moe_chunk_max_len = custom_params.get("moe_chunk_max_len", 65536)
        enable_afd = custom_params.get("enable_afd", False)
        next_n = self.infer_config.model_config.next_n
        attn_tp_size = self.infer_config.parallel_config.attn_tp_size
        o_proj_tp_size = self.infer_config.parallel_config.o_proj_tp_size
        cp_size = self.infer_config.parallel_config.cp_size
        world_size = self.infer_config.parallel_config.world_size

        if exe_mode not in ["eager", "ge_graph", "npugraph_ex"]:
            raise ValueError(f"{exe_mode=} does not supported!")
        if moe_chunk_max_len <= 0:
            raise ValueError(f"{moe_chunk_max_len=} should be a positive integer.")
        if enable_afd and world_size % 2 != 0:
            raise ValueError(f"AFD is only supported when world_size % 2 == 0, but now {world_size=}.")
        if cp_size != 1:
            raise ValueError(f"LongCat-Flash does not support cp_size > 1, got {cp_size}.")
        if attn_tp_size > 1 and attn_tp_size != o_proj_tp_size:
            raise ValueError(f"when attn_tp_size > 1, {attn_tp_size=} must be equal to {o_proj_tp_size=}")
        if enable_multi_streams and exe_mode == "eager":
            raise ValueError("LongCat-Flash enable_multi_streams not supports exe_mode='eager'.")
        if exe_mode == "eager" and (enable_cache_compile or enable_superkernel):
            raise ValueError(f"{exe_mode=} does not support cache compile or superkernel!")
        if exe_mode == "npugraph_ex" and enable_superkernel:
            raise ValueError("LongCat-Flash npugraph_ex does not support superkernel.")
        if enable_prefetch and not enable_multi_streams:
            raise ValueError("LongCat-Flash enable_prefetch requires enable_multi_streams > 0.")
        if next_n > 2:
            raise ValueError(f"{next_n=}, currently only support 0 or 1 or 2")

    def _iter_attention_modules(self):
        if self.is_mtp:
            for layer in self.model.mtp.layers:
                yield layer.self_attn
            return
        for layer in self.model.layers:
            for attn in layer.self_attn:
                yield attn

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
        self.embed_tp_size = self.infer_config.parallel_config.embed_tp_size
        self.attn_dp_size = self.infer_config.parallel_config.attn_dp_size
        self.attn_tp_size = self.infer_config.parallel_config.attn_tp_size
        self.o_proj_tp_size = self.infer_config.parallel_config.o_proj_tp_size
        self.moe_ep_size = self.infer_config.parallel_config.moe_ep_size
        self.moe_tp_size = self.infer_config.parallel_config.moe_tp_size
        self.lmhead_tp_size = self.infer_config.parallel_config.lmhead_tp_size
        self.dense_tp_size = self.infer_config.parallel_config.dense_tp_size
        self.use_attn_sp = (
            self.attn_tp_size > 1
            and self.moe_ep_size > 1
        )

    def init_parallel_comm_group(self, comm_manager: CommManager):
        world_size = self.infer_config.parallel_config.world_size
        if self.enable_afd:
            # AFD comm groups are owned by ffn.py (the FFN role); import lazily to
            # avoid a modeling <-> ffn import cycle.
            from .ffn import init_afd_parallel_comm_group
            init_afd_parallel_comm_group(comm_manager, self.infer_config, self.config)
            return

        comm_manager.register_group(
            name="attn_tp_group",
            group_num=world_size // self.attn_tp_size,
            group_size=self.attn_tp_size,
        )
        comm_manager.register_group(
            name="o_proj_tp_group",
            group_num=world_size // self.o_proj_tp_size,
            group_size=self.o_proj_tp_size,
        )
        comm_manager.register_group(
            name="embed_tp_group",
            group_num=world_size // self.embed_tp_size,
            group_size=self.embed_tp_size,
        )
        comm_manager.register_group(
            name="lmhead_tp_group",
            group_num=world_size // self.lmhead_tp_size,
            group_size=self.lmhead_tp_size,
        )
        comm_manager.register_group(
            name="dense_tp_group",
            group_num=world_size // self.dense_tp_size,
            group_size=self.dense_tp_size,
        )
        comm_manager.register_group(
            name="moe_tp_group",
            group_num=world_size // self.moe_tp_size,
            group_size=self.moe_tp_size,
        )
        comm_manager.register_group(
            name="moe_ep_group",
            group_num=world_size // self.moe_ep_size,
            group_size=self.moe_ep_size,
            group_stride=self.moe_tp_size,
            return_name=True,
        )
        moe_ep_mc2_buffer_size = None
        if self.moe_ep_size > 1 and self.moe_tp_size == 1:
            moe_ep_mc2_buffer_size = _calc_longcat_moe_mc2_hccl_buffer_size(
                self.infer_config, self.config, world_size
            )
        comm_manager.register_group(
            name="moe_ep_group_mc2",
            group_num=world_size // self.moe_ep_size,
            group_size=self.moe_ep_size,
            group_stride=self.moe_tp_size,
            return_name=True,
            allow_physical_reuse=False,
            hccl_buffer_size=moe_ep_mc2_buffer_size,
        )

    def forward_lm_head(
        self,
        outputs,
        is_prefill=True,
        actual_seq_lengths_q=None,
        decode_batch_size=None,
    ):
        num_tokens = outputs.shape[0]
        hidden_size = outputs.shape[-1]
        if is_prefill:
            if actual_seq_lengths_q is None:
                raise RuntimeError("actual_seq_lengths_q is required for packed prefill lm_head.")
            actual_seq_lengths_q = actual_seq_lengths_q.to(dtype=torch.long, device=outputs.device)
            bs = actual_seq_lengths_q.numel()
            seq_index = actual_seq_lengths_q - 1
            outputs = torch.index_select(outputs.view(-1, hidden_size), 0, seq_index).view(bs, 1, hidden_size)
            q_len = 1
        else:
            if decode_batch_size is not None:
                bs = decode_batch_size
                q_len = num_tokens // bs
            else:
                bs = num_tokens
                q_len = 1
            outputs = outputs.view(bs * q_len, 1, hidden_size)

        if (self.attn_dp_size == 1) or (self.lmhead_tp_size == 1):
            hidden_states = outputs
        else:
            hidden_states = torch.empty_like(outputs).repeat(self.lmhead_tp_size, 1, 1)
            dist.all_gather_into_tensor(hidden_states, outputs, group=self.comm_manager.get_group("lmhead_tp_group"))

        logits = self.lm_head(hidden_states)
        if self.lmhead_tp_size > 1:
            if self.attn_dp_size == 1:
                new_logits = torch.empty_like(logits).repeat(self.lmhead_tp_size, 1, 1)
                dist.all_gather_into_tensor(new_logits, logits, group=self.comm_manager.get_group("lmhead_tp_group"))
            else:
                new_logits = torch.empty_like(logits).view(-1)
                dist.all_to_all_single(
                    new_logits,
                    logits.view(-1),
                    group=self.comm_manager.get_group("lmhead_tp_group"),
                )

            new_logits = new_logits.reshape(
                self.lmhead_tp_size, bs * q_len, logits.shape[1], -1
            ).permute(1, 2, 0, 3)
            logits = new_logits.reshape(bs * q_len, logits.shape[1], self.config.vocab_size)
        logits = logits.reshape(bs, q_len, -1).float()
        return logits

    def gather_attn_sp_outputs(self, outputs, seq_length_unpad: int, is_prefill: bool):
        if not is_prefill or not self.use_attn_sp:
            return outputs
        local_num_tokens, hidden_size = outputs.shape
        new_outputs = outputs.new_empty(local_num_tokens * self.attn_tp_size, hidden_size)
        dist.all_gather_into_tensor(
            new_outputs,
            outputs.contiguous(),
            group=self.comm_manager.get_group("attn_tp_group"),
        )
        return new_outputs[:seq_length_unpad]

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        forward_metadata: Optional[ForwardMetaData] = None,
        **kwargs,
    ):
        prev_hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            forward_metadata=forward_metadata,
            cur_topk_list=kwargs.get('cur_topk_list', None),
        )
        prev_hidden_states = prev_hidden_states.reshape(-1, self.config.hidden_size)
        prev_hidden_states = self.gather_attn_sp_outputs(
            prev_hidden_states,
            input_ids.shape[0],
            forward_metadata.is_prefill,
        )
        logits = self.forward_lm_head(
            prev_hidden_states,
            is_prefill=forward_metadata.is_prefill,
            actual_seq_lengths_q=forward_metadata.actual_seq_lengths_cu_q,
            decode_batch_size=forward_metadata.kv_len.shape[0] if not forward_metadata.is_prefill else None,
        )
        return logits, prev_hidden_states

    def get_cache_info(self) -> ModelCacheInfo:
        layer_infos = []
        for layer_idx, attn in enumerate(self._iter_attention_modules()):
            layer_infos.append(
                LayerCacheInfo(
                    layer_idx=layer_idx,
                    caches=list(attn.cache_entries),
                )
            )
        return ModelCacheInfo(
            num_layers=len(layer_infos),
            layer_infos=layer_infos,
            is_mla_backend=True,
        )

    def init_splited_kv_b_weight(self):
        def for_each_to_init_splited_k_b_weight(layer):
            try:
                if hasattr(self.model, 'mtp'):
                    data_tensor = attrgetter("kv_b_proj_w_k_data")(layer.self_attn)
                    layer.self_attn.kv_b_proj_w_k = nn.Parameter(data_tensor.contiguous(), requires_grad=False)
                else:
                    for attn in layer.self_attn:
                        data_tensor = attrgetter("kv_b_proj_w_k_data")(attn)
                        attn.kv_b_proj_w_k = nn.Parameter(data_tensor.contiguous(), requires_grad=False)
            except AttributeError:
                pass

        def for_each_to_init_splited_v_b_weight(layer):
            try:
                if hasattr(self.model, 'mtp'):
                    data_tensor = attrgetter("kv_b_proj_w_v_data")(layer.self_attn)
                    layer.self_attn.kv_b_proj_w_v = nn.Parameter(data_tensor.contiguous(), requires_grad=False)
                else:
                    for attn in layer.self_attn:
                        data_tensor = attrgetter("kv_b_proj_w_v_data")(attn)
                        attn.kv_b_proj_w_v = nn.Parameter(data_tensor.contiguous(), requires_grad=False)
            except AttributeError:
                pass

        def for_each_to_offload_kv_b_weight(layer):
            try:
                if hasattr(self.model, 'mtp'):
                    layer.self_attn.kv_b_proj.weight = None
                else:
                    for attn in layer.self_attn:
                        attn.kv_b_proj.weight = None
            except AttributeError:
                pass

        layers = self.model.mtp.layers if self.is_mtp and hasattr(self.model, "mtp") else self.model.layers
        for layer in layers:
            for_each_to_init_splited_k_b_weight(layer)
            for_each_to_init_splited_v_b_weight(layer)
            for_each_to_offload_kv_b_weight(layer)

    def process_weights_after_loading(self):
        self.init_splited_kv_b_weight()
        for attn in self._iter_attention_modules():
            attn.fold_mla_kv_lora_scale_into_norm()
        float_scales_map = [
            "gate_up_proj",
        ]
        float_smooth_scales_map = [
            "down_proj",
        ]
        enable_mla_prolog = self.infer_config.model_config.custom_params.get("enable_mla_prolog", False)
        if enable_mla_prolog and self.config.q_lora_rank is not None:
            float_scales_map += [
                "q_a_proj",
                "q_b_proj",
                "kv_a_proj_with_mqa",
            ]
        if not self.enable_online_split_weight:
            if hasattr(self, "scale_dtype_adapter"):
                self.scale_dtype_adapter()
            if hasattr(self, "cast_format"):
                self.cast_format()
            return

        for module_name, module in self.named_modules():
            if "kv_b_proj" in module_name:
                continue
            quant_method = getattr(module, "quant_method", None)
            scales_dtype = {}
            for scale_name in float_scales_map:
                if scale_name in module_name:
                    scales_dtype['scale_dtype'] = torch.float
                    break
            for smooth_scale_name in float_smooth_scales_map:
                if smooth_scale_name in module_name:
                    scales_dtype['smooth_scale_dtype'] = torch.float
                    break

            if isinstance(quant_method, QuantizeMethodBase):
                quant_method.process_weights_after_loading(
                    module, is_nz=self.enable_weight_nz, scales_dtype=scales_dtype)
            if isinstance(quant_method, CompressedTensorW8A8Int8MoEGMMMethod):
                if self.moe_ep_size > 1:
                    group = self.comm_manager.get_group("moe_ep_group")
                    all_experts_smooth_scale = module.smooth_scale_1.data.new_empty(
                        module.smooth_scale_1.data.shape[0] * self.moe_ep_size,
                        module.smooth_scale_1.data.shape[1],
                    )
                    dist.all_gather_into_tensor(all_experts_smooth_scale, module.smooth_scale_1.data, group=group)
                    module.smooth_scale_1.data = all_experts_smooth_scale

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


class LongcatFlashMTPDecoderLayer(nn.Module):
    def __init__(self, config: LongcatFlashConfig, infer_config: InferenceConfig, comm_manager: CommManager,
                 layer_idx: int, prefix: str, **kwargs):
        super().__init__()
        self.self_attn = LongcatFlashAttention(config=config, infer_config=infer_config, comm_manager=comm_manager, \
                                               layer_idx=layer_idx, prefix=f"{prefix}.self_attn", **kwargs)
        self.mlp = LongcatFlashMLP(config, infer_config, comm_manager, f"{prefix}.mlp", **kwargs)

        self.input_layernorm = LongcatFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps) # eps=1e-5
        self.post_attention_layernorm = LongcatFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_len: torch.IntTensor,
        actual_seq_lengths_kv: list,
        actual_seq_lengths_q: list = None,
        cos_sin: torch.Tensor = None,
        past_residual: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        is_prefill: Optional[bool] = False,
        slot_mapping: Optional[torch.Tensor] = None,
        block_table: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            actual_seq_lengths_q=actual_seq_lengths_q,
            position_embeddings=cos_sin,
            slot_mapping=slot_mapping,
            is_prefill=is_prefill,
            kv_len=kv_len,
            block_table=block_table,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _, _ = self.mlp(hidden_states, is_prefill=is_prefill)
        hidden_states = residual + hidden_states
        return hidden_states


class LongcatFlashMTPLayer(nn.Module):
    def __init__(self, config: LongcatFlashConfig, infer_config: InferenceConfig, comm_manager: CommManager,
                 layer_idx: int, **kwargs):
        super().__init__()
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.attn_tp_size = self.infer_config.parallel_config.attn_tp_size
        self.mtp = nn.Module()
        self.mtp.layers = nn.ModuleList(
            [
                LongcatFlashMTPDecoderLayer(config, self.infer_config, self.comm_manager, layer_idx, \
                                            prefix=f"model.mtp.layers.{i}", **kwargs)
                for i in range(config.num_nextn_predict_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_len: torch.IntTensor,
        actual_seq_lengths_kv: list,
        actual_seq_lengths_q: list = None,
        cos_sin: torch.Tensor = None,
        past_residual: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        is_prefill: Optional[bool] = False,
        slot_mapping: Optional[torch.Tensor] = None,
        block_table: Optional[Dict[str, torch.Tensor]] = None,
        mtp_layer_idx: Optional[int] = 0,
        input_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.mtp.layers[mtp_layer_idx](
            hidden_states,
            kv_len,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            actual_seq_lengths_q=actual_seq_lengths_q,
            cos_sin=cos_sin,
            past_residual=past_residual,
            attention_mask=attention_mask,
            position_ids=position_ids,
            is_prefill=is_prefill,
            slot_mapping=slot_mapping,
            block_table=block_table,
            mtp_layer_idx=mtp_layer_idx,
            input_ids=input_ids,
        )


class LongcatFlashModelMTP(LongcatFlashForCausalLM):
    def __init__(self, config: LongcatFlashConfig, infer_config: InferenceConfig, comm_manager: CommManager = None,
                 prefix: str = "", **kwargs):
        super().__init__(config, infer_config, comm_manager=comm_manager, prefix=prefix, is_mtp=True)
        self.global_rank = dist.get_rank()
        self.ffn_world_size = (
            self.infer_config.parallel_config.world_size // 2
            if self.enable_afd
            else 0
        )
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.vocab_size_per_rank = self.vocab_size // self.embed_tp_size
        self.ignore_share_weight = False

        self.mtp_layer_idx = config.num_hidden_layers * 2 # MTP is the last layer
        self.model = LongcatFlashMTPLayer(config, self.infer_config, self.comm_manager, self.mtp_layer_idx)

        # no reuse
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            self.padding_idx,
            torch.bfloat16,
            tp_size=self.embed_tp_size,
            tp_rank=self.comm_manager.get_rank("embed_tp_group") if self.embed_tp_size > 1 else 0)

        # lm_head is shared from the main model by MTPWorker.
        self.lm_head = None
        _init_rope(self)

        self.norm = LongcatFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.enorm = LongcatFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = LongcatFlashRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # prev_hidden_states and input_hidden_state feature fusion
        self.eh_proj = ReplicatedLinear(2 * config.hidden_size, config.hidden_size, bias=False)

    def calc_input_embeddings(self, input_ids, is_prefill=False):
        num_tokens = input_ids.shape[0]
        sp_prefill = is_prefill and self.use_attn_sp
        seq_length_unpad = num_tokens
        if sp_prefill:
            padding_size = align_up(num_tokens, self.attn_tp_size) - num_tokens
            if padding_size > 0:
                input_ids = F.pad(input_ids, (0, padding_size), value=0)
            num_tokens = input_ids.shape[0]
            attn_tp_rank = self.comm_manager.get_rank("attn_tp_group")
            attn_local_num_tokens = num_tokens // self.attn_tp_size
            local_start = attn_tp_rank * attn_local_num_tokens
            valid_tokens = max(0, min(seq_length_unpad - local_start, attn_local_num_tokens))

        if self.embed_tp_size > 1:
            embed_tp_group = self.comm_manager.get_group("embed_tp_group")
            local_num_tokens = num_tokens
            if self.embed_tp_size == self.attn_tp_size or (sp_prefill and self.embed_tp_size < self.attn_tp_size):
                embed_input_ids = input_ids
            else:
                if sp_prefill and self.embed_tp_size % self.attn_tp_size != 0:
                    raise ValueError(
                        f"{self.embed_tp_size=} must be divisible by {self.attn_tp_size=} "
                        "when attention SP is enabled."
                    )
                if is_prefill:
                    max_num_tokens = torch.tensor([local_num_tokens], dtype=torch.long, device=input_ids.device)
                    dist.all_reduce(max_num_tokens, op=dist.ReduceOp.MAX, group=embed_tp_group)
                    max_num_tokens = int(max_num_tokens.item())
                    padded_input_ids = input_ids
                    if local_num_tokens < max_num_tokens:
                        padded_input_ids = F.pad(input_ids, (0, max_num_tokens - local_num_tokens), value=0)
                    embed_input_ids = input_ids.new_empty(max_num_tokens * self.embed_tp_size)
                    dist.all_gather_into_tensor(embed_input_ids, padded_input_ids, group=embed_tp_group)
                else:
                    embed_input_ids = input_ids.new_empty(local_num_tokens * self.embed_tp_size)
                    dist.all_gather_into_tensor(embed_input_ids, input_ids, group=embed_tp_group)
                if sp_prefill:
                    allgather_ratio = self.embed_tp_size // self.attn_tp_size
                    embed_input_ids = embed_input_ids.view(
                        self.embed_tp_size,
                        max_num_tokens,
                    )[::self.attn_tp_size].contiguous().view(-1)

            vocab_rank = (self.global_rank - self.ffn_world_size) % self.embed_tp_size
            new_input_ids = embed_input_ids - vocab_rank * self.vocab_size_per_rank
            mask = (new_input_ids >= 0) & (new_input_ids < self.vocab_size_per_rank)
            new_input_ids_per_rank = new_input_ids * mask
            inputs_embeds = self.embed_tokens(new_input_ids_per_rank) * mask.unsqueeze(-1)
            if sp_prefill:
                if self.embed_tp_size == self.attn_tp_size:
                    inputs_embeds_attn = inputs_embeds.new_empty(attn_local_num_tokens, inputs_embeds.shape[-1])
                    dist.reduce_scatter_tensor(inputs_embeds_attn, inputs_embeds, group=embed_tp_group)
                    inputs_embeds = inputs_embeds_attn
                else:
                    dist.all_reduce(inputs_embeds, group=embed_tp_group)
                    if self.embed_tp_size > self.attn_tp_size:
                        local_dp_rank = self.comm_manager.get_rank("embed_tp_group") // self.attn_tp_size
                        inputs_embeds = inputs_embeds.view(
                            allgather_ratio,
                            max_num_tokens,
                            inputs_embeds.shape[-1],
                        )[local_dp_rank][:num_tokens]
                    elif self.attn_tp_size % self.embed_tp_size != 0:
                        raise ValueError(
                            f"{self.attn_tp_size=} must be divisible by {self.embed_tp_size=} "
                            "when attention SP is enabled."
                        )
                    inputs_embeds = inputs_embeds[
                        local_start:local_start + attn_local_num_tokens
                    ].contiguous()
            elif self.embed_tp_size == self.attn_tp_size:
                dist.all_reduce(inputs_embeds, group=embed_tp_group)
            elif is_prefill:
                inputs_embeds_attn = inputs_embeds.new_empty(max_num_tokens, inputs_embeds.shape[-1])
                dist.reduce_scatter_tensor(inputs_embeds_attn, inputs_embeds, group=embed_tp_group)
                inputs_embeds = inputs_embeds_attn[:local_num_tokens]
            else:
                inputs_embeds_attn = inputs_embeds.new_empty(local_num_tokens, inputs_embeds.shape[-1])
                dist.reduce_scatter_tensor(inputs_embeds_attn, inputs_embeds, group=embed_tp_group)
                inputs_embeds = inputs_embeds_attn
        else:
            inputs_embeds = self.embed_tokens(input_ids)
            if sp_prefill:
                inputs_embeds = inputs_embeds[
                    local_start:local_start + attn_local_num_tokens
                ].contiguous()
        if sp_prefill:
            return _trim_attn_sp_padding(inputs_embeds, valid_tokens, attn_local_num_tokens)
        return inputs_embeds

    def slice_attn_sp_prev_hidden_states(self, prev_hidden_states, seq_length_unpad, is_prefill: bool):
        if not is_prefill or not self.use_attn_sp:
            return prev_hidden_states[:seq_length_unpad]
        padded_num_tokens = align_up(seq_length_unpad, self.attn_tp_size)
        padding_size = padded_num_tokens - seq_length_unpad
        prev_hidden_states = prev_hidden_states[:seq_length_unpad]
        if padding_size > 0:
            prev_hidden_states = F.pad(prev_hidden_states, (0, 0, 0, padding_size), value=0)
        attn_tp_rank = self.comm_manager.get_rank("attn_tp_group")
        local_num_tokens = padded_num_tokens // self.attn_tp_size
        local_start = attn_tp_rank * local_num_tokens
        prev_hidden_states = prev_hidden_states[local_start:local_start + local_num_tokens].contiguous()
        valid_tokens = max(0, min(seq_length_unpad - local_start, local_num_tokens))
        return _trim_attn_sp_padding(prev_hidden_states, valid_tokens, local_num_tokens)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        forward_metadata: Optional[ForwardMetaData] = None,
        prev_hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if prev_hidden_states is None:
            raise RuntimeError("LongCat-Flash MTP requires prev_hidden_states from the main model.")

        is_prefill = forward_metadata.is_prefill
        hidden_states = self.calc_input_embeddings(input_ids, is_prefill)
        prev_hidden_states = prev_hidden_states.view(-1, prev_hidden_states.shape[-1])
        prev_hidden_states = self.slice_attn_sp_prev_hidden_states(prev_hidden_states, input_ids.shape[0], is_prefill)

        hidden_states = self.enorm(hidden_states)
        prev_hidden_states = self.hnorm(prev_hidden_states)
        hidden_states_eh = torch.cat([hidden_states, prev_hidden_states], dim=-1)
        hidden_states = self.eh_proj(hidden_states_eh)

        if is_prefill:
            kv_len = forward_metadata.kv_len
            actual_seq_lengths_kv = forward_metadata.actual_seq_lengths_cu_kv
        else:
            batch_size = forward_metadata.kv_len.shape[0]
            kv_len = position_ids.view(batch_size, -1)
            actual_seq_lengths_kv = forward_metadata.actual_seq_lengths_kv
        actual_seq_lengths_q = forward_metadata.actual_seq_lengths_cu_q
        cos_sin = self.rotary_emb(
            hidden_states,
            kv_len,
            self.config.max_position_embeddings,
            position_ids=position_ids,
        )
        is_npugraph_ex_decode = (
            not is_prefill and self.infer_config.model_config.exe_mode == "npugraph_ex"
        )
        attn_actual_seq_lengths_kv = (
            forward_metadata.actual_seq_lengths_list_kv
            if is_npugraph_ex_decode
            else actual_seq_lengths_kv
        )
        attn_actual_seq_lengths_q = (
            forward_metadata.actual_seq_lengths_cu_list_q
            if is_npugraph_ex_decode
            else actual_seq_lengths_q
        )

        hidden_states = self.model(
            hidden_states,
            kv_len,
            attn_actual_seq_lengths_kv,
            actual_seq_lengths_q=attn_actual_seq_lengths_q,
            input_ids=input_ids,
            cos_sin=cos_sin,
            past_residual=None,
            attention_mask=forward_metadata.attention_mask,
            position_ids=position_ids,
            is_prefill=is_prefill,
            cur_topk_list=kwargs.get('cur_topk_list', None),
            slot_mapping=forward_metadata.slot_mapping,
            block_table=forward_metadata.block_table,
        )

        prev_hidden_states = self.norm(hidden_states).reshape(-1, self.config.hidden_size)
        prev_hidden_states = self.gather_attn_sp_outputs(
            prev_hidden_states,
            input_ids.shape[0],
            is_prefill,
        )
        logits = self.forward_lm_head(
            outputs=prev_hidden_states,
            is_prefill=is_prefill,
            actual_seq_lengths_q=forward_metadata.actual_seq_lengths_cu_q,
            decode_batch_size=forward_metadata.kv_len.shape[0] if not is_prefill else None,
        )
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



__all__ = [
    "LongcatFlashPreTrainedModel",
    "LongcatFlashModel",
    "LongcatFlashForCausalLM",
    "LongcatFlashModelMTP",
]
