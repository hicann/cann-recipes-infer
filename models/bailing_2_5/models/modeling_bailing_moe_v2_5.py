# coding=utf-8
# Adapted from
# https://huggingface.co/inclusionAI/Ling-2.5-1T/blob/main/modeling_bailing_moe_v2_5.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Copyright 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
#
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

from typing import Iterable, NamedTuple, Optional, Tuple, List, Set
import math
import gc
from operator import attrgetter

import torch
from torch import nn
import torch.distributed as dist
import torch_npu

from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import auto_docstring, can_return_tuple, logging

from executor.core.config import InferenceConfig
from executor.core.kv_cache.cache_info import (
    CacheEntry,
    LayerCacheInfo,
    MambaCacheEntry,
    ModelCacheInfo,
)
from executor.model_loader.weight_utils import default_weight_loader
from executor.utils import align_up
from executor.utils import calc_moe_hccl_buffer_size, get_default_group
from executor.utils.stream_utils import record_stream
from module.fuse_moe_gmm import FusedMoEGMM
from module.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
    QKVParallelLinear
    )
from module.quantization.utils.quant_utils import reshape_mx_scale
from module.quantization import QuantizeMethodBase

from .configuration_bailing_moe_v2_5 import BailingMoeV25Config
from .simple_gla_torch import (
    chunk_simple_gla_torch,
    fused_recurrent_simple_gla_torch,
)


logger = logging.get_logger(__name__)


def get_pa_max_length(infer_config: InferenceConfig) -> int:
    mc, dc, sc = infer_config.model_config, infer_config.data_config, infer_config.scheduler_config
    return align_up(dc.input_truncated_len + sc.max_new_tokens * (mc.next_n + 1), sc.block_size)


def get_batch_size_per_rank(infer_config: InferenceConfig) -> int:
    sc, pc = infer_config.scheduler_config, infer_config.parallel_config
    return max(sc.batch_size_per_dp_rank // pc.attn_tp_size, 1)


def uses_mc2_full_mesh(infer_config: InferenceConfig) -> bool:
    """950 runs MC2 on an aiv group; the other platforms use the fullmesh_v2 window layout."""
    pv = infer_config.model_config.platform_version
    return getattr(pv, "value", pv) != "950"


class TokenShard(NamedTuple):
    """How the token stream splits over attn_tp: gathered_shape after the all_gather, local_shape held."""
    tokens: int
    padded_tokens: int
    tokens_per_rank: int
    gathered_shape: tuple
    local_shape: tuple


def prefill_token_shard(num_tokens, attn_tp_size):
    """Prefill runs packed: one row of num_tokens, request bounds carried by actual_seq_lengths."""
    padded = align_up(num_tokens, attn_tp_size)
    per_rank = padded // attn_tp_size
    return TokenShard(num_tokens, padded, per_rank, (1, num_tokens), (1, per_rank))


def decode_token_shard(full_bsz, attn_tp_size):
    """Decode rows are one token each, so the split is per request and sized by the graph."""
    per_rank = full_bsz // attn_tp_size
    return TokenShard(full_bsz, full_bsz, per_rank, (full_bsz, 1), (per_rank, 1))


def sp_gather(hidden_states, token_shard, group):
    """local [*local_shape, H] -> the whole stream [*gathered_shape, H]."""
    hidden_size = hidden_states.shape[-1]
    gathered = torch.empty(
        [token_shard.padded_tokens, hidden_size], dtype=hidden_states.dtype, device=hidden_states.device)
    dist.all_gather_into_tensor(
        gathered, hidden_states.reshape(-1, hidden_size).contiguous(), group=group)
    return gathered[:token_shard.tokens].view(*token_shard.gathered_shape, hidden_size)


def _scatter_to_rows(packed, scatter_index, num_requests, row_len):
    """[T, ...] -> [num_requests, row_len, ...], zero where a request is shorter than the longest."""
    tail = packed.shape[1:]
    rows = packed.new_zeros((num_requests * row_len, *tail))
    rows.index_copy_(0, scatter_index, packed.reshape(-1, *tail))
    return rows.view(num_requests, row_len, *tail)


def _gather_from_rows(rows, scatter_index):
    """Inverse of _scatter_to_rows: pick the real tokens back out, in packed order."""
    tail = rows.shape[2:]
    return torch.index_select(rows.reshape(-1, *tail), 0, scatter_index)


def sp_reduce_scatter(x, token_shard, group):
    """[tokens, D] -> this rank's shard; re-pads the tail sp_gather trimmed."""
    x = x.reshape(token_shard.tokens, -1)
    if token_shard.padded_tokens > token_shard.tokens:
        x = torch.cat([x, x.new_zeros((token_shard.padded_tokens - token_shard.tokens, x.shape[-1]))])
    out = torch.empty((token_shard.tokens_per_rank, x.shape[-1]), dtype=x.dtype, device=x.device)
    dist.reduce_scatter_tensor(out, x.contiguous(), group=group)
    return out.view(*token_shard.local_shape, -1)


def uses_fia_v2(infer_config: InferenceConfig) -> bool:
    return infer_config.model_config.exe_mode == "npugraph_ex"


class LayerInputs(NamedTuple):
    hidden_states: torch.Tensor
    residual: Optional[torch.Tensor]
    kv_len: torch.Tensor
    position_embeddings: Tuple[torch.Tensor, torch.Tensor]
    position_embeddings_mla: Tuple[torch.Tensor, torch.Tensor]
    slot_mapping: torch.Tensor
    actual_seq_lengths_kv: object


class BailingMoeV25RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        BailingMoeV25RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def rms_norm(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

    def forward(self, hidden_states, *args):
        if len(args) == 0:
            result = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
            return result
        elif len(args) == 1 and args[0] is None:
            result = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
            residual = hidden_states
            return (result, residual)
        elif len(args) == 1:
            residual = args[0]
            result, _, r = torch_npu.npu_add_rms_norm(residual, hidden_states, self.weight, self.variance_epsilon)
            return (result, r)
        else:
            raise NotImplementedError(f"unsupported args len: {len(args) + 1}")


class BailingMoeV25GroupRMSNorm(nn.Module):
    def __init__(self, hidden_size, group_norm_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.group_norm_size = group_norm_size
        if hidden_size % group_norm_size != 0:
            raise ValueError(f"hidden_size must be divisible by group_norm_size, got {hidden_size=} {group_norm_size=}")
        self.variance_epsilon = eps
        self.register_buffer("ones_gamma", torch.ones(hidden_size // group_norm_size), persistent=False)

    def forward(self, hidden_states, *args):
        input_dtype = hidden_states.dtype
        orig_shape = hidden_states.shape
        group_hidden_size = orig_shape[-1] // self.group_norm_size

        x = hidden_states.view(*orig_shape[:-1], self.group_norm_size, group_hidden_size)

        y = torch_npu.npu_rms_norm(x, self.ones_gamma, self.variance_epsilon)[0]
        y = y * self.weight.view(self.group_norm_size, group_hidden_size)

        return y.view(orig_shape).to(input_dtype)


class BailingMoeV25RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

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
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, kv_len, max_seq_len=None, is_prefill=True, position_ids=None):
        if max_seq_len is None:
            self._set_cos_sin_cache(seq_len=kv_len, device=x.device, dtype=x.dtype)
        elif max_seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=max_seq_len, device=x.device, dtype=x.dtype)

        if is_prefill:
            # SD -> TND, one entry per packed token so request bounds and offsets both hold
            index = position_ids.reshape(-1)
            cos = torch.index_select(self.cos_cached, 0, index).unsqueeze(1)
            sin = torch.index_select(self.sin_cached, 0, index).unsqueeze(1)
        else:
            # BD -> BNSD
            cos = torch.index_select(self.cos_cached, dim=0, index=kv_len.view(-1)).unsqueeze(1).unsqueeze(1)
            sin = torch.index_select(self.sin_cached, dim=0, index=kv_len.view(-1)).unsqueeze(1).unsqueeze(1)

        return (
            cos.to(dtype=x.dtype),
            sin.to(dtype=x.dtype),
        )


class BailingMoeV25MLP(nn.Module):
    def __init__(self, config, infer_config: InferenceConfig, prefix, hidden_size=None,
                 intermediate_size=None, **kwargs):
        super().__init__()
        self.infer_config = infer_config
        self.dense_tp_size = infer_config.parallel_config.dense_tp_size
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
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
        self.mlp_forward = self.forward_normal

    def forward(self, x, is_prefill=False, cur_topk_list=None, gmm_event=None):
        if self.dense_tp_size > 1:
            bsz, q_len, _ = x.size()
            x_output = torch.empty([bsz * q_len * self.dense_tp_size, self.hidden_size], \
                                   dtype=x.dtype, device="npu")
            dist.all_gather_into_tensor(x_output, x, group=self.hccl_comm_dict.get("dense_tp_group", None))
            x = x_output.view(-1, q_len, self.hidden_size)

        down_proj = self.mlp_forward(x, gmm_event)

        if self.dense_tp_size > 1:
            mlp_res = down_proj.new_empty(bsz, q_len, down_proj.shape[-1])
            dist.reduce_scatter_tensor(mlp_res, down_proj, group=self.hccl_comm_dict.get("dense_tp_group", None))
        else:
            mlp_res = down_proj

        return mlp_res

    def forward_normal(self, x, gmm_event=None):
        merged_x = self.gate_up_proj(x)
        intermediate_hidden_states = torch_npu.npu_swiglu(merged_x)
        # Multi-stream: wait until the routed GMM is issued so down_proj overlaps the combine comm.
        if gmm_event is not None:
            gmm_event.wait()
        return self.down_proj(intermediate_hidden_states)


class BailingMoeV25Gate(nn.Module):
    def __init__(self, config, prefix):
        super().__init__()
        self.config = config
        self.top_k = config.moe_topk
        self.routed_scaling_factor = config.routed_scaling_factor
        self.router_bias = config.router_bias
        self.num_experts = config.n_routed_experts

        self.n_group = config.n_group
        self.topk_group = config.topk_group

        self.classifier = ReplicatedLinear(self.config.hidden_size,
                                     self.num_experts,
                                     bias=self.router_bias,
                                     quant_config=None,
                                     params_dtype=torch.float32,
                                     prefix=f"{prefix}.classifier")
        self.expert_bias = nn.Parameter(torch.empty((self.num_experts), dtype=torch.float32))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.classifier.weight, a=math.sqrt(5))

    def group_limited_topk(self, scores: torch.Tensor):
        num_tokens, _ = scores.size()
        group_scores = scores.view(num_tokens, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)

        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, self.n_group, self.num_experts // self.n_group)
            .reshape(num_tokens, -1)
        )

        masked_scores = scores.masked_fill(~score_mask.bool(), float('-inf'))

        _, topk_indices, _ = torch_npu.npu_moe_gating_top_k(
                masked_scores,
                k=self.top_k,
                renorm=0,  # 0: softmax->topk; 1: topk->softmax
                norm_type=1,  # 0: softmax; 1: sigmoid
                routed_scaling_factor=self.routed_scaling_factor,
                eps=float(1e-20)
            )

        return topk_indices

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = self.classifier(hidden_states.type(torch.float32))

        scores = torch.sigmoid(router_logits.float()).type_as(router_logits)
        scores_for_routing = scores + self.expert_bias

        topk_indices = self.group_limited_topk(scores_for_routing)

        scores = torch.gather(scores, dim=1, index=topk_indices).type_as(router_logits)

        topk_weights = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if self.top_k > 1 else scores
        topk_weights = topk_weights * self.routed_scaling_factor

        return topk_indices.to(torch.int32), topk_weights, scores_for_routing


class BailingMoeV25MoE(nn.Module):
    def __init__(self, config, infer_config: InferenceConfig, layer_idx, prefix, **kwargs):
        super().__init__()
        self.config = config
        self.infer_config = infer_config
        mc, pc = infer_config.model_config, infer_config.parallel_config
        self.gmm_quant_mode = (
            config.quant_config.gmm_quant_mode
            if config.quant_config is not None
            else "w16a16")
        self.dispatch_quant_mode = {
            "w16a16": 0,
            "w8a8int8": 2,
            "w4a8int4": 2,
            "w8a8float8": 3,
            "w8a8mxfloat8": 4,
            "w4a8mxfloat4": 4,
            "w4a4mxfloat4": 4,
        }
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.moe_tp_size = pc.moe_tp_size
        self.moe_ep_size = pc.moe_ep_size
        self.moe_chunk_max_len = mc.custom_params.get("moe_chunk_max_len", 65536)
        self.enable_multi_streams = bool(mc.custom_params.get("enable_multi_streams", False))

        self.exe_mode = mc.exe_mode
        self.enable_npugraph_ex = self.exe_mode == "npugraph_ex"
        self.enable_npugraphex_and_multistream = self.enable_multi_streams and self.enable_npugraph_ex
        self.shared_expert_stream = None
        self.shared_expert_in_event = torch.npu.Event() if self.enable_npugraphex_and_multistream else None
        self.shared_expert_gmm_event = torch.npu.Event() if self.enable_npugraphex_and_multistream else None
        self.shared_expert_event = torch.npu.Event() if self.enable_npugraphex_and_multistream else None
        self.hccl_comm_dict = kwargs.get("hccl_comm_dict", None)
        self.moe_ep_group = self.hccl_comm_dict.get("moe_ep_group", None)

        self.n_routed_experts = config.n_routed_experts
        self.n_routed_experts_per_rank = self.n_routed_experts // self.moe_ep_size
        self.router = BailingMoeV25Gate(config, f"{prefix}.router")
        self.force_eplb = mc.force_eplb
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

        if config.num_shared_experts is not None:
            self.shared_experts = BailingMoeV25MLP(config, infer_config, f"{prefix}.shared_experts", \
                intermediate_size=config.moe_intermediate_size * config.num_shared_experts, **kwargs)

    def _split_tensors(self, bs_qlen, x, topk_ids, topk_weight, hidden_states_share):
        if bs_qlen > self.moe_chunk_max_len:  # chunk moe seq_len dim to avoid OOM
            num_chunks = (bs_qlen + self.moe_chunk_max_len - 1) // self.moe_chunk_max_len
            x_list = x.chunk(num_chunks, dim=0)
            topk_ids_list = topk_ids.chunk(num_chunks, dim=0)
            topk_weight_list = topk_weight.chunk(num_chunks, dim=0)
            hidden_states_share_list = hidden_states_share.chunk(num_chunks, dim=0)
        else:
            x_list = [x]
            topk_ids_list = [topk_ids]
            topk_weight_list = [topk_weight]
            hidden_states_share_list = [hidden_states_share]
        return x_list, topk_ids_list, topk_weight_list, hidden_states_share_list

    def forward_expert(self, gathered_tokens, tokens_per_expert_group, gathered_pertoken_scale):
        if gathered_tokens.shape[0] == 0:
            return gathered_tokens

        hidden_states_ordered_by_experts, gathered_pertoken_scale, gathered_ids_unsort, tokens_per_local_expert = \
                torch_npu.npu_moe_re_routing(gathered_tokens, tokens_per_expert_group.view(self.moe_ep_size, -1),
                per_token_scales=gathered_pertoken_scale)

        gmm_args = {
            "x": hidden_states_ordered_by_experts,
            "expert_tokens": tokens_per_local_expert,
            "group_list_type": 1,
        }

        hidden_states_ordered_by_experts = self.experts(**gmm_args)
        new_x = torch.index_select(hidden_states_ordered_by_experts, 0, gathered_ids_unsort.float().argsort().int())
        return new_x

    def dispatch_double_routing(self, tokens_per_expert, expanded_x, pertoken_scale):
        tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
        dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert, group=self.moe_ep_group)
        # stack first so the reduceSum and the D2H copy happen once instead of twice
        combine_tokens = torch.stack([tokens_per_expert_group, tokens_per_expert], dim=0)
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

        gathered_pertoken_scale = None
        if pertoken_scale is not None and "a8" in self.gmm_quant_mode:
            gathered_pertoken_scale = pertoken_scale.new_empty(gathered_tokens.shape[0])
            dist.all_to_all_single(gathered_pertoken_scale, pertoken_scale,
                                   output_splits, input_splits, group=self.moe_ep_group)
        return tokens_per_expert_group, gathered_tokens, gathered_pertoken_scale, input_splits, output_splits

    def forward_combine_double_routing(self, new_x, expanded_x, input_splits, output_splits):
        gathered_tokens = new_x.new_empty(*expanded_x.shape)
        dist.all_to_all_single(gathered_tokens, new_x, input_splits, output_splits, group=self.moe_ep_group)
        return gathered_tokens

    def moe_infer_double_routing(self, x, topk_ids, topk_weight, hidden_states_share):
        batch_size, sequence_length, h = x.shape
        x = x.view(-1, h)

        # quant_mode: -1 non-quant; 1 dynamic; 2/3 mxfp8 (e5m2/e4m3); 4/5 fp8 (e5m2/e4m3)
        routing_args = {"quant_mode": -1}
        hidden_states_list = []
        for hidden_states, topk_ids, topk_weight, hidden_states_share in zip(
                *self._split_tensors(batch_size * sequence_length, x, topk_ids, topk_weight, hidden_states_share)):
            bs_qlen = hidden_states.shape[0]
            expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = torch_npu.npu_moe_init_routing_v2(
                hidden_states,
                expert_idx=topk_ids,
                active_num=topk_ids.shape[0] * topk_ids.shape[1],
                scale=None,
                expert_num=self.n_routed_experts,
                expert_tokens_num_type=1,  # 0: cumsum mode(not supported now); 1: count mode
                expert_tokens_num_flag=True,
                active_expert_range=[0, self.n_routed_experts],
                **routing_args
            )
            tokens_per_expert_group, gathered_tokens, gathered_pertoken_scale, input_splits, output_splits =\
                self.dispatch_double_routing(tokens_per_expert, expanded_x, pertoken_scale)

            new_x = self.forward_expert(gathered_tokens, tokens_per_expert_group, gathered_pertoken_scale)

            gathered_tokens = self.forward_combine_double_routing(new_x, expanded_x, input_splits, output_splits)

            hidden_states = torch_npu.npu_moe_finalize_routing(
                gathered_tokens, skip1=hidden_states_share.view(-1, hidden_states_share.shape[-1]),
                skip2=None, bias=None, scales=topk_weight.to(gathered_tokens.dtype),
                expanded_src_to_dst_row=expanded_row_idx,
                export_for_source_row=None, drop_pad_mode=2
            )

            hidden_states = hidden_states.view(bs_qlen, self.hidden_size)
            hidden_states_list.append(hidden_states)

        hidden_states = torch.cat(hidden_states_list, dim=0) if len(hidden_states_list) > 1 else hidden_states_list[0]
        return hidden_states.view(batch_size, -1, h)

    def set_mc2_kwargs(self):
        global_rank = dist.get_rank()
        moe_ep_group_name = self.hccl_comm_dict.get("moe_ep_group_mc2_name", None)
        if self.gmm_quant_mode not in self.dispatch_quant_mode:
            quant_mode = self.dispatch_quant_mode["w16a16"]
        else:
            quant_mode = self.dispatch_quant_mode[self.gmm_quant_mode]
        self.dispatch_kwargs = {
                "x_active_mask": None,
                "expert_shard_type": 0,
                "shared_expert_num": 0,
                "shared_expert_rank_num": 0,
                "moe_expert_num": self.n_routed_experts,
                "global_bs": 0,
                "quant_mode": quant_mode,
                "scales": None,
                "group_ep": moe_ep_group_name,
                "ep_world_size": self.moe_ep_size,
                "ep_rank_id": global_rank // self.moe_tp_size,
                "group_tp": moe_ep_group_name,
                "tp_world_size": self.moe_tp_size,
                "tp_rank_id": global_rank % self.moe_tp_size,
            }
        if quant_mode in (self.dispatch_quant_mode["w8a8float8"], self.dispatch_quant_mode["w8a8mxfloat8"]):
            self.dispatch_kwargs["y_dtype"] = torch.float8_e4m3fn
        # must agree with the group_type / buffer size init_parallel_comm_group picked
        if uses_mc2_full_mesh(self.infer_config):
            self.dispatch_kwargs["comm_alg"] = "fullmesh_v2"

        self.combine_kwargs = {
                "x_active_mask": None,
                "expert_shard_type": 0,
                "shared_expert_num": 0,
                "shared_expert_rank_num": 0,
                "moe_expert_num": self.n_routed_experts,
                "global_bs": 0,
                "group_ep": moe_ep_group_name,
                "ep_world_size": self.moe_ep_size,
                "ep_rank_id": global_rank // self.moe_tp_size,
                "group_tp": moe_ep_group_name,
                "tp_world_size": self.moe_tp_size,
                "tp_rank_id": global_rank % self.moe_tp_size
            }

    def moe_infer_dispatch_combine(self, x, topk_ids, topk_weight, hidden_states_share,
                                   shared_expert_event=None):
        """tp+ep mix strategy, for decode stage."""
        batch_size, sequence_length, h = x.shape
        hidden_states = x.view(-1, h)
        self.set_mc2_kwargs()

        dispatch_args = {
            "x": hidden_states,
            "expert_ids": topk_ids, # [n*topk]
            **self.dispatch_kwargs
        }
        output = torch_npu.npu_moe_distribute_dispatch_v2(**dispatch_args)
        expand_x, dynamic_scale, expand_idx, expert_token_num, ep_recv_counts, tp_recv_counts = output[:6]

        gmm_args = {
            "x": expand_x,
            "expert_tokens": expert_token_num,
            "group_list_type": 1,
        }
        if "a8" in self.gmm_quant_mode:
            if "mxfloat" in self.gmm_quant_mode:
                dynamic_scale = reshape_mx_scale(dynamic_scale)
            gmm_args.update({"pertoken_scale": dynamic_scale})

        hidden_states_ordered_by_experts = self.experts(**gmm_args)

        compute_shared_on_side = (
            shared_expert_event is not None and self.config.num_shared_experts is not None
        )
        if compute_shared_on_side:
            self.shared_expert_gmm_event.record()
            with torch.npu.stream(self.shared_expert_stream):
                self.shared_expert_in_event.wait()
                hidden_states_share = self.shared_experts(x, gmm_event=self.shared_expert_gmm_event)
                shared_expert_event.record()

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
        if not compute_shared_on_side and hidden_states_share is not None:
            combine_args["shared_expert_x"] = hidden_states_share.view(-1, hidden_states_share.shape[-1])
        hidden_states = torch_npu.npu_moe_distribute_combine_v2(**combine_args)

        if compute_shared_on_side:
            shared_expert_event.wait()
            hidden_states = hidden_states + hidden_states_share.view(-1, hidden_states_share.shape[-1])

        hidden_states = hidden_states.view(batch_size, sequence_length, self.hidden_size)
        return hidden_states, hidden_states_ordered_by_experts

    def forward(self, hidden_states, is_prefill, cur_topk_list=None):
        enable_npugraphex_and_multistream = self.enable_npugraphex_and_multistream and not is_prefill

        topk_indices, topk_weights, _ = self.router(hidden_states)
        if self.force_eplb:
            topk_indices = cur_topk_list
        topk_indices = topk_indices.to(torch.int32)

        hidden_states_share = None
        if self.config.num_shared_experts is not None:
            if enable_npugraphex_and_multistream:
                record_stream(True, hidden_states, self.shared_expert_stream)
                self.shared_expert_in_event.record()
            else:
                hidden_states_share = self.shared_experts(hidden_states)

        if is_prefill:
            output = self.moe_infer_double_routing(
                hidden_states, topk_indices, topk_weights, hidden_states_share)
        else:
            output = self.moe_infer_dispatch_combine(
                hidden_states, topk_indices, topk_weights, hidden_states_share,
                shared_expert_event=self.shared_expert_event if enable_npugraphex_and_multistream else None)[0]

        return output


class BailingMoeV25MLA(nn.Module):

    def __init__(self, config: BailingMoeV25Config, infer_config: InferenceConfig,
                 layer_idx: Optional[int] = None, prefix: Optional[str] = "", **kwargs):
        super().__init__()
        self.config = config
        self.infer_config = infer_config
        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.o_proj_tp_size = infer_config.parallel_config.o_proj_tp_size
        self.batch_size_per_rank = get_batch_size_per_rank(infer_config)
        self.layer_idx = layer_idx
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
            self.q_a_layernorm = BailingMoeV25RMSNorm(config.q_lora_rank)
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
        self.kv_a_layernorm = BailingMoeV25RMSNorm(self.kv_lora_rank)

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

        self.softmax_scale = self.qk_head_dim ** (-0.5)

        self.block_size = infer_config.scheduler_config.block_size
        self.attn_tp_group = self.hccl_comm_dict.get("attn_tp_group", None)
        self.use_fia_v2 = uses_fia_v2(infer_config)

        self.attn_type = "FullAttention"
        self.nope_cache = torch.Tensor([])
        self.rope_cache = torch.Tensor([])
        self.cache_entries = []
        for name, dim in (("nope_cache", self.kv_lora_rank), ("rope_cache", self.qk_rope_head_dim)):
            self.cache_entries.append(CacheEntry(
                cache_name=name,
                attn_type=self.attn_type,
                dim=dim,
                num_head=1,
                dtype=torch.get_default_dtype(),
                needs_block=True,
                block_size=self.block_size,
                tensor_setter=(
                    lambda tensor, layer=self, attr=name: setattr(layer, attr, tensor)
                ),
            ))

    def o_proj_forward(
        self,
        attn_output: torch.Tensor = None,
        token_shard: Optional[TokenShard] = None,
    ):
        bsz, q_len, _ = attn_output.shape
        bsz = (bsz + self.attn_tp_size - 1) // self.attn_tp_size

        # after view: (o_proj_tp_size * bs*q_len, num_heads // self.o_proj_tp_size * v_head_dim)
        attn_output = self.o_proj(attn_output.view(-1, self.num_heads // self.o_proj_tp_size * self.v_head_dim))
        if self.o_proj_tp_size > 1:
            group = self.hccl_comm_dict.get("o_proj_tp_group", None)
            if token_shard is not None:
                return sp_reduce_scatter(attn_output, token_shard, group)
            reduce_scatter_output = torch.empty((attn_output.size()[0] // self.o_proj_tp_size, attn_output.size()[1]),
                                                dtype=attn_output.dtype, device=attn_output.device)
            dist.reduce_scatter_tensor(reduce_scatter_output, attn_output, group=group)
            attn_output = reduce_scatter_output

        return attn_output.view(bsz, q_len, -1)

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
        slot_mapping: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        token_shard: Optional[TokenShard] = None,
    ):
        if self.attn_tp_size > 1:
            hidden_states = sp_gather(
                hidden_states, token_shard, self.hccl_comm_dict.get("attn_tp_group", None))
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

        q_pe = q_pe.transpose(1, 2)
        cos = cos.view(bsz, 1, -1, self.qk_rope_head_dim)
        sin = sin.view(bsz, 1, -1, self.qk_rope_head_dim)
        q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin)# (B, N, S, D)
        q_pe = q_pe.view(bsz, self.num_heads_per_rank, -1, self.qk_rope_head_dim).transpose(1, 2)
        # (B, S, N, D)
        query_states = [q_nope, q_pe]

        latent_cache = latent_cache.view(-1, 1, 1, self.kv_lora_rank + self.qk_rope_head_dim)  # (B,N,S,D)
        nope_cache = self.nope_cache
        rope_cache = self.rope_cache
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

        k_nope_out = torch.matmul(k_nope.view(1, -1, self.kv_lora_rank), self.kv_b_proj_w_k.permute(0, 2, 1))
        v_out = torch.matmul(k_nope.view(1, -1, self.kv_lora_rank), self.kv_b_proj_w_v)

        # NTD foramt, repeat in N
        k_rope = k_rope.view(1, -1, self.qk_rope_head_dim).repeat(self.num_heads_per_rank, 1, 1)

        attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            query_states[0].flatten(0, 1).permute(1, 0, 2), k_nope_out, v_out,
            query_rope=query_states[1].flatten(0, 1).permute(1, 0, 2), key_rope=k_rope,
            num_heads=self.num_heads_per_rank,
            num_key_value_heads=self.num_heads_per_rank,
            input_layout="NTD_TND",
            atten_mask=attention_mask, sparse_mode=3,
            actual_seq_lengths=actual_seq_lengths_kv,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            scale=self.softmax_scale,
            antiquant_mode=0, antiquant_scale=None,
            next_tokens=0
        )
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj_forward(attn_output, token_shard)
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
        slot_mapping: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        token_shard: Optional[TokenShard] = None,
    ):
        query_states, k_nope, k_rope = self.prepare_qkv(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            kv_len=kv_len,
            slot_mapping=slot_mapping,
            block_table=block_table,
            past_key_value=past_key_value,
            token_shard=token_shard,
        )

        attn_output = self.attn_output_forward(
            query_states=query_states,
            k_nope=k_nope,
            k_rope=k_rope,
            attention_mask=attention_mask,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            block_table=block_table,
        )

        return attn_output

    def prepare_qkv(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        kv_len: torch.IntTensor = None,
        slot_mapping: torch.IntTensor = None,
        block_table: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        token_shard: Optional[TokenShard] = None,
    ):
        if self.attn_tp_size > 1:
            hidden_states = sp_gather(
                hidden_states, token_shard, self.hccl_comm_dict.get("attn_tp_group", None))
        input_kwargs = {
            "hidden_states": hidden_states,
            "position_embeddings": position_embeddings,
            "kv_len": kv_len,
            "slot_mapping": slot_mapping,
            "block_table": block_table,
            "past_key_value": past_key_value,
        }
        return self.prepare_qkv_absorb(**input_kwargs)

    def prepare_qkv_absorb(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        kv_len: torch.IntTensor = None,
        slot_mapping: torch.IntTensor = None,
        block_table: Optional[torch.Tensor] = None,
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
        nope_cache = self.nope_cache
        rope_cache = self.rope_cache
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

        kv_cache_nz_dim = 16  # bf16 dtype is 16 for nz format, avoid dynamic shape in high torch version
        k_nope = k_nope.view(block_num, 1, self.kv_lora_rank // kv_cache_nz_dim,
                             block_size, kv_cache_nz_dim)
        k_rope = k_rope.view(block_num, 1, self.qk_rope_head_dim // kv_cache_nz_dim,
                             block_size, kv_cache_nz_dim)

        return query_states, k_nope, k_rope

    def attn_output_forward(
        self,
        query_states: torch.Tensor,
        k_nope: torch.Tensor,
        k_rope: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        actual_seq_lengths_kv: list = None,
        block_table: Optional[torch.Tensor] = None,
    ):
        # query_states here is a list of [q_nope, q_pe] with shape (b, s, n, D)
        bsz, q_len, _, _ = query_states[0].size()

        if q_len > 1: # mtp
            sparse_mode = 3
        else:
            sparse_mode = 0
            attention_mask = None

        block_table = block_table[self.attn_type]

        if self.use_fia_v2:
            num_tokens = bsz * q_len
            q_nope = query_states[0].reshape(num_tokens, self.num_heads_per_rank, self.kv_lora_rank)
            q_pe = query_states[1].reshape(num_tokens, self.num_heads_per_rank, self.qk_rope_head_dim)
            # cumulative q lengths: one request per row, q_len tokens each
            actual_seq_qlen = [(i + 1) * q_len for i in range(bsz)]
            attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score_v2(
                q_nope, k_nope, k_nope,
                query_rope=q_pe, key_rope=k_rope,
                atten_mask=attention_mask,
                actual_seq_qlen=actual_seq_qlen,
                actual_seq_kvlen=actual_seq_lengths_kv,
                block_table=block_table,
                num_query_heads=self.num_heads_per_rank,
                num_key_value_heads=self.num_key_value_heads_per_rank,
                softmax_scale=self.softmax_scale,
                input_layout="TND_NTD",
                sparse_mode=sparse_mode,
                block_size=self.block_size,
            )
            # attn_output: (T, N, kv_lora_rank); project values: (T, N*v_head_dim)
            attn_output = torch_npu.npu_transpose_batchmatmul(
                attn_output, self.kv_b_proj_w_v, bias=None, scale=None,
                perm_x1=(0, 1, 2), perm_x2=(0, 1, 2), perm_y=(1, 0, 2),
            )
            attn_output = attn_output.reshape(bsz, q_len, -1)
            return self.o_proj_forward(attn_output)

        attn_partial, lse_partial = torch.ops.npu.npu_fused_infer_attention_score(
            query_states[0], k_nope, k_nope,
            query_rope=query_states[1], key_rope=k_rope,
            num_heads=self.num_heads_per_rank,
            num_key_value_heads=self.num_key_value_heads_per_rank,
            input_layout="BSND_NBSD",
            block_table=block_table,
            block_size=self.block_size,
            atten_mask=attention_mask,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            scale=self.softmax_scale,
            antiquant_mode=0, antiquant_scale=None,
            sparse_mode=sparse_mode,
            softmax_lse_flag=False,
        )
        attn_partial = attn_partial.view(self.num_heads_per_rank, -1, self.kv_lora_rank)
        attn_partial = (
            torch.matmul(attn_partial, self.kv_b_proj_w_v)
        )
        attn_output = attn_partial
        # (N, B*S, D) -> (B*S, N, D) -> (B, S, N * D)
        attn_output = attn_output.transpose(1, 0).reshape(bsz, q_len, -1)
        return self.o_proj_forward(attn_output)

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
        block_table: Optional[torch.Tensor] = None,
        token_shard: Optional[TokenShard] = None,
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
            "slot_mapping": slot_mapping,
            "block_table": block_table,
            "token_shard": token_shard,
        }
        if is_prefill:
            fn = self.forward_page_attention_normal
        else:
            fn = self.forward_page_attention_absorb
        return fn(**input_kwargs)


class BailingMoeV25LinearAttention(nn.Module):
    """
    BailingMoeAttention implements a linear attention mechanism based on Lightning Attention-2
    (https://arxiv.org/abs/2401.04658) with efficient computation using flash-linear-attention operators.
    The implementation leverages optimized kernels from the flash-linear-attention library
    (https://github.com/fla-org/flash-linear-attention) for maximum performance.
    """
    def __init__(self, config: BailingMoeV25Config, infer_config: InferenceConfig,
                 layer_idx: Optional[int] = None, prefix: Optional[str] = "", **kwargs):
        super().__init__()
        self.config = config

        self.infer_config = infer_config
        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        self.o_proj_tp_size = infer_config.parallel_config.o_proj_tp_size

        self.hccl_comm_dict = kwargs.get("hccl_comm_dict", None)

        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_heads_per_rank = self.num_heads // self.attn_tp_size
        self.head_dim = config.head_dim or self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_attention_heads
        self.num_key_value_heads_per_rank = config.num_attention_heads // self.attn_tp_size

        self.attn_type = "Mamba"
        self.gla_state_cache = torch.Tensor([])
        self.cache_entries = [
            MambaCacheEntry(
                cache_name="gla_recurrent_state",
                dtype=torch.get_default_dtype(),
                needs_block=True,
                shape=[self.num_heads_per_rank, self.head_dim, self.head_dim],
                tensor_setter=(
                    lambda tensor, layer=self: setattr(layer, "gla_state_cache", tensor)
                ),
            )
        ]

        self.rms_norm_eps = config.rms_norm_eps

        self.query_key_value = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_key_value_heads,
            bias=config.use_qkv_bias,
            skip_bias_add=False,
            tp_size=self.attn_tp_size,
            tp_rank=dist.get_rank(self.hccl_comm_dict["attn_tp_group"]) if self.attn_tp_size > 1 else 0,
            quant_config=config.quant_config,
            prefix=f"{prefix}.merged_qkv_proj",
            return_bias=False
        )

        if self.config.use_qk_norm:
            self.query_layernorm = BailingMoeV25RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = BailingMoeV25RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.o_proj = RowParallelLinear(self.num_heads * self.head_dim,
                                self.hidden_size,
                                tp_size=self.o_proj_tp_size,
                                tp_rank=dist.get_rank(self.hccl_comm_dict["o_proj_tp_group"])
                                if self.o_proj_tp_size > 1 else 0,
                                bias=config.use_bias,
                                input_is_parallel=True,
                                quant_config=config.quant_config,
                                prefix=f"{prefix}.o_proj")

        # output gate, column-parallel by head (same head layout as query_key_value / o_proj / slope)
        self.g_proj = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.num_heads * self.head_dim,
            bias=False,
            tp_size=self.attn_tp_size,
            tp_rank=dist.get_rank(self.hccl_comm_dict["attn_tp_group"]) if self.attn_tp_size > 1 else 0,
            quant_config=config.quant_config,
            prefix=f"{prefix}.g_proj",
        )
        # Group RMSNorm must keep the same channels-per-group as the unsharded HF model, so the local
        # group count shrinks with the shard. Refuse shards that split an HF group across ranks.
        hf_group_channels = (self.num_heads * self.head_dim) // config.group_norm_size
        local_gnorm_width = self.num_heads_per_rank * self.head_dim
        if local_gnorm_width % hf_group_channels != 0:
            raise ValueError(
                f"attn_tp_size ({self.attn_tp_size}) splits a g_norm group across ranks: local width "
                f"{local_gnorm_width} is not divisible by group channels {hf_group_channels}; "
                f"attn_tp_size must divide group_norm_size={config.group_norm_size}.")
        self.g_norm = BailingMoeV25GroupRMSNorm(local_gnorm_width,
            group_norm_size=local_gnorm_width // hf_group_channels,
            eps=self.rms_norm_eps,
        )
        # the checkpoint stores the full num_heads*head_dim g_norm weight, so attach a narrowing
        # loader for this rank's head slice
        if self.attn_tp_size > 1:
            g_tp_rank = dist.get_rank(self.hccl_comm_dict["attn_tp_group"])

            def _g_norm_weight_loader(param, loaded_weight, _w=local_gnorm_width, _r=g_tp_rank):
                param.data.copy_(loaded_weight.narrow(0, _r * _w, _w))
            self.g_norm.weight.weight_loader = _g_norm_weight_loader

        slope_decay = 1 - (self.layer_idx - 1) / (self.config.num_hidden_layers - 1) + 1e-5
        full_slope = -BailingMoeV25LinearAttention.build_slope_tensor(self.num_heads) * slope_decay

        if self.attn_tp_size > 1:
            tp_rank = dist.get_rank(self.hccl_comm_dict["attn_tp_group"])
            h = self.num_heads_per_rank
            slope = full_slope[tp_rank * h:(tp_rank + 1) * h].contiguous()
        else:
            slope = full_slope

        self.register_buffer('slope', slope, persistent=False)

        self.lightning_attn_ops = {
            'chunk': chunk_simple_gla_torch,
            'fused_recurrent': fused_recurrent_simple_gla_torch,
        }

    @staticmethod
    def build_slope_tensor(n_attention_heads: int):
        """
        Build a tensor of slopes for Lightning Attention-2 as described in the paper:
        "Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models"
        (https://arxiv.org/abs/2401.04658)
        This function computes the slope values that control the decay rate of attention scores
        based on the number of attention heads. The slopes are designed to have specific
        mathematical properties that work optimally when the number of heads is a power of 2.
        For non-power-of-2 head counts, a workaround is implemented to maintain similar properties.
        Args:
            n_attention_heads (int): Number of attention heads in the model
        Returns:
            torch.Tensor: A tensor of shape [n_attention_heads] containing the computed slopes
        Note:
            Code copied from lightning-attention (OpenNLPLab), lightning_attn/utils/utils.py#L6:
            https://github.com/OpenNLPLab/lightning-attention/blob/d15c3852/lightning_attn/utils/utils.py
        """

        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(
                    n
                )  # In the paper, we only train models that have 2^a heads for some a. This function has
            else:  # some good properties that only occur when the input is a power of 2. To maintain that even
                closest_power_of_2 = 2 ** math.floor(
                    math.log2(n)
                )  # when the number of heads is not a power of 2, we use this workaround.
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
                )

        slopes = torch.tensor(get_slopes(n_attention_heads), dtype=torch.float)
        return slopes

    @staticmethod
    def _request_rows(q_lens, num_tokens, device):
        """Row index in [num_requests, row_len] for every packed token."""
        q_lens = q_lens.to(device=device, dtype=torch.long).reshape(-1)
        num_requests = int(q_lens.numel())
        row_len = int(q_lens.max().item())
        starts = torch.cumsum(q_lens, dim=0) - q_lens
        token_idx = torch.arange(num_tokens, device=device)
        request = torch.bucketize(token_idx, starts[1:], right=True) if num_requests > 1 \
            else torch.zeros(num_tokens, device=device, dtype=torch.long)
        return request * row_len + (token_idx - starts[request]), num_requests, row_len

    def o_proj_forward(
        self,
        attn_output: torch.Tensor = None,
        token_shard: Optional[TokenShard] = None,
    ):
        bsz, q_len, _ = attn_output.shape
        bsz = (bsz + self.attn_tp_size - 1) // self.attn_tp_size

        # after view: (o_proj_tp_size * bs*q_len, num_heads // self.o_proj_tp_size * head_dim)
        attn_output = self.o_proj(attn_output.view(-1, self.num_heads // self.o_proj_tp_size * self.head_dim))
        if self.o_proj_tp_size > 1:
            group = self.hccl_comm_dict.get("o_proj_tp_group", None)
            if token_shard is not None:
                return sp_reduce_scatter(attn_output, token_shard, group)
            reduce_scatter_output = torch.empty((attn_output.size()[0] // self.o_proj_tp_size, attn_output.size()[1]),
                                                dtype=attn_output.dtype, device=attn_output.device)
            dist.reduce_scatter_tensor(reduce_scatter_output, attn_output, group=group)
            attn_output = reduce_scatter_output

        return attn_output.view(bsz, q_len, -1)

    def forward_linear_attention(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = True,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        is_prefill: bool = True,
        token_shard: Optional[TokenShard] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if self.attn_tp_size > 1:
            hidden_states = sp_gather(
                hidden_states, token_shard, self.hccl_comm_dict.get("attn_tp_group", None))

        bsz, q_len, _ = hidden_states.size()

        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(bsz, q_len, self.num_heads_per_rank + 2 * self.num_key_value_heads_per_rank, self.head_dim)
        query_states, key_states, value_states = qkv.split(
            [self.num_heads_per_rank, self.num_key_value_heads_per_rank, self.num_key_value_heads_per_rank], dim=-2
        )
        if self.config.use_qk_norm:
            query_states = self.query_layernorm(query_states)
            key_states = self.key_layernorm(key_states)

        cos, sin = position_embeddings  # [seq_len, 1, head_dim]

        if cos.dim() == 3:
            # prefill phase: [T, 1, D]
            cos = cos[:q_len].unsqueeze(0)   # [1, S, 1, D]
            sin = sin[:q_len].unsqueeze(0)
        else:
            if cos.shape[0] != bsz: # Divide by TP
                attn_tp_group = self.hccl_comm_dict.get("attn_tp_group", None)
                tp_rank = (
                    dist.get_rank(group=attn_tp_group)
                    if attn_tp_group is not None else 0
                )
                cos = cos[tp_rank * bsz:(tp_rank + 1) * bsz]   # [B_local, 1, 1, D]
                sin = sin[tp_rank * bsz:(tp_rank + 1) * bsz]

        rope_dim = cos.shape[-1]
        q_rot, q_pass = query_states[..., :rope_dim], query_states[..., rope_dim:]
        k_rot, k_pass = key_states[..., :rope_dim], key_states[..., rope_dim:]
        q_rot, k_rot = torch_npu.npu_apply_rotary_pos_emb(q_rot, k_rot, cos, sin, layout='BSND')
        query_states = torch.cat([q_rot, q_pass], dim=-1)
        key_states = torch.cat([k_rot, k_pass], dim=-1)

        gk = self.slope[None, None, :].expand(bsz, q_len, self.num_heads_per_rank)
        rows = bsz
        if is_prefill:
            # the recurrence runs per request, so lay the packed stream out one request per row;
            # the zero fill leaves gk=0 (decay 1) and v=0 on the tail, carrying the state through
            scatter_index, rows, row_len = self._request_rows(
                kwargs.get("kv_len"), q_len, value_states.device)
            query_states, key_states, value_states = (
                _scatter_to_rows(t.reshape(q_len, *t.shape[2:]), scatter_index, rows, row_len)
                for t in (query_states, key_states, value_states))
            gk = _scatter_to_rows(gk.reshape(q_len, -1), scatter_index, rows, row_len)

        block_table = kwargs.get("block_table")
        state_ids = None
        initial_state = None
        if block_table is not None:
            table = block_table[self.attn_type]
            if table.shape[0] == rows:
                state_ids = table[:, 0].to(torch.int32)
            else:
                num_requests = min(table.shape[0], rows)
                state_ids = table.new_zeros((rows,), dtype=torch.int32)
                state_ids[:num_requests] = table[:num_requests, 0].to(torch.int32)
            if not is_prefill:
                initial_state = self.gla_state_cache.index_select(0, state_ids)

        if is_prefill:
            attn_out, new_state = self.lightning_attn_ops['chunk'](
                    q=query_states,
                    k=key_states,
                    v=value_states,
                    gk=gk,
                    initial_state=initial_state,
                    output_final_state=use_cache,
                )
            attn_out = _gather_from_rows(attn_out, scatter_index)
        else:
            attn_out, new_state = self.lightning_attn_ops['fused_recurrent'](
                    q=query_states,
                    k=key_states,
                    v=value_states,
                    gk=gk,
                    initial_state=initial_state,
                    output_final_state=use_cache,
                    inplace_state=initial_state is not None,
                )

        if use_cache and state_ids is not None and new_state is not None:
            self.gla_state_cache.index_copy_(0, state_ids, new_state.to(self.gla_state_cache.dtype))

        attn_out = attn_out.reshape(bsz, q_len, -1)
        attn_out = self.g_norm(attn_out)
        g_proj = self.g_proj(hidden_states)
        attn_out = attn_out * torch.sigmoid_(g_proj)
        attn_out = self.o_proj_forward(attn_out, token_shard)

        return attn_out

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
        block_table: Optional[torch.Tensor] = None,
        token_shard: Optional[TokenShard] = None,
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
            "slot_mapping": slot_mapping,
            "block_table": block_table,
            "token_shard": token_shard,
        }
        return self.forward_linear_attention(**input_kwargs)


class BailingMoeV25DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: BailingMoeV25Config, infer_config: InferenceConfig, layer_idx: int,
                 prefix: str, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.infer_config = infer_config
        self.hidden_size = config.hidden_size
        self.hccl_comm_dict = kwargs.get("hccl_comm_dict", None)
        self.global_rank = dist.get_rank()

        self.attention_layer_type = (
            "attention"
            if (layer_idx + 1) % config.layer_group_size == 0
            or layer_idx >= config.num_hidden_layers // config.layer_group_size * config.layer_group_size
            else "linear_attention"
        )
        if self.attention_layer_type == "attention":
            self.self_attn = BailingMoeV25MLA(
                config=config,
                infer_config=infer_config,
                layer_idx=layer_idx,
                prefix=f"{prefix}.self_attn",
                **kwargs
            )
        else:
            self.self_attn = BailingMoeV25LinearAttention(
                config=config,
                infer_config=infer_config,
                layer_idx=layer_idx,
                prefix=f"{prefix}.linear_attn",
                **kwargs
            )

        self.mlp = (
            BailingMoeV25MoE(config, infer_config, layer_idx, prefix=f"{prefix}.mlp", **kwargs)
            if (config.num_experts is not None and layer_idx >= config.first_k_dense_replace)
            else BailingMoeV25MLP(config, infer_config, f"{prefix}.mlps", **kwargs)
        )

        self.input_layernorm = BailingMoeV25RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = BailingMoeV25RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_len: torch.IntTensor,
        actual_seq_lengths_kv: list,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        position_embeddings_mla: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        is_prefill: Optional[bool] = False,
        slot_mapping: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        past_residual: Optional[torch.Tensor] = None,
        cur_topk_list: Optional[torch.Tensor] = None,
        token_shard: Optional[TokenShard] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = past_residual
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if self.attention_layer_type == "attention":
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                kv_len=kv_len,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                position_embeddings=position_embeddings_mla,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                is_prefill=is_prefill,
                slot_mapping=slot_mapping,
                block_table=block_table,
                token_shard=token_shard,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                kv_len=kv_len,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                is_prefill=is_prefill,
                slot_mapping=slot_mapping,
                block_table=block_table,
                token_shard=token_shard,
            )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states = self.mlp(
            hidden_states,
            is_prefill,
            cur_topk_list=cur_topk_list
        )

        outputs = (hidden_states, residual)
        return outputs


class BailingMoeV25PreTrainedModel(PreTrainedModel):
    config: BailingMoeV25Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BailingMoeV25DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": BailingMoeV25DecoderLayer,
        "attentions": BailingMoeV25MLA,
    }


class BailingMoeV25Model(BailingMoeV25PreTrainedModel):

    def __init__(self, config: BailingMoeV25Config, infer_config: InferenceConfig, **kwargs):
        super().__init__(config)
        self.config = config
        self.global_rank = dist.get_rank()
        self.infer_config = infer_config
        self.embed_tp_size = infer_config.parallel_config.embed_tp_size

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.vocab_size_per_rank = self.vocab_size // self.embed_tp_size


        self.hccl_comm_dict = kwargs.get("hccl_comm_dict", None)
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            self.padding_idx,
            torch.bfloat16,
            tp_size=self.embed_tp_size,
            tp_rank=dist.get_rank(self.hccl_comm_dict["embed_tp_group"]) if self.embed_tp_size > 1 else 0)

        if config.num_nextn_predict_layers > 0:
            raise NotImplementedError("num_nextn_predict_layers > 0 (MTP) is not supported")
        self.layers = []
        for layer_idx in range(config.num_hidden_layers):
            self.layers.append(BailingMoeV25DecoderLayer(config, infer_config, layer_idx, \
                                         prefix=f"model.layers.{layer_idx}", **kwargs))
        self.layers = nn.ModuleList(self.layers)

        enable_multi_streams = infer_config.model_config.custom_params.get("enable_multi_streams", False)
        enable_npugraph_ex = infer_config.model_config.exe_mode == "npugraph_ex"
        shared_expert_stream = (
            torch.npu.Stream() if (enable_multi_streams and enable_npugraph_ex) else None
        )
        for layer in self.layers:
            if isinstance(layer.mlp, BailingMoeV25MoE):
                layer.mlp.shared_expert_stream = shared_expert_stream

        self.norm = BailingMoeV25RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if config.rope_scaling is not None and config.rope_scaling.get("rope_type") != "default":
            raise NotImplementedError(
                f"rope_scaling={config.rope_scaling} is not supported; only default RoPE is implemented")
        # MLA uses interleaved RoPE, linear attention use partial rotate-half
        self.rotary_emb_mla = BailingMoeV25RotaryEmbedding(
            self.config.qk_rope_head_dim,
            max_position_embeddings=self.config.max_position_embeddings,
            base=self.config.rope_theta,
        )
        self.rotary_emb = BailingMoeV25RotaryEmbedding(
            int(self.config.head_dim * self.config.partial_rotary_factor),
            max_position_embeddings=self.config.max_position_embeddings,
            base=self.config.rope_theta,
        )
        self.gradient_checkpointing = False
        self.post_init()

    def prepare_inputs_for_layer(self, input_ids, kv_len, global_position_ids, actual_seq_lengths_kv, is_prefill,
                                 slot_mapping=None, token_shard=None):
        if input_ids.dim() != 2:
            raise RuntimeError(f"expect a 2-D token shard, got {tuple(input_ids.shape)}")
        batch_size, seq_length = input_ids.shape

        if self.embed_tp_size > 1:
            embed_tp_group = self.hccl_comm_dict.get("embed_tp_group", None)
            all_input_ids = input_ids.new_empty(batch_size * self.embed_tp_size, seq_length)
            dist.all_gather_into_tensor(all_input_ids, input_ids, group=embed_tp_group)

            # shift into this rank's vocab shard, then zero out (and mask off the embedding of)
            # the tokens that fall outside it
            new_input_ids = all_input_ids - (
                    self.global_rank % self.embed_tp_size) * self.vocab_size_per_rank
            mask = (new_input_ids >= 0) & (new_input_ids < self.vocab_size_per_rank) # (bs, qlen)
            new_input_ids_per_rank = new_input_ids * mask
            inputs_embeds = self.embed_tokens(new_input_ids_per_rank) * mask.unsqueeze(-1)

            inputs_embeds_attn = inputs_embeds.new_empty(batch_size, seq_length, inputs_embeds.shape[-1])
            dist.reduce_scatter_tensor(inputs_embeds_attn, inputs_embeds, group=embed_tp_group)
            inputs_embeds = inputs_embeds_attn

        else:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        # rope and slot_mapping cover the stream the attention sees after its all_gather
        position_embeddings = self.rotary_emb(hidden_states, kv_len, self.config.max_position_embeddings, \
                                              is_prefill=is_prefill, position_ids=global_position_ids)
        position_embeddings_mla = self.rotary_emb_mla(hidden_states, kv_len, self.config.max_position_embeddings, \
                                              is_prefill=is_prefill, position_ids=global_position_ids)

        residual = None
        return LayerInputs(
            hidden_states=hidden_states,
            residual=residual,
            kv_len=kv_len,
            position_embeddings=position_embeddings,
            position_embeddings_mla=position_embeddings_mla,
            slot_mapping=slot_mapping,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        global_position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_len: Optional[torch.IntTensor] = None,
        actual_seq_lengths_kv: Optional[list] = None,
        is_prefill: Optional[bool] = False,
        cur_topk_list: Optional[torch.Tensor] = None,
        slot_mapping=None,
        block_table=None,
        token_shard=None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        layer_inputs = self.prepare_inputs_for_layer(
            input_ids, kv_len, global_position_ids, actual_seq_lengths_kv, is_prefill,
            slot_mapping=slot_mapping, token_shard=token_shard)
        hidden_states = layer_inputs.hidden_states
        residual = layer_inputs.residual
        kv_len = layer_inputs.kv_len
        position_embeddings = layer_inputs.position_embeddings
        position_embeddings_mla = layer_inputs.position_embeddings_mla
        slot_mapping = layer_inputs.slot_mapping
        actual_seq_lengths_kv = layer_inputs.actual_seq_lengths_kv

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states, residual = decoder_layer(
                hidden_states,
                kv_len,
                actual_seq_lengths_kv,
                position_embeddings=position_embeddings,
                position_embeddings_mla=position_embeddings_mla,
                attention_mask=attention_mask,
                position_ids=global_position_ids,
                past_key_value=past_key_values,
                is_prefill=is_prefill,
                slot_mapping=slot_mapping,
                block_table=block_table,
                past_residual=residual,
                cur_topk_list=cur_topk_list,
                token_shard=token_shard,
                **kwargs,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


@auto_docstring
class BailingMoeV25ForCausalLM(BailingMoeV25PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config, infer_config: InferenceConfig, comm_manager=None, **kwargs):
        super().__init__(config)
        self.config = config
        self.comm_manager = comm_manager
        self.infer_config = infer_config
        self.num_experts = config.n_routed_experts
        self.num_experts_per_tok = config.moe_topk
        self.get_parallel_settings()
        kwargs = {}
        default_pg = get_default_group()
        if default_pg is not None:
            if dist.get_world_size() > 1:
                self.hccl_comm_dict = self.init_parallel_comm_group()
                kwargs.update({"hccl_comm_dict": self.hccl_comm_dict})

        self.attn_tp_rank = (
            dist.get_rank(self.hccl_comm_dict["attn_tp_group"]) if self.attn_tp_size > 1 else 0)
        self.batch_size_per_rank = get_batch_size_per_rank(infer_config)
        self.pa_max_length = get_pa_max_length(infer_config)
        self.block_size = infer_config.scheduler_config.block_size
        self.model = BailingMoeV25Model(config, infer_config, **kwargs)
        self.vocab_size = config.vocab_size
        self.lm_head = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            bias=False,
            tp_size=self.lmhead_tp_size,
            tp_rank=dist.get_rank(self.hccl_comm_dict.get("lmhead_tp_group")) if self.lmhead_tp_size > 1 else 0
        )
        self.post_init()

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def _shard_topk_list_for_attn_tp(self, cur_topk_list, is_prefill):
        """force_eplb builds the prefill list per rank already; the decode one covers the batch."""
        if cur_topk_list is None or is_prefill or self.attn_tp_size == 1:
            return cur_topk_list
        rows = self.batch_size_per_rank
        start = self.attn_tp_rank * rows
        return cur_topk_list[start:start + rows]

    def _shard_for_attn_tp(self, input_ids, token_shard):
        """Contiguous token slice per rank, so the attention all_gather restores the token order."""
        flat = input_ids.reshape(-1)
        if token_shard.padded_tokens > flat.numel():
            flat = torch.cat([flat, flat.new_zeros(token_shard.padded_tokens - flat.numel())])
        start = self.attn_tp_rank * token_shard.tokens_per_rank
        return flat[start:start + token_shard.tokens_per_rank].view(*token_shard.local_shape)

    def _build_attn_inputs(self, position_ids, forward_metadata, token_shard):
        fm = forward_metadata
        if fm.is_prefill:
            # request bounds inside the packed stream; unequal lengths need no padding here.
            # Prefill starts from scratch, so a request's kv length is its query length.
            kv_len = torch.as_tensor(
                fm.actual_seq_lengths_q, device=position_ids.device).view(-1).to(torch.long)
            actual_seq_lengths_kv = torch.cumsum(kv_len, dim=0)
            attention_mask = fm.attention_mask
        else:
            kv_len = fm.kv_len
            attention_mask = None
            position_ids = kv_len.view(-1, 1)
            framework_list = getattr(fm, "actual_seq_lengths_list_kv", None)
            if framework_list is not None:
                actual_seq_lengths_kv = list(framework_list)
                return kv_len, actual_seq_lengths_kv, attention_mask, position_ids
            actual_seq_lengths_kv = kv_len + 1

        actual_seq_lengths_kv = actual_seq_lengths_kv.view(-1).cpu().detach().tolist()
        return kv_len, actual_seq_lengths_kv, attention_mask, position_ids

    def get_cache_info(self) -> Optional[ModelCacheInfo]:
        layer_infos = []
        for layer_idx, layer in enumerate(self.model.layers):
            entries = getattr(layer.self_attn, "cache_entries", None)
            if not entries:
                continue
            layer_infos.append(LayerCacheInfo(layer_idx=layer_idx, caches=list(entries)))
        if not layer_infos:
            return None
        return ModelCacheInfo(
            num_layers=len(layer_infos),
            layer_infos=layer_infos,
            is_mla_backend=True,
        )

    def check_model_settings(self):
        from .model_setting import check_vars
        check_vars(self.infer_config)

    def init_splited_kv_b_weight(self):
        def _init_k(layer):
            try:
                data_tensor = attrgetter("kv_b_proj_w_k_data")(layer.self_attn)
                layer.self_attn.kv_b_proj_w_k = nn.Parameter(data_tensor.contiguous(), requires_grad=False)
            except AttributeError:
                pass

        def _init_v(layer):
            try:
                data_tensor = attrgetter("kv_b_proj_w_v_data")(layer.self_attn)
                layer.self_attn.kv_b_proj_w_v = nn.Parameter(data_tensor.contiguous(), requires_grad=False)
            except AttributeError:
                pass

        def _offload_kv_b(layer):
            try:
                layer.self_attn.kv_b_proj.weight = None
            except AttributeError:
                pass

        for layer in self.model.layers:
            _init_k(layer)
            _init_v(layer)
            _offload_kv_b(layer)
        gc.collect()

    def process_weights_after_loading(self):
        self.init_splited_kv_b_weight()
        enable_weight_nz = self.infer_config.model_config.enable_weight_nz
        float_scales_map = ["gate_up_proj"]
        float_smooth_scales_map = ["down_proj"]
        for module_name, module in self.named_modules():
            if "kv_b_proj" in module_name:
                continue
            quant_method = getattr(module, "quant_method", None)
            scales_dtype = {}
            for scale_name in float_scales_map:
                if scale_name in module_name:
                    scales_dtype["scale_dtype"] = torch.float
                    break
            for smooth_scale_name in float_smooth_scales_map:
                if smooth_scale_name in module_name:
                    scales_dtype["smooth_scale_dtype"] = torch.float
                    break
            if isinstance(quant_method, QuantizeMethodBase):
                quant_method.process_weights_after_loading(module, is_nz=enable_weight_nz, scales_dtype=scales_dtype)

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

    def get_parallel_settings(self):
        pc = self.infer_config.parallel_config
        self.embed_tp_size = pc.embed_tp_size
        self.attn_dp_size = pc.attn_dp_size
        self.attn_tp_size = pc.attn_tp_size
        self.o_proj_tp_size = pc.o_proj_tp_size
        self.moe_ep_size = pc.moe_ep_size
        self.moe_tp_size = pc.moe_tp_size
        self.lmhead_tp_size = pc.lmhead_tp_size
        self.dense_tp_size = pc.dense_tp_size

    def init_parallel_comm_group(self):
        world_size = dist.get_world_size()
        cm = self.comm_manager

        def tp_group(name, tp_size, stride=1):
            return cm.register_group(name=name, group_num=world_size // tp_size,
                                     group_size=tp_size, group_stride=stride)

        # CommManager dedupes groups that end up on the same ranks, so every logical group
        # is registered by name rather than aliased by hand
        attn_tp_group = tp_group("attn_tp_group", self.attn_tp_size)
        o_proj_tp_group = tp_group("o_proj_tp_group", self.o_proj_tp_size)
        embed_tp_group = tp_group("embed_tp_group", self.embed_tp_size)
        lmhead_tp_group = tp_group("lmhead_tp_group", self.lmhead_tp_size)
        dense_tp_group = tp_group("dense_tp_group", self.dense_tp_size)
        moe_tp_group = tp_group("moe_tp_group", self.moe_tp_size)
        moe_ep_group = tp_group("moe_ep_group", self.moe_ep_size, stride=self.moe_tp_size)

        # MC2 needs its own communicator: a reused one is built without the aiv expansion mode
        # and the dispatch then fails its link-protocol check. group_type, buffer size and the
        # comm_alg set_mc2_kwargs passes all follow the platform and must not disagree.
        full_mesh = uses_mc2_full_mesh(self.infer_config)
        cm.register_group(
            name="moe_ep_group_mc2", group_num=self.moe_tp_size,
            group_size=world_size // self.moe_tp_size, group_stride=self.moe_tp_size,
            hccl_buffer_size=calc_moe_hccl_buffer_size(
                self.infer_config, self.config, is_full_mesh_v2=full_mesh),
            group_type=None if full_mesh else 3,
            return_name=True, allow_physical_reuse=False)

        return {
                "default_pg": get_default_group(),
                "attn_tp_group": attn_tp_group, "embed_tp_group": embed_tp_group,
                "o_proj_tp_group": o_proj_tp_group,
                "moe_tp_group": moe_tp_group,
                "moe_ep_group": moe_ep_group,
                "moe_ep_group_mc2": cm.get_group("moe_ep_group_mc2"),
                "moe_ep_group_mc2_name": cm.get_group_name("moe_ep_group_mc2"),
                "lmhead_tp_group": lmhead_tp_group,
                "dense_tp_group": dense_tp_group,
            }

    def forward_lm_head(self, outputs, is_prefill=False, actual_seq_lengths_cu_q=None):
        hidden_size = outputs.shape[-1]
        q_len = 1
        if is_prefill:
            # packed stream: each request's last token sits at cu_q - 1, whatever its length
            seq_index = actual_seq_lengths_cu_q.to(dtype=torch.long, device=outputs.device) - 1
            bs = seq_index.numel()
            outputs = torch.index_select(outputs.view(-1, hidden_size), 0, seq_index).view(bs, 1, hidden_size)
        else:
            bs = outputs.shape[0]
            outputs = outputs.reshape(bs, 1, hidden_size)

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
        position_ids: Optional[torch.LongTensor] = None,
        forward_metadata: "ForwardMetaData" = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Framework interface: forward(input_ids, position_ids, forward_metadata)
        -> (logits, prev_hidden_states), unpacking forward_metadata for the layer stack.
        """
        fm = forward_metadata
        is_prefill = fm.is_prefill

        if input_ids is not None and input_ids.dim() == 1:
            # prefill runs as one packed row, decode as one token per request
            shape = (1, -1) if is_prefill else (-1, 1)
            input_ids = input_ids.view(*shape)
            position_ids = position_ids.view(*shape) if position_ids is not None else None

        if is_prefill:
            token_shard = prefill_token_shard(input_ids.numel(), self.attn_tp_size)
        else:
            token_shard = decode_token_shard(self.batch_size_per_rank * self.attn_tp_size, self.attn_tp_size)
        request_bsz = input_ids.shape[0]

        kv_len, actual_seq_lengths_kv, attention_mask, global_position_ids = self._build_attn_inputs(
            position_ids, fm, token_shard)
        input_ids = self._shard_for_attn_tp(input_ids, token_shard)

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_position_ids=global_position_ids,
            past_key_values=None,
            inputs_embeds=None,
            kv_len=kv_len,
            is_prefill=is_prefill,
            cur_topk_list=self._shard_topk_list_for_attn_tp(kwargs.get("cur_topk_list", None), is_prefill),
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            slot_mapping=fm.slot_mapping["FullAttention"],
            block_table=fm.block_table,
            token_shard=token_shard,
        )

        hidden_states = outputs.last_hidden_state
        if self.attn_tp_size > 1:
            hidden_states = sp_gather(
                hidden_states, token_shard, self.hccl_comm_dict.get("attn_tp_group", None))
        if is_prefill:
            logits = self.forward_lm_head(hidden_states, is_prefill=True,
                                          actual_seq_lengths_cu_q=fm.actual_seq_lengths_cu_q)
        else:
            logits = self.forward_lm_head(hidden_states, is_prefill=False)[:request_bsz]
        prev_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
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
            # Bring original-HF checkpoint names to the runtime module names, so a raw bf16 HF
            # checkpoint loads without name mismatches. Mirrors rename_hf_to_runtime in
            # utils/convert_model.py, and is idempotent on already-converted checkpoints.
            if name.startswith("model.word_embeddings."):
                name = name.replace("model.word_embeddings.", "model.embed_tokens.", 1)
            if ".attention." in name:
                name = name.replace(".attention.", ".self_attn.")
                name = name.replace(".self_attn.dense.", ".self_attn.o_proj.")
            if name.endswith(".mlp.gate.weight"):
                name = name[: -len(".mlp.gate.weight")] + ".mlp.router.classifier.weight"
            elif name.endswith(".mlp.gate.expert_bias"):
                name = name[: -len(".mlp.gate.expert_bias")] + ".mlp.router.e_score_correction_bias"

            # the checkpoint stores the router correction bias as `e_score_correction_bias`,
            # the gate registers it as `expert_bias`
            name = name.replace("e_score_correction_bias", "expert_bias")

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


__all__ = ["BailingMoeV25PreTrainedModel", "BailingMoeV25Model", "BailingMoeV25ForCausalLM"]
