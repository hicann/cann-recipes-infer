# coding=utf-8
# Adapted from
# https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

""" PyTorch DeepSeek model."""
import os
import math
import json
from typing import List, Optional, Tuple, Union, Dict, Iterable, Set, Literal
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from torch import nn
import torch.distributed as dist

import torch_npu
import torchair as tng
import custom_ops

from transformers.cache_utils import Cache
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)

from executor.utils import (
    override, get_init_attn_mask, align_up, calc_moe_hccl_buffer_size,
    init_comm_group, get_default_group, get_decode_mask)

from executor.model_loader.weight_utils import default_weight_loader
from executor.utils import (
    superkernel_scope, weight_dequant,
    limit_core_num)
from executor.utils.stream_utils import npu_stream_switch, record_event, wait_event, record_stream
from module.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding
    )
from module.fuse_moe_gmm import FusedMoEGMM
from module.quantization.utils.quant_utils import reshape_mx_scale
from .configuration_deepseek import DeepseekV3Config
from .modules import (get_window_topk_idxs, get_compress_topk_idxs,
                      one_hot, yarn_get_mscale,
                      DeepseekV3RMSNorm, _init_rope, DEEPSEEKV3_START_DOCSTRING,
                      DEEPSEEKV3_INPUTS_DOCSTRING, DeepseekV3PreTrainedModel, apply_rotary_emb
                    )
from .modules import Indexer, Compressor, AttnMetaData, CacheData
from .modules.registry import OpKernel
logger = logging.get_logger(__name__)

HADAMARD_SIZE = 128


class DeepseekV3SharedExpert(nn.Module):
    def __init__(self, config, runner_settings, is_moe_layer=False, prefix="", **kwargs):
        super().__init__()
        self.runner_settings = runner_settings
        self.mm_quant_mode = (
            config.quant_config.mm_quant_mode
            if config.quant_config is not None
            else "w16a16")
        self.moe_tp_size = self.runner_settings.get("parallel_config").get("moe_tp_size", 1)
        self.moe_ep_size = self.runner_settings.get("parallel_config").get("moe_ep_size", 1)
        self.config = config
        self.hidden_size = config.hidden_size
        self.is_moe_layer = is_moe_layer
        self.hccl_comm_dict = kwargs.get("hccl_comm_dict", None)
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[config.moe_intermediate_size * config.n_shared_experts] * 2,
            bias=False,
            tp_size=self.moe_tp_size,
            tp_rank=dist.get_rank(self.hccl_comm_dict.get("moe_tp_group")) if self.moe_tp_size > 1 else 0,
            quant_config=config.quant_config,
            prefix=f"{prefix}.gate_up_proj",
            )
        self.down_proj = RowParallelLinear(
            config.moe_intermediate_size * config.n_shared_experts,
            config.hidden_size,
            bias=False,
            tp_size=self.moe_tp_size,
            tp_rank=dist.get_rank(self.hccl_comm_dict["moe_tp_group"]) if self.moe_tp_size > 1 else 0,
            quant_config=config.quant_config,
            prefix=f"{prefix}.down_proj",
            )
        if self.mm_quant_mode == "w8a8int8":
            self.forward = self.forward_w8a8int8
        elif "float8" in self.mm_quant_mode and "a8" in self.mm_quant_mode:
            self.forward = self.forward_a8float8
        else:
            self.forward = self.forward_normal

    def forward_normal(self, x):
        merged_x = self.gate_up_proj(x)
        intermediate_hidden_states = torch_npu.npu_swiglu(merged_x)
        return self.down_proj(intermediate_hidden_states)

    def forward_w8a8int8(self, x):
        merged_x, pertoken_scale = self.gate_up_proj(x, out_dtype=torch.int32)
        intermediate_hidden_states, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
            merged_x, weight_scale=self.gate_up_proj.weight_scale,
            quant_scale=self.down_proj.smooth_scales,
            quant_mode=1, activate_left=True,
            activation_scale=pertoken_scale,
        )
        return self.down_proj(intermediate_hidden_states, pertoken_scale)

    def forward_a8float8(self, x):
        merged_x = self.gate_up_proj(x)
        intermediate_hidden_states, pergroup_scale , _ = torch.ops.custom.npu_swiglu_group_quant(
            merged_x,
            dst_type=torch.float8_e4m3fn,
            quant_mode=2 if "mx" in self.mm_quant_mode else 1,
            )
        return self.down_proj(intermediate_hidden_states, pergroup_scale)


class DeepseekV3MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config, runner_settings, prefix, **kwargs):
        super().__init__()
        self.config = config
        self.layer_idx = kwargs.get("layer_idx")
        self.hash = self.layer_idx < config.num_hash_layers
        self.runner_settings = runner_settings
        self.gmm_quant_mode = (
            config.quant_config.gmm_quant_mode
            if config.quant_config is not None
            else "w16a16")
        self.swiglu_limit = config.swiglu_limit if hasattr(config, "swiglu_limit") else None
        self.hidden_dim = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.moe_tp_size = self.runner_settings.get("parallel_config").get("moe_tp_size", 1)
        self.moe_ep_size = self.runner_settings.get("parallel_config").get("moe_ep_size", 1)
        self.moe_chunk_max_len = self.runner_settings.get("model_config").get("moe_chunk_max_len", 65536)
        self.platform_version = self.runner_settings.get("model_config").get("platform_version", "A3")
        self.exe_mode = self.runner_settings.get("exe_mode", "eager")
        self.enable_multi_streams = runner_settings.get("model_config").get("enable_multi_streams", False)
        self.perfect_eplb = self.runner_settings.get("model_config").get("perfect_eplb", False)
        self.num_experts_per_tok = config.num_experts_per_tok
        # total experts num
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok

        self.npu_events = []
        if self.enable_multi_streams:
            self.npu_events = [torch.npu.Event(), torch.npu.Event()]

        self.intermediate_size_per_rank = self.intermediate_size // self.moe_tp_size
        self.shared_expert_rank_num = 0 # route and share on same card
        self.n_shared_experts = config.n_shared_experts
        self.n_routed_experts = config.n_routed_experts
        self.experts_per_rank = config.n_routed_experts // self.moe_ep_size
        self.hccl_comm_dict = kwargs.get("hccl_comm_dict", None)
        self.experts = FusedMoEGMM(
            num_experts=config.n_routed_experts,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_size,
            # when W4A8 is enabled, gmm kernel needs an auxiliary matrix, it will be passed in as a bias
            bias=True if self.gmm_quant_mode == "w4a8int4" else False,
            quant_config=config.quant_config,
            tp_size=self.moe_tp_size,
            tp_rank=dist.get_rank(self.hccl_comm_dict["moe_tp_group"]) if self.moe_tp_size > 1 else 0,
            ep_size=self.moe_ep_size,
            ep_rank=dist.get_rank(self.hccl_comm_dict["moe_ep_group"]) if self.moe_ep_size > 1 else 0,
            prefix=f"{prefix}.experts",
        )
        self.gmm_int_quant = "a8" in self.gmm_quant_mode and "float" not in self.gmm_quant_mode

        self._init_gate(prefix)
        if config.n_shared_experts is not None:
            self.shared_experts = DeepseekV3SharedExpert(config, self.runner_settings,
                                        is_moe_layer=True, prefix=f"{prefix}.shared_experts", **kwargs)
        self.dispatch_quant_mode = {
            "w16a16": 0,
            "w8a8int8": 2,
            "w8a8float8": 3,
            "w8a8mxfloat8": 4,
            "w4a8mxfloat4": 4,
            "w4a4mxfloat4": 4,
        }

        self.dispatch_kwargs = None
        self.combine_kwargs = None

    def _init_gate(self, prefix):
        self.n_routed_experts = self.config.n_routed_experts
        self.routed_scaling_factor = self.config.routed_scaling_factor
        self.scoring_func = self.config.scoring_func
        self.topk_method = self.config.topk_method
        self.n_group = self.config.n_group
        self.topk_group = self.config.topk_group
        # topk selection algorithm
        self.norm_topk_prob = self.config.norm_topk_prob
        self.gate = ReplicatedLinear(self.config.hidden_size,
                                     self.n_routed_experts,
                                     bias=False,
                                     quant_config=None,
                                     params_dtype=torch.bfloat16 if self.platform_version == "950" else torch.float32,
                                     prefix=f"{prefix}.gate",
                                     )
        self._reset_parameters()
        if self.hash:
            self.gate.e_score_correction_bias = None
            self.tid2eid = nn.Parameter(torch.randint(high=self.n_routed_experts, size=(self.config.vocab_size, self.top_k), dtype=torch.int32), requires_grad=False)
        else:
            self.gate.e_score_correction_bias = nn.Parameter(torch.empty((self.n_routed_experts), dtype=torch.float32))
            self.tid2eid = None

    def _reset_parameters(self) -> None:
        pass

    def _split_tensors(self, bs_qlen, x, topk_ids, topk_weight, hidden_states_share):
        if bs_qlen > self.moe_chunk_max_len:  # need to chunk moe seq_len dim to avoid OOM
            num_chunks = (bs_qlen + self.moe_chunk_max_len - 1) // self.moe_chunk_max_len
            x_list = x.chunk(num_chunks, dim=0)
            topk_ids_list = topk_ids.chunk(num_chunks, dim=0)
            topk_weight_list = topk_weight.chunk(num_chunks, dim=0)
            if hidden_states_share is None:
                hidden_states_share_list = [None] * num_chunks
            else:
                hidden_states_share_list = hidden_states_share.chunk(num_chunks, dim=0)
        else:
            x_list = [x]
            topk_ids_list = [topk_ids]
            topk_weight_list = [topk_weight]
            hidden_states_share_list = [hidden_states_share]
        return x_list, topk_ids_list, topk_weight_list, hidden_states_share_list

    def set_mc2_kwargs(self):
        global_rank = dist.get_rank()
        moe_ep_group_name = self.hccl_comm_dict.get("moe_ep_group_mc2_name", None)
        if self.gmm_quant_mode not in self.dispatch_quant_mode:
            quant_mode = self.dispatch_quant_mode["w16a16"]
        else:
            quant_mode = self.dispatch_quant_mode[self.gmm_quant_mode]
        enable_smooth_scale = quant_mode == self.dispatch_quant_mode["w8a8int8"]
        self.dispatch_kwargs = {
                "x_active_mask": None,
                "expert_shard_type": 0,
                "shared_expert_rank_num": self.shared_expert_rank_num,
                "moe_expert_num": self.n_routed_experts,
                "global_bs": 0,
                "scales": self.experts.smooth_scale_1 if enable_smooth_scale else None,
                "quant_mode": quant_mode,
                "group_ep": moe_ep_group_name,
                "ep_world_size": self.moe_ep_size,
                "ep_rank_id": global_rank // self.moe_tp_size,
                "group_tp": moe_ep_group_name,
                "tp_world_size": self.moe_tp_size,
                "tp_rank_id": global_rank % self.moe_tp_size,
            }
        if self.gmm_quant_mode == "w4a4mxfloat4":
            self.dispatch_kwargs['y_dtype'] = torch_npu.float4_e2m1fn_x2
        elif quant_mode in (self.dispatch_quant_mode["w8a8float8"], self.dispatch_quant_mode["w8a8mxfloat8"]):
            self.dispatch_kwargs['y_dtype'] = torch.float8_e4m3fn
        self.combine_kwargs = {
                "x_active_mask": None,
                "expert_shard_type": 0,
                "shared_expert_rank_num": self.shared_expert_rank_num,
                "moe_expert_num": self.n_routed_experts,
                "global_bs": 0,
                "group_ep": moe_ep_group_name,
                "ep_world_size": self.moe_ep_size,
                "ep_rank_id": global_rank // self.moe_tp_size,
                "group_tp": moe_ep_group_name,
                "tp_world_size": self.moe_tp_size,
                "tp_rank_id": global_rank % self.moe_tp_size
            }
        if self.platform_version != "950":
            self.dispatch_kwargs["comm_alg"] = "fullmesh_v2"

    def forward(self, hidden_states, is_prefill=False, cur_topk_list=None, input_ids=None, shared_expert_stream=None):
        bsz, seq_len, h = hidden_states.shape

        if self.n_shared_experts > 0:
            hidden_states_share = self.forward_shared_expert(hidden_states, shared_expert_stream)
        else:
            hidden_states_share = None

        # compute gating score
        if self.platform_version == "950":
            logits = torch_npu.npu_fused_matmul(hidden_states.view(-1, h), self.gate.weight, fused_op_type="16cast32")
        else:
            logits = self.gate(hidden_states.view(-1, h).to(torch.float32))
        topk_idx, topk_weight, _ = OpKernel.gate_topk(self, logits, input_ids)
        if self.perfect_eplb:
            topk_idx = cur_topk_list
        topk_idx = topk_idx.to(torch.int32)

        # MOE EP
        if is_prefill:
            return self.moe_infer_double_routing(
                hidden_states, topk_idx, topk_weight, hidden_states_share)
        else:
            return self.moe_infer_dispatch_combine(hidden_states, topk_idx, topk_weight, hidden_states_share)

    def forward_shared_expert(self, hidden_states, shared_expert_stream=None):
        record_stream(self.enable_multi_streams, hidden_states, shared_expert_stream)
        record_event(self.enable_multi_streams, self.npu_events, 0)
        with npu_stream_switch(self.enable_multi_streams, shared_expert_stream):
            wait_event(self.enable_multi_streams, self.npu_events, 0)
            # shared_expert use multi streams
            hidden_states_share = self.shared_experts(hidden_states.view(-1, hidden_states.shape[-1]))
            record_event(self.enable_multi_streams, self.npu_events, 1)
        return hidden_states_share

    def forward_expert(self, gathered_tokens, tokens_per_expert_group, gathered_pertoken_scale):
        # reroute
        if "mx" in self.gmm_quant_mode:
            gathered_pertoken_scale = gathered_pertoken_scale.flatten(1)
        hidden_states_ordered_by_experts, gathered_pertoken_scale, gathered_ids_unsort, tokens_per_local_expert = \
                torch_npu.npu_moe_re_routing(gathered_tokens, tokens_per_expert_group.view(self.moe_ep_size, -1),
                per_token_scales=gathered_pertoken_scale)

        # compute experts
        gmm_args = {
            "x": hidden_states_ordered_by_experts,
            "expert_tokens": tokens_per_local_expert,
            "group_list_type": 1,
            "swiglu_limit": self.swiglu_limit,
            "enable_custom_ops": True,
        }

        if "a16" not in self.gmm_quant_mode:
            if "mxfloat" in self.gmm_quant_mode:
                # match GMM operator requirement (dim0, dim1)->(dim0, dim1//2, 2)
                gathered_pertoken_scale = reshape_mx_scale(gathered_pertoken_scale)
            gmm_args.update({"pertoken_scale": gathered_pertoken_scale})
        hidden_states_ordered_by_experts = self.experts(**gmm_args)
        # finalize-rerouting
        new_x = torch.index_select(hidden_states_ordered_by_experts, 0, gathered_ids_unsort.float().argsort().int())
        return new_x

    def combine_double_routing(self, new_x, expanded_x, input_splits, output_splits):
        moe_ep_group = self.hccl_comm_dict.get("moe_ep_group", None)
        gathered_tokens = new_x.new_empty(expanded_x.shape[0], new_x.shape[1])
        dist.all_to_all_single(gathered_tokens, new_x, input_splits, output_splits, group=moe_ep_group)
        return gathered_tokens

    def dispatch_double_routing(self, tokens_per_expert, expanded_x, pertoken_scale):
        moe_ep_group = self.hccl_comm_dict.get("moe_ep_group", None)
        tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
        # (total_experts,)->(total_ranks*n_routed_experts_per_rank)
        dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert, group=moe_ep_group)
        # combine tensors, do reduceSum and D2H togather
        combine_tokens = torch.stack([tokens_per_expert_group, tokens_per_expert], dim=0)
        # view: EP, E // EP
        combine_tokens = combine_tokens.view(2, self.moe_ep_size, -1).sum(2)
        all_tokens = combine_tokens[0].sum()
        combine_tokens_cpu = combine_tokens.cpu().tolist()
        input_splits = combine_tokens_cpu[1]
        output_splits = combine_tokens_cpu[0]
        if "a4mxfloat4" in self.gmm_quant_mode:
            expanded_x = expanded_x.view(torch.float8_e4m3fn)
        gathered_tokens = expanded_x.new_empty(all_tokens.item(), expanded_x.shape[1])
        dist.all_to_all_single(gathered_tokens, expanded_x, output_splits, input_splits, group=moe_ep_group)

        gathered_pertoken_scale = None
        if pertoken_scale is not None:
            if self.gmm_quant_mode == "w8a8float8":
                gathered_pertoken_scale = pertoken_scale.new_empty(gathered_tokens.shape[0], pertoken_scale.shape[1])
            elif "mxfloat" in self.gmm_quant_mode:
                pertoken_scale = pertoken_scale.view(torch.int8)
                gathered_pertoken_scale = pertoken_scale.new_empty(gathered_tokens.shape[0], pertoken_scale.shape[1], pertoken_scale.shape[2])
            else:
                gathered_pertoken_scale = pertoken_scale.new_empty(gathered_tokens.shape[0])
        if "a16" not in self.gmm_quant_mode:
            dist.all_to_all_single(gathered_pertoken_scale, \
                                   pertoken_scale, output_splits, input_splits, group=moe_ep_group)
        if "mxfloat" in self.gmm_quant_mode:
            gathered_pertoken_scale = gathered_pertoken_scale.view(torch.float8_e8m0fnu)
        return tokens_per_expert_group, gathered_tokens, gathered_pertoken_scale, input_splits, output_splits

    def moe_infer_double_routing(self, x, topk_ids, topk_weight, hidden_states_share):
        """
        pure ep strategy, for prefill stage mainly, only support eager mode
        """
        batch_size, sequence_length, h = x.shape
        x = x.view(-1, h)

        # -1: non-quant; 1: dynamic quant; 0: static quant(not supported now)
        # 2: mxfp8 for dst_dtype e5m2; 3: mxfp8 for dst_dtype e4m3; 4: fp8 for dst_dtype e5m2; 5: fp8 for dst_dtype e4m3
        routing_args = {"quant_mode": -1}
        if self.gmm_quant_mode == "w8a8float8":
            moe_init_routing = torch_npu.npu_moe_init_routing_group_quant
            routing_args.update({
                "quant_mode": 5,
                "row_idx_type": 0,
                "drop_pad_mode": 0,
                "group_size": 128, # currently only support 128
            })
        elif "a4mxfloat4" in self.gmm_quant_mode:
            moe_init_routing = torch_npu.npu_moe_init_routing_group_quant
            routing_args.update({
                "quant_mode": 6,
                "row_idx_type": 0,
                "drop_pad_mode": 0,
            })
        elif "a8mxfloat" in self.gmm_quant_mode:
            moe_init_routing = torch_npu.npu_moe_init_routing_group_quant
            routing_args.update({
                "quant_mode": 3,
                "row_idx_type": 0,
                "drop_pad_mode": 0,
            })
        else:
            moe_init_routing = torch_npu.npu_moe_init_routing_v2
            if self.gmm_int_quant:
                routing_args.update({
                    "quant_mode": 1,
                })

        enable_smooth_scale = "w8a8" in self.gmm_quant_mode and "float8" not in self.gmm_quant_mode
        hidden_states_list = []
        for hidden_states, topk_ids, topk_weight, hidden_states_share in zip(
                *self._split_tensors(batch_size * sequence_length, x, topk_ids, topk_weight, hidden_states_share)):
            sequence_length = hidden_states.shape[0]
            expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = moe_init_routing(
                hidden_states,
                expert_idx=topk_ids,
                active_num=topk_ids.shape[0] * topk_ids.shape[1],
                scale=self.experts.smooth_scale_1 if enable_smooth_scale else None,
                expert_num=self.num_experts,
                expert_tokens_num_type=1,  # 0: cumsum mode(not supported now); 1: count mode
                expert_tokens_num_flag=True, active_expert_range=[0, self.num_experts],
                **routing_args
            )

            tokens_per_expert_group, gathered_tokens, gathered_pertoken_scale, input_splits, output_splits =\
                self.dispatch_double_routing(tokens_per_expert, expanded_x, pertoken_scale)

            new_x = self.forward_expert(gathered_tokens, tokens_per_expert_group, gathered_pertoken_scale)

            gathered_tokens = self.combine_double_routing(new_x, expanded_x, input_splits, output_splits)

            wait_event(self.enable_multi_streams, self.npu_events, 1)

            # finalize-routing
            hidden_states = torch_npu.npu_moe_finalize_routing(
                gathered_tokens, skip1=hidden_states_share, skip2=None, bias=None,
                scales=topk_weight.to(gathered_tokens.dtype),
                expanded_src_to_dst_row=expanded_row_idx,
                export_for_source_row=None, drop_pad_mode=2
            )
            hidden_states = hidden_states.view(sequence_length, self.hidden_dim)
            hidden_states_list.append(hidden_states)

        hidden_states = torch.cat(hidden_states_list, dim=0) if len(hidden_states_list) > 1 else hidden_states_list[0]

        return hidden_states.view(batch_size, -1, h)

    def moe_infer_dispatch_combine(self, x, topk_ids, topk_weight, hidden_states_share):
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
            "swiglu_limit": self.swiglu_limit,
            "enable_custom_ops": True
        }

        if "a16" not in self.gmm_quant_mode:
            if "mxfloat" in self.gmm_quant_mode:
                # match GMM operator requirement (dim0, dim1)->(dim0, dim1//2, 2)
                dynamic_scale = reshape_mx_scale(dynamic_scale)
            gmm_args.update({"pertoken_scale": dynamic_scale})

        hidden_states_ordered_by_experts = self.experts(**gmm_args)

        wait_event(self.enable_multi_streams, self.npu_events, 1)

        # moe combine
        combine_args = {
            "expand_x": hidden_states_ordered_by_experts,
            "shared_expert_x": hidden_states_share,
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


class Attention(nn.Module):
    """Multi-Query Attention (MQA) Layer."""
    def __init__(self, config: DeepseekV3Config, runner_settings: Dict, layer_idx: Optional[int] = None,
                 prefix: Optional[str] = "", **kwargs):
        super().__init__()
        self.config = config
        self.runner_settings = runner_settings
        self.batch_size = self.runner_settings.get("data_config").get("batch_size", 16)
        self.batch_size_per_rank = self.runner_settings.get("data_config").get("batch_size_per_rank", 1)
        self.attn_tp_size = self.runner_settings.get("parallel_config").get("attn_tp_size", 1)
        self.attn_dp_size = self.runner_settings.get("parallel_config").get("attn_dp_size", 1)
        self.oproj_tp_size = self.runner_settings.get("parallel_config").get("oproj_tp_size", 1)
        self.moe_tp_size = self.runner_settings.get("parallel_config").get("moe_tp_size", 1)
        self.moe_ep_size = self.runner_settings.get("parallel_config").get("moe_ep_size", 1)
        self.world_size = self.runner_settings.get("world_size", 16)
        self.enable_pypto = self.runner_settings.get("model_config").get("enable_pypto", False)
        # enable_global_multi_streams: global multistream, to wait for attention kernel metadata
        self.enable_global_multi_streams = self.runner_settings.get("model_config").get("enable_multi_streams", False)
        self.enable_npugraph_ex = runner_settings.get("exe_mode", "ge_graph") == "npugraph_ex"
        # enable_multi_streams: multistream within current class
        self.enable_multi_streams = self.enable_global_multi_streams and not self.enable_pypto
        self.platform_version = self.runner_settings.get("model_config").get("platform_version", "A3")
        self.layer_idx = layer_idx
        self.is_mtp = False
        if layer_idx == config.num_hidden_layers:
            self.layer_idx = 0 # MTP model only has one layer of cache
            self.is_mtp = True
        self.mla_events = []
        if self.enable_multi_streams and self.enable_npugraph_ex:
            # 3 is number of events used for event synchronization
            self.mla_events = [torch.npu.Event(), torch.npu.Event(), torch.npu.Event()]

        self.enable_limit_core = self.runner_settings.get("model_config").get("enable_limit_core", False)
        self.compress_ratio = 1 if self.is_mtp else config.compress_ratios[layer_idx]
        if self.enable_multi_streams and self.platform_version == "A3":
            self.enable_compressor_parallel = self.compress_ratio == 128
            if self.compress_ratio == 4: # c4a supports compressor parallel only if it supports limit core num
                self.enable_compressor_parallel = self.enable_limit_core
        else:
            self.enable_compressor_parallel = False
        self.total_aic_num = 24 # enable_limit_core only suppots A3 (24 cube and 48 vector cores)
        self.cmpr_aic_num = 0
        self.cmpr_events = []
        if self.enable_compressor_parallel:
            self.cmpr_aic_num = 16
            # 2 is number of events used for event synchronization
            self.cmpr_events = [torch.npu.Event(), torch.npu.Event()]

        self.aiv_to_aic_ratio = 2 # keep aiv_num = 2 * aic_num
        self.mm_quant_mode = (
            config.quant_config.mm_quant_mode
            if config.quant_config is not None
            else "w16a16")

        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads

        self.hccl_comm_dict = kwargs.get("hccl_comm_dict", None)

        self.num_heads_per_rank = self.n_heads // self.attn_tp_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        # set self.partial_slice, used for inplace_partial_rotary_mul
        self.partial_slice = [self.head_dim - self.rope_head_dim, self.head_dim]
        self.nope_head_dim = config.head_dim - config.qk_rope_head_dim
        self.n_groups = config.o_groups

        self.num_groups_per_rank = max(self.n_groups // self.attn_tp_size, 1)
        self.window_size = config.sliding_window
        self.eps = config.rms_norm_eps

        self.attn_sink = nn.Parameter(torch.empty(self.num_heads_per_rank, dtype=torch.float32))
        self.wq_a = ReplicatedLinear(self.dim,
                                     self.q_lora_rank,
                                     params_dtype=torch.bfloat16,
                                     quant_config=config.quant_config,
                                     prefix=f"{prefix}.wq_a",
                                     )
        self.q_norm = DeepseekV3RMSNorm(self.q_lora_rank, self.eps)
        self.q_b_norm = DeepseekV3RMSNorm(self.head_dim, self.eps)

        self.wq_b = ColumnParallelLinear(config.q_lora_rank,
                                        self.n_heads * self.head_dim,
                                        bias=False,
                                        quant_config=config.quant_config,
                                        tp_size=self.attn_tp_size,
                                        tp_rank=dist.get_rank(self.hccl_comm_dict["attn_tp_group"])
                                        if self.attn_tp_size > 1 else 0,
                                        prefix=f"{prefix}.wq_b",
                                        )
        self.wkv = ReplicatedLinear(self.dim,
                                    self.head_dim,
                                    params_dtype=torch.bfloat16,
                                    quant_config=config.quant_config,
                                    prefix=f"{prefix}.wkv",
                                    )
        self.kv_norm = DeepseekV3RMSNorm(self.head_dim, self.eps)

        # consider oproj_tp
        if self.oproj_tp_size == 1:
            wo_tp_size = self.attn_tp_size
            wo_tp_rank = dist.get_rank(self.hccl_comm_dict["attn_tp_group"]) if self.attn_tp_size > 1 else 0
        else:
            wo_tp_size = self.oproj_tp_size
            wo_tp_rank = dist.get_rank(self.hccl_comm_dict["oproj_tp_group"])
        quant_config = config.quant_config if self.mm_quant_mode == "w8a8mxfloat8" else None
        self.wo_a = ColumnParallelLinear(self.n_heads * self.head_dim // self.n_groups,
                                        self.n_groups * self.o_lora_rank,
                                        params_dtype=torch.bfloat16,
                                        bias=False,
                                        quant_config=quant_config,
                                        tp_size=wo_tp_size,
                                        tp_rank=wo_tp_rank,
                                        prefix=f"{prefix}.wo_a",
                                        )
        self.wo_b = RowParallelLinear(self.n_groups * self.o_lora_rank,
                                    self.dim,
                                    tp_size=wo_tp_size,
                                    tp_rank=wo_tp_rank,
                                    bias=False,
                                    input_is_parallel=True,
                                    quant_config=config.quant_config,
                                    prefix=f"{prefix}.wo_b",
                                    )
        self.softmax_scale = self.head_dim ** -0.5

        if self.compress_ratio > 1:
            self.compressor = Compressor(config, runner_settings, layer_idx, self.compress_ratio,
                                         head_dim = self.head_dim, prefix=f"{prefix}.compressor", **kwargs)
            if self.compress_ratio == 4:
                self.indexer = Indexer(config, runner_settings, layer_idx, self.compress_ratio,
                                       prefix=f"{prefix}.indexer", **kwargs)
            else:
                self.indexer = None

        self.max_position_embeddings = runner_settings.get("data_config").get("max_position_embeddings", 2048)
        self.original_seq_len = config.max_seq_len

        self.li_cache_quant_mode = config.quant_config.li_cache_quant_mode \
            if config.quant_config is not None else "unquant"

        self.kv_cache_quant_mode = config.quant_config.kv_cache_quant_mode \
            if config.quant_config is not None else "unquant"

        self.sparse_attn_ops = torch.ops.custom.npu_sparse_attn_sharedkv
        if self.kv_cache_quant_mode == "float8":
            self.sparse_attn_ops = torch.ops.custom.npu_kv_quant_sparse_attn_sharedkv

        self.cp_size = self.runner_settings.get("parallel_config").get("cp_size", 1)
        self.global_rank = kwargs.get("global_rank")

    def apply_norm_dynamic_quant(self, x):
        if self.mm_quant_mode == "w8a8float8":
            x = self.q_norm(x)
            bsz, seq_len, _ = x.shape
            x, x_scale = torch_npu.npu_dynamic_block_quant(x.view(-1, x.size(-1)),
                dst_type=torch.float8_e4m3fn,
                row_block_size=1,
                col_block_size=self.config.quant_config.weight_block_size[1])
            return x.view(bsz, seq_len, -1), x_scale.view(bsz, seq_len, -1)
        elif self.mm_quant_mode == "w8a8mxfloat8":
            x = self.q_norm(x)
            return torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
        elif self.mm_quant_mode == "w8a8int8":
            if self.platform_version == "950":
                x = self.q_norm(x)
                return torch_npu.npu_dynamic_quant(x, smooth_scales=self.wq_b.smooth_scales)
            else:
                return torch.ops.custom.npu_rms_norm_dynamic_quant(x, self.q_norm.weight, \
                                                                        smooth_scale=self.wq_b.smooth_scales, \
                                                                        epsilon=self.eps)
        else:
            return x, None

    def mla_prolog_pypto(
        self,
        x: torch.Tensor,
        attn_metadata: Optional[Dict] = None,
        cache_data: Optional[Tuple[Dict]] = None,
        is_prefill: bool = True,
        **kwargs,
    ):
        cos_sin = attn_metadata["cos_sin"]
        cos, sin = cos_sin["c1a"] if self.compress_ratio == 1 else cos_sin["comp"]

        b, s, h = x.shape
        _, q_lora_rank = self.wq_a.weight.shape
        _, nq_headdim = self.wq_b.weight.shape
        _, head_dim = self.wkv.weight.shape
        n_q = nq_headdim // head_dim

        p_x = x.reshape(b * s, h)
        p_cos = cos.reshape(-1, cos.shape[-1]) # [t, d]
        p_sin = sin.reshape(-1, sin.shape[-1]) # [t, d]

        if self.config.quant_config is None:
            from ops.pypto_python.impl.mla_prolog_pypto import mla_prolog_pypto

            # token_x, wq_a, wq_b, wkv, rope_cos, rope_sin, gamma_cq, gamma_ckv
            q, kv, qr = mla_prolog_pypto(p_x, self.wq_a.weight.contiguous(), self.wq_b.weight.contiguous(),
                                self.wkv.weight.contiguous(), p_cos, p_sin, self.q_norm.weight, self.kv_norm.weight)
        else:
            from ops.pypto_python.impl.mla_prolog_quant_pypto import mla_prolog_quant_pypto
            # token_x, wq_a, wq_b, wkv, rope_cos, rope_sin, gamma_cq, gamma_ckv, wq_b_scale
            q, kv, qr, qr_scale = mla_prolog_quant_pypto(p_x, self.wq_a.weight.contiguous(),
                                self.wq_b.weight.contiguous(), self.wkv.weight.contiguous(), p_cos, p_sin,
                                self.q_norm.weight, self.kv_norm.weight, self.wq_b.weight_scale.to(torch.float32))
        q = q.reshape(b, s, n_q, head_dim)
        kv = kv.reshape(b, s, head_dim)
        qr = qr.reshape(b, s, q_lora_rank)
        if is_prefill:
            # update temporary full cache
            full_kv_cache = attn_metadata["full_kv_cache"]
            full_kv_slot_mapping = attn_metadata["slot_mapping"]["full_kv"]
            self.update_win_kv(kv, full_kv_slot_mapping, full_kv_cache)

            # extract tail cache
            gather_indices = attn_metadata["slot_mapping"]["full_kv_gather_indices"]
            kv = torch.gather(kv.view(b, s, -1), 1, gather_indices.unsqueeze(-1).expand(-1, -1, self.head_dim))

        win_cache = cache_data["win_kv"]
        win_kv_slot_mapping = attn_metadata["slot_mapping"]["win_kv"]
        self.update_win_kv(kv, win_kv_slot_mapping, win_cache)
        return q, qr, qr_scale, x

    def get_cp_window(
        self,
        x: torch.Tensor,
        attn_metadata: Optional[Dict] = None,
    ):
        bsz, seqlen, _ = x.size()

        # Collect tail windows (last 128 tokens) of all CP segments
        x_prev, x_next = x.split(x.shape[1] // 2, dim=1)
        cur_win_list = []
        for zz_flag in ["prev", "next"]:
            x_seg = x_prev if zz_flag == "prev" else x_next
            cur_kv_len = attn_metadata[zz_flag]["cur_kv_len"]
            if cur_kv_len >= self.window_size:
                cur_win = x_seg[:, cur_kv_len - self.window_size: cur_kv_len].view(-1, x.shape[-1])
            else:
                cur_win = x_seg[:, :cur_kv_len]
                cur_win = torch.cat([cur_win, torch.zeros([bsz, self.window_size - cur_kv_len, x.shape[-1]], dtype=x.dtype).npu()],
                                    dim=1).view(-1, x.shape[-1])
            cur_win_list.append(cur_win)
        cur_win = torch.cat(cur_win_list, dim=0)
        all_win = x.new_empty([cur_win.shape[0] * self.cp_size, x.shape[-1]])
        dist.all_gather_into_tensor(all_win, cur_win, group=self.hccl_comm_dict["cp_group"])
        all_win = all_win.view(-1, bsz, self.window_size,
                               x.shape[-1])[attn_metadata["cp_metadata"]["reverse_index"]]

        # Retrieve the window from the tail of the previous CP segment,
        # and concatenate it to the current CP segment
        x_list = []
        for zz_flag in ["prev", "next"]:
            x_seg = x_prev if zz_flag == "prev" else x_next
            if not (attn_metadata[zz_flag]["is_start"] and zz_flag == "prev"):
                # get pre win except rank 0
                if zz_flag == "prev":
                    pre_win = all_win[self.global_rank - 1]
                else:
                    pre_win = all_win[2 * self.cp_size - self.global_rank - 2]
                x_seg = torch.cat([pre_win, x_seg], dim=1)
            x_list.append(x_seg)

        # Compute the last window kv of the entire sequence
        cos_sin = attn_metadata["prev"]["cos_sin"]
        cos, sin = cos_sin["c1a_last_win"] if self.compress_ratio == 1 \
            else cos_sin["comp_last_win"]
        last_kv_len = attn_metadata["prev"]["last_kv_len"]
        last_rank = attn_metadata["cp_metadata"]["last_rank"]
        if last_kv_len >= self.window_size:
            last_win = all_win[last_rank]
        else:
            last_win = all_win[last_rank, :, :last_kv_len]
            second_last_win = all_win[last_rank - 1]
            last_win = torch.cat([second_last_win[:, -(self.window_size - last_kv_len):], last_win], dim=1)
        last_win_kv = self.wkv(last_win)
        last_win_kv = self.kv_norm(last_win_kv)
        torch.ops.custom.inplace_partial_rotary_mul(
            last_win_kv.view(-1, 1, 1, self.head_dim), cos, sin,
            rotary_mode="interleave",
            partial_slice=self.partial_slice,
        )
        return last_win_kv, x_list

    def mla_prolog_prefill(
        self,
        x: torch.Tensor,
        attn_metadata: Optional[Dict] = None,
        cache_data: Optional[Tuple[Dict]] = None,
        is_prefill: bool = True,
        **kwargs,
    ):
        bsz, seqlen, _ = x.size()
        if self.cp_size > 1:
            cp_metadata_prev, cp_metadata_next = attn_metadata["prev"], attn_metadata["next"]
            cos_sin_prev, cos_sin_next = cp_metadata_prev["cos_sin"], cp_metadata_next["cos_sin"]
            cos_prev, sin_prev = cos_sin_prev["c1a"] if self.compress_ratio == 1 else cos_sin_prev["comp"]
            cos_next, sin_next = cos_sin_next["c1a"] if self.compress_ratio == 1 else cos_sin_next["comp"]
            cos = torch.cat([cos_prev, cos_next], dim=0)
            sin = torch.cat([sin_prev, sin_next], dim=0)
        else:
            cos_sin = attn_metadata["cos_sin"]
            cos, sin = cos_sin["c1a"] if self.compress_ratio == 1 else cos_sin["comp"]

        # q
        qr = self.wq_a(x)
        qr, qr_scale = self.apply_norm_dynamic_quant(qr)
        q = self.wq_b(qr, dynamic_scale=qr_scale).unflatten(-1, (self.num_heads_per_rank, self.head_dim))
        q = self.q_b_norm(q)
        torch.ops.custom.inplace_partial_rotary_mul(   # x: (T, 1, N, D); cos(T, 1, 1, D)
            q.flatten(0, 1).unsqueeze(2), cos, sin,
            rotary_mode="interleave",
            partial_slice=self.partial_slice,
        )

        if self.cp_size > 1:
            last_win_kv, x_with_pre_win = self.get_cp_window(x, attn_metadata)
            cos_prev, sin_prev = cos_sin_prev["c1a_with_pre_win"] if self.compress_ratio == 1 \
                else cos_sin_prev["comp_with_pre_win"]
            cos_next, sin_next = cos_sin_next["c1a_with_pre_win"] if self.compress_ratio == 1 \
                else cos_sin_next["comp_with_pre_win"]
            cos = torch.cat([cos_prev, cos_next], dim=0)
            sin = torch.cat([sin_prev, sin_next], dim=0)
            x = torch.cat(x_with_pre_win, dim=1)

        # win kv & topk_idxs
        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        torch.ops.custom.inplace_partial_rotary_mul(
            kv.view(-1, 1, 1, self.head_dim), cos, sin,
            rotary_mode="interleave",
            partial_slice=self.partial_slice,
        )

        # update temporary full cache
        full_kv_cache = attn_metadata["full_kv_cache"]
        if self.cp_size > 1:
            full_kv_slot_mapping_prev = cp_metadata_prev["slot_mapping_ori_kv"]
            full_kv_slot_mapping_next = cp_metadata_next["slot_mapping_ori_kv"]
            full_kv_slot_mapping = torch.cat([full_kv_slot_mapping_prev, full_kv_slot_mapping_next], dim=-1)
        else:
            full_kv_slot_mapping = attn_metadata["slot_mapping"]["full_kv"]
        self.update_win_kv(kv, full_kv_slot_mapping, full_kv_cache)

        # extract tail cache
        if self.cp_size > 1:
            kv = last_win_kv
        else:
            gather_indices = attn_metadata["slot_mapping"]["full_kv_gather_indices"]
            kv = torch.gather(kv.view(bsz, seqlen, -1), 1, \
                            gather_indices.unsqueeze(-1).expand(-1, -1, self.head_dim))

        win_cache = cache_data["win_kv"]
        win_kv_slot_mapping = attn_metadata["slot_mapping"]["win_kv"]
        self.update_win_kv(kv, win_kv_slot_mapping, win_cache)
        return q, qr, qr_scale, x

    def mla_prolog_decode(
        self,
        x: torch.Tensor,
        attn_metadata: Optional[Dict] = None,
        cache_data: Optional[Tuple[Dict]] = None,
        is_prefill: bool = True,
        **kwargs,
    ):
        cos_sin = attn_metadata["cos_sin"]
        cos, sin = cos_sin["c1a"] if self.compress_ratio == 1 else cos_sin["comp"]
        enable_multi_streams = self.enable_multi_streams and self.enable_npugraph_ex
        enable_limit_core = self.enable_limit_core and not is_prefill

        kv_aic_num = self.total_aic_num // 2 # half of total corenums to support parallelism with q_b qbmm
        qb_aic_num = self.total_aic_num - self.cmpr_aic_num
        enable_cmpr_stream = self.enable_compressor_parallel and not is_prefill

        record_event(enable_cmpr_stream, self.cmpr_events, 0)
        record_stream(enable_cmpr_stream, x, attn_metadata.get('compressor_stream', None))

        if self.platform_version == "A3":
            qa = self.wq_a(x)
        record_event(enable_multi_streams, self.mla_events, 0)
        record_stream(enable_multi_streams, x, attn_metadata.get('mla_stream', None))
        if self.platform_version != "A3":
            qa = self.wq_a(x)
        qr, qr_scale = self.apply_norm_dynamic_quant(qa)

        cur_stream = torch.npu.current_stream()
        with npu_stream_switch(enable_multi_streams, attn_metadata.get('mla_stream', None)):
            # ensure wkv matmul does not overlap with wq_a or wq_b
            wait_event(enable_multi_streams, self.mla_events, 0)
            kv = self.wkv(x)
            record_event(enable_multi_streams, self.mla_events, 1)
            with limit_core_num(enable_limit_core, kv_aic_num, kv_aic_num * self.aiv_to_aic_ratio): # parallel to wq_b
                kv = self.kv_norm(kv)
                torch.ops.custom.inplace_partial_rotary_mul(
                    kv.view(-1, 1, 1, self.head_dim), cos, sin,
                    rotary_mode="interleave",
                    partial_slice=self.partial_slice,
                )
                record_stream(enable_multi_streams, kv, cur_stream)
                record_event(enable_multi_streams, self.mla_events, 2)

        wait_event(enable_multi_streams, self.mla_events, 1)
        q = self.wq_b(qr, dynamic_scale=qr_scale).unflatten(-1, (self.num_heads_per_rank, self.head_dim))

        # start compressor and li qb stream after wq_b, to run parallel with wkv vectors
        record_event(enable_cmpr_stream, self.cmpr_events, 0)
        if self.compress_ratio == 4:
            record_stream(enable_multi_streams, qr, attn_metadata.get('indexer_stream', None))
            record_stream(enable_multi_streams, qr_scale, attn_metadata.get('indexer_stream', None))
            record_event(enable_multi_streams, self.indexer.indexer_events, 0)
        with limit_core_num(enable_limit_core, qb_aic_num, qb_aic_num * self.aiv_to_aic_ratio):
            q = self.q_b_norm(q)
            torch.ops.custom.inplace_partial_rotary_mul( # x: (T, 1, N, D); cos(T, 1, 1, D)
                q.flatten(0, 1).unsqueeze(2), cos, sin,
                rotary_mode="interleave",
                partial_slice=self.partial_slice,
            )
            # update kv cache in default stream can remove tensormove
            wait_event(enable_multi_streams, self.mla_events, 2)
            self.update_win_kv(kv, attn_metadata["slot_mapping"]["win_kv"], cache_data["win_kv"])
        return q, qr, qr_scale, x

    def mla_prolog(
        self,
        x: torch.Tensor,
        attn_metadata: Optional[Dict] = None,
        cache_data: Optional[Tuple[Dict]] = None,
        is_prefill: bool = True,
        **kwargs,
    ):
        if self.enable_pypto:
            return self.mla_prolog_pypto(x, attn_metadata, cache_data, is_prefill, **kwargs)
        elif is_prefill:
            return self.mla_prolog_prefill(x, attn_metadata, cache_data, is_prefill, **kwargs)
        else:
            return self.mla_prolog_decode(x, attn_metadata, cache_data, is_prefill, **kwargs)

    def prepare_fa_kwargs(
        self,
        attn_metadata,
        q: torch.Tensor,
        cmp_sparse_indices: torch.Tensor = None,
        cache_data: Optional[Tuple[Dict]] = None,
        is_prefill: Optional[bool] = False
    ):
        if self.compress_ratio > 1:
            cmp_block_table = attn_metadata["block_table"][f'c{self.compress_ratio}a_cmp_kv']
        else:
            cmp_block_table = None
        win_cache = attn_metadata["full_kv_cache"] if is_prefill else cache_data["win_kv"]
        win_block_table_str = "full_kv" if is_prefill else "win_kv"
        metadata = attn_metadata["kernel_metadata"][f'c{self.compress_ratio}a_metadata']

        cu_seqlens_q = attn_metadata["cu_seq_lens_q"]
        seqused_kv = attn_metadata["actual_seq_k"]

        attn_kwargs = {
            "cu_seqlens_q": cu_seqlens_q,
            "seqused_kv": seqused_kv,
            "cmp_ratio": self.compress_ratio,
            "ori_mask_mode": 4, # sliding window
            "cmp_mask_mode": 3, # causal
            "ori_win_left": 127,
            "ori_win_right": 0,
            "layout_q": "TND",
            "layout_kv": "PA_ND",
            "q": q,
            "ori_kv": win_cache,   # get from prefill is full cache, transfer bsnd to bbnd
            "cmp_kv": cache_data["sfa_cmp_kv"],
            "cmp_sparse_indices": cmp_sparse_indices,  # only for C4A
            "ori_block_table": attn_metadata["block_table"][win_block_table_str],
            "cmp_block_table": cmp_block_table,
            "sinks": self.attn_sink,
            "metadata": metadata, # get from operator sparse_attn_sharedkv_metadata for fa tiling
            "softmax_scale": self.softmax_scale,
        }
        if self.kv_cache_quant_mode == "float8":
            attn_kwargs["tile_size"] = 64 # quant per tile size
            attn_kwargs["rope_head_dim"] = self.rope_head_dim
            attn_kwargs["kv_quant_mode"] = 1
        return attn_kwargs

    def update_win_kv(
        self,
        kv: torch.Tensor,
        win_kv_slot_mapping: torch.Tensor,
        win_cache: torch.Tensor,
    ):
        if self.kv_cache_quant_mode == "float8":
            torch.ops.custom.kv_compress_epilog(
                    x=kv.view(-1, self.head_dim),
                    slot_mapping=win_kv_slot_mapping,
                    kv_compress_cache=win_cache
                )
        else:
            torch.ops.custom.scatter_nd_update_asc(win_cache.view(-1, win_cache.shape[-1]),
                                            win_kv_slot_mapping.view(-1, 1),
                                            kv.view(-1, self.head_dim))

    def sparse_attn(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        qr: torch.Tensor,
        qr_scale: torch.Tensor,
        attn_metadata: Optional[Dict] = None,
        cache_data: Optional[Tuple[Dict]] = None,
        is_prefill: bool = True,
        **kwargs
    ):
        bsz, seq_len, _ = x.size()
        if is_prefill and self.cp_size > 1:
            q_len = q.shape[1] // 2
            x = list(x.split([x.shape[1] - q_len - self.window_size, q_len + self.window_size], dim=1))

        enable_cmpr_stream = self.enable_compressor_parallel and not is_prefill
        enable_limit_core = self.enable_limit_core and not is_prefill

        # compressor for C4A and C128A
        # cmpr event 0 is recorded in self.mal_prolog function, after calling mla qb
        # cmpr event 1 is waited inside self.sparse_attn function, before calling sfa(c128a) / li qb dynamic_quant(c4a)
        if self.compress_ratio > 1:
            with npu_stream_switch(enable_cmpr_stream, attn_metadata.get('compressor_stream', None)):
                wait_event(enable_cmpr_stream, self.cmpr_events, 0)
                with limit_core_num(enable_limit_core, self.cmpr_aic_num, self.cmpr_aic_num * self.aiv_to_aic_ratio):
                    self.compressor(x, cache_data, attn_metadata, is_prefill)
                record_event(enable_cmpr_stream, self.cmpr_events, 1)

        enable_metadata_stream = self.enable_multi_streams and not is_prefill
        wait_event(enable_metadata_stream, attn_metadata.get('metadata_event'), 1)

        # indexer for C4A
        # separate sfa compressor and li qb dynamic quant
        if self.compress_ratio == 4:
            # wait self.cmpr_events[1] in self.indexer
            topk_idxs = self.indexer(x, qr, qr_scale, cache_data, attn_metadata, enable_cmpr_stream, self.cmpr_events, 1, is_prefill)
        else:
            topk_idxs = None

        if self.compress_ratio > 1:
            wait_event(enable_cmpr_stream, self.cmpr_events, 1) # finish compressor before sfa

        if is_prefill and self.cp_size > 1:
            q_prev, q_next = q.split(q.shape[1] // 2, dim=1)
            if self.compress_ratio == 4:
                topk_idxs_prev, topk_idxs_next = topk_idxs
            else:
                topk_idxs_prev, topk_idxs_next = None, None
            o_prev = self.attn_kernel(q_prev, topk_idxs_prev, attn_metadata["prev"], cache_data, is_prefill)
            o_next = self.attn_kernel(q_next, topk_idxs_next, attn_metadata["next"], cache_data, is_prefill)
            o = torch.cat([o_prev, o_next], dim=1)
        else:
            o = self.attn_kernel(q, topk_idxs, attn_metadata, cache_data, is_prefill)
        return o

    def attn_kernel(
        self,
        q: torch.Tensor,
        topk_idxs: torch.Tensor,
        attn_metadata: Optional[Dict] = None,
        cache_data: Optional[Tuple[Dict]] = None,
        is_prefill: bool = True,
    ):
        bsz, _ = q.shape[:2]
        # get fa input dict
        fa_input_kwargs = self.prepare_fa_kwargs(
                attn_metadata,
                q.flatten(0, 1),
                topk_idxs,
                cache_data,
                is_prefill
        )
        o = self.sparse_attn_ops(**fa_input_kwargs)[0].unflatten(0, (bsz, q.shape[1]))
        return o

    def attn_post(
        self,
        o: torch.Tensor,
        attn_metadata: Optional[Dict] = None,
        is_prefill: bool = True,
    ):
        '''
        oproj_tp: split on group dim, o: [B, S, G, N, D/G] -> [B, S, G/tp_size, N, D/G]
        transpose to make the splitted dim to be the primary
        split o_a on group dim (batch); split o_b on group dim (reduce)
        '''
        bsz, seq_len = o.shape[:2]
        if is_prefill and self.cp_size > 1:
            cos_sin_prev, cos_sin_next = attn_metadata["prev"]["cos_sin"], attn_metadata["next"]["cos_sin"]
            cos_prev = cos_sin_prev["c1a"][0] if self.compress_ratio == 1 else cos_sin_prev["comp"][0]
            cos_next = cos_sin_next["c1a"][0] if self.compress_ratio == 1 else cos_sin_next["comp"][0]
            sin_prev = cos_sin_prev["c1a_neg_sin"] if self.compress_ratio == 1 else cos_sin_prev["comp_neg_sin"]
            sin_next = cos_sin_next["c1a_neg_sin"] if self.compress_ratio == 1 else cos_sin_next["comp_neg_sin"]
            cos = torch.cat([cos_prev, cos_next], dim=0)
            sin = torch.cat([sin_prev, sin_next], dim=0)
        else:
            cos_sin = attn_metadata["cos_sin"]
            cos = cos_sin["c1a"][0] if self.compress_ratio == 1 else cos_sin["comp"][0]
            sin = cos_sin["c1a_neg_sin"] if self.compress_ratio == 1 else cos_sin["comp_neg_sin"]

        torch.ops.custom.inplace_partial_rotary_mul(
            o.flatten(0, 1).unsqueeze(2), cos, sin,
            rotary_mode="interleave",
            partial_slice=self.partial_slice,
        )

        o = o.view(bsz * seq_len, self.num_groups_per_rank, -1).to(torch.bfloat16)
        if self.oproj_tp_size > 1:
            # [BS, tp_size, G/tp_size, ND/G] -> [tp_size, BS, G/tp_size, ND/G]
            o = o.view(bsz * seq_len, self.oproj_tp_size, self.num_groups_per_rank // self.oproj_tp_size, -1)
            o = o.transpose(1, 0).contiguous().view(-1)
            all2all_output = torch.empty_like(o)
            dist.all_to_all_single(all2all_output, o,
                                   group=self.hccl_comm_dict.get("oproj_tp_group", None))
            o = all2all_output.view(self.oproj_tp_size * bsz * seq_len,
                                    self.num_groups_per_rank // self.oproj_tp_size, -1)

        # o_a_proj
        if self.mm_quant_mode == "w8a8mxfloat8":
            o, o_scale = torch_npu.npu_dynamic_mx_quant(o, dst_type=torch.float8_e4m3fn)
            o = torch_npu.npu_transpose_quant_batchmatmul(o, self.wo_a.weight, dtype=torch.bfloat16,
 	                                                        x1_scale=o_scale.view(torch.float8_e8m0fnu),
                                                            x2_scale=self.wo_a.weight_scale.view(torch.float8_e8m0fnu),
 	                                                        group_sizes=(0, 0, 32),perm_x1=(1, 0, 2),perm_x2=(0, 1, 2),
                                                            perm_y=(1, 0, 2))
        else:
            o = torch_npu.npu_transpose_batchmatmul(o, self.wo_a.weight, perm_x1=(1, 0, 2), perm_y=(1, 0, 2))
        if self.oproj_tp_size > 1:
            # [oproj_tp_size, bsz * seq_len, num_groups_per_rank // oproj_tp_size * o_lora_rank]
            o = o.view(self.oproj_tp_size, bsz * seq_len, -1)
        else:
            o = o.view(bsz, seq_len, -1)

        # o_b_proj
        x = self.wo_b(o)

        if self.oproj_tp_size > 1:
            # [oproj_tp_size, bsz * seq_len, dim] --> [oproj_tp_size * bsz * seq_len, dim]
            x = x.view(self.oproj_tp_size * bsz * seq_len, -1)
            reduce_scatter_output = torch.empty((bsz * seq_len, x.shape[-1]), dtype=x.dtype, device=x.device)
            dist.reduce_scatter_tensor(reduce_scatter_output, x,
                                       group=self.hccl_comm_dict.get("oproj_tp_group", None))
            x = reduce_scatter_output.view(bsz, seq_len, x.shape[-1])
        return x

    def forward(
        self,
        x: torch.Tensor,
        attn_metadata: Optional[Dict] = None,
        cache_data: Optional[Tuple[Dict]] = None,
        is_prefill: bool = True,
        **kwargs,
    ):
        cache_data = cache_data[self.layer_idx]
        q, qr, qr_scale, x = self.mla_prolog(x, attn_metadata, cache_data, is_prefill, **kwargs)

        # o TND -> BSND
        o = self.sparse_attn(
            x, q, qr, qr_scale, attn_metadata, cache_data, is_prefill, **kwargs)
        x = self.attn_post(o, attn_metadata, is_prefill)

        return x

class DeepseekV3DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV3Config, runner_settings: Dict, layer_idx: int, prefix: str, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.runner_settings = runner_settings
        self.hidden_size = config.hidden_size
        self.attn = Attention(
            config=config,
            runner_settings=self.runner_settings,
            layer_idx=layer_idx,
            prefix=f"{prefix}.attn",
            **kwargs)

        self.ffn = (
            DeepseekV3MoE(config, self.runner_settings, layer_idx=layer_idx, prefix=f"{prefix}.mlp", **kwargs)
        )
        self.input_layernorm = DeepseekV3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = DeepseekV3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # mhc parameters
        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.hidden_size
        origin_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
        self.hc_attn_scale = nn.Parameter(torch.empty(3))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3))
        torch.set_default_dtype(origin_dtype)

        self.attn_norm = DeepseekV3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = DeepseekV3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm_eps = config.rms_norm_eps

        self.cp_size = self.runner_settings.get("parallel_config").get("cp_size", 1)
        self.global_rank = kwargs.get("global_rank")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: Optional[Dict] = None,
        past_residual: Optional[torch.Tensor] = None,
        cache_data: Optional[Tuple[Dict]] = None,
        is_prefill: Optional[bool] = False,
        cur_topk_list: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states
        hidden_states, post, comb = OpKernel.hc_pre(hidden_states, self.hc_attn_fn, self.hc_attn_scale,
                                                    self.hc_attn_base, self.hc_mult, self.hc_sinkhorn_iters,
                                                    self.norm_eps, self.hc_eps)

        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.attn(
            x=hidden_states,
            attn_metadata=attn_metadata,
            cache_data=cache_data,
            is_prefill=is_prefill,
        )
        hidden_states = OpKernel.hc_post(hidden_states, residual, post, comb)

        residual = hidden_states
        hidden_states, post, comb = OpKernel.hc_pre(hidden_states, self.hc_ffn_fn, self.hc_ffn_scale,
                                                    self.hc_ffn_base, self.hc_mult, self.hc_sinkhorn_iters,
                                                    self.norm_eps, self.hc_eps)
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states,
            is_prefill=is_prefill,
            cur_topk_list=cur_topk_list,
            input_ids=input_ids,
            shared_expert_stream =attn_metadata.get('shared_expert_stream', None),
        )
        hidden_states = OpKernel.hc_post(hidden_states, residual, post, comb)

        return hidden_states

@add_start_docstrings(
    "The bare DeepseekV3 Model outputting raw hidden-states without any specific head on top.",
    DEEPSEEKV3_START_DOCSTRING,
)
class DeepseekV3Model(DeepseekV3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DeepseekV3DecoderLayer`]

    Args:
        config: DeepseekV3Config
    """

    def __init__(self, config: DeepseekV3Config, runner_settings: Dict, prefix: str, **kwargs):
        super().__init__(config)
        self.config = config
        self.runner_settings = runner_settings
        self.embed_tp_size = self.runner_settings.get("parallel_config").get("embed_tp_size", 1)
        self.embed_dp_size = self.runner_settings.get("parallel_config").get("embed_dp_size", 1)
        self.attn_tp_size = self.runner_settings.get("parallel_config").get("attn_tp_size", 1)
        self.cp_size = self.runner_settings.get("parallel_config").get("cp_size", 1)
        self.moe_ep_size = self.runner_settings.get("parallel_config").get("moe_ep_size", 1)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.vocab_size_per_rank = self.vocab_size // self.embed_tp_size
        self.global_rank = kwargs.get("global_rank")
        self.enable_superkernel = self.runner_settings.get("model_config").get("enable_superkernel", False)
        self.enable_multi_streams = runner_settings.get("model_config").get("enable_multi_streams", False)
        self.world_size = self.runner_settings.get("world_size", 16)
        self.hccl_comm_dict = kwargs.get("hccl_comm_dict", None)
        self.max_position_embeddings = self.runner_settings.get("data_config").get("max_position_embeddings", 2048)

        is_mtp = kwargs.get("is_mtp")
        if not is_mtp:
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                self.padding_idx,
                torch.bfloat16,
                tp_size=self.embed_tp_size,
                tp_rank=dist.get_rank(self.hccl_comm_dict["embed_tp_group"]) if self.embed_tp_size > 1 else 0)
            self.layers = nn.ModuleList(
                [
                    DeepseekV3DecoderLayer(
                        config,
                        self.runner_settings,
                        layer_idx,
                        prefix=f"layers.{layer_idx}",
                        **kwargs)
                    for layer_idx in range(config.num_hidden_layers)
                ]
            )
        else:
            self.embed_tokens = None
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # mhc
        self.hc_eps = config.hc_eps
        self.hc_mult = hc_mult = config.hc_mult
        hc_dim = hc_mult * config.hidden_size
        origin_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim))
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult))
        self.hc_head_scale = nn.Parameter(torch.empty(1))
        torch.set_default_dtype(origin_dtype)
        self.norm_eps = config.rms_norm_eps

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
        _init_rope(self)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def prepare_inputs_for_layer(self, inputs_embeds, input_ids):
        batch_size, seq_length = input_ids.shape

        step = batch_size * seq_length // self.attn_tp_size
        tp_rank = dist.get_rank(group=self.hccl_comm_dict.get("attn_tp_group", None)) % self.attn_tp_size
        end = step * (tp_rank + 1)

        inputs_embeds = inputs_embeds.view(batch_size * seq_length, self.config.hidden_size)
        hidden_states = inputs_embeds[step * tp_rank: end]

        # batch_size * seq_length: SP
        hidden_states = hidden_states.view(-1, step, self.config.hidden_size)

        return hidden_states

    def hc_head(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype)

    def calc_input_embeddings(self, input_ids, is_prefill):
        batch_size, seq_length = input_ids.shape
        cp_size = self.cp_size if is_prefill else 1
        attn_dp_size = self.world_size // self.attn_tp_size // cp_size
        if self.embed_tp_size > 1:
            embed_tp_group = self.hccl_comm_dict.get("embed_tp_group", None)
            if attn_dp_size > self.embed_dp_size:
                allgather_ratio = self.embed_tp_size // self.attn_tp_size
                if input_ids.ndim == 1:
                    all_input_ids = input_ids.new_empty(seq_length * allgather_ratio)
                else:
                    all_input_ids = input_ids.new_empty(batch_size * allgather_ratio, seq_length)
                dist.all_gather_into_tensor(all_input_ids, input_ids, group=embed_tp_group)
                input_ids = all_input_ids

            new_input_ids = input_ids - (self.global_rank % self.embed_tp_size) * self.vocab_size_per_rank
            mask = (new_input_ids >= 0) & (new_input_ids < self.vocab_size_per_rank) # (bs, qlen)
            new_input_ids_per_rank = new_input_ids * mask
            inputs_embeds = self.embed_tokens(new_input_ids_per_rank) * mask.unsqueeze(-1)

            if attn_dp_size <= self.embed_dp_size:
                dist.all_reduce(inputs_embeds, group=embed_tp_group)
            else:
                if input_ids.ndim == 1:
                    inputs_embeds_attn = inputs_embeds.new_empty(seq_length, inputs_embeds.shape[-1])
                else:
                    inputs_embeds_attn = inputs_embeds.new_empty(batch_size, seq_length, inputs_embeds.shape[-1])
                dist.reduce_scatter_tensor(inputs_embeds_attn, inputs_embeds, group=embed_tp_group)
                inputs_embeds = inputs_embeds_attn
        else:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        return hidden_states

    def generate_cos_sin(self, attn_metadata, hidden_states, is_mtp=False):
        # Hash layer: use cos_sin['c1a'] as position_embdding
        # Compress layer: -- Attention & Indexer: use cos_sin['comp'] as position_embdding
        #                 -- Compressor(ratio:[4, 128]): use cos_sin['c4a'] and  cos_sin['c128a'] as position embedding
        position_ids = attn_metadata["position_ids"]
        kv_len = attn_metadata["kv_len"]
        cos_sin = {
            "c1a": self.rotary_emb(hidden_states, position_ids, kv_len, self.max_position_embeddings),
        }
        cos_sin.update({"c1a_neg_sin": -cos_sin["c1a"][1]})
        if not is_mtp:
            position_ids_c = attn_metadata["position_ids_c"]
            cos_sin.update({
                "comp": self.compress_rotary_emb(hidden_states, position_ids, kv_len, self.max_position_embeddings),
                "c4a": self.compress_rotary_emb(
                    hidden_states, position_ids_c["4"], kv_len, self.max_position_embeddings),
                "c128a": self.compress_rotary_emb(
                    hidden_states, position_ids_c["128"], kv_len, self.max_position_embeddings),
            })
            cos_sin.update({"comp_neg_sin": -cos_sin["comp"][1]})
        return cos_sin

    @add_start_docstrings_to_model_forward(DEEPSEEKV3_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor,
        attn_metadata: Optional[Dict] = None,
        cache_data: Optional[Tuple[Dict]] = None,
        is_prefill: Optional[bool] = False,
        cur_topk_list: Optional[torch.Tensor] = None,
    ):
        batch_size, seq_length = input_ids.shape

        inputs_embeds = self.calc_input_embeddings(input_ids, is_prefill)
        hidden_states = inputs_embeds

        cos_sin = self.generate_cos_sin(attn_metadata, hidden_states)
        attn_metadata.update({'cos_sin': cos_sin})

        position_ids = attn_metadata["position_ids"]
        kv_len = attn_metadata["kv_len"]
        input_ids_cp = input_ids
        if is_prefill and self.cp_size > 1:
            hidden_states_list = list(
                torch.split(hidden_states, attn_metadata["cp_metadata"]["split_list"], dim=1))
            hidden_states = torch.cat(
                [hidden_states_list[i] for i in attn_metadata["cp_metadata"]["zigzag_idx"]], dim=1
            )

            input_ids_list = list(
                torch.split(input_ids, attn_metadata["cp_metadata"]["split_list"], dim=1))
            input_ids_cp = torch.cat(
                [input_ids_list[i] for i in attn_metadata["cp_metadata"]["zigzag_idx"]], dim=1
            )

            for zigzag_flag in ["prev", "next"]:
                position_ids_cur = attn_metadata[zigzag_flag]["position_ids_cur"]
                position_ids_with_pre_win = attn_metadata[zigzag_flag]["position_ids_with_pre_win"]
                position_ids_last_win = attn_metadata[zigzag_flag]["position_ids_last_win"]
                position_ids_cmp = attn_metadata[zigzag_flag]["position_ids_cmp_for_rope"]
                cos_sin = {
                    "c1a": self.rotary_emb(hidden_states, position_ids_cur, kv_len, self.max_position_embeddings),
                    "comp": self.compress_rotary_emb(hidden_states, position_ids_cur, kv_len, self.max_position_embeddings),
                    "c1a_with_pre_win": self.rotary_emb(hidden_states, position_ids_with_pre_win,
                                                        kv_len, self.max_position_embeddings),
                    "comp_with_pre_win": self.compress_rotary_emb(hidden_states, position_ids_with_pre_win,
                                                                kv_len, self.max_position_embeddings),
                    "c1a_last_win": self.rotary_emb(hidden_states, position_ids_last_win,
                                                        kv_len, self.max_position_embeddings),
                    "comp_last_win": self.compress_rotary_emb(hidden_states, position_ids_last_win,
                                                                kv_len, self.max_position_embeddings),
                    "c4a": self.compress_rotary_emb(
                        hidden_states, position_ids_cmp["4"], kv_len, self.max_position_embeddings),
                    "c128a": self.compress_rotary_emb(
                        hidden_states, position_ids_cmp["128"], kv_len, self.max_position_embeddings),
                }
                cos_sin.update({"c1a_neg_sin": -cos_sin["c1a"][1]})
                cos_sin.update({"comp_neg_sin": -cos_sin["comp"][1]})
                attn_metadata[zigzag_flag].update({
                    "cos_sin": cos_sin,
                })

        if is_prefill and self.attn_tp_size > 1 and self.moe_ep_size > 1:
            hidden_states = self.prepare_inputs_for_layer(inputs_embeds, input_ids)
        residual = None

        label = f'decode_layer'
        if self.enable_multi_streams:
            option = "stream-fusion=1"
        else:
            option = "option_xxx2"

        # mhc
        hidden_states = hidden_states.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)

        with superkernel_scope(self.enable_superkernel and not is_prefill, label, option):
            for decoder_layer in self.layers:
                hidden_states = decoder_layer(
                    hidden_states,
                    attn_metadata=attn_metadata,
                    past_residual=residual,
                    cache_data=cache_data,
                    is_prefill=is_prefill,
                    cur_topk_list=cur_topk_list,
                    input_ids=input_ids_cp,
                )
        hidden_states = self.hc_head(hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class DeepseekV3ModelMTPLayer(DeepseekV3Model):
    def __init__(self, config: DeepseekV3Config, runner_settings: Dict, layer_idx: int, prefix: str, **kwargs):
        super().__init__(config, runner_settings, prefix=prefix, **kwargs)
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.layers = nn.ModuleDict(
            {
                str(self.mtp_start_layer_idx + i):
                DeepseekV3DecoderLayer(
                    config,
                    runner_settings,
                    layer_idx,
                    prefix="mtp.0",
                    **kwargs)
                for i in range(config.num_nextn_predict_layers)
        })

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: Optional[Dict] = None,
        past_residual: Optional[torch.Tensor] = None,
        is_prefill: Optional[bool] = False,
        cache_data: Optional[Tuple[Dict]] = None,
        cur_topk_list: Optional[torch.Tensor] = None,
        mtp_layer_idx: Optional[int] = 0,
        input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # mhc
        hidden_states = hidden_states.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        hidden_states = self.layers[str(self.mtp_start_layer_idx + mtp_layer_idx)](
            hidden_states,
            attn_metadata=attn_metadata,
            past_residual=past_residual,
            is_prefill=is_prefill,
            cache_data=cache_data,
            cur_topk_list=cur_topk_list,
            input_ids=input_ids,
        )
        hidden_states = self.hc_head(hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
        return hidden_states


class DeepseekV3ForCausalLM(DeepseekV3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, runner_settings, is_mtp=False, prefix: str = ""):
        super().__init__(config)
        self.config = config
        self.runner_settings = runner_settings
        self.input_max_len = self.runner_settings.get("data_config").get("input_max_len", 32)
        self.platform_version = self.runner_settings.get("model_config").get("platform_version", "A3")
        self.get_parallel_settings()
        self.experts_per_rank = config.n_routed_experts // self.moe_ep_size
        self.top_k = config.num_experts_per_tok
        self.max_position_embeddings = self.runner_settings.get("data_config").get("max_position_embeddings", 2048)
        self.perfect_eplb = self.runner_settings.get("model_config").get("perfect_eplb", False)
        self.is_mtp = is_mtp
        self.enable_cache_compile = self.runner_settings.get("model_config").get("enable_cache_compile", False)
        self.mm_quant_mode = (
            config.quant_config.mm_quant_mode
            if config.quant_config is not None
            else "w16a16")
        self.update_kv_quant_settings()
        self.update_gmm_quant_mode()
        self.kv_cache_quant_mode = config.quant_config.kv_cache_quant_mode
        self.li_cache_quant_mode = config.quant_config.li_cache_quant_mode

        self.enable_static_kernel = self.runner_settings.get("model_config").get("enable_static_kernel", False)
        self.enable_npugraph_ex = self.runner_settings.get("exe_mode", "ge_graph") == "npugraph_ex"
        self.enable_multi_streams = self.runner_settings.get("model_config").get("enable_multi_streams", False)
        self.metadata_event = []
        if self.enable_multi_streams:
            self.metadata_event = [torch.npu.Event(), torch.npu.Event()]

        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.rank_offset = int(os.getenv("RANK_OFFSET", "0"))
        self.global_rank = self.local_rank + self.rank_offset
        self.world_size = self.runner_settings.get("world_size", 16)
        kwargs = {
                    "global_rank": self.global_rank,
                    "is_mtp": is_mtp
                }
        default_pg = get_default_group()
        if default_pg is not None:
            if dist.get_world_size() > 1:
                self.hccl_comm_dict = self.init_parallel_comm_group()
                kwargs.update({"hccl_comm_dict": self.hccl_comm_dict})
        self.batch_size_per_rank = self.runner_settings.get("data_config").get("batch_size_per_rank", 1)

        mtp_layer_idx = config.num_hidden_layers # MTP is the last layer
        self.model = DeepseekV3ModelMTPLayer(config, self.runner_settings, mtp_layer_idx, prefix, **kwargs) \
                    if is_mtp else DeepseekV3Model(config, self.runner_settings, prefix, **kwargs)
        self.vocab_size = config.vocab_size
        self.rope_head_dim = config.qk_rope_head_dim
        self.attn_metadata = AttnMetaData(config, runner_settings, is_mtp)
        if not is_mtp:
            self.lm_head = ColumnParallelLinear(
                input_size=config.hidden_size,
                output_size=config.vocab_size,
                bias=False,
                tp_size=self.lmhead_tp_size,
                tp_rank=dist.get_rank(self.hccl_comm_dict.get("lmhead_tp_group")) if self.lmhead_tp_size > 1 else 0,
                quant_config=None,
                prefix="lm_head"
                )
        else:
            self.lm_head = None

        # Initialize weights and apply final processing
        self.post_init()

        if self.enable_cache_compile:
            self.cached_decode = self.get_cached_graph()

        # for prefill minibatch
        self.prefill_mini_batch_size = self.runner_settings.get("model_config").get("prefill_mini_batch_size", 0)
        self.pa_max_length = self.runner_settings.get("model_config").get("pa_max_length", 2048)
        self.block_size = self.runner_settings.get("model_config").get("pa_block_size", 128)
        self.cache_len = self.pa_max_length // self.block_size
        self.sas_metadata_ops = torch.ops.custom.npu_sparse_attn_sharedkv_metadata
        if self.kv_cache_quant_mode == "float8":
            self.sas_metadata_ops = torch.ops.custom.npu_kv_quant_sparse_attn_sharedkv_metadata
        self.enable_prefill_multi_cycle = self.runner_settings.get("model_config").get("prefill_mini_batch_size", 0) > 0
        self.window_size = config.sliding_window
        self.init_cache_dim()
        self.first_layer_idx = mtp_layer_idx if is_mtp else 0
        self.first_layer_ratio = self.config.compress_ratios[self.first_layer_idx]

    @staticmethod
    def _repeat_batch(tensor, repeat_num):
        if repeat_num == 1:
            return tensor
        return tensor.repeat(repeat_num, *[1] * (tensor.dim() - 1))

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

    def update_kv_quant_settings(self):
        # set li_cache_quant_mode to quant_config
        self.config.quant_config.set_quant_mode("li_cache_quant_mode", "unquant")

        # if quant to fp8 or mxfp8, set kv cache and li cache to fp8; if quant to int8, set li cache to int8
        if self.platform_version == "950" and "float" in self.mm_quant_mode:
            self.config.quant_config.kv_cache_quant_mode = "float8"
            self.config.quant_config.li_cache_quant_mode = "float8"
        else:
            self.config.quant_config.li_cache_quant_mode = "int8"

    def update_gmm_quant_mode(self):
        if self.platform_version == "950" and "w4" in self.config.quant_config.gmm_quant_mode and "mx" not in self.config.quant_config.gmm_quant_mode:
            self.config.quant_config.gmm_quant_mode = self.config.quant_config.gmm_quant_mode.replace("float", "mxfloat")

    def get_parallel_settings(self):
        self.embed_tp_size = self.runner_settings.get("parallel_config").get("embed_tp_size", 1)
        self.attn_dp_size = self.runner_settings.get("parallel_config").get("attn_dp_size", 1)
        self.attn_tp_size = self.runner_settings.get("parallel_config").get("attn_tp_size", 1)
        self.oproj_tp_size = self.runner_settings.get("parallel_config").get("oproj_tp_size", 1)
        self.cp_size = self.runner_settings.get("parallel_config").get("cp_size", 1)
        self.moe_ep_size = self.runner_settings.get("parallel_config").get("moe_ep_size", 1)
        self.moe_tp_size = self.runner_settings.get("parallel_config").get("moe_tp_size", 1)
        self.lmhead_tp_size = self.runner_settings.get("parallel_config").get("lmhead_tp_size", self.embed_tp_size)
        self.moe_dp_size = self.runner_settings.get("parallel_config").get("moe_dp_size", 1)
        self.embed_dp_size = self.runner_settings.get("parallel_config").get("embed_dp_size", 1)
        self.attn_dp_size = self.runner_settings.get("parallel_config").get("attn_dp_size", 1)

    def get_cached_graph(self):
        case_name = "compile_cache/" + os.getenv("CASE_NAME")
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), case_name)
        cache_model = self.main_decode
        if self.is_mtp:
            cache_model = self.main_decode_mtp

        torch._dynamo.config.inline_inbuilt_nn_modules = False
        if self.enable_npugraph_ex:
            compile_options = {
                "frozen_parameter": True,
                "static_kernel_compile": self.enable_static_kernel,
            }
            cached_decode = torch.npu.npugraph_ex.inference.cache_compile(cache_model, cache_dir=cache_dir,
                                                                          dynamic=False, options=compile_options)
        else:
            tng_config = tng.CompilerConfig()
            tng_config.experimental_config.frozen_parameter = True
            tng_config.experimental_config.tiling_schedule_optimize = True
            tng_config.experimental_config.topology_sorting_strategy = "StableRDFS"

            cached_decode = tng.inference.cache_compile(cache_model, cache_dir=cache_dir, config=tng_config,
                                                        dynamic=False, fullgraph=True, ge_cache=True)

        return cached_decode

    def init_parallel_comm_group(self):
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()

        attn_tp_group = init_comm_group(
            global_rank=global_rank, group_num=self.attn_dp_size, world_size=world_size,
            group_stride=1, group_name="attn_tp_group", platform_version=self.platform_version)

        if self.oproj_tp_size == self.attn_tp_size:
            oproj_tp_group = attn_tp_group
        else:
            oproj_tp_group = init_comm_group(
                global_rank=global_rank, group_num=world_size // self.oproj_tp_size, world_size=world_size,
                group_stride=1, group_name="oproj_tp_group", platform_version=self.platform_version)

        if self.embed_tp_size == self.attn_tp_size:
            embed_tp_group = attn_tp_group
        else:
            embed_tp_group = init_comm_group(
                global_rank=global_rank, group_num=self.embed_dp_size, world_size=world_size,
                group_stride=1, group_name="embed_tp_group", platform_version=self.platform_version)

        if self.lmhead_tp_size == self.embed_tp_size:
            lmhead_tp_group = embed_tp_group
        else:
            lmhead_tp_group = init_comm_group(
                global_rank=global_rank, group_num=world_size // self.lmhead_tp_size, world_size=world_size,
                group_stride=1, group_name="lmhead_tp_group", platform_version=self.platform_version)

        if self.moe_tp_size == self.attn_tp_size:
            moe_tp_group = attn_tp_group
        else:
            moe_tp_group = init_comm_group(
                global_rank=global_rank, group_num=self.moe_dp_size, world_size=world_size,
                group_stride=1, group_name="moe_tp_group", platform_version=self.platform_version)

        group_type = None if self.platform_version != "950" else 0 # 950 use default group for prefill moe
        moe_ep_group = init_comm_group(
            global_rank=global_rank, group_num=self.moe_tp_size, world_size=world_size,
            group_stride=self.moe_tp_size, group_name="moe_ep_group", group_type=group_type, platform_version=self.platform_version)

        # used for fullmesh v2
        group_type = None if self.platform_version != "950" else 3 # 950 use aiv group for mc2
        is_full_mesh_v2 = self.platform_version != "950"
        hccl_buffer_size = calc_moe_hccl_buffer_size(self.runner_settings, self.config, is_full_mesh_v2=is_full_mesh_v2)
        moe_ep_group_mc2, moe_ep_group_mc2_name = init_comm_group(
            global_rank=global_rank, group_num=self.moe_tp_size, world_size=world_size,
            group_stride=self.moe_tp_size, group_name="moe_ep_group_mc2", return_name=True,
            hccl_buffer_size=hccl_buffer_size, group_type=group_type, platform_version=self.platform_version)

        cp_group = init_comm_group(
            global_rank=global_rank, group_num=world_size // self.cp_size, world_size=world_size,
            group_stride=1, group_name="cp_group", platform_version=self.platform_version)

        hccl_comm_dict = {
                "default_pg": get_default_group(),
                "attn_tp_group": attn_tp_group, "embed_tp_group": embed_tp_group,
                "moe_tp_group": moe_tp_group, "moe_ep_group": moe_ep_group,
                "moe_ep_group_mc2": moe_ep_group_mc2,
                "moe_ep_group_mc2_name": moe_ep_group_mc2_name,
                "lmhead_tp_group": lmhead_tp_group,
                "oproj_tp_group": oproj_tp_group,
                "cp_group": cp_group
            }
        return hccl_comm_dict

    def forward_lm_head(self, outputs, kv_len, is_prefill=True, attn_metadata=None):
        bs, q_len, hidden_size = outputs.shape
        if is_prefill and self.cp_size > 1:
            outputs_all = outputs.new_empty([bs * q_len * self.cp_size, hidden_size])
            dist.all_gather_into_tensor(outputs_all, outputs.view(bs * q_len, -1), \
                                    group=self.hccl_comm_dict.get("cp_group", None))
            outputs_all = outputs_all.view(-1, bs, q_len // 2,
                                           hidden_size)[attn_metadata["cp_metadata"]["reverse_index"]]
            outputs = outputs_all.permute(1, 0, 2, 3).reshape(bs, -1, hidden_size)
        if is_prefill:
            # attention: SP + TP，moe：DP + EP
            if self.attn_tp_size > 1 and self.moe_ep_size > 1:
                new_outputs = torch.zeros_like(outputs).repeat(self.attn_tp_size, 1, 1)
                dist.all_gather_into_tensor(new_outputs, outputs,
                                            group=self.hccl_comm_dict.get("attn_tp_group", None))
                outputs = new_outputs
            gather_index = kv_len - 1
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
            dist.all_gather_into_tensor(hidden_states, outputs,
                                        group=self.hccl_comm_dict.get("lmhead_tp_group", None))

        logits = self.lm_head(hidden_states) # (lmhead_tp_size * bs / attn_dp, 1, vocab_size / lmhead_tp_size)
        if self.lmhead_tp_size > 1: # -> (bs / attn_dp, 1, vocab_size)
            if self.attn_dp_size == 1:
                new_logits = torch.zeros_like(logits).repeat(self.lmhead_tp_size, 1, 1)
                dist.all_gather_into_tensor(new_logits, logits,
                                            group=self.hccl_comm_dict.get("lmhead_tp_group", None))
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

    def calc_sas_metadata(self, attn_metadata, metadata_kwargs):
        metadata_kwargs["cu_seqlens_q"] = attn_metadata['cu_seq_lens_q']
        actual_seq_k = attn_metadata['actual_seq_k']
        metadata_kwargs["seqused_kv"] = actual_seq_k
        metadata_kwargs["batch_size"] = actual_seq_k.shape[0]
        return self.sas_metadata_ops(**metadata_kwargs)

    def calc_li_metadata(self, attn_metadata, metadata_kwargs):
        metadata_kwargs["actual_seq_lengths_query"] = attn_metadata['actual_seq_q']
        metadata_kwargs["actual_seq_lengths_key"] = attn_metadata['actual_seq_k']
        return torch.ops.custom.npu_quant_lightning_indexer_metadata(**metadata_kwargs)

    def generate_metadata(self, attn_metadata, metadata_kwargs, is_prefill, metadata_desc, is_li=False):
        metadata_ops = self.calc_li_metadata if is_li else self.calc_sas_metadata
        if is_prefill and self.cp_size > 1:
            for zz_flag in ['prev', 'next']:
                attn_metadata[zz_flag]['kernel_metadata'][metadata_desc] = \
                    metadata_ops(attn_metadata[zz_flag], metadata_kwargs)
        else:
            attn_metadata['kernel_metadata'][metadata_desc] = metadata_ops(attn_metadata, metadata_kwargs)

    def init_cache_dim(self):
        cache_dim = self.config.head_dim
        if self.kv_cache_quant_mode == "float8":
            rope_dim = self.config.qk_rope_head_dim
            nope_dim = self.config.head_dim - rope_dim
            cache_dim = align_up(nope_dim + 2 * rope_dim + nope_dim // 64, 128)
        self.cache_dim = cache_dim

    def generate_sas_metadata_kwargs(self):
        sas_metadata_kwargs = {
            "cmp_ratio": 1,
            "ori_mask_mode": 4, # sliding window
            "cmp_mask_mode": 3, # causal
            "ori_win_left": 127, # default
            "ori_win_right": 0,
            "layout_q": "TND",
            "layout_kv": "PA_ND",
            "num_heads_q": self.config.num_attention_heads,
            "num_heads_kv": 1,
            "head_dim": self.cache_dim,
            "has_ori_kv": True,
            "has_cmp_kv": False,
        }
        if self.kv_cache_quant_mode == "float8":
            sas_metadata_kwargs.update(
                {"kv_quant_mode": 1, "tile_size": 64, "rope_head_dim": self.rope_head_dim}
            )
        return sas_metadata_kwargs

    def generate_li_metadata_kwargs(self):
        li_metadata_kwarges = {
            "layout_key": 'PA_BSND',
            "sparse_count": self.config.index_topk,
            "sparse_mode": 3,
            "layout_query": "TND",
            "cmp_ratio": 4, # only c4a have li module
            "key_quant_mode": 0,
            "query_quant_mode": 0,
            "num_heads_q": self.config.index_n_heads,
            "num_heads_k": 1,
            "head_dim": self.config.index_head_dim,
        }
        return li_metadata_kwarges

    def generate_kernel_metadata(self, attn_metadata, is_prefill, is_mtp=False):
        metadata_stream = attn_metadata.get('metadata_stream', None)
        c1a_metadata_kwargs = None
        if self.first_layer_ratio == 1:
            c1a_metadata_kwargs = self.generate_sas_metadata_kwargs()

        c4a_metadata_kwargs = self.generate_sas_metadata_kwargs()
        c4a_metadata_kwargs.update({"cmp_ratio": 4, "has_cmp_kv": True, "cmp_topk": self.config.index_topk})

        c128a_metadata_kwargs = self.generate_sas_metadata_kwargs()
        c128a_metadata_kwargs.update({"cmp_ratio": 128, "has_cmp_kv": True})

        enable_metadata_multi_streams = self.enable_multi_streams and not is_prefill
        record_event(enable_metadata_multi_streams , attn_metadata.get('metadata_event'), 0)
        if is_mtp:  # only c1a in mtp
            with npu_stream_switch(enable_metadata_multi_streams, metadata_stream):
                wait_event(enable_metadata_multi_streams, attn_metadata.get('metadata_event'), 0)
                self.generate_metadata(attn_metadata, c1a_metadata_kwargs, is_prefill, "c1a_metadata")
                record_event(enable_metadata_multi_streams, attn_metadata.get('metadata_event'), 1)
        else:
            with npu_stream_switch(enable_metadata_multi_streams, metadata_stream):
                wait_event(enable_metadata_multi_streams, attn_metadata.get('metadata_event'), 0)
                # scfa_metadata
                if self.first_layer_ratio == 1:
                    self.generate_metadata(attn_metadata, c1a_metadata_kwargs, is_prefill, "c1a_metadata")
                    record_event(enable_metadata_multi_streams, attn_metadata.get('metadata_event'), 1)
                self.generate_metadata(attn_metadata, c128a_metadata_kwargs, is_prefill, "c128a_metadata")
                record_event(enable_metadata_multi_streams and self.first_layer_ratio == 128, attn_metadata.get('metadata_event'), 1)
                self.generate_metadata(attn_metadata, c4a_metadata_kwargs, is_prefill, "c4a_metadata")
                if self.li_cache_quant_mode in ["int8", "float8"]:
                    li_metadata_kwargs = self.generate_li_metadata_kwargs()
                    self.generate_metadata(attn_metadata, li_metadata_kwargs,\
                                            is_prefill, "lightning_indexer_quant", is_li=True)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attn_metadata: Optional[Dict] = None,
        cache_data: Optional[Tuple[Dict]] = None,
        is_prefill: Optional[bool] = False,
        cur_topk_list: Optional[torch.Tensor] = None,
        prev_hidden_states: Optional[torch.Tensor] = None,
        **kwargs
    ):
        self.generate_kernel_metadata(attn_metadata, is_prefill)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            cache_data=cache_data,
            is_prefill=is_prefill,
            cur_topk_list=cur_topk_list,
        ) # (bs / attn_dp, S, hidden_size)

        prev_hidden_states = outputs
        logits = self.forward_lm_head(outputs, attn_metadata["kv_len"], is_prefill, attn_metadata)
        return logits, prev_hidden_states

    def prefill(
        self,
        **kwargs
    ):
        logits, prev_hidden_states = self.forward(
            is_prefill=True,
            **kwargs
        )
        return logits, prev_hidden_states

    def main_decode(
        self,
        **kwargs
    ):
        logits, prev_hidden_states = self.forward(
            is_prefill=False,
            **kwargs
        )
        return logits, prev_hidden_states

    def main_decode_mtp(
        self,
        **kwargs
    ):
        logits, prev_hidden_states = self.forward(
            is_prefill=False,
            **kwargs
        )
        return logits, prev_hidden_states

    def decode(
        self,
        **kwargs
    ):
        if self.enable_cache_compile:
            logits = self.cached_decode(**kwargs)
        else:
            logits = self.main_decode(**kwargs)
        return logits

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
            tokens_per_rank_prefill = batch_size * seq_len // self.attn_tp_size \
            if self.moe_ep_size != 1 else batch_size * seq_len * self.attn_dp_size
            step_prefill = tokens_per_rank_prefill * self.top_k
            cur_topk_list_prefill = [
                (i + global_rank) % self.config.n_routed_experts for i in range(step_prefill)]
            cur_topk_list = torch.Tensor(cur_topk_list_prefill).int().view(tokens_per_rank_prefill, -1).npu()
        else:
            if self.moe_tp_size > 1:
                tokens_per_rank_decode = batch_size * self.top_k * seq_len
                cur_topk_list_decode = []
                for offset in range(self.moe_ep_size):
                    for i in range(offset * self.experts_per_rank, \
                                   offset * self.experts_per_rank + tokens_per_rank_decode):
                        cur_topk_list_decode.append(i)
                cur_topk_list = torch.Tensor(cur_topk_list_decode).int().view(batch_size * seq_len, -1).npu()
            else:
                expanded_tokens = batch_size * self.top_k * seq_len  # Total tokens to be allocated to experts
                step_gap = self.config.n_routed_experts // self.moe_ep_size # Number of experts per rank
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

    def prepare_inputs_for_generation(
        self,
        input_ids,
        cache_data=None,
        attention_mask=None,
        is_prefill=None,
        kv_len=None,
        prev_hidden_states=None,
        **kwargs
    ):
        # input shape: [B, S]
        batch_size, seq_len = input_ids.size()
        # use reshape to avoid stride change, which will cause recompile in mtp case
        input_ids = input_ids.contiguous().reshape(batch_size, seq_len)

        attn_metadata = self.attn_metadata.get_attn_metadata(input_ids, attention_mask, kv_len, is_prefill)
        if is_prefill:
            # temporary full cache for win attention
            cache = CacheData(self.config, self.runner_settings, self.is_mtp, \
                              self.kv_cache_quant_mode, self.li_cache_quant_mode)
            full_kv_cache = cache.init_full_buffer_c1a()
            attn_metadata.update({"full_kv_cache": full_kv_cache})

        attn_metadata.update({"metadata_event": self.metadata_event})

        if is_prefill and self.cp_size > 1:
            attn_metadata = self.attn_metadata.get_cp_metadata(input_ids, is_prefill, attn_metadata, self.is_mtp, self.hccl_comm_dict)

        model_inputs = {
            "input_ids": input_ids.to(torch.int32),
            "cache_data": cache_data,
            "attn_metadata": attn_metadata,
            "prev_hidden_states": prev_hidden_states,
        }
        return model_inputs

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
        ]

        repeat_loaded_weights_mapping = [] # (origin_name: repeat_loaded_name)

        # Params for weights, int8 weight scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoEGMM.make_expert_params_mapping(
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts)

        params_dict = dict(self.named_parameters())
        params_dict = adapt_safetensors_field(params_dict)
        loaded_params: Set[str] = set()
        dequant_cache = {}
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if self.config.num_nextn_predict_layers > 0:
                mtp_prefix = [f"model.layers.{self.config.num_hidden_layers + layer_idx}"
                              for layer_idx in range(self.config.num_nextn_predict_layers)]
                if name.startswith(tuple(mtp_prefix)):
                    continue

            for (origin_name, repeat_loaded_name) in repeat_loaded_weights_mapping:
                if origin_name not in name:
                    continue
                if name.replace(origin_name, repeat_loaded_name) not in params_dict:
                    continue
                param = params_dict[name.replace(origin_name, repeat_loaded_name)]
                weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name.replace(origin_name, repeat_loaded_name))


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
                if (("ffn.experts." in name) and name not in params_dict):
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if self.config.quant_config.mm_quant_mode != "w8a8float8":
                    name = name.replace(".scale", ".weight_scale")

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
                    name = name.replace("scale", "weight_scale")

                    if name not in params_dict:
                        continue
                    is_gmm_w4mxfloat = ("w4" in self.config.quant_config.gmm_quant_mode and
                                       "mxfloat" in self.config.quant_config.gmm_quant_mode)
                    if is_gmm_w4mxfloat:
                        loaded_weight = loaded_weight.view(torch.uint8)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    # The npu_transpose_batchmatmul op doesn't support the fp8 data type. The weight of wo_a needs to be
                    # converted to bf16.
                    if "wo_a" in name and self.config.quant_config.mm_quant_mode == "w8a8float8":
                        base_name, attr = name.rsplit(".", 1)
                        if base_name not in dequant_cache:
                            dequant_cache[base_name] = {}
                        dequant_cache[base_name][attr] = loaded_weight
                        if "weight" in dequant_cache[base_name] and "scale" in dequant_cache[base_name]:
                            data = dequant_cache.pop(base_name)
                            q_weight = data["weight"]
                            scale = data["scale"]
                            loaded_weight = weight_dequant(q_weight, scale)
                            name = f"{base_name}.weight"

                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    if self.config.quant_config.mm_quant_mode != "w8a8float8":
                        name = name.replace(".scale", ".weight_scale")

                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)

        # add checkpoint load check
        weights_not_loaded = set(params_dict.keys()) - loaded_params
        if weights_not_loaded:
            if all("smooth_scale" in name or "q_b_norm" in name for name in weights_not_loaded):
                logger.warning(
                    "Smooth scales were not initialized from checkpoint.")
            else:
                raise ValueError(
                    "Following weights were not initialized from "
                    f"checkpoint: {weights_not_loaded}")
        return loaded_params


class DeepseekV3ModelMTP(DeepseekV3ForCausalLM):

    def __init__(self, config: DeepseekV3Config, runner_settings: Dict, **kwargs):
        super().__init__(config, runner_settings, is_mtp=True)
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.rank_offset = int(os.getenv("RANK_OFFSET", "0"))
        self.global_rank = self.local_rank + self.rank_offset
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.vocab_size_per_rank = self.vocab_size // self.embed_tp_size
        self.ignore_share_weight = True

        # reuse embed_tokens, lm_head, rotary_emb from main model
        self.embed_tokens = None
        self.lm_head = None
        self.rotary_emb = None

        self.shared_head_norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.enorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.e_proj = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=config.quant_config,
            prefix=f"model.layers.{self.mtp_start_layer_idx}.e_proj",
            )
        self.h_proj = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=config.quant_config,
            prefix=f"model.layers.{self.mtp_start_layer_idx}.h_proj",
            )

    @add_start_docstrings_to_model_forward(DEEPSEEKV3_INPUTS_DOCSTRING)
    @override
    def forward(
        self,
        input_ids: torch.LongTensor,
        prev_hidden_states: Optional[torch.Tensor] = None,
        attn_metadata: Optional[Dict] = None,
        is_prefill: Optional[bool] = False,
        cache_data: Optional[Tuple[Dict]] = None,
        cur_topk_list: Optional[torch.Tensor] = None,
        **kwargs
    ):
        position_ids = attn_metadata["position_ids"]
        kv_len = attn_metadata["kv_len"]
        self.generate_kernel_metadata(attn_metadata, is_prefill, True)

        batch_size, seq_length = input_ids.shape
        hidden_states = self.model.calc_input_embeddings(input_ids, is_prefill)

        cos_sin = self.model.generate_cos_sin(attn_metadata, hidden_states, is_mtp=True)
        attn_metadata.update({'cos_sin': cos_sin})
        residual = None

        if is_prefill and self.cp_size > 1:
            hidden_states_list = list(
                torch.split(hidden_states, attn_metadata["cp_metadata"]["split_list"], dim=1))
            hidden_states = torch.cat(
                [hidden_states_list[i] for i in attn_metadata["cp_metadata"]["zigzag_idx"]], dim=1
            )

            for zigzag_flag in ["prev", "next"]:
                position_ids_cur = attn_metadata[zigzag_flag]["position_ids_cur"]
                position_ids_with_pre_win = attn_metadata[zigzag_flag]["position_ids_with_pre_win"]
                position_ids_last_win = attn_metadata[zigzag_flag]["position_ids_last_win"]
                cos_sin = {
                    "c1a": self.rotary_emb(hidden_states, position_ids_cur, kv_len, self.max_position_embeddings),
                    "c1a_with_pre_win": self.rotary_emb(hidden_states, position_ids_with_pre_win,
                                                        kv_len, self.max_position_embeddings),
                    "c1a_last_win": self.rotary_emb(hidden_states, position_ids_last_win,
                                                        kv_len, self.max_position_embeddings),
                }
                cos_sin.update({"c1a_neg_sin": -cos_sin["c1a"][1]})
                attn_metadata[zigzag_flag].update({
                    "cos_sin": cos_sin,
                })

        hidden_states = self.enorm(hidden_states)
        prev_hidden_states = self.hnorm(prev_hidden_states)
        hidden_states_e = self.e_proj(hidden_states)
        hidden_states_h = self.h_proj(prev_hidden_states)
        hidden_states = hidden_states_e + hidden_states_h

        hidden_states = self.model(
            hidden_states,
            past_residual=residual,
            attn_metadata=attn_metadata,
            is_prefill=is_prefill,
            cache_data=cache_data,
            cur_topk_list=cur_topk_list,
        )

        prev_hidden_states = self.shared_head_norm(hidden_states)

        outputs = prev_hidden_states
        logits = self.forward_lm_head(
            outputs=outputs, kv_len=attn_metadata["kv_len"], is_prefill=is_prefill, \
            attn_metadata=attn_metadata)

        return logits, prev_hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping, mtp_unique_weight_mapping, expert_params_mapping, repeat_loaded_weights_mapping \
            = self._load_weight_map()

        params_dict = dict(self.named_parameters())
        params_dict = adapt_safetensors_field_mtp(params_dict, self.config.num_hidden_layers)
        loaded_params: Set[str] = set()
        dequant_cache = {}
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.ignore_share_weight and any(
                    substring in name for substring in ["lm_head.weight", "emb.tok_emb"]):
                continue
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if not spec_layer:
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

            for (origin_name, repeat_loaded_name) in repeat_loaded_weights_mapping:
                if origin_name not in name:
                    continue
                if name.replace(origin_name, repeat_loaded_name) not in params_dict:
                    continue
                param = params_dict[name.replace(origin_name, repeat_loaded_name)]
                weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name.replace(origin_name, repeat_loaded_name))

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
                if (("ffn.experts." in name) and name not in params_dict):
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if self.config.quant_config.mm_quant_mode != "w8a8float8":
                    name = name.replace(".scale", ".weight_scale")

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
                    name = name.replace("scale", "weight_scale")

                    if name not in params_dict:
                        continue
                    is_gmm_w4mxfloat = ("w4" in self.config.quant_config.gmm_quant_mode and
                                       "mxfloat" in self.config.quant_config.gmm_quant_mode)
                    if is_gmm_w4mxfloat:
                        loaded_weight = loaded_weight.view(torch.uint8)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    # The npu_transpose_batchmatmul op doesn't support the fp8 data type. The weight of wo_a needs to be
                    # converted to bf16.
                    if "wo_a" in name and self.config.quant_config.mm_quant_mode == "w8a8float8":
                        base_name, attr = name.rsplit(".", 1)
                        if base_name not in dequant_cache:
                            dequant_cache[base_name] = {}
                        dequant_cache[base_name][attr] = loaded_weight
                        if "weight" in dequant_cache[base_name] and "scale" in dequant_cache[base_name]:
                            data = dequant_cache.pop(base_name)
                            q_weight = data["weight"]
                            scale = data["scale"]
                            loaded_weight = weight_dequant(q_weight, scale)
                            name = f"{base_name}.weight"

                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    if self.config.quant_config.mm_quant_mode != "w8a8float8":
                        name = name.replace(".scale", ".weight_scale")

                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)

        weights_not_loaded = set(params_dict.keys()) - loaded_params
        if weights_not_loaded:
            if all("smooth_scale" in name or "q_b_norm" in name for name in weights_not_loaded):
                logger.warning(
                    "Smooth scales were not initialized from checkpoint.")
            else:
                raise ValueError(
                    "Following weights were not initialized from "
                    f"checkpoint: {weights_not_loaded}")
        return loaded_params

    def _load_weight_map(self):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
        ]

        mtp_unique_weight_mapping = []

        # Params for weights, int8 weight scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoEGMM.make_expert_params_mapping(
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.config.n_routed_experts)

        repeat_loaded_weights_mapping = [] # (origin_name: repeat_loaded_name)
        return stacked_params_mapping, mtp_unique_weight_mapping, expert_params_mapping, repeat_loaded_weights_mapping


def get_spec_layer_idx_from_weight_name(config,
                                        weight_name: str) -> Optional[int]:
    if hasattr(config,
               "num_nextn_predict_layers") and (config.num_nextn_predict_layers
                                                > 0):
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(f"mtp.0."):
                return True
    return False


def adapt_safetensors_field(params_dict: Dict):
    fix_dict = {}
    for k, v in params_dict.items():
        if "model." in k:
            k = k.removeprefix("model.")
        if "tid2eid" in k:
            k = k.replace("tid2eid", "gate.tid2eid")
        if "e_score_correction_bias" in k:
            k = k.replace("e_score_correction_bias", "bias")
        if "shared_experts.down_proj" in k:
            k = k.replace("shared_experts.down_proj", "shared_experts.w2")
        if "input_layernorm" in k:
            k = k.replace("input_layernorm", "attn_norm")
        if "post_attention_layernorm" in k:
            k = k.replace("post_attention_layernorm", "ffn_norm")
        if "embed_tokens" in k:
            k = k.replace("embed_tokens", "embed")
        if "lm_head" in k:
            k = k.replace("lm_head", "head")
        fix_dict[k] = v
    return fix_dict


def adapt_safetensors_field_mtp(params_dict: Dict, num_hidden_layers: int):
    fix_dict = {}
    for k, v in params_dict.items():
        if "model" in k:
            k = k.replace("model", "mtp.0")
        if f"layers.{num_hidden_layers}." in k:
            k = k.replace(f"layers.{num_hidden_layers}.", "")
        if k in ["enorm.weight", "hnorm.weight", "shared_head_norm.weight", "e_proj.weight",
            "h_proj.weight", "e_proj.weight_scale", "h_proj.weight_scale", "e_proj.scale", "h_proj.scale"]:
            k = "mtp.0." + k
        if "tid2eid" in k:
            k = k.replace("tid2eid", "gate.tid2eid")
        if "e_score_correction_bias" in k:
            k = k.replace("e_score_correction_bias", "bias")
        if "shared_experts.down_proj" in k:
            k = k.replace("shared_experts.down_proj", "shared_experts.w2")
        if "input_layernorm" in k:
            k = k.replace("input_layernorm", "attn_norm")
        if "post_attention_layernorm" in k:
            k = k.replace("post_attention_layernorm", "ffn_norm")
        if "shared_head_norm" in k:
            k = k.replace("shared_head_norm", "norm")
        fix_dict[k] = v
    return fix_dict