# coding=utf-8
# Adapted from
# https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
from typing import List, Optional, Tuple, Dict, Iterable, Set

import torch

import torch.utils.checkpoint

from torch import nn
import torch.distributed as dist

import torch_npu
import torchair as tng

from transformers.cache_utils import Cache
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)

from executor.utils import (
    override, get_init_attn_mask,
    init_comm_group, get_default_group, get_decode_mask)

from executor.model_loader.weight_utils import default_weight_loader
from executor.utils import npu_stream_switch, superkernel_scope, MicroBatchMode
from module.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding
    )
from module.fuse_moe_gmm import FusedMoEGMM
from .configuration_deepseek import DeepseekV3Config
from .modules import (_prepare_4d_causal_attention_mask, one_hot, yarn_get_mscale,
                      DeepseekV3RMSNorm, apply_rotary_pos_emb, _init_rope, DEEPSEEKV3_START_DOCSTRING,
                      DEEPSEEKV3_INPUTS_DOCSTRING, DeepseekV3PreTrainedModel
                    )

logger = logging.get_logger(__name__)
events = [tng.ops.npu_create_tagged_event(tag=f"tag_{i}") for i in range(256)]  # 256: pre-allocated number of events


class DeepseekV3DenseMLP(nn.Module):
    def __init__(self, config, runner_settings, prefix, **kwargs):
        super().__init__()
        self.runner_settings = runner_settings
        self.mm_quant_mode = runner_settings.get("model_config").get("mm_quant_mode", "A16W16")
        self.moe_ep_size = self.runner_settings.get("parallel_config").get("moe_ep_size", 1)
        self.moe_tp_size = self.runner_settings.get("parallel_config").get("moe_tp_size", 1)
        self.dense_tp_size = self.runner_settings.get("parallel_config").get("dense_tp_size", 1)
        self.config = config
        self.hidden_size = config.hidden_size
        self.hccl_comm_dict = kwargs.get("hccl_comm_dict", None)
        self.merge_up_gate_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[config.intermediate_size] * 2,
            bias=False,
            tp_size=self.dense_tp_size,
            tp_rank=dist.get_rank(self.hccl_comm_dict["dense_tp_group"]) if self.dense_tp_size > 1 else 0,
            quant_config=config.quant_config,
            prefix=f"{prefix}.merge_up_gate_proj"
            )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            tp_size=self.dense_tp_size,
            tp_rank=dist.get_rank(self.hccl_comm_dict["dense_tp_group"]) if self.dense_tp_size > 1 else 0,
            quant_config=config.quant_config,
            prefix=f"{prefix}.down_proj")

        if self.mm_quant_mode == "A8W8":
            self.down_proj_forward = self.forward_a8w8
        else:
            self.down_proj_forward = self.forward_normal

    def forward(self, x, is_prefill=False):
        # input_DP + attention_TP + moe_EP
        if is_prefill and self.dense_tp_size > 1 and self.moe_ep_size > 1:
            bsz, q_len, _ = x.size()
            x_output = torch.empty([bsz * q_len * self.dense_tp_size, self.hidden_size], \
                                   dtype=x.dtype, device="npu")
            dist.all_gather_into_tensor(x_output, x, group=self.hccl_comm_dict.get("dense_tp_group", None))
            x = x_output.view(bsz, -1, self.hidden_size)

        down_proj = self.down_proj_forward(x)

        if is_prefill and self.dense_tp_size > 1 and self.moe_ep_size > 1:
            mlp_res = down_proj.new_empty(bsz, q_len, down_proj.shape[-1])
            dist.reduce_scatter_tensor(mlp_res, down_proj, group=self.hccl_comm_dict.get("dense_tp_group", None))
            down_proj = mlp_res
        elif self.dense_tp_size > 1 and self.moe_tp_size > 1:
            dist.all_reduce(down_proj, group=self.hccl_comm_dict.get("dense_tp_group", None))

        return down_proj

    def forward_normal(self, x):
        merged_x = self.merge_up_gate_proj(x)
        intermediate_hidden_states = torch_npu.npu_swiglu(merged_x)
        return self.down_proj(intermediate_hidden_states)

    def forward_a8w8(self, x):
        merged_x, pertoken_scale = self.merge_up_gate_proj(x, out_dtype=torch.int32)
        intermediate_hidden_states, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
            merged_x, weight_scale=self.merge_up_gate_proj.weight_scale,
            quant_scale=self.down_proj.smooth_scales,
            quant_mode=1, activate_left=True,
            activation_scale=pertoken_scale
        )
        return self.down_proj(intermediate_hidden_states, pertoken_scale)


class DeepseekV3SharedExpert(nn.Module):
    def __init__(self, config, runner_settings, is_moe_layer=False, prefix="", **kwargs):
        super().__init__()
        self.runner_settings = runner_settings
        self.mm_quant_mode = runner_settings.get("model_config").get("mm_quant_mode", "A16W16")
        self.moe_tp_size = self.runner_settings.get("parallel_config").get("moe_tp_size", 1)
        self.moe_ep_size = self.runner_settings.get("parallel_config").get("moe_ep_size", 1)
        self.config = config
        self.hidden_size = config.hidden_size
        self.is_moe_layer = is_moe_layer
        self.hccl_comm_dict = kwargs.get("hccl_comm_dict", None)
        self.merge_up_gate_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[config.moe_intermediate_size * config.n_shared_experts] * 2,
            bias=False,
            tp_size=self.moe_tp_size,
            tp_rank=dist.get_rank(self.hccl_comm_dict.get("moe_tp_group")) if self.moe_tp_size > 1 else 0,
            quant_config=config.quant_config,
            prefix=f"{prefix}.merge_up_gate_proj")
        self.down_proj = RowParallelLinear(
            config.moe_intermediate_size * config.n_shared_experts,
            config.hidden_size,
            bias=False,
            tp_size=self.moe_tp_size,
            tp_rank=dist.get_rank(self.hccl_comm_dict["moe_tp_group"]) if self.moe_tp_size > 1 else 0,
            quant_config=config.quant_config,
            prefix=f"{prefix}.down_proj")
        if self.mm_quant_mode == "A8W8":
            self.down_proj_forward = self.forward_a8w8
        else:
            self.down_proj_forward = self.forward_normal


    def forward(self, x):
        down_proj = self.down_proj_forward(x)
        return down_proj

    def forward_normal(self, x):
        merged_x = self.merge_up_gate_proj(x)
        intermediate_hidden_states = torch_npu.npu_swiglu(merged_x)
        return self.down_proj(intermediate_hidden_states)

    def forward_a8w8(self, x):
        merged_x, pertoken_scale = self.merge_up_gate_proj(x, out_dtype=torch.int32)
        intermediate_hidden_states, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
            merged_x, weight_scale=self.merge_up_gate_proj.weight_scale,
            quant_scale=self.down_proj.smooth_scales,
            quant_mode=1, activate_left=True,
            activation_scale=pertoken_scale
        )
        return self.down_proj(intermediate_hidden_states, pertoken_scale)


class DeepseekV3MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config, runner_settings, prefix, **kwargs):
        super().__init__()
        self.config = config
        self.layer_idx = kwargs.get("layer_idx")
        self.runner_settings = runner_settings
        self.gmm_quant_mode = runner_settings.get("model_config").get("gmm_quant_mode", "A16W16")
        self.hidden_dim = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.moe_tp_size = self.runner_settings.get("parallel_config").get("moe_tp_size", 1)
        self.moe_ep_size = self.runner_settings.get("parallel_config").get("moe_ep_size", 1)
        self.exe_mode = self.runner_settings.get("exe_mode", "eager")
        self.enable_multi_streams = runner_settings.get("model_config").get("enable_multi_streams", False)
        self.enable_aclgraph = runner_settings.get("exe_mode", "ge_graph") == "acl_graph"
        self.perfect_eplb = self.runner_settings.get("model_config").get("perfect_eplb", False)
        self.moe_chunk_max_len = self.runner_settings.get("model_config").get("moe_chunk_max_len", 65536)
        self.num_experts_per_tok = config.num_experts_per_tok
        # total experts num
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok

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
            bias=False,
            quant_config=config.quant_config,
            tp_size=self.moe_tp_size,
            tp_rank=dist.get_rank(self.hccl_comm_dict["moe_tp_group"]) if self.moe_tp_size > 1 else 0,
            ep_size=self.moe_ep_size,
            ep_rank=dist.get_rank(self.hccl_comm_dict["moe_ep_group"]) if self.moe_ep_size > 1 else 0,
            prefix=f"{prefix}.experts",
        )

        self._init_gate(prefix)
        if config.n_shared_experts is not None:
            self.shared_experts = DeepseekV3SharedExpert(config, self.runner_settings,
                                        is_moe_layer=True, prefix=f"{prefix}.shared_experts", **kwargs)

        self.dispatch_kwargs = None
        self.combine_kwargs = None
        self.micro_batch_mode = MicroBatchMode(self.runner_settings.get("model_config").get("micro_batch_mode", 0))
        self.moe_ep_group = self.hccl_comm_dict.get("moe_ep_group_stream1", None) if\
            self.micro_batch_mode == MicroBatchMode.PREFILL_MICRO_BATCH_SP_TP_EP else\
            self.hccl_comm_dict.get("moe_ep_group", None)

    def _init_gate(self, prefix):
        self.top_k = self.config.num_experts_per_tok
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
                                     params_dtype=torch.float32,
                                     prefix=f"{prefix}.gate")
        self._reset_parameters()
        if self.topk_method == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty((self.n_routed_experts), dtype=torch.float32)
            )
        else:
            self.gate.e_score_correction_bias = None

    def _reset_parameters(self) -> None:
        pass

    def _forward_gate(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h).to(torch.float32)
        logits = self.gate(hidden_states)

        # use fused kernel, currently only support 256 or 384 experts
        if self.topk_method == "noaux_tc" and self.n_routed_experts in [256, 384]:
            topk_weight, topk_idx, _ = torch_npu.npu_moe_gating_top_k(
                logits,
                k=self.top_k,
                bias=self.gate.e_score_correction_bias.float(),
                k_group=self.topk_group,
                group_count=self.n_group,
                group_select_mode=1,  # 0: group中的最大; 1: topk2.sum
                renorm=0,  # 0: softmax->topk; 1: topk->softmax
                norm_type=1,  # 0: softmax; 1: sigmoid
                routed_scaling_factor=self.routed_scaling_factor,
                eps=float(1e-20)
            )
            return topk_idx, topk_weight, None

        if self.scoring_func == "sigmoid":
            scores = logits.sigmoid()
        elif self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1, dtype=torch.float32)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        ### select top-k experts
        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(
                scores, k=self.top_k, dim=-1, sorted=False
            )
        elif self.topk_method == "group_limited_greedy":
            group_scores = (
                scores.view(bsz * seq_len, self.n_group, -1).max(dim=-1).values
            )  # [n, n_group]
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[
                1
            ]  # [n, top_k_group]
            group_mask = torch.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            topk_weight, topk_idx = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False
            )
        elif self.topk_method == "noaux_tc":
            assert not self.training
            scores_for_choice = scores.view(bsz * seq_len, -1) + self.gate.e_score_correction_bias.unsqueeze(0)
            group_scores = (
                scores_for_choice.view(bsz * seq_len, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
            )  # [n, n_group]
            group_idx = torch.topk(
                group_scores, k=self.topk_group, dim=-1, sorted=False
            )[
                1
            ]  # [n, top_k_group]
            group_mask = one_hot(group_idx, self.n_group)  # [n, n_group]
            group_mask = torch.sum(group_mask, dim=1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand(
                    bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            _, topk_idx = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False
            )
            topk_weight = scores.gather(1, topk_idx)
        else:
            raise NotImplementedError(
                f"insupportable TopK function for MoE gating: {self.topk_method}"
            )

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor # must multiply the scaling factor

        return topk_idx, topk_weight, None

    def set_mc2_kwargs(self):
        global_rank = dist.get_rank()
        moe_ep_group_name = self.hccl_comm_dict.get("moe_ep_group_name", None)
        self.dispatch_kwargs = {
                "x_active_mask": None,
                "expert_shard_type": 0,
                "shared_expert_rank_num": self.shared_expert_rank_num,
                "moe_expert_num": self.n_routed_experts,
                "global_bs": 0,
                "scales": self.experts.smooth_scale_1 if self.gmm_quant_mode == "A8W8" else None,
                "quant_mode": 2 if self.gmm_quant_mode == "A8W8" else 0,
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

    def forward(self, hidden_states, is_prefill=False, cur_topk_list=None):
        if self.n_shared_experts > 0:
            enable_multi_streams = self.enable_multi_streams and not is_prefill
            use_aclgraph_event = enable_multi_streams and self.enable_aclgraph
            if use_aclgraph_event:
                tng.ops.npu_tagged_event_record(events[self.layer_idx])
            with npu_stream_switch(enable_multi_streams, "11"):
                # shared_expert use multi streams
                if use_aclgraph_event:
                    tng.ops.npu_tagged_event_wait(events[self.layer_idx])
                hidden_states_share = self.shared_experts(hidden_states.view(-1, hidden_states.shape[-1]))
                if use_aclgraph_event:
                    tng.ops.npu_tagged_event_record(events[self.layer_idx + self.config.num_hidden_layers])
        else:
            use_aclgraph_event = False
            hidden_states_share = None

        topk_idx, topk_weight, _ = self._forward_gate(hidden_states)
        if self.perfect_eplb:
            topk_idx = cur_topk_list
        topk_idx = topk_idx.to(torch.int32)
        if self.moe_tp_size > 1:
            # MoE TP
            return self.moe_infer_tp(hidden_states, topk_idx, topk_weight, hidden_states_share, use_aclgraph_event)
        else:
            # MoE EP
            if is_prefill:
                return self.moe_infer_double_routing(hidden_states, topk_idx, topk_weight, hidden_states_share)
            else:
                return self.moe_infer_dispatch_combine(hidden_states, topk_idx, topk_weight, hidden_states_share,
                                                       use_aclgraph_event)

    def forward_gate_init_routing(self, hidden_states, cur_topk_list=None):
        # gate
        topk_idx, topk_weight, _ = self._forward_gate(hidden_states)
        if self.perfect_eplb:
            topk_idx = cur_topk_list
        topk_idx = topk_idx.to(torch.int32)

        # init_routing
        _, _, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            expert_idx=topk_idx,
            active_num=topk_idx.shape[0] * topk_idx.shape[1],
            scale=self.experts.smooth_scale_1 if self.gmm_quant_mode == "A8W8" else None,
            expert_num=self.num_experts,
            expert_tokens_num_type=1,  # 0: cumsum mode(not supported now); 1: count mode
            expert_tokens_num_flag=True, active_expert_range=[0, self.num_experts],
            quant_mode=1 if self.gmm_quant_mode == "A8W8" else -1
            # -1: non-quant; 1: dynamic quant; 0: static quant(not supported now)
        )
        return expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale, topk_weight

    def forward_shared_expert(self, hidden_states, is_prefill):
        if self.n_shared_experts > 0:
            hidden_states_share = self.shared_experts(hidden_states.view(-1, hidden_states.shape[-1]))
        else:
            hidden_states_share = None
        return hidden_states_share

    def forward_expert(self, gathered_tokens, tokens_per_expert_group, gathered_pertoken_scale):
        # reroute
        hidden_states_ordered_by_experts, gathered_pertoken_scale, gathered_ids_unsort, tokens_per_local_expert = \
                torch_npu.npu_moe_re_routing(gathered_tokens, tokens_per_expert_group.view(self.moe_ep_size, -1),
                per_token_scales=gathered_pertoken_scale)

        # compute experts
        gmm_args = {
            "x": hidden_states_ordered_by_experts,
            "expert_tokens": tokens_per_local_expert,
            "group_list_type": 1,
        }

        if self.gmm_quant_mode == "A8W8":
            gmm_args.update({"pertoken_scale": gathered_pertoken_scale})
        hidden_states_ordered_by_experts = self.experts(**gmm_args)
        # finalize-rerouting
        new_x = torch.index_select(hidden_states_ordered_by_experts, 0, gathered_ids_unsort.float().argsort().int())
        return new_x

    def forward_combine_double_routing(self, new_x, expanded_x, input_splits, output_splits):
        gathered_tokens = new_x.new_empty(*expanded_x.shape)
        dist.all_to_all_single(gathered_tokens, new_x, input_splits, output_splits, group=self.moe_ep_group)
        return gathered_tokens

    def forward_finalize_routing(self, hidden_states, gathered_tokens, hidden_states_share, topk_weight,
                                  expanded_row_idx):
        batch_size, sequence_length, h = hidden_states.shape
        # finalize-routing
        hidden_states = torch_npu.npu_moe_finalize_routing(
            gathered_tokens, skip1=hidden_states_share, skip2=None, bias=None,
            scales=topk_weight.to(gathered_tokens.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None, drop_pad_mode=2
        )

        hidden_states = hidden_states.view(batch_size, sequence_length, h)
        return hidden_states

    def moe_infer_tp(self, x, topk_ids, topk_weight, hidden_states_share, use_aclgraph_event):
        batch_size, sequence_length, h = x.shape
        hidden_states = x.view(-1, h)
        routing_args = {
            "expert_idx": topk_ids,
            "active_num": batch_size * sequence_length * self.top_k,
            "expert_num": self.num_experts,
            "expert_tokens_num_type": 1,  # 0: cumsum mode(not supported now); 1: count mode
            "expert_tokens_num_flag": True,
            "active_expert_range": [0, self.num_experts],
            "quant_mode": -1
        }
        if self.gmm_quant_mode == "A8W8":
            routing_args.update({
                "scale": self.experts.smooth_scale_1,
                "expert_tokens_num_type": 2,
                "quant_mode": 1,
                "row_idx_type": 0,
                "drop_pad_mode": 0
            })
        # A8W8量化场景npu_moe_init_routing_v2算子融合了后面的dynamicquant算子，输出INT8数据及对应的pertoken_scale
        expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = torch_npu.npu_moe_init_routing_v2(
            hidden_states, **routing_args
        )

        moe_args = {"group_list_type": 1}
        if self.gmm_quant_mode == "A8W8":
            moe_args.update({
                "group_list_type": 2,
                "pertoken_scale": pertoken_scale
            })
        hidden_states_ordered_by_experts = self.experts(expanded_x, tokens_per_expert, **moe_args)
        if use_aclgraph_event:
            tng.ops.npu_tagged_event_wait(events[self.layer_idx + self.config.num_hidden_layers])
        hidden_states = torch_npu.npu_moe_finalize_routing(
            hidden_states_ordered_by_experts,
            skip1=hidden_states_share.view(-1, h), skip2=None,
            bias=None,
            scales=topk_weight.to(hidden_states_ordered_by_experts.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None, drop_pad_mode=2
        )
        if self.moe_tp_size > 1:
            dist.all_reduce(hidden_states, group=self.hccl_comm_dict.get("moe_tp_group"))
        hidden_states = hidden_states.view(batch_size, -1, self.hidden_dim)
        return hidden_states

    def dispatch_double_routing(self, tokens_per_expert, expanded_x, pertoken_scale):
        tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
        # (total_experts,)->(total_ranks*n_routed_experts_per_rank)
        dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert, group=self.moe_ep_group)
        # combine tensors, do reduceSum and D2H togather
        combine_tokens = torch.stack([tokens_per_expert_group, tokens_per_expert], dim=0)
        # view: EP, E // EP
        # sum: EP, 每个rank
        combine_tokens = combine_tokens.view(2, self.moe_ep_size, -1).sum(2)
        all_tokens = combine_tokens[0].sum()
        combine_tokens_cpu = combine_tokens.cpu().tolist()
        # alltoall input splits, 大小为当前rank路由到其他rank的tokens数总和
        input_splits = combine_tokens_cpu[1]
        # alltoall output splits, 每个rank拿到的其他rank的tokens数
        output_splits = combine_tokens_cpu[0]
        # alltoall output, 展开成一维，大小为其他卡路由到当前rank的tokens数总和
        gathered_tokens = expanded_x.new_empty(all_tokens.item(), expanded_x.shape[1])
        dist.all_to_all_single(gathered_tokens, expanded_x, output_splits, input_splits, group=self.moe_ep_group)

        gathered_pertoken_scale = None if pertoken_scale is None else\
                            pertoken_scale.new_empty(gathered_tokens.shape[0])
        if self.gmm_quant_mode == "A8W8":
            dist.all_to_all_single(gathered_pertoken_scale,\
                                   pertoken_scale, output_splits, input_splits, group=self.moe_ep_group)
        return tokens_per_expert_group, gathered_tokens, gathered_pertoken_scale, input_splits, output_splits

    def moe_infer_double_routing(self, x, topk_ids, topk_weight, hidden_states_share):
        """
        pure ep strategy, for prefill stage mainly, only support eager mode
        """
        batch_size, sequence_length, h = x.shape
        x = x.view(-1, h)
        hidden_states_list = []
        for hidden_states, topk_ids, topk_weight, hidden_states_share in zip(
                *self._split_tensors(batch_size * sequence_length, x, topk_ids, topk_weight, hidden_states_share)):
            sequence_length = hidden_states.shape[0]
            expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = torch_npu.npu_moe_init_routing_v2(
                hidden_states,
                expert_idx=topk_ids,
                active_num=topk_ids.shape[0] * topk_ids.shape[1],
                scale=self.experts.smooth_scale_1 if self.gmm_quant_mode == "A8W8" else None,
                expert_num=self.num_experts,
                expert_tokens_num_type=1,  # 0: cumsum mode(not supported now); 1: count mode
                expert_tokens_num_flag=True, active_expert_range=[0, self.num_experts],
                quant_mode=1 if self.gmm_quant_mode == "A8W8" else -1
                # -1: non-quant; 1: dynamic quant; 0: static quant(not supported now)
            )

            tokens_per_expert_group, gathered_tokens, gathered_pertoken_scale, input_splits, output_splits =\
                self.dispatch_double_routing(tokens_per_expert, expanded_x, pertoken_scale)

            new_x = self.forward_expert(gathered_tokens, tokens_per_expert_group, gathered_pertoken_scale)

            gathered_tokens = self.forward_combine_double_routing(new_x, expanded_x, input_splits, output_splits)

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

    def moe_infer_dispatch_combine(self, x, topk_ids, topk_weight, hidden_states_share, use_aclgraph_event):
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

        if self.gmm_quant_mode == "A8W8":
            gmm_args.update({"pertoken_scale": dynamic_scale})

        hidden_states_ordered_by_experts = self.experts(**gmm_args)

        if use_aclgraph_event:
            tng.ops.npu_tagged_event_wait(events[self.layer_idx + self.config.num_hidden_layers])
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


# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->DeepseekV3
class DeepseekV3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DeepseekV3Config, runner_settings: Dict,
                 layer_idx: Optional[int] = None, prefix: Optional[str] = "", **kwargs):
        super().__init__()
        self.config = config
        self.runner_settings = runner_settings
        self.mm_quant_mode = runner_settings.get("model_config").get("mm_quant_mode", "A16W16")
        self.batch_size = self.runner_settings.get("data_config").get("batch_size", 16)
        self.batch_size_per_rank = self.runner_settings.get("data_config").get("batch_size_per_rank", 1)
        self.attn_tp_size = self.runner_settings.get("parallel_config").get("attn_tp_size", 1)
        self.attn_dp_size = self.runner_settings.get("parallel_config").get("attn_dp_size", 1)
        self.moe_tp_size = self.runner_settings.get("parallel_config").get("moe_tp_size", 1)
        self.moe_ep_size = self.runner_settings.get("parallel_config").get("moe_ep_size", 1)
        self.enable_o_proj_alltoall = self.runner_settings.get("parallel_config").get("enable_o_proj_alltoall", False)
        self.is_sp = kwargs.get("is_sp", False)
        self.layer_idx = layer_idx
        if layer_idx == config.num_hidden_layers: # mtp model
            self.layer_idx = 0 # mtp model only has one layer of cache
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
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.is_causal = True
        self.hccl_comm_dict = kwargs.get("hccl_comm_dict", None)

        if self.q_lora_rank is None:
            self.q_proj = ColumnParallelLinear(self.hidden_size,
                                               self.num_heads * self.q_head_dim,
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
            self.q_a_layernorm = DeepseekV3RMSNorm(config.q_lora_rank)
            self.q_b_proj = ColumnParallelLinear(config.q_lora_rank,
                                                 self.num_heads * self.q_head_dim,
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
        self.kv_a_layernorm = DeepseekV3RMSNorm(config.kv_lora_rank)

        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=None,
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

        if self.enable_o_proj_alltoall:
            self.o_proj_ata = ReplicatedLinear(
                self.num_heads * self.v_head_dim,
                self.hidden_size,
                bias=config.attention_bias,
                quant_config=config.quant_config,
                prefix=f"{prefix}.o_proj"
            )
        self.o_proj = RowParallelLinear(self.num_heads * self.v_head_dim,
                                        config.hidden_size,
                                        tp_size=self.attn_tp_size,
                                        tp_rank=dist.get_rank(self.hccl_comm_dict["attn_tp_group"])
                                        if self.attn_tp_size > 1 else 0,
                                        bias=False,
                                        input_is_parallel=True,
                                        quant_config=config.quant_config,
                                        prefix=f"{prefix}.o_proj")

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

        self.enable_pa = self.runner_settings.get("model_config").get("enable_pa", False)
        if self.enable_pa:
            max_length = self.runner_settings.get("model_config").get("pa_max_length", 2048)
            self.block_size = self.runner_settings.get("model_config").get("pa_block_size", 128)
            cache_len = max_length // self.block_size
            self.block_table = torch.arange(0, self.batch_size_per_rank * cache_len
                                            ).reshape(self.batch_size_per_rank, -1)
            self.block_table = self.block_table.to(dtype=torch.int32, device="npu")
        self.enable_weight_nz = runner_settings.get("model_config").get(
            "enable_weight_nz", True
        )
        self.enable_mla_prolog = runner_settings.get("model_config").get(
            "enable_mla_prolog", False
        )
        self.enable_mla_prolog = (
            self.enable_mla_prolog
            and self.q_lora_rank is not None
            and self.enable_pa
            and self.enable_weight_nz
        )
        self.use_faquant = False
        self.kv_scale = None
        self.enable_aclgraph = runner_settings.get("exe_mode", "eager") == "acl_graph"
        self.enable_gegraph = runner_settings.get("exe_mode", "eager") == "ge_graph"
        self.fa_ops = torch.ops.npu
        if self.enable_gegraph and not self.enable_aclgraph:
            self.fa_ops = tng.ops
        self.micro_batch_mode = MicroBatchMode(self.runner_settings.get("model_config").get("micro_batch_mode", 0))
        self.attn_tp_group = self.hccl_comm_dict.get("attn_tp_group_stream1", None) if\
            self.micro_batch_mode == MicroBatchMode.PREFILL_MICRO_BATCH_SP_TP_EP else\
            self.hccl_comm_dict.get("attn_tp_group", None)

    def forward_absorb(
            self,
            hidden_states: torch.Tensor,
            kv_len: torch.IntTensor = None,
            actual_seq_lengths_kv: list = None,
            cos_sin: torch.Tensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            is_prefill: bool = True,
            **kwargs,
        ):
        query_states, key_states, value_states, kv_seq_len = self.prepare_qkv(
            hidden_states=hidden_states,
            cos_sin=cos_sin,
            kv_len=kv_len,
            position_ids=position_ids,
            past_key_value=past_key_value,
            is_prefill=is_prefill
        )
        output = self.apply_attention_npu(
            query_states=query_states, key_states=key_states, value_states=value_states,
            kv_seq_len=kv_seq_len,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            attention_mask=attention_mask,
            past_key_value=past_key_value
        )
        return output

    def prepare_qkv(
        self,
        hidden_states: torch.Tensor,
        cos_sin: torch.Tensor = None,
        kv_len: torch.IntTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        is_prefill: bool = True,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))

        latent_cache = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            latent_cache, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        q = q.view(bsz, q_len, self.num_heads_per_rank, self.q_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_pe = q_pe.transpose(1, 2)

        q_nope = q_nope.view(-1, self.num_heads_per_rank, self.qk_nope_head_dim)
        if self.kv_b_proj_w_k.shape[0] * self.kv_b_proj_w_k.shape[1] <= 65535:  # 65535: max value of uint16
            q_nope = torch_npu.npu_transpose_batchmatmul(q_nope, self.kv_b_proj_w_k, bias=None, scale=None,
                                                         perm_x1=(1, 0, 2), perm_x2=(0, 1, 2), perm_y=(1, 0, 2)
                                                        )  # (b*s, n, d)
            q_nope = q_nope.view(bsz, q_len, self.num_heads_per_rank, self.kv_lora_rank).transpose(1, 2)
        else:
            q_nope = (
                torch.matmul(q_nope.transpose(0, 1), self.kv_b_proj_w_k)
                .transpose(0, 1)
                .view(bsz, q_len, self.num_heads_per_rank, self.kv_lora_rank)
                .transpose(1, 2)  # (b, n, s, d)
            )

        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        k_nope = (
            self.kv_a_layernorm(compressed_kv)
            .view(bsz, -1, 1, self.kv_lora_rank)
            .transpose(1, 2)
        ) # (bs, 1, q_len, kv_lora_rank)

        # rope
        cos, sin = cos_sin
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = torch.cat([q_nope, q_pe], dim=-1)
        key_states = torch.cat([k_nope, k_pe], dim=-1)

        kv_seq_len = k_nope.shape[-2]
        if past_key_value is not None:
            past_key_states = past_key_value[self.layer_idx][0]
            torch_npu.scatter_update_(past_key_states, kv_len, key_states, -2)
            if not is_prefill:
                key_states = past_key_states
            kv_seq_len = past_key_value[0][0].size()[-2]
        value_states = key_states
        return query_states, key_states, value_states, kv_seq_len

    def apply_attention_npu(
        self,
        query_states, key_states, value_states, kv_seq_len,
        attention_mask: Optional[torch.Tensor] = None,
        actual_seq_lengths_kv: list = None,
        past_key_value: Optional[Cache] = None,
    ):
        # repeat k/v heads if n_kv_heads < n_heads
        bsz, _, q_len, _ = query_states.size()
        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        )
        assert attention_mask is not None
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        value_states = value_states[..., :self.kv_lora_rank]
        attn_output = torch.matmul(attn_weights, value_states)

        # kv rank opt
        attn_output = attn_output.transpose(1, 2).contiguous()  # (b, s, n, d)
        attn_output = attn_output.reshape(bsz * q_len, self.num_heads_per_rank, self.kv_lora_rank).transpose(0, 1)
        attn_output = torch.matmul(
            attn_output,
            self.kv_b_proj_w_v
        ).transpose(0, 1)  # (bs*q_len, num_heads, kv_lora_rank)
        attn_output = self.o_proj(attn_output.reshape(bsz, q_len, -1))

        if self.attn_tp_size > 1:
            dist.all_reduce(attn_output, group=self.hccl_comm_dict.get("attn_tp_group", None))
        return attn_output


    def forward_page_attention_normal(
        self,
        hidden_states: torch.Tensor,
        cos_sin: torch.Tensor = None,
        kv_len: torch.IntTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        actual_seq_lengths_kv: list = None,
        is_prefill: bool = True,
        slot_mapping: Optional[torch.Tensor] = None
    ):
        bsz, q_len, _ = hidden_states.size()
        cos, sin = cos_sin

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q_hidden_states = self.q_a_layernorm(self.q_a_proj(hidden_states))
            if self.is_sp:
                q_a_output = torch.empty([bsz * q_len * self.attn_tp_size, self.q_lora_rank], \
                                        dtype=hidden_states.dtype, device="npu")
                dist.all_gather_into_tensor(q_a_output, q_hidden_states.view(bsz * q_len, -1), \
                                        group=self.hccl_comm_dict.get("attn_tp_group", None))
                q_hidden_states = q_a_output
            q = self.q_b_proj(q_hidden_states)

        latent_cache = self.kv_a_proj_with_mqa(hidden_states)

        # TND format, [1, T, N, D]
        q = q.view(1, -1, self.num_heads_per_rank, self.q_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        q_pe = q_pe.transpose(1, 2)
        cos = cos.view(1, 1, -1, self.qk_rope_head_dim)
        sin = sin.view(1, 1, -1, self.qk_rope_head_dim)
        q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin)
        # NTD format, [1, N, T, D]
        q_pe = q_pe.view(1, self.num_heads_per_rank, -1, self.qk_rope_head_dim).transpose(1, 2)
        # TND format, [1, T, N, D]
        query_states = [q_nope, q_pe]

        if self.is_sp:
            latent_cache_output = torch.empty([bsz * q_len * self.attn_tp_size, \
                                    self.kv_lora_rank + self.qk_rope_head_dim], \
                                    dtype=hidden_states.dtype, device="npu")
            dist.all_gather_into_tensor(latent_cache_output, latent_cache, \
                                    group=self.hccl_comm_dict.get("attn_tp_group", None))
            latent_cache = latent_cache_output

        latent_cache = latent_cache.view(-1, 1, 1, self.kv_lora_rank + self.qk_rope_head_dim)  # (B,N,S,D)
        nope_cache = past_key_value[self.layer_idx][0].unsqueeze(2)
        rope_cache = past_key_value[self.layer_idx][1].unsqueeze(2)
        block_num, block_size, key_head_num, cache_dim = nope_cache.size()
        # prefill stage needs to view shapes as the following.
        cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)
        sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)
        _, _, k_rope, k_nope = torch_npu.npu_kv_rmsnorm_rope_cache(
            latent_cache,
            self.kv_a_layernorm.weight,
            cos,
            sin,
            slot_mapping,
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

        assert actual_seq_lengths_kv is not None
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

        if self.enable_o_proj_alltoall and self.attn_tp_size > 1:
            attn_output_ata = attn_output.new_empty(*attn_output.shape)
            dist.all_to_all_single(attn_output_ata, attn_output, group=self.attn_tp_group)
            attn_output_ata = attn_output_ata.reshape(self.attn_tp_size, -1, \
                                    self.num_heads_per_rank, self.v_head_dim).transpose(0, 1)
            attn_output = self.o_proj_ata(attn_output_ata.reshape(bsz, -1, self.num_heads * self.v_head_dim))
        else:
            attn_output = self.o_proj(attn_output.reshape(bsz, -1, self.num_heads_per_rank * self.v_head_dim))

            if self.attn_tp_size > 1:
                # attention_TP + moe_TP
                if self.moe_tp_size > 1:
                    dist.all_reduce(attn_output, group=self.hccl_comm_dict.get("attn_tp_group", None))
                elif self.moe_ep_size > 1:
                    attn_res = attn_output.new_empty(bsz, q_len, attn_output.shape[-1])
                    dist.reduce_scatter_tensor(attn_res, attn_output, \
                                            group=self.hccl_comm_dict.get("attn_tp_group", None))
                    attn_output = attn_res

        return attn_output

    def forward_page_attention_absorb(
        self,
        hidden_states: torch.Tensor,
        cos_sin: torch.Tensor = None,
        kv_len: torch.IntTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        actual_seq_lengths_kv: list = None,
        is_prefill: bool = False,
        slot_mapping: Optional[torch.Tensor] = None
    ):
        bsz, q_len, _ = hidden_states.size()
        cos, sin = cos_sin

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))

        latent_cache = self.kv_a_proj_with_mqa(hidden_states)

        q = q.view(bsz, q_len, self.num_heads_per_rank, self.q_head_dim)
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

        tmp_slot_mapping = kv_len.view(-1)
        latent_cache = latent_cache.view(
            bsz * q_len, 1, 1, self.kv_lora_rank + self.qk_rope_head_dim
        )  # (b*s, n, 1, d)
        cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)  # (b*s, n, 1, d)
        sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)  # (b*s, n, 1, d)
        nope_cache = past_key_value[self.layer_idx][0].unsqueeze(2)
        rope_cache = past_key_value[self.layer_idx][1].unsqueeze(2)
        block_num, block_size, _, _ = nope_cache.size()

        k_rope, k_nope, _, _ = torch_npu.npu_kv_rmsnorm_rope_cache(
            latent_cache,
            self.kv_a_layernorm.weight,
            cos,
            sin,
            tmp_slot_mapping,
            rope_cache,
            nope_cache,
            epsilon=self.kv_a_layernorm.variance_epsilon,
            cache_mode="PA_NZ"
        )

        # adapter nz
        KV_CACHE_NZ_DIM = 16  # bf16 dtype is 16 for nz format, avoid dynamic shape in high torch version
        k_nope = k_nope.view(block_num, 1, self.kv_lora_rank // KV_CACHE_NZ_DIM,
                             block_size, KV_CACHE_NZ_DIM)
        k_rope = k_rope.view(block_num, 1, self.qk_rope_head_dim // KV_CACHE_NZ_DIM,
                             block_size, KV_CACHE_NZ_DIM)

        if q_len > 1: # mtp
            sparse_mode = 3
        else:
            sparse_mode = 0
            attention_mask = None

        attn_output, _ = self.fa_ops.npu_fused_infer_attention_score(
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
            sparse_mode=sparse_mode
        )
        attn_output = attn_output.view(self.num_heads_per_rank, -1, self.kv_lora_rank)
        attn_output = torch_npu.npu_transpose_batchmatmul(
            attn_output,
            self.kv_b_proj_w_v,
            bias=None,
            scale=None,
            perm_x1=(0, 1, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
        )  # (B, S, N*v_head_dim)
        attn_output = self.o_proj(attn_output.view(bsz, q_len, -1))

        if self.attn_tp_size > 1:
            dist.all_reduce(attn_output, group=self.hccl_comm_dict.get("attn_tp_group", None))
        return attn_output

    def forward_page_attention_mla_prolog(
        self,
        hidden_states: torch.Tensor,
        cos_sin: torch.Tensor = None,
        kv_len: torch.IntTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        actual_seq_lengths_kv: list = None,
        is_prefill: bool = False,
        slot_mapping: Optional[torch.Tensor] = None
    ):
        bsz, q_len, _ = hidden_states.size()
        cos, sin = cos_sin
        cos = cos.view(bsz, 1, -1, self.qk_rope_head_dim)
        sin = sin.view(bsz, 1, -1, self.qk_rope_head_dim)
        cache_index = kv_len.view(bsz, -1)
        nope_cache = past_key_value[self.layer_idx][0].unsqueeze(2)
        rope_cache = past_key_value[self.layer_idx][1].unsqueeze(2)
        block_num, block_size, key_head_num, cache_dim = nope_cache.size()

        enable_quant = self.mm_quant_mode == "A8W8"
        if enable_quant:
            hidden_states_int8, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states.flatten(0, 1))
            hidden_states_int8 = hidden_states_int8.view(bsz, q_len, -1)
            pertoken_scale = pertoken_scale.view(-1, 1)

        q_nope, q_pe, k_nope, k_rope, dequant_scale_q_nope = torch.ops.npu.npu_mla_prolog_v2(
            token_x=hidden_states_int8 if enable_quant else hidden_states,
            weight_dq=self.q_a_proj.weight, weight_uq_qr=self.q_b_proj.weight,
            weight_uk=self.kv_b_proj_w_k, weight_dkv_kr=self.kv_a_proj_with_mqa.weight,
            rmsnorm_gamma_cq=self.q_a_layernorm.weight,
            rmsnorm_gamma_ckv=self.kv_a_layernorm.weight,
            rope_sin=sin.squeeze(1), rope_cos=cos.squeeze(1),
            cache_index=cache_index,
            kv_cache=nope_cache,
            kr_cache=rope_cache,
            dequant_scale_x=pertoken_scale if enable_quant else None,
            dequant_scale_w_dq=self.q_a_proj.weight_scale.view(1, -1) if enable_quant else None,
            dequant_scale_w_uq_qr=self.q_b_proj.weight_scale.view(1, -1) if enable_quant else None,
            dequant_scale_w_dkv_kr=self.kv_a_proj_with_mqa.weight_scale.view(1, -1) if enable_quant else None,
            quant_scale_ckv=torch.reciprocal(self.kv_scale).repeat(self.kv_lora_rank).view(1, -1) \
                if self.use_faquant else None,
            quant_scale_ckr=None,
            smooth_scales_cq=None,
            rmsnorm_epsilon_cq=self.q_a_layernorm.variance_epsilon,
            rmsnorm_epsilon_ckv=self.kv_a_layernorm.variance_epsilon,
            cache_mode="PA_NZ"
        )

        # adapter nz
        factor = 2 if self.use_faquant else 1
        KV_CACHE_NZ_DIM = 16  # bf16 dtype is 16 for nz format, avoid dynamic shape in high torch version
        k_nope = k_nope.view(block_num, 1, self.kv_lora_rank // (KV_CACHE_NZ_DIM * factor),
                             block_size, KV_CACHE_NZ_DIM * factor)
        k_rope = k_rope.view(block_num, 1, self.qk_rope_head_dim // KV_CACHE_NZ_DIM,
                             block_size, KV_CACHE_NZ_DIM)

        if q_len > 1: # mtp
            sparse_mode = 3
        else:
            sparse_mode = 0
            attention_mask = None

        attn_output, _ = self.fa_ops.npu_fused_infer_attention_score(
            q_nope, k_nope, k_nope,
            query_rope=q_pe, key_rope=k_rope,
            num_heads=self.num_heads_per_rank,
            num_key_value_heads=self.num_key_value_heads_per_rank,
            input_layout="BSND_NBSD",
            block_table=self.block_table,
            block_size=self.block_size,
            atten_mask=attention_mask,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            scale=self.softmax_scale,
            antiquant_mode=0, antiquant_scale=None,
            sparse_mode=sparse_mode
        )
        attn_output = attn_output.view(self.num_heads_per_rank, -1, self.kv_lora_rank)
        attn_output = torch_npu.npu_transpose_batchmatmul(
            attn_output,
            self.kv_b_proj_w_v,
            bias=None,
            scale=None,
            perm_x1=(0, 1, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
        )  # (B, S, N*v_head_dim)
        attn_output = self.o_proj(attn_output.view(bsz, q_len, -1))

        if self.attn_tp_size > 1:
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
        is_prefill: bool = True,
        output_attentions: bool = False,
        slot_mapping: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_kwargs = {
            "hidden_states": hidden_states,
            "cos_sin": cos_sin,
            "kv_len": kv_len,
            "position_ids": position_ids,
            "past_key_value": past_key_value,
            "actual_seq_lengths_kv": actual_seq_lengths_kv,
            "attention_mask": attention_mask,
            "is_prefill": is_prefill,
            "slot_mapping": slot_mapping
        }
        if self.enable_pa:
            if is_prefill:
                fn = self.forward_page_attention_normal
            elif self.enable_mla_prolog:
                fn = self.forward_page_attention_mla_prolog
            else:
                fn = self.forward_page_attention_absorb
            return fn(**input_kwargs)
        else:
            return self.forward_absorb(**input_kwargs)


class DeepseekV3DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV3Config, runner_settings: Dict, layer_idx: int, prefix: str, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.runner_settings = runner_settings
        self.hidden_size = config.hidden_size

        self.self_attn = DeepseekV3Attention(
            config=config, runner_settings=self.runner_settings, layer_idx=layer_idx,
            prefix=f"{prefix}.self_attn", **kwargs
        )

        self.is_moe = config.n_routed_experts is not None and \
                layer_idx >= config.first_k_dense_replace and \
                layer_idx % config.moe_layer_freq == 0

        self.mlp = (
            DeepseekV3MoE(config, self.runner_settings, layer_idx=layer_idx, prefix=f"{prefix}.mlp", **kwargs)
            if self.is_moe
            else DeepseekV3DenseMLP(config, self.runner_settings, prefix=f"{prefix}.mlp", **kwargs)
        )
        self.input_layernorm = DeepseekV3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = DeepseekV3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
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
        cur_topk_list: Optional[torch.Tensor] = None,
        slot_mapping: Optional[torch.Tensor] = None,
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
            is_prefill=is_prefill,
            slot_mapping=slot_mapping
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if self.is_moe:
            hidden_states = self.mlp(hidden_states, is_prefill=is_prefill, cur_topk_list=cur_topk_list)
        else:
            hidden_states = self.mlp(hidden_states, is_prefill=is_prefill)

        outputs = (residual, hidden_states)
        return outputs

    def forward_post_attention_layernorm(
        self,
        hidden_states: torch.Tensor,
        past_residual: Optional[torch.Tensor] = None
    ):
        hidden_states, residual = self.post_attention_layernorm(hidden_states, past_residual)
        return hidden_states, residual

    def forward_attn(
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
    ):
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
            is_prefill=is_prefill,
            slot_mapping=slot_mapping
        )
        return hidden_states, residual

    def forward_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        is_prefill: Optional[bool] = False,
        cur_topk_list: Optional[torch.Tensor] = None
    ):
        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if self.is_moe:
            hidden_states = self.mlp(hidden_states, is_prefill=is_prefill, cur_topk_list=cur_topk_list)
        else:
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def forward_gate_init_routing(self, *arg, **kwargs):
        return self.mlp.forward_gate_init_routing(*arg, **kwargs)

    def forward_dispatch_double_routing(self, *arg, **kwargs):
        return self.mlp.dispatch_double_routing(*arg, **kwargs)

    def forward_shared_expert(self, *arg, **kwargs):
        return self.mlp.forward_shared_expert(*arg, **kwargs)

    def forward_expert(self, *arg, **kwargs):
        return self.mlp.forward_expert(*arg, **kwargs)

    def forward_combine_double_routing(self, *arg, **kwargs):
        return self.mlp.forward_combine_double_routing(*arg, **kwargs)

    def forward_finalize_routing(self, *arg, **kwargs):
        return self.mlp.forward_finalize_routing(*arg, **kwargs)


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

    def __init__(self, config: DeepseekV3Config, runner_settings: Dict, **kwargs):
        super().__init__(config)
        self.init_params(config, runner_settings, **kwargs)
        self.init_modules(config, **kwargs)

    def init_params(self, config, runner_settings, **kwargs):
        self.config = config
        self.runner_settings = runner_settings
        self.embed_tp_size = self.runner_settings.get("parallel_config").get("embed_tp_size", 1)
        self.attn_tp_size = self.runner_settings.get("parallel_config").get("attn_tp_size", 1)
        self.moe_ep_size = self.runner_settings.get("parallel_config").get("moe_ep_size", 1)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.vocab_size_per_rank = self.vocab_size // self.embed_tp_size
        self.enable_pa = self.runner_settings.get("model_config").get("enable_pa", False)
        self.kv_len_offset = kwargs.get("kv_len_offset", None)
        self.global_rank = kwargs.get("global_rank")
        self.enable_superkernel = self.runner_settings.get("model_config").get("enable_superkernel", False)
        self.enable_multi_streams = runner_settings.get("model_config").get("enable_multi_streams", False)
        self.is_sp = kwargs.get("is_sp", False)

        self.pa_max_length = self.runner_settings.get("model_config").get("pa_max_length", 2048)
        self.hccl_comm_dict = kwargs.get("hccl_comm_dict", None)
        self.gradient_checkpointing = False
        self.perfect_eplb = self.runner_settings.get("model_config").get("perfect_eplb", False)
        self.attn_dp_size = self.runner_settings.get("parallel_config").get("attn_dp_size", 1)
        self.top_k = config.num_experts_per_tok
        self.max_position_embeddings = self.runner_settings.get("data_config").get("max_position_embeddings", 4096)

    def init_modules(self, config, **kwargs):
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            self.padding_idx,
            torch.bfloat16,
            tp_size=self.embed_tp_size,
            tp_rank=dist.get_rank(self.hccl_comm_dict["embed_tp_group"]) if self.embed_tp_size > 1 else 0)
        self.layers = nn.ModuleList(
            [
                DeepseekV3DecoderLayer(config, self.runner_settings, layer_idx,
                                       prefix=f"model.layers.{layer_idx}", **kwargs)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()
        _init_rope(self)

        self.micro_batch_mode = MicroBatchMode(self.runner_settings.get("model_config").get("micro_batch_mode", 0))
        if self.micro_batch_mode != MicroBatchMode.DISABLE:
            self.stream1 = torch.npu.Stream()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_slot_mapping(self, kv_len, is_prefill, device):
        if not is_prefill:
            return None
        all_tensors = []
        for i, seq_len in enumerate(kv_len):
            new_index = torch.arange(self.pa_max_length * i, seq_len.item() + self.pa_max_length * i,
                                     dtype=kv_len.dtype, device=device)
            all_tensors.append(new_index)
        return torch.cat(all_tensors)

    def prepare_inputs_for_prefill_layer(self, inputs_embeds, input_ids):
        batch_size, seq_length = input_ids.shape

        step = batch_size * seq_length // self.attn_tp_size
        tp_rank = dist.get_rank(group=self.hccl_comm_dict.get("attn_tp_group", None)) % self.attn_tp_size
        end = step * (tp_rank + 1)

        inputs_embeds = inputs_embeds.view(batch_size * seq_length, self.config.hidden_size)
        hidden_states = inputs_embeds[step * tp_rank: end]

        # batch_size * seq_length: SP
        hidden_states = hidden_states.view(-1, step, self.config.hidden_size)

        return hidden_states

    @add_start_docstrings_to_model_forward(DEEPSEEKV3_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor,
        kv_len: torch.IntTensor = None,
        actual_seq_lengths_kv: list = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        is_prefill: Optional[bool] = False,
        cur_topk_list: Optional[torch.Tensor] = None,
    ):
        label = f'decode_layer'
        if self.enable_multi_streams:
            option = "stream-fusion=1" # 使能双流后，启用sk的双流选项
        else:
            option = "option_xxx2"
        with superkernel_scope(self.enable_superkernel and not is_prefill, label, option):
            if is_prefill and self.micro_batch_mode != MicroBatchMode.DISABLE:
                input_ids_mb, kv_len_mb, actual_seq_lengths_kv_mb, topk_idx_mb = self.gen_microbatch_input(
                    input_ids, kv_len)
                if self.micro_batch_mode == MicroBatchMode.PREFILL_MICRO_BATCH_DP_EP:
                    fn = self.forward_microbatch_v1
                elif self.micro_batch_mode == MicroBatchMode.PREFILL_MICRO_BATCH_SP_TP_EP:
                    fn = self.forward_microbatch_v2
                hidden_states = fn(
                    input_ids_mb,
                    kv_len_mb,
                    actual_seq_lengths_kv_mb,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    is_prefill=is_prefill,
                    topk_idx_mb=topk_idx_mb,
                )
            else:
                hidden_states, residual, kv_len, cos_sin, slot_mapping, actual_seq_lengths_kv =\
                self.prepare_inputs_for_layer(input_ids, kv_len, position_ids, actual_seq_lengths_kv, is_prefill)
                for decoder_layer in self.layers:
                    residual, hidden_states = decoder_layer(
                        hidden_states,
                        kv_len,
                        actual_seq_lengths_kv,
                        cos_sin=cos_sin,
                        past_residual=residual,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        is_prefill=is_prefill,
                        cur_topk_list=cur_topk_list,
                        slot_mapping=slot_mapping
                    )

                hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    def gen_microbatch_input(self, input_ids, kv_len):
        assert len(kv_len) >= 2, "query num must be greater than 2 when use micro_batch"
        kv_len_mb, split_sections = self.get_split_section(kv_len)
        # gen micro_batch topk_idx
        topk_idx_mb0 = self.gen_topk_idx_mb(split_sections[0])
        topk_idx_mb1 = self.gen_topk_idx_mb(split_sections[1])
        topk_idx_mb = [topk_idx_mb0, topk_idx_mb1]
        # gen micro_batch input_idx
        input_ids_mb = input_ids.split(split_sections, dim=1)
        # gen micro_batch actual_seq_lengths_kv
        actual_seq_lengths_kv_mb0 = torch.cumsum(kv_len_mb[0], dim=0).tolist()
        actual_seq_lengths_kv_mb1 = torch.cumsum(kv_len_mb[1], dim=0).tolist()
        actual_seq_lengths_kv_mb = [actual_seq_lengths_kv_mb0, actual_seq_lengths_kv_mb1]

        return input_ids_mb, kv_len_mb, actual_seq_lengths_kv_mb, topk_idx_mb

    def gen_topk_idx_mb(
        self,
        input_len
    ):
        if not self.perfect_eplb:
            return None
        # if use perfect_eplb
        global_rank = dist.get_rank()
        tokens_per_rank_prefill = input_len // self.attn_tp_size \
            if self.moe_ep_size != 1 else input_len * self.attn_dp_size
        step_prefill = tokens_per_rank_prefill * self.top_k
        cur_topk_list_prefill = [
            (i + global_rank) % self.config.n_routed_experts for i in range(step_prefill)]
        cur_topk_list = torch.Tensor(cur_topk_list_prefill).int().view(tokens_per_rank_prefill, -1).npu()
        return cur_topk_list

    def get_split_section(self, kv_len):
        # kv_len is seq_list
        seq_len_list = torch.cumsum(kv_len, dim=0).tolist()
        total_seq_num = seq_len_list[-1]
        half_seq = total_seq_num // 2
        balance_split = 0
        import bisect
        balance_split = bisect.bisect_right(seq_len_list, half_seq)
        kv_len_mb0 = kv_len[:balance_split]
        kv_len_mb1 = kv_len[balance_split:]
        kv_len_mb = [kv_len_mb0, kv_len_mb1]
        # The total length of seq for two batches
        split_sections = [torch.sum(kv_len_mb0), torch.sum(kv_len_mb1)]
        return kv_len_mb, split_sections

    def prepare_inputs_for_layer(self, input_ids, kv_len, position_ids, actual_seq_lengths_kv, is_prefill):
        sp_prefill = is_prefill and self.is_sp
        batch_size, seq_length = input_ids.shape
        input_ids, actual_seq_lengths_kv, hidden_states, _, seq_length_unpad = \
            self.calc_input_embeddings(input_ids, actual_seq_lengths_kv, sp_prefill)

        if sp_prefill:
            batch_size, seq_length = input_ids.shape
            # padding data adds to the last value
            kv_len[-1] = kv_len[batch_size - 1] + (seq_length - seq_length_unpad)

        cos_sin = self.rotary_emb(hidden_states, kv_len, self.max_position_embeddings, is_prefill=is_prefill,
                                  enable_pa=self.enable_pa)

        if sp_prefill:
            hidden_states = self.prepare_inputs_for_prefill_layer(hidden_states, input_ids)

        if self.enable_pa and not is_prefill:
            kv_len = kv_len.view(batch_size, -1) + self.kv_len_offset[:batch_size]
        residual = None
        slot_mapping = self.get_slot_mapping(kv_len, is_prefill, position_ids.device)
        return hidden_states, residual, kv_len, cos_sin, slot_mapping, actual_seq_lengths_kv

    def calc_input_embeddings(self, input_ids, actual_seq_lengths_kv, sp_prefill, prev_hidden_states=None):
        batch_size, seq_length = input_ids.shape

        seq_length_unpad = seq_length
        if sp_prefill:
            # seq pad in attention (SP + TP) and MoE(EP)
            padding_size = (seq_length_unpad + self.attn_tp_size - 1) // self.attn_tp_size * self.attn_tp_size \
                            - seq_length_unpad
            input_ids = torch.nn.functional.pad(input_ids, (0, padding_size, 0, 0), value=0)
            if prev_hidden_states is not None:
                prev_hidden_states = torch.nn.functional.pad(prev_hidden_states, (0, 0, 0, padding_size, 0, 0), value=0)
            batch_size, seq_length = input_ids.shape
            # padding data needs cal in PFA
            actual_seq_lengths_kv.append(seq_length)

        if self.embed_tp_size > 1:
            embed_tp_group = self.hccl_comm_dict.get("embed_tp_group", None)
            if self.embed_tp_size > self.attn_tp_size:
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

            if self.embed_tp_size == self.attn_tp_size:
                dist.all_reduce(inputs_embeds, group=embed_tp_group)
            elif self.embed_tp_size > self.attn_tp_size:
                if input_ids.ndim == 1:
                    inputs_embeds_attn = inputs_embeds.new_empty(seq_length, inputs_embeds.shape[-1])
                else:
                    inputs_embeds_attn = inputs_embeds.new_empty(batch_size, seq_length, inputs_embeds.shape[-1])
                dist.reduce_scatter_tensor(inputs_embeds_attn, inputs_embeds, group=embed_tp_group)
                inputs_embeds = inputs_embeds_attn
        else:
            inputs_embeds = self.embed_tokens(input_ids)
        return input_ids, actual_seq_lengths_kv, inputs_embeds, prev_hidden_states, seq_length_unpad

    def forward_microbatch_v2(
        self,
        input_ids_mb: torch.Tensor,
        kv_len_mb: torch.IntTensor,
        actual_seq_lengths_kv_mb: list,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        is_prefill: Optional[bool] = False,
        topk_idx_mb: Optional[list] = None,
        **kwargs,
    ):
        cur_stream = torch.npu.current_stream()
        # generate input
        with torch.npu.stream(cur_stream):
            # gengrate mb0 layer input
            input_ids_mb0 = input_ids_mb[0]
            kv_len_mb0 = kv_len_mb[0]
            actual_seq_lengths_kv_mb0 = actual_seq_lengths_kv_mb[0]
            hidden_states_mb0, residual_mb0, _, cos_sin_mb0, slot_mapping_mb0, actual_seq_lengths_kv_mb0 =\
                self.prepare_inputs_for_layer(input_ids_mb0, kv_len_mb0, position_ids, actual_seq_lengths_kv_mb0,
                                              is_prefill)

        with torch.npu.stream(self.stream1):
            # gengrate mb1 layer input
            input_ids_mb1 = input_ids_mb[1]
            kv_len_mb1 = kv_len_mb[1]
            actual_seq_lengths_kv_mb1 = actual_seq_lengths_kv_mb[1]
            hidden_states_mb1, residual_mb1, _, cos_sin_mb1, slot_mapping_mb1, actual_seq_lengths_kv_mb1 =\
            self.prepare_inputs_for_layer(input_ids_mb1, kv_len_mb1, position_ids, actual_seq_lengths_kv_mb1,
                                          is_prefill)
            slot_mapping_mb1 = slot_mapping_mb1 + len(kv_len_mb0) * self.pa_max_length

        for decode_layer in self.layers:
            with torch.npu.stream(cur_stream):
                hidden_states_mb0, residual_mb0 = decode_layer.forward_attn(
                    hidden_states_mb0,
                    kv_len_mb0,
                    actual_seq_lengths_kv_mb0,
                    cos_sin=cos_sin_mb0,
                    past_residual=residual_mb0,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    is_prefill=is_prefill,
                    slot_mapping=slot_mapping_mb0
                )

            with torch.npu.stream(self.stream1):
                hidden_states_mb1, residual_mb1 = decode_layer.forward_attn(
                    hidden_states_mb1,
                    kv_len_mb1,
                    actual_seq_lengths_kv_mb1,
                    cos_sin=cos_sin_mb1,
                    past_residual=residual_mb1,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    is_prefill=is_prefill,
                    slot_mapping=slot_mapping_mb1
                )

            with torch.npu.stream(cur_stream):
                hidden_states_mb0, residual_mb0 = decode_layer.forward_mlp(
                    hidden_states_mb0,
                    residual_mb0,
                    is_prefill,
                    topk_idx_mb[0]
                )

            with torch.npu.stream(self.stream1):
                hidden_states_mb1, residual_mb1 = decode_layer.forward_mlp(
                    hidden_states_mb1,
                    residual_mb1,
                    is_prefill,
                    topk_idx_mb[1]
                )

        cur_stream.wait_stream(self.stream1)
        self.stream1.wait_stream(cur_stream)
        hidden_states_mb0, _ = self.norm(hidden_states_mb0, residual_mb0)
        hidden_states_mb1, _ = self.norm(hidden_states_mb1, residual_mb1)

        # [B,S,H] concat S
        return torch.cat([hidden_states_mb0, hidden_states_mb1], dim=1)


    def forward_microbatch_v1(
        self,
        input_ids_mb: torch.Tensor,
        kv_len_mb: torch.IntTensor,
        actual_seq_lengths_kv_mb: list,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        is_prefill: Optional[bool] = False,
        topk_idx_mb: Optional[list] = None,
        **kwargs,
    ):
        cur_stream = torch.npu.current_stream()
        # generate input
        # gengrate mb0 layer input
        input_ids_mb0 = input_ids_mb[0]
        kv_len_mb0 = kv_len_mb[0]
        actual_seq_lengths_kv_mb0 = actual_seq_lengths_kv_mb[0]
        hidden_states_mb0, residual_mb0, _, cos_sin_mb0, slot_mapping_mb0, actual_seq_lengths_kv_mb0 =\
        self.prepare_inputs_for_layer(input_ids_mb0, kv_len_mb0, position_ids, actual_seq_lengths_kv_mb0, is_prefill)

        # gengrate mb1 layer input
        input_ids_mb1 = input_ids_mb[1]
        kv_len_mb1 = kv_len_mb[1]
        actual_seq_lengths_kv_mb1 = actual_seq_lengths_kv_mb[1]
        hidden_states_mb1, residual_mb1, _, cos_sin_mb1, slot_mapping_mb1, actual_seq_lengths_kv_mb1 =\
        self.prepare_inputs_for_layer(input_ids_mb1, kv_len_mb1, position_ids, actual_seq_lengths_kv_mb1, is_prefill)
        slot_mapping_mb1 = slot_mapping_mb1 + len(kv_len_mb0) * self.pa_max_length

        for layer_id, decode_layer in enumerate(self.layers):
            # disable micro_batch in dense layers
            if layer_id < self.config.first_k_dense_replace:
                hidden_states_mb0, residual_mb0 = decode_layer.forward_attn(
                    hidden_states_mb0,
                    kv_len_mb0,
                    actual_seq_lengths_kv_mb0,
                    cos_sin=cos_sin_mb0,
                    past_residual=residual_mb0,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    is_prefill=is_prefill,
                    slot_mapping=slot_mapping_mb0
                )
                hidden_states_mb1, residual_mb1 = decode_layer.forward_attn(
                    hidden_states_mb1,
                    kv_len_mb1,
                    actual_seq_lengths_kv_mb1,
                    cos_sin=cos_sin_mb1,
                    past_residual=residual_mb1,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    is_prefill=is_prefill,
                    slot_mapping=slot_mapping_mb1
                )

                hidden_states_mb0, residual_mb0 = decode_layer.forward_mlp(
                    hidden_states_mb0,
                    residual_mb0,
                    is_prefill,
                    topk_idx_mb[0]
                )
                hidden_states_mb1, residual_mb1 = decode_layer.forward_mlp(
                    hidden_states_mb1,
                    residual_mb1,
                    is_prefill,
                    topk_idx_mb[1]
                )
            else:
                hidden_states_mb0, residual_mb0 = decode_layer.forward_attn(
                    hidden_states_mb0,
                    kv_len_mb0,
                    actual_seq_lengths_kv_mb0,
                    cos_sin=cos_sin_mb0,
                    past_residual=residual_mb0,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    is_prefill=is_prefill,
                    slot_mapping=slot_mapping_mb0
                )
                # Cover up the combination of the previous round with the next round's atten
                if layer_id > self.config.first_k_dense_replace:
                    cur_stream.wait_event(event_combine_finalize_routing_mb1)
                    hidden_states_mb1 = decode_layer.forward_finalize_routing(hidden_states_mb1, gathered_tokens_mb1,
                    hidden_states_share_mb1, topk_weight_mb1, expanded_row_idx_mb1)

                hidden_states_mb0, residual_mb0 = decode_layer.forward_post_attention_layernorm(hidden_states_mb0,
                                                                                                 residual_mb0)
                expanded_x_mb0, expanded_row_idx_mb0, tokens_per_expert_mb0, pertoken_scale_mb0, topk_weight_mb0 = \
                    decode_layer.forward_gate_init_routing(hidden_states_mb0, topk_idx_mb[0])
                event_routing_dispatch_mb0 = cur_stream.record_event()

                hidden_states_mb1, residual_mb1 = decode_layer.forward_attn(
                    hidden_states_mb1,
                    kv_len_mb1,
                    actual_seq_lengths_kv_mb1,
                    cos_sin=cos_sin_mb1,
                    past_residual=residual_mb1,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    is_prefill=is_prefill,
                    slot_mapping=slot_mapping_mb1
                )
                hidden_states_mb1, residual_mb1 = decode_layer.forward_post_attention_layernorm(hidden_states_mb1,
                                                                                                 residual_mb1)
                expanded_x_mb1, expanded_row_idx_mb1, tokens_per_expert_mb1, pertoken_scale_mb1, topk_weight_mb1 = \
                    decode_layer.forward_gate_init_routing(hidden_states_mb1, topk_idx_mb[1])
                event_routing_dispatch_mb1 = cur_stream.record_event()

                hidden_states_share_mb0 = decode_layer.forward_shared_expert(hidden_states_mb0, is_prefill)
                hidden_states_share_mb1 = decode_layer.forward_shared_expert(hidden_states_mb1, is_prefill)

                with torch.npu.stream(self.stream1):
                    self.stream1.wait_event(event_routing_dispatch_mb0)
                    tokens_per_expert_group_mb0, gathered_tokens_mb0, gathered_pertoken_scale_mb0, input_splits_mb0,\
                          output_splits_mb0 = decode_layer.forward_dispatch_double_routing(tokens_per_expert_mb0,
                                                                            expanded_x_mb0, pertoken_scale_mb0)
                    event_dispatch_expert_mb0 = self.stream1.record_event()

                cur_stream.wait_event(event_dispatch_expert_mb0)
                new_x_mb0 = decode_layer.forward_expert(gathered_tokens_mb0, tokens_per_expert_group_mb0,
                                                         gathered_pertoken_scale_mb0)
                event_expert_combine_mb0 = cur_stream.record_event()

                with torch.npu.stream(self.stream1):
                    self.stream1.wait_event(event_routing_dispatch_mb1)
                    tokens_per_expert_group_mb1, gathered_tokens_mb1, gathered_pertoken_scale_mb1, input_splits_mb1,\
                          output_splits_mb1 = decode_layer.forward_dispatch_double_routing(tokens_per_expert_mb1,
                                                                            expanded_x_mb1, pertoken_scale_mb1)
                    event_dispatch_expert_mb1 = self.stream1.record_event()

                with torch.npu.stream(self.stream1):
                    self.stream1.wait_event(event_expert_combine_mb0)
                    gathered_tokens_mb0 = decode_layer.forward_combine_double_routing(new_x_mb0, expanded_x_mb0,
                                                                        input_splits_mb0, output_splits_mb0)
                    event_combine_finalize_routing_mb0 = self.stream1.record_event()

                cur_stream.wait_event(event_dispatch_expert_mb1)
                new_x_mb1 = decode_layer.forward_expert(gathered_tokens_mb1, tokens_per_expert_group_mb1,
                                                         gathered_pertoken_scale_mb1)
                event_expert_combine_mb1 = cur_stream.record_event()
                with torch.npu.stream(self.stream1):
                    self.stream1.wait_event(event_expert_combine_mb1)
                    gathered_tokens_mb1 = decode_layer.forward_combine_double_routing(new_x_mb1, expanded_x_mb1,
                                                                        input_splits_mb1, output_splits_mb1)
                    event_combine_finalize_routing_mb1 = self.stream1.record_event()

                cur_stream.wait_event(event_combine_finalize_routing_mb0)
                hidden_states_mb0 = decode_layer.forward_finalize_routing(hidden_states_mb0, gathered_tokens_mb0,
                    hidden_states_share_mb0, topk_weight_mb0, expanded_row_idx_mb0)

        cur_stream.wait_event(event_combine_finalize_routing_mb1)
        hidden_states_mb1 = decode_layer.forward_finalize_routing(hidden_states_mb1, gathered_tokens_mb1,
                hidden_states_share_mb1, topk_weight_mb1, expanded_row_idx_mb1)

        cur_stream.wait_stream(self.stream1)
        self.stream1.wait_stream(cur_stream)
        hidden_states_mb0, _ = self.norm(hidden_states_mb0, residual_mb0)
        hidden_states_mb1, _ = self.norm(hidden_states_mb1, residual_mb1)

        # [B,S,H] concat S
        return torch.cat([hidden_states_mb0, hidden_states_mb1], dim=1)


class DeepseekV3ModelMTPLayer(DeepseekV3Model):
    def __init__(self, config: DeepseekV3Config, runner_settings: Dict, **kwargs):
        super().__init__(config, runner_settings, **kwargs)

    def init_modules(self, config, **kwargs):
        self.embed_tokens = None
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.layers = nn.ModuleDict(
            {
                str(self.mtp_start_layer_idx + i):
                DeepseekV3DecoderLayer(config, self.runner_settings, self.mtp_start_layer_idx + i,
                                       prefix=f"model.layers.{self.mtp_start_layer_idx + i}", **kwargs)
                for i in range(config.num_nextn_predict_layers)
        })

    def get_layer(self, i):
        return list(self.layers.values())[i]

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
        cur_topk_list: Optional[torch.Tensor] = None,
        slot_mapping: Optional[torch.Tensor] = None,
        mtp_layer_idx: Optional[int] = 0,
        **kwargs,
    ) -> torch.Tensor:
        return self.get_layer(mtp_layer_idx)(
            hidden_states,
            kv_len,
            actual_seq_lengths_kv,
            cos_sin=cos_sin,
            past_residual=past_residual,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            is_prefill=is_prefill,
            cur_topk_list=cur_topk_list,
            slot_mapping=slot_mapping
        )


class DeepseekV3ForCausalLM(DeepseekV3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, runner_settings):
        super().__init__(config)
        self.config = config
        self.runner_settings = runner_settings
        self.input_max_len = int(os.getenv("INPUT_MAX_LEN", 1024))
        self.get_parallel_settings()
        self.experts_per_rank = config.n_routed_experts // self.moe_ep_size
        self.top_k = config.num_experts_per_tok
        self.max_position_embeddings = self.runner_settings.get("data_config").get("max_position_embeddings", 4096)
        self.perfect_eplb = self.runner_settings.get("model_config").get("perfect_eplb", False)
        self.next_n = self.runner_settings.get("model_config").get("next_n", 0)
        self.enable_o_proj_alltoall = self.runner_settings.get("parallel_config").get("enable_o_proj_alltoall", False)
        self.hidden_size = config.hidden_size
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.rank_offset = int(os.getenv("RANK_OFFSET", "0"))
        self.global_rank = self.local_rank + self.rank_offset
        self.kwargs = {"global_rank": self.global_rank}
        self.kwargs.update({"is_sp": self.is_sp})
        self.default_pg = get_default_group()
        if self.default_pg is not None:
            if dist.get_world_size() > 1:
                self.hccl_comm_dict = self.init_parallel_comm_group()
                self.kwargs.update({"hccl_comm_dict": self.hccl_comm_dict})
        self.enable_pa = self.runner_settings.get("model_config").get("enable_pa", False)
        self.use_aclgraph = self.runner_settings.get("exe_mode", "ge_graph") == "acl_graph"
        if self.enable_pa:
            batch_size_per_rank = self.runner_settings.get("data_config").get("batch_size_per_rank", 1)
            self.pa_max_length = self.runner_settings.get("model_config").get("pa_max_length", 2048)
            self.block_size = self.runner_settings.get("model_config").get("pa_block_size", 128)
            self.cache_len = self.pa_max_length // self.block_size
            self.kv_cache_num_block = self.cache_len * batch_size_per_rank
            self.kv_len_offset = torch.arange(
                0,
                batch_size_per_rank * self.pa_max_length,
                self.pa_max_length,
                dtype=torch.int64,
                device="npu",
            ).view(-1, 1)
            self.kwargs.update({"kv_len_offset": self.kv_len_offset})

        self.model = DeepseekV3Model(config, self.runner_settings, **self.kwargs)
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

        self.enable_prefill_multi_cycle = runner_settings.get("model_config").get("enable_prefill_multi_cycle", False)
        self.micro_batch_mode = MicroBatchMode(self.runner_settings.get("model_config").get("micro_batch_mode", 0))

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

    def get_parallel_settings(self):
        self.embed_tp_size = self.runner_settings.get("parallel_config").get("embed_tp_size", 1)
        self.attn_dp_size = self.runner_settings.get("parallel_config").get("attn_dp_size", 1)
        self.attn_tp_size = self.runner_settings.get("parallel_config").get("attn_tp_size", 1)
        self.moe_ep_size = self.runner_settings.get("parallel_config").get("moe_ep_size", 1)
        self.moe_tp_size = self.runner_settings.get("parallel_config").get("moe_tp_size", 1)
        self.lmhead_tp_size = self.runner_settings.get("parallel_config").get("lmhead_tp_size", self.embed_tp_size)
        self.moe_dp_size = self.runner_settings.get("parallel_config").get("moe_dp_size", 1)
        self.embed_dp_size = self.runner_settings.get("parallel_config").get("embed_dp_size", 1)
        self.attn_dp_size = self.runner_settings.get("parallel_config").get("attn_dp_size", 1)
        self.dense_tp_size = self.runner_settings.get("parallel_config").get("dense_tp_size", 1)
        self.is_sp = self.attn_tp_size > 1 and self.moe_ep_size > 1
        self.micro_batch_mode = MicroBatchMode(self.runner_settings.get("model_config").get("micro_batch_mode", 0))

    def init_parallel_comm_group(self):
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()

        attn_tp_group = init_comm_group(
            global_rank=global_rank, group_num=self.attn_dp_size, world_size=world_size,
            group_stride=1, group_name="attn_tp_group")

        attn_tp_group_stream1 = None
        if self.micro_batch_mode == MicroBatchMode.PREFILL_MICRO_BATCH_SP_TP_EP:
            attn_tp_group_stream1 = init_comm_group(
                global_rank=global_rank, group_num=self.attn_dp_size, world_size=world_size,
                group_stride=1, group_name="attn_tp_group_stream1")

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

        moe_ep_group_stream1 = None
        if self.micro_batch_mode == MicroBatchMode.PREFILL_MICRO_BATCH_SP_TP_EP:
            moe_ep_group_stream1 = init_comm_group(
                global_rank=global_rank, group_num=self.moe_tp_size, world_size=world_size,
                group_stride=self.moe_tp_size, group_name="moe_ep_group_stream1")

        hccl_comm_dict = {
                "default_pg": get_default_group(),
                "attn_tp_group": attn_tp_group, "embed_tp_group": embed_tp_group,
                "moe_tp_group": moe_tp_group, "moe_ep_group": moe_ep_group,
                "moe_ep_group_name": moe_ep_group_name,
                "lmhead_tp_group": lmhead_tp_group,
                "dense_tp_group": dense_tp_group,
                "attn_tp_group_stream1": attn_tp_group_stream1,
                "moe_ep_group_stream1": moe_ep_group_stream1,
            }
        return hccl_comm_dict

    def forward_lm_head(self, outputs, position_ids, is_prefill=True):
        bs, q_len, hidden_size = outputs.shape
        if is_prefill:
            if not self.enable_pa:
                gather_index, _ = torch.max(position_ids, dim=-1)
                gather_index = gather_index.unsqueeze(1).unsqueeze(2).repeat(1, 1, outputs.shape[-1])
                outputs = torch.gather(outputs, 1, gather_index)
            else:
                bs = position_ids.shape[0]
                gather_index, _ = torch.max(position_ids, dim=-1)
                seq_index = ((gather_index + 1).to(torch.int32).cumsum(-1) - 1).npu()
                outputs = (torch.index_select(outputs.view(1, -1, hidden_size), 1, seq_index.view(-1))).view(bs, 1, -1)
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

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        kv_len: torch.IntTensor = None,
        actual_seq_lengths_kv: list = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        is_prefill: Optional[bool] = False,
        cur_topk_list: Optional[torch.Tensor] = None,
        prev_hidden_states: Optional[torch.Tensor] = None,
        **kwargs
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
            cur_topk_list=cur_topk_list,
        ) # (bs / attn_dp, S, hidden_size)

        # attention: SP + TP，moe：DP + EP
        if is_prefill and self.is_sp:
            new_outputs = torch.zeros_like(outputs).repeat(self.attn_tp_size, 1, 1)
            dist.all_gather_into_tensor(new_outputs, outputs, group=self.hccl_comm_dict.get("attn_tp_group", None))
            outputs = new_outputs
        prev_hidden_states = outputs
        if is_prefill:
            prev_hidden_states = prev_hidden_states.view(1, -1, self.hidden_size)
        logits = self.forward_lm_head(outputs, position_ids, is_prefill)
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

    def decode(
        self,
        **kwargs
    ):
        logits, prev_hidden_states = self.forward(
            is_prefill=False,
            **kwargs
        )
        return logits, prev_hidden_states

    def init_cache(
        self,
        input_ids
    ):
        batch_size, seq_len = input_ids.size()
        cache_seq_len = self.max_position_embeddings
        dtype = self.config.torch_dtype

        past_key_values = ()

        num_hidden_layers = len(self.model.layers)
        if self.enable_pa:
            cache_nope_shape = (
                            self.kv_cache_num_block,
                            self.block_size,
                            self.config.kv_lora_rank
                        )

            cache_rope_shape = (
                            self.kv_cache_num_block,
                            self.block_size,
                            self.config.qk_rope_head_dim
                        )

            for _ in range(num_hidden_layers):
                cache_nope = torch.zeros(cache_nope_shape, dtype=dtype, device=input_ids.device)
                cache_rope = torch.zeros(cache_rope_shape, dtype=dtype, device=input_ids.device)
                past_key_values += ((cache_nope, cache_rope),)
        else:
            cache_key_shape = (
                            batch_size,
                            1,
                            cache_seq_len,
                            self.config.kv_lora_rank + self.config.qk_rope_head_dim
                        )

            for _ in range(num_hidden_layers):
                key_cache = torch.zeros(cache_key_shape, dtype=dtype, device=input_ids.device)
                past_key_values += ((key_cache, ),)

        return past_key_values

    def generate_decoder_mask(self, batch_size: int, s: int, sq: int, kv_len: torch.Tensor) -> torch.Tensor:

        compressed_count = (s - self.compress_block_size) // self.sliding_stride + 1

        seq_indices = torch.arange(sq).view(1, sq, 1).npu()
        k_indices = torch.arange(compressed_count).view(1, 1, compressed_count).npu()

        start_pos = kv_len.view(batch_size, 1, 1) + seq_indices
        valid_mask = ((k_indices < (start_pos - self.compress_block_size) // self.sliding_stride + 1))
        valid_mask = valid_mask & (start_pos >= 0)
        valid_mask = k_indices.masked_fill(~valid_mask, torch.iinfo(k_indices.dtype).min)
        return valid_mask

    def generate_prefill_mask(self, batch_size: int, s_max: int, actual_lengths: torch.Tensor) -> torch.Tensor:

        compressed_count = (s_max - self.compress_block_size) // self.sliding_stride + 1

        s_indices = torch.arange(s_max).view(1, s_max, 1).npu()
        i_indices = torch.arange(compressed_count).view(1, 1, compressed_count).npu()

        mask = i_indices < ((s_indices - self.compress_block_size) // self.sliding_stride + 1)

        actual_lengths = torch.Tensor(actual_lengths).npu()
        seq_mask = torch.arange(s_max).expand(batch_size, s_max).npu() < actual_lengths.unsqueeze(0)
        seq_mask_expanded = seq_mask.unsqueeze(-1).expand(-1, -1, compressed_count)
        final_mask = mask & seq_mask_expanded
        mask = mask.to(torch.int32)
        final_mask = mask.masked_fill(~final_mask, torch.iinfo(mask.dtype).min)
        return final_mask

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
            tokens_per_rank_prefill = (batch_size * seq_len + self.attn_tp_size - 1) // self.attn_tp_size \
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
                    cur_topk_list_decode = cur_topk_list_decode + \
                    [i for i in range(offset * self.experts_per_rank, \
                                        offset * self.experts_per_rank + tokens_per_rank_decode)]
                cur_topk_list = torch.Tensor(cur_topk_list_decode).int().view(batch_size * seq_len, -1).npu()
            else:
                step_decode = batch_size * self.top_k * seq_len
                cur_topk_list_decode = [
                    (i + global_rank) % self.config.n_routed_experts for i in range(step_decode)
                ]
                cur_topk_list = torch.Tensor(cur_topk_list_decode).int().view(batch_size * seq_len, -1).npu()
        return cur_topk_list

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
        # use reshape to avoid stride change, which will cause recompile in mtp case
        input_ids = input_ids.contiguous().reshape(batch_size, seq_len)
        if past_key_values is None:
            raise ValueError("past_key_values should be initialized first!")
        if is_prefill:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            attention_mask = share_mask_tril
            if self.enable_pa:
                # Obtain the actual length of the request
                kv_len = torch.max(position_ids, axis=1)[0] + 1
                result = []
                for i in range(batch_size):
                    result.append(input_ids[i, :kv_len[i]])
                result = torch.cat(result, dim=0)
                 # add batch dimension
                input_ids = result.unsqueeze(0)
                actual_seq_lengths_kv = torch.cumsum(kv_len, dim=0).tolist()
            else:
                kv_len = torch.zeros(batch_size, dtype=torch.int64, device=input_ids.device)
                actual_seq_lengths_kv = None
        else:
            if seq_len > 1: # fa requires sparse mode 3 and 2048 * 2048 mask for mtp
                attention_mask = get_init_attn_mask(2048, kv_len.device)
                last_kv = torch.max(kv_len, axis=1)[0]
                if self.runner_settings.get("exe_mode") == "ge_graph":
                    # dynamo use fa_tensor
                    actual_seq_lengths_kv = (last_kv + 1)
                else:
                    actual_seq_lengths_kv = (last_kv + 1).cpu().detach().tolist()

            else:
                attention_mask = None
                if self.runner_settings.get("exe_mode") == "ge_graph":
                    # dynamo use fa_tensor
                    actual_seq_lengths_kv = (kv_len + 1)
                else:
                    actual_seq_lengths_kv = (kv_len + 1).cpu().detach().tolist()

            position_ids = kv_len.view(-1, 1)
        # attention_mask set
        if not self.enable_pa:
            if is_prefill:
                past_key_values_length = 0
                sliding_window = self.input_max_len
                input_mask = None
            else:
                past_key_values_length = self.max_position_embeddings - seq_len
                sliding_window = min(self.max_position_embeddings, input_lens)
                share_mask_tril = get_decode_mask(mask_length=self.max_position_embeddings,
                                                  device=input_ids.device,
                                                  position=input_lens)
                share_mask_tril = self._repeat_batch(share_mask_tril, seq_len)
                share_mask_tril = share_mask_tril[None, None, ...]
                input_mask = self._repeat_batch(share_mask_tril, batch_size)

            attention_mask = _prepare_4d_causal_attention_mask(
                input_mask,
                (batch_size, seq_len),
                input_ids.float(),
                past_key_values_length,
                sliding_window
            )

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

    # Adapted from vllm.model_executor.models.deepseek_v2.DeepseekV2ForCausalLM.load_weights
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("merge_up_gate_proj", "gate_proj", 0),
            ("merge_up_gate_proj", "up_proj", 1),
        ]

        repeat_loaded_weights_mapping = [] # (origin_name: repeat_loaded_name)
        if self.enable_o_proj_alltoall:
            repeat_loaded_weights_mapping.append(("o_proj", "o_proj_ata"))


        # Params for weights, int8 weight scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoEGMM.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts)

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if self.config.architectures[0] == 'DeepseekV3ForCausalLM' and self.config.num_nextn_predict_layers > 0:
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


class DeepseekV3ModelMTP(DeepseekV3ForCausalLM):

    def __init__(self, config: DeepseekV3Config, runner_settings: Dict):
        super().__init__(config, runner_settings)
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.rank_offset = int(os.getenv("RANK_OFFSET", "0"))
        self.global_rank = self.local_rank + self.rank_offset
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.vocab_size_per_rank = self.vocab_size // self.embed_tp_size
        self.ignore_share_weight = False

        # reuse lm_head, rotary_emb from main model
        self.lm_head = None
        self.rotary_emb = None
        self.model = DeepseekV3ModelMTPLayer(config, self.runner_settings, **self.kwargs)

        self.shared_head_norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.enorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # prev_hidden_states and input_hidden_state feature fusion
        self.eh_proj = ReplicatedLinear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.enable_pa = self.runner_settings.get("model_config").get("enable_pa", False)

    def get_slot_mapping(self, kv_len, is_prefill, device):
        if not is_prefill:
            return None
        all_tensors = []
        for i, seq_len in enumerate(kv_len):
            new_index = torch.arange(self.pa_max_length * i, seq_len.item() + self.pa_max_length * i,
                                     dtype=kv_len.dtype, device=device)
            all_tensors.append(new_index)
        return torch.cat(all_tensors)

    def set_share_weight(self, target_model):
        if self.ignore_share_weight:
            for _, layer in self.layers.items():
                layer.embed_tokens = target_model.model.embed_tokens
                layer.shared_head.head = target_model.lm_head

    @add_start_docstrings_to_model_forward(DEEPSEEKV3_INPUTS_DOCSTRING)
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
        cur_topk_list: Optional[torch.Tensor] = None,
    ):
        sp_prefill = is_prefill and self.is_sp
        batch_size, seq_length = input_ids.shape
        input_ids, actual_seq_lengths_kv, hidden_states, prev_hidden_states, seq_length_unpad = \
            self.model.calc_input_embeddings(input_ids, actual_seq_lengths_kv, sp_prefill,
                                             prev_hidden_states=prev_hidden_states)

        hidden_states = self.enorm(hidden_states)
        prev_hidden_states = self.hnorm(prev_hidden_states)
        hidden_states_eh = torch.cat([hidden_states, prev_hidden_states], dim=-1)
        hidden_states = self.eh_proj(hidden_states_eh)

        if sp_prefill:
            batch_size, seq_length = input_ids.shape
            # padding data adds to the last value
            kv_len[-1] = kv_len[batch_size - 1] + (seq_length - seq_length_unpad)

        cos_sin = self.rotary_emb(hidden_states, kv_len, self.max_position_embeddings, is_prefill=is_prefill,
                                  enable_pa=self.enable_pa)

        if sp_prefill:
            hidden_states = self.model.prepare_inputs_for_prefill_layer(hidden_states, input_ids)

        if self.enable_pa and not is_prefill:
            kv_len = kv_len.view(batch_size, -1) + self.kv_len_offset[:batch_size]
        residual = None
        slot_mapping = self.get_slot_mapping(kv_len, is_prefill, position_ids.device)

        residual, hidden_states = self.model(
            hidden_states,
            kv_len,
            actual_seq_lengths_kv,
            cos_sin=cos_sin,
            past_residual=residual,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            is_prefill=is_prefill,
            cur_topk_list=cur_topk_list,
            slot_mapping=slot_mapping
        )

        prev_hidden_states, _ = self.shared_head_norm(hidden_states, residual)
        # attention: SP + TP，moe：DP + EP
        if is_prefill and self.is_sp:
            new_outputs = torch.zeros_like(prev_hidden_states).repeat(self.attn_tp_size, 1, 1)
            dist.all_gather_into_tensor(new_outputs, prev_hidden_states,
                                        group=self.hccl_comm_dict.get("attn_tp_group", None))
            prev_hidden_states = new_outputs
        outputs = prev_hidden_states
        if is_prefill:
            prev_hidden_states = prev_hidden_states.view(1, -1, self.hidden_size)
        logits = self.forward_lm_head(outputs=outputs, position_ids=position_ids, is_prefill=is_prefill)

        return logits, prev_hidden_states

    # Adapted from vllm.model_executor.models.deepseek_mtp.DeepSeekMTP.load_weights
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping, mtp_unique_weight_mapping, expert_params_mapping, repeat_loaded_weights_mapping \
            = self._load_weight_map()

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.ignore_share_weight and any(
                    substring in name for substring in ["embed_tokens.weight", "shared_head.head"]):
                continue
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
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

    def _load_weight_map(self):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("merge_up_gate_proj", "gate_proj", 0),
            ("merge_up_gate_proj", "up_proj", 1),
        ]

        mtp_unique_weight_mapping = [
            # (param_name, weight_name)
            ("shared_head_norm", "shared_head.norm"),
            ("enorm", "enorm"),
            ("hnorm", "hnorm"),
            ("eh_proj", "eh_proj")
        ]

        # Params for weights, int8 weight scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoEGMM.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts)

        repeat_loaded_weights_mapping = [] # (origin_name: repeat_loaded_name)
        if self.enable_o_proj_alltoall:
            repeat_loaded_weights_mapping.append(("o_proj", "o_proj_ata"))
        return stacked_params_mapping, mtp_unique_weight_mapping, expert_params_mapping, repeat_loaded_weights_mapping


def get_spec_layer_idx_from_weight_name(config,
                                        weight_name: str) -> Optional[int]:
    if hasattr(config,
               "num_nextn_predict_layers") and (config.num_nextn_predict_layers
                                                > 0):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(f"model.layers.{layer_idx+i}."):
                return layer_idx + i
    return None
