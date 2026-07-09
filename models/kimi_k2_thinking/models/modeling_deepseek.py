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
import gc
from operator import attrgetter
from typing import List, Optional, Tuple, Union, Dict, Iterable, Set

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from torch import nn
import torch.distributed as dist

import torch_npu
import torchair as tng

from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)

from executor.utils import (
    override, calc_moe_hccl_buffer_size,
)
from executor.utils.forward_metadata import ForwardMetaData, PrefillCPMetaData
from executor.core.config import InferenceConfig, CommManager

from executor.core.kv_cache.cache_info import CacheEntry, LayerCacheInfo, ModelCacheInfo

from executor.model_loader.weight_utils import default_weight_loader
from executor.utils import npu_stream_switch, npu_wait_tensor, superkernel_scope
from executor.utils.stream_utils import (
    record_event,
    wait_event,
    record_stream,
    npu_stream_switch as npu_stream_switch_npugraph,
)
from module.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding
    )
from module.fuse_moe_gmm import FusedMoEGMM
from module.quantization import QuantizeMethodBase
from module.quantization.compressed_tensors.compressed_tensors_moe_gmm import CompressedTensorW8A8Int8MoEGMMMethod
from .configuration_deepseek import DeepseekV3Config
from .modules import (_prepare_4d_causal_attention_mask, one_hot, yarn_get_mscale,
                      DeepseekV3RMSNorm, apply_rotary_pos_emb, _init_rope, DEEPSEEKV3_START_DOCSTRING,
                      DEEPSEEKV3_INPUTS_DOCSTRING
                    )

logger = logging.get_logger(__name__)


class DeepseekV3DenseMLP(nn.Module):
    def __init__(self, config, infer_config: InferenceConfig, comm_manager: CommManager, prefix, **kwargs):
        super().__init__()
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.mm_quant_mode = (
            config.quant_config.mm_quant_mode
            if config.quant_config is not None
            else "w16a16")
        self.moe_ep_size = self.infer_config.parallel_config.moe_ep_size
        self.moe_tp_size = self.infer_config.parallel_config.moe_tp_size
        self.dense_tp_size = self.infer_config.parallel_config.dense_tp_size
        self.config = config
        self.hidden_size = config.hidden_size
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[config.intermediate_size] * 2,
            bias=False,
            tp_size=self.dense_tp_size,
            tp_rank=self.comm_manager.get_rank("dense_tp_group") if self.dense_tp_size > 1 else 0,
            quant_config=config.quant_config,
            prefix=f"{prefix}.gate_up_proj"
            )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            tp_size=self.dense_tp_size,
            tp_rank=self.comm_manager.get_rank("dense_tp_group") if self.dense_tp_size > 1 else 0,
            quant_config=config.quant_config,
            prefix=f"{prefix}.down_proj")

    def forward(self, x, is_prefill=False):
        # input_DP + attention_TP + moe_EP. Packed token-first [T, H].
        if self.dense_tp_size > 1 and self.moe_ep_size > 1:
            num_tokens, _ = x.shape
            x_output = torch.empty([num_tokens * self.dense_tp_size, self.hidden_size], \
                                   dtype=x.dtype, device="npu")
            dist.all_gather_into_tensor(x_output, x, group=self.comm_manager.get_group("dense_tp_group"))
            x = x_output

        if self.mm_quant_mode == "w8a8int8":
            down_proj = self.forward_w8a8int8(x)
        else:
            down_proj = self.forward_normal(x)

        if self.dense_tp_size > 1 and self.moe_ep_size > 1:
            mlp_res = down_proj.new_empty(num_tokens, down_proj.shape[-1])
            dist.reduce_scatter_tensor(mlp_res, down_proj, group=self.comm_manager.get_group("dense_tp_group"))
            down_proj = mlp_res
        elif self.dense_tp_size > 1 and self.moe_tp_size > 1:
            dist.all_reduce(down_proj, group=self.comm_manager.get_group("dense_tp_group"))

        return down_proj

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
            activation_scale=pertoken_scale
        )
        return self.down_proj(intermediate_hidden_states, pertoken_scale)


class DeepseekV3SharedExpert(nn.Module):
    def __init__(self, config, infer_config: InferenceConfig, comm_manager: CommManager,
                 is_moe_layer=False, prefix="", **kwargs):
        super().__init__()
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.mm_quant_mode = (
            config.quant_config.mm_quant_mode
            if config.quant_config is not None
            else "w16a16")
        self.moe_tp_size = self.infer_config.parallel_config.moe_tp_size
        self.moe_ep_size = self.infer_config.parallel_config.moe_ep_size
        self.config = config
        self.hidden_size = config.hidden_size
        self.is_moe_layer = is_moe_layer
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[config.moe_intermediate_size * config.n_shared_experts] * 2,
            bias=False,
            tp_size=self.moe_tp_size,
            tp_rank=self.comm_manager.get_rank("moe_tp_group") if self.moe_tp_size > 1 else 0,
            quant_config=config.quant_config,
            prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(
            config.moe_intermediate_size * config.n_shared_experts,
            config.hidden_size,
            bias=False,
            tp_size=self.moe_tp_size,
            tp_rank=self.comm_manager.get_rank("moe_tp_group") if self.moe_tp_size > 1 else 0,
            quant_config=config.quant_config,
            prefix=f"{prefix}.down_proj")

    def forward(self, x, enable_decode_stream=False, shared_expert_event=None):
        if self.mm_quant_mode == "w8a8int8":
            down_proj = self.forward_w8a8int8(x, enable_decode_stream, shared_expert_event)
        else:
            down_proj = self.forward_normal(x, enable_decode_stream, shared_expert_event)
        return down_proj

    def forward_normal(self, x, enable_decode_stream=False, shared_expert_event=None):
        merged_x = self.gate_up_proj(x)
        intermediate_hidden_states = torch_npu.npu_swiglu(merged_x)
        wait_event(enable_decode_stream, shared_expert_event, 0)
        return self.down_proj(intermediate_hidden_states)

    def forward_w8a8int8(self, x, enable_decode_stream=False, shared_expert_event=None):
        merged_x, pertoken_scale = self.gate_up_proj(x, out_dtype=torch.int32)
        intermediate_hidden_states, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
            merged_x, weight_scale=self.gate_up_proj.weight_scale,
            quant_scale=self.down_proj.smooth_scales,
            quant_mode=1, activate_left=True,
            activation_scale=pertoken_scale
        )
        wait_event(enable_decode_stream, shared_expert_event, 0)
        return self.down_proj(intermediate_hidden_states, pertoken_scale)


class DeepseekV3MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config, infer_config: InferenceConfig, comm_manager: CommManager,
                 layer_idx: int, prefix: str, **kwargs):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.gmm_quant_mode = (
            config.quant_config.gmm_quant_mode
            if config.quant_config is not None
            else "w16a16")
        self.mm_quant_mode = (
            config.quant_config.mm_quant_mode
            if config.quant_config is not None
            else "w16a16")
        self.hidden_dim = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.moe_tp_size = self.infer_config.parallel_config.moe_tp_size
        self.moe_ep_size = self.infer_config.parallel_config.moe_ep_size
        self.exe_mode = self.infer_config.model_config.exe_mode
        self.enable_multi_streams = self.infer_config.model_config.custom_params.get("enable_multi_streams", False)
        self.enable_gegraph = self.exe_mode == "ge_graph"
        self.enable_npugraph_ex = self.exe_mode == "npugraph_ex"
        self.enable_gegraph_and_multistream = self.enable_multi_streams and self.enable_gegraph
        self.enable_npugraphex_and_multistream = self.enable_multi_streams and self.enable_npugraph_ex
        self.npu_events = []
        self.shared_expert_event = []
        if self.enable_npugraphex_and_multistream:
            self.npu_events = [torch.npu.Event(), torch.npu.Event()]
            self.shared_expert_event = [torch.npu.Event()]
        self.force_eplb = self.infer_config.model_config.force_eplb
        self.num_experts_per_tok = config.num_experts_per_tok
        # total experts num
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok

        self.intermediate_size_per_rank = self.intermediate_size // self.moe_tp_size
        self.shared_expert_rank_num = 0 # route and share on same card
        self.n_shared_experts = config.n_shared_experts
        self.n_routed_experts = config.n_routed_experts
        self.experts_per_rank = config.n_routed_experts // self.moe_ep_size
        self.experts = FusedMoEGMM(
            num_experts=config.n_routed_experts,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_size,
            bias=False,
            quant_config=config.quant_config,
            tp_size=self.moe_tp_size,
            tp_rank=self.comm_manager.get_rank("moe_tp_group") if self.moe_tp_size > 1 else 0,
            ep_size=self.moe_ep_size,
            ep_rank=self.comm_manager.get_rank("moe_ep_group") if self.moe_ep_size > 1 else 0,
            prefix=f"{prefix}.experts",
        )

        self._init_gate(prefix)
        if config.n_shared_experts is not None:
            self.shared_experts = DeepseekV3SharedExpert(config, self.infer_config, self.comm_manager,
                                        is_moe_layer=True, prefix=f"{prefix}.shared_experts", **kwargs)

        self.moe_ep_group = self.comm_manager.get_group("moe_ep_group") if self.moe_ep_size > 1 else None
        self.dispatch_kwargs = None
        self.combine_kwargs = None

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
        num_tokens, h = hidden_states.shape   # packed [T, H]
        # compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.gate.weight)

        # use fused kernel, currently only support 256 or 384 experts
        if self.topk_method == "noaux_tc" and self.n_routed_experts in [256, 384]:
            topk_weight, topk_idx, _ = torch_npu.npu_moe_gating_top_k(
                logits,
                k=self.top_k,
                bias=self.gate.e_score_correction_bias.float(),
                k_group=self.topk_group,
                group_count=self.n_group,
                group_select_mode=1,
                renorm=0,       # 0: softmax->topk; 1: topk->softmax
                norm_type=1,    # 0: softmax; 1: sigmoid
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

        # select top-k experts
        if self.topk_method == "greedy":
            topk_weight, topk_idx = torch.topk(
                scores, k=self.top_k, dim=-1, sorted=False
            )
        elif self.topk_method == "group_limited_greedy":
            group_scores = (
                scores.view(num_tokens, self.n_group, -1).max(dim=-1).values
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
                    num_tokens, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(num_tokens, -1)
            )  # [n, e]
            tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            topk_weight, topk_idx = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False
            )
        elif self.topk_method == "noaux_tc":
            assert not self.training
            scores_for_choice = scores.view(num_tokens, -1) + self.gate.e_score_correction_bias.unsqueeze(0)
            group_scores = (
                scores_for_choice.view(num_tokens, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
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
                    num_tokens, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(num_tokens, -1)
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

        # norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor # must multiply the scaling factor

        return topk_idx, topk_weight, None

    def set_mc2_kwargs(self):
        global_rank = dist.get_rank()
        mc2_group_name = self.comm_manager.get_group_name("moe_ep_group_mc2")
        self.dispatch_kwargs = {
                "x_active_mask": None,
                "expert_shard_type": 0,
                "shared_expert_rank_num": self.shared_expert_rank_num,
                "moe_expert_num": self.n_routed_experts,
                "global_bs": 0,
                "scales": self.experts.smooth_scale_1 if "a8" in self.gmm_quant_mode else None,
                "quant_mode": 2 if "a8" in self.gmm_quant_mode else 0,
                "group_ep": mc2_group_name,
                "ep_world_size": self.moe_ep_size,
                "ep_rank_id": global_rank // self.moe_tp_size,
                "group_tp": mc2_group_name,
                "tp_world_size": self.moe_tp_size,
                "tp_rank_id": global_rank % self.moe_tp_size,
                "comm_alg": "fullmesh_v2",
            }
        self.combine_kwargs = {
                "x_active_mask": None,
                "expert_shard_type": 0,
                "shared_expert_rank_num": self.shared_expert_rank_num,
                "moe_expert_num": self.n_routed_experts,
                "global_bs": 0,
                "group_ep": mc2_group_name,
                "ep_world_size": self.moe_ep_size,
                "ep_rank_id": global_rank // self.moe_tp_size,
                "group_tp": mc2_group_name,
                "tp_world_size": self.moe_tp_size,
                "tp_rank_id": global_rank % self.moe_tp_size
            }

    def forward(self, hidden_states, is_prefill=False, cur_topk_list=None, shared_expert_stream=None):
        topk_idx, topk_weight, _ = self._forward_gate(hidden_states.float())
        if self.force_eplb:
            topk_idx = cur_topk_list
        topk_idx = topk_idx.to(torch.int32)
        # we convert 2d to 3d, and then 3d back to 2d to adapte fusion pass rule
        hidden_states_3d = hidden_states.unsqueeze(1)
        hidden_states = hidden_states_3d.squeeze(1)

        enable_gegraph_and_multistream = self.enable_gegraph_and_multistream and not is_prefill
        enable_npugraphex_and_multistream = self.enable_npugraphex_and_multistream and not is_prefill
        merged_x = None
        pertoken_scale = None
        if self.n_shared_experts > 0:
            if enable_gegraph_and_multistream:
                hidden_states_share = None
                with npu_stream_switch(True, "shared_expert"):
                    hs = npu_wait_tensor(True, hidden_states, topk_idx)
                    if "a8" in self.mm_quant_mode:
                        merged_x, pertoken_scale = self.shared_experts.gate_up_proj(
                            hs.view(-1, hs.shape[-1]),
                            out_dtype=torch.int32,
                        )
                    else:
                        merged_x = self.shared_experts.gate_up_proj(hs.view(-1, hs.shape[-1]))
            elif enable_npugraphex_and_multistream:
                hidden_states_share = None
            else:
                hidden_states_share = self.shared_experts(hidden_states.view(-1, hidden_states.shape[-1]))
        else:
            hidden_states_share = None

        hidden_states_params = (hidden_states_share, merged_x, pertoken_scale)

        if self.moe_tp_size > 1:
            # MOE TP
            return self.moe_infer_tp(hidden_states, topk_idx, topk_weight, hidden_states_share, is_prefill)
        else:
            # MOE EP
            if is_prefill:
                return self.moe_infer_double_routing(hidden_states, topk_idx, topk_weight, hidden_states_share)
            else:
                return self.moe_infer_dispatch_combine(
                    hidden_states, topk_idx, topk_weight, hidden_states_params, shared_expert_stream
                )

    def shared_experts_down_proj(
        self, intermediate_hidden_states, hidden_states_ordered_by_experts, pertoken_scale=None
    ):
        with npu_stream_switch(True, "shared_expert"):
            intermediate_hidden_states = npu_wait_tensor(
                True, intermediate_hidden_states, hidden_states_ordered_by_experts
            )
            if "a8" in self.mm_quant_mode:
                hidden_states_share = self.shared_experts.down_proj(intermediate_hidden_states, pertoken_scale)
            else:
                hidden_states_share = self.shared_experts.down_proj(intermediate_hidden_states)
        return hidden_states_share

    def forward_gate_init_routing(self, hidden_states, cur_topk_list=None):
        # gate
        topk_idx, topk_weight, _ = self._forward_gate(hidden_states)
        if self.force_eplb:
            topk_idx = cur_topk_list
        topk_idx = topk_idx.to(torch.int32)

        # init_routing
        _, h = hidden_states.shape   # packed [T, H] (NOTE: helper has no callers)
        hidden_states = hidden_states.view(-1, h)
        expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = torch_npu.npu_moe_init_routing_v2(
            hidden_states,
            expert_idx=topk_idx,
            active_num=topk_idx.shape[0] * topk_idx.shape[1],
            scale=self.experts.smooth_scale_1 if "a8" in self.gmm_quant_mode else None,
            expert_num=self.num_experts,
            expert_tokens_num_type=1,  # 0: cumsum mode(not supported now); 1: count mode
            expert_tokens_num_flag=True, active_expert_range=[0, self.num_experts],
            quant_mode=1 if "a8" in self.gmm_quant_mode else -1
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

        if "a8" in self.gmm_quant_mode:
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
        num_tokens, h = hidden_states.shape   # packed [T, H] (NOTE: helper has no callers)
        # finalize-routing
        hidden_states = torch_npu.npu_moe_finalize_routing(
            gathered_tokens, skip1=hidden_states_share, skip2=None, bias=None,
            scales=topk_weight.to(gathered_tokens.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None, drop_pad_mode=2
        )

        hidden_states = hidden_states.view(num_tokens, h)
        return hidden_states

    def moe_infer_tp(self, x, topk_ids, topk_weight, hidden_states_share, is_prefill=False):
        num_tokens, h = x.shape   # packed [T, H]
        hidden_states = x.view(-1, h)
        routing_args = {
            "expert_idx": topk_ids,
            "active_num": num_tokens * self.top_k,
            "expert_num": self.num_experts,
            "expert_tokens_num_type": 1,  # 0: cumsum mode(not supported now); 1: count mode
            "expert_tokens_num_flag": True,
            "active_expert_range": [0, self.num_experts],
            "quant_mode": -1
        }
        if "a8" in self.gmm_quant_mode:
            routing_args.update({
                "scale": self.experts.smooth_scale_1,
                "expert_tokens_num_type": 2,
                "quant_mode": 1,
                "row_idx_type": 0,
                "drop_pad_mode": 0
            })
        expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = torch_npu.npu_moe_init_routing_v2(
            hidden_states, **routing_args
        )

        moe_args = {"group_list_type": 1}
        if "a8" in self.gmm_quant_mode:
            moe_args.update({
                "group_list_type": 2,
                "pertoken_scale": pertoken_scale
            })
        hidden_states_ordered_by_experts = self.experts(expanded_x, tokens_per_expert, **moe_args)
        enable_multi_streams = self.enable_npugraphex_and_multistream and not is_prefill
        wait_event(enable_multi_streams, self.npu_events, 1)
        hidden_states = torch_npu.npu_moe_finalize_routing(
            hidden_states_ordered_by_experts,
            skip1=hidden_states_share.view(-1, h), skip2=None,
            bias=None,
            scales=topk_weight.to(hidden_states_ordered_by_experts.dtype),
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=None, drop_pad_mode=2
        )
        if self.moe_tp_size > 1:
            dist.all_reduce(hidden_states, group=self.comm_manager.get_group("moe_tp_group"))
        hidden_states = hidden_states.view(num_tokens, self.hidden_dim)
        return hidden_states

    def dispatch_double_routing(self, tokens_per_expert, expanded_x, pertoken_scale):
        moe_ep_group = self.moe_ep_group
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
        gathered_tokens = expanded_x.new_empty(all_tokens.item(), expanded_x.shape[1])
        dist.all_to_all_single(gathered_tokens, expanded_x, output_splits, input_splits, group=moe_ep_group)

        gathered_pertoken_scale = None if pertoken_scale is None else\
                            pertoken_scale.new_empty(gathered_tokens.shape[0])
        if "a8" in self.gmm_quant_mode:
            dist.all_to_all_single(gathered_pertoken_scale, \
                                   pertoken_scale, output_splits, input_splits, group=moe_ep_group)
        return tokens_per_expert_group, gathered_tokens, gathered_pertoken_scale, input_splits, output_splits

    def moe_infer_double_routing(self, x, topk_ids, topk_weight, hidden_states_share):
        """
        pure ep strategy, for prefill stage mainly, only support eager mode
        """
        num_tokens, h = x.shape   # packed [T, H]
        x = x.view(-1, h)

        expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = torch_npu.npu_moe_init_routing_v2(
            x,
            expert_idx=topk_ids,
            active_num=topk_ids.shape[0] * topk_ids.shape[1],
            scale=self.experts.smooth_scale_1 if "a8" in self.gmm_quant_mode else None,
            expert_num=self.num_experts,
            expert_tokens_num_type=1,  # 0: cumsum mode(not supported now); 1: count mode
            expert_tokens_num_flag=True, active_expert_range=[0, self.num_experts],
            quant_mode=1 if "a8" in self.gmm_quant_mode else -1
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

        return hidden_states.view(num_tokens, h)

    def moe_infer_dispatch_combine(self, x, topk_ids, topk_weight, hidden_states_params, shared_expert_stream=None):
        """
        tp+ep mix strategy, for decode stage
        """
        if isinstance(hidden_states_params, tuple):
            hidden_states_share, merged_x, pertoken_scale = hidden_states_params
        else:
            hidden_states_share = hidden_states_params
            merged_x = None
            pertoken_scale = None
        enable_gegraph_and_multistream = merged_x is not None and self.n_shared_experts > 0

        num_tokens, h = x.shape   # packed [T, H]
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

        if enable_gegraph_and_multistream:
            with npu_stream_switch(True, "shared_expert"):
                merged_x = npu_wait_tensor(True, merged_x, expand_x)
                if "a8" in self.mm_quant_mode:
                    intermediate_hidden_states, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
                        merged_x,
                        weight_scale=self.shared_experts.gate_up_proj.weight_scale,
                        quant_scale=self.shared_experts.down_proj.smooth_scales,
                        quant_mode=1,
                        activate_left=True,
                        activation_scale=pertoken_scale,
                    )
                else:
                    intermediate_hidden_states = torch_npu.npu_swiglu(merged_x)

        # compute experts
        gmm_args = {
            "x": expand_x,
            "expert_tokens": expert_token_num,
            "group_list_type": 1,
        }

        if "a8" in self.gmm_quant_mode:
            gmm_args.update({"pertoken_scale": dynamic_scale})

        hidden_states_ordered_by_experts = self.experts(**gmm_args)

        if self.enable_npugraphex_and_multistream:
            hidden_states.record_stream(shared_expert_stream)
            self.npu_events[0].record()
            with torch.npu.stream(shared_expert_stream):
                self.npu_events[0].wait()
                hidden_states_share = self.shared_experts(hidden_states.view(-1, hidden_states.shape[-1]))
                self.npu_events[1].record()

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
        if enable_gegraph_and_multistream:
            combine_args["shared_expert_x"] = None
        elif not self.enable_npugraphex_and_multistream:
            combine_args["shared_expert_x"] = hidden_states_share
            
        hidden_states = torch_npu.npu_moe_distribute_combine_v2(**combine_args)

        if enable_gegraph_and_multistream:
            hidden_states_share = self.shared_experts_down_proj(
                intermediate_hidden_states,
                hidden_states_ordered_by_experts,
                pertoken_scale,
            )
        if self.enable_npugraphex_and_multistream:
            hidden_states_share.record_stream(torch.npu.current_stream())
            self.npu_events[1].wait()
        if enable_gegraph_and_multistream or self.enable_npugraphex_and_multistream:
            hidden_states = hidden_states + hidden_states_share

        hidden_states = hidden_states.view(num_tokens, self.hidden_dim)
        return hidden_states


# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->DeepseekV3
class DeepseekAttention(nn.Module):
    def __init__(self, config: DeepseekV3Config, infer_config: InferenceConfig, comm_manager: CommManager,
                 layer_idx: Optional[int] = None, prefix: Optional[str] = "", **kwargs):
        super().__init__()
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.attn_type = "FullAttention"
        self.mm_quant_mode = (
            config.quant_config.mm_quant_mode
            if config.quant_config is not None
            else "w16a16")
        self.kv_cache_quant_mode = config.quant_config.kv_cache_quant_mode \
            if config.quant_config is not None else "unquant"
        self.batch_size = self.infer_config.scheduler_config.batch_size
        self.batch_size_per_rank = self.infer_config.scheduler_config.batch_size_per_dp_rank
        self.attn_tp_size = self.infer_config.parallel_config.attn_tp_size
        self.attn_dp_size = self.infer_config.parallel_config.attn_dp_size
        self.o_proj_tp_size = self.infer_config.parallel_config.o_proj_tp_size
        self.cp_size = self.infer_config.parallel_config.cp_size
        self.moe_tp_size = self.infer_config.parallel_config.moe_tp_size
        self.moe_ep_size = self.infer_config.parallel_config.moe_ep_size
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
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim  # 64

        self.is_causal = True

        if self.q_lora_rank is None:
            self.q_proj = ColumnParallelLinear(self.hidden_size,
                                               self.num_heads * self.q_head_dim,
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
            self.q_a_layernorm = DeepseekV3RMSNorm(config.q_lora_rank)
            self.q_b_proj = ColumnParallelLinear(config.q_lora_rank,
                                                 self.num_heads * self.q_head_dim,
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
        self.kv_a_layernorm = DeepseekV3RMSNorm(config.kv_lora_rank)

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

        if self.o_proj_tp_size == 1:
            self.o_proj = RowParallelLinear(self.num_heads * self.v_head_dim,
                                            config.hidden_size,
                                            tp_size=self.attn_tp_size,
                                            tp_rank=self.comm_manager.get_rank("attn_tp_group")
                                            if self.attn_tp_size > 1 else 0,
                                            bias=False,
                                            input_is_parallel=True,
                                            quant_config=config.quant_config,
                                            prefix=f"{prefix}.o_proj")
        else:
            self.o_proj = RowParallelLinear(self.num_heads * self.v_head_dim,
                                            config.hidden_size,
                                            tp_size=self.o_proj_tp_size,
                                            tp_rank=self.comm_manager.get_rank("o_proj_tp_group")
                                            if self.o_proj_tp_size > 1 else 0,
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

        # Framework KV: cache and block_table are supplied by executor/core/kv_cache.
        # block_table is threaded into forward via forward_metadata.block_table[self.attn_type];
        # nope_cache / rope_cache storage is injected by KVCacheManager through cache_entries' tensor_setter.
        self.block_size = self.infer_config.scheduler_config.block_size
        self.nope_cache = torch.Tensor([])
        self.rope_cache = torch.Tensor([])
        self.prefill_cp_nope_cache = None
        self.prefill_cp_rope_cache = None
        dtype_nope = torch.int8 if self.kv_cache_quant_mode == "int8" else self.config.torch_dtype
        dtype_rope = self.config.torch_dtype
        self.cache_entries = [
            CacheEntry(
                cache_name="nope_cache",
                attn_type=self.attn_type,
                dim=self.config.kv_lora_rank,
                num_head=1,
                dtype=dtype_nope,
                needs_block=True,
                block_size=self.block_size,
                tensor_setter=lambda tensor, layer=self: setattr(layer, "nope_cache", tensor),
            ),
            CacheEntry(
                cache_name="rope_cache",
                attn_type=self.attn_type,
                dim=self.config.qk_rope_head_dim,
                num_head=1,
                dtype=dtype_rope,
                needs_block=True,
                block_size=self.block_size,
                tensor_setter=lambda tensor, layer=self: setattr(layer, "rope_cache", tensor),
            ),
        ]

        self.enable_weight_nz = self.infer_config.model_config.enable_weight_nz

        self.attn_func = self.apply_attention_fusion
        self.exe_mode = self.infer_config.model_config.exe_mode
        self.global_rank = kwargs.get("global_rank")
        self.enable_multi_streams = self.infer_config.model_config.custom_params.get("enable_multi_streams", False)
        self.enable_gegraph = self.exe_mode == "ge_graph"
        self.attn_tp_group = self.comm_manager.get_group("attn_tp_group")
        self.fa_ops_prefill = torch.ops.npu
        self.fa_ops_decode = torch.ops.npu
        if self.enable_gegraph:
            self.fa_ops_decode = tng.ops

    def mla_epilog(
        self,
        attn_output: torch.Tensor = None,
        absorb: bool = False,
        is_prefill: bool = False,
        prefill_o_proj_padded_tokens: Optional[int] = None,
    ):
        if absorb:
            # input shape [N//attn_tp_size, T(bs*q_len), D]
            # output shape [T(bs*q_len), N//attn_tp_size, D]
            attn_output = torch.matmul(
                attn_output,
                self.kv_b_proj_w_v
            ).transpose(0, 1)
            # Note: Considering the fusion rules of TBMM, attn_output shape requires a 3-dim shape, and
            # with appropriate tensor stride for the later 'view' operation if o_proj_tp_size > 1.
            # after reshape: [T(bs*q_len), 1, N//attn_tp_size*D]
            attn_output = attn_output.reshape(-1, 1, self.num_heads // self.attn_tp_size * self.v_head_dim)

        local_token_num = attn_output.shape[0]
        if is_prefill and self.o_proj_tp_size > 1 and prefill_o_proj_padded_tokens is not None:
            pad_tokens = prefill_o_proj_padded_tokens - local_token_num
            if pad_tokens > 0:
                pad_shape = (pad_tokens, *attn_output.shape[1:])
                attn_output = torch.cat([attn_output, attn_output.new_zeros(pad_shape)], dim=0)

        if self.o_proj_tp_size > 1:
            # after view: (bs*q_len, o_proj_tp_size, num_heads // o_proj_tp_size * v_head_dim)
            attn_output = attn_output.view(-1, self.o_proj_tp_size,
                                           self.num_heads // self.o_proj_tp_size * self.v_head_dim)
            # after transpose: (o_proj_tp_size, bs*q_len, num_heads // o_proj_tp_size * v_head_dim)
            # after view: (o_proj_tp_size * bs*q_len * num_heads // o_proj_tp_size * v_head_dim)
            attn_output = attn_output.transpose(1, 0).contiguous().view(-1)
            all2all_output = torch.empty_like(attn_output)
            # after all2all: (o_proj_tp_size * bs*q_len * num_heads // o_proj_tp_size * v_head_dim)
            dist.all_to_all_single(all2all_output, attn_output,
                                   group=self.comm_manager.get_group("o_proj_tp_group"))
            # after view: (o_proj_tp_size * bs*q_len, num_heads // o_proj_tp_size * v_head_dim)
            attn_output = all2all_output.view(-1, self.num_heads // self.o_proj_tp_size * self.v_head_dim)

        attn_output = self.o_proj(attn_output.reshape(attn_output.shape[0], -1))

        if self.o_proj_tp_size > 1:
            reduce_scatter_output = torch.empty((attn_output.size()[0] // self.o_proj_tp_size, attn_output.size()[1]),
                                                dtype=attn_output.dtype, device=attn_output.device)
            dist.reduce_scatter_tensor(reduce_scatter_output, attn_output,
                                       group=self.comm_manager.get_group("o_proj_tp_group"))
            attn_output = reduce_scatter_output

        if is_prefill and attn_output.shape[0] > local_token_num:
            attn_output = attn_output[:local_token_num]

        if self.attn_tp_size > 1:
            dist.all_reduce(attn_output, group=self.comm_manager.get_group("attn_tp_group"))

        return attn_output

    def get_prefill_o_proj_padded_tokens(self, hidden_states, is_prefill):
        if not is_prefill or self.o_proj_tp_size <= 1:
            return None
        local_token_num = hidden_states.shape[0]
        max_token_num = torch.tensor([local_token_num], dtype=torch.long, device=hidden_states.device)
        dist.all_reduce(
            max_token_num,
            op=dist.ReduceOp.MAX,
            group=self.comm_manager.get_group("o_proj_tp_group"),
        )
        return int(max_token_num.item())

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_len: torch.IntTensor = None,
        actual_seq_lengths_kv: torch.Tensor = None,
        actual_seq_lengths_q: torch.Tensor = None,
        cos_sin: torch.Tensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        is_prefill: bool = True,
        output_attentions: bool = False,
        slot_mapping: Optional[Dict[str, torch.Tensor]] = None,
        block_table: Optional[Dict[str, torch.Tensor]] = None,
        cp_metadata: Optional[PrefillCPMetaData] = None,
        attention_mask: Optional[torch.Tensor] = None,
        prefill_o_proj_padded_tokens: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        slot_mapping = slot_mapping[self.attn_type] if isinstance(slot_mapping, dict) else slot_mapping
        block_table = block_table[self.attn_type] if isinstance(block_table, dict) else block_table
        input_kwargs = {
            "hidden_states": hidden_states,
            "cos_sin": cos_sin,
            "kv_len": kv_len,
            "position_ids": position_ids,
            "actual_seq_lengths_kv": actual_seq_lengths_kv,
            "is_prefill": is_prefill,
            "slot_mapping": slot_mapping,
            "block_table": block_table,
            "cp_metadata": cp_metadata,
            "attention_mask": attention_mask,
            "prefill_o_proj_padded_tokens": prefill_o_proj_padded_tokens,
        }
        if is_prefill and cp_metadata is not None and cp_metadata.enabled:
            return self.forward_absorb_cp(**input_kwargs)
        else:
            input_kwargs.update({"actual_seq_lengths_q": actual_seq_lengths_q})
            return self.forward_absorb(**input_kwargs)

    def forward_absorb(
        self,
        hidden_states: torch.Tensor,
        kv_len: torch.IntTensor = None,
        actual_seq_lengths_kv: torch.Tensor = None,
        actual_seq_lengths_q: torch.Tensor = None,
        cos_sin: torch.Tensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        is_prefill: bool = True,
        slot_mapping: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        cp_metadata: Optional[PrefillCPMetaData] = None,
        attention_mask: Optional[torch.Tensor] = None,
        prefill_o_proj_padded_tokens: Optional[int] = None,
        **kwargs,
    ):
        query_states = self.prepare_qkv(
            hidden_states=hidden_states,
            cos_sin=cos_sin,
            kv_len=kv_len,
            position_ids=position_ids,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            actual_seq_lengths_q=actual_seq_lengths_q,
            is_prefill=is_prefill,
            slot_mapping=slot_mapping
        )
        q_nope, q_rope = query_states

        attn_output = self.attn_func(
            query_states=query_states,
            actual_seq_qlen=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            block_table=block_table,
            is_prefill=is_prefill,
            attention_mask=attention_mask
        )

        output = self.mla_epilog(
            attn_output,
            absorb=True,
            is_prefill=is_prefill,
            prefill_o_proj_padded_tokens=prefill_o_proj_padded_tokens,
        )

        return output

    def forward_absorb_cp(
        self,
        hidden_states: torch.Tensor,
        kv_len: torch.IntTensor = None,
        actual_seq_lengths_kv: torch.Tensor = None,
        cos_sin: torch.Tensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        is_prefill: bool = True,
        slot_mapping: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        cp_metadata: Optional[PrefillCPMetaData] = None,
        attention_mask: Optional[torch.Tensor] = None,
        prefill_o_proj_padded_tokens: Optional[int] = None,
        **kwargs,
    ):
        # Prefill:[1, B*S, H] Decode:[B, S, H]
        cp_compute_block_table = cp_metadata.global_block_table["FullAttention"]
        query_states = self.prepare_qkv(
            hidden_states=hidden_states,
            cos_sin=cos_sin,
            kv_len=kv_len,
            position_ids=position_ids,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            is_prefill=is_prefill,
            cp_metadata=cp_metadata,
            slot_mapping=slot_mapping,
            block_table=cp_compute_block_table
        )
        q_nope, q_rope = query_states

        q_nope_prev, q_nope_next = torch.split(
            q_nope,
            [cp_metadata.local_prev_token_num, cp_metadata.local_next_token_num],
            dim=0,
        )
        q_rope_prev, q_rope_next = torch.split(
            q_rope,
            [cp_metadata.local_prev_token_num, cp_metadata.local_next_token_num],
            dim=0,
        )
        query_states_prev = q_nope_prev, q_rope_prev
        query_states_next = q_nope_next, q_rope_next

        attn_output_prev = self.attn_func(
            query_states=query_states_prev,
            actual_seq_qlen=cp_metadata.actual_seq_q_prev.tolist(),
            actual_seq_lengths_kv=cp_metadata.kv_len_prev.tolist(),
            block_table=cp_compute_block_table,
            is_prefill=is_prefill,
            attention_mask=attention_mask,
            label="prev"
        )
        attn_output_next = self.attn_func(
            query_states=query_states_next,
            actual_seq_qlen=cp_metadata.actual_seq_q_next.tolist(),
            actual_seq_lengths_kv=cp_metadata.kv_len_next.tolist(),
            block_table=cp_compute_block_table,
            is_prefill=is_prefill,
            attention_mask=attention_mask,
            label="next"
        )

        attn_output = torch.cat([attn_output_prev, attn_output_next], dim=1)  # [T,N,D]
        # Drop references to the temporary full CP compute cache after FA consumes it.
        self.prefill_cp_nope_cache = None
        self.prefill_cp_rope_cache = None

        output = self.mla_epilog(
            attn_output,
            absorb=True,
            is_prefill=is_prefill,
            prefill_o_proj_padded_tokens=prefill_o_proj_padded_tokens,
        )

        return output

    def mlaprolog_prefill(
        self,
        hidden_states: torch.Tensor,
        cos_sin: torch.Tensor = None,
        slot_mapping: Optional[torch.Tensor] = None,
        cp_metadata: Optional[PrefillCPMetaData] = None,
        block_table: Optional[torch.Tensor] = None,
    ):
        num_tokens, _ = hidden_states.shape
        enable_cp = cp_metadata is not None and cp_metadata.enabled
        
        if enable_cp:
            # CP mode: create temporary PA buffer for local tokens
            cp_tmpkv_cache_num_block = (num_tokens + self.block_size) // self.block_size
            dtype = self.config.torch_dtype
            nope_cache = torch.zeros(
                (cp_tmpkv_cache_num_block, self.block_size, 1, self.config.kv_lora_rank),
                dtype=dtype,
                device=hidden_states.device,
            )
            rope_cache = torch.zeros(
                (cp_tmpkv_cache_num_block, self.block_size, 1, self.config.qk_rope_head_dim),
                dtype=dtype,
                device=hidden_states.device,
            )
            # Use continuous slot_mapping for temporary buffer
            mla_prolog_slot_mapping = torch.arange(num_tokens, dtype=torch.long, device=hidden_states.device)
        else:
            nope_cache = self.nope_cache
            rope_cache = self.rope_cache
            mla_prolog_slot_mapping = slot_mapping
        
        cos, sin = cos_sin

        cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)
        sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)
        hidden_states = hidden_states.view(num_tokens, -1)

        if enable_cp:
            cache_mode = "PA_BSND"
        else:
            cache_mode = "PA_NZ"

        mla_prolog_input_args = {
            "token_x": hidden_states,
            "weight_dq": self.q_a_proj.weight,
            "weight_uq_qr": self.q_b_proj.weight,
            "weight_uk": self.kv_b_proj_w_k,
            "weight_dkv_kr": self.kv_a_proj_with_mqa.weight,
            "rmsnorm_gamma_cq": self.q_a_layernorm.weight,
            "rmsnorm_gamma_ckv": self.kv_a_layernorm.weight,
            "rope_sin": sin.squeeze(1).squeeze(1),
            "rope_cos": cos.squeeze(1).squeeze(1),
            "kv_cache": nope_cache,
            "kr_cache": rope_cache,
            "cache_index": mla_prolog_slot_mapping.view(-1),
            "rmsnorm_epsilon_cq": self.q_a_layernorm.variance_epsilon,
            "rmsnorm_epsilon_ckv": self.kv_a_layernorm.variance_epsilon,
            "cache_mode": cache_mode,
            "query_norm_flag": True,
            "weight_quant_mode": 0
        }

        q_nope, q_pe, _, qr, _ = torch_npu.npu_mla_prolog_v3(
            **mla_prolog_input_args
        )

        if enable_cp:
            # Read from temporary PA buffer using continuous indices
            latent_cache = torch.cat([
                nope_cache.view(-1, nope_cache.shape[-1])[:num_tokens, :],
                rope_cache.view(-1, self.qk_rope_head_dim)[:num_tokens, :]], dim=-1)
        else:
            # Read from actual PA cache using slot_mapping
            latent_cache = torch.cat([
                nope_cache.view(-1, nope_cache.shape[-1])[slot_mapping, :],
                rope_cache.view(-1, self.qk_rope_head_dim)[slot_mapping, :]], dim=-1)

        if enable_cp:
            # CP prolog writes only local tokens first. Gather all CP shards,
            # restore the global padded order.
            kv_all = latent_cache.new_empty(
                [cp_metadata.local_token_num * cp_metadata.cp_size, latent_cache.shape[-1]]
            )
            dist.all_gather_into_tensor(
                kv_all,
                latent_cache.view(num_tokens, -1),
                group=self.comm_manager.get_group("cp_group"),
            )
            latent_cache = torch.index_select(kv_all, 0, cp_metadata.restore_indices)

            # CP prefill uses two PA cache spaces:
            # 1. full_*_cache: full KV restored from all CP ranks, used only
            #    by the current layer's prefill FA.
            # 2. self.nope_cache/self.rope_cache: this rank's assigned requests,
            #    kept for later decode.
            decode_nope_cache = self.nope_cache
            decode_rope_cache = self.rope_cache
            full_block_num = cp_metadata.global_block_table[self.attn_type].numel()
            global_slot_mapping = cp_metadata.global_slot_mapping[self.attn_type]
            decode_token_indices = cp_metadata.persistent_valid_indices
            persistent_slot_mapping = cp_metadata.persistent_slot_mapping[self.attn_type]
            has_decode_requests = decode_token_indices.numel() > 0

            k_nope = latent_cache[:, : self.config.kv_lora_rank]
            k_pe = latent_cache[:, self.config.kv_lora_rank:]

            block_num, block_size, kv_num, _ = decode_nope_cache.shape
            factor = 2 if self.kv_cache_quant_mode == "int8" else 1
            KV_CACHE_NZ_DIM = 16 * factor

            # Scatter ALL gathered tokens to full cache (for prefill FA)
            full_nope_cache = latent_cache.new_zeros(full_block_num, block_size, 1, self.config.kv_lora_rank)
            full_rope_cache = latent_cache.new_zeros(full_block_num, block_size, 1, self.config.qk_rope_head_dim)

            torch_npu.npu_scatter_pa_kv_cache(k_nope.contiguous().unsqueeze(1),
                                              k_pe.contiguous().unsqueeze(1),
                                              full_nope_cache.view(full_block_num,
                                                                   self.config.kv_lora_rank // KV_CACHE_NZ_DIM,
                                                                   block_size,
                                                                   KV_CACHE_NZ_DIM),
                                              full_rope_cache.view(full_block_num,
                                                                   self.config.qk_rope_head_dim // KV_CACHE_NZ_DIM,
                                                                   block_size,
                                                                   KV_CACHE_NZ_DIM),
                                              global_slot_mapping)

            # Scatter ONLY owned tokens to persistent cache (for decode)
            if has_decode_requests:
                decode_k_nope = torch.index_select(k_nope, 0, decode_token_indices)
                decode_k_pe = torch.index_select(k_pe, 0, decode_token_indices)
                decode_nope_dim = (kv_num * decode_nope_cache.shape[-1]) // KV_CACHE_NZ_DIM
                decode_rope_dim = (kv_num * decode_rope_cache.shape[-1]) // KV_CACHE_NZ_DIM
                torch_npu.npu_scatter_pa_kv_cache(
                    decode_k_nope.contiguous().unsqueeze(1),
                    decode_k_pe.contiguous().unsqueeze(1),
                    decode_nope_cache.view(block_num, decode_nope_dim,
                                           block_size, KV_CACHE_NZ_DIM),
                    decode_rope_cache.view(block_num, decode_rope_dim,
                                           block_size, KV_CACHE_NZ_DIM),
                    persistent_slot_mapping,
                )

            self.prefill_cp_nope_cache = full_nope_cache
            self.prefill_cp_rope_cache = full_rope_cache
            nope_cache = decode_nope_cache
            rope_cache = decode_rope_cache

        return q_nope, q_pe, qr, nope_cache, rope_cache

    def mlaprolog_decode(
        self,
        hidden_states: torch.Tensor,
        cos_sin: torch.Tensor = None,
        slot_mapping: Optional[torch.Tensor] = None,
    ):
        num_tokens, _ = hidden_states.shape   # packed [T, H]
        # framework-injected per-layer KV cache (cache_entries' tensor_setter)
        nope_cache = self.nope_cache
        rope_cache = self.rope_cache
        cos, sin = cos_sin
        cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)
        sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)
        hidden_states = hidden_states.view(num_tokens, -1)

        mla_prolog_input_args = {
            "token_x": hidden_states,
            "weight_dq": self.q_a_proj.weight,
            "weight_uq_qr": self.q_b_proj.weight,
            "weight_uk": self.kv_b_proj_w_k,
            "weight_dkv_kr": self.kv_a_proj_with_mqa.weight,
            "rmsnorm_gamma_cq": self.q_a_layernorm.weight,
            "rmsnorm_gamma_ckv": self.kv_a_layernorm.weight,
            "rope_sin": sin.squeeze(1).squeeze(1),
            "rope_cos": cos.squeeze(1).squeeze(1),
            "kv_cache": nope_cache,
            "kr_cache": rope_cache,
            "cache_index": slot_mapping.view(-1),
            "rmsnorm_epsilon_cq": self.q_a_layernorm.variance_epsilon,
            "rmsnorm_epsilon_ckv": self.kv_a_layernorm.variance_epsilon,
            "cache_mode": "PA_NZ",
            "query_norm_flag": True,
            "weight_quant_mode": 0
        }
        q_nope, q_pe, _, qr, _ = torch_npu.npu_mla_prolog_v3(
            **mla_prolog_input_args
        )

        return q_nope, q_pe, qr, nope_cache, rope_cache

    def prepare_qkv(
        self,
        hidden_states: torch.Tensor,
        cos_sin: torch.Tensor = None,
        kv_len: torch.IntTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        actual_seq_lengths_kv: Optional[torch.Tensor] = None,
        actual_seq_lengths_q: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
        slot_mapping: Optional[torch.Tensor] = None,
        cp_metadata: Optional[PrefillCPMetaData] = None,
        block_table: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if not is_prefill:
            q_nope, q_pe, qr, nope_cache, rope_cache = self.mlaprolog_decode(
                    hidden_states=hidden_states, cos_sin=cos_sin,
                    slot_mapping=slot_mapping)
        else:
            q_nope, q_pe, qr, nope_cache, rope_cache = self.mlaprolog_prefill(
                hidden_states=hidden_states, cos_sin=cos_sin,
                slot_mapping=slot_mapping,
                cp_metadata=cp_metadata,
                block_table=block_table)

        query_states = (q_nope.view(-1, self.num_heads_per_rank, self.kv_lora_rank), \
                    q_pe.view(-1, self.num_heads_per_rank, self.qk_rope_head_dim))

        return query_states

    def apply_attention_fusion(
        self,
        query_states,
        actual_seq_qlen: torch.Tensor = None,
        actual_seq_lengths_kv: torch.Tensor = None,
        block_table: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
        label: str = "",
    ):
        q_nope, q_pe = query_states

        if is_prefill and self.prefill_cp_nope_cache is not None:
            k_nope = self.prefill_cp_nope_cache
            k_pe = self.prefill_cp_rope_cache
        else:
            k_nope = self.nope_cache
            k_pe = self.rope_cache
        num_tokens, num_heads, _ = q_nope.shape

        q_nope = q_nope.contiguous().view(num_tokens, num_heads, -1)
        q_pe = q_pe.contiguous().view(num_tokens, num_heads, -1)

        block_num, block_size, kv_num, dim = k_nope.shape
        factor = 2 if self.kv_cache_quant_mode == "int8" else 1
        KV_CACHE_NZ_DIM = 16 * factor
        k_nope = k_nope.view(block_num, kv_num, self.kv_lora_rank // KV_CACHE_NZ_DIM, block_size, KV_CACHE_NZ_DIM)
        k_pe = k_pe.view(block_num, kv_num, self.qk_rope_head_dim // KV_CACHE_NZ_DIM, block_size, KV_CACHE_NZ_DIM)

        fa_input_kwargs = {
            "query": q_nope,
            "key": k_nope,
            "value": k_nope,
            "query_rope": q_pe,
            "key_rope": k_pe,
            "num_heads": self.num_heads_per_rank,
            "num_key_value_heads": self.num_key_value_heads_per_rank,
            "input_layout": "TND_NTD",
            "actual_seq_lengths": actual_seq_qlen,
            "actual_seq_lengths_kv": actual_seq_lengths_kv,
            "sparse_mode": 3,
            "atten_mask": attention_mask,
            "block_table": block_table,
            "block_size": self.block_size,
            "scale": self.softmax_scale
        }
        if is_prefill:
            attn_output, _ = self.fa_ops_prefill.npu_fused_infer_attention_score(**fa_input_kwargs)
        else:
            attn_output, _ = self.fa_ops_decode.npu_fused_infer_attention_score(**fa_input_kwargs)

        return attn_output


class DeepseekV3DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV3Config, infer_config: InferenceConfig, comm_manager: CommManager,
                 layer_idx: int, prefix: str, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.hidden_size = config.hidden_size

        self.self_attn = DeepseekAttention(
            config=config,
            infer_config=self.infer_config,
            comm_manager=self.comm_manager,
            layer_idx=layer_idx,
            prefix=f"{prefix}.self_attn",
            **kwargs
        )

        self.is_moe = config.n_routed_experts is not None and \
                layer_idx >= config.first_k_dense_replace and \
                layer_idx % config.moe_layer_freq == 0

        self.mlp = (
            DeepseekV3MoE(config, self.infer_config, self.comm_manager,
                          layer_idx=layer_idx, prefix=f"{prefix}.mlp", **kwargs)
            if self.is_moe
            else DeepseekV3DenseMLP(config, self.infer_config, self.comm_manager,
                                    prefix=f"{prefix}.mlp", **kwargs)
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
        actual_seq_lengths_kv: torch.Tensor,
        cos_sin: torch.Tensor,
        actual_seq_lengths_q: Optional[torch.Tensor] = None,
        past_residual: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        is_prefill: Optional[bool] = False,
        cur_topk_list: Optional[torch.Tensor] = None,
        slot_mapping: Optional[Dict[str, torch.Tensor]] = None,
        block_table: Optional[Dict[str, torch.Tensor]] = None,
        cp_metadata: Optional[PrefillCPMetaData] = None,
        attention_mask: Optional[torch.Tensor] = None,
        shared_expert_stream: Optional[torch.npu.Stream] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:
        hidden_states, residual = self.input_layernorm(hidden_states, past_residual)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            kv_len=kv_len,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            cos_sin=cos_sin,
            actual_seq_lengths_q=actual_seq_lengths_q,
            position_ids=position_ids,
            is_prefill=is_prefill,
            slot_mapping=slot_mapping,
            block_table=block_table,
            cp_metadata=cp_metadata,
            attention_mask=attention_mask
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if self.is_moe:
            hidden_states = self.mlp(
                hidden_states, is_prefill=is_prefill, cur_topk_list=cur_topk_list,
                shared_expert_stream=shared_expert_stream
            )
        else:
            hidden_states = self.mlp(hidden_states)

        outputs = (residual, hidden_states)
        return outputs


@add_start_docstrings(
    "The bare DeepseekV3 Model outputting raw hidden-states without any specific head on top.",
    DEEPSEEKV3_START_DOCSTRING,
)
class DeepseekV3Model(nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DeepseekV3DecoderLayer`]

    Args:
        config: DeepseekV3Config
    """

    def __init__(self, config: DeepseekV3Config, infer_config: InferenceConfig, comm_manager: CommManager,
                 prefix: str, **kwargs):
        super().__init__()
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.embed_tp_size = self.infer_config.parallel_config.embed_tp_size
        self.embed_dp_size = self.infer_config.parallel_config.embed_dp_size
        self.attn_tp_size = self.infer_config.parallel_config.attn_tp_size
        self.attn_dp_size = self.infer_config.parallel_config.attn_dp_size
        self.cp_size = self.infer_config.parallel_config.cp_size
        self.moe_ep_size = self.infer_config.parallel_config.moe_ep_size
        self.padding_idx = None
        self.vocab_size = config.vocab_size
        self.vocab_size_per_rank = self.vocab_size // self.embed_tp_size
        self.global_rank = kwargs.get("global_rank")
        self.enable_superkernel = self.infer_config.model_config.custom_params.get("enable_superkernel", False)
        self.enable_multi_streams = self.infer_config.model_config.custom_params.get("enable_multi_streams", False)
        self.enable_npugraph_ex = self.infer_config.model_config.exe_mode == "npugraph_ex"
        self.world_size = self.infer_config.parallel_config.world_size

        self.shared_expert_stream = None
        if self.enable_multi_streams and self.enable_npugraph_ex:
            self.shared_expert_stream = torch.npu.Stream()

        self.max_position_embeddings = config.max_position_embeddings

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            self.padding_idx,
            torch.bfloat16,
            tp_size=self.embed_tp_size,
            tp_rank=self.comm_manager.get_rank("embed_tp_group") if self.embed_tp_size > 1 else 0)
        self.layers = nn.ModuleList(
            [
                DeepseekV3DecoderLayer(config, self.infer_config, self.comm_manager,
                                       layer_idx, f"model.layers.{layer_idx}", **kwargs)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        _init_rope(self)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def prepare_inputs_for_layer(self, inputs_embeds, input_ids):
        seq_length = input_ids.shape[0]

        step = seq_length // self.attn_tp_size
        tp_rank = self.comm_manager.get_rank("attn_tp_group") % self.attn_tp_size
        end = step * (tp_rank + 1)

        inputs_embeds = inputs_embeds.view(seq_length, self.config.hidden_size)
        hidden_states = inputs_embeds[step * tp_rank: end]

        return hidden_states

    def calc_input_embeddings(self, input_ids, is_prefill):
        num_tokens = input_ids.shape[0]
        attn_dp_size = self.attn_dp_size
        if self.embed_tp_size > 1:
            embed_tp_group = self.comm_manager.get_group("embed_tp_group")
            if attn_dp_size > self.embed_dp_size:
                allgather_ratio = self.embed_tp_size // self.attn_tp_size
                if is_prefill:
                    local_num_tokens = num_tokens
                    max_num_tokens = torch.tensor([local_num_tokens], dtype=torch.long, device=input_ids.device)
                    dist.all_reduce(max_num_tokens, op=dist.ReduceOp.MAX, group=embed_tp_group)
                    max_num_tokens = int(max_num_tokens.item())
                    if local_num_tokens < max_num_tokens:
                        input_ids = F.pad(input_ids, (0, max_num_tokens - local_num_tokens), value=0)
                    all_input_ids = input_ids.new_empty(max_num_tokens * allgather_ratio)
                    dist.all_gather_into_tensor(all_input_ids, input_ids, group=embed_tp_group)
                else:
                    all_input_ids = input_ids.new_empty(num_tokens * allgather_ratio)
                    dist.all_gather_into_tensor(all_input_ids, input_ids, group=embed_tp_group)
                embed_input_ids = all_input_ids
            else:
                embed_input_ids = input_ids

            new_input_ids = embed_input_ids - (self.global_rank % self.embed_tp_size) * self.vocab_size_per_rank
            mask = (new_input_ids >= 0) & (new_input_ids < self.vocab_size_per_rank) # (bs, qlen)
            new_input_ids_per_rank = new_input_ids * mask
            inputs_embeds = self.embed_tokens(new_input_ids_per_rank) * mask.unsqueeze(-1)

            if attn_dp_size <= self.embed_dp_size:
                dist.all_reduce(inputs_embeds, group=embed_tp_group)
            else:
                scatter_tokens = max_num_tokens if is_prefill else num_tokens
                inputs_embeds_attn = inputs_embeds.new_empty(scatter_tokens, inputs_embeds.shape[-1])
                dist.reduce_scatter_tensor(inputs_embeds_attn, inputs_embeds, group=embed_tp_group)
                inputs_embeds = inputs_embeds_attn[:local_num_tokens] if is_prefill else inputs_embeds_attn
        else:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        return hidden_states

    def select_prefill_cp_local_inputs(self, input_ids, position_ids, slot_mapping, cp_metadata, is_prefill):
        if not (is_prefill and cp_metadata is not None and cp_metadata.enabled):
            return input_ids, position_ids, slot_mapping

        local_indices = cp_metadata.local_indices
        input_ids = torch.index_select(input_ids, 0, local_indices)
        position_ids = torch.index_select(position_ids, 0, local_indices)
        return input_ids, position_ids, None

    @add_start_docstrings_to_model_forward(DEEPSEEKV3_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor = None,
        forward_metadata: ForwardMetaData = None,
        cur_topk_list: torch.Tensor = None,
        **kwargs,
    ):
        is_prefill = forward_metadata.is_prefill
        cp_metadata = getattr(forward_metadata, "cp_metadata", None)
        is_npugraph_ex_decode = (
            not is_prefill and self.infer_config.model_config.exe_mode == "npugraph_ex"
        )
        slot_mapping = forward_metadata.slot_mapping
        block_table = forward_metadata.block_table
        attention_mask = forward_metadata.attention_mask
        if slot_mapping is not None:
            slot_mapping = slot_mapping["FullAttention"]
        if block_table is not None:
            block_table = block_table["FullAttention"]

        # Prefill: FA expects per-sequence format for actual_seq_lengths_kv
        # Decode: use per-request format for ge_graph, list format for npugraph_ex
        if is_prefill:
            fa_actual_seq_lengths_kv = forward_metadata.actual_seq_lengths_kv
        elif is_npugraph_ex_decode:
            fa_actual_seq_lengths_kv = forward_metadata.actual_seq_lengths_list_kv
        else:
            fa_actual_seq_lengths_kv = forward_metadata.actual_seq_lengths_kv
        fa_actual_seq_lengths_q = (
            forward_metadata.actual_seq_lengths_cu_list_q
            if is_npugraph_ex_decode
            else forward_metadata.actual_seq_lengths_cu_q
        )

        input_ids = input_ids.to(torch.int32).view(-1)
        position_ids = position_ids.view(-1)
        input_ids, position_ids, slot_mapping = self.select_prefill_cp_local_inputs(
            input_ids, position_ids, slot_mapping, cp_metadata, is_prefill
        )

        hidden_states = self.calc_input_embeddings(input_ids, is_prefill)

        cos_sin = self.rotary_emb(hidden_states, position_ids, forward_metadata.kv_len,
                                  self.max_position_embeddings)

        if is_prefill and self.attn_tp_size > 1 and self.moe_ep_size > 1:
            hidden_states = self.prepare_inputs_for_layer(hidden_states, input_ids)

        residual = None

        label = f'decode_layer'
        option = "stream-fusion=1" if self.enable_multi_streams else "option_xxx2"
        with superkernel_scope(self.enable_superkernel and not is_prefill, label, option):
            for decoder_layer in self.layers:
                residual, hidden_states = decoder_layer(
                    hidden_states,
                    forward_metadata.kv_len,
                    fa_actual_seq_lengths_kv,
                    cos_sin=cos_sin,
                    actual_seq_lengths_q=fa_actual_seq_lengths_q,
                    past_residual=residual,
                    position_ids=position_ids,
                    is_prefill=is_prefill,
                    cur_topk_list=cur_topk_list,
                    slot_mapping=slot_mapping,
                    block_table=block_table,
                    cp_metadata=cp_metadata,
                    attention_mask=attention_mask,
                    shared_expert_stream=getattr(self, 'shared_expert_stream', None),
                )

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class DeepseekV3ModelMTPLayer(DeepseekV3Model):
    def __init__(self, config: DeepseekV3Config, infer_config: InferenceConfig, comm_manager: CommManager,
                 layer_idx: int, prefix: str, **kwargs):
        super().__init__(config, infer_config, comm_manager, prefix=prefix, **kwargs)
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.layers = nn.ModuleDict(
            {
                str(self.mtp_start_layer_idx + i):
                DeepseekV3DecoderLayer(config,
                    infer_config,
                    comm_manager,
                    layer_idx,
                    f"model.layers.{self.mtp_start_layer_idx + i}",
                    **kwargs)
                for i in range(config.num_nextn_predict_layers)
        })

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_len: torch.IntTensor,
        actual_seq_lengths_kv: list,
        cos_sin: torch.Tensor,
        actual_seq_lengths_q: Optional[torch.Tensor] = None,
        past_residual: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        is_prefill: Optional[bool] = False,
        cur_topk_list: Optional[torch.Tensor] = None,
        cp_metadata: Optional[PrefillCPMetaData] = None,
        slot_mapping: Optional[Dict[str, torch.Tensor]] = None,
        block_table: Optional[Dict[str, torch.Tensor]] = None,
        mtp_layer_idx: Optional[int] = 0,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.layers[str(self.mtp_start_layer_idx + mtp_layer_idx)](
            hidden_states,
            kv_len,
            actual_seq_lengths_kv,
            cos_sin=cos_sin,
            actual_seq_lengths_q=actual_seq_lengths_q,
            past_residual=past_residual,
            position_ids=position_ids,
            is_prefill=is_prefill,
            cur_topk_list=cur_topk_list,
            cp_metadata=cp_metadata,
            slot_mapping=slot_mapping,
            block_table=block_table,
            attention_mask=attention_mask,
            shared_expert_stream=getattr(self, 'shared_expert_stream', None),
        )


class DeepseekV3ForCausalLM(nn.Module):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, infer_config: InferenceConfig, comm_manager: CommManager = None,
                 is_mtp=False, prefix: str = ""):
        super().__init__()
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.kv_cache_quant_mode = config.quant_config.kv_cache_quant_mode \
            if config.quant_config is not None else "unquant"
        self.get_parallel_settings()
        self.experts_per_rank = config.n_routed_experts // self.moe_ep_size
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.force_eplb = self.infer_config.model_config.force_eplb
        self.next_n = self.infer_config.model_config.next_n
        self.is_mtp = is_mtp
        self.enable_cache_compile = self.infer_config.model_config.enable_cache_compile

        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.rank_offset = int(os.getenv("RANK_OFFSET", "0"))
        self.global_rank = self.local_rank + self.rank_offset
        self.world_size = self.infer_config.parallel_config.world_size
        self.enable_weight_nz = self.infer_config.model_config.enable_weight_nz
        self.enable_mla_prolog = self.infer_config.model_config.custom_params.get("enable_mla_prolog", False)
        self.block_size = self.infer_config.scheduler_config.block_size
        kwargs = {"global_rank": self.global_rank}

        # Declare HCCL comm groups before constructing any submodule that calls
        # comm_manager.get_group(...). Per executor_design.md 6.2, the refactored
        # CommManager no longer auto-creates parallel groups; each model must register
        # the groups it uses (otherwise get_group raises KeyError).
        self.init_parallel_comm_group()

        mtp_layer_idx = config.num_hidden_layers # MTP is the last layer
        self.model = DeepseekV3ModelMTPLayer(config, self.infer_config, self.comm_manager,
                                             mtp_layer_idx, prefix, **kwargs) \
                    if is_mtp else DeepseekV3Model(config, self.infer_config, self.comm_manager, prefix, **kwargs)
        self.vocab_size = config.vocab_size
        self.lm_head = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            bias=False,
            tp_size=self.lmhead_tp_size,
            tp_rank=self.comm_manager.get_rank("lmhead_tp_group") if self.lmhead_tp_size > 1 else 0,
            quant_config=None,
            prefix="lm_head"
            )

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
        pc = self.infer_config.parallel_config
        self.embed_tp_size = pc.embed_tp_size
        self.attn_dp_size = pc.attn_dp_size
        self.attn_tp_size = pc.attn_tp_size
        self.o_proj_tp_size = pc.o_proj_tp_size
        self.cp_size = pc.cp_size
        self.moe_ep_size = pc.moe_ep_size
        self.moe_tp_size = pc.moe_tp_size
        self.lmhead_tp_size = pc.lmhead_tp_size
        self.moe_dp_size = pc.world_size // self.moe_tp_size if self.moe_tp_size else 1
        self.embed_dp_size = pc.embed_dp_size
        self.dense_tp_size = pc.dense_tp_size

    def init_parallel_comm_group(self):
        world_size = self.world_size
        self.comm_manager.register_group(
            name="attn_tp_group",
            group_num=world_size // self.attn_tp_size,
            group_size=self.attn_tp_size,
        )
        self.comm_manager.register_group(
            name="o_proj_tp_group",
            group_num=world_size // self.o_proj_tp_size,
            group_size=self.o_proj_tp_size,
        )
        self.comm_manager.register_group(
            name="embed_tp_group",
            group_num=world_size // self.embed_tp_size,
            group_size=self.embed_tp_size,
        )
        self.comm_manager.register_group(
            name="lmhead_tp_group",
            group_num=world_size // self.lmhead_tp_size,
            group_size=self.lmhead_tp_size,
        )
        if self.dense_tp_size > 1:
            self.comm_manager.register_group(
                name="dense_tp_group",
                group_num=world_size // self.dense_tp_size,
                group_size=self.dense_tp_size,
            )
        if self.moe_tp_size > 1:
            self.comm_manager.register_group(
                name="moe_tp_group",
                group_num=world_size // self.moe_tp_size,
                group_size=self.moe_tp_size,
            )
        if self.moe_ep_size > 1:
            moe_ep_group_num = world_size // self.moe_ep_size
            self.comm_manager.register_group(
                name="moe_ep_group",
                group_num=moe_ep_group_num,
                group_size=self.moe_ep_size,
                group_stride=moe_ep_group_num,
                return_name=True,
                hccl_buffer_size=calc_moe_hccl_buffer_size(self.infer_config, self.config),
            )
        if self.moe_ep_size > 1 and self.moe_tp_size == 1:
            moe_ep_group_num = world_size // self.moe_ep_size
            self.comm_manager.register_group(
                name="moe_ep_group_mc2",
                group_num=moe_ep_group_num,
                group_size=self.moe_ep_size,
                group_stride=moe_ep_group_num,
                return_name=True,
                allow_physical_reuse=False,
                hccl_buffer_size=calc_moe_hccl_buffer_size(self.infer_config, self.config, is_full_mesh_v2=True),
            )
        if self.cp_size > 1:
            self.comm_manager.register_group(
                name="cp_group",
                group_num=world_size // self.cp_size,
                group_size=self.cp_size,
            )

    def restore_prefill_cp_outputs(self, outputs, cp_metadata):
        if cp_metadata is None or not cp_metadata.enabled:
            return outputs
        gathered_outputs = outputs.new_empty([cp_metadata.local_token_num * cp_metadata.cp_size,
                                              outputs.shape[-1]])
        dist.all_gather_into_tensor(
            gathered_outputs,
            outputs.contiguous(),
            group=self.comm_manager.get_group("cp_group"),
        )
        restored_outputs = torch.index_select(gathered_outputs, 0, cp_metadata.restore_indices)
        restored_outputs = torch.index_select(restored_outputs, 0, cp_metadata.global_valid_indices)
        return restored_outputs

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
            # attention: SP + TP，moe：DP + EP
            if self.attn_tp_size > 1 and self.moe_ep_size > 1:
                new_outputs = torch.empty_like(outputs).repeat(self.attn_tp_size, 1)
                dist.all_gather_into_tensor(new_outputs, outputs,
                                            group=self.comm_manager.get_group("attn_tp_group"))
                outputs = new_outputs
            bs = actual_seq_lengths_q.numel()
            seq_index = actual_seq_lengths_q.to(dtype=torch.long, device=outputs.device) - 1
            outputs = torch.index_select(outputs.view(-1, hidden_size), 0, seq_index).view(bs, 1, hidden_size)
            q_len = 1 # prefill takes th last token
        else: # combine bs and q_len axes for lm_head
            bs = decode_batch_size if decode_batch_size is not None else num_tokens
            q_len = num_tokens // bs
            outputs = outputs.view(bs * q_len, 1, hidden_size)

        if (self.attn_dp_size == 1) or (self.lmhead_tp_size == 1):
            hidden_states = outputs
        else:
            # allgather: (bs / attn_dp, hidden_size) -> (bs, hidden_size)
            hidden_states = torch.empty_like(outputs).repeat(self.lmhead_tp_size, 1, 1)
            dist.all_gather_into_tensor(hidden_states, outputs,
                                        group=self.comm_manager.get_group("lmhead_tp_group"))

        logits = self.lm_head(hidden_states) # (lmhead_tp_size * bs / attn_dp, 1, vocab_size / lmhead_tp_size)
        if self.lmhead_tp_size > 1: # -> (bs / attn_dp, 1, vocab_size)
            if self.attn_dp_size == 1:
                new_logits = torch.empty_like(logits).repeat(self.lmhead_tp_size, 1, 1)
                dist.all_gather_into_tensor(new_logits, logits,
                                            group=self.comm_manager.get_group("lmhead_tp_group"))
            else:
                new_logits = torch.empty_like(logits).view(-1)
                dist.all_to_all_single(new_logits, logits.view(-1), \
                        group=self.comm_manager.get_group("lmhead_tp_group"))

            # transpose: (lmhead_tp_size * bs / attn_dp, vocab_size / lmhead_tp_size) -> (bs / attn_dp, vocab_size)
            new_logits = new_logits.reshape(
                self.lmhead_tp_size, bs * q_len, logits.shape[1], -1).permute(1, 2, 0, 3)
            logits = new_logits.reshape(bs * q_len, logits.shape[1], self.config.vocab_size)
        logits = logits.reshape(bs, q_len, -1).float()
        return logits

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor = None,
        forward_metadata: ForwardMetaData = None,
        **kwargs
    ):
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            forward_metadata=forward_metadata,
            cur_topk_list=kwargs.get("cur_topk_list", None),
        )

        outputs = self.restore_prefill_cp_outputs(
            outputs,
            forward_metadata.cp_metadata if forward_metadata.is_prefill else None,
        )
        prev_hidden_states = outputs

        logits = self.forward_lm_head(
            outputs,
            is_prefill=forward_metadata.is_prefill,
            actual_seq_lengths_q=forward_metadata.actual_seq_lengths_cu_q,
            decode_batch_size=forward_metadata.kv_len.shape[0] if not forward_metadata.is_prefill else None,
        )
        return logits, prev_hidden_states

    def prefill(
        self,
        **kwargs
    ):
        logits, prev_hidden_states = self.forward(
            **kwargs
        )
        return logits, prev_hidden_states

    def main_decode(
        self,
        **kwargs
    ):
        logits, prev_hidden_states = self.forward(
            **kwargs
        )
        return logits, prev_hidden_states

    def main_decode_mtp(
        self,
        **kwargs
    ):
        logits, prev_hidden_states = self.forward(
            **kwargs
        )
        return logits, prev_hidden_states

    def decode(
        self,
        **kwargs
    ):
        logits = self.main_decode(**kwargs)
        return logits

    def get_cache_info(self) -> ModelCacheInfo:
        layers = self.model.layers if not self.is_mtp else self.model.layers.values()
        layer_infos = []
        for layer_idx, layer in enumerate(layers):
            layer_infos.append(
                LayerCacheInfo(
                    layer_idx=layer_idx,
                    caches=list(layer.self_attn.cache_entries),
                )
            )

        return ModelCacheInfo(
            num_layers=len(layer_infos),
            layer_infos=layer_infos,
            is_mla_backend=True,
        )

    def check_model_settings(self):
        exe_mode = self.infer_config.model_config.exe_mode
        custom_params = self.infer_config.model_config.custom_params
        enable_cache_compile = self.infer_config.model_config.enable_cache_compile
        moe_chunk_max_len = custom_params.get("moe_chunk_max_len", 65536)
        enable_multi_streams = custom_params.get("enable_multi_streams", False)
        enable_superkernel = custom_params.get("enable_superkernel", False)
        next_n = self.infer_config.model_config.next_n

        if exe_mode not in ["ge_graph", "npugraph_ex", "eager"]:
            raise ValueError(f"{exe_mode=} does not supported!")
        if moe_chunk_max_len <= 0:
            raise ValueError(f"{moe_chunk_max_len=} should be a positive integer.")
        dynamo_feat = (enable_cache_compile or enable_multi_streams or enable_superkernel)
        if exe_mode == "eager" and dynamo_feat:
            raise ValueError(f"{exe_mode=} does not support cache compile, npugraph_ex, multi_streams or superkernel!")
        if next_n > 3:
            raise ValueError(f"{next_n=}, currently only support 0, 1, 2, 3")
        if self.kv_cache_quant_mode == "int8" and not self.enable_mla_prolog:
            raise ValueError("if kv_cache_quant_mode is C8, then enable_mla_prolog must be set to True.")

    def init_splited_kv_b_weight(self):
        def for_each_to_init_splited_k_b_weight(layer):
            try:
                data_getter = attrgetter("self_attn.kv_b_proj_w_k_data")
                data_tensor = data_getter(layer)
                layer.self_attn.kv_b_proj_w_k = nn.Parameter(data_tensor.contiguous(), requires_grad=False)
            except AttributeError:
                pass

        def for_each_to_init_splited_v_b_weight(layer):
            try:
                data_getter = attrgetter("self_attn.kv_b_proj_w_v_data")
                data_tensor = data_getter(layer)
                layer.self_attn.kv_b_proj_w_v = nn.Parameter(data_tensor.contiguous(), requires_grad=False)
            except AttributeError:
                pass

        def for_each_to_offload_kv_b_weight(layer):
            try:
                layer.self_attn.kv_b_proj.weight = None
            except AttributeError:
                pass

        layers = self.model.layers.values() if self.is_mtp else self.model.layers
        for layer in layers:
            for_each_to_init_splited_k_b_weight(layer)
            for_each_to_init_splited_v_b_weight(layer)
            for_each_to_offload_kv_b_weight(layer)
        gc.collect()

    def process_weights_after_loading(self):
        '''
        Doing weight transpose, format cast to nz, and scale type cast after loading weights from files.
        Folds the removed runner's _process_weight_after_loading + init_splited_kv_b_weight.
        '''
        self.init_splited_kv_b_weight()
        # map for scales need to cast to float when apply w8a8int8 quant method
        float_scales_map = [
            "gate_up_proj",
            "q_b_proj",
            "wq_b",
        ]
        # map for smooth scales need to cast to float when apply w8a8int8 quant method
        float_smooth_scales_map = [
            "down_proj"
        ]
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

            is_nz = False if ("mlp.gate" in module_name and "proj" not in module_name) else self.enable_weight_nz
            is_transpose = False if ("mlp.gate" in module_name and "proj" not in module_name) else True
            if isinstance(quant_method, QuantizeMethodBase):
                quant_method.process_weights_after_loading(module, is_nz=is_nz, is_transpose=is_transpose,\
                                                           scales_dtype=scales_dtype)
            # Dynamic quant for input_activation of first grouped matmul requires complete smooth scale.
            # Only applies to W8A8 MoE GMM; for W4 (CompressedTensorW4A16Int4MoEGMMMethod) this isinstance
            # check is False, so the all_gather block is skipped (harmless).
            if isinstance(quant_method, CompressedTensorW8A8Int8MoEGMMMethod):
                moe_ep_size = self.moe_ep_size
                if moe_ep_size > 1:
                    all_experts_smooth_scale = module.smooth_scale_1.data.new_empty(
                        module.smooth_scale_1.data.shape[0] * moe_ep_size, module.smooth_scale_1.data.shape[1])
                    dist.all_gather_into_tensor(all_experts_smooth_scale, module.smooth_scale_1.data,
                                                group=self.comm_manager.get_group("moe_ep_group"))
                    module.smooth_scale_1.data = all_experts_smooth_scale

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        repeat_loaded_weights_mapping = [] # (origin_name: repeat_loaded_name)

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
                    name = name.replace("weight_packed", "weight")

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

    def __init__(self, config: DeepseekV3Config, infer_config: InferenceConfig, comm_manager: CommManager = None):
        super().__init__(config, infer_config, comm_manager, is_mtp=True)
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.rank_offset = int(os.getenv("RANK_OFFSET", "0"))
        self.global_rank = self.local_rank + self.rank_offset
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.vocab_size_per_rank = self.vocab_size // self.embed_tp_size
        self.ignore_share_weight = False

        # reuse embed_tokens, lm_head, rotary_emb from main model
        self.embed_tokens = None
        self.lm_head = None
        self.rotary_emb = None

        self.shared_head_norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.enorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # prev_hidden_states and input_hidden_state feature fusion
        self.eh_proj = ReplicatedLinear(2 * config.hidden_size, config.hidden_size, bias=False)

    @add_start_docstrings_to_model_forward(DEEPSEEKV3_INPUTS_DOCSTRING)
    @override
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        forward_metadata: ForwardMetaData = None,
        prev_hidden_states: Optional[torch.Tensor] = None,
        **kwargs
    ):
        is_prefill = forward_metadata.is_prefill
        cp_metadata = getattr(forward_metadata, "cp_metadata", None)
        is_npugraph_ex_decode = (
            not is_prefill and self.infer_config.model_config.exe_mode == "npugraph_ex"
        )
        slot_mapping = forward_metadata.slot_mapping
        block_table = forward_metadata.block_table
        attention_mask = forward_metadata.attention_mask
        cur_topk_list = kwargs.get("cur_topk_list", None)
        if slot_mapping is not None:
            slot_mapping = slot_mapping["FullAttention"]
        if block_table is not None:
            block_table = block_table["FullAttention"]

        # Prefill: use per-request format
        # Decode: use per-request format for ge_graph, list format for npugraph_ex
        if is_prefill:
            fa_actual_seq_lengths_kv = forward_metadata.actual_seq_lengths_kv
        elif is_npugraph_ex_decode:
            fa_actual_seq_lengths_kv = forward_metadata.actual_seq_lengths_list_kv
        else:
            fa_actual_seq_lengths_kv = forward_metadata.actual_seq_lengths_kv
        fa_actual_seq_lengths_q = (
            forward_metadata.actual_seq_lengths_cu_list_q
            if is_npugraph_ex_decode
            else forward_metadata.actual_seq_lengths_cu_q
        )

        input_ids = input_ids.to(torch.int32).view(-1)
        position_ids = position_ids.view(-1)
        input_ids, position_ids, slot_mapping = self.model.select_prefill_cp_local_inputs(
            input_ids, position_ids, slot_mapping, cp_metadata, is_prefill
        )

        hidden_states = self.model.calc_input_embeddings(input_ids, is_prefill)

        if is_prefill and cp_metadata is not None and cp_metadata.enabled:
            prev_hidden_states = torch.index_select(prev_hidden_states, 0, cp_metadata.local_indices)

        cos_sin = self.rotary_emb(hidden_states, position_ids, forward_metadata.kv_len,
                                  self.max_position_embeddings)

        hidden_states = self.enorm(hidden_states)
        prev_hidden_states = self.hnorm(prev_hidden_states)
        hidden_states_eh = torch.cat([hidden_states, prev_hidden_states], dim=-1)
        hidden_states = self.eh_proj(hidden_states_eh)

        residual = None

        residual, hidden_states = self.model(
            hidden_states,
            forward_metadata.kv_len,
            fa_actual_seq_lengths_kv,
            actual_seq_lengths_q=fa_actual_seq_lengths_q,
            cos_sin=cos_sin,
            past_residual=residual,
            position_ids=position_ids,
            is_prefill=is_prefill,
            cur_topk_list=cur_topk_list,
            slot_mapping=slot_mapping,
            block_table=block_table,
            cp_metadata=cp_metadata,
            attention_mask=attention_mask
        )

        prev_hidden_states, _ = self.shared_head_norm(hidden_states, residual)

        outputs = self.restore_prefill_cp_outputs(
            prev_hidden_states,
            cp_metadata if is_prefill else None,
        )
        logits = self.forward_lm_head(
            outputs=outputs,
            is_prefill=is_prefill,
            actual_seq_lengths_q=forward_metadata.actual_seq_lengths_cu_q,
            decode_batch_size=forward_metadata.kv_len.shape[0] if not is_prefill else None,
        )

        return logits, prev_hidden_states

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
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
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
