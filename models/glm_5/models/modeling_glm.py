# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Copyright 2024 The GLM & ZhipuAI team and HuggingFace Inc. team. All rights reserved.
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

import os
import gc
from typing import Optional, Tuple, Dict, Iterable, Set

import torch
import torch.nn.functional as F

from torch import nn
import torch.distributed as dist

import custom_ops
import torch_npu

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from executor.utils import (
    override, weight_dequant, calc_moe_hccl_buffer_size)

from executor.model_loader.weight_utils import default_weight_loader
from executor.core.config import InferenceConfig, CommManager
from executor.core.kv_cache.cache_info import (
    CacheAllocator,
    CacheEntry,
    LayerCacheInfo,
    ModelCacheInfo,
    OffloadWorkspaceMemoryInfo,
)
from executor.utils import superkernel_scope
from executor.utils.forward_metadata import ForwardMetaData, PrefillCPMetaData
from executor.utils.stream_utils import (
    create_event, create_stream, npu_stream_switch,
    record_event, record_stream, wait_event)
from module.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding
    )
from module.fuse_moe_gmm import FusedMoEGMM
from module.quantization import QuantizeMethodBase
from module.quantization.compressed_tensors.compressed_tensors_moe_gmm import (
    CompressedTensorW8A8Int8MoEGMMMethod,
    CompressedTensorW4A8Int8MoEGMMMethod,
)
from module.quantization.mxfp8 import reshape_mx_scale
from .configuration_glm import GlmMoeDsaConfig
from .modules import one_hot, DeepseekV3RMSNorm, _init_rope
from .indexer import GlmMoeDsaIndexer
from .offload_cache import OffloadCache

logger = logging.get_logger(__name__)


class GlmFFN(nn.Module):
    def __init__(self, config, infer_config: InferenceConfig, comm_manager: CommManager, prefix, **kwargs):
        super().__init__()
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.platform_version = self.infer_config.model_config.platform_version.value
        self.mm_quant_mode = (
            config.quant_config.mm_quant_mode
            if config.quant_config is not None
            else "w16a16")
        self.ffn_mode = {
            "w8a8int8": self.forward_w8a8int8,
            "w8a8mxfloat8": self.forward_w8a8mxfloat8,
            "w8a8float8": self.forward_w8a8float8,
        }
        self.ffn_forward = self.ffn_mode[self.mm_quant_mode] if "w8a8" in self.mm_quant_mode else self.forward_normal

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

    def forward_w8a8float8(self, x):
        merged_x = self.gate_up_proj(x)
        intermediate_hidden_states = torch_npu.npu_swiglu(merged_x)
        intermediate_hidden_states, pertoken_scale = torch_npu.npu_dynamic_block_quant(
            intermediate_hidden_states, dst_type=torch.float8_e4m3fn)
        return self.down_proj(intermediate_hidden_states, pertoken_scale)

    def forward_w8a8mxfloat8(self, x):
        merged_x = self.gate_up_proj(x)
        intermediate_hidden_states, pertoken_scale = torch_npu.npu_swiglu_mx_quant(
                merged_x,
                group_index=None,
                dst_type=torch_npu.float8_e4m3fn,
                activate_left=True
            )
        return self.down_proj(intermediate_hidden_states, pertoken_scale)


class GlmMoeDsaMLP(GlmFFN):
    def __init__(self, config, infer_config: InferenceConfig, comm_manager: CommManager, prefix, **kwargs):
        super().__init__(config, infer_config, comm_manager, prefix, **kwargs)
        self.mm_quant_mode = (
            config.quant_config.mm_quant_mode
            if config.quant_config is not None
            else "w16a16")
        parallel_config = self.infer_config.parallel_config
        self.moe_ep_size = parallel_config.moe_ep_size
        self.moe_tp_size = parallel_config.moe_tp_size
        self.dense_tp_size = parallel_config.dense_tp_size
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

    def forward(self, x, is_prefill=False, prefill_dense_padded_tokens: Optional[int] = None):
        # input_DP + attention_TP + moe_EP
        if self.dense_tp_size > 1 and self.moe_ep_size > 1:
            token_num = x.shape[0]
            padded_token_num = prefill_dense_padded_tokens if prefill_dense_padded_tokens is not None else token_num
            if padded_token_num > token_num:
                pad_x = x.new_zeros((padded_token_num - token_num, self.hidden_size))
                x = torch.cat([x.view(token_num, self.hidden_size), pad_x], dim=0)
            else:
                x = x.view(token_num, self.hidden_size)
            x_output = torch.empty([padded_token_num * self.dense_tp_size, self.hidden_size],
                                   dtype=x.dtype, device=x.device)
            dist.all_gather_into_tensor(x_output, x, group=self.comm_manager.get_group("dense_tp_group"))
            x = x_output

        down_proj = self.ffn_forward(x)

        if self.dense_tp_size > 1 and self.moe_ep_size > 1:
            mlp_res = down_proj.new_empty(padded_token_num, down_proj.shape[-1])
            dist.reduce_scatter_tensor(mlp_res, down_proj, group=self.comm_manager.get_group("dense_tp_group"))
            down_proj = mlp_res[:token_num]
        elif self.dense_tp_size > 1 and self.moe_tp_size > 1:
            dist.all_reduce(down_proj, group=self.comm_manager.get_group("dense_tp_group"))

        return down_proj


class GlmMoeDsaSharedExpert(GlmFFN):
    def __init__(self, config, infer_config: InferenceConfig, comm_manager: CommManager,
                 is_moe_layer=False, prefix="", **kwargs):
        super().__init__(config, infer_config, comm_manager, prefix, **kwargs)
        self.mm_quant_mode = (
            config.quant_config.mm_quant_mode
            if config.quant_config is not None
            else "w16a16")
        self.moe_tp_size = self.infer_config.parallel_config.moe_tp_size
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

    def forward(self, x):
        down_proj = self.ffn_forward(x)
        return down_proj


class GlmMoeDsaMoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config, infer_config: InferenceConfig, comm_manager: CommManager, prefix, **kwargs):
        super().__init__()
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.layer_idx = kwargs.get("layer_idx")
        self.gmm_quant_mode = (
            config.quant_config.gmm_quant_mode
            if config.quant_config is not None
            else "w16a16")
        self.hidden_dim = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        model_config = self.infer_config.model_config
        parallel_config = self.infer_config.parallel_config
        custom_params = model_config.custom_params
        self.moe_tp_size = parallel_config.moe_tp_size
        self.moe_ep_size = parallel_config.moe_ep_size
        self.platform_version = model_config.platform_version.value
        self.exe_mode = model_config.exe_mode
        self.enable_multi_streams = custom_params.get("enable_multi_streams", False)
        self.enable_npugraph_ex = model_config.exe_mode == "npugraph_ex"
        self.perfect_eplb = model_config.force_eplb
        self.shared_expert_stream = kwargs.get("shared_expert_stream", None)
        self.npu_events = tuple(create_event(self.exe_mode, self.enable_multi_streams) for i in range(2))
        self.num_experts_per_tok = config.num_experts_per_tok
        # total experts num
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok

        self.intermediate_size_per_rank = self.intermediate_size // self.moe_tp_size
        self.shared_expert_rank_num = 0  # route and share on same card
        self.n_shared_experts = config.n_shared_experts
        self.n_routed_experts = config.n_routed_experts
        self.experts_per_rank = config.n_routed_experts // self.moe_ep_size
        self.experts = FusedMoEGMM(
            num_experts=config.n_routed_experts,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_size,
            # when W4A8 is enabled, gmm kernel needs an auxiliary matrix, it will be passed in as a bias
            bias=True if self.gmm_quant_mode == "w4a8int4" else False,
            quant_config=config.quant_config,
            tp_size=self.moe_tp_size,
            tp_rank=self.comm_manager.get_rank("moe_tp_group") if self.moe_tp_size > 1 else 0,
            ep_size=self.moe_ep_size,
            ep_rank=self.comm_manager.get_rank("moe_ep_group") if self.moe_ep_size > 1 else 0,
            prefix=f"{prefix}.experts",
        )

        self._init_gate(prefix)
        if config.n_shared_experts is not None:
            self.shared_experts = GlmMoeDsaSharedExpert(
                config,
                self.infer_config,
                self.comm_manager,
                is_moe_layer=True,
                prefix=f"{prefix}.shared_experts",
                **kwargs,
            )

        self.dispatch_kwargs = None
        self.combine_kwargs = None
        self.dispatch_quant_mode = {
            "w16a16": 0,
            "w8a8int8": 2,
            "w8a8float8": 3,
            "w8a8mxfloat8": 4,
        }

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

    @staticmethod
    def _token_shape(hidden_states):
        return hidden_states.shape[0], hidden_states.shape[-1]

    def _forward_gate(self, hidden_states):
        token_num, h = self._token_shape(hidden_states)
        # compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.gate.weight)

        # use fused kernel
        if self.topk_method == "noaux_tc":
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
                scores.view(token_num, self.n_group, -1).max(dim=-1).values
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
                    token_num, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(token_num, -1)
            )  # [n, e]
            tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            topk_weight, topk_idx = torch.topk(
                tmp_scores, k=self.top_k, dim=-1, sorted=False
            )
        elif self.topk_method == "noaux_tc":
            assert not self.training
            scores_for_choice = scores.view(token_num, -1) + self.gate.e_score_correction_bias.unsqueeze(0)
            group_scores = (
                scores_for_choice.view(token_num, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
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
                    token_num, self.n_group, self.n_routed_experts // self.n_group
                )
                .reshape(token_num, -1)
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
        topk_weight = topk_weight * self.routed_scaling_factor  # must multiply the scaling factor

        return topk_idx, topk_weight, None

    def set_mc2_kwargs(self):
        global_rank = dist.get_rank()
        mc2_group_name = self.comm_manager.get_group_name("moe_ep_group_mc2")
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
                "group_ep": mc2_group_name,
                "ep_world_size": self.moe_ep_size,
                "ep_rank_id": global_rank // self.moe_tp_size,
                "group_tp": mc2_group_name,
                "tp_world_size": self.moe_tp_size,
                "tp_rank_id": global_rank % self.moe_tp_size,
            }
        if quant_mode in (self.dispatch_quant_mode["w8a8float8"], self.dispatch_quant_mode["w8a8mxfloat8"]):
            self.dispatch_kwargs['y_dtype'] = torch.float8_e4m3fn
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
        if self.platform_version != "950":
            self.dispatch_kwargs["comm_alg"] = "fullmesh_v2"

    def forward(self, hidden_states, is_prefill=False, cur_topk_list=None, prefill_moe_global_chunks=None):
        if self.n_shared_experts > 0:
            hidden_states_share = self.forward_shared_expert(hidden_states, self.shared_expert_stream)
        else:
            hidden_states_share = None
        topk_idx, topk_weight, _ = self._forward_gate(hidden_states.float())
        if self.perfect_eplb:
            topk_idx = cur_topk_list
        topk_idx = topk_idx.to(torch.int32)

        if self.moe_tp_size > 1:
            # MOE TP
            return self.moe_infer_tp(hidden_states, topk_idx, topk_weight, hidden_states_share, is_prefill)
        else:
            # MOE EP
            if is_prefill:
                return self.moe_infer_double_routing(hidden_states, topk_idx, topk_weight, hidden_states_share,
                                                     prefill_moe_global_chunks)
            else:
                return self.moe_infer_dispatch_combine(hidden_states, topk_idx, topk_weight, hidden_states_share)

    def forward_shared_expert(self, hidden_states, shared_expert_stream=None):
        record_stream(self.enable_multi_streams, hidden_states, self.shared_expert_stream, exe_mode=self.exe_mode)
        record_event(self.enable_multi_streams, self.npu_events, 0, exe_mode=self.exe_mode)
        with npu_stream_switch(self.enable_multi_streams, shared_expert_stream, exe_mode=self.exe_mode):
            wait_event(self.enable_multi_streams, self.npu_events, 0, exe_mode=self.exe_mode)
            # shared_expert use multi streams
            hidden_states_share = self.shared_experts(hidden_states.view(-1, hidden_states.shape[-1]))
            record_event(self.enable_multi_streams, self.npu_events, 1, exe_mode=self.exe_mode)
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
            if "a8mxfloat" in self.gmm_quant_mode:
                # match GMM operator requirement (dim0, dim1)->(dim0, dim1//2, 2)
                gathered_pertoken_scale = reshape_mx_scale(gathered_pertoken_scale)
            if "a8float" in self.gmm_quant_mode:
                gathered_pertoken_scale = None
            gmm_args.update({"pertoken_scale": gathered_pertoken_scale})
        hidden_states_ordered_by_experts = self.experts(**gmm_args)
        # finalize-rerouting
        new_x = torch.index_select(hidden_states_ordered_by_experts, 0, gathered_ids_unsort.float().argsort().int())
        return new_x

    def forward_combine_double_routing(self, new_x, expanded_x, input_splits, output_splits):
        moe_ep_group = self.comm_manager.get_group("moe_ep_group")
        gathered_tokens = new_x.new_empty(*expanded_x.shape)
        dist.all_to_all_single(gathered_tokens, new_x, input_splits, output_splits, group=moe_ep_group)
        return gathered_tokens

    def moe_infer_tp(self, x, topk_ids, topk_weight, hidden_states_share, is_prefill):
        token_num, h = self._token_shape(x)
        hidden_states = x.view(-1, h)
        routing_args = {
            "expert_idx": topk_ids,
            "active_num": token_num * self.top_k,
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
        enable_multi_streams = self.enable_multi_streams
        wait_event(enable_multi_streams and hidden_states_share is not None, self.npu_events, 1, exe_mode=self.exe_mode)
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
        return hidden_states

    def dispatch_double_routing(self, tokens_per_expert, expanded_x, pertoken_scale):
        moe_ep_group = self.comm_manager.get_group("moe_ep_group")
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

        gathered_pertoken_scale = None
        if pertoken_scale is not None:
            # for w8a8float8, use bf16 routing now
            if "a8mxfloat" in self.gmm_quant_mode:
                pertoken_scale = pertoken_scale.view(torch.int8)
                gathered_pertoken_scale = pertoken_scale.new_empty(gathered_tokens.shape[0], pertoken_scale.shape[1])
            elif self.gmm_quant_mode == "w8a8int8":
                gathered_pertoken_scale = pertoken_scale.new_empty(gathered_tokens.shape[0])
        if "a16" not in self.gmm_quant_mode and "a8float" not in self.gmm_quant_mode:   # bf16 routing when fp8 mode now
            dist.all_to_all_single(gathered_pertoken_scale, \
                                   pertoken_scale, output_splits, input_splits, group=moe_ep_group)
        if "a8mxfloat" in self.gmm_quant_mode:
            gathered_pertoken_scale = gathered_pertoken_scale.view(torch.float8_e8m0fnu)
        return tokens_per_expert_group, gathered_tokens, gathered_pertoken_scale, input_splits, output_splits

    def moe_infer_double_routing(self, x, topk_ids, topk_weight, hidden_states_share, prefill_moe_global_chunks=None):
        """
        pure ep strategy, for prefill stage mainly, only support eager mode
        """
        token_num, h = self._token_shape(x)
        x = x.view(-1, h)
        hidden_states_list = []
        # EP-aligned chunk count, pre-computed once at the forward entry to avoid per-layer sync.
        global_num_chunks = prefill_moe_global_chunks

        # -1: non-quant; 1: dynamic quant; 0: static quant
        # 2: mxfp8 for dst_dtype e5m2; 3: mxfp8 for dst_dtype e4m3;
        # 4: fp8 for dst_dtype e5m2; 5: fp8 for dst_dtype e4m3
        # init_routing do not support fp8 now
        routing_args = {}
        if "a8mxfloat" in self.gmm_quant_mode:
            routing_args["quant_mode"] = 3
        elif "a8int" in self.gmm_quant_mode:
            routing_args["quant_mode"] = 1
        else:
            routing_args["quant_mode"] = -1  # bf16 routing, quant in MoeGmm if necessary

        enable_smooth_scale = "w8a8" in self.gmm_quant_mode and "float8" not in self.gmm_quant_mode
        # Run prefill MoE chunk by chunk to reduce routing/GMM peak memory.
        for hidden_states, topk_ids, topk_weight, hidden_states_share in zip(
                *self._split_tensors(x, topk_ids, topk_weight, hidden_states_share, global_num_chunks)):
            expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = torch_npu.npu_moe_init_routing_v2(
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

            gathered_tokens = self.forward_combine_double_routing(new_x, expanded_x, input_splits, output_splits)
            wait_event(self.enable_multi_streams and hidden_states_share is not None,
                       self.npu_events, 1, exe_mode=self.exe_mode)
            # finalize-routing
            hidden_states = torch_npu.npu_moe_finalize_routing(
                gathered_tokens, skip1=hidden_states_share, skip2=None, bias=None,
                scales=topk_weight.to(gathered_tokens.dtype),
                expanded_src_to_dst_row=expanded_row_idx,
                export_for_source_row=None, drop_pad_mode=2
            )

            if hidden_states.shape[0] > 0:
                hidden_states_list.append(hidden_states)

        if len(hidden_states_list) == 0:
            return x.new_empty(0, h)
        return torch.cat(hidden_states_list, dim=0) if len(hidden_states_list) > 1 else hidden_states_list[0]

    def _split_tensors(self, x, topk_ids, topk_weight, hidden_states_share, target_num_chunks):
        """Split prefill MoE along the token dimension to cap routing/GMM peak memory.

        Use torch.tensor_split to guarantee returning target_num_chunks chunks, automatically
        distributing tokens evenly or padding with empty chunks when token_num < target.
        Works correctly even when x.shape[0] == 0 (returns target empty chunks).
        """
        x_list = list(torch.tensor_split(x, target_num_chunks, dim=0))
        topk_ids_list = list(torch.tensor_split(topk_ids, target_num_chunks, dim=0))
        topk_weight_list = list(torch.tensor_split(topk_weight, target_num_chunks, dim=0))
        if hidden_states_share is None:
            hidden_states_share_list = [None] * target_num_chunks
        else:
            hidden_states_share_list = list(torch.tensor_split(hidden_states_share, target_num_chunks, dim=0))
        return x_list, topk_ids_list, topk_weight_list, hidden_states_share_list

    def moe_infer_dispatch_combine(self, x, topk_ids, topk_weight, hidden_states_share):
        """
        tp+ep mix strategy, for decode stage
        """
        _, h = self._token_shape(x)
        hidden_states = x.view(-1, h)
        self.set_mc2_kwargs()

        # moe dispatch
        dispatch_args = {
            "x": hidden_states,
            "expert_ids": topk_ids,  # [n*topk]
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
            if "mxfloat" in self.gmm_quant_mode:
                dynamic_scale = reshape_mx_scale(dynamic_scale)
            gmm_args.update({"pertoken_scale": dynamic_scale})

        hidden_states_ordered_by_experts = self.experts(**gmm_args)

        # is_prefill is always false in this branch
        wait_event(self.enable_multi_streams and hidden_states_share is not None,
                   self.npu_events, 1, exe_mode=self.exe_mode)

        # moe combine
        combine_args = {
            "expand_x": hidden_states_ordered_by_experts,
            "shared_expert_x": hidden_states_share,
            "expert_ids": topk_ids,
            "assist_info_for_combine": expand_idx,
            "expert_scales": topk_weight.to(torch.float32),  # [n*topk]
            "ep_send_counts": ep_recv_counts,
            "tp_send_counts": tp_recv_counts,
            **self.combine_kwargs
        }
        hidden_states = torch_npu.npu_moe_distribute_combine_v2(**combine_args)

        return hidden_states


class GlmMoeDsaAttention(nn.Module):
    def __init__(self, config: GlmMoeDsaConfig, infer_config: InferenceConfig, comm_manager: CommManager,
                 layer_idx: Optional[int] = None, prefix: Optional[str] = "", **kwargs):
        super().__init__()
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        model_config = self.infer_config.model_config
        parallel_config = self.infer_config.parallel_config
        scheduler_config = self.infer_config.scheduler_config
        custom_params = model_config.custom_params
        self.mm_quant_mode = (
            config.quant_config.mm_quant_mode
            if config.quant_config is not None
            else "w16a16")
        self.attn_tp_size = parallel_config.attn_tp_size
        self.o_proj_tp_size = parallel_config.o_proj_tp_size
        self.layer_idx = layer_idx
        self.is_mtp = False
        if layer_idx == config.num_hidden_layers:  # mtp model
            self.layer_idx = 0  # mtp model only has one layer of cache
            self.is_mtp = True
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
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim  # 256

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
            prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = DeepseekV3RMSNorm(config.kv_lora_rank)

        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=None,     # anti-quantize in load_weights for fp8
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
                                            tp_rank=self.comm_manager.get_rank("oproj_tp_group")
                                            if self.o_proj_tp_size > 1 else 0,
                                            bias=False,
                                            input_is_parallel=True,
                                            quant_config=config.quant_config,
                                            prefix=f"{prefix}.o_proj")

        self.softmax_scale = self.q_head_dim ** (-0.5)

        self.block_size = scheduler_config.block_size

        self.enable_weight_nz = model_config.enable_weight_nz

        self.indexer = GlmMoeDsaIndexer(
            self.config,
            self.infer_config,
            self.comm_manager,
            layer_idx,
            prefix=f"{prefix}.indexer",
            **kwargs,
        )
        self.attn_func = self.apply_attention_fusion
        self.select_block_count = config.index_topk
        self.exe_mode = model_config.exe_mode
        self.global_rank = kwargs.get("global_rank")
        self.enable_multi_streams = custom_params.get("enable_multi_streams", False)

        self.kv_cache_quant_mode = config.quant_config.kv_cache_quant_mode \
            if config.quant_config is not None else "unquant"

        self.li_cache_quant_mode = config.quant_config.li_cache_quant_mode \
            if config.quant_config is not None else "unquant"

        if self.kv_cache_quant_mode != "unquant":
            self.ckv_a_alpha = torch.nn.Parameter(torch.ones(1, dtype=torch.float), requires_grad=False)
            # empty tensor input for kr_cache
            # Its shape can be specified arbitrarily, but one of its dimensions must be 0
            self.fake_kr_cache = torch.empty((1, 2, 1, 0), dtype=torch.bfloat16, device="npu")

        self.enable_offload = custom_params.get("enable_offload", False)
        full_kv_allocator = CacheAllocator.SWAPPED_MEMORY if self.enable_offload else CacheAllocator.HBM
        self.index_topk = self.config.index_topk
        self.last_dim = self.kv_lora_rank + self.qk_rope_head_dim * 2 + 4 * 4 \
            if self.kv_cache_quant_mode != "unquant" else self.kv_lora_rank
        self.attn_type = "FullAttention"
        cache_dtype_map = {
            "int8": torch.int8,
            "float8": torch.float8_e4m3fn,
            "unquant": self.config.torch_dtype,
        }
        dtype_nope = cache_dtype_map.get(self.kv_cache_quant_mode, self.config.torch_dtype)
        dtype_rope = self.config.torch_dtype
        dtype_indexer = cache_dtype_map.get(self.li_cache_quant_mode, self.config.torch_dtype)
        self.nope_cache = torch.Tensor([])
        self.rope_cache = torch.Tensor([])
        self.indexer_key_cache = torch.Tensor([])
        self.indexer_key_scale_cache = torch.Tensor([])
        self.prefill_cp_nope_cache = None
        self.prefill_cp_rope_cache = None
        self.cache_entries = [
            CacheEntry(
                cache_name="nope_cache",
                attn_type=self.attn_type,
                dim=self.last_dim,
                num_head=1,
                dtype=dtype_nope,
                needs_block=True,
                block_size=self.block_size,
                tensor_setter=lambda tensor, layer=self: setattr(layer, "nope_cache", tensor),
                allocator=full_kv_allocator,
            ),
            CacheEntry(
                cache_name="indexer_key_cache",
                attn_type=self.attn_type,
                dim=self.config.index_head_dim,
                num_head=1,
                dtype=dtype_indexer,
                needs_block=True,
                block_size=self.block_size,
                tensor_setter=lambda tensor, layer=self: setattr(layer, "indexer_key_cache", tensor),
            ),
        ]
        if self.li_cache_quant_mode != "unquant":
            scale_dtype = torch.float16 if self.li_cache_quant_mode == "int8" else torch.float32
            self.cache_entries.append(
                CacheEntry(
                    cache_name="indexer_key_scale_cache",
                    attn_type=self.attn_type,
                    dim=1,
                    num_head=1,
                    dtype=scale_dtype,
                    needs_block=True,
                    block_size=self.block_size,
                    tensor_setter=lambda tensor, layer=self: setattr(layer, "indexer_key_scale_cache", tensor),
                )
            )
        # Only allocate rope_cache in non-quantized mode.
        # In quantized mode, rope data is merged into nope_cache and passed via k_nope input.
        if self.kv_cache_quant_mode == "unquant":
            self.cache_entries.append(
                CacheEntry(
                    cache_name="rope_cache",
                    attn_type=self.attn_type,
                    dim=self.qk_rope_head_dim,
                    num_head=1,
                    dtype=dtype_rope,
                    needs_block=True,
                    block_size=self.block_size,
                    tensor_setter=lambda tensor, layer=self: setattr(layer, "rope_cache", tensor),
                    allocator=full_kv_allocator,
                )
            )
        self.mlaprolog_quant_mode = {
            "w16a16": 0,
            "w8a8int8": 1,
            "w8a8float8": 4,
            "w8a8mxfloat8": 3,
        }

    def mla_epilog(
        self,
        attn_output: torch.Tensor = None,
        absorb: bool = False,
        is_prefill: bool = False,
        prefill_oproj_padded_tokens: Optional[int] = None,
    ):
        if absorb:
            # input shape [N//attn_tp_size, T, D]
            # output shape [T, N//attn_tp_size, D]
            attn_output = torch.matmul(
                attn_output,
                self.kv_b_proj_w_v
            ).transpose(0, 1)
            # Note: Considering the fusion rules of TBMM, attn_output shape requires a 3-dim shape, and
            # with appropriate tensor stride for the later 'view' operation if o_proj_tp_size > 1.
            # after reshape: [T, 1, N//attn_tp_size*D]
            attn_output = attn_output.reshape(-1, 1, self.num_heads // self.attn_tp_size * self.v_head_dim)

        local_token_num = attn_output.shape[0]
        if is_prefill and self.o_proj_tp_size > 1 and prefill_oproj_padded_tokens is not None:
            pad_tokens = prefill_oproj_padded_tokens - local_token_num
            if pad_tokens > 0:
                pad_shape = (pad_tokens, *attn_output.shape[1:])
                attn_output = torch.cat([attn_output, attn_output.new_zeros(pad_shape)], dim=0)

        if self.o_proj_tp_size > 1:
            # after view: (T, o_proj_tp_size, num_heads // o_proj_tp_size * v_head_dim)
            attn_output = attn_output.view(-1, self.o_proj_tp_size,
                                           self.num_heads // self.o_proj_tp_size * self.v_head_dim)
            # after transpose: (o_proj_tp_size, T, num_heads // o_proj_tp_size * v_head_dim)
            # after view: (o_proj_tp_size * T * num_heads // o_proj_tp_size * v_head_dim)
            attn_output = attn_output.transpose(1, 0).contiguous().view(-1)
            all2all_output = torch.empty_like(attn_output)
            # after all2all: (o_proj_tp_size * T * num_heads // o_proj_tp_size * v_head_dim)
            dist.all_to_all_single(all2all_output, attn_output,
                                   group=self.comm_manager.get_group("oproj_tp_group"))
            # after view: (o_proj_tp_size * T, num_heads // o_proj_tp_size * v_head_dim)
            attn_output = all2all_output.view(-1, self.num_heads // self.o_proj_tp_size * self.v_head_dim)

        attn_output = self.o_proj(attn_output.reshape(attn_output.shape[0], -1))

        if self.o_proj_tp_size > 1:
            reduce_scatter_output = torch.empty((attn_output.size()[0] // self.o_proj_tp_size, attn_output.size()[1]),
                                                dtype=attn_output.dtype, device=attn_output.device)
            dist.reduce_scatter_tensor(reduce_scatter_output, attn_output,
                                       group=self.comm_manager.get_group("oproj_tp_group"))
            attn_output = reduce_scatter_output

        if is_prefill and attn_output.shape[0] > local_token_num:
            attn_output = attn_output[:local_token_num]

        if self.attn_tp_size > 1:
            dist.all_reduce(attn_output, group=self.comm_manager.get_group("attn_tp_group"))

        return attn_output

    def forward(
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
        offload_cache: Optional[OffloadCache] = None,
        prefill_oproj_padded_tokens: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
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
            "offload_cache": offload_cache,
            "prefill_oproj_padded_tokens": prefill_oproj_padded_tokens,
        }
        input_kwargs.update({"actual_seq_lengths_q": actual_seq_lengths_q})
        if is_prefill and cp_metadata is not None and cp_metadata.enabled:
            return self.forward_absorb_cp(**input_kwargs)
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
        offload_cache: Optional[OffloadCache] = None,
        prefill_oproj_padded_tokens: Optional[int] = None,
        **kwargs,
    ):
        query_states, topk_indices = self.prepare_qkv(
            hidden_states=hidden_states,
            cos_sin=cos_sin,
            kv_len=kv_len,
            position_ids=position_ids,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            actual_seq_lengths_q=actual_seq_lengths_q,
            is_prefill=is_prefill,
            block_table=block_table,
            cp_metadata=cp_metadata,
            slot_mapping=slot_mapping,
            offload_cache=offload_cache
        )
        attn_output = self.attn_func(
            query_states=query_states,
            actual_seq_qlen=actual_seq_lengths_q,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            topk_indices=topk_indices,
            is_prefill=is_prefill,
            block_table=block_table,
            offload_cache=offload_cache
        )

        output = self.mla_epilog(
            attn_output,
            absorb=True,
            is_prefill=is_prefill,
            prefill_oproj_padded_tokens=prefill_oproj_padded_tokens,
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
        offload_cache: Optional[OffloadCache] = None,
        prefill_oproj_padded_tokens: Optional[int] = None,
        **kwargs,
    ):
        cp_compute_block_table = cp_metadata.global_block_table["FullAttention"]
        query_states, topk_indices = self.prepare_qkv(
            hidden_states=hidden_states,
            cos_sin=cos_sin,
            kv_len=kv_len,
            position_ids=position_ids,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            is_prefill=is_prefill,
            cp_metadata=cp_metadata,
            slot_mapping=slot_mapping,
            block_table=cp_compute_block_table,
            offload_cache=offload_cache,
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
        topk_indices_prev, topk_indices_next = topk_indices

        attn_output_prev = self.attn_func(
            query_states=(q_nope_prev, q_rope_prev),
            actual_seq_qlen=cp_metadata.actual_seq_q_prev,
            actual_seq_lengths_kv=cp_metadata.kv_len_prev,
            topk_indices=topk_indices_prev,
            is_prefill=is_prefill,
            block_table=cp_compute_block_table,
            offload_cache=offload_cache,
        )
        attn_output_next = self.attn_func(
            query_states=(q_nope_next, q_rope_next),
            actual_seq_qlen=cp_metadata.actual_seq_q_next,
            actual_seq_lengths_kv=cp_metadata.kv_len_next,
            topk_indices=topk_indices_next,
            is_prefill=is_prefill,
            block_table=cp_compute_block_table,
            offload_cache=offload_cache,
        )
        attn_output = torch.cat([attn_output_prev, attn_output_next], dim=1)
        self.prefill_cp_nope_cache = None
        self.prefill_cp_rope_cache = None

        output = self.mla_epilog(
            attn_output,
            absorb=True,
            is_prefill=is_prefill,
            prefill_oproj_padded_tokens=prefill_oproj_padded_tokens,
        )
        return output

    def apply_dynamic_quant(self, x):
        if self.mm_quant_mode == "w8a8float8":
            init_shape = x.shape
            x, x_scale = torch_npu.npu_dynamic_block_quant(x.view(-1, x.size(-1)),
                dst_type=torch.float8_e4m3fn,
                row_block_size=1,
                col_block_size=self.config.quant_config.weight_block_size[1])
            x, x_scale = x.view(*init_shape[:-1], -1), x_scale.view(*init_shape[:-1], -1)
        elif self.mm_quant_mode == "w8a8mxfloat8":
            x, x_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
        return x, x_scale

    def mlaprolog_prefill(
        self,
        hidden_states: torch.Tensor,
        cos_sin: torch.Tensor = None,
        slot_mapping: Optional[torch.Tensor] = None,
        cp_metadata: Optional[PrefillCPMetaData] = None,
        c8_input_dict: Optional[Dict] = None,
        offload_cache: Optional[OffloadCache] = None,
    ):
        token_num = hidden_states.shape[0]
        mla_prolog_slot_mapping = slot_mapping
        enable_cp = cp_metadata is not None and cp_metadata.enabled
        if enable_cp:
            cache_dtype_map = {
                "int8": torch.int8,
                "float8": torch.float8_e4m3fn,
                "unquant": self.config.torch_dtype,
            }
            dtype = cache_dtype_map.get(self.kv_cache_quant_mode, self.config.torch_dtype)
            nope_cache = torch.zeros(
                (1, token_num, 1, self.last_dim),
                dtype=dtype,
                device=hidden_states.device,
            )
            rope_cache = None if self.kv_cache_quant_mode != "unquant" else torch.zeros(
                (1, token_num, 1, self.qk_rope_head_dim),
                dtype=dtype,
                device=hidden_states.device,
            )
        else:
            nope_cache = self.nope_cache
            rope_cache = self.rope_cache
        if self.enable_offload and not enable_cp:
            nope_cache = offload_cache.temp_kv_cache[0]
            rope_cache = offload_cache.temp_kv_cache[1]

        cos, sin = cos_sin

        cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)
        sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)

        scale_name = 'scale' if self.mm_quant_mode == 'w8a8float8' else 'weight_scale'
        dequant_scale_w_uq_qr = getattr(self.q_b_proj, scale_name, None)
        dequant_scale_w_dq = getattr(self.q_a_proj, scale_name, None)
        dequant_scale_w_dkv_kr = getattr(self.kv_a_proj_with_mqa, scale_name, None)

        # 0: no quant applied; 1: only qb quantization is enabled
        # 3: mxfp8 quantization for weightDq, weightUqQr, and weightDkvKr;
        # 4: fp8 quantization for weightDq, weightUqQr, and weightDkvKr;
        weight_quant_mode = self.mlaprolog_quant_mode.get(self.mm_quant_mode, 0)
        if weight_quant_mode == 1:
            dequant_scale_w_uq_qr = self.q_b_proj.weight_scale.view(1, -1)

        mla_prolog_input_args = {
            "weight_dq": self.q_a_proj.weight,
            "weight_uq_qr": self.q_b_proj.weight,
            "weight_uk": self.kv_b_proj_w_k,     # bf16
            "weight_dkv_kr": self.kv_a_proj_with_mqa.weight,
            "rmsnorm_gamma_cq": self.q_a_layernorm.weight,
            "rmsnorm_gamma_ckv": self.kv_a_layernorm.weight,
            "kv_cache": nope_cache,
            "kr_cache": rope_cache,
            "dequant_scale_w_uq_qr": dequant_scale_w_uq_qr,
            "dequant_scale_w_dq": dequant_scale_w_dq,
            "dequant_scale_w_dkv_kr": dequant_scale_w_dkv_kr,
            "rmsnorm_epsilon_cq": self.q_a_layernorm.variance_epsilon,
            "rmsnorm_epsilon_ckv": self.kv_a_layernorm.variance_epsilon,
            "query_norm_flag": True,
            "weight_quant_mode": weight_quant_mode
        }

        if enable_cp:
            cache_mode = "BSND"
            hidden_states = hidden_states.view(1, token_num, -1)
            mla_prolog_input_args.update({
                "rope_sin": sin.view(1, token_num, -1),
                "rope_cos": cos.view(1, token_num, -1),
            })
        else:
            cache_mode = "PA_BSND"
            hidden_states = hidden_states.view(token_num, -1)
            mla_prolog_input_args.update({
                "cache_index": mla_prolog_slot_mapping.view(-1),
                "rope_sin": sin.squeeze(1).squeeze(1),
                "rope_cos": cos.squeeze(1).squeeze(1)
            })

        mla_prolog_input_args.update({
            "token_x": hidden_states,
            "cache_mode": cache_mode
        })

        if weight_quant_mode in (3, 4):
            hidden_states, dequant_scale_x = self.apply_dynamic_quant(hidden_states)
            dequant_scale_x = dequant_scale_x.flatten(-2)
            mla_prolog_input_args.update({
                "token_x": hidden_states,
                "dequant_scale_x": dequant_scale_x.view(dtype=torch.float8_e8m0fnu)
            })

        # When kv_cache_c8 is enabled, nope_cache, rope_cache, and scales
        # are concatenated and passed via the k_nope input.
        # In this case, an empty tensor must be passed for the k_rope input.
        if self.kv_cache_quant_mode != "unquant":
            mla_prolog_input_args.update({
                "kr_cache": self.fake_kr_cache,
                "kv_cache_quant_mode": 3,
                "query_quant_mode": 0,
                "ckvkr_repo_mode": 1,
                "quant_scale_repo_mode": 1,
                "tile_size": 128,
            })
        if self.kv_cache_quant_mode != "unquant" and "mxfloat" not in self.mm_quant_mode:
            mla_prolog_input_args.update({"k_nope_clip_alpha": self.ckv_a_alpha})

        q_nope, q_pe, dequant_scale_q_nope, qr, dequant_q_norm = torch_npu.npu_mla_prolog_v3(
            **mla_prolog_input_args
        )

        if self.kv_cache_quant_mode != "unquant":
            latent_cache = nope_cache.view(-1, nope_cache.shape[-1])
        else:
            latent_cache = torch.cat([
                nope_cache.view(-1, nope_cache.shape[-1]),
                rope_cache.view(-1, self.qk_rope_head_dim)], dim=-1)

        if enable_cp:
            kv_all = latent_cache.new_empty(
                [cp_metadata.local_token_num * cp_metadata.cp_size, latent_cache.shape[-1]]
            )
            dist.all_gather_into_tensor(
                kv_all,
                latent_cache.view(token_num, -1),
                group=self.comm_manager.get_group("cp_group"),
            )
            latent_cache = torch.index_select(kv_all, 0, cp_metadata.restore_indices)

            decode_nope_cache = offload_cache.temp_kv_cache[0] if self.enable_offload else self.nope_cache
            decode_rope_cache = offload_cache.temp_kv_cache[1] if self.enable_offload else self.rope_cache
            full_block_num = cp_metadata.global_block_table["FullAttention"].numel()
            full_slot_mapping = cp_metadata.global_slot_mapping["FullAttention"].view(-1, 1)
            decode_token_indices = cp_metadata.persistent_valid_indices
            decode_slot_mapping = cp_metadata.persistent_slot_mapping["FullAttention"].view(-1, 1)
            has_decode_requests = decode_token_indices.numel() > 0

            if self.kv_cache_quant_mode != "unquant":
                full_nope_cache = decode_nope_cache.new_zeros(
                    full_block_num,
                    self.block_size,
                    1,
                    latent_cache.shape[-1],
                )
                torch_npu.npu_scatter_nd_update_(
                    full_nope_cache.view(-1, latent_cache.shape[-1]),
                    full_slot_mapping,
                    latent_cache.view(-1, latent_cache.shape[-1]),
                )
                if has_decode_requests:
                    decode_latent_cache = torch.index_select(latent_cache, 0, decode_token_indices)
                    torch_npu.npu_scatter_nd_update_(
                        decode_nope_cache.view(-1, latent_cache.shape[-1]),
                        decode_slot_mapping,
                        decode_latent_cache.view(-1, latent_cache.shape[-1]),
                    )
                self.prefill_cp_nope_cache = full_nope_cache
                self.prefill_cp_rope_cache = self.fake_kr_cache
            else:
                full_nope_cache = decode_nope_cache.new_zeros(
                    full_block_num,
                    self.block_size,
                    1,
                    self.kv_lora_rank,
                )
                full_rope_cache = decode_rope_cache.new_zeros(
                    full_block_num,
                    self.block_size,
                    1,
                    self.qk_rope_head_dim,
                )
                k_nope = latent_cache.view(-1, latent_cache.shape[-1])[:, : self.kv_lora_rank]
                k_pe = latent_cache.view(-1, latent_cache.shape[-1])[:, self.kv_lora_rank:]
                torch_npu.npu_scatter_nd_update_(
                    full_nope_cache.view(-1, self.kv_lora_rank),
                    full_slot_mapping,
                    k_nope.view(-1, self.kv_lora_rank),
                )
                torch_npu.npu_scatter_nd_update_(
                    full_rope_cache.view(-1, self.qk_rope_head_dim),
                    full_slot_mapping,
                    k_pe.view(-1, self.qk_rope_head_dim),
                )
                if has_decode_requests:
                    decode_latent_cache = torch.index_select(latent_cache, 0, decode_token_indices)
                    decode_k_nope = decode_latent_cache[:, : self.kv_lora_rank]
                    decode_k_pe = decode_latent_cache[:, self.kv_lora_rank:]
                    torch_npu.npu_scatter_nd_update_(
                        decode_nope_cache.view(-1, self.kv_lora_rank),
                        decode_slot_mapping,
                        decode_k_nope.view(-1, self.kv_lora_rank),
                    )
                    torch_npu.npu_scatter_nd_update_(
                        decode_rope_cache.view(-1, self.qk_rope_head_dim),
                        decode_slot_mapping,
                        decode_k_pe.view(-1, self.qk_rope_head_dim),
                    )
                self.prefill_cp_nope_cache = full_nope_cache
                self.prefill_cp_rope_cache = full_rope_cache
            nope_cache = decode_nope_cache
            rope_cache = decode_rope_cache

        if weight_quant_mode in (1, 3, 4):
            if weight_quant_mode == 3:
                dequant_q_norm = dequant_q_norm.view(*dequant_q_norm.shape[:-1], -1, 2)
            c8_input_dict.update({'pertoken_scale': dequant_q_norm})

        return q_nope, q_pe, qr, nope_cache, rope_cache

    def mlaprolog_decode(
        self,
        hidden_states: torch.Tensor,
        cos_sin: torch.Tensor = None,
        slot_mapping: Optional[torch.Tensor] = None,
        c8_input_dict: Optional[Dict] = None,
    ):
        token_num = hidden_states.shape[0]
        nope_cache = self.nope_cache
        rope_cache = self.rope_cache
        cos, sin = cos_sin
        cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)
        sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)
        hidden_states = hidden_states.view(token_num, -1)

        scale_name = 'scale' if self.mm_quant_mode == 'w8a8float8' else 'weight_scale'
        dequant_scale_w_uq_qr = getattr(self.q_b_proj, scale_name, None)
        dequant_scale_w_dq = getattr(self.q_a_proj, scale_name, None)
        dequant_scale_w_dkv_kr = getattr(self.kv_a_proj_with_mqa, scale_name, None)

        # 0: no quant applied; 1: only qb quantization is enabled
        # 3: mxfp8 quantization for weightDq, weightUqQr, and weightDkvKr;
        # 4: fp8 quantization for weightDq, weightUqQr, and weightDkvKr;
        weight_quant_mode = self.mlaprolog_quant_mode.get(self.mm_quant_mode, 0)
        if weight_quant_mode == 1:
            dequant_scale_w_uq_qr = self.q_b_proj.weight_scale.view(1, -1)

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
            "dequant_scale_w_uq_qr": dequant_scale_w_uq_qr,
            "dequant_scale_w_dq": dequant_scale_w_dq,
            "dequant_scale_w_dkv_kr": dequant_scale_w_dkv_kr,
            "rmsnorm_epsilon_cq": self.q_a_layernorm.variance_epsilon,
            "rmsnorm_epsilon_ckv": self.kv_a_layernorm.variance_epsilon,
            "cache_mode": "PA_BSND",
            "query_norm_flag": True,
            "weight_quant_mode": weight_quant_mode
        }
        if weight_quant_mode in (3, 4):
            hidden_states, dequant_scale_x = self.apply_dynamic_quant(hidden_states)
            mla_prolog_input_args.update({
                "token_x": hidden_states,
                "dequant_scale_x": dequant_scale_x.flatten(1).view(dtype=torch.float8_e8m0fnu)
            })

        if self.kv_cache_quant_mode != "unquant":
            mla_prolog_input_args.update({
                "kr_cache": self.fake_kr_cache,
                "kv_cache_quant_mode": 3,
                "query_quant_mode": 0,
                "ckvkr_repo_mode": 1,
                "quant_scale_repo_mode": 1,
                "tile_size": 128,
            })
        if self.kv_cache_quant_mode != "unquant" and "mxfloat" not in self.mm_quant_mode:
            mla_prolog_input_args.update({"k_nope_clip_alpha": self.ckv_a_alpha})

        q_nope, q_pe, dequant_scale_q_nope, qr, dequant_q_norm = torch_npu.npu_mla_prolog_v3(
            **mla_prolog_input_args
        )

        if weight_quant_mode in (1, 3, 4):
            if weight_quant_mode == 3:
                dequant_q_norm = dequant_q_norm.view(*dequant_q_norm.shape[:-1], -1, 2)
            c8_input_dict.update({'pertoken_scale': dequant_q_norm})
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
        block_table: Optional[torch.Tensor] = None,
        cp_metadata: Optional[PrefillCPMetaData] = None,
        offload_cache: Optional[OffloadCache] = None,
        **kwargs,
    ):
        token_num = hidden_states.shape[0]

        c8_input_dict = {}
        if not is_prefill:
            q_nope, q_pe, qr, nope_cache, rope_cache = self.mlaprolog_decode(
                hidden_states=hidden_states, cos_sin=cos_sin,
                slot_mapping=slot_mapping,
                c8_input_dict=c8_input_dict
            )
        else:
            q_nope, q_pe, qr, nope_cache, rope_cache = self.mlaprolog_prefill(
                hidden_states=hidden_states, cos_sin=cos_sin,
                slot_mapping=slot_mapping,
                cp_metadata=cp_metadata,
                c8_input_dict=c8_input_dict,
                offload_cache=offload_cache
            )
            if self.enable_offload:
                offload_cache.d2h_event.record()
                with torch.npu.stream(offload_cache.d2h_stream):
                    offload_cache.d2h_event.wait()
                    nope_cache_dst = self.nope_cache.view(-1, 1, self.last_dim)[:nope_cache.numel() // self.last_dim]
                    nope_cache_src = nope_cache.view(-1, 1, self.last_dim)
                    if cp_metadata is not None and cp_metadata.enabled:
                        owner_slot_mapping = cp_metadata.persistent_slot_mapping["FullAttention"].view(-1).long()
                        if owner_slot_mapping.numel() > 0:
                            owner_nope_cache = torch.index_select(nope_cache_src, 0, owner_slot_mapping)
                            nope_cache_dst[owner_slot_mapping] = owner_nope_cache
                    else:
                        nope_cache_dst.copy_(nope_cache_src, non_blocking=True)
                    if self.kv_cache_quant_mode == "unquant":
                        rope_cache_dst = self.rope_cache.view(-1, 1, self.qk_rope_head_dim)[
                            :rope_cache.numel() // self.qk_rope_head_dim
                        ]
                        rope_cache_src = rope_cache.view(-1, 1, self.qk_rope_head_dim)
                        if cp_metadata is not None and cp_metadata.enabled:
                            if owner_slot_mapping.numel() > 0:
                                owner_rope_cache = torch.index_select(rope_cache_src, 0, owner_slot_mapping)
                                rope_cache_dst[owner_slot_mapping] = owner_rope_cache
                        else:
                            rope_cache_dst.copy_(rope_cache_src, non_blocking=True)
                    offload_cache.d2h_event.record()

        query_states = (
            q_nope.view(token_num, self.num_heads_per_rank, self.kv_lora_rank),
            q_pe.view(token_num, self.num_heads_per_rank, self.qk_rope_head_dim),
        )
        topk_indices = self.indexer(
            hidden_states, qr, actual_seq_lengths_kv, kv_len, cos_sin, position_ids,
            query_states, slot_mapping, block_table, actual_seq_lengths_q, cp_metadata,
            c8_input_dict, self.indexer_key_cache, self.indexer_key_scale_cache, is_prefill
        )

        return query_states, topk_indices

    def apply_attention_fusion(
        self,
        query_states, topk_indices,
        actual_seq_qlen: torch.Tensor = None,
        actual_seq_lengths_kv: torch.Tensor = None,
        is_prefill: bool = True,
        block_table: Optional[torch.Tensor] = None,
        offload_cache: Optional[OffloadCache] = None,
    ):
        # repeat k/v heads if n_kv_heads < n_heads
        q_nope, q_pe = query_states

        if is_prefill and self.prefill_cp_nope_cache is not None:
            k_nope = self.prefill_cp_nope_cache
            k_pe = self.prefill_cp_rope_cache
        elif self.enable_offload and is_prefill:
            k_nope = offload_cache.temp_kv_cache[0]
            k_pe = offload_cache.temp_kv_cache[1]
        else:
            k_nope = self.nope_cache
            k_pe = self.rope_cache

        token_num, num_heads, _ = q_nope.shape

        q_nope = q_nope.contiguous().view(token_num, num_heads, -1)
        q_pe = q_pe.contiguous().view(token_num, num_heads, -1)

        if self.enable_offload and not is_prefill:
            selection_k_rope = offload_cache.selected_key_values[self.layer_idx][1]
            selection_kv_cache = offload_cache.selected_key_values[self.layer_idx][0]
            full_kv_cache = k_nope.squeeze(2)
            full_k_rope = k_pe.squeeze(2) if self.kv_cache_quant_mode == "unquant" else offload_cache.empty_rope

            selection_kv_block_table = offload_cache.selection_kv_block_table[self.layer_idx][:token_num]
            selection_kv_block_status = offload_cache.selection_kv_block_status[self.layer_idx][:token_num]

            topk_indices = topk_indices.view(token_num, 1, self.index_topk)
            selection_kv_actual_seq = torch_npu.npu_gather_selection_kv_cache(
                selection_k_rope, selection_kv_cache,
                selection_kv_block_table, selection_kv_block_status,
                topk_indices, full_k_rope, full_kv_cache,
                block_table, actual_seq_lengths_kv.to(torch.int32),
                actual_seq_qlen.to(torch.int32), selection_topk_block_size=1)

            default_topk_indices = offload_cache.default_topk_indices[:token_num]
            sparse_indices = torch.where(default_topk_indices < selection_kv_actual_seq.unsqueeze(1), \
                                         default_topk_indices, -1)

            slc_fa_input_kwargs = {
                "query": q_nope,
                "sparse_indices": sparse_indices.view(-1, 1, self.index_topk),  # [T, 1, topk]
                "scale_value": self.softmax_scale,
                "actual_seq_lengths_query": torch.arange(1, token_num + 1, dtype=torch.int32, device="npu"),
                "actual_seq_lengths_kv": selection_kv_actual_seq,  # [T]
                "block_table": selection_kv_block_table,
                "sparse_block_size": 1,
                "layout_query": 'TND',
                "layout_kv": 'PA_BSND',
                "sparse_mode": 3,
            }

            slc_fa_input_kwargs.update({
                "key": selection_kv_cache.unsqueeze(2),
                "value": selection_kv_cache.unsqueeze(2),
            })
            if self.kv_cache_quant_mode == "unquant":
                k_pe = selection_k_rope.unsqueeze(2)

        else:
            slc_fa_input_kwargs = {
                "query": q_nope,
                "key": k_nope,
                "value": k_nope,
                "sparse_indices": topk_indices,
                "scale_value": self.softmax_scale,
                "actual_seq_lengths_query": actual_seq_qlen.to(torch.int32),
                "actual_seq_lengths_kv": actual_seq_lengths_kv.to(torch.int32),
                "block_table": block_table,
                "sparse_block_size": 1,
                "layout_query": 'TND',  # default is BSND
                "layout_kv": 'PA_BSND',
                "sparse_mode": 3,
            }

        if self.kv_cache_quant_mode != "unquant":
            q = torch.cat([q_nope, q_pe], dim=-1).contiguous()
            slc_fa_input_kwargs.update({
                "query": q,
                "key_quant_mode": 2,
                "value_quant_mode": 2,
                "attention_mode": 2,
                "quant_scale_repo_mode": 1,
                "tile_size": 128,
                "rope_head_dim": 64,
                "key_dequant_scale": None,
                "value_dequant_scale": None
            })

            slc_fa_fusion = torch_npu.npu_kv_quant_sparse_flash_attention(**slc_fa_input_kwargs)

        else:
            slc_fa_input_kwargs.update({
                "query_rope": q_pe,
                "key_rope": k_pe,
                "attention_mode": 2,
            })
            slc_fa_fusion, _, _ = torch_npu.npu_sparse_flash_attention(**slc_fa_input_kwargs)

        slc_fa_fusion = slc_fa_fusion.transpose(0, 1)
        return slc_fa_fusion


class GlmMoeDsaDecoderLayer(nn.Module):
    def __init__(self, config: GlmMoeDsaConfig, infer_config: InferenceConfig, comm_manager: CommManager,
                 layer_idx: int, prefix: str, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.hidden_size = config.hidden_size
        self.enable_offload = self.infer_config.model_config.custom_params.get("enable_offload", False)

        self.self_attn = GlmMoeDsaAttention(
            config=config,
            infer_config=self.infer_config,
            comm_manager=self.comm_manager,
            layer_idx=layer_idx,
            prefix=f"{prefix}.self_attn",
            **kwargs)

        self.is_moe = config.n_routed_experts is not None and \
                layer_idx >= config.first_k_dense_replace and \
                layer_idx % config.moe_layer_freq == 0

        self.mlp = (
            GlmMoeDsaMoE(
                config,
                self.infer_config,
                self.comm_manager,
                layer_idx=layer_idx,
                prefix=f"{prefix}.mlp",
                **kwargs,
            )
            if self.is_moe
            else GlmMoeDsaMLP(config, self.infer_config, self.comm_manager, prefix=f"{prefix}.mlp", **kwargs)
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
        slot_mapping: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        cp_metadata: Optional[PrefillCPMetaData] = None,
        offload_cache: Optional[OffloadCache] = None,
        prefill_oproj_padded_tokens: Optional[int] = None,
        prefill_dense_padded_tokens: Optional[int] = None,
        prefill_moe_global_chunks: Optional[int] = None,
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
            offload_cache=offload_cache,
            prefill_oproj_padded_tokens=prefill_oproj_padded_tokens,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if self.is_moe:
            hidden_states = self.mlp(hidden_states, is_prefill=is_prefill, cur_topk_list=cur_topk_list,
                                     prefill_moe_global_chunks=prefill_moe_global_chunks)
        else:
            hidden_states = self.mlp(
                hidden_states,
                is_prefill=is_prefill,
                prefill_dense_padded_tokens=prefill_dense_padded_tokens,
            )

        if is_prefill and self.enable_offload:
            offload_cache.d2h_event.wait()

        outputs = (residual, hidden_states)
        return outputs


class GlmMoeDsaPreTrainedModel(PreTrainedModel):
    config: GlmMoeDsaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GlmMoeDsaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = False  # flash-mla kernels need a bit more work in the way we enable them!
    _supports_sdpa = True
    _supports_flex_attn = False

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": GlmMoeDsaDecoderLayer,
        "attentions": GlmMoeDsaAttention,
    }
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.78.*"]
    # NOTE: FP8 quantization uses `_keep_in_fp32_modules` (not `_strict`) to decide which modules to NOT convert.
    # We must keep `indexer.weights_proj` as a plain Linear to match the checkpoint (no `weight_scale_inv`).
    _keep_in_fp32_modules = ["indexer.weights_proj"]
    _compatible_flash_implementations = ["kernels-community/flash-mla"]


class GlmMoeDsaModel(GlmMoeDsaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`GlmMoeDsaDecoderLayer`]

    Args:
        config: GlmMoeDsaConfig
    """

    def __init__(self, config: GlmMoeDsaConfig, infer_config: InferenceConfig, comm_manager: CommManager,
                 prefix: str, **kwargs):
        super().__init__(config)
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        model_config = self.infer_config.model_config
        parallel_config = self.infer_config.parallel_config
        scheduler_config = self.infer_config.scheduler_config
        custom_params = model_config.custom_params
        self.embed_tp_size = parallel_config.embed_tp_size
        self.embed_dp_size = parallel_config.embed_dp_size
        self.attn_tp_size = parallel_config.attn_tp_size
        self.attn_dp_size = parallel_config.attn_dp_size
        self.o_proj_tp_size = parallel_config.o_proj_tp_size
        self.dense_tp_size = parallel_config.dense_tp_size
        self.cp_size = parallel_config.cp_size
        self.moe_ep_size = parallel_config.moe_ep_size
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.vocab_size_per_rank = self.vocab_size // self.embed_tp_size
        self.global_rank = kwargs.get("global_rank")
        self.enable_superkernel = custom_params.get("enable_superkernel", False)
        self.enable_multi_streams = custom_params.get("enable_multi_streams", False)
        self.world_size = parallel_config.world_size

        self.pa_max_length = custom_params.get(
            "pa_max_length",
            self.infer_config.data_config.input_truncated_len
            + scheduler_config.max_new_tokens * (model_config.next_n + 1)
            + model_config.next_n,
        )
        self.max_position_embeddings = custom_params.get("max_position_embeddings", self.pa_max_length)

        is_mtp = kwargs.get("is_mtp")
        if not is_mtp:
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                None,           # tentative, emb tp happened: weight.size(0) < pad_token_id
                torch.bfloat16,
                tp_size=self.embed_tp_size,
                tp_rank=self.comm_manager.get_rank("embed_tp_group") if self.embed_tp_size > 1 else 0)
        else:
            self.embed_tokens = None

        self.layers = nn.ModuleList(
            [
                GlmMoeDsaDecoderLayer(
                    config,
                    self.infer_config,
                    self.comm_manager,
                    layer_idx,
                    prefix=f"model.layers.{layer_idx}",
                    **kwargs)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()
        _init_rope(self)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

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
                else:
                    all_input_ids = input_ids.new_empty(num_tokens * allgather_ratio)
                dist.all_gather_into_tensor(all_input_ids, input_ids, group=embed_tp_group)
                embed_input_ids = all_input_ids
            else:
                embed_input_ids = input_ids

            new_input_ids = embed_input_ids - (self.global_rank % self.embed_tp_size) * self.vocab_size_per_rank
            mask = (new_input_ids >= 0) & (new_input_ids < self.vocab_size_per_rank)  # [T]
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

    def get_prefill_oproj_padded_tokens(self, hidden_states):
        if self.o_proj_tp_size == 1:
            return None
        local_token_num = hidden_states.shape[0]
        max_token_num = torch.tensor([local_token_num], dtype=torch.long, device=hidden_states.device)
        dist.all_reduce(max_token_num, op=dist.ReduceOp.MAX, group=self.comm_manager.get_group("oproj_tp_group"))
        return int(max_token_num.item())

    def get_prefill_dense_padded_tokens(self, hidden_states):
        if self.dense_tp_size == 1 or self.moe_ep_size == 1:
            return None
        local_token_num = hidden_states.shape[0]
        max_token_num = torch.tensor([local_token_num], dtype=torch.long, device=hidden_states.device)
        dist.all_reduce(max_token_num, op=dist.ReduceOp.MAX, group=self.comm_manager.get_group("dense_tp_group"))
        return int(max_token_num.item())

    def get_prefill_moe_global_chunks(self, hidden_states):
        """Pre-compute the global MoE chunk count once at the forward entry.

        All MoE layers share the same token count within a forward pass, so computing the
        EP-group-aligned chunk count here (a single all_reduce + device-to-host sync) avoids
        repeating that sync in every MoE layer.
        """
        moe_chunk_max_len = self.infer_config.model_config.custom_params.get("moe_chunk_max_len", 65536)
        local_token_num = hidden_states.shape[0]
        local_num_chunks = (local_token_num + moe_chunk_max_len - 1) // moe_chunk_max_len
        if self.moe_ep_size == 1:
            return local_num_chunks
        max_num_chunks = torch.tensor([local_num_chunks], dtype=torch.int32, device=hidden_states.device)
        dist.all_reduce(max_num_chunks, op=dist.ReduceOp.MAX, group=self.comm_manager.get_group("moe_ep_group"))
        return int(max_num_chunks.item())

    def select_prefill_cp_local_inputs(self, input_ids, position_ids, slot_mapping, cp_metadata, is_prefill):
        if not (is_prefill and cp_metadata is not None and cp_metadata.enabled):
            return input_ids, position_ids, slot_mapping

        local_indices = cp_metadata.local_indices
        input_ids = torch.index_select(input_ids, 0, local_indices)
        position_ids = torch.index_select(position_ids, 0, local_indices)
        return input_ids, position_ids, None

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        forward_metadata: Optional[ForwardMetaData] = None,
        cur_topk_list: Optional[torch.Tensor] = None,
        offload_cache: Optional[OffloadCache] = None,
        **kwargs,
    ):
        is_prefill = forward_metadata.is_prefill
        kv_len = forward_metadata.kv_len
        actual_seq_lengths_kv = forward_metadata.actual_seq_lengths_kv
        actual_seq_lengths_q = forward_metadata.actual_seq_lengths_cu_q
        slot_mapping = forward_metadata.slot_mapping
        block_table = forward_metadata.block_table
        cp_metadata = getattr(forward_metadata, "cp_metadata", None)
        if slot_mapping is not None:
            slot_mapping = slot_mapping["FullAttention"]
        if block_table is not None:
            block_table = block_table["FullAttention"]

        input_ids, position_ids, slot_mapping = self.select_prefill_cp_local_inputs(
            input_ids, position_ids, slot_mapping, cp_metadata, is_prefill
        )
        input_ids = input_ids.to(torch.int32)
        inputs_embeds = self.calc_input_embeddings(input_ids, is_prefill)
        hidden_states = inputs_embeds

        cos_sin = self.rotary_emb(hidden_states, position_ids, kv_len, self.max_position_embeddings)

        prefill_oproj_padded_tokens = None
        prefill_dense_padded_tokens = None
        prefill_moe_global_chunks = None
        if is_prefill:
            prefill_oproj_padded_tokens = self.get_prefill_oproj_padded_tokens(hidden_states)
            prefill_dense_padded_tokens = self.get_prefill_dense_padded_tokens(hidden_states)
            prefill_moe_global_chunks = self.get_prefill_moe_global_chunks(hidden_states)
        residual = None

        label = f'decode_layer'
        if self.enable_multi_streams:
            option = "stream-fusion=1"
        else:
            option = "option_xxx2"
        with superkernel_scope(self.enable_superkernel and not is_prefill, label, option):
            for decoder_layer in self.layers:
                residual, hidden_states = decoder_layer(
                    hidden_states,
                    kv_len,
                    actual_seq_lengths_kv,
                    cos_sin=cos_sin,
                    actual_seq_lengths_q=actual_seq_lengths_q,
                    past_residual=residual,
                    position_ids=position_ids,
                    is_prefill=is_prefill,
                    cur_topk_list=cur_topk_list,
                    slot_mapping=slot_mapping,
                    block_table=block_table,
                    cp_metadata=cp_metadata,
                    offload_cache=offload_cache,
                    prefill_oproj_padded_tokens=prefill_oproj_padded_tokens,
                    prefill_dense_padded_tokens=prefill_dense_padded_tokens,
                    prefill_moe_global_chunks=prefill_moe_global_chunks,
                )

        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class GlmMoeDsaModelMTPLayer(GlmMoeDsaModel):
    def __init__(self, config: GlmMoeDsaConfig, infer_config: InferenceConfig, comm_manager: CommManager,
                 layer_idx: int, prefix: str, **kwargs):
        super().__init__(config, infer_config, comm_manager, prefix=prefix, **kwargs)
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.layers = nn.ModuleDict(
            {
                str(self.mtp_start_layer_idx + i):
                GlmMoeDsaDecoderLayer(
                    config,
                    self.infer_config,
                    self.comm_manager,
                    layer_idx,
                    prefix=f"model.layers.{self.mtp_start_layer_idx + i}",
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
        slot_mapping: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        mtp_layer_idx: Optional[int] = 0,
        cp_metadata: Optional[PrefillCPMetaData] = None,
        offload_cache: Optional[OffloadCache] = None,
        **kwargs,
    ) -> torch.Tensor:
        prefill_oproj_padded_tokens = None
        prefill_dense_padded_tokens = None
        prefill_moe_global_chunks = None
        if is_prefill:
            prefill_oproj_padded_tokens = self.get_prefill_oproj_padded_tokens(hidden_states)
            prefill_dense_padded_tokens = self.get_prefill_dense_padded_tokens(hidden_states)
            prefill_moe_global_chunks = self.get_prefill_moe_global_chunks(hidden_states)
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
            slot_mapping=slot_mapping,
            block_table=block_table,
            cp_metadata=cp_metadata,
            offload_cache=offload_cache,
            prefill_oproj_padded_tokens=prefill_oproj_padded_tokens,
            prefill_dense_padded_tokens=prefill_dense_padded_tokens,
            prefill_moe_global_chunks=prefill_moe_global_chunks,
        )


class GlmMoeDsaForCausalLM(GlmMoeDsaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self,
        config,
        infer_config: InferenceConfig,
        is_mtp=False,
        prefix: str = "",
        comm_manager: CommManager = None,
    ):
        super().__init__(config)
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        model_config = self.infer_config.model_config
        data_config = self.infer_config.data_config
        scheduler_config = self.infer_config.scheduler_config
        custom_params = model_config.custom_params
        self.platform_version = model_config.platform_version.value
        self.get_parallel_settings()
        self.num_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.pa_max_length = custom_params.get(
            "pa_max_length",
            data_config.input_truncated_len
            + scheduler_config.max_new_tokens * (model_config.next_n + 1)
            + model_config.next_n,
        )
        self.max_position_embeddings = custom_params.get("max_position_embeddings", self.pa_max_length)
        self.is_mtp = is_mtp
        self.mm_quant_mode = (
            config.quant_config.mm_quant_mode
            if config.quant_config is not None
            else "w16a16")
        self.update_kv_quant_settings()
        self.kv_cache_quant_mode = config.quant_config.kv_cache_quant_mode \
            if config.quant_config is not None else "unquant"

        self.exe_mode = model_config.exe_mode
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.rank_offset = int(os.getenv("RANK_OFFSET", "0"))
        self.global_rank = self.local_rank + self.rank_offset
        self.world_size = self.infer_config.parallel_config.world_size
        kwargs = {
                    "global_rank": self.global_rank,
                    "is_mtp": is_mtp
                }
        if self.comm_manager is None:
            raise RuntimeError("GlmMoeDsaForCausalLM requires CommManager in the refactored framework.")
        self.check_model_settings()
        self.init_parallel_comm_group()

        kwargs.update({
            "shared_expert_stream": create_stream("11", self.exe_mode),
            "indexer_stream": create_stream("22", self.exe_mode),
            "weights_stream": create_stream("33", self.exe_mode),
        })

        mtp_layer_idx = config.num_hidden_layers  # MTP is the last layer
        self.model = GlmMoeDsaModelMTPLayer(
            config,
            self.infer_config,
            self.comm_manager,
            mtp_layer_idx,
            prefix,
            **kwargs,
        ) if is_mtp else GlmMoeDsaModel(config, self.infer_config, self.comm_manager, prefix, **kwargs)
        self.vocab_size = config.vocab_size
        if not is_mtp:
            self.lm_head = ColumnParallelLinear(
                input_size=config.hidden_size,
                output_size=config.vocab_size,
                bias=False,
                tp_size=self.lmhead_tp_size,
                tp_rank=self.comm_manager.get_rank("lmhead_tp_group") if self.lmhead_tp_size > 1 else 0,
                quant_config=None,
                prefix="lm_head"
                )
        else:
            self.lm_head = None

        # Initialize weights and apply final processing
        self.post_init()

        self.enable_offload = custom_params.get("enable_offload", False)
        self.offload_cache = None

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

    def get_offload_workspace_memory_info(self) -> Optional[OffloadWorkspaceMemoryInfo]:
        if not self.enable_offload:
            return None
        return OffloadCache.build_workspace_spec(self.infer_config, self).memory_info()

    def init_offload_workspace(self, device):
        if not self.enable_offload:
            return
        self.offload_cache = OffloadCache(self.infer_config, self)
        self.offload_cache.init_workspace(device)

    def get_parallel_settings(self):
        parallel_config = self.infer_config.parallel_config
        self.embed_tp_size = parallel_config.embed_tp_size
        self.attn_dp_size = parallel_config.attn_dp_size
        self.attn_tp_size = parallel_config.attn_tp_size
        self.o_proj_tp_size = parallel_config.o_proj_tp_size
        self.cp_size = parallel_config.cp_size
        self.moe_ep_size = parallel_config.moe_ep_size
        self.moe_tp_size = parallel_config.moe_tp_size
        self.lmhead_tp_size = parallel_config.lmhead_tp_size
        self.moe_dp_size = parallel_config.moe_ep_size
        self.embed_dp_size = parallel_config.embed_dp_size
        self.dense_tp_size = parallel_config.dense_tp_size

    def check_model_settings(self):
        model_config = self.infer_config.model_config
        custom_params = model_config.custom_params
        exe_mode = model_config.exe_mode
        enable_multi_streams = custom_params.get("enable_multi_streams", False)
        enable_superkernel = custom_params.get("enable_superkernel", False)
        enable_offload = custom_params.get("enable_offload", False)
        moe_chunk_max_len = custom_params.get("moe_chunk_max_len", 65536)
        enable_cache_compile = model_config.enable_cache_compile
        disaggregation_mode = self.infer_config.disagg_config.disaggregation_mode

        if exe_mode not in ["eager", "ge_graph", "npugraph_ex"]:
            raise ValueError(f"{exe_mode=} is not supported!")
        if moe_chunk_max_len <= 0:
            raise ValueError(f"{moe_chunk_max_len=} should be a positive integer.")
        if self.attn_tp_size != 1:
            raise ValueError(f"GLM-5 only supports attn_tp_size == 1, got {self.attn_tp_size}.")
        if enable_offload and disaggregation_mode != "NONE":
            raise ValueError(
                "GLM-5 KVCache offload currently supports offline mode only; "
                f"got disaggregation_mode={disaggregation_mode!r}."
            )
        if enable_offload and self.kv_cache_quant_mode not in ("unquant", "int8"):
            raise ValueError(
                "GLM-5 KVCache offload only supports unquant and int8 KV cache; "
                f"got kv_cache_quant_mode={self.kv_cache_quant_mode!r}."
            )
        graph_only_feat_enabled = enable_multi_streams or enable_superkernel or enable_cache_compile
        if exe_mode == "eager" and graph_only_feat_enabled:
            raise ValueError(
                f"{exe_mode=} does not support graph-only features: "
                "enable_multi_streams, enable_superkernel or enable_cache_compile."
            )
        if exe_mode == "npugraph_ex" and enable_superkernel:
            raise ValueError("npugraph_ex does not support superkernel.")

    def init_parallel_comm_group(self):
        world_size = self.world_size
        self.comm_manager.register_group(
            name="attn_tp_group",
            group_num=world_size // self.attn_tp_size,
            group_size=self.attn_tp_size,
        )
        self.comm_manager.register_group(
            name="oproj_tp_group",
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
        self.comm_manager.register_group(
            name="dense_tp_group",
            group_num=world_size // self.dense_tp_size,
            group_size=self.dense_tp_size,
        )
        self.comm_manager.register_group(
            name="moe_tp_group",
            group_num=self.moe_dp_size,
            group_size=self.moe_tp_size,
        )
        moe_group_type = None if self.platform_version != "950" else 0
        self.comm_manager.register_group(
            name="moe_ep_group",
            group_num=self.moe_tp_size,
            group_size=world_size // self.moe_tp_size,
            group_stride=self.moe_tp_size,
            group_type=moe_group_type,
        )
        mc2_group_type = None if self.platform_version != "950" else 3
        is_full_mesh_v2 = self.platform_version != "950"
        hccl_buffer_size = calc_moe_hccl_buffer_size(self.infer_config, self.config, is_full_mesh_v2=is_full_mesh_v2)
        self.comm_manager.register_group(
            name="moe_ep_group_mc2",
            group_num=self.moe_tp_size,
            group_size=world_size // self.moe_tp_size,
            group_stride=self.moe_tp_size,
            return_name=True,
            hccl_buffer_size=hccl_buffer_size,
            group_type=mc2_group_type,
            allow_physical_reuse=False,
        )
        self.comm_manager.register_group(
            name="cp_group",
            group_num=world_size // self.cp_size,
            group_size=self.cp_size,
        )

    def forward_lm_head(
        self,
        outputs,
        is_prefill=True,
        actual_seq_lengths_q=None,
    ):
        num_tokens = outputs.shape[0]
        hidden_size = outputs.shape[-1]
        bs = actual_seq_lengths_q.shape[0]
        if is_prefill:
            # attention: SP + TP，moe：DP + EP
            if self.attn_tp_size > 1 and self.moe_ep_size > 1:
                new_outputs = torch.empty_like(outputs).repeat(self.attn_tp_size, 1)
                dist.all_gather_into_tensor(new_outputs, outputs,
                                            group=self.comm_manager.get_group("attn_tp_group"))
                outputs = new_outputs
            seq_index = actual_seq_lengths_q.to(dtype=torch.long, device=outputs.device) - 1
            outputs = torch.index_select(outputs.view(-1, hidden_size), 0, seq_index).view(bs, 1, hidden_size)
            q_len = 1  # prefill takes the last token
        else:  # flatten request/token axes for lm_head
            q_len = num_tokens // bs
            outputs = outputs.view(bs * q_len, 1, hidden_size)

        if (self.attn_dp_size == 1) or (self.lmhead_tp_size == 1):
            hidden_states = outputs
        else:
            # allgather: (bs / attn_dp, hidden_size) -> (bs, hidden_size)
            hidden_states = torch.zeros_like(outputs).repeat(self.lmhead_tp_size, 1, 1)
            dist.all_gather_into_tensor(hidden_states, outputs,
                                        group=self.comm_manager.get_group("lmhead_tp_group"))

        logits = self.lm_head(hidden_states)  # (lmhead_tp_size * bs / attn_dp, 1, vocab_size / lmhead_tp_size)
        if self.lmhead_tp_size > 1:  # -> (bs / attn_dp, 1, vocab_size)
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

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        forward_metadata: Optional[ForwardMetaData] = None,
        cur_topk_list: Optional[torch.Tensor] = None,
        **kwargs
    ):
        if forward_metadata is None:
            raise ValueError("GLM-5 refactored framework path requires forward_metadata.")
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            forward_metadata=forward_metadata,
            cur_topk_list=cur_topk_list,
            offload_cache=self.offload_cache,
        )

        prev_hidden_states = outputs
        cp_metadata = forward_metadata.cp_metadata if forward_metadata.is_prefill else None
        if cp_metadata is not None and cp_metadata.enabled:
            outputs = self.restore_prefill_cp_outputs(outputs, cp_metadata)

        logits = self.forward_lm_head(
            outputs,
            is_prefill=forward_metadata.is_prefill,
            actual_seq_lengths_q=forward_metadata.actual_seq_lengths_cu_q,
        )

        # Keep selected-cache status lifecycle local until the framework owns offload workspace.
        if forward_metadata.is_prefill and self.offload_cache is not None:
            self.offload_cache.reinit_status()

        return logits, prev_hidden_states

    def get_cache_info(self) -> ModelCacheInfo:
        layers = self.model.layers.values() if self.is_mtp else self.model.layers
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

    def update_kv_quant_settings(self):
        if self.config.quant_config is None:
            return
        # set li_cache_quant_mode to quant_config
        self.config.quant_config.set_quant_mode("li_cache_quant_mode", "unquant")

        # if quant to fp8 or mxfp8, set kv cache and li cache to fp8; if quant to int8, set li cache to int8
        if self.platform_version == "950" and "float" in self.mm_quant_mode:
            self.config.quant_config.kv_cache_quant_mode = "float8"
            self.config.quant_config.li_cache_quant_mode = "float8"
        elif self.config.quant_config:
            self.config.quant_config.li_cache_quant_mode = "int8"

    def process_weights_after_loading(self):
        layers = self.model.layers.values() if self.is_mtp else self.model.layers
        for layer in layers:
            try:
                attn = layer.self_attn
                attn.kv_b_proj_w_k = nn.Parameter(attn.kv_b_proj_w_k_data.contiguous(), requires_grad=False)
                attn.kv_b_proj_w_v = nn.Parameter(attn.kv_b_proj_w_v_data.contiguous(), requires_grad=False)
                attn.kv_b_proj.weight = None
            except AttributeError:
                continue

        float_scales_map = [
            "gate_up_proj",
            "q_b_proj",
            "wq_b",
        ]
        float_smooth_scales_map = [
            "down_proj",
        ]
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

            enable_weight_nz = self.infer_config.model_config.enable_weight_nz
            if self.platform_version == "950":
                # On 950, only selected attention projections use NZ; other weights stay non-NZ regardless of YAML.
                enable_weight_nz = any(
                    attn_proj_name in module_name
                    for attn_proj_name in ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa"]
                )

            is_nz = False if ("mlp.gate" in module_name and "proj" not in module_name) else enable_weight_nz
            is_transpose = False if ("mlp.gate" in module_name and "proj" not in module_name) else True
            if isinstance(quant_method, QuantizeMethodBase):
                quant_method.process_weights_after_loading(
                    module, is_nz=is_nz, is_transpose=is_transpose, scales_dtype=scales_dtype
                )

            if self.platform_version == "950" and self.model.config.quant_config is not None \
                    and self.model.config.quant_config.mm_quant_mode == "w8a8mxfloat8":
                if any(
                    attn_proj_name in module_name
                    for attn_proj_name in ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa"]
                ):
                    module.weight_scale = nn.Parameter(
                        module.weight_scale.transpose(0, 1).flatten(1).view(dtype=torch.float8_e8m0fnu),
                        requires_grad=False,
                    )

            if isinstance(quant_method, CompressedTensorW8A8Int8MoEGMMMethod) or \
                    isinstance(quant_method, CompressedTensorW4A8Int8MoEGMMMethod):
                moe_ep_size = self.infer_config.parallel_config.moe_ep_size
                if moe_ep_size > 1:
                    all_experts_smooth_scale = module.smooth_scale_1.data.new_empty(
                        module.smooth_scale_1.data.shape[0] * moe_ep_size,
                        module.smooth_scale_1.data.shape[1],
                    )
                    dist.all_gather_into_tensor(
                        all_experts_smooth_scale,
                        module.smooth_scale_1.data,
                        group=self.comm_manager.get_group("moe_ep_group"),
                    )
                    module.smooth_scale_1.data = all_experts_smooth_scale
        gc.collect()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        repeat_loaded_weights_mapping = []  # (origin_name: repeat_loaded_name)

        # Params for weights, int8 weight scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoEGMM.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts)

        params_dict = dict(self.named_parameters())
        dequant_cache = {}
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if self.config.architectures[0] == 'GlmMoeDsaForCausalLM' and self.config.num_nextn_predict_layers > 0:
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
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
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

                if self.mm_quant_mode == "w8a8float8":
                    name = name.replace(".weight_scale_inv", ".scale")

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
                    if self.mm_quant_mode == "w8a8float8":
                        name = name.replace(".weight_scale_inv", ".scale")

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

                    if self.mm_quant_mode == "w8a8float8":
                        name = name.replace(".weight_scale_inv", ".scale")

                    if "kv_b_proj" in name and self.mm_quant_mode == "w8a8float8":
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

                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class GlmMoeDsaModelMTP(GlmMoeDsaForCausalLM):

    def __init__(
        self,
        config: GlmMoeDsaConfig,
        infer_config: InferenceConfig,
        comm_manager: CommManager = None,
        **kwargs,
    ):
        super().__init__(config, infer_config, is_mtp=True, comm_manager=comm_manager)
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.rank_offset = int(os.getenv("RANK_OFFSET", "0"))
        self.global_rank = self.local_rank + self.rank_offset
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.vocab_size_per_rank = self.vocab_size // self.embed_tp_size
        self.ignore_share_weight = False

        # reuse embed_tokens and lm_head from main model; rotary has no learned state.
        self.embed_tokens = None
        self.lm_head = None
        self.rotary_emb = self.model.rotary_emb

        self.shared_head_norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.enorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # prev_hidden_states and input_hidden_state feature fusion
        self.eh_proj = ReplicatedLinear(
            2 * config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=None,
            prefix=f"model.layers.{self.mtp_start_layer_idx}.eh_proj")

    @override
    def forward(
        self,
        input_ids: torch.LongTensor,
        prev_hidden_states: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cur_topk_list: Optional[torch.Tensor] = None,
        forward_metadata: Optional[ForwardMetaData] = None,
        **kwargs
    ):
        if forward_metadata is None:
            raise ValueError("GLM-5 MTP requires forward_metadata from the new framework.")

        is_prefill = forward_metadata.is_prefill
        kv_len = forward_metadata.kv_len
        actual_seq_lengths_kv = forward_metadata.actual_seq_lengths_kv
        actual_seq_lengths_q = forward_metadata.actual_seq_lengths_cu_q
        slot_mapping = forward_metadata.slot_mapping
        block_table = forward_metadata.block_table
        cp_metadata = getattr(forward_metadata, "cp_metadata", None)
        if slot_mapping is not None:
            slot_mapping = slot_mapping["FullAttention"]
        if block_table is not None:
            block_table = block_table["FullAttention"]

        input_ids, position_ids, slot_mapping = self.model.select_prefill_cp_local_inputs(
            input_ids, position_ids, slot_mapping, cp_metadata, is_prefill
        )
        input_ids = input_ids.to(torch.int32)
        hidden_states = self.model.calc_input_embeddings(input_ids, is_prefill)
        cos_sin = self.rotary_emb(hidden_states, position_ids, kv_len, self.max_position_embeddings)

        hidden_states = self.enorm(hidden_states)
        prev_hidden_states = self.hnorm(prev_hidden_states)
        hidden_states = self.eh_proj(torch.cat([hidden_states, prev_hidden_states], dim=-1))
        residual = None

        residual, hidden_states = self.model(
            hidden_states,
            kv_len,
            actual_seq_lengths_kv,
            actual_seq_lengths_q=actual_seq_lengths_q,
            cos_sin=cos_sin,
            past_residual=residual,
            position_ids=position_ids,
            is_prefill=is_prefill,
            cur_topk_list=cur_topk_list,
            slot_mapping=slot_mapping,
            block_table=block_table,
            cp_metadata=cp_metadata,
            offload_cache=self.offload_cache,
        )

        prev_hidden_states, _ = self.shared_head_norm(hidden_states, residual)
        prev_hidden_states = self.restore_prefill_cp_outputs(
            prev_hidden_states,
            cp_metadata if is_prefill else None,
        )
        logits = self.forward_lm_head(
            outputs=prev_hidden_states,
            is_prefill=is_prefill,
            actual_seq_lengths_q=forward_metadata.actual_seq_lengths_cu_q,
        )

        # Keep selected-cache status lifecycle local until the framework owns offload workspace.
        if is_prefill and self.offload_cache is not None:
            self.offload_cache.reinit_status()

        return logits, prev_hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping, mtp_unique_weight_mapping, expert_params_mapping, repeat_loaded_weights_mapping \
            = self._load_weight_map()

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        dequant_cache = {}
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
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
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
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
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
                if self.mm_quant_mode == "w8a8float8":
                    name = name.replace(".weight_scale_inv", ".scale")

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
                    if self.mm_quant_mode == "w8a8float8":
                        name = name.replace(".weight_scale_inv", ".scale")

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
                    if self.mm_quant_mode == "w8a8float8":
                        name = name.replace(".weight_scale_inv", ".scale")
                    if "kv_b_proj" in name and self.mm_quant_mode == "w8a8float8":
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

        repeat_loaded_weights_mapping = []  # (origin_name: repeat_loaded_name)
        return stacked_params_mapping, mtp_unique_weight_mapping, expert_params_mapping, repeat_loaded_weights_mapping


def get_spec_layer_idx_from_weight_name(config, weight_name: str) -> Optional[int]:
    if (
        hasattr(config, "num_nextn_predict_layers")
        and config.num_nextn_predict_layers > 0
    ):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(f"model.layers.{layer_idx+i}."):
                return layer_idx + i
    return None
