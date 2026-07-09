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
    get_decode_mask)

from executor.model_loader.weight_utils import default_weight_loader
from executor.utils import (
    superkernel_scope, weight_dequant,
    limit_core_num)
from executor.utils.stream_utils import npu_stream_switch, record_event, wait_event, record_stream
from executor.core.config import InferenceConfig, CommManager, PlatformVersion
from executor.core.kv_cache.cache_info import CacheEntry, LayerCacheInfo, ModelCacheInfo
from executor.utils.forward_metadata import ForwardMetaData
from module.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding
    )
from module.fuse_moe_gmm import FusedMoEGMM
from module.quantization.utils.quant_utils import reshape_mx_scale
from module.quantization import QuantizeMethodBase
from module.quantization.compressed_tensors.compressed_tensors_moe_gmm import (
    CompressedTensorW8A8Int8MoEGMMMethod,
    CompressedTensorW4A8Int8MoEGMMMethod,
)
from module.quantization.fp8 import Fp8MoEGMMMethod
from .configuration_deepseek import DeepseekV3Config
from .modules import (get_window_topk_idxs, get_compress_topk_idxs,
                      one_hot, yarn_get_mscale,
                      DeepseekV3RMSNorm, _init_rope, DEEPSEEKV3_START_DOCSTRING,
                      DEEPSEEKV3_INPUTS_DOCSTRING, DeepseekV3PreTrainedModel, apply_rotary_emb,
                      partial_rotary_mul_quant, AttnMetaData
                    )
from .modules import Indexer, Compressor
from .modules.registry import OpKernel, auto_import_modules
logger = logging.get_logger(__name__)

HADAMARD_SIZE = 128
MATMUL_MAX_AXIS_VALUE = 65535


def get_max_position_embeddings(infer_config: InferenceConfig, is_mtp: bool = False):
    input_truncated_len = infer_config.data_config.input_truncated_len
    max_new_tokens = infer_config.scheduler_config.max_new_tokens
    if is_mtp:
        next_n = infer_config.model_config.next_n
        return max_new_tokens * (next_n + 1) + input_truncated_len + next_n - 1
    return input_truncated_len + max_new_tokens


class DeepseekV3SharedExpert(nn.Module):
    def __init__(
        self,
        config,
        infer_config: InferenceConfig,
        comm_manager: CommManager = None,
        is_moe_layer=False,
        prefix="",
        **kwargs,
    ):
        super().__init__()
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.mm_quant_mode = (
            config.quant_config.mm_quant_mode
            if config.quant_config is not None
            else "w16a16")
        self.swiglu_limit = config.swiglu_limit if hasattr(config, "swiglu_limit") else None
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
            prefix=f"{prefix}.gate_up_proj",
            )
        self.down_proj = RowParallelLinear(
            config.moe_intermediate_size * config.n_shared_experts,
            config.hidden_size,
            bias=False,
            tp_size=self.moe_tp_size,
            tp_rank=self.comm_manager.get_rank("moe_tp_group") if self.moe_tp_size > 1 else 0,
            quant_config=config.quant_config,
            prefix=f"{prefix}.down_proj",
            )
        if self.mm_quant_mode == "w8a8int8":
            self.forward = self.forward_w8a8int8
        elif "hifloat8" in self.mm_quant_mode:
            self.forward = self.forward_a8hifloat8
        elif "float8" in self.mm_quant_mode and "a8" in self.mm_quant_mode:
            self.forward = self.forward_a8float8
        else:
            self.forward = self.forward_normal

    def forward_normal(self, x, enable_decode_stream=False, shared_expert_event=None):
        merged_x = self.gate_up_proj(x)
        intermediate_hidden_states = torch_npu.npu_swiglu(merged_x)
        wait_event(enable_decode_stream, shared_expert_event, 0)
        return self.down_proj(intermediate_hidden_states)

    def forward_w8a8int8(self, x, enable_decode_stream=False, shared_expert_event=None):
        merged_x, pertoken_scale = self.gate_up_proj(x, out_dtype=torch.int32)
        swiglu_limit_args = {}
        if self.swiglu_limit:
            swiglu_limit_args.update({
                "swiglu_mode": 1,
                "clamp_limit": self.swiglu_limit,
                "glu_alpha": 1,
                "glu_bias": 0
            })
        intermediate_hidden_states, pertoken_scale = torch_npu.npu_dequant_swiglu_clamp_quant(
            merged_x, weight_scale=self.gate_up_proj.weight_scale,
            quant_scale=self.down_proj.smooth_scales,
            quant_mode=1, activate_left=True,
            activation_scale=pertoken_scale,
            **swiglu_limit_args
        )
        wait_event(enable_decode_stream, shared_expert_event, 0)
        return self.down_proj(intermediate_hidden_states, pertoken_scale)

    def forward_a8float8(self, x, enable_decode_stream=False, shared_expert_event=None):
        merged_x = self.gate_up_proj(x)
        intermediate_hidden_states, pergroup_scale, _ = torch.ops.custom.npu_swiglu_group_quant(
            merged_x,
            dst_type=torch.float8_e4m3fn,
            round_scale=True if "mx" in self.mm_quant_mode else False,
            quant_mode=1 if "mx" in self.mm_quant_mode else 0,
            clamp_limit=self.swiglu_limit, 
            )
        wait_event(enable_decode_stream, shared_expert_event, 0)
        return self.down_proj(intermediate_hidden_states, pergroup_scale)

    def forward_a8hifloat8(self, x, enable_decode_stream=False, shared_expert_event=None):
        merged_x = self.gate_up_proj(x)
        if self.swiglu_limit:
            half = merged_x.size(-1) // 2
            gate = merged_x[..., :half].clamp(max=self.swiglu_limit)
            up = merged_x[..., half:].clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            merged_x = torch.cat([gate, up], dim=-1)
        intermediate_hidden_states = torch_npu.npu_swiglu(merged_x)
        wait_event(enable_decode_stream, shared_expert_event, 0)
        return self.down_proj(intermediate_hidden_states)


class DeepseekV3MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(
        self,
        config,
        infer_config: InferenceConfig,
        comm_manager: CommManager = None,
        prefix="",
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = kwargs.get("layer_idx")
        self.hash = self.layer_idx < config.num_hash_layers
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.gmm_quant_mode = (
            config.quant_config.gmm_quant_mode
            if config.quant_config is not None
            else "w16a16")
        self.swiglu_limit = config.swiglu_limit if hasattr(config, "swiglu_limit") else None
        self.hidden_dim = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.moe_tp_size = self.infer_config.parallel_config.moe_tp_size
        self.moe_ep_size = self.infer_config.parallel_config.moe_ep_size
        self.moe_chunk_max_len = self.infer_config.model_config.custom_params.get("moe_chunk_max_len", 65536)
        self.platform_version = self.infer_config.model_config.platform_version
        self.exe_mode = self.infer_config.model_config.exe_mode
        self.enable_multi_streams = self.infer_config.model_config.custom_params.get("enable_multi_streams", False)
        self.force_eplb = self.infer_config.model_config.force_eplb
        self.num_experts_per_tok = config.num_experts_per_tok
        # total experts num
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok

        self.npu_events = []
        self.shared_expert_event = []
        if self.enable_multi_streams:
            self.npu_events = [torch.npu.Event(), torch.npu.Event()]
            self.shared_expert_event = [torch.npu.Event()]

        self.intermediate_size_per_rank = self.intermediate_size // self.moe_tp_size
        self.shared_expert_rank_num = 0 # route and share on same card
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
        self.gmm_int_quant = "a8" in self.gmm_quant_mode and "float" not in self.gmm_quant_mode
        self.moe_ffn = self.experts_w4a8int4 if self.gmm_quant_mode == "w4a8int4" else self.experts
        self._init_gate(prefix)
        if config.n_shared_experts is not None:
            self.shared_experts = DeepseekV3SharedExpert(
                config,
                self.infer_config,
                self.comm_manager,
                is_moe_layer=True,
                prefix=f"{prefix}.shared_experts",
                **kwargs,
            )
        self.dispatch_quant_mode = {
            "w16a16": 0,
            "w8a8int8": 2,
            "w4a8int4": 2,
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
                                     params_dtype=torch.bfloat16 \
                                        if self.platform_version == PlatformVersion.ASCEND_950 else torch.float32,
                                     prefix=f"{prefix}.gate",
                                     )
        self._reset_parameters()
        if self.hash:
            self.gate.e_score_correction_bias = None
            self.tid2eid = nn.Parameter(torch.randint(high=self.n_routed_experts, \
                                                      size=(self.config.vocab_size, self.top_k), \
                                                        dtype=torch.int32), requires_grad=False)
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
        moe_ep_group_name = self.comm_manager.get_group_name("moe_ep_group_mc2")
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
        if self.platform_version != PlatformVersion.ASCEND_950:
            self.dispatch_kwargs["comm_alg"] = "fullmesh_v2"

    def forward(self, hidden_states, is_prefill=False, cur_topk_list=None, input_ids=None, shared_expert_stream=None):
        _, h = hidden_states.shape
        if is_prefill:
            if self.n_shared_experts > 0:
                hidden_states_share = self.forward_shared_expert(hidden_states, shared_expert_stream)
            else:
                hidden_states_share = None
        else:
            record_stream(self.enable_multi_streams, hidden_states, shared_expert_stream)
            record_event(self.enable_multi_streams, self.npu_events, 0)

        # compute gating score
        if self.platform_version == PlatformVersion.ASCEND_950:
            logits = torch_npu.npu_fused_matmul(hidden_states.view(-1, h), self.gate.weight, fused_op_type="16cast32")
        else:
            logits = self.gate(hidden_states.view(-1, h).to(torch.float32))
        topk_idx, topk_weight, _ = OpKernel.gate_topk(self, logits, input_ids)
        if self.force_eplb:
            topk_idx = cur_topk_list
        topk_idx = topk_idx.to(torch.int32)

        # MOE EP
        if is_prefill:
            return self.moe_infer_double_routing(
                hidden_states, topk_idx, topk_weight, hidden_states_share)
        else:
            return self.moe_infer_dispatch_combine(hidden_states, topk_idx, topk_weight, shared_expert_stream)

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
        hidden_states_ordered_by_experts = self.moe_ffn(**gmm_args)
        # finalize-rerouting
        new_x = torch.index_select(hidden_states_ordered_by_experts, 0, gathered_ids_unsort.float().argsort().int())
        return new_x

    def experts_w4a8int4(
        self,
        x: torch.Tensor,
        expert_tokens: torch.Tensor,
        group_list_type: int,
        pertoken_scale: torch.Tensor = None,
        final_output_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        hidden_size = x.size(-1)

        if pertoken_scale is None:
            x, pertoken_scale = torch_npu.npu_dynamic_quant(x)

        if pertoken_scale.dim() > 1:
            pertoken_scale = pertoken_scale.reshape(-1)
            x = x.view(-1, hidden_size)

        mm1_mm3 = torch_npu.npu_grouped_matmul(
            [x], [self.experts.w13_weight], bias=[self.experts.w13_bias],
            scale=[self.experts.w13_weight_scale], per_token_scale=[pertoken_scale],
            group_list=expert_tokens, split_item=3,
            output_dtype=final_output_dtype, group_type=0,
            group_list_type=group_list_type,
            act_type=0
        )[0]

        # fusion kernel: swiglu + clip + dynamic_quant
        # To enhance quantization accuracy, a clip kernel has been introduced prior to the dynamic_quant kernel
        swiglu_limit = kwargs.get("swiglu_limit", None)
        swiglu_limit_args = {}
        if swiglu_limit:
            swiglu_limit_args.update({
                "swiglu_mode": 1,
                "clamp_limit": swiglu_limit,
                "glu_alpha": 1,
                "glu_bias": 0
            })
        intermediate_h, pertoken_scale = torch_npu.npu_dequant_swiglu_clamp_quant(
            mm1_mm3,
            quant_scale=self.experts.smooth_scale_2,
            group_index=expert_tokens,
            activate_left=True,
            quant_mode=1,
            **swiglu_limit_args,
            )

        out_hidden = torch_npu.npu_grouped_matmul(
            [intermediate_h], [self.experts.w2_weight], bias=[self.experts.w2_bias],
            scale=[self.experts.w2_weight_scale], per_token_scale=[pertoken_scale],
            group_list=expert_tokens, split_item=3,
            output_dtype=final_output_dtype, group_type=0,
            group_list_type=group_list_type,
            act_type=0
        )[0]

        return out_hidden

    def combine_double_routing(self, new_x, expanded_x, input_splits, output_splits):
        moe_ep_group = self.comm_manager.get_group("moe_ep_group")
        gathered_tokens = new_x.new_empty(expanded_x.shape[0], new_x.shape[1])
        dist.all_to_all_single(gathered_tokens, new_x, input_splits, output_splits, group=moe_ep_group)
        return gathered_tokens

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
                gathered_pertoken_scale = pertoken_scale.new_empty(gathered_tokens.shape[0], pertoken_scale.shape[1], \
                    pertoken_scale.shape[2])
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
        num_tokens, h = x.shape

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
                *self._split_tensors(num_tokens, x, topk_ids, topk_weight, hidden_states_share)):
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

        return hidden_states.view(num_tokens, h)

    def moe_infer_dispatch_combine(self, x, topk_ids, topk_weight, shared_expert_stream):
        """
        tp+ep mix strategy, for decode stage
        """
        num_tokens, h = x.shape
        hidden_states = x
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

        hidden_states_ordered_by_experts = self.moe_ffn(**gmm_args)

        record_event(self.enable_multi_streams, self.shared_expert_event, 0)
        with npu_stream_switch(self.enable_multi_streams, shared_expert_stream):
            wait_event(self.enable_multi_streams, self.npu_events, 0)
            # shared_expert use multi streams
            hidden_states_share = self.shared_experts(hidden_states.view(-1, hidden_states.shape[-1]), \
                enable_decode_stream=self.enable_multi_streams, shared_expert_event=self.shared_expert_event)
            record_event(self.enable_multi_streams, self.npu_events, 1)

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
        wait_event(self.enable_multi_streams, self.npu_events, 1)
        hidden_states = hidden_states + hidden_states_share
        hidden_states = hidden_states.view(num_tokens, self.hidden_dim)
        return hidden_states


class Attention(nn.Module):
    """Multi-Query Attention (MQA) Layer."""
    def __init__(
        self,
        config: DeepseekV3Config,
        infer_config: InferenceConfig,
        comm_manager: CommManager = None,
        layer_idx: Optional[int] = None,
        prefix: Optional[str] = "",
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.is_online = (
            infer_config.disagg_config.disaggregation_mode in ("PREFILL", "DECODE")
        )
        self.batch_size = self.infer_config.scheduler_config.batch_size
        self.batch_size_per_rank = self.infer_config.scheduler_config.batch_size_per_dp_rank
        self.attn_tp_size = self.infer_config.parallel_config.attn_tp_size
        self.attn_dp_size = self.infer_config.parallel_config.attn_dp_size
        self.oproj_tp_size = self.infer_config.parallel_config.o_proj_tp_size
        if self.oproj_tp_size > config.o_groups:
            raise ValueError(f"{self.oproj_tp_size=} should not be greater than {config.o_groups =}")
        self.moe_tp_size = self.infer_config.parallel_config.moe_tp_size
        self.moe_ep_size = self.infer_config.parallel_config.moe_ep_size
        self.world_size = self.infer_config.parallel_config.world_size
        self.enable_pypto = self.infer_config.model_config.custom_params.get("enable_pypto", False)
        # enable_global_multi_streams: global multistream, to wait for attention kernel metadata
        self.enable_global_multi_streams = \
            self.infer_config.model_config.custom_params.get("enable_multi_streams", False)
        self.enable_npugraph_ex = self.infer_config.model_config.exe_mode == "npugraph_ex"
        # enable_multi_streams: multistream within current class
        self.enable_multi_streams = self.enable_global_multi_streams and not self.enable_pypto
        self.platform_version = self.infer_config.model_config.platform_version
        self.layer_idx = layer_idx
        self.is_mtp = False
        if layer_idx == config.num_hidden_layers:
            self.layer_idx = 0 # MTP model only has one layer of cache
            self.is_mtp = True
        self.mla_events = []
        if self.enable_multi_streams:
            # 4 is number of events used for event synchronization
            self.mla_events = [torch.npu.Event(), torch.npu.Event(), torch.npu.Event(), torch.npu.Event()]

        self.enable_limit_core = self.infer_config.model_config.custom_params.get("enable_limit_core", False)
        self.compress_ratio = 1 if self.is_mtp else config.compress_ratios[layer_idx]
        if self.enable_multi_streams and self.platform_version == PlatformVersion.A3:
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
                                        tp_rank=self.comm_manager.get_rank("attn_tp_group") \
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
            wo_tp_rank = self.comm_manager.get_rank("attn_tp_group") if self.attn_tp_size > 1 else 0
        else:
            wo_tp_size = self.oproj_tp_size
            wo_tp_rank = self.comm_manager.get_rank("oproj_tp_group")
        quant_config = config.quant_config if self.mm_quant_mode == "w8a8mxfloat8" \
            or self.mm_quant_mode == "w8a8hifloat8" else None
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

        self.max_position_embeddings = get_max_position_embeddings(self.infer_config, self.is_mtp)
        self.original_seq_len = config.max_seq_len

        self.li_cache_quant_mode = config.quant_config.li_cache_quant_mode \
            if config.quant_config is not None else "unquant"

        self.kv_cache_quant_mode = config.quant_config.kv_cache_quant_mode \
            if config.quant_config is not None else "unquant"

        self.window_size = config.sliding_window
        self.block_size = self.infer_config.scheduler_config.block_size

        self._init_cache_entries()

        if self.compress_ratio > 1:
            self.compressor = Compressor(config, self.infer_config, layer_idx, self.compress_ratio,
                                         head_dim=self.head_dim, prefix=f"{prefix}.compressor",
                                         comm_manager=self.comm_manager, cache_getter=self._get_cache_tensor, **kwargs)
            if self.compress_ratio == 4:
                self.indexer = Indexer(config, self.infer_config, layer_idx, self.compress_ratio,
                                       prefix=f"{prefix}.indexer", comm_manager=self.comm_manager,
                                       cache_getter=self._get_cache_tensor, **kwargs)
            else:
                self.indexer = None

        self.sparse_attn_ops = torch.ops.custom.npu_sparse_attn_sharedkv
        if self.kv_cache_quant_mode == "float8" or self.kv_cache_quant_mode == "hifloat8":
            self.sparse_attn_ops = torch.ops.custom.npu_kv_quant_sparse_attn_sharedkv

        self.cp_size = self.infer_config.parallel_config.cp_size
        self.global_rank = kwargs.get("global_rank")

        if self.mm_quant_mode == "w8a8hifloat8":
            self.register_buffer("beta", torch.zeros(self.q_lora_rank, dtype=torch.bfloat16))

    def _get_cache_dtype_map(self):
        return {
            "int8": torch.int8,
            "float8": torch.float8_e4m3fn,
            "unquant": torch.bfloat16,
            "hifloat8": torch.uint8
        }

    def _get_cache_dim_and_cmp_dtype(self):
        cache_dim = self.config.head_dim
        use_fused_kernel_compressor = (
            self.infer_config.model_config.custom_params.get("kernel_config", {}).get("compressor", "native")
            == "ascendc"
        )
        cmp_kv_dtype = torch.float8_e4m3fn if (
            self.platform_version == PlatformVersion.ASCEND_950 and use_fused_kernel_compressor
        ) else torch.bfloat16

        if "float" in self.kv_cache_quant_mode:
            rope_dim = self.config.qk_rope_head_dim
            nope_dim = self.config.head_dim - rope_dim
            cache_dim = align_up(nope_dim + 2 * rope_dim + nope_dim // 64, 128)
            cmp_kv_dtype = torch.float8_e4m3fn
        return cache_dim, cmp_kv_dtype

    def _make_cache_setter(self, cache_name):
        return lambda tensor, layer=self, name=cache_name: setattr(layer, name, tensor)

    def _get_cache_tensor(self, cache_name):
        cache_tensor = getattr(self, cache_name, None)
        if cache_tensor is None:
            raise KeyError(f"cache tensor {cache_name} is not initialized in Attention.")
        return cache_tensor

    def _get_runtime_cmp_cache(self, attn_metadata, is_prefill):
        if is_prefill and self.cp_size > 1 and not self.is_online and attn_metadata["cp_tmp_cache"]:
            tmp_cache = attn_metadata["cp_tmp_cache"]
            if tmp_cache is not None:
                return tmp_cache[f"{self.compress_ratio}"]["sfa_cmp_kv"]
            else:
                raise ValueError("When cp is enabled, a temporary cache is required, but no temporary cache is found.")
        return self.sfa_cmp_kv

    def _init_cache_entries(self):
        self.cache_entries = []

        cache_dtype_map = self._get_cache_dtype_map()
        cache_dtype = cache_dtype_map[self.kv_cache_quant_mode]
        li_cache_dtype = cache_dtype_map[self.li_cache_quant_mode]

        if self.kv_cache_quant_mode == "hifloat8":
            cache_dtype = torch.float8_e4m3fn
        cache_dim, cmp_kv_dtype = self._get_cache_dim_and_cmp_dtype()

        self.win_kv = torch.Tensor([])
        self.cache_entries.append(
            CacheEntry(
                cache_name="win_kv",
                attn_type="SlidingWindow",
                dim=cache_dim,
                num_head=1,
                dtype=cache_dtype,
                needs_block=True,
                block_size=self.block_size,
                manager_key="win_kv",
                tensor_setter=self._make_cache_setter("win_kv"),
                sliding_window=self.window_size,
            )
        )

        if self.compress_ratio == 4:
            self.sfa_cmp_kv = torch.Tensor([])
            self.li_cmp_kv = torch.Tensor([])
            self.sfa_kv_state = torch.Tensor([])
            self.li_kv_state = torch.Tensor([])
            self.li_key_dequant_scale = torch.Tensor([])
            li_scale_dtype = torch.float16 if self.li_cache_quant_mode == "int8" else torch.float32
            self.cache_entries.extend([
                CacheEntry(
                    cache_name="sfa_cmp_kv",
                    attn_type="FullAttention",
                    dim=cache_dim,
                    num_head=1,
                    dtype=cmp_kv_dtype,
                    needs_block=True,
                    block_size=self.block_size * 4,
                    manager_key="c4a_cmp_kv",
                    tensor_setter=self._make_cache_setter("sfa_cmp_kv"),
                    compress_ratio=4,
                ),
                CacheEntry(
                    cache_name="li_cmp_kv",
                    attn_type="FullAttention",
                    dim=self.config.index_head_dim,
                    num_head=1,
                    dtype=li_cache_dtype,
                    needs_block=True,
                    block_size=self.block_size * 4,
                    manager_key="c4a_cmp_kv",
                    tensor_setter=self._make_cache_setter("li_cmp_kv"),
                    compress_ratio=4,
                ),
                CacheEntry(
                    cache_name="sfa_kv_state",
                    attn_type="SlidingWindow",
                    dim=[2, self.config.head_dim],
                    num_head=2, # state + score
                    dtype=torch.float32,
                    needs_block=True,
                    block_size=self.block_size,
                    manager_key="c4a_cmp_state",
                    tensor_setter=self._make_cache_setter("sfa_kv_state"),
                    sliding_window=8,
                ),
                CacheEntry(
                    cache_name="li_kv_state",
                    attn_type="SlidingWindow",
                    dim=[2, self.config.index_head_dim],
                    num_head=2, # state + score
                    dtype=torch.float32,
                    needs_block=True,
                    block_size=self.block_size,
                    manager_key="c4a_cmp_state",
                    tensor_setter=self._make_cache_setter("li_kv_state"),
                    sliding_window=8,
                )])
            if self.li_cache_quant_mode in ["int8", "float8", "hifloat8"]:
                dtype = torch.float16 if self.li_cache_quant_mode == "int8" else torch.float32
                self.cache_entries.extend([
                    CacheEntry(
                        cache_name="li_key_dequant_scale",
                        attn_type="FullAttention",
                        dim=1,
                        num_head=1,
                        dtype=dtype,
                        needs_block=True,
                        block_size=self.block_size * 4,
                        manager_key="c4a_cmp_kv",
                        tensor_setter=self._make_cache_setter("li_key_dequant_scale"),
                        compress_ratio=4,
                    ),
                ])
        elif self.compress_ratio == 128:
            self.sfa_cmp_kv = torch.Tensor([])
            self.sfa_kv_state = torch.Tensor([])
            self.cache_entries.extend([
                CacheEntry(
                    cache_name="sfa_cmp_kv",
                    attn_type="FullAttention",
                    dim=cache_dim,
                    num_head=1,
                    dtype=cmp_kv_dtype,
                    needs_block=True,
                    block_size=self.block_size * 128,
                    manager_key="c128a_cmp_kv",
                    tensor_setter=self._make_cache_setter("sfa_cmp_kv"),
                    compress_ratio=128,
                ),
                CacheEntry(
                    cache_name="sfa_kv_state",
                    attn_type="SlidingWindow",
                    dim=[1, self.config.head_dim],
                    num_head=2,
                    dtype=torch.float32,
                    needs_block=True,
                    block_size=self.block_size,
                    manager_key="c128a_cmp_state",
                    tensor_setter=self._make_cache_setter("sfa_kv_state"),
                    sliding_window=128,
                    compress_ratio=1,
                ),
            ])

    def apply_norm_dynamic_quant(self, x):
        if self.mm_quant_mode == "w8a8float8":
            x = self.q_norm(x)
            num_tokens, _ = x.shape
            x, x_scale = torch_npu.npu_dynamic_block_quant(x,
                dst_type=torch.float8_e4m3fn,
                row_block_size=1,
                col_block_size=self.config.quant_config.weight_block_size[1])
            return x.view(num_tokens, -1), x_scale.view(num_tokens, -1)
        elif self.mm_quant_mode == "w8a8hifloat8":
            num_tokens, _ = x.shape
            x = torch_npu.npu_rms_norm_quant(x, self.q_norm.weight, \
                self.beta, self.wq_b.scale, self.wq_b.offset, self.eps, \
                dst_dtype=torch_npu.hifloat8)
            return x.view(num_tokens, -1), self.wq_b.scale
        elif self.mm_quant_mode == "w8a8mxfloat8":
            x = self.q_norm(x)
            return torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
        elif self.mm_quant_mode == "w8a8int8":
            if self.platform_version == PlatformVersion.ASCEND_950:
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

        win_cache = self.win_kv
        win_kv_slot_mapping = attn_metadata["slot_mapping"]["win_kv"]
        self.update_win_kv(kv, win_kv_slot_mapping, win_cache)
        return q, qr, qr_scale, x

    def get_cp_window(
        self,
        x: torch.Tensor,
        attn_metadata: Optional[Dict] = None,
    ):
        hidden_size = x.shape[-1]
        cp_metadata = attn_metadata["cp_metadata"]
        cp_rank = cp_metadata.get("cp_rank", self.global_rank % self.cp_size)

        # Collect tail windows (last 128 tokens) of all CP segments
        x_prev, x_next = x.split(x.shape[0] // 2, dim=0)
        cur_win_list = []
        for zz_flag in ["prev", "next"]:
            x_seg = x_prev if zz_flag == "prev" else x_next
            cur_kv_len = attn_metadata[zz_flag]["cur_kv_len"]
            if cur_kv_len >= self.window_size:
                cur_win = x_seg[cur_kv_len - self.window_size: cur_kv_len]
            else:
                cur_win = x_seg[:cur_kv_len]
                pad_win = x.new_zeros((self.window_size - cur_kv_len, hidden_size))
                cur_win = torch.cat([cur_win, pad_win], dim=0)
            cur_win_list.append(cur_win)
        cur_win = torch.cat(cur_win_list, dim=0)
        all_win = x.new_empty((cur_win.shape[0] * self.cp_size, hidden_size))
        dist.all_gather_into_tensor(all_win, cur_win, group=self.comm_manager.get_group("cp_group"))
        all_win = all_win.view(-1, self.window_size, hidden_size)[cp_metadata["reverse_index"]]

        # Retrieve the window from the tail of the previous CP segment,
        # and concatenate it to the current CP segment
        x_list = []
        for zz_flag in ["prev", "next"]:
            x_seg = x_prev if zz_flag == "prev" else x_next
            if not (attn_metadata[zz_flag]["is_start"] and zz_flag == "prev"):
                # get pre win except rank 0
                if zz_flag == "prev":
                    pre_win = all_win[cp_rank - 1]
                else:
                    pre_win = all_win[2 * self.cp_size - cp_rank - 2]
                x_seg = torch.cat([pre_win, x_seg], dim=0)
            x_list.append(x_seg)

        # Compute the last window kv of the entire sequence
        cos_sin = attn_metadata["prev"]["cos_sin"]
        cos, sin = cos_sin["c1a_last_win"] if self.compress_ratio == 1 \
            else cos_sin["comp_last_win"]
        last_kv_len = attn_metadata["prev"]["last_kv_len"]
        last_rank = attn_metadata["cp_metadata"]["last_rank"]
        if last_kv_len >= self.window_size:
            last_win = all_win[last_rank]
        elif last_rank == 0:
            last_win = all_win[last_rank]
        else:
            last_win = all_win[last_rank, :last_kv_len]
            second_last_win = all_win[last_rank - 1]
            last_win = torch.cat([second_last_win[-(self.window_size - last_kv_len):], last_win], dim=0)
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
        is_prefill: bool = True,
        **kwargs,
    ):
        num_tokens, _ = x.size()
        enable_multi_streams = self.enable_multi_streams and self.platform_version != PlatformVersion.ASCEND_950
        x_scale = None
        move_quant_before = "float8" in self.mm_quant_mode and self.cp_size == 1
        if self.cp_size == 1:
            if self.mm_quant_mode == "w8a8float8":
                x_q, x_scale = torch_npu.npu_dynamic_block_quant(
                    x.view(-1, x.size(-1)),
                    dst_type=torch.float8_e4m3fn,
                    row_block_size=1,
                    col_block_size=self.config.quant_config.weight_block_size[1],
                )
            elif self.mm_quant_mode == "w8a8mxfloat8":
                x_q, x_scale = torch_npu.npu_dynamic_mx_quant(
                    x.view(-1, x.size(-1)), dst_type=torch.float8_e4m3fn
                )

        qr = self.wq_a(x_q if move_quant_before else x, dynamic_scale=x_scale).view(num_tokens, -1)
        if self.cp_size > 1:
            cp_metadata_prev, cp_metadata_next = attn_metadata["prev"], attn_metadata["next"]
            cos_sin_prev, cos_sin_next = cp_metadata_prev["cos_sin"], cp_metadata_next["cos_sin"]
            cos_prev, sin_prev = cos_sin_prev["c1a"] if self.compress_ratio == 1 else cos_sin_prev["comp"]
            cos_next, sin_next = cos_sin_next["c1a"] if self.compress_ratio == 1 else cos_sin_next["comp"]
            cos_q = torch.cat([cos_prev, cos_next], dim=0)
            sin_q = torch.cat([sin_prev, sin_next], dim=0)
            last_win_kv, x_with_pre_win = self.get_cp_window(x, attn_metadata)
            cos_prev, sin_prev = cos_sin_prev["c1a_with_pre_win"] if self.compress_ratio == 1 \
                else cos_sin_prev["comp_with_pre_win"]
            cos_next, sin_next = cos_sin_next["c1a_with_pre_win"] if self.compress_ratio == 1 \
                else cos_sin_next["comp_with_pre_win"]
            cos = torch.cat([cos_prev, cos_next], dim=0)
            sin = torch.cat([sin_prev, sin_next], dim=0)
            x = torch.cat(x_with_pre_win, dim=0)
        else:
            cos_sin = attn_metadata["cos_sin"]
            cos, sin = cos_sin["c1a"] if self.compress_ratio == 1 else cos_sin["comp"]
            cos_q, sin_q = cos, sin
        record_event(enable_multi_streams, self.mla_events, 0)
        with npu_stream_switch(enable_multi_streams, attn_metadata.get('mla_stream', None)):
            wait_event(enable_multi_streams, self.mla_events, 0)
            # win kv & topk_idxs
            kv = self.wkv(x_q if move_quant_before else x, dynamic_scale=x_scale)
            record_event(enable_multi_streams, self.mla_events, 2)
        qr, qr_scale = self.apply_norm_dynamic_quant(qr)
        wait_event(enable_multi_streams, self.mla_events, 2)
        record_event(enable_multi_streams, self.mla_events, 1)
        q = self.wq_b(qr, dynamic_scale=qr_scale).unflatten(-1, (self.num_heads_per_rank, self.head_dim))
        cur_stream = torch.npu.current_stream()
        with npu_stream_switch(enable_multi_streams, attn_metadata.get('mla_stream', None)):
            wait_event(enable_multi_streams, self.mla_events, 1)
            kv = self.kv_norm(kv)
            torch.ops.custom.inplace_partial_rotary_mul(
                kv.view(-1, 1, 1, self.head_dim), cos, sin,
                rotary_mode="interleave",
                partial_slice=self.partial_slice,
            )
            record_stream(enable_multi_streams, kv, cur_stream)
            record_event(enable_multi_streams, self.mla_events, 3)
        wait_event(enable_multi_streams, self.mla_events, 3)

        q = self.q_b_norm(q)
        torch.ops.custom.inplace_partial_rotary_mul(
            q.unsqueeze(2), cos_q, sin_q,
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
            decode_token_indices = attn_metadata["cp_metadata"]["decode_token_indices"]
            has_decode_requests = decode_token_indices.numel() > 0
            if not has_decode_requests:
                return q, qr, qr_scale, x

        win_cache = self.win_kv
        win_kv_slot_mapping = attn_metadata["slot_mapping"]["win_kv"]

        self.update_win_kv(kv, win_kv_slot_mapping, win_cache)
        return q, qr, qr_scale, x

    def mla_prolog_decode(
        self,
        x: torch.Tensor,
        attn_metadata: Optional[Dict] = None,
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

        if self.platform_version == PlatformVersion.A3:
            qa = self.wq_a(x)
        record_event(enable_multi_streams, self.mla_events, 0)
        record_stream(enable_multi_streams, x, attn_metadata.get('mla_stream', None))
        if self.platform_version != PlatformVersion.A3:
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
            torch.ops.custom.inplace_partial_rotary_mul(
                q.unsqueeze(2), cos, sin,
                rotary_mode="interleave",
                partial_slice=self.partial_slice,
            )
            # update kv cache in default stream can remove tensormove
            wait_event(enable_multi_streams, self.mla_events, 2)
            self.update_win_kv(kv, attn_metadata["slot_mapping"]["win_kv"], self.win_kv)
        return q, qr, qr_scale, x

    def mla_prolog(
        self,
        x: torch.Tensor,
        attn_metadata: Optional[Dict] = None,
        is_prefill: bool = True,
        **kwargs,
    ):
        if self.enable_pypto:
            return self.mla_prolog_pypto(x, attn_metadata, is_prefill, **kwargs)
        elif is_prefill:
            return self.mla_prolog_prefill(x, attn_metadata, is_prefill, **kwargs)
        else:
            return self.mla_prolog_decode(x, attn_metadata, is_prefill, **kwargs)

    def prepare_fa_kwargs(
        self,
        attn_metadata,
        q: torch.Tensor,
        cmp_sparse_indices: torch.Tensor = None,
        is_prefill: Optional[bool] = False
    ):
        if self.compress_ratio > 1:
            if attn_metadata.get("tmp_block_table", None): # not cp
                cmp_block_table = attn_metadata["tmp_block_table"][f'c{self.compress_ratio}a_cmp_kv']
            else:
                cmp_block_table = attn_metadata["block_table"][f'c{self.compress_ratio}a_cmp_kv']

        else:
            cmp_block_table = None
        win_cache = attn_metadata["full_kv_cache"] if is_prefill else self.win_kv
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
            "cmp_kv": self._get_runtime_cmp_cache(attn_metadata, is_prefill) if self.compress_ratio > 1 else None,
            "cmp_sparse_indices": cmp_sparse_indices,  # only for C4A
            "ori_block_table": attn_metadata["block_table"][win_block_table_str],
            "cmp_block_table": cmp_block_table,
            "sinks": self.attn_sink,
            "metadata": metadata, # get from operator sparse_attn_sharedkv_metadata for fa tiling
            "softmax_scale": self.softmax_scale,
        }
        if self.kv_cache_quant_mode == "float8" or self.kv_cache_quant_mode == "hifloat8":
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
        if self.kv_cache_quant_mode == "float8" or self.kv_cache_quant_mode == "hifloat8":
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
        is_prefill: bool = True,
        **kwargs
    ):
        if is_prefill and self.cp_size > 1:
            q_len = q.shape[0] // 2
            x = list(x.split([x.shape[0] - q_len - self.window_size, q_len + self.window_size], dim=0))

        enable_cmpr_stream = self.enable_compressor_parallel and not is_prefill
        enable_limit_core = self.enable_limit_core and not is_prefill

        # compressor for C4A and C128A
        # cmpr event 0 is recorded in self.mal_prolog function, after calling mla qb
        # cmpr event 1 is waited inside self.sparse_attn function, before calling sfa(c128a) / li qb dynamic_quant(c4a)
        if self.compress_ratio > 1:
            with npu_stream_switch(enable_cmpr_stream, attn_metadata.get('compressor_stream', None)):
                wait_event(enable_cmpr_stream, self.cmpr_events, 0)
                with limit_core_num(enable_limit_core, self.cmpr_aic_num, self.cmpr_aic_num * self.aiv_to_aic_ratio):
                    self.compressor(x, attn_metadata, is_prefill)
                record_event(enable_cmpr_stream, self.cmpr_events, 1)

        enable_metadata_stream = self.enable_multi_streams and not is_prefill
        wait_event(enable_metadata_stream, attn_metadata.get('metadata_event'), 1)

        # indexer for C4A
        # separate sfa compressor and li qb dynamic quant
        if self.compress_ratio == 4:
            # wait self.cmpr_events[1] in self.indexer
            topk_idxs = \
                self.indexer(x, qr, qr_scale, attn_metadata, enable_cmpr_stream, self.cmpr_events, 1, is_prefill)
        else:
            topk_idxs = None

        if self.compress_ratio > 1:
            wait_event(enable_cmpr_stream, self.cmpr_events, 1) # finish compressor before sfa

        if is_prefill and self.cp_size > 1:
            q_prev, q_next = q.split(q.shape[0] // 2, dim=0)
            if self.compress_ratio == 4:
                topk_idxs_prev, topk_idxs_next = topk_idxs
            else:
                topk_idxs_prev, topk_idxs_next = None, None
            o_prev = self.attn_kernel(q_prev, topk_idxs_prev, attn_metadata["prev"], is_prefill)
            o_next = self.attn_kernel(q_next, topk_idxs_next, attn_metadata["next"], is_prefill)
            o = torch.cat([o_prev, o_next], dim=0)
        else:
            o = self.attn_kernel(q, topk_idxs, attn_metadata, is_prefill)
        return o

    def attn_kernel(
        self,
        q: torch.Tensor,
        topk_idxs: torch.Tensor,
        attn_metadata: Optional[Dict] = None,
        is_prefill: bool = True,
    ):
        # get fa input dict
        fa_input_kwargs = self.prepare_fa_kwargs(
                attn_metadata,
                q,
                topk_idxs,
                is_prefill
        )
        o = self.sparse_attn_ops(**fa_input_kwargs)[0]
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
        num_tokens = o.shape[0]
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

        if self.mm_quant_mode == "w8a8hifloat8":
            o = partial_rotary_mul_quant(
                o.unsqueeze(2), cos, sin,
                partial_slice=self.partial_slice,
                platform_version=self.platform_version,
                origin_shape=o.shape,
            )
            o = o.view(num_tokens, self.num_groups_per_rank, -1)
        else:
            torch.ops.custom.inplace_partial_rotary_mul(
                o.unsqueeze(2), cos, sin,
                rotary_mode="interleave",
                partial_slice=self.partial_slice,
            )
            o = o.view(num_tokens, self.num_groups_per_rank, -1).to(torch.bfloat16)
        if self.oproj_tp_size > 1:
            # [num_tokens, tp_size, G/tp_size, ND/G] -> [tp_size, BS, G/tp_size, ND/G]
            o = o.view(num_tokens, self.oproj_tp_size, self.num_groups_per_rank // self.oproj_tp_size, -1)
            o = o.transpose(1, 0).contiguous().view(-1)
            all2all_output = torch.empty_like(o)
            dist.all_to_all_single(all2all_output, o,
                                   group=self.comm_manager.get_group("oproj_tp_group"))
            o = all2all_output.view(self.oproj_tp_size * num_tokens,
                                    self.num_groups_per_rank // self.oproj_tp_size, -1)

        # o_a_proj
        if self.mm_quant_mode == "w8a8mxfloat8":
            o, o_scale = torch_npu.npu_dynamic_mx_quant(o, dst_type=torch.float8_e4m3fn)
            o = torch_npu.npu_transpose_quant_batchmatmul(o, self.wo_a.weight, dtype=torch.bfloat16,
                                                            x1_scale=o_scale.view(torch.float8_e8m0fnu),
                                                            x2_scale=self.wo_a.weight_scale.view(torch.float8_e8m0fnu),
                                                            group_sizes=(0, 0, 32), perm_x1=(1, 0, 2),
                                                            perm_x2=(0, 1, 2), perm_y=(1, 0, 2))
        elif self.mm_quant_mode == "w8a8hifloat8":
            o = torch_npu.npu_transpose_quant_batchmatmul(o, self.wo_a.weight, dtype=torch_npu.hifloat8,
                                                            x1_scale=self.wo_a.x_scale, x2_scale=self.wo_a.weight_scale,
                                                            group_sizes=(0, 0, 0), perm_x1=(1, 0, 2),
                                                            perm_x2=(0, 1, 2), perm_y=(1, 0, 2), batch_split_factor=1,
                                                            x1_dtype=torch_npu.hifloat8, x2_dtype=torch_npu.hifloat8)
        else:
            if self.n_heads * self.head_dim // self.oproj_tp_size > MATMUL_MAX_AXIS_VALUE and \
               self.platform_version == PlatformVersion.A3:
                o = torch.matmul(o.transpose(0, 1), self.wo_a.weight).transpose(0, 1).contiguous()
            else:
                o = torch_npu.npu_transpose_batchmatmul(o, self.wo_a.weight, perm_x1=(1, 0, 2), perm_y=(1, 0, 2))
        if self.oproj_tp_size > 1:
            # [oproj_tp_size, num_tokens, num_groups_per_rank // oproj_tp_size * o_lora_rank]
            o = o.view(self.oproj_tp_size, num_tokens, -1)
        else:
            o = o.view(num_tokens, -1)

        # o_b_proj
        x = self.wo_b(o)

        if self.oproj_tp_size > 1:
            # [oproj_tp_size, num_tokens, dim] --> [oproj_tp_size * num_tokens, dim]
            x = x.view(self.oproj_tp_size * num_tokens, -1)
            reduce_scatter_output = torch.empty((num_tokens, x.shape[-1]), dtype=x.dtype, device=x.device)
            dist.reduce_scatter_tensor(reduce_scatter_output, x,
                                       group=self.comm_manager.get_group("oproj_tp_group"))
            x = reduce_scatter_output.view(num_tokens, x.shape[-1])
        return x

    def forward(
        self,
        x: torch.Tensor,
        attn_metadata: Optional[Dict] = None,
        is_prefill: bool = True,
        **kwargs,
    ):
        q, qr, qr_scale, x = self.mla_prolog(x, attn_metadata, is_prefill, **kwargs)

        # o TND
        o = self.sparse_attn(
            x, q, qr, qr_scale, attn_metadata, is_prefill, **kwargs)
        x = self.attn_post(o, attn_metadata, is_prefill)

        return x


class DeepseekV3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: DeepseekV3Config,
        infer_config: InferenceConfig,
        comm_manager: CommManager = None,
        layer_idx: int = 0,
        prefix: str = "",
        **kwargs,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.hidden_size = config.hidden_size
        self.attn = Attention(
            config=config,
            infer_config=self.infer_config,
            comm_manager=self.comm_manager,
            layer_idx=layer_idx,
            prefix=f"{prefix}.attn",
            **kwargs)

        self.ffn = (
            DeepseekV3MoE(
                config,
                self.infer_config,
                self.comm_manager,
                layer_idx=layer_idx,
                prefix=f"{prefix}.mlp",
                **kwargs,
            )
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

        self.cp_size = self.infer_config.parallel_config.cp_size
        self.global_rank = kwargs.get("global_rank")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: Optional[Dict] = None,
        past_residual: Optional[torch.Tensor] = None,
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
            shared_expert_stream=attn_metadata.get('shared_expert_stream', None),
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

    def __init__(
        self,
        config: DeepseekV3Config,
        infer_config: InferenceConfig,
        comm_manager: CommManager = None,
        prefix: str = "",
        **kwargs,
    ):
        super().__init__(config)
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.embed_tp_size = self.infer_config.parallel_config.embed_tp_size
        self.embed_dp_size = self.infer_config.parallel_config.embed_dp_size
        self.attn_tp_size = self.infer_config.parallel_config.attn_tp_size
        self.cp_size = self.infer_config.parallel_config.cp_size
        self.moe_ep_size = self.infer_config.parallel_config.moe_ep_size
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.vocab_size_per_rank = self.vocab_size // self.embed_tp_size
        self.global_rank = kwargs.get("global_rank")
        self.enable_superkernel = self.infer_config.model_config.custom_params.get("enable_superkernel", False)
        self.enable_multi_streams = self.infer_config.model_config.custom_params.get("enable_multi_streams", False)
        self.world_size = self.infer_config.parallel_config.world_size
        self.max_position_embeddings = get_max_position_embeddings(
            self.infer_config,
            kwargs.get("is_mtp", False),
        )

        is_mtp = kwargs.get("is_mtp")
        if not is_mtp:
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                self.padding_idx,
                torch.bfloat16,
                tp_size=self.embed_tp_size,
                tp_rank=self.comm_manager.get_rank("embed_tp_group") if self.embed_tp_size > 1 else 0)
            self.layers = nn.ModuleList(
                [
                    DeepseekV3DecoderLayer(
                        config,
                        self.infer_config,
                        self.comm_manager,
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

    def hc_head(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(1).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
        return y.to(dtype)

    def calc_input_embeddings(
        self,
        input_ids,
        is_prefill,
    ):
        num_tokens = input_ids.shape[0]
        cp_size = self.cp_size if is_prefill else 1
        attn_dp_size = self.world_size // self.attn_tp_size // cp_size
        if self.embed_tp_size > 1:
            embed_tp_group = self.comm_manager.get_group("embed_tp_group")
            if attn_dp_size > self.embed_dp_size:
                allgather_ratio = self.embed_tp_size // self.attn_tp_size
                if is_prefill:
                    local_num_tokens = num_tokens
                    max_num_tokens = torch.tensor([local_num_tokens], dtype=torch.long, device=input_ids.device)
                    dist.all_reduce(max_num_tokens, op=dist.ReduceOp.MAX, group=embed_tp_group)
                    max_num_tokens = int(max_num_tokens.item())

                    padded_input_ids = input_ids
                    if local_num_tokens < max_num_tokens:
                        padded_input_ids = torch.nn.functional.pad(
                            input_ids, (0, max_num_tokens - local_num_tokens), value=0
                        )
                    all_input_ids = input_ids.new_empty(max_num_tokens * allgather_ratio)
                    dist.all_gather_into_tensor(all_input_ids, padded_input_ids, group=embed_tp_group)
                else:
                    all_input_ids = input_ids.new_empty(num_tokens * allgather_ratio)
                    dist.all_gather_into_tensor(all_input_ids, input_ids, group=embed_tp_group)
                input_ids = all_input_ids

            new_input_ids = input_ids - (self.global_rank % self.embed_tp_size) * self.vocab_size_per_rank
            mask = (new_input_ids >= 0) & (new_input_ids < self.vocab_size_per_rank) # [T]
            new_input_ids_per_rank = new_input_ids * mask
            inputs_embeds = self.embed_tokens(new_input_ids_per_rank) * mask.unsqueeze(-1)

            if attn_dp_size <= self.embed_dp_size:
                dist.all_reduce(inputs_embeds, group=embed_tp_group)
            else:
                if is_prefill:
                    inputs_embeds_attn = inputs_embeds.new_empty(max_num_tokens, inputs_embeds.shape[-1])
                    dist.reduce_scatter_tensor(inputs_embeds_attn, inputs_embeds, group=embed_tp_group)
                    inputs_embeds = inputs_embeds_attn[:local_num_tokens]
                else:
                    inputs_embeds_attn = inputs_embeds.new_empty(num_tokens, inputs_embeds.shape[-1])
                    dist.reduce_scatter_tensor(inputs_embeds_attn, inputs_embeds, group=embed_tp_group)
                    inputs_embeds = inputs_embeds_attn
        else:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        return hidden_states

    def get_cp_input_ids(self, input_ids, attn_metadata):
        input_ids_split = torch.split(input_ids, attn_metadata["cp_metadata"]["split_list"], dim=0)
        cp_input_ids = torch.cat(
            [input_ids_split[i] for i in attn_metadata["cp_metadata"]["zigzag_idx"]], dim=0
        )
        return cp_input_ids

    def get_cp_hidden_states(self, hidden_states, attn_metadata):
        hidden_states_split = torch.split(hidden_states, attn_metadata["cp_metadata"]["split_list"], dim=0)
        hidden_states_cp = torch.cat(
            [hidden_states_split[i] for i in attn_metadata["cp_metadata"]["zigzag_idx"]], dim=0
        )
        return hidden_states_cp

    def update_cp_cos_sin(self, attn_metadata, hidden_states, kv_len, is_mtp=False):
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
            if not is_mtp:
                position_ids_cmp = attn_metadata[zigzag_flag]["position_ids_cmp_for_rope"]
                cos_sin.update({
                    "comp": self.compress_rotary_emb(
                        hidden_states, position_ids_cur, kv_len, self.max_position_embeddings),
                    "comp_with_pre_win": self.compress_rotary_emb(hidden_states, position_ids_with_pre_win,
                                                                  kv_len, self.max_position_embeddings),
                    "comp_last_win": self.compress_rotary_emb(hidden_states, position_ids_last_win,
                                                              kv_len, self.max_position_embeddings),
                    "c4a": self.compress_rotary_emb(
                        hidden_states, position_ids_cmp["4"], kv_len, self.max_position_embeddings),
                    "c128a": self.compress_rotary_emb(
                        hidden_states, position_ids_cmp["128"], kv_len, self.max_position_embeddings),
                })
                cos_sin.update({"comp_neg_sin": -cos_sin.get("comp")[1]})
            attn_metadata[zigzag_flag].update({
                "cos_sin": cos_sin,
            })

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
                "comp": self.compress_rotary_emb(
                    hidden_states, position_ids, kv_len, self.max_position_embeddings),
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
        position_ids: torch.Tensor,
        attn_metadata: Optional[Dict] = None,
        is_prefill: Optional[bool] = False,
        cur_topk_list: Optional[torch.Tensor] = None,
    ):
        if is_prefill and self.cp_size > 1 and self.embed_tp_size == 1:
            input_ids = self.get_cp_input_ids(input_ids, attn_metadata)

        inputs_embeds = self.calc_input_embeddings(input_ids, is_prefill)
        hidden_states = inputs_embeds
        if is_prefill and self.cp_size > 1 and self.embed_tp_size > 1:
            # Keep full input_ids for embedding TP, then switch hidden states to attention CP layout.
            input_ids = self.get_cp_input_ids(input_ids, attn_metadata)
            hidden_states = self.get_cp_hidden_states(inputs_embeds, attn_metadata)
            del inputs_embeds

        kv_len = attn_metadata["kv_len"]
        if is_prefill and self.cp_size > 1:
            self.update_cp_cos_sin(attn_metadata, hidden_states, kv_len)
        else:
            cos_sin = self.generate_cos_sin(attn_metadata, hidden_states)
            attn_metadata.update({'cos_sin': cos_sin})

        residual = None

        label = f'decode_layer'
        if self.enable_multi_streams:
            option = "stream-fusion=1"
        else:
            option = "option_xxx2"

        # mhc
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)

        with superkernel_scope(self.enable_superkernel and not is_prefill, label, option):
            for decoder_layer in self.layers:
                hidden_states = decoder_layer(
                    hidden_states,
                    attn_metadata=attn_metadata,
                    past_residual=residual,
                    is_prefill=is_prefill,
                    cur_topk_list=cur_topk_list,
                    input_ids=input_ids,
                )
        hidden_states = self.hc_head(hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class DeepseekV3ModelMTPLayer(DeepseekV3Model):
    def __init__(
        self,
        config: DeepseekV3Config,
        infer_config: InferenceConfig,
        comm_manager: CommManager = None,
        layer_idx: int = 0,
        prefix: str = "",
        **kwargs,
    ):
        super().__init__(config, infer_config, comm_manager, prefix=prefix, **kwargs)
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.layers = nn.ModuleDict(
            {
                str(self.mtp_start_layer_idx + i):
                DeepseekV3DecoderLayer(
                    config,
                    self.infer_config,
                    self.comm_manager,
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
        cur_topk_list: Optional[torch.Tensor] = None,
        mtp_layer_idx: Optional[int] = 0,
        input_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # mhc
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        hidden_states = self.layers[str(self.mtp_start_layer_idx + mtp_layer_idx)](
            hidden_states,
            attn_metadata=attn_metadata,
            past_residual=past_residual,
            is_prefill=is_prefill,
            cur_topk_list=cur_topk_list,
            input_ids=input_ids,
        )
        hidden_states = self.hc_head(hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
        return hidden_states


class DeepseekV3ForCausalLM(DeepseekV3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    @staticmethod
    def update_model_cfg(config, infer_config: InferenceConfig):
        if config.compress_ratios is not None:
            # When compress_ratio == 0, set compress_ratio to 1 for convenience.
            for i, val in enumerate(config.compress_ratios):
                if val == 0:
                    config.compress_ratios[i] = 1

    def __init__(
        self,
        config,
        infer_config: InferenceConfig,
        comm_manager: CommManager = None,
        is_mtp=False,
        prefix: str = "",
    ):
        super().__init__(config)
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.input_max_len = self.infer_config.data_config.input_truncated_len
        self.platform_version = self.infer_config.model_config.platform_version
        self.get_parallel_settings()
        self.experts_per_rank = config.n_routed_experts // self.moe_ep_size
        self.top_k = config.num_experts_per_tok
        self.max_position_embeddings = get_max_position_embeddings(self.infer_config, is_mtp)
        self.force_eplb = self.infer_config.model_config.force_eplb
        self.num_experts_per_tok = config.num_experts_per_tok
        # total experts num
        self.num_experts = config.n_routed_experts
        self.is_mtp = is_mtp
        self.mm_quant_mode = (
            config.quant_config.mm_quant_mode
            if config.quant_config is not None
            else "w16a16")
        self.update_kv_quant_settings()
        self.update_gmm_quant_mode()
        self.kv_cache_quant_mode = config.quant_config.kv_cache_quant_mode
        self.li_cache_quant_mode = config.quant_config.li_cache_quant_mode
        self.attention_data = AttnMetaData(self.config, comm_manager, self.infer_config, is_mtp)
        self.enable_cache_compile = self.infer_config.model_config.enable_cache_compile

        self.enable_static_kernel = self.infer_config.model_config.enable_static_kernel
        self.enable_npugraph_ex = self.infer_config.model_config.exe_mode == "npugraph_ex"
        self.enable_multi_streams = self.infer_config.model_config.custom_params.get("enable_multi_streams", False)
        self.metadata_event = []
        if self.enable_multi_streams:
            self.metadata_event = [torch.npu.Event(), torch.npu.Event()]

        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.rank_offset = int(os.getenv("RANK_OFFSET", "0"))
        self.global_rank = self.local_rank + self.rank_offset
        self.world_size = self.infer_config.parallel_config.world_size
        kwargs = {
                    "global_rank": self.global_rank,
                    "is_mtp": is_mtp
                }
        self.init_parallel_comm_group()
        self.batch_size_per_rank = self.infer_config.scheduler_config.batch_size_per_dp_rank

        mtp_layer_idx = config.num_hidden_layers # MTP is the last layer
        self.model = DeepseekV3ModelMTPLayer(
            config, self.infer_config, self.comm_manager, mtp_layer_idx, prefix, **kwargs
        ) if is_mtp else DeepseekV3Model(
            config, self.infer_config, self.comm_manager, prefix, **kwargs
        )
        self.vocab_size = config.vocab_size
        self.rope_head_dim = config.qk_rope_head_dim
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
        self.block_size = self.infer_config.scheduler_config.block_size
        self.sas_metadata_ops = torch.ops.custom.npu_sparse_attn_sharedkv_metadata
        if self.kv_cache_quant_mode == "float8" or self.kv_cache_quant_mode == "hifloat8":
            self.sas_metadata_ops = torch.ops.custom.npu_kv_quant_sparse_attn_sharedkv_metadata
        self.window_size = config.sliding_window
        self.cp_segment_min_len = self.window_size
        self.init_cache_dim()
        self.first_layer_idx = mtp_layer_idx if is_mtp else 0
        self.first_layer_ratio = self.config.compress_ratios[self.first_layer_idx]

    def check_model_settings(self):
        model_config = self.infer_config.model_config
        custom_params = model_config.custom_params
        parallel_config = self.infer_config.parallel_config
        scheduler_config = self.infer_config.scheduler_config


        exe_mode = model_config.exe_mode
        enable_cache_compile = model_config.enable_cache_compile
        enable_multi_streams = custom_params.get("enable_multi_streams", False)
        enable_limit_core = custom_params.get("enable_limit_core", False)
        platform_version = model_config.platform_version
        enable_superkernel = custom_params.get("enable_superkernel", False)
        enable_pypto = custom_params.get("enable_pypto", False)
        next_n = model_config.next_n
        with_ckpt = model_config.with_ckpt

        if not with_ckpt and not model_config.force_eplb:
            raise ValueError(f"{model_config.force_eplb=} must be True if {with_ckpt =}!")

        if exe_mode not in ["eager", "npugraph_ex"]:
            raise ValueError(f"{exe_mode=} does not supported!")
        if parallel_config.attn_tp_size > 1:
            raise ValueError(f"{parallel_config.attn_tp_size=} is not supported yet!")

        dynamo_feat = enable_cache_compile or enable_superkernel
        if exe_mode == "eager" and dynamo_feat:
            raise ValueError(f"{exe_mode=} does not support cache compile or superkernel!")
        if exe_mode == "eager" and enable_multi_streams:
            logger.info(
                "When using eager execution mode, enable-multi-streams only takes effect during the prefill phase.")
        if enable_limit_core and not enable_multi_streams:
            raise ValueError(f"{enable_limit_core=} only if enable_multi_streams!")
        if enable_limit_core and platform_version != PlatformVersion.A3:
            raise ValueError(f"{enable_limit_core=} only supports platform A3!")
        if enable_limit_core and enable_pypto:
            raise ValueError(f"{enable_pypto=} does not support {enable_limit_core=}!")
        if next_n > 3:
            raise ValueError(f"{next_n=} must equal or smaller than 3")

        if parallel_config.cp_size > 1 and scheduler_config.cp_mini_batch != 1:
            raise ValueError(f"when cp enabled, {scheduler_config.cp_mini_batch=} should be 1")

        model_config.enable_weight_nz = platform_version != PlatformVersion.ASCEND_950
        self.update_op_kernel_dict()

    def update_op_kernel_dict(self):
        """
        kernel_config: Dict, op impls defined by user, {op_type: op_impls}
        OpKernel.OP_TYPE: List, contain ops with different impls
        OpKernel.KERNEL_MAP: Dict, contain op_impls {op_impl_name: op_impl}
        """
        # import all mudules under models.modules.op_impls files
        auto_import_modules(f"{__package__}.modules.op_impls")
        custom_params = self.infer_config.model_config.custom_params
        kernel_config = custom_params.get("kernel_config", {})
        platform_version = self.infer_config.model_config.platform_version.value.lower()
        enable_pypto = custom_params.get("enable_pypto", False)
        if enable_pypto and platform_version != "a3":
            raise ValueError(f"PYPTO kernel for this model on {platform_version=} is not supported yet.")

        for op_type in OpKernel.OP_TYPE:
            if op_type in kernel_config:
                kernel_impl = kernel_config[op_type]
                used_kernel = f"{op_type}_{kernel_impl}_{platform_version}"
            else:
                default_kernel = f"{op_type}_ascendc_{platform_version}"
                if default_kernel in OpKernel.KERNEL_MAP:
                    used_kernel = default_kernel
                else:
                    used_kernel = f"{op_type}_native_{platform_version}"
            if enable_pypto:
                pypto_kernel = f"{op_type}_pypto_a3"
                if pypto_kernel in OpKernel.KERNEL_MAP:
                    used_kernel = pypto_kernel
            OpKernel.op_impl_apply(op_type, used_kernel)
            logger.info("%s use impl %s", op_type, used_kernel)

    def process_weights_after_loading(self):
        """
        Do weight transpose, format cast to NZ, and scale dtype cast after loading weights.
        """
        float_scales_map = [
            "gate_up_proj",
            "q_b_proj",
            "wq_b",
        ]
        float_smooth_scales_map = [
            "down_proj"
        ]
        enable_weight_nz = self.infer_config.model_config.enable_weight_nz

        for module_name, module in self.named_modules():
            if "wo_a" in module_name:
                config = self.config
                head_dim_per_group = config.num_attention_heads * config.head_dim // config.o_groups
                module.weight.data = module.weight.data.view(-1, config.o_lora_rank, head_dim_per_group) \
                                                   .transpose(1, 2).contiguous()
                if config.quant_config.mm_quant_mode == "w8a8mxfloat8":
                    scale_data = reshape_mx_scale(module.weight_scale.data)
                    module.weight_scale.data = scale_data.view(-1, config.o_lora_rank, *scale_data.shape[1:]) \
                                                   .transpose(1, 2).contiguous()
                elif config.quant_config.mm_quant_mode == "w8a8hifloat8":
                    scale_data = torch_npu.npu_trans_quant_param(module.weight_scale.data.to(torch.float32).npu())
                    module.weight_scale.data = scale_data.view(-1, config.o_lora_rank, *scale_data.shape[1:]) \
                                                   .transpose(1, 2)[0, 0].contiguous()
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

            is_wq_b_transpose = self.config.num_attention_heads * self.config.head_dim > MATMUL_MAX_AXIS_VALUE
            is_weight_nz = False if "compressor" in module_name or (is_wq_b_transpose and "attn.wq_b" in module_name) \
                                 else enable_weight_nz
            is_transpose = False if "compressor" in module_name else True
            if isinstance(quant_method, QuantizeMethodBase):
                quant_method.process_weights_after_loading(
                    module,
                    is_nz=is_weight_nz,
                    is_transpose=is_transpose,
                    scales_dtype=scales_dtype,
                )

            moe_quant_methods = (
                CompressedTensorW8A8Int8MoEGMMMethod,
                CompressedTensorW4A8Int8MoEGMMMethod,
                Fp8MoEGMMMethod,
            )
            if isinstance(quant_method, moe_quant_methods) and self.moe_ep_size > 1:
                all_experts_smooth_scale = module.smooth_scale_1.data.new_empty(
                    module.smooth_scale_1.data.shape[0] * self.moe_ep_size,
                    module.smooth_scale_1.data.shape[1],
                )
                dist.all_gather_into_tensor(
                    all_experts_smooth_scale,
                    module.smooth_scale_1.data,
                    group=self.comm_manager.get_group("moe_ep_group"),
                )
                module.smooth_scale_1.data = all_experts_smooth_scale

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
        if self.platform_version == PlatformVersion.ASCEND_950 and "hifloat" in self.mm_quant_mode:
            self.config.quant_config.kv_cache_quant_mode = "hifloat8"
            self.config.quant_config.li_cache_quant_mode = "hifloat8"
        elif self.platform_version == PlatformVersion.ASCEND_950 and "float" in self.mm_quant_mode:
            self.config.quant_config.kv_cache_quant_mode = "float8"
            self.config.quant_config.li_cache_quant_mode = "float8"
        else:
            self.config.quant_config.li_cache_quant_mode = "int8"

    def update_gmm_quant_mode(self):
        if self.platform_version == PlatformVersion.ASCEND_950 and "w4" in self.config.quant_config.gmm_quant_mode \
            and "mx" not in self.config.quant_config.gmm_quant_mode:
            self.config.quant_config.gmm_quant_mode = \
                self.config.quant_config.gmm_quant_mode.replace("float", "mxfloat")

    def get_parallel_settings(self):
        self.embed_tp_size = self.infer_config.parallel_config.embed_tp_size
        self.attn_dp_size = self.infer_config.parallel_config.attn_dp_size
        self.attn_tp_size = self.infer_config.parallel_config.attn_tp_size
        self.oproj_tp_size = self.infer_config.parallel_config.o_proj_tp_size
        self.cp_size = self.infer_config.parallel_config.cp_size
        self.moe_ep_size = self.infer_config.parallel_config.moe_ep_size
        self.moe_tp_size = self.infer_config.parallel_config.moe_tp_size
        self.lmhead_tp_size = self.infer_config.parallel_config.lmhead_tp_size
        self.moe_dp_size = self.infer_config.parallel_config.world_size // self.moe_tp_size
        self.embed_dp_size = self.infer_config.parallel_config.embed_dp_size

    def init_parallel_comm_group(self):
        if self.comm_manager is None:
            raise ValueError("DeepseekV3ForCausalLM requires comm_manager to initialize communication groups.")

        world_size = self.world_size
        platform_version = self.platform_version
        self.comm_manager.register_group(
            name="attn_tp_group",
            group_num=self.world_size // self.attn_tp_size,
            group_size=self.attn_tp_size,
            platform_version=platform_version,
        )
        self.comm_manager.register_group(
            name="oproj_tp_group",
            group_num=world_size // self.oproj_tp_size,
            group_size=self.oproj_tp_size,
            platform_version=platform_version,
        )
        self.comm_manager.register_group(
            name="embed_tp_group",
            group_num=self.embed_dp_size,
            group_size=self.embed_tp_size,
            platform_version=platform_version,
        )
        self.comm_manager.register_group(
            name="lmhead_tp_group",
            group_num=world_size // self.lmhead_tp_size,
            group_size=self.lmhead_tp_size,
            platform_version=platform_version,
        )
        
        self.comm_manager.register_group(
            name="moe_tp_group",
            group_num=self.moe_dp_size,
            group_size=world_size // self.moe_dp_size,
            platform_version=platform_version,
        )

        moe_ep_group_type = None if self.platform_version != PlatformVersion.ASCEND_950 else 0
        # 950 use default group for prefill moe
        self.comm_manager.register_group(
            name="moe_ep_group",
            group_num=self.moe_tp_size,
            group_size=world_size // self.moe_tp_size,
            group_stride=self.moe_tp_size,
            group_type=moe_ep_group_type,
            platform_version=platform_version,
        )

        # used for fullmesh v2
        moe_ep_mc2_group_type = None if self.platform_version != PlatformVersion.ASCEND_950 else 3
        # 950 use aiv group for mc2
        is_full_mesh_v2 = self.platform_version != PlatformVersion.ASCEND_950
        hccl_buffer_size = calc_moe_hccl_buffer_size(
            self.infer_config, self.config, is_full_mesh_v2=is_full_mesh_v2
        )
        self.comm_manager.register_group(
            name="moe_ep_group_mc2",
            group_num=self.moe_tp_size,
            group_size=self.moe_ep_size,
            group_stride=self.moe_tp_size,
            return_name=True,
            allow_physical_reuse=False,
            hccl_buffer_size=hccl_buffer_size,
            group_type=moe_ep_mc2_group_type,
            platform_version=platform_version,
        )
        self.comm_manager.register_group(
            name="cp_group",
            group_num=world_size // self.cp_size,
            group_size=self.cp_size,
            group_stride=1,
            platform_version=platform_version,
        )

    def gather_cp_last_token_hidden(self, outputs, attn_metadata):
        num_tokens, hidden_size = outputs.shape
        cp_metadata = attn_metadata["cp_metadata"]
        segment_len = num_tokens // 2
        last_segment_idx = cp_metadata["last_rank"]
        if last_segment_idx < 0:
            raise ValueError("CP prefill requires at least one valid token")
        last_segment_len = cp_metadata["split_kv_len"][last_segment_idx].item()

        local_offset = last_segment_len - 1
        if cp_metadata["last_rank_flag"] == "next":
            local_offset += segment_len

        last_hidden = outputs.new_zeros((1, hidden_size))
        if self.global_rank == cp_metadata["last_rank_zz"]:
            last_hidden.copy_(outputs[local_offset: local_offset + 1, :])
        # Only the owner rank writes data; sum broadcasts the selected hidden state across the CP group.
        dist.all_reduce(last_hidden, group=self.comm_manager.get_group("cp_group"))
        return last_hidden

    def forward_lm_head(self, outputs, kv_len, is_prefill=True, attn_metadata=None):
        num_tokens, hidden_size = outputs.shape
        seq_used_q = attn_metadata.get("seq_used_q")
        bs = seq_used_q.numel()
        if is_prefill:
            if self.cp_size > 1:
                outputs = self.gather_cp_last_token_hidden(outputs, attn_metadata)
            else:
                gather_index = kv_len - 1
                gather_index = gather_index.unsqueeze(1).repeat(1, outputs.shape[-1])
                outputs = torch.gather(outputs, 0, gather_index).view(-1, 1, hidden_size)
            q_len = 1 # prefill takes the last token
        else: # combine bs and q_len axes for lm_head
            outputs = outputs.view(num_tokens, 1, hidden_size)
            q_len = num_tokens // bs
        if (self.attn_dp_size == 1) or (self.lmhead_tp_size == 1):
            hidden_states = outputs
        else:
            # allgather: (bs / attn_dp, hidden_size) -> (bs, hidden_size)
            hidden_states = torch.empty_like(outputs).repeat(self.lmhead_tp_size, 1, 1)
            dist.all_gather_into_tensor(hidden_states, outputs, group=self.comm_manager.get_group("lmhead_tp_group"))

        logits = self.lm_head(hidden_states) # (lmhead_tp_size * bs / attn_dp, 1, vocab_size / lmhead_tp_size)
        if self.lmhead_tp_size > 1: # -> (bs / attn_dp, 1, vocab_size)
            if self.attn_dp_size == 1:
                new_logits = torch.empty_like(logits).repeat(self.lmhead_tp_size, 1, 1)
                dist.all_gather_into_tensor(new_logits, logits, group=self.comm_manager.get_group("lmhead_tp_group"))
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
        if self.kv_cache_quant_mode == "float8" or self.kv_cache_quant_mode == "hifloat8":
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
        if self.kv_cache_quant_mode == "float8" or self.kv_cache_quant_mode == "hifloat8":
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
        record_event(enable_metadata_multi_streams, attn_metadata.get('metadata_event'), 0)
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
                record_event(enable_metadata_multi_streams and self.first_layer_ratio == 128,
                             attn_metadata.get('metadata_event'), 1)
                self.generate_metadata(attn_metadata, c4a_metadata_kwargs, is_prefill, "c4a_metadata")
                if self.li_cache_quant_mode in ["int8", "float8", "hifloat8"]:
                    li_metadata_kwargs = self.generate_li_metadata_kwargs()
                    self.generate_metadata(attn_metadata, li_metadata_kwargs,\
                                            is_prefill, "lightning_indexer_quant", is_li=True)

    def preprocess_model_inputs(self, model_inputs: Dict, is_prefill=False, is_mtp=False, **kwargs):
        model_inputs = dict(model_inputs)

        attn_metadata = self.attention_data.build_attn_metadata(
            model_inputs.get("input_ids"),
            model_inputs.get("position_ids"),
            model_inputs.get("forward_metadata"),
        )
        attn_metadata.update({"metadata_event": self.metadata_event})
        model_inputs["attn_metadata"] = attn_metadata

        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor = None,
        forward_metadata: ForwardMetaData = None,
        attn_metadata: Optional[Dict] = None,
        cur_topk_list: Optional[torch.Tensor] = None,
        **kwargs
    ):
        is_prefill = forward_metadata.is_prefill

        self.generate_kernel_metadata(attn_metadata, is_prefill)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attn_metadata=attn_metadata,
            is_prefill=is_prefill,
            cur_topk_list=cur_topk_list,
        ) # (num_tokens, hidden_size)

        prev_hidden_states = outputs

        logits = self.forward_lm_head(outputs, attn_metadata["actual_seq_q"], is_prefill, attn_metadata)
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

    def mtp_decode(
        self,
        **kwargs
    ):
        logits, prev_hidden_states = self.forward(
            is_prefill=False,
            **kwargs
        )
        return logits, prev_hidden_states

    def get_cache_info(
        self,
    ) -> ModelCacheInfo:
        layers = self.model.layers.values() if self.is_mtp else self.model.layers
        layer_infos = []
        for layer_idx, layer in enumerate(layers):
            layer_infos.append(
                LayerCacheInfo(
                    layer_idx=layer_idx,
                    caches=list(layer.attn.cache_entries),
                )
            )

        return ModelCacheInfo(
            num_layers=len(layer_infos),
            layer_infos=layer_infos,
            is_mla_backend=True,
        )

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
        is_replace_expert_scale_name = any("w13_weight_scale" in key for key in params_dict)
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
                    if is_replace_expert_scale_name:
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
                    # The npu_transpose_batchmatmul op doesn't support the fp8 data type. The weight of wo_a needs
                    # to be converted to bf16.
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
        key_weights = {"smooth_scale", "q_b_norm", "w2_alpha"}
        if weights_not_loaded:
            if all(any(key in name for key in key_weights) for name in weights_not_loaded):
                logger.warning(
                    "Smooth scales were not initialized from checkpoint.")
            else:
                raise ValueError(
                    "Following weights were not initialized from "
                    f"checkpoint: {weights_not_loaded}")
        return loaded_params


class DeepseekV3ModelMTP(DeepseekV3ForCausalLM):

    def __init__(
        self,
        config: DeepseekV3Config,
        infer_config: InferenceConfig,
        comm_manager: CommManager = None,
        **kwargs,
    ):
        super().__init__(config, infer_config, comm_manager, is_mtp=True)
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.rank_offset = int(os.getenv("RANK_OFFSET", "0"))
        self.global_rank = self.local_rank + self.rank_offset
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.vocab_size_per_rank = self.vocab_size // self.embed_tp_size
        self.ignore_share_weight = True
        self.mm_quant_mode = (
            config.quant_config.mm_quant_mode
            if config.quant_config is not None
            else "w16a16")

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
        position_ids: torch.LongTensor = None,
        forward_metadata: ForwardMetaData = None,
        prev_hidden_states: Optional[torch.Tensor] = None,
        attn_metadata: Optional[Dict] = None,
        cur_topk_list: Optional[torch.Tensor] = None,
        **kwargs
    ):
        is_prefill = forward_metadata.is_prefill
        model_input_ids = input_ids

        position_ids = attn_metadata["position_ids"]
        kv_len = attn_metadata["kv_len"]
        if not attn_metadata.get("kernel_metadata"):
            self.generate_kernel_metadata(attn_metadata, is_prefill, True)

        if is_prefill and self.cp_size > 1 and self.embed_tp_size == 1:
            input_ids = self.model.get_cp_input_ids(input_ids, attn_metadata)

        inputs_embeds = self.model.calc_input_embeddings(input_ids, is_prefill)
        hidden_states = inputs_embeds
        if is_prefill and self.cp_size > 1 and self.embed_tp_size > 1:
            # Preserve embedding TP correctness, then use local attention CP hidden states for MTP.
            hidden_states = self.model.get_cp_hidden_states(inputs_embeds, attn_metadata)
            del inputs_embeds

        kv_len = attn_metadata["kv_len"]
        if is_prefill and self.cp_size > 1:
            self.model.update_cp_cos_sin(attn_metadata, hidden_states, kv_len, is_mtp=True)
        else:
            cos_sin = self.model.generate_cos_sin(attn_metadata, hidden_states, is_mtp=True)
            attn_metadata.update({'cos_sin': cos_sin})
        residual = None


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
            cur_topk_list=cur_topk_list,
        )

        outputs = self.shared_head_norm(hidden_states)
        prev_hidden_states = outputs

        logits = self.forward_lm_head(
            outputs=outputs, kv_len=attn_metadata["actual_seq_q"], is_prefill=is_prefill, \
            attn_metadata=attn_metadata)

        return logits, prev_hidden_states

    def decode(self, **kwargs):
        return self.forward(**kwargs)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping, mtp_unique_weight_mapping, expert_params_mapping, repeat_loaded_weights_mapping \
            = self._load_weight_map()

        params_dict = dict(self.named_parameters())
        params_dict = adapt_safetensors_field_mtp(params_dict, self.config.num_hidden_layers)
        loaded_params: Set[str] = set()
        dequant_cache = {}
        is_replace_expert_scale_name = any("w13_weight_scale" in key for key in params_dict)
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
                    if is_replace_expert_scale_name:
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
        key_weights = {"smooth_scale", "q_b_norm", "w2_alpha"}
        if weights_not_loaded:
            if all(any(key in name for key in key_weights) for name in weights_not_loaded):
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
