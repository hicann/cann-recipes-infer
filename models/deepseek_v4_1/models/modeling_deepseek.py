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
from enum import Enum, auto
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from torch import nn
import torch.distributed as dist

import torch_npu
import torchair as tng
import cann_ops_transformer
import cann_ops_nn
from transformers.cache_utils import Cache
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)

from executor.utils import (
    override, get_init_attn_mask, calc_moe_hccl_buffer_size,
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
from module.utils import get_moe_num_chunks, split_moe_tensors
from module.quantization.utils.quant_utils import reshape_mx_scale
from module.quantization import QuantizeMethodBase
from module.quantization.fp8 import Fp8PerTileMoEGMMMethod
from .configuration_deepseek import DeepseekV3Config
from .modules import (get_window_topk_idxs, get_compress_topk_idxs,
                      one_hot, yarn_get_mscale,
                      DeepseekV3RMSNorm, _init_rope, DEEPSEEKV3_START_DOCSTRING,
                      DEEPSEEKV3_INPUTS_DOCSTRING, DeepseekV3PreTrainedModel, apply_rotary_emb,
                      AttnMetaData, PACKED_KV_STORAGE_DTYPE,
                      PACKED_KV_COMPUTE_DTYPE, get_kv_cache_dim, is_packed_kv_layout
                    )
from .modules import Indexer, Compressor, Engram, EngramLayout, NgramHashState
from .modules.registry import OpKernel, auto_import_modules
from .modeling_deepseek_vision import DeepseekV41VisionModel
from .modules.op_impls.mhc import make_identity_pre_mix, hc_pre_mix
logger = logging.get_logger(__name__)

HADAMARD_SIZE = 128
MATMUL_MAX_AXIS_VALUE = 65535


class KVCache:
    """Per-layer view of the model-owned KV resources."""
    def __init__(self, layer_idx, source_layer=None):
        self.layer_idx = layer_idx
        self.source_layer = source_layer
        self.is_source = source_layer is not None and layer_idx == source_layer
        self.shared_idx = None
        self.candidates = None
        self.cmp_cache = None
        self.indexer_cache = None
        self.win_kv = None
        self.state_cache = None


class _Scope(Enum):
    PER_LAYER = auto()
    SHARED = auto()
    SHARED_HEAD = auto()


def build_cache_mapping(kv_source_layers, compress_ratios, num_layers):
    """Build the source/consumer topology used by shared KV caches.

    A source layer owns one compressed cache and serves the consecutive layers
    up to (but excluding) the next source. Layers before the first source are
    left unmapped because they do not participate in the shared path. The
    returned forward map is used for O(1) per-layer lookup, while the reverse
    map is used by cache setters to fan one allocated tensor out to consumers.
    """
    # Every model layer must have a corresponding compression ratio so that
    # source-group validation cannot silently read an unrelated index.
    if len(compress_ratios) < num_layers:
        raise ValueError("compress_ratios length must be at least num_layers")

    # Normalize and sort source declarations. Duplicate source IDs do not
    # create additional cache groups.
    sources = sorted({int(x) for x in (kv_source_layers or [])})
    if sources and (sources[0] < 0 or sources[-1] >= num_layers):
        raise ValueError("kv_source_layers contains an out-of-range layer")

    layer_to_source = {}
    source_to_layers = {source: [] for source in sources}
    for layer_idx in range(num_layers):
        # The latest source at or before this layer is its cache provider.
        eligible = [source for source in sources if source <= layer_idx]
        if eligible:
            source = eligible[-1]
            layer_to_source[layer_idx] = source
            source_to_layers[source].append(layer_idx)

    for source, group in source_to_layers.items():
        # Shared NPU tensors require one contiguous execution interval, and a
        # source must always be the first layer in its own interval.
        if not group or group[0] != source or group != list(range(group[0], group[-1] + 1)):
            raise ValueError(f"source layer {source} has a non-contiguous group")

        # All consumers of a source must use the same compression layout.
        ratio = compress_ratios[source]
        if any(compress_ratios[i] != ratio for i in group):
            raise ValueError(f"source layer {source} group has mismatched compress_ratios")
    return layer_to_source, source_to_layers


def get_max_position_embeddings(infer_config: InferenceConfig):
    return infer_config.data_config.input_truncated_len + infer_config.scheduler_config.max_new_tokens


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
        if "float8" in self.mm_quant_mode and "a8" in self.mm_quant_mode:
            self.forward = self.forward_a8float8
        else:
            self.forward = self.forward_normal

    def forward_normal(self, x, enable_decode_stream=False, shared_expert_event=None):
        merged_x = self.gate_up_proj(x)
        intermediate_hidden_states = torch_npu.npu_swiglu(merged_x)
        wait_event(enable_decode_stream, shared_expert_event, 0)
        return self.down_proj(intermediate_hidden_states)

    def forward_a8float8(self, x, enable_decode_stream=False, shared_expert_event=None):
        merged_x = self.gate_up_proj(x)
        swiglu_limit_args = {}
        if self.swiglu_limit is not None:
            swiglu_limit_args["clamp_limit"] = self.swiglu_limit
        intermediate_hidden_states, pergroup_scale, _ = torch.ops.cann_ops_nn.swiglu_group_quant(
            merged_x,
            dst_type=torch.float8_e4m3fn,
            round_scale=True if "mx" in self.mm_quant_mode else False,
            # 1: dynamic quantization; 0: static quantization.
            quant_mode=1 if "mx" in self.mm_quant_mode else 0,
            **swiglu_limit_args,
            )
        wait_event(enable_decode_stream, shared_expert_event, 0)
        return self.down_proj(intermediate_hidden_states, pergroup_scale)


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
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.gmm_quant_mode = config.quant_config.gmm_quant_mode
        self.swiglu_limit = config.swiglu_limit if hasattr(config, "swiglu_limit") else None
        self.hidden_dim = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.moe_tp_size = self.infer_config.parallel_config.moe_tp_size
        self.moe_ep_size = self.infer_config.parallel_config.moe_ep_size
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
            bias=False,
            quant_config=config.quant_config,
            tp_size=self.moe_tp_size,
            tp_rank=self.comm_manager.get_rank("moe_tp_group") if self.moe_tp_size > 1 else 0,
            ep_size=self.moe_ep_size,
            ep_rank=self.comm_manager.get_rank("moe_ep_group") if self.moe_ep_size > 1 else 0,
            prefix=f"{prefix}.experts",
        )
        self.moe_ffn = self.experts
        self._init_gate(prefix)
        self.use_native_gate_topk = False
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
                                     params_dtype=torch.bfloat16,
                                     prefix=f"{prefix}.gate",
                                     )
        self._reset_parameters()
        enable_vision = (
            getattr(self.config, "vision_n_layers", 0) > 0
            and self.infer_config.disagg_config.disaggregation_mode != "DECODE"
        )
        self.gate.bias_vl = (
            nn.Parameter(torch.empty(self.n_routed_experts, dtype=torch.float32))
            if enable_vision else None
        )
        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(self.n_routed_experts, dtype=torch.float32)
        )

    def _reset_parameters(self) -> None:
        pass

    def set_mc2_kwargs(self):
        global_rank = dist.get_rank()
        moe_ep_group_name = self.comm_manager.get_group_name("moe_ep_group_mc2")
        if self.gmm_quant_mode not in self.dispatch_quant_mode:
            quant_mode = self.dispatch_quant_mode["w16a16"]
        else:
            quant_mode = self.dispatch_quant_mode[self.gmm_quant_mode]
        enable_smooth_scale = False
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

    def forward(self, hidden_states, is_prefill=False, cur_topk_list=None, input_ids=None, image_mask=None,
                shared_expert_stream=None,
                prefill_moe_global_chunks=None):
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
        topk_idx, topk_weight, _ = OpKernel.gate_topk(self, logits, input_ids, image_mask)
        if self.force_eplb:
            topk_idx = cur_topk_list
        topk_idx = topk_idx.to(torch.int32)

        # MOE EP
        if is_prefill:
            return self.moe_infer_double_routing(
                hidden_states, topk_idx, topk_weight, hidden_states_share, prefill_moe_global_chunks)
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
            "enable_cann_ops_nn": True,
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
                gathered_pertoken_scale = pertoken_scale.new_empty(gathered_tokens.shape[0], *pertoken_scale.shape[1:])
            else:
                gathered_pertoken_scale = pertoken_scale.new_empty(gathered_tokens.shape[0])
        if "a16" not in self.gmm_quant_mode:
            dist.all_to_all_single(gathered_pertoken_scale, \
                                   pertoken_scale, output_splits, input_splits, group=moe_ep_group)
        if "mxfloat" in self.gmm_quant_mode:
            gathered_pertoken_scale = gathered_pertoken_scale.view(torch.float8_e8m0fnu)
        return tokens_per_expert_group, gathered_tokens, gathered_pertoken_scale, input_splits, output_splits

    def moe_infer_double_routing(self, x, topk_ids, topk_weight, hidden_states_share, prefill_moe_global_chunks=None):
        """
        pure ep strategy, for prefill stage mainly, only support eager mode
        """
        num_tokens, h = x.shape

        # -1: non-quant; 1: dynamic quant; 0: static quant(not supported now)
        # 2: MXFP8 E5M2; 3: MXFP8 E4M3; 4: FP8 E5M2; 5: FP8 E4M3; 6: A4 MXFP4
        routing_args = {"quant_mode": -1}
        if self.gmm_quant_mode == "w8a8float8":
            moe_init_routing = torch_npu.npu_moe_init_routing_v2
            routing_args.update({
                "quant_mode": 5,
                "row_idx_type": 0,
                "drop_pad_mode": 0,
            })
        elif "a4mxfloat4" in self.gmm_quant_mode:
            moe_init_routing = torch_npu.npu_moe_init_routing_v2
            routing_args.update({
                "quant_mode": 6,
                "row_idx_type": 0,
                "drop_pad_mode": 0,
            })
        elif "a8mxfloat" in self.gmm_quant_mode:
            moe_init_routing = torch_npu.npu_moe_init_routing_v2
            routing_args.update({
                "quant_mode": 3,
                "row_idx_type": 0,
                "drop_pad_mode": 0,
            })
        else:
            moe_init_routing = torch_npu.npu_moe_init_routing_v2

        hidden_states_list = []
        # EP-aligned chunk count, pre-computed once at the forward entry to avoid per-layer sync.
        global_num_chunks = prefill_moe_global_chunks
        for hidden_states, topk_ids, topk_weight, hidden_states_share in zip(
                *split_moe_tensors(x, topk_ids, topk_weight, hidden_states_share, num_chunks=global_num_chunks)):
            expanded_x, expanded_row_idx, tokens_per_expert, pertoken_scale = moe_init_routing(
                hidden_states,
                expert_idx=topk_ids,
                active_num=topk_ids.shape[0] * topk_ids.shape[1],
                scale=None,
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
            if hidden_states.shape[0] > 0:
                hidden_states_list.append(hidden_states)

        if len(hidden_states_list) == 0:
            return x.new_empty(0, h)
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
        expert_ids = topk_ids
        dispatch_args = {
            "x": hidden_states,
            "expert_ids": expert_ids, # [n*topk]
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
            "enable_cann_ops_nn": True
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
        self.platform_version = self.infer_config.model_config.platform_version
        self.layer_idx = layer_idx
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.attention_type = config.attention_types[layer_idx]
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

        self.max_position_embeddings = get_max_position_embeddings(self.infer_config)
        self.original_seq_len = config.max_seq_len

        self.block_size = self.infer_config.scheduler_config.block_size

        self.compressor = None
        self.indexer = None
        self.is_kv_source = layer_idx in config.kv_source_layers
        self.is_index_source = layer_idx in config.index_source_layers
        index_source_kv_share_layers = list(set(config.index_source_layers) - set(config.kv_source_layers))
        self.is_index_source_kv_share = layer_idx in index_source_kv_share_layers
        if self.is_kv_source:
            self.compressor = Compressor(config, self.infer_config, layer_idx, self.compress_ratio,
                                         head_dim=self.head_dim, prefix=f"{prefix}.compressor",
                                         comm_manager=self.comm_manager, **kwargs)
        if self.is_index_source:
            self.indexer = Indexer(config, self.infer_config, layer_idx, self.compress_ratio,
                                   prefix=f"{prefix}.indexer", comm_manager=self.comm_manager,
                                   **kwargs)
        # TODO: remove the following code after the new kernel is ready
        # self.sparse_attn_ops = torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla
        self.global_rank = kwargs.get("global_rank")

    def apply_rope(self, x, attn_metadata, inverse=False):
        """Apply interleaved RoPE to the tail of a packed [T, N, D] tensor."""
        rope_key = "win" if self.attention_type == "win" else "comp"
        cos_sin = attn_metadata["cos_sin"]
        cos, sin = cos_sin[rope_key]
        if inverse:
            sin = cos_sin[f"{rope_key}_neg_sin"]
        torch.ops.cann_ops_transformer.inplace_partial_rotary_mul(
            x.unsqueeze(2), cos, sin,
            rotary_mode="interleave",
            partial_slice=self.partial_slice,
        )
        return x

    def compute_win_cache(self, x: torch.Tensor, kv_cache: KVCache, attn_metadata):
        # The recipe Linear modules handle their own weight/activation quantization.
        # Keep normalized qr in BF16 for the Indexer and window KV in BF16 storage.
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.num_heads_per_rank, self.head_dim))
        kv = self.kv_norm(self.wkv(x)).unsqueeze(1)
        q = self.apply_rope(q, attn_metadata)
        kv = self.apply_rope(kv, attn_metadata)

        self.update_win_kv(kv, attn_metadata["slot_mapping"]["win_kv"], kv_cache.win_kv)
        win_kv = kv_cache.win_kv
        if attn_metadata["is_prefill"]:
            # The persistent SWA pages only retain the tail; early prefill queries
            # need the full current prompt. This temporary buffer is reused by layers.
            win_kv = attn_metadata["full_kv_cache"]
            self.update_win_kv(kv, attn_metadata["slot_mapping"]["full_kv"], win_kv)
        return win_kv, attn_metadata["win_idx"], qr, q

    def update_win_kv(self, kv, slot_mapping, cache):
        slots = slot_mapping.reshape(-1).long()
        # Evicted/padded pages use null block 0 as well as slot -1.
        valid = slots >= cache.shape[1]
        cache.view(-1, self.head_dim).index_copy_(
            0, slots[valid], kv.reshape(-1, self.head_dim)[valid].to(cache.dtype))

    def compute_comp_kv(self, x: torch.Tensor, qr: torch.Tensor, kv_cache: KVCache, attn_metadata, is_prefill, offset):
        if self.is_kv_source:
            latent = self.compressor(x, attn_metadata, is_prefill, kv_cache)
            # calc & store comp_idx, store index_cache if latent
            comp_idx = self.indexer(x, qr, attn_metadata, latent, kv_cache, offset)
            kv_cache.shared_idx = comp_idx

        elif self.is_index_source_kv_share:
            # calc & store comp_idx
            comp_idx = self.indexer(x, qr, attn_metadata, None, kv_cache, offset)
            kv_cache.shared_idx = comp_idx

        else:
            comp_idx = kv_cache.shared_idx

        return kv_cache.cmp_cache, comp_idx

    def sparse_attn(
        self,
        query_states,
        kv_states,
        topk_idxs
    ):
        """BSND shared-KV attention; topk_idxs uses -1 for invalid candidates.

        Other indices must be in [0, N). Computation dtype and sink handling are
        unchanged. Invalid candidates contribute neither to softmax nor to output.
        """
        if query_states.ndim == 3:
            T, H, D = query_states.shape
            query_states = query_states.view(1, T, H, D)

        B, M, H, D = query_states.shape
        N = kv_states.shape[0] #T,1,D
        K = topk_idxs.shape[-1] #M,K

        valid = topk_idxs != -1  #M,K
        # Replace -1 before gather; index 0 is only a temporary safe address.
        safe_idxs = topk_idxs.masked_fill(~valid, 0).long()
        kv_selected = torch.gather(
            kv_states.expand(N, K, D),
            dim=0,
            index=safe_idxs.unsqueeze(-1).expand(M, K, D)
        ).view(1, M, K, D)#B,M,K,D
        valid = valid.unsqueeze(0) # 1,m,k
        kv_selected.mul_(valid.unsqueeze(-1).to(kv_selected.dtype))

        logits = torch.einsum(
            "bsnd,bskd->bsnk",
            query_states,
            kv_selected
        )
        logits.mul_(self.softmax_scale)
        # Zero KV alone still contributes exp(0) to the softmax denominator.
        logits.masked_fill_(~valid.unsqueeze(2), float("-inf"))

        sink = self.attn_sink.reshape(1, 1, H, 1).to(logits)
        max_val = torch.maximum(
            logits.amax(-1, keepdim=True),
            sink
        )

        exp_logits = torch.exp(logits-max_val)
        exp_sink = torch.exp(sink-max_val)
        probs = exp_logits.div_(exp_logits.sum(-1, keepdim=True) + exp_sink)
        output = torch.einsum(
            "bsnk,bskd->bsnd",
            probs,
            kv_selected
        )

        return output.view(M, H, D)

    def attn_post(
        self,
        o: torch.Tensor,
        attn_metadata: Optional[Dict] = None,
    ):
        '''
        oproj_tp: split on group dim, o: [B, S, G, N, D/G] -> [B, S, G/tp_size, N, D/G]
        transpose to make the splitted dim to be the primary
        split o_a on group dim (batch); split o_b on group dim (reduce)
        '''
        num_tokens = o.shape[0]
        o = self.apply_rope(o, attn_metadata, inverse=True)
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

    def select_kv(self, cache, topk_ids, block_table, attn_metadata = None):
        # cache: [num_blocks, block_size, N, D]
        # topk_ids: [T, K] (or [T, 1, K])
        # block_table: block ids for the current request
        if cache is None or topk_ids is None or block_table is None:
            return None, None

        if cache.ndim < 3:
            raise ValueError(
                f"cache must have shape [num_blocks, block_size, N, D], got {tuple(cache.shape)}"
            )

        # The indexer may keep a singleton head dimension in its output.
        if topk_ids.ndim < 2:
            raise ValueError(
                f"topk_ids must have shape [T, K], got {tuple(topk_ids.shape)}"
            )
        topk_ids = topk_ids.reshape(topk_ids.shape[0], -1).to(device=cache.device)

        block_ids = block_table.reshape(-1).to(device=cache.device, dtype=torch.long)
        request_cache = cache.index_select(0, block_ids).reshape(-1, *cache.shape[2:])
        if attn_metadata:
            # win_kv
            safe_topk_ids = attn_metadata["win_topk_ids"]
            valid_mask = attn_metadata["win_topk_mask"]
        else:
            valid_mask = topk_ids >= 0
            safe_topk_ids = topk_ids.masked_fill(~valid_mask, 0).to(dtype=torch.long)
        gather_index = safe_topk_ids.reshape(-1, 1, 1).expand(
            -1, request_cache.shape[-2], request_cache.shape[-1]
        )
        selected_kv = torch.gather(request_cache, dim=0, index=gather_index)
        selected_kv = selected_kv.reshape(
            topk_ids.shape[0], topk_ids.shape[1], *request_cache.shape[1:]
        ) # T, K, D
        return selected_kv, valid_mask

    def forward(
        self,
        x: torch.Tensor,
        attn_metadata: Optional[Dict] = None,
        is_prefill: bool = True,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ):
        if kv_cache is not None and kv_cache.layer_idx != self.layer_idx:
            raise RuntimeError("KVCache layer does not match Attention layer")
        win_kv, win_idx, qr, q = self.compute_win_cache(x, kv_cache, attn_metadata)
        if is_prefill:
            block_table = attn_metadata["block_table"]["full_kv"]
        else:
            block_table = attn_metadata["block_table"]["win_kv"]
        block_ids = block_table.reshape(-1)
        win_select_kv = win_kv.index_select(0, block_ids).reshape(-1, *win_kv.shape[2:]) # T,1,d
        win_idx = attn_metadata["win_topk_ids"]  # T,128k
        offset = win_select_kv.shape[0]
        # selected_win_kv, valid_win_mask = self.select_kv(win_kv, win_idx, block_table)
        if self.compress_ratio > 0:
            compress_kv, compress_idxs = self.compute_comp_kv(x, qr, kv_cache, attn_metadata, is_prefill, offset)
            block_table = attn_metadata["block_table"][f"c{self.compress_ratio}a_cmp_kv"]
            block_ids = block_table.reshape(-1)
            cmp_select_kv = compress_kv.index_select(0, block_ids).reshape(-1, *compress_kv.shape[2:]) # T(m),1,d
            # selected_cmp_kv, valid_cmp_mask = self.select_kv(compress_kv, compress_idxs, block_table)
            if win_select_kv is not None and cmp_select_kv is not None:
                kv = torch.cat([win_select_kv, cmp_select_kv], dim=0)
                topk_idx = torch.cat([win_idx, compress_idxs.view(-1,compress_idxs.shape[-1])], dim=-1)
        else:
            kv = win_select_kv
            topk_idx = win_idx
        o = self.sparse_attn(q, kv, topk_idx)
        x = self.attn_post(o, attn_metadata)

        return x


class DeepseekV3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: DeepseekV3Config,
        infer_config: InferenceConfig,
        comm_manager: CommManager = None,
        layer_idx: int = 0,
        prefix: str = "",
        engram_layout: EngramLayout = None,
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

        self.engram = None
        if engram_layout is not None and layer_idx in engram_layout.layer_ids:
            self.engram = Engram(config, self.infer_config, layer_idx, engram_layout, f"{prefix}.engram",
                                 self.comm_manager)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pre_mix: torch.Tensor,
        attn_metadata: Optional[Dict] = None,
        past_residual: Optional[torch.Tensor] = None,
        is_prefill: Optional[bool] = False,
        cur_topk_list: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        image_mask: Optional[torch.Tensor] = None,
        prefill_moe_global_chunks: Optional[int] = None,
        kv_cache: Optional[KVCache] = None,
        engram_hashes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states
        if self.engram is not None and engram_hashes is not None:
            engram_metadata = attn_metadata.get("engram", {})
            hidden_states = self.engram(
                hidden_states, engram_hashes, is_prefill=is_prefill,
                engram_metadata=engram_metadata, image_mask=image_mask)
        hidden_states, post, comb, attn_pre = OpKernel.hc_pre(hidden_states, pre_mix,
                                                        self.hc_attn_fn, self.hc_attn_scale,
                                                        self.hc_attn_base, self.hc_mult, self.hc_sinkhorn_iters,
                                                        self.norm_eps, self.hc_eps)

        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.attn(
            x=hidden_states,
            attn_metadata=attn_metadata,
            is_prefill=is_prefill,
            kv_cache=kv_cache,
        )
        hidden_states = OpKernel.hc_post(hidden_states, residual, post, comb)

        residual = hidden_states
        hidden_states, post, comb, ffn_pre = OpKernel.hc_pre(hidden_states, attn_pre, self.hc_ffn_fn, self.hc_ffn_scale,
                                                    self.hc_ffn_base, self.hc_mult, self.hc_sinkhorn_iters,
                                                    self.norm_eps, self.hc_eps)

        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states,
            is_prefill=is_prefill,
            cur_topk_list=cur_topk_list,
            input_ids=input_ids,
            image_mask=image_mask,
            shared_expert_stream=attn_metadata.get('shared_expert_stream', None),
            prefill_moe_global_chunks=prefill_moe_global_chunks,
        )
        hidden_states = OpKernel.hc_post(hidden_states, residual, post, comb)
        return hidden_states, ffn_pre


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
        self.mm_quant_mode = (
            config.quant_config.mm_quant_mode
            if config.quant_config is not None
            else "w16a16")
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
        self.max_position_embeddings = get_max_position_embeddings(self.infer_config)

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            self.padding_idx,
            torch.bfloat16,
            tp_size=self.embed_tp_size,
            tp_rank=self.comm_manager.get_rank("embed_tp_group") if self.embed_tp_size > 1 else 0)

        self.engram_layout = EngramLayout.from_args(config)
        self.engram_hash = NgramHashState(
            config, self.infer_config, self.engram_layout, config.tokenizer
        )

        self.layers = nn.ModuleList(
            [
                DeepseekV3DecoderLayer(
                    config,
                    self.infer_config,
                    self.comm_manager,
                    layer_idx,
                    prefix=f"layers.{layer_idx}",
                    engram_layout=self.engram_layout,
                    **kwargs)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.kv_source_layers = config.kv_source_layers
        self.compress_ratios = config.compress_ratios

        # layer_to_source: dict, key = layer_id, value = corresponding source layer id
        # e.g. layer_id -> source_layer, maps each layer to its source layer
        # source_layers: dict, key is shared source layer id, value is list of layers sharing this source layer.
        # Example: 2: [3,4,5,6,7] means layer 3~7 share source layer 2
        self.layer_to_source, self.source_to_layers = build_cache_mapping(
            self.kv_source_layers or [], self.compress_ratios, len(self.layers)
        )

        self._init_model_cache_layout()
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # mhc
        self.hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
        _init_rope(self)

    def _entry_link_setter(self, layer_idx, name, tensor):
        for field, store, scope in self._entry_specs:
            if not name.startswith(field):
                continue
            key = layer_idx if scope is _Scope.PER_LAYER else self.layer_to_source.get(layer_idx, layer_idx)
            store[key] = tensor
            targets = self.source_to_layers.get(key, [key]) if scope is _Scope.SHARED else [key]
            for target in targets:
                setattr(self.kv_cache_by_layer[target], field, tensor)
                self.kv_cache_by_layer[target].shared_idx = self.shared_idx
            return
        raise KeyError(f"unknown cache entry name: {name}")

    def _init_model_cache_layout(self):
        self.cache_entries_by_layer = {i: [] for i in range(len(self.layers))}
        compressed_sources = {
            source for source in self.kv_source_layers
            if self.compress_ratios[source] > 1
        }
        # source_layer --> cmp_cache
        self.cmp_cache_by_layer = {source: torch.tensor([]) for source in self.kv_source_layers}
        # source_layer --> indexer_cache
        self.indexer_cache_by_layer = {
            source: torch.tensor([]) for source in self.kv_source_layers
        }
        # layer --> win_kv
        self.win_kv_by_layer = {i: torch.tensor([]) for i in range(len(self.layers))}
        # source_layer & cmp_ratio == 2 --> state_cache
        self.state_cache_by_layer = {source: torch.tensor([]) for source in compressed_sources}
        # layer --> kv_cache
        self.kv_cache_by_layer = {
            i: KVCache(i, self.layer_to_source.get(i)) for i in range(len(self.layers))
        }
        max_tokens = max(
            1,
            int(self.infer_config.scheduler_config.batch_size_per_dp_rank),
            int(self.infer_config.scheduler_config.max_prefill_tokens) + \
            int(self.infer_config.scheduler_config.max_new_tokens),
        )
        topk = int(getattr(self.config, "index_topk", 1))
        self.register_buffer("shared_idx", torch.empty((max_tokens, topk), dtype=torch.int32), persistent=False)
        self.register_buffer("candidates", torch.tensor([], dtype=torch.int32), persistent=False)

        self.head_dim = self.config.head_dim
        self.indexer_head_dim = self.config.index_head_dim
        self.cache_dtype = torch.bfloat16
        self.state_cache_dtype = torch.float32
        self.block_size = self.infer_config.scheduler_config.block_size
        self.window_size = self.config.sliding_window
        self._entry_specs = (
            ("win_kv", self.win_kv_by_layer, _Scope.PER_LAYER),
            ("cmp_cache", self.cmp_cache_by_layer, _Scope.SHARED),
            ("indexer_cache", self.indexer_cache_by_layer, _Scope.SHARED),
            ("state_cache", self.state_cache_by_layer, _Scope.SHARED_HEAD),
        )

        def add_entry(common, layer_idx, name, dim, dtype, **kwargs):
            num_head = kwargs.pop("num_head", 1)
            entry_common = dict(common)
            entry_common.update(kwargs)
            setter = lambda tensor, n=name, i=layer_idx: self._entry_link_setter(i, n, tensor)
            cache_entries.append(CacheEntry(
                cache_name=name,
                dim=dim,
                num_head=num_head,
                dtype=dtype,
                tensor_setter=setter,
                **entry_common))

        for layer_idx, _ in enumerate(self.layers):
            cache_entries = self.cache_entries_by_layer[layer_idx]
            win_setter = lambda tensor, i=layer_idx: self._entry_link_setter(
                i, f"win_kv_layer{i}", tensor)
            # every layer owns one win_kv
            cache_entries.append(CacheEntry(
                cache_name=f"win_kv_layer{layer_idx}",
                attn_type="SlidingWindow",
                dim=self.head_dim,
                num_head=1,
                dtype=self.cache_dtype,
                needs_block=True,
                block_size=self.block_size,
                manager_key="win_kv",
                tensor_setter=win_setter,
                sliding_window=self.window_size,
            ))

            ratio = self.compress_ratios[layer_idx]
            if layer_idx not in self.source_to_layers or ratio < 1:
                continue

            # create cmp_cache/indexer cache for source layers
            common = dict(attn_type="FullAttention",
                          needs_block=True,
                          block_size=self.block_size * ratio,
                          manager_key=f"c{ratio}a_cmp_kv",
                          compress_ratio=ratio)

            add_entry(common, layer_idx, f"cmp_cache_src{layer_idx}", self.head_dim, self.cache_dtype)
            add_entry(common, layer_idx, f"indexer_cache_src{layer_idx}", self.indexer_head_dim, self.cache_dtype)

            if ratio == 1:
                continue
            # create state_cache for source layers (cmp_ratio > 1)
            cache_entries.append(CacheEntry(
                cache_name=f"state_cache_src{layer_idx}",
                attn_type="SlidingWindow",
                dim=self.head_dim,
                num_head=2,
                dtype=torch.float32,
                needs_block=True,
                block_size=self.block_size,
                manager_key=f"c{ratio}a_cmp_state",
                sliding_window=ratio,
                tensor_setter=lambda tensor, i=layer_idx:
                self._entry_link_setter(i, "state_cache", tensor)))

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

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

    def update_cp_cos_sin(self, attn_metadata, hidden_states, kv_len):
        for zigzag_flag in ["prev", "next"]:
            position_ids_cur = attn_metadata[zigzag_flag]["position_ids_cur"]
            position_ids_with_pre_win = attn_metadata[zigzag_flag]["position_ids_with_pre_win"]
            position_ids_last_win = attn_metadata[zigzag_flag]["position_ids_last_win"]
            cos_sin = {
                "win": self.rotary_emb(hidden_states, position_ids_cur, kv_len, self.max_position_embeddings),
                "win_with_pre_win": self.rotary_emb(hidden_states, position_ids_with_pre_win,
                                                    kv_len, self.max_position_embeddings),
                "win_last_win": self.rotary_emb(hidden_states, position_ids_last_win,
                                                kv_len, self.max_position_embeddings),
            }
            cos_sin.update({"win_neg_sin": -cos_sin["win"][1]})

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

    def generate_cos_sin(self, attn_metadata, hidden_states):
        # WIN uses base RoPE; C2A/C1A Attention and Indexer use comp.
        # C1A Compressor uses per-token comp; C2A Compressor uses group positions in c2a.
        position_ids = attn_metadata["position_ids"]
        kv_len = attn_metadata["kv_len"]
        cos_sin = {
            "win": self.rotary_emb(hidden_states, position_ids, kv_len, self.max_position_embeddings),
        }
        cos_sin.update({"win_neg_sin": -cos_sin["win"][1]})

        position_ids_c = attn_metadata["position_ids_c"]
        cos_sin.update({
            "comp": self.compress_rotary_emb(
                hidden_states, position_ids, kv_len, self.max_position_embeddings),
            "c2a": self.compress_rotary_emb(
                hidden_states, position_ids_c["2"], kv_len, self.max_position_embeddings),
        })
        cos_sin.update({"comp_neg_sin": -cos_sin["comp"][1]})
        cos_sin["c1a"] = cos_sin["comp"]
        return cos_sin

    def merge_visual_embeddings(self, hidden_states, full_input_ids, visual_embeddings):
        image_mask = full_input_ids == self.config.image_token_id
        if int(image_mask.sum().item()) != visual_embeddings.shape[0]:
            raise RuntimeError(
                f"Visual embedding rows ({visual_embeddings.shape[0]}) do not match "
                f"image tokens ({int(image_mask.sum().item())})")
        visual_ordinals = image_mask.long().cumsum(0) - 1
        hidden_states[image_mask] = visual_embeddings[visual_ordinals[image_mask]]
        return hidden_states, image_mask

    def get_prefill_moe_global_chunks(self, hidden_states, is_prefill):
        if not is_prefill:
            return None
        moe_chunk_max_len = self.infer_config.model_config.custom_params.get("moe_chunk_max_len", 65536)
        moe_ep_group = None
        if self.moe_ep_size > 1:
            moe_ep_group = self.comm_manager.get_group("moe_ep_group")
        return get_moe_num_chunks(hidden_states, moe_chunk_max_len, moe_ep_group)

    @add_start_docstrings_to_model_forward(DEEPSEEKV3_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.Tensor,
        attn_metadata: Optional[Dict] = None,
        is_prefill: Optional[bool] = False,
        cur_topk_list: Optional[torch.Tensor] = None,
        visual_embeddings: Optional[torch.Tensor] = None,
    ):
        self.shared_idx.fill_(-1)
        full_input_ids = input_ids
        if is_prefill and self.cp_size > 1 and self.embed_tp_size == 1:
            input_ids = self.get_cp_input_ids(input_ids, attn_metadata)

        inputs_embeds = self.calc_input_embeddings(input_ids, is_prefill)
        hidden_states = inputs_embeds
        image_mask = None
        if visual_embeddings is not None:
            hidden_states, image_mask = self.merge_visual_embeddings(
                hidden_states, full_input_ids, visual_embeddings,
            )
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
        prefill_moe_global_chunks = self.get_prefill_moe_global_chunks(hidden_states, is_prefill)

        # Compute engram n-gram hashes centrally (cf. Transformer.forward in reference model.py)
        engram_hashes = None
        engram_meta = attn_metadata.get("engram", {})
        engram_hashes = self.engram_hash(
            input_ids, is_prefill,
            prefix_input_ids=engram_meta.get("prefix_input_ids"),
            ngram_shift_mask=engram_meta.get("shift_mask"))

        # mhc init one-hot pre_mix
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        pre_mix = make_identity_pre_mix(hidden_states, self.hc_mult)

        for decoder_layer in self.layers:
            layer_idx = decoder_layer.layer_idx
            layer_engram_hashes = None
            if engram_hashes is not None and decoder_layer.engram is not None:
                layer_hash_index = decoder_layer.engram.layer_hash_index
                layer_engram_hashes = engram_hashes[:, :, layer_hash_index, :]
            kv_cache = self.kv_cache_by_layer[layer_idx]
            kv_cache.shared_idx = self.shared_idx
            kv_cache.candidates = self.candidates
            hidden_states, pre_mix = decoder_layer(
                hidden_states,
                pre_mix,
                attn_metadata=attn_metadata,
                past_residual=residual,
                is_prefill=is_prefill,
                cur_topk_list=cur_topk_list,
                input_ids=input_ids,
                image_mask=image_mask,
                prefill_moe_global_chunks=prefill_moe_global_chunks,
                kv_cache=self.kv_cache_by_layer[layer_idx],
                engram_hashes=layer_engram_hashes,
            )
            self.shared_idx = kv_cache.shared_idx
            self.candidates = kv_cache.candidates
        hidden_states = hc_pre_mix(hidden_states, pre_mix)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class DeepseekV3ForCausalLM(DeepseekV3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self,
        config,
        infer_config: InferenceConfig,
        comm_manager: CommManager = None,
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
        self.max_position_embeddings = get_max_position_embeddings(self.infer_config)
        self.force_eplb = self.infer_config.model_config.force_eplb
        self.num_experts_per_tok = config.num_experts_per_tok
        # total experts num
        self.num_experts = config.n_routed_experts
        self.mm_quant_mode = (
            config.quant_config.mm_quant_mode
            if config.quant_config is not None
            else "w16a16")
        self.update_kv_quant_settings()
        self.update_gmm_quant_mode()
        self.kv_cache_quant_mode = config.quant_config.kv_cache_quant_mode
        self.li_cache_quant_mode = config.quant_config.li_cache_quant_mode
        self.attention_data = AttnMetaData(self.config, comm_manager, self.infer_config)
        self.enable_cache_compile = self.infer_config.model_config.enable_cache_compile

        self.enable_static_kernel = self.infer_config.model_config.enable_static_kernel
        self.enable_npugraph_ex = self.infer_config.model_config.exe_mode == "npugraph_ex"
        self.enable_multi_streams = self.infer_config.model_config.custom_params.get("enable_multi_streams", False)
        self.engram_tp_size = self.infer_config.model_config.custom_params.get("engram_tp_size", 1)
        self.enable_engram_offload = self.infer_config.model_config.custom_params.get("enable_engram_offload", False)

        self.metadata_event = []
        if self.enable_multi_streams:
            self.metadata_event = [torch.npu.Event(), torch.npu.Event()]

        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.rank_offset = int(os.getenv("RANK_OFFSET", "0"))
        self.global_rank = self.local_rank + self.rank_offset
        self.world_size = self.infer_config.parallel_config.world_size
        kwargs = {"global_rank": self.global_rank}
        self.init_parallel_comm_group()
        self.batch_size_per_rank = self.infer_config.scheduler_config.batch_size_per_dp_rank

        self.model = DeepseekV3Model(config, self.infer_config, self.comm_manager, prefix, **kwargs)
        self.vocab_size = config.vocab_size
        self.rope_head_dim = config.qk_rope_head_dim
        self.lm_head = ColumnParallelLinear(
                input_size=config.hidden_size,
                output_size=config.vocab_size,
                bias=False,
                tp_size=self.lmhead_tp_size,
                tp_rank=self.comm_manager.get_rank("lmhead_tp_group") if self.lmhead_tp_size > 1 else 0,
                quant_config=None,
                prefix="lm_head"
                )

        # Initialize weights and apply final processing
        self.post_init()
        self.block_size = self.infer_config.scheduler_config.block_size
        self.sas_metadata_ops = torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla_metadata
        self.window_size = config.sliding_window
        self.cp_segment_min_len = self.window_size
        self.init_cache_dim()
        self.first_layer_idx = 0
        self.first_layer_ratio = self.config.compress_ratios[self.first_layer_idx]

    @staticmethod
    def check_model_config_before_loading(config, infer_config):
        model_config = infer_config.model_config
        custom_params = model_config.custom_params
        parallel_config = infer_config.parallel_config
        scheduler_config = infer_config.scheduler_config

        engram_tp_size = int(custom_params.get("engram_tp_size", 1))
        enable_engram_offload = custom_params.get("enable_engram_offload", False)
        world_size = parallel_config.world_size
        if engram_tp_size <= 0 or world_size % engram_tp_size != 0:
            raise ValueError(
                f"engram_tp_size={engram_tp_size} must be positive and divide world_size={world_size}"
            )
        if engram_tp_size > world_size:
            raise ValueError(
                f"engram_tp_size={engram_tp_size} can not be greater than world_size={world_size}"
            )
        if enable_engram_offload and world_size > 1:
            device_count = torch.npu.device_count()
            if world_size <= device_count and engram_tp_size < world_size:
                raise ValueError(
                    f"Got engram_tp_size={engram_tp_size}, "
                    f"world_size={world_size}, device_count={device_count}; "
                    "set engram_tp_size=world_size to keep host storage "
                    "non-redundant."
                )
            if world_size > device_count and engram_tp_size < device_count:
                raise ValueError(
                    f"Got engram_tp_size={engram_tp_size}, "
                    f"world_size={world_size}, device_count={device_count}; "
                    "set engram_tp_size to at least device_count so host "
                    "storage remains non-redundant."
                )

        exe_mode = model_config.exe_mode
        enable_cache_compile = model_config.enable_cache_compile
        platform_version = model_config.platform_version
        enable_superkernel = custom_params.get("enable_superkernel", False)
        moe_chunk_max_len = custom_params.get("moe_chunk_max_len", 65536)
        with_ckpt = model_config.with_ckpt

        if (
            isinstance(moe_chunk_max_len, bool)
            or not isinstance(moe_chunk_max_len, int)
            or moe_chunk_max_len <= 0
        ):
            raise ValueError(f"{moe_chunk_max_len=} should be a positive integer.")
        if not with_ckpt and not model_config.force_eplb:
            raise ValueError(f"{model_config.force_eplb=} must be True if {with_ckpt =}!")

        if exe_mode != "eager":
            raise ValueError(f"{exe_mode=} does not supported!")

        if parallel_config.attn_tp_size > 1:
            raise ValueError(f"{parallel_config.attn_tp_size=} is not supported yet!")

        if parallel_config.cp_size > 1:
            raise ValueError(f"{parallel_config.cp_size=} is not supported yet!")

        if model_config.next_n > 0:
            raise ValueError(f"{model_config.next_n=} is not supported yet! Please set next_n = 0")
        
        if scheduler_config.batch_size_per_dp_rank > 1:
            raise ValueError(f"{scheduler_config.batch_size_per_dp_rank=} is not supported yet!" + \
                             " Please set scheduler_config.batch_size = world_size")

        dynamo_feat = enable_cache_compile or enable_superkernel
        if exe_mode == "eager" and dynamo_feat:
            raise ValueError(f"{exe_mode=} does not support cache compile or superkernel!")

        if parallel_config.cp_size > 1 and scheduler_config.cp_mini_batch != 1:
            raise ValueError(f"when cp enabled, {scheduler_config.cp_mini_batch=} should be 1")

        model_config.enable_weight_nz = platform_version != PlatformVersion.ASCEND_950

    def check_model_settings(self):
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
            if module_name.endswith("compressor.norm"):
                module.weight.data = module.weight.data.to(torch.float32)

            if "wo_a" in module_name:
                config = self.config
                head_dim_per_group = config.num_attention_heads * config.head_dim // config.o_groups
                module.weight.data = module.weight.data.view(-1, config.o_lora_rank, head_dim_per_group) \
                                                   .transpose(1, 2).contiguous()
                if config.quant_config.mm_quant_mode == "w8a8mxfloat8":
                    scale_data = reshape_mx_scale(module.weight_scale.data)
                    module.weight_scale.data = scale_data.view(-1, config.o_lora_rank, *scale_data.shape[1:]) \
                                                   .transpose(1, 2).contiguous()
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
                Fp8PerTileMoEGMMMethod,
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
        # Cache storage is BF16 independently of the Linear weight quantization.
        self.config.quant_config.set_quant_mode("kv_cache_quant_mode", "unquant")
        self.config.quant_config.set_quant_mode("li_cache_quant_mode", "unquant")

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

        if self.engram_tp_size > 1:
            for layer_id in self.config.engram_layer_ids:
                self.comm_manager.register_group(
                    name=f"engram_tp_group_{layer_id}",
                    group_num=world_size // self.engram_tp_size,
                    group_size=self.engram_tp_size,
                    platform_version=platform_version,
                    # Keep one physical communication domain per Engram layer.
                    allow_physical_reuse=False,
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
        metadata_kwargs["seqused_q"] = attn_metadata['seq_used_q']
        actual_seq_k = attn_metadata['actual_seq_k']
        metadata_kwargs["seqused_ori_kv"] = actual_seq_k
        if metadata_kwargs.get("has_cmp_kv", False):
            cmp_ratio = metadata_kwargs["cmp_ratio"]
            ratio_key = f"{cmp_ratio}"
            metadata_kwargs.update({
                "seqused_cmp_kv": attn_metadata["compressed_seq_lens"][ratio_key],
                "cmp_residual_kv": attn_metadata["compressed_seq_remainders"][ratio_key],
            })
        metadata_kwargs["batch_size"] = actual_seq_k.shape[0]
        return self.sas_metadata_ops(**metadata_kwargs)

    def calc_li_metadata(self, attn_metadata, metadata_kwargs):
        metadata_kwargs["cu_seqlens_q"] = attn_metadata['cu_seq_lens_q']
        metadata_kwargs["seqused_q"] = attn_metadata['seq_used_q']
        actual_seq_k = attn_metadata['actual_seq_k']
        cmp_ratio = metadata_kwargs["cmp_ratio"]
        ratio_key = f"{cmp_ratio}"
        # seqused_k is the valid length in the compressed cache; cmp_residual_k
        # carries the remainder needed to reconstruct the original sequence length.
        metadata_kwargs["seqused_k"] = attn_metadata["compressed_seq_lens"][ratio_key]
        metadata_kwargs["cmp_residual_k"] = attn_metadata["compressed_seq_remainders"][ratio_key]
        metadata_kwargs["batch_size"] = actual_seq_k.shape[0]
        return torch.ops.cann_ops_transformer.quant_lightning_indexer_metadata(**metadata_kwargs)

    def generate_metadata(self, attn_metadata, metadata_kwargs, is_prefill, metadata_desc, is_li=False):
        metadata_ops = self.calc_li_metadata if is_li else self.calc_sas_metadata
        if is_prefill and self.cp_size > 1:
            for zz_flag in ['prev', 'next']:
                attn_metadata[zz_flag]['kernel_metadata'][metadata_desc] = \
                    metadata_ops(attn_metadata[zz_flag], metadata_kwargs)
        else:
            attn_metadata['kernel_metadata'][metadata_desc] = metadata_ops(attn_metadata, metadata_kwargs)

    def init_cache_dim(self):
        self.cache_dim = get_kv_cache_dim(
            self.config.head_dim, self.config.qk_rope_head_dim, self.kv_cache_quant_mode)

    def generate_sas_metadata_kwargs(self):
        # Sparse MLA mask modes: 0: no mask, 3: right-down causal, 4: sliding window.
        sas_metadata_kwargs = {
            "cmp_ratio": 1,
            "ori_mask_mode": 4, # 4: sliding-window mask
            # The C1A metadata kernel uses causal mode 3.
            "cmp_mask_mode": 0 if self.platform_version == PlatformVersion.ASCEND_950 else 3,
            "ori_win_left": 127, # default
            "ori_win_right": 0,
            "layout_q": "TND",
            "layout_kv": "PA_BBND",
            "num_heads_q": self.config.num_attention_heads,
            "num_heads_kv": 1,
            "head_dim": self.config.kv_lora_rank,
            "has_ori_kv": True,
            "has_cmp_kv": False,
        }
        sas_metadata_kwargs.update(
            {"quant_mode": 1,  # 1: quantized KV-cache mode for mixed sparse MLA
                "rope_head_dim": self.rope_head_dim}
        )
        return sas_metadata_kwargs

    def generate_li_metadata_kwargs(self):
        # Lightning Indexer quant mode: 1: FP8 E4M3.
        quant_mode = 1
        li_metadata_kwargs = {
            "layout_k": "PA_BBND",
            "topk": self.config.index_topk,
            "mask_mode": 3,  # 3: causal mask
            "layout_q": "TND",
            "cmp_ratio": 4, # only c4a have li module
            "quant_mode": quant_mode,
            "num_heads_q": self.config.index_n_heads,
            "num_heads_k": 1,
            "head_dim": self.config.index_head_dim,
        }
        return li_metadata_kwargs

    def generate_kernel_metadata(self, attn_metadata, is_prefill):
        metadata_stream = attn_metadata.get('metadata_stream', None)
        c1a_metadata_kwargs = None
        if self.first_layer_ratio == 1:
            c1a_metadata_kwargs = self.generate_sas_metadata_kwargs()

        c4a_metadata_kwargs = self.generate_sas_metadata_kwargs()
        c4a_metadata_kwargs.update(
            {"cmp_ratio": 4, "cmp_mask_mode": 3, "has_cmp_kv": True, "cmp_topk": self.config.index_topk}
        )

        c128a_metadata_kwargs = self.generate_sas_metadata_kwargs()
        c128a_metadata_kwargs.update({"cmp_ratio": 128, "cmp_mask_mode": 3, "has_cmp_kv": True})

        enable_metadata_multi_streams = self.enable_multi_streams and not is_prefill
        record_event(enable_metadata_multi_streams, attn_metadata.get('metadata_event'), 0)

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
            li_metadata_kwargs = self.generate_li_metadata_kwargs()
            self.generate_metadata(attn_metadata, li_metadata_kwargs,\
                                    is_prefill, "lightning_indexer_quant", is_li=True)

    def preprocess_model_inputs(self, model_inputs: Dict, is_prefill=False, **kwargs):
        model_inputs = dict(model_inputs)

        attn_metadata = self.attention_data.build_attn_metadata(
            model_inputs.get("input_ids"),
            model_inputs.get("position_ids"),
            model_inputs.get("forward_metadata"),
            batch=model_inputs.get("batch"),
        )
        attn_metadata.update({"metadata_event": self.metadata_event})
        model_inputs["attn_metadata"] = attn_metadata
        model_inputs.pop("batch", None)

        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: torch.LongTensor = None,
        forward_metadata: ForwardMetaData = None,
        attn_metadata: Optional[Dict] = None,
        cur_topk_list: Optional[torch.Tensor] = None,
        visual_embeddings: Optional[torch.Tensor] = None,
        **kwargs
    ):
        is_prefill = forward_metadata.is_prefill

        # TODO: generate kernel metadata after supporting ascendc kernel(fa/indexer)
        # self.generate_kernel_metadata(attn_metadata, is_prefill)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attn_metadata=attn_metadata,
            is_prefill=is_prefill,
            cur_topk_list=cur_topk_list,
            visual_embeddings=visual_embeddings,
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

    def get_cache_info(
        self,
    ) -> ModelCacheInfo:
        layer_infos = []
        for layer_idx in range(len(self.model.layers)):
            layer_infos.append(
                LayerCacheInfo(
                    layer_idx=layer_idx,
                    caches=list(self.model.cache_entries_by_layer[layer_idx]),
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

        # Params for expert weights and weight scales
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
        engram_dequant_cache = {}
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ".ffn.shared_experts." in name or ".attn." in name:
                name = name.replace(".scale", ".weight_scale")
                if name.endswith(".weight_scale"):
                    # Scales are repeated offline; preserve their E8M0 bits when loading.
                    loaded_weight = loaded_weight.view(torch.uint8)
            elif name.endswith(".engram.wkv.scale"):
                loaded_weight = loaded_weight.view(torch.uint8)

            if self.enable_engram_offload and ".engram.embed." in name:
                base_name, attr = name.rsplit(".", 1)
                if attr in {"weight", "scale"}:
                    engram_dequant_cache.setdefault(base_name, {})[attr] = loaded_weight
                    cached = engram_dequant_cache[base_name]
                    if "weight" in cached and "scale" in cached:
                        param_name = f"{base_name}.weight"
                        if param_name not in params_dict:
                            raise KeyError(
                                f"Engram embedding parameter {param_name} was not found"
                            )
                        param = params_dict[param_name]
                        fp8_weight_loader = getattr(
                            param, "engram_fp8_weight_loader", None
                        )
                        if fp8_weight_loader is None:
                            raise RuntimeError(
                                f"Engram parameter {param_name} has no FP8 weight loader"
                            )
                        fp8_weight_loader(param, cached["weight"], cached["scale"])
                        loaded_params.add(param_name)
                        engram_dequant_cache.pop(base_name)
                    continue

            if name.startswith(("vision.", "aligner.", "image_")):
                param = params_dict.get(name)
                if param is not None:
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
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
                        name = name.replace("w13_scale", "w13_weight_scale").replace("w2_scale", "w2_weight_scale")

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

        if engram_dequant_cache:
            missing = {
                base_name: sorted({"weight", "scale"} - set(parts))
                for base_name, parts in engram_dequant_cache.items()
            }
            raise ValueError(
                f"Incomplete Engram FP8 checkpoint tensors: {missing}"
            )

        # add checkpoint load check
        weights_not_loaded = set(params_dict.keys()) - loaded_params
        key_weights = {"smooth_scale", "w2_alpha"}
        if weights_not_loaded:
            if all(any(key in name for key in key_weights) for name in weights_not_loaded):
                logger.warning(
                    "Smooth scales were not initialized from checkpoint.")
            else:
                raise ValueError(
                    "Following weights were not initialized from "
                    f"checkpoint: {weights_not_loaded}")
        return loaded_params

    def offload_weights(self):
        if not self.enable_engram_offload:
            return
        for module in self.modules():
            if hasattr(module, 'engram_offload') and module.engram_offload:
                if hasattr(module, 'offload_weights') and not module._engram_offloaded:
                    module.offload_weights()


class DeepseekV41VisionForCausalLM(DeepseekV3ForCausalLM):
    def init_parallel_comm_group(self):
        super().init_parallel_comm_group()
        self.view_parallel_size = self.attn_tp_size * self.cp_size
        if self.infer_config.disagg_config.disaggregation_mode != "DECODE":
            self.comm_manager.register_group(
                name="view_parallel_group",
                group_num=self.world_size // self.view_parallel_size,
                group_size=self.view_parallel_size,
                platform_version=self.platform_version,
            )

    def __init__(self, config, infer_config, comm_manager=None, prefix=""):
        super().__init__(config, infer_config, comm_manager, prefix=prefix)
        is_decode = infer_config.disagg_config.disaggregation_mode == "DECODE"
        self.vision_model = None if is_decode else DeepseekV41VisionModel(
            config,
            view_parallel_group=comm_manager.get_group("view_parallel_group"),
            view_parallel_rank=comm_manager.get_rank("view_parallel_group"),
            view_parallel_size=self.view_parallel_size,
        )

    def encode_multimodal(self, mm_inputs_list):
        if self.vision_model is None:
            raise RuntimeError(
                "encode_multimodal requires a vision model, but this rank was "
                "built without one."
            )
        return self.vision_model(mm_inputs_list)

    def build_multimodal_warmup_inputs(self, seq_len) -> Optional[Tuple[torch.Tensor, Dict]]:
        if self.vision_model is None:
            return None
        from models.deepseek_v4_1.utils.image_processor import image_token_types
        n_vit_h = n_vit_w = 28
        n_llm_h = n_llm_w = 10
        patches = torch.zeros(
            n_vit_h * n_vit_w,
            3,
            self.config.vision_patch_size,
            self.config.vision_patch_size,
            dtype=torch.bfloat16,
        )
        image_token_id = self.config.image_token_id
        tokens = []
        images = []
        for _ in range(self.view_parallel_size):
            types = image_token_types(n_llm_h, n_llm_w)
            tokens.extend([image_token_id] * types.numel())
            images.append({
                "patches": patches,
                "n_vit_h": n_vit_h,
                "n_vit_w": n_vit_w,
                "types": types,
            })
        if len(tokens) > seq_len:
            return None
        input_ids = torch.tensor(
            tokens + [0] * (seq_len - len(tokens)), dtype=torch.long
        )
        return input_ids, {"images": images}


def adapt_safetensors_field(params_dict: Dict):
    fix_dict = {}
    for k, v in params_dict.items():
        if "model." in k:
            k = k.removeprefix("model.")
        if k.startswith("vision_model."):
            k = k.removeprefix("vision_model.")
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
        if ".engram." in k:
            k = k.replace(".multi_head_embedding.", ".embed.")
        fix_dict[k] = v
    return fix_dict
