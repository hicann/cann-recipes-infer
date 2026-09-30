# coding=utf-8
# Adapted from
# https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py
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
from executor.utils.stream_utils import npu_stream_switch, record_event, wait_event, record_stream, create_stream, create_event
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
from module.quantization.mxfp4 import W4A8MxFp4MoEGMMMethod
from .configuration_deepseek import DeepseekV41Config
from .modules import (get_window_topk_idxs, get_compress_topk_idxs,
                      one_hot, yarn_get_mscale,
                      DeepseekV41RMSNorm, _init_rope, DEEPSEEKV41_START_DOCSTRING,
                      DEEPSEEKV41_INPUTS_DOCSTRING, DeepseekV41PreTrainedModel, apply_rotary_emb,
                      AttnMetaData, PACKED_KV_STORAGE_DTYPE,
                      PACKED_KV_COMPUTE_DTYPE, gather_cp_segments, get_cmp_block_table, get_cp_tmp_cache,
                      get_kv_cache_dim, has_cp_decode_requests, pick_request_tail,
                      scatter_cache_rows, select_cp_segments
                    )
from .modules import Compressor, Engram, EngramLayout, NgramHashState
from .modules.indexer import Indexer, MXFP4_VALUES_PER_BYTE, MXFP4_SCALE_GROUP_SIZE
from .modules.compressor import write_compressed_kv
from .modules.registry import OpKernel, auto_import_modules
from .modeling_deepseek_vision import DeepseekV41VisionModel
from .modules.op_impls.mhc import make_identity_pre_mix, hc_pre_mix
logger = logging.get_logger(__name__)

HADAMARD_SIZE = 128
MATMUL_MAX_AXIS_VALUE = 65535

SEPARATE_MLA_KV = 0
SEPARATE_MLA_INDICES = 1
SEPARATE_MLA_DONE = 2


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
        self.indexer_cache_scale = None
        self.indexer_qsli_cache = None
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


def get_max_local_prefill_tokens(infer_config: InferenceConfig, segment_min_len: int):
    """Upper bound on the prefill tokens one rank computes in a single step."""
    max_prefill_tokens = infer_config.scheduler_config.max_prefill_tokens
    cp_size = infer_config.parallel_config.cp_size
    if cp_size <= 1:
        return max_prefill_tokens
    # CP pads every request to 2 * cp_size segments and hands two of them to each rank.
    segments = 2 * cp_size
    segment_len = max((max_prefill_tokens + segments - 1) // segments, segment_min_len)
    extra_requests = infer_config.scheduler_config.cp_mini_batch - 1
    return 2 * segment_len + 2 * extra_requests * (segment_min_len + 1)


class DeepseekV41SharedExpert(nn.Module):
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
        self.low_latency_tp = bool(
            self.infer_config.model_config.custom_params.get("low_latency_tp", False)
        )
        # SwiGLU via the repo custom op when its package is deployed in the
        # image; falls back to the mainline op otherwise.
        kernel_config = self.infer_config.model_config.custom_params.get("kernel_config", {})
        self.enable_custom_swiglu = kernel_config.get("swiglu", "custom") == "custom"
        self.config = config
        self.hidden_size = config.hidden_size
        self.is_moe_layer = is_moe_layer
        self.tp_rank = self.comm_manager.get_rank("moe_tp_group") if self.moe_tp_size > 1 else 0
        from module.fuse_moe_gmm import FusedMoEGMM
        self.intermediate_size = config.moe_intermediate_size * config.n_shared_experts
        self.enable_nonuniform_tp_split = (
            self.moe_tp_size > 1
            and "mxfloat" in self.mm_quant_mode
            and self.intermediate_size % FusedMoEGMM.TP_INTERMEDIATE_ALIGN == 0
            and (self.intermediate_size // self.moe_tp_size) % FusedMoEGMM.TP_INTERMEDIATE_ALIGN != 0)
        if self.enable_nonuniform_tp_split:
            split_sizes = FusedMoEGMM.split_intermediate_aligned(
                self.intermediate_size, self.moe_tp_size, FusedMoEGMM.TP_INTERMEDIATE_ALIGN)
            self.intermediate_size_per_partition = split_sizes[self.tp_rank]
            self.intermediate_offset = sum(split_sizes[:self.tp_rank])
        else:
            self.intermediate_size_per_partition = self.intermediate_size // self.moe_tp_size
            self.intermediate_offset = (self.intermediate_size // self.moe_tp_size) * self.tp_rank
        if self.enable_nonuniform_tp_split:
            gu_tp_size, gu_tp_rank = 1, 0
            dp_tp_size, dp_tp_rank = 1, 0
            gu_output_sizes = [self.intermediate_size_per_partition] * 2
            dp_input_size = self.intermediate_size_per_partition
            shard_kwargs = dict(
                shard_full_size=self.intermediate_size,
                shard_offset=self.intermediate_offset)
        else:
            # Uniform: original behavior, let the generic Linear shard itself.
            gu_tp_size, gu_tp_rank = self.moe_tp_size, self.tp_rank
            dp_tp_size, dp_tp_rank = self.moe_tp_size, self.tp_rank
            gu_output_sizes = [self.intermediate_size] * 2
            dp_input_size = self.intermediate_size
            shard_kwargs = {}
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=gu_output_sizes,
            bias=False,
            tp_size=gu_tp_size,
            tp_rank=gu_tp_rank,
            quant_config=config.quant_config,
            prefix=f"{prefix}.gate_up_proj",
            **shard_kwargs,
            )
        self.down_proj = RowParallelLinear(
            dp_input_size,
            config.hidden_size,
            bias=False,
            tp_size=dp_tp_size,
            tp_rank=dp_tp_rank,
            quant_config=config.quant_config,
            prefix=f"{prefix}.down_proj",
            **shard_kwargs,
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

    def swiglu_a8float8(self, merged_x):
        swiglu_limit_args = {}
        if self.swiglu_limit is not None:
            swiglu_limit_args["clamp_limit"] = self.swiglu_limit
        d_limit = 64 if "mx" in self.mm_quant_mode else 256
        if self.enable_custom_swiglu and merged_x.shape[-1] % d_limit == 0:
            intermediate_hidden_states, pergroup_scale, _ = torch.ops.custom.npu_swiglu_group_quant(
                merged_x,
                dst_type=torch.float8_e4m3fn,
                round_scale=True if "mx" in self.mm_quant_mode else False,
                quant_mode=1 if "mx" in self.mm_quant_mode else 0,
                clamp_limit=self.swiglu_limit,
                )
            return intermediate_hidden_states, pergroup_scale
        if merged_x.shape[-1] % 256 == 0:
            intermediate_hidden_states, pergroup_scale, _ = torch.ops.cann_ops_nn.swiglu_group_quant(
                merged_x,
                dst_type=torch.float8_e4m3fn,
                round_scale=True if "mx" in self.mm_quant_mode else False,
                quant_mode=1 if "mx" in self.mm_quant_mode else 0,
                **swiglu_limit_args,
                )
            return intermediate_hidden_states, pergroup_scale
        # Fused kernels reject this width: clamp before silu, matching the
        # fused kernel's A=Min(A,limit), B=Clamp(B,+-limit) semantics.
        gate, up = merged_x.chunk(2, dim=-1)
        if self.swiglu_limit is not None:
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        intermediate_hidden_states = F.silu(gate) * up
        return intermediate_hidden_states, None

    def forward_a8float8(self, x, enable_decode_stream=False, shared_expert_event=None):
        merged_x = self.gate_up_proj(x)
        intermediate_hidden_states, pergroup_scale = self.swiglu_a8float8(merged_x)
        wait_event(enable_decode_stream, shared_expert_event, 0)
        if pergroup_scale is None:
            return self.down_proj(intermediate_hidden_states)
        return self.down_proj(intermediate_hidden_states, pergroup_scale)


class DeepseekV41MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(
        self,
        config,
        infer_config: InferenceConfig,
        comm_manager: CommManager = None,
        prefix="",
        is_mtp: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = kwargs.get("layer_idx")
        self.is_mtp = is_mtp
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.gmm_quant_mode = config.quant_config.gmm_quant_mode
        self.swiglu_limit = config.swiglu_limit if hasattr(config, "swiglu_limit") else None
        self.hidden_dim = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.moe_tp_size = self.infer_config.parallel_config.moe_tp_size
        self.moe_ep_size = self.infer_config.parallel_config.moe_ep_size
        self.low_latency_tp = bool(
            self.infer_config.model_config.custom_params.get("low_latency_tp", False)
        )
        self.platform_version = self.infer_config.model_config.platform_version
        self.exe_mode = self.infer_config.model_config.exe_mode
        self.enable_multi_streams = self.infer_config.model_config.custom_params.get("enable_multi_streams", False)
        self.megamoe_backend = kwargs.get("megamoe_backend")
        self.enable_mega_moe = self.megamoe_backend is not None
        self.force_eplb = self.infer_config.model_config.force_eplb
        self.n_routed_experts, self.num_experts_per_tok = self.get_moe_config(self.layer_idx)
        # Total expert counts are layer-specific: DSpark/MTP layers may use a
        # smaller routed-expert pool and a different top-k than the backbone.
        self.num_experts = self.n_routed_experts
        self.top_k = self.num_experts_per_tok
        self.is_a4w4_mxfp4 = self.gmm_quant_mode == "w4a4mxfloat4"
        self.is_a8w4_mxfp4 = self.gmm_quant_mode == "w4a8mxfloat4"
        self.enable_grouplist_type2 = self.infer_config.model_config.custom_params.get("enable_grouplist_type2", False)

        self.intermediate_size_per_rank = self.intermediate_size // self.moe_tp_size
        self.shared_expert_rank_num = 0 # route and share on same card
        self.n_shared_experts = config.n_shared_experts
        self.experts_per_rank = self.n_routed_experts // self.moe_ep_size
        kernel_config = self.infer_config.model_config.custom_params.get("kernel_config", {})
        self.enable_custom_swiglu = kernel_config.get("swiglu", "custom") == "custom"
        self.experts = FusedMoEGMM(
            num_experts=self.n_routed_experts,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_size,
            bias=False,
            quant_config=config.quant_config,
            tp_size=self.moe_tp_size,
            tp_rank=self.comm_manager.get_rank("moe_tp_group") if self.moe_tp_size > 1 else 0,
            ep_size=self.moe_ep_size,
            ep_rank=self.comm_manager.get_rank("moe_ep_group") if self.moe_ep_size > 1 else 0,
            prefix=f"{prefix}.experts",
            swiglu_limit=self.swiglu_limit,
            enable_cann_ops_nn=True,
            enable_custom_swiglu=self.enable_custom_swiglu,
        )
        self.moe_ffn = self.experts
        self._init_gate(prefix)
        if config.n_shared_experts is not None:
            self.shared_experts = DeepseekV41SharedExpert(
                config,
                self.infer_config,
                self.comm_manager,
                is_moe_layer=True,
                prefix=f"{prefix}.shared_experts",
                **kwargs,
            )
        shared = getattr(self, "shared_experts", None)
        self.enable_shared_pipeline = (
            self.low_latency_tp and self.enable_multi_streams and shared is not None
            and "float8" in shared.mm_quant_mode and "a8" in shared.mm_quant_mode
            and isinstance(self.experts.quant_method, W4A8MxFp4MoEGMMMethod)
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

    def get_moe_config(self, layer_id: int) -> tuple[int, int]:
        """Return routed and activated expert counts for a model layer.

        The target backbone uses the regular MoE configuration. Proposal
        layers appended after ``num_hidden_layers`` use the optional DSpark
        configuration and fall back to the backbone values when it is unset.
        """
        main_routed = int(self.config.n_routed_experts)
        main_top_k = int(self.config.num_experts_per_tok)
        if layer_id < self.config.num_hidden_layers:
            return main_routed, main_top_k

        dspark_routed = int(getattr(self.config, "dspark_n_routed_experts", 0) or 0)
        dspark_top_k = int(getattr(self.config, "dspark_n_activated_experts", 0) or 0)
        return dspark_routed or main_routed, dspark_top_k or main_top_k

    def _init_gate(self, prefix):
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
        mc2_group_name = "dspark_moe_ep_group_mc2" if self.is_mtp else "moe_ep_group_mc2"
        moe_ep_group_name = self.comm_manager.get_group_name(mc2_group_name)
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
                shared_expert_stream=None, prefill_moe_global_chunks=None, engram_resource_events=None,
                moe_events=None):
        moe_events = moe_events or {}
        shared_expert_events = moe_events.get("shared_expert", [])
        _, h = hidden_states.shape
        shared_merged_x = None
        if self.low_latency_tp:
            hidden_states_share = None
            if self.n_shared_experts > 0:
                if self.enable_shared_pipeline and not is_prefill:
                    shared_merged_x = self.shared_expert_gate_up(
                        hidden_states, shared_expert_stream, shared_expert_events)
                else:
                    hidden_states_share = self.forward_shared_expert(
                        hidden_states, shared_expert_stream, shared_expert_events)
        elif is_prefill:
            # With MegaMoE the shared experts run inside the operator, so the
            # model-side computation must be skipped to avoid double counting.
            if self.n_shared_experts > 0 and not self.enable_mega_moe:
                hidden_states_share = self.forward_shared_expert(
                    hidden_states, shared_expert_stream, shared_expert_events)
            else:
                hidden_states_share = None
        else:
            record_stream(self.enable_multi_streams, hidden_states, shared_expert_stream)
            record_event(self.enable_multi_streams, shared_expert_events, 0)

        # compute gating score
        if self.platform_version == PlatformVersion.ASCEND_950:
            logits = torch_npu.npu_fused_matmul(hidden_states.view(-1, h), self.gate.weight, fused_op_type="16cast32")
        else:
            logits = self.gate(hidden_states.view(-1, h).to(torch.float32))
        topk_idx, topk_weight, _ = OpKernel.gate_topk(
            self, logits, input_ids, image_mask, is_prefill=is_prefill
        )
        if self.force_eplb:
            topk_idx = cur_topk_list
        topk_idx = topk_idx.to(torch.int32)

        if self.low_latency_tp:
            return self.moe_infer_tp(
                hidden_states, topk_idx, topk_weight, hidden_states_share,
                moe_events, engram_resource_events, shared_expert_stream, shared_merged_x)

        # MOE EP
        if is_prefill:
            if self.enable_mega_moe:
                return self.megamoe_backend.prefill(
                    self,
                    hidden_states,
                    topk_idx,
                    topk_weight,
                )
            return self.moe_infer_double_routing(
                hidden_states, topk_idx, topk_weight, hidden_states_share,
                shared_expert_events, prefill_moe_global_chunks)
        else:
            return self.moe_infer_dispatch_combine(
                hidden_states, topk_idx, topk_weight, shared_expert_stream,
                moe_events, engram_resource_events=engram_resource_events)

    def _build_moe_tp_routing_args(self, x, topk_ids):
        routing_quantizes_input = self.gmm_quant_mode in ("w8a8mxfloat8", "w4a8mxfloat4")
        routing_args = {
            "expert_idx": topk_ids,
            "active_num": x.shape[0] * self.top_k,
            "expert_num": self.num_experts,
            # type=2: routing emits the sparse [E, 2] (expert_id, count)
            # directly in expert order; type=1 keeps the count list.
            "expert_tokens_num_type": 2 if self.enable_grouplist_type2 else 1,
            "expert_tokens_num_flag": True,
            "active_expert_range": [0, self.num_experts],
            "quant_mode": 3 if routing_quantizes_input else -1,
        }
        return routing_args, routing_quantizes_input

    def _build_moe_tp_gmm_args(self, expanded_x, tokens_per_expert, pertoken_scale, routing_quantizes_input):
        group_list_type = 1
        gmm_group_list = tokens_per_expert
        swiglu_group_list = None
        if self.enable_grouplist_type2:
            if tokens_per_expert.dim() == 1:
                # Pre-fix op form: build the sparse pairs locally.
                expert_ids = torch.arange(self.num_experts, dtype=torch.int64,
                                          device=tokens_per_expert.device)
                sparse_pairs = torch.stack(
                    [expert_ids, tokens_per_expert.to(torch.int64)], dim=1)
            else:
                # type=2 direct output: [E, 2] (expert_id, count).
                sparse_pairs = tokens_per_expert
            # The GMM walks the pairs in order; they must keep expert order.
            group_list_type = 2
            gmm_group_list = sparse_pairs.to(torch.int64)
            swiglu_group_list = gmm_group_list
        gmm_args = {
            "x": expanded_x,
            "expert_tokens": gmm_group_list,
            "group_list_type": group_list_type,
            "swiglu_expert_tokens": swiglu_group_list,
        }
        if routing_quantizes_input:
            gmm_args["pertoken_scale"] = reshape_mx_scale(pertoken_scale)
        return gmm_args

    def moe_infer_tp(self, x, topk_ids, topk_weight, hidden_states_share, moe_events,
                     engram_resource_events=None, shared_expert_stream=None,
                     shared_merged_x=None):
        """Local routing over all experts; sum the TP-sharded FFN outputs."""
        shared_expert_events = moe_events.get("shared_expert", [])
        if not self.low_latency_tp:
            wait_event(self.enable_multi_streams and hidden_states_share is not None, shared_expert_events, 1)
        routing_args, routing_quantizes_input = self._build_moe_tp_routing_args(x, topk_ids)
        if engram_resource_events is not None:
            record_event(True, engram_resource_events, 4, self.exe_mode)
        expanded_x, row_idx, expert_tokens, pertoken_scale = torch_npu.npu_moe_init_routing_v2(
            x, **routing_args)
        if engram_resource_events is not None:
            record_event(True, engram_resource_events, 5, self.exe_mode)
        shared_h, shared_scale = None, None
        if shared_merged_x is not None:
            record_event(True, moe_events["shared_pipeline"], 0)
            with npu_stream_switch(True, shared_expert_stream):
                wait_event(True, moe_events["shared_pipeline"], 0)
                shared_h, shared_scale = self.shared_experts.swiglu_a8float8(shared_merged_x)
        gmm_args = self._build_moe_tp_gmm_args(
            expanded_x, expert_tokens, pertoken_scale, routing_quantizes_input)
        self.experts.gmm1_events = moe_events["moe_gmm1"] if shared_merged_x is not None else None
        expert_output = self.moe_ffn(**gmm_args)
        if shared_merged_x is not None:
            main_stream = torch.npu.current_stream()
            with npu_stream_switch(True, shared_expert_stream):
                wait_event(True, moe_events["moe_gmm1"], 0)
                hidden_states_share = self.shared_experts.down_proj(shared_h, shared_scale)
                record_stream(True, hidden_states_share, main_stream)
                record_event(True, shared_expert_events, 1)
        if self.low_latency_tp:
            # Only Finalize consumes the shared branch; routing and GMM are independent.
            wait_event(self.enable_multi_streams and hidden_states_share is not None, shared_expert_events, 1)
        output = torch_npu.npu_moe_finalize_routing(
            expert_output, skip1=hidden_states_share, skip2=None, bias=None,
            scales=topk_weight.to(expert_output.dtype),
            expanded_src_to_dst_row=row_idx,
            export_for_source_row=None, drop_pad_mode=2)
        # Shared experts are intermediate-dim sharded too: one reduce for
        # routed + shared partial outputs.
        dist.all_reduce(output, group=self.comm_manager.get_group("moe_tp_group"))
        return output

    def shared_expert_gate_up(self, hidden_states, shared_expert_stream, shared_expert_events):
        """Start the shared gate/up projection while the main stream computes routing."""
        record_stream(True, hidden_states, shared_expert_stream)
        record_event(True, shared_expert_events, 0)
        with npu_stream_switch(True, shared_expert_stream):
            wait_event(True, shared_expert_events, 0)
            return self.shared_experts.gate_up_proj(hidden_states.view(-1, hidden_states.shape[-1]))

    def forward_shared_expert(self, hidden_states, shared_expert_stream, shared_expert_events):
        main_stream = torch.npu.current_stream() if self.enable_multi_streams else None
        record_stream(self.enable_multi_streams, hidden_states, shared_expert_stream)
        record_event(self.enable_multi_streams, shared_expert_events, 0)
        with npu_stream_switch(self.enable_multi_streams, shared_expert_stream):
            wait_event(self.enable_multi_streams, shared_expert_events, 0)
            # shared_expert use multi streams
            hidden_states_share = self.shared_experts(hidden_states.view(-1, hidden_states.shape[-1]))
            record_event(self.enable_multi_streams, shared_expert_events, 1)
            record_stream(self.enable_multi_streams, hidden_states_share, main_stream)
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

    def moe_infer_double_routing(self, x, topk_ids, topk_weight, hidden_states_share,
                                 shared_expert_events, prefill_moe_global_chunks=None):
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

            wait_event(self.enable_multi_streams, shared_expert_events, 1)

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

    def moe_infer_dispatch_combine(self, x, topk_ids, topk_weight, shared_expert_stream, moe_events,
                                   engram_resource_events=None):
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
        # Resource boundary 4: dispatch is MIX_AIV (Vector window for wkv).
        if engram_resource_events is not None:
            record_event(True, engram_resource_events, 4,
                         self.infer_config.model_config.exe_mode)
        output = torch_npu.npu_moe_distribute_dispatch_v2(**dispatch_args)
        expand_x, dynamic_scale, expand_idx, expert_token_num, ep_recv_counts, tp_recv_counts = output[:6]

        # compute experts
        gmm_args = {
            "x": expand_x,
            "expert_tokens": expert_token_num,
            "group_list_type": 1,
        }

        if "a16" not in self.gmm_quant_mode:
            if "mxfloat" in self.gmm_quant_mode:
                # match GMM operator requirement (dim0, dim1)->(dim0, dim1//2, 2)
                dynamic_scale = reshape_mx_scale(dynamic_scale)
            gmm_args.update({"pertoken_scale": dynamic_scale})

        # Resource boundary 5: GroupedMM is Cube-heavy, releases Vector window.
        if engram_resource_events is not None:
            record_event(True, engram_resource_events, 5,
                         self.infer_config.model_config.exe_mode)
        hidden_states_ordered_by_experts = self.moe_ffn(**gmm_args)

        shared_expert_events = moe_events.get("shared_expert", [])
        shared_expert_internal_events = moe_events.get("shared_expert_internal", [])
        record_event(self.enable_multi_streams, shared_expert_internal_events, 0)
        main_stream = torch.npu.current_stream() if self.enable_multi_streams else None
        with npu_stream_switch(self.enable_multi_streams, shared_expert_stream):
            wait_event(self.enable_multi_streams, shared_expert_events, 0)
            # shared_expert use multi streams
            hidden_states_share = self.shared_experts(hidden_states.view(-1, hidden_states.shape[-1]), \
                enable_decode_stream=self.enable_multi_streams,
                shared_expert_event=shared_expert_internal_events)
            record_event(self.enable_multi_streams, shared_expert_events, 1)
            record_stream(self.enable_multi_streams, hidden_states_share, main_stream)

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
        wait_event(self.enable_multi_streams, shared_expert_events, 1)
        hidden_states = hidden_states + hidden_states_share
        hidden_states = hidden_states.view(num_tokens, self.hidden_dim)
        return hidden_states


class Attention(nn.Module):
    """Multi-Query Attention (MQA) Layer."""
    def __init__(
        self,
        config: DeepseekV41Config,
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
        self.low_latency_tp = bool(
            self.infer_config.model_config.custom_params.get("low_latency_tp", False)
        )
        self.attn_dp_size = self.infer_config.parallel_config.attn_dp_size
        self.oproj_tp_size = self.infer_config.parallel_config.o_proj_tp_size
        if self.oproj_tp_size > config.o_groups:
            raise ValueError(f"{self.oproj_tp_size=} should not be greater than {config.o_groups =}")
        self.moe_tp_size = self.infer_config.parallel_config.moe_tp_size
        self.moe_ep_size = self.infer_config.parallel_config.moe_ep_size
        self.world_size = self.infer_config.parallel_config.world_size
        self.cp_size = self.infer_config.parallel_config.cp_size
        self.platform_version = self.infer_config.model_config.platform_version
        self.layer_idx = layer_idx
        self.enable_multi_streams = self.infer_config.model_config.custom_params.get("enable_multi_streams", False)
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.attention_type = config.attention_types[layer_idx]
        self.global_rank = kwargs["global_rank"]
        self.is_mla_rank = not self.low_latency_tp or self.compress_ratio == 0 or self.global_rank % 2 == 0
        self.attn_group_name = "attn_tp_group"
        self.attn_tp_rank = self.comm_manager.get_rank(self.attn_group_name) if self.attn_tp_size > 1 else 0
        if self.low_latency_tp and self.compress_ratio > 0:
            self.separate_mla_peer_rank = self.global_rank ^ 1
            self.separate_mla_stream = kwargs["separate_mla_stream"]
            self.attn_group_name = "separate_mla_tp"
            self.attn_tp_size //= 2
            self.oproj_tp_size //= 2
            self.attn_tp_rank = self.global_rank // 2
        self.mm_quant_mode = (
            config.quant_config.mm_quant_mode
            if config.quant_config is not None
            else "w16a16")

        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads

        self.num_heads_per_rank = self.n_heads // self.attn_tp_size
        # Low-latency TP computes full-head Q on each MLA rank.
        self.q_num_heads = self.n_heads if self.low_latency_tp else self.num_heads_per_rank
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

        self.is_kv_source = layer_idx in config.kv_source_layers
        self.is_index_source = layer_idx in config.index_source_layers
        if self.low_latency_tp and self.compress_ratio > 0 and not self.is_mla_rank and self.is_kv_source:
            self.compressor_stream = kwargs["separate_mla_compressor_stream"]
        runs_mla = self.is_mla_rank
        runs_indexer = not self.low_latency_tp or self.compress_ratio == 0 or not self.is_mla_rank

        if runs_mla:
            sink_heads = self.n_heads if self.low_latency_tp and self.compress_ratio == 0 else self.num_heads_per_rank
            self.attn_sink = nn.Parameter(torch.empty(sink_heads, dtype=torch.float32))
            self.attn_sink.weight_loader = self.load_attn_sink
            self.register_buffer("full_attn_sink", None, persistent=False)
        if runs_mla or self.is_index_source:
            self.wq_a = ReplicatedLinear(self.dim,
                                         self.q_lora_rank,
                                         params_dtype=torch.bfloat16,
                                         quant_config=config.quant_config,
                                         prefix=f"{prefix}.wq_a",
                                         )
            self.q_norm = DeepseekV41RMSNorm(self.q_lora_rank, self.eps)

        if runs_mla:
            # Sparse FA consumes all Q heads; keep Q-B replicated for low-latency TP.
            self.wq_b = ColumnParallelLinear(config.q_lora_rank,
                                            self.n_heads * self.head_dim,
                                            bias=False,
                                            quant_config=config.quant_config,
                                            tp_size=1 if self.low_latency_tp else self.attn_tp_size,
                                            tp_rank=0 if self.low_latency_tp else self.attn_tp_rank,
                                            prefix=f"{prefix}.wq_b",
                                            )
            self.wkv = ReplicatedLinear(self.dim,
                                        self.head_dim,
                                        params_dtype=torch.bfloat16,
                                        quant_config=config.quant_config,
                                        prefix=f"{prefix}.wkv",
                                        )
            self.kv_norm = DeepseekV41RMSNorm(self.head_dim, self.eps)

            # consider oproj_tp
            if self.low_latency_tp and self.compress_ratio > 0:
                wo_tp_size = self.attn_tp_size
                wo_tp_rank = self.attn_tp_rank
            elif self.oproj_tp_size == 1:
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
        index_source_kv_share_layers = list(set(config.index_source_layers) - set(config.kv_source_layers))
        self.is_index_source_kv_share = layer_idx in index_source_kv_share_layers
        if self.is_kv_source and runs_indexer:
            self.compressor = Compressor(config, self.infer_config, layer_idx, self.compress_ratio,
                                         head_dim=self.head_dim, prefix=f"{prefix}.compressor",
                                         comm_manager=self.comm_manager, **kwargs)
        if self.is_index_source and runs_indexer:
            self.indexer = Indexer(config, self.infer_config, layer_idx, self.compress_ratio,
                                   prefix=f"{prefix}.indexer", comm_manager=self.comm_manager,
                                   **kwargs)
        import custom_ops  # Registers the repository's KV quantization writer.
        self.sparse_attn_ops = torch.ops.cann_ops_transformer.ds41.mixed_quant_sparse_flash_mla
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
        if attn_metadata["is_prefill"] and self.cp_size > 1:
            return self.compute_win_cache_cp(x, kv_cache, attn_metadata)
        # The recipe Linear modules handle their own weight/activation quantization.
        # Keep normalized qr in BF16 for the Indexer; quantize KV only when writing cache.
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.q_num_heads, self.head_dim))
        kv = self.kv_norm(self.wkv(x)).unsqueeze(1)
        q = self.apply_rope(q, attn_metadata)
        kv = self.apply_rope(kv, attn_metadata)

        self.update_win_kv(kv, attn_metadata["cache_slot_mapping"]["win_kv"], kv_cache.win_kv)
        win_kv = kv_cache.win_kv
        if attn_metadata["is_prefill"]:
            # The persistent SWA pages only retain the tail; early prefill queries
            # need the full current prompt. This temporary buffer is reused by layers.
            win_kv = attn_metadata["full_kv_cache"]
            self.update_win_kv(kv, attn_metadata["cache_slot_mapping"]["full_kv"], win_kv)
        return win_kv, attn_metadata["win_idx"], qr, q

    def compute_win_cache_cp(self, x: torch.Tensor, kv_cache: KVCache, attn_metadata):
        """Write this rank's window rows, then exchange the tail window of every segment.

        Each segment attends to the window that ends inside the previous segment, which lives on
        another rank, so the quantized tail rows are gathered and written into the temporary cache.
        """
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.q_num_heads, self.head_dim))
        kv = self.kv_norm(self.wkv(x)).unsqueeze(1)
        q = self.apply_rope(q, attn_metadata)
        kv = self.apply_rope(kv, attn_metadata)

        win_kv = attn_metadata["full_kv_cache"]
        # Row gather and the zigzag restore have no fp8 kernels, so the rows travel as bytes.
        cache_rows = win_kv.view(-1, win_kv.shape[-1]).view(torch.uint8)
        tail_rows = []
        for zigzag_flag, kv_seg in zip(("prev", "next"), kv.chunk(2, dim=0)):
            seg_metadata = attn_metadata[zigzag_flag]
            own_slots = seg_metadata["slot_mapping_seg"].long()
            self.update_win_kv(kv_seg, own_slots, win_kv)
            tail_rows.append(self.take_window_tails(cache_rows, own_slots, seg_metadata))

        all_win = gather_cp_segments(
            torch.cat(tail_rows, dim=0), attn_metadata["cp_metadata"],
            self.comm_manager.get_group("cp_group"))
        batch_size = attn_metadata["cp_metadata"]["batch_size"]
        all_win = all_win.view(win_kv.dtype).view(-1, batch_size, self.window_size, win_kv.shape[-1])
        for zigzag_flag in ("prev", "next"):
            seg_metadata = attn_metadata[zigzag_flag]
            if seg_metadata["is_start"]:
                continue
            scatter_cache_rows(
                win_kv, seg_metadata["slot_mapping_pre_win"],
                all_win[seg_metadata["segment_idx"] - 1].flatten(0, 1))
        if has_cp_decode_requests(attn_metadata["cp_metadata"]):
            self.update_decode_win_kv(all_win, attn_metadata, kv_cache)
        return win_kv, None, qr, q

    def take_window_tails(self, cache_rows, own_slots, seg_metadata):
        """Last window_size rows this segment holds for every request, zero padded when shorter."""
        tails, offset = [], 0
        for segment_len, cur_kv_len in zip(seg_metadata["segment_lens"], seg_metadata["cur_kv_len"]):
            slots = own_slots[offset:offset + segment_len]
            offset += segment_len
            if cur_kv_len >= self.window_size:
                tails.append(cache_rows.index_select(0, slots[cur_kv_len - self.window_size:cur_kv_len]))
            else:
                rows = cache_rows.index_select(0, slots[:cur_kv_len])
                tails.append(F.pad(rows, (0, 0, 0, self.window_size - cur_kv_len)))
        return torch.cat(tails, dim=0)

    def update_decode_win_kv(self, all_win, attn_metadata, kv_cache: KVCache):
        """Keep the window of the last valid tokens in the persistent cache of the decoding rank."""
        cp_metadata = attn_metadata["cp_metadata"]
        win_slots = cp_metadata["slot_mapping_last_win"]
        last_wins, slots = [], []
        for request in cp_metadata["owner_request_indices"].tolist():
            last_wins.append(pick_request_tail(all_win, cp_metadata, request, self.window_size))
            slots.append(win_slots[request * self.window_size:(request + 1) * self.window_size])
        scatter_cache_rows(kv_cache.win_kv, torch.cat(slots, dim=0), torch.cat(last_wins, dim=0))

    def update_win_kv(self, kv, slot_mapping, cache):
        torch.ops.custom.kv_compress_epilog_v2(
            cache=cache.view(-1, cache.shape[-1]),
            x=kv.view(-1, self.head_dim),
            slot_mapping=slot_mapping.view(-1),
            quant_group_size=MXFP4_SCALE_GROUP_SIZE,
            quant_mode="mxfp8_bf16",
            round_scale=True,
        )

    def get_runtime_cmp_cache(self, attn_metadata, kv_cache: KVCache):
        """Offline CP prefill reads the full-length temporary compressed cache."""
        tmp_cache = get_cp_tmp_cache(attn_metadata, self.compress_ratio)
        if tmp_cache is not None:
            return tmp_cache["cmp_cache"]
        return kv_cache.cmp_cache

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

        return self.get_runtime_cmp_cache(attn_metadata, kv_cache), comp_idx

    def prepare_fa_kwargs(
        self,
        q: torch.Tensor,
        win_kv: torch.Tensor,
        cmp_kv: Optional[torch.Tensor],
        cmp_sparse_indices: Optional[torch.Tensor],
        attn_metadata: Dict,
        is_prefill: bool,
    ):
        # V4.1 ratio 0 is window-only; ratio 1 still has indexed shared KV.
        has_cmp_kv = self.compress_ratio > 0
        win_key = "full_kv" if is_prefill else "win_kv"
        attn_kwargs = {
            "q": q,
            "ori_kv": win_kv.view(torch.uint8),
            "ori_sparse_indices": attn_metadata["win_sparse_indices"],
            "ori_block_table": attn_metadata["block_table"][win_key],
            "cu_seqlens_q": attn_metadata["cu_seq_lens_q"],
            "seqused_q": None,
            "seqused_ori_kv": attn_metadata["actual_seq_k"],
            "ori_topk_length": attn_metadata["win_topk_length"],
            "sinks": self.attn_sink.detach(),  # The DSL ABI requires a plain Tensor.
            "quant_mode": 1,
            "softmax_scale": self.softmax_scale,
            "layout_q": "TND",
            "layout_kv": "PA_BBND",
            "return_softmax_lse": False,
            "metadata": attn_metadata["kernel_metadata"][f"{self.attention_type}_metadata"],
        }
        if has_cmp_kv:
            # Preserve shared_idx; only the FA input uses zero-based [T, 1, K] indices.
            cmp_sparse_indices = cmp_sparse_indices.reshape(q.shape[0], 1, cmp_sparse_indices.shape[-1])
            cmp_sparse_indices = F.pad(
                cmp_sparse_indices, (0, self.config.index_topk - cmp_sparse_indices.shape[-1]), value=-1)
            ratio_key = str(self.compress_ratio)
            attn_kwargs.update({
                "cmp_kv": cmp_kv,
                "cmp_sparse_indices": cmp_sparse_indices,
                "cmp_block_table": get_cmp_block_table(attn_metadata, self.compress_ratio),
                "seqused_cmp_kv": attn_metadata["compressed_seq_lens"][ratio_key],
            })
            attn_kwargs["cmp_topk_length"] = attn_metadata["cmp_topk_length_by_ratio"][ratio_key]
        return attn_kwargs

    def sparse_attn(self, q, win_kv, cmp_kv, cmp_sparse_indices, attn_metadata, is_prefill):
        if is_prefill and self.cp_size > 1:
            if self.layer_idx == 0:
                wait_event(self.enable_multi_streams, attn_metadata["metadata_event"], 1)
            # The two zigzag segments see different causal ranges, so each runs with its own metadata.
            q_prev, q_next = q.chunk(2, dim=0)
            if cmp_sparse_indices is None:
                idx_prev, idx_next = None, None
            else:
                idx_prev, idx_next = cmp_sparse_indices.chunk(2, dim=0)
            o_prev = self.sparse_attn_ops(**self.prepare_fa_kwargs(
                q_prev, win_kv, cmp_kv, idx_prev, attn_metadata["prev"], is_prefill))[0]
            o_next = self.sparse_attn_ops(**self.prepare_fa_kwargs(
                q_next, win_kv, cmp_kv, idx_next, attn_metadata["next"], is_prefill))[0]
            output = torch.cat([o_prev, o_next], dim=0)
            return self.select_local_attn_heads(output)
        fa_kwargs = self.prepare_fa_kwargs(q, win_kv, cmp_kv, cmp_sparse_indices, attn_metadata, is_prefill)
        if self.layer_idx == 0:
            wait_event(self.enable_multi_streams, attn_metadata["metadata_event"], 1)
        return self.select_local_attn_heads(self.sparse_attn_ops(**fa_kwargs)[0])

    def select_local_attn_heads(self, output):
        if self.low_latency_tp and self.compress_ratio == 0:
            start = self.attn_tp_rank * self.num_heads_per_rank
            return output[:, start:start + self.num_heads_per_rank].contiguous()
        return output

    def sparse_attn_low_latency(self, q, win_kv, cmp_kv, cmp_sparse_indices, attn_metadata, is_prefill):
        if self.is_index_source:
            wait_event(True, attn_metadata["attention_events"]["separate_mla"], SEPARATE_MLA_DONE)
        fa_kwargs = self.prepare_fa_kwargs(q, win_kv, cmp_kv, cmp_sparse_indices, attn_metadata, is_prefill)
        fa_kwargs["sinks"] = self.full_attn_sink if self.attn_tp_size > 1 else self.attn_sink.detach()
        output = self.sparse_attn_ops(**fa_kwargs)[0]
        start = self.attn_tp_rank * self.num_heads_per_rank
        return output[:, start:start + self.num_heads_per_rank].contiguous()

    def attn_post(
        self,
        o: torch.Tensor,
        attn_metadata: Optional[Dict] = None,
        is_dspark: bool = False,
    ):
        '''
        oproj_tp: split on group dim, o: [B, S, G, N, D/G] -> [B, S, G/tp_size, N, D/G]
        transpose to make the splitted dim to be the primary
        split o_a on group dim (batch); split o_b on group dim (reduce)
        '''
        num_tokens = o.shape[0]
        if is_dspark:
            rope_slices = attn_metadata["dspark_rope_slices"]
            draft_cos, _ = rope_slices["draft"]
            draft_neg_sin = rope_slices["draft_neg_sin"]
            o = self._apply_win_rope(o, (draft_cos, draft_neg_sin))
        else:
            o = self.apply_rope(o, attn_metadata, inverse=True)
        o = o.view(num_tokens, self.num_groups_per_rank, -1).to(torch.bfloat16)

        local_num_tokens = num_tokens
        if self.oproj_tp_size > 1 and self.attn_tp_size == 1:
            oproj_max_tokens = attn_metadata.get("oproj_max_tokens", None)
            if attn_metadata["is_prefill"] and oproj_max_tokens is not None:
                o = F.pad(o, (0, 0, 0, 0, 0, oproj_max_tokens - num_tokens))
                num_tokens = oproj_max_tokens
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
        if self.oproj_tp_size > 1 and self.attn_tp_size == 1:
            # [oproj_tp_size, num_tokens, num_groups_per_rank // oproj_tp_size * o_lora_rank]
            o = o.view(self.oproj_tp_size, num_tokens, -1)
        else:
            o = o.view(num_tokens, -1)

        # o_b_proj
        x = self.wo_b(o)

        if self.low_latency_tp:
            if self.compress_ratio == 0:
                dist.all_reduce(x, group=self.comm_manager.get_group("oproj_tp_group"))
        elif self.oproj_tp_size > 1:
            # [oproj_tp_size, num_tokens, dim] --> [oproj_tp_size * num_tokens, dim]
            x = x.view(self.oproj_tp_size * num_tokens, -1)
            reduce_scatter_output = torch.empty((num_tokens, x.shape[-1]), dtype=x.dtype, device=x.device)
            dist.reduce_scatter_tensor(reduce_scatter_output, x,
                                       group=self.comm_manager.get_group("oproj_tp_group"))
            x = reduce_scatter_output.view(num_tokens, x.shape[-1])

            # trim tokens if padded in prefill
            if attn_metadata["is_prefill"] and (local_num_tokens < num_tokens):
                x = x[:local_num_tokens]
        return x

    def load_attn_sink(self, param, loaded_weight):
        if self.low_latency_tp and self.compress_ratio == 0:
            shard = loaded_weight
        else:
            rank = self.attn_tp_rank if self.low_latency_tp else 0
            shard = loaded_weight.narrow(0, rank * self.num_heads_per_rank, self.num_heads_per_rank)
        default_weight_loader(param, shard)

    def separate_mla_send_recv(self, tensor, is_prefill, payload):
        group = self.comm_manager.get_group("separate_mla_pair")
        if is_prefill:
            source = self.separate_mla_peer_rank if self.is_mla_rank else self.global_rank
            dist.broadcast(tensor, src=source, group=group)
        elif self.is_mla_rank:
            dist.recv(tensor, src=self.separate_mla_peer_rank, group=group, tag=self.layer_idx * 2 + payload)
        else:
            dist.send(tensor, dst=self.separate_mla_peer_rank, group=group, tag=self.layer_idx * 2 + payload)

    def separate_mla_send(self, tensor, attn_metadata, is_prefill, payload):
        separate_mla_events = attn_metadata["attention_events"]["separate_mla"]
        record_event(True, separate_mla_events, payload)
        record_stream(True, tensor, self.separate_mla_stream)
        with npu_stream_switch(True, self.separate_mla_stream):
            wait_event(True, separate_mla_events, payload)
            self.separate_mla_send_recv(tensor, is_prefill, payload)
            record_event(True, separate_mla_events, SEPARATE_MLA_DONE)

    def process_li_rank_inputs(self, x, attn_metadata, is_prefill, kv_cache):
        """Receive source-layer KV and indices before their first FA consumer."""
        separate_mla_events = attn_metadata["attention_events"]["separate_mla"]
        indices = torch.empty((x.shape[0], self.config.index_topk), dtype=torch.int32, device=x.device)
        # The receiver reuses the KV event to mark its input/buffer readiness.
        record_event(True, separate_mla_events, SEPARATE_MLA_KV)
        with npu_stream_switch(True, self.separate_mla_stream):
            wait_event(True, separate_mla_events, SEPARATE_MLA_KV)
            if self.is_kv_source:
                slots = attn_metadata['slot_mapping'][f'c{self.compress_ratio}a_cmp_kv']
                kv = x.new_empty((slots.numel(), self.head_dim))
                if kv.numel():
                    self.separate_mla_send_recv(kv, is_prefill, SEPARATE_MLA_KV)
                    write_compressed_kv(kv, slots, kv_cache.cmp_cache)
            self.separate_mla_send_recv(indices, is_prefill, SEPARATE_MLA_INDICES)
            record_event(True, separate_mla_events, SEPARATE_MLA_DONE)
        record_stream(True, indices, self.separate_mla_stream)
        return indices

    def attention_prolog_mla_rank(self, x, attn_metadata, is_prefill, kv_cache):
        if self.is_index_source:
            kv_cache.shared_idx = self.process_li_rank_inputs(x, attn_metadata, is_prefill, kv_cache)
        win_kv, _, _, q = self.compute_win_cache(x, kv_cache, attn_metadata)
        return win_kv, q

    def attention_prolog_li_rank(self, x, attn_metadata, is_prefill, kv_cache):
        if not self.is_index_source:
            return
        if is_prefill:
            qr = self.q_norm(self.wq_a(x))
            latent = None
            if self.is_kv_source:
                latent = self.compress_and_send_kv(x, attn_metadata, True, kv_cache)
            indices = self.indexer(x, qr, attn_metadata, latent, kv_cache)
        else:
            indices = self.decode_indexer(x, attn_metadata, kv_cache)
        self.separate_mla_send(indices.contiguous(), attn_metadata, is_prefill, SEPARATE_MLA_INDICES)
        wait_event(True, attn_metadata["attention_events"]["separate_mla"], SEPARATE_MLA_DONE)

    def decode_indexer(self, x, attn_metadata, kv_cache):
        """Overlap Q/W preparation with compressed KV and Indexer K updates."""
        attention_events = attn_metadata["attention_events"]
        use_fused_prolog_qw = self.indexer.use_fused_prolog_qw
        if not use_fused_prolog_qw:
            main_stream = torch.npu.current_stream()
            record_event(True, attention_events["indexer_weights"], 0)
            record_stream(True, x, self.indexer_weights_stream)
            with npu_stream_switch(True, self.indexer_weights_stream):
                wait_event(True, attention_events["indexer_weights"], 0)
                weights = self.indexer.prepare_weights(x)
                record_event(True, attention_events["indexer_weights"], 1)
            record_stream(True, weights, main_stream)
        if self.is_kv_source:
            record_event(True, attention_events["compressor"], 0)
            record_stream(True, x, self.compressor_stream)
            with npu_stream_switch(True, self.compressor_stream):
                wait_event(True, attention_events["compressor"], 0)
                latent = self.compress_and_send_kv(x, attn_metadata, False, kv_cache)
                self.indexer.update_indexer_cache(latent, attn_metadata, kv_cache)
                record_event(True, attention_events["compressor"], 1)
        qr = self.q_norm(self.wq_a(x))
        cos, sin = attn_metadata["cos_sin"]["comp"]
        if use_fused_prolog_qw:
            q, q_scale, weights = OpKernel.indexer_prolog_qw(self.indexer, x, qr, cos, sin)
        else:
            q, q_scale = self.indexer.prepare_query(qr, cos, sin)
            wait_event(True, attention_events["indexer_weights"], 1)
        if self.is_kv_source:
            wait_event(True, attention_events["compressor"], 1)
        return self.indexer.select_indices(q, q_scale, weights, attn_metadata, kv_cache)

    def compress_and_send_kv(self, x, attn_metadata, is_prefill, kv_cache):
        kv, latent = self.compressor.compressor_prolog(x, attn_metadata, kv_cache)
        if kv.numel():
            self.separate_mla_send(kv.contiguous(), attn_metadata, is_prefill, SEPARATE_MLA_KV)
        return latent

    def forward_separate_mla(self, x, attn_metadata, is_prefill, kv_cache):
        if self.is_mla_rank:
            win_kv, q = self.attention_prolog_mla_rank(x, attn_metadata, is_prefill, kv_cache)
            o = self.sparse_attn_low_latency(
                q, win_kv, kv_cache.cmp_cache, kv_cache.shared_idx, attn_metadata, is_prefill,
            )
            output = self.attn_post(o, attn_metadata)
        else:
            self.attention_prolog_li_rank(x, attn_metadata, is_prefill, kv_cache)
            output = torch.zeros_like(x)
        # MLA shards contribute once; the same result reaches every MoE/mHC rank.
        dist.all_reduce(output, group=self.comm_manager.get_group("oproj_tp_group"))
        return output

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
        if self.low_latency_tp and self.compress_ratio > 0:
            return self.forward_separate_mla(x, attn_metadata, is_prefill, kv_cache)
        win_kv, win_idx, qr, q = self.compute_win_cache(x, kv_cache, attn_metadata)
        if is_prefill:
            block_table = attn_metadata["block_table"]["full_kv"]
        else:
            block_table = attn_metadata["block_table"]["win_kv"]
        offset = block_table.numel() * win_kv.shape[1]
        compress_kv, compress_idxs = None, None
        if self.compress_ratio > 0:
            compress_kv, compress_idxs = self.compute_comp_kv(x, qr, kv_cache, attn_metadata, is_prefill, offset)
        o = self.sparse_attn(q, win_kv, compress_kv, compress_idxs, attn_metadata, is_prefill)
        x = self.attn_post(o, attn_metadata)

        return x


class DeepseekV41DecoderLayer(nn.Module):
    def __init__(
        self,
        config: DeepseekV41Config,
        infer_config: InferenceConfig,
        comm_manager: CommManager = None,
        layer_idx: int = 0,
        prefix: str = "",
        engram_layout: EngramLayout = None,
        is_mtp: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.hidden_size = config.hidden_size
        if not is_mtp:
            self.attn = Attention(
                config=config,
                infer_config=self.infer_config,
                comm_manager=self.comm_manager,
                layer_idx=layer_idx,
                prefix=f"{prefix}.attn",
                **kwargs)

        self.ffn = (
            DeepseekV41MoE(
                config,
                self.infer_config,
                self.comm_manager,
                layer_idx=layer_idx,
                prefix=f"{prefix}.mlp",
                is_mtp=is_mtp,
                **kwargs,
            )
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

        self.attn_norm = DeepseekV41RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = DeepseekV41RMSNorm(config.hidden_size, config.rms_norm_eps)
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
        engram_precomputed=None,
        engram_resource_events=None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:
        if self.engram is not None and (engram_hashes is not None or engram_precomputed is not None):
            engram_metadata = attn_metadata.get("engram", {})
            hidden_states = self.engram(
                hidden_states, engram_hashes, is_prefill=is_prefill,
                engram_metadata=engram_metadata, image_mask=image_mask,
                precomputed=engram_precomputed)
        residual = hidden_states
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
        hidden_states = OpKernel.hc_post(
            hidden_states, residual, post, comb, is_prefill=is_prefill
        )

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
            engram_resource_events=engram_resource_events,
            moe_events=attn_metadata.get("moe_events"),
        )
        hidden_states = OpKernel.hc_post(
            hidden_states, residual, post, comb, is_prefill=is_prefill
        )
        return hidden_states, ffn_pre


@add_start_docstrings(
    "The bare DeepseekV41 Model outputting raw hidden-states without any specific head on top.",
    DEEPSEEKV41_START_DOCSTRING,
)
class DeepseekV41Model(DeepseekV41PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DeepseekV41DecoderLayer`]

    Args:
        config: DeepseekV41Config
    """

    def __init__(
        self,
        config: DeepseekV41Config,
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
        self.oproj_tp_size = self.infer_config.parallel_config.o_proj_tp_size
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.vocab_size_per_rank = self.vocab_size // self.embed_tp_size
        self.global_rank = kwargs.get("global_rank")
        self.enable_superkernel = self.infer_config.model_config.custom_params.get("enable_superkernel", False)
        self.enable_multi_streams = self.infer_config.model_config.custom_params.get("enable_multi_streams", False)
        self.collect_layer_ids = self.config.dspark_target_layer_ids

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
        self.with_engram = self.engram_layout is not None

        self.layers = nn.ModuleList(
            [
                DeepseekV41DecoderLayer(
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
        self.norm = DeepseekV41RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Engram multi-stream configuration
        custom_params = self.infer_config.model_config.custom_params
        self.engram_tp_size = int(custom_params.get("engram_tp_size", 1))
        self.enable_engram_offload = bool(custom_params.get("enable_engram_offload", False))
        self.enable_engram_multi_stream = bool(
            custom_params.get("enable_engram_multi_stream", self.enable_multi_streams)
        )
        exe_mode = self.infer_config.model_config.exe_mode
        self.engram_stream = None
        self.engram_layer_ids = ()
        self.engram_ready_events = ()
        self.engram_resource_events = None
        if self.enable_engram_multi_stream and self.with_engram:
            self.engram_stream = create_stream("engram_precompute", exe_mode)
            self.engram_layer_ids = tuple(
                self.engram_layout.layer_ids if self.engram_layout else ())
            self.engram_ready_events = tuple(
                create_event(exe_mode, True) for _ in self.engram_layer_ids)
            # Resource events gate Cube/Vector overlap on Layer-0 (ev[5] =
            # post-GroupedMM boundary, consumed by the wkv wait on the side
            # stream).  Eager only.
            if exe_mode not in ("ge_graph", "npugraph_ex"):
                self.engram_resource_events = tuple(
                    create_event(exe_mode, True) for _ in range(6)
                )

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
        self.indexer_cache_scale_by_layer = {
            source: torch.tensor([]) for source in self.kv_source_layers
        }
        index_layers = set(self.config.index_source_layers)
        qsli_sources = {
            source
            for source, consumers in self.source_to_layers.items()
            if any(consumer > source and consumer in index_layers for consumer in consumers)
        }
        # QSLI reads the combined cache owned by its source layer; QLI-only
        # sources do not need a QSLI cache placeholder.
        self.indexer_qsli_cache_by_layer = {
            source: torch.tensor([]) for source in qsli_sources
        }
        # layer --> win_kv
        self.win_kv_by_layer = {i: torch.tensor([]) for i in range(len(self.layers))}
        # source_layer & cmp_ratio == 2 --> state_cache
        self.state_cache_by_layer = {source: torch.tensor([]) for source in compressed_sources}
        # layer --> kv_cache
        self.kv_cache_by_layer = {
            i: KVCache(i, self.layer_to_source.get(i)) for i in range(len(self.layers))
        }

        self.shared_idx = None
        self.candidates = None

        self.head_dim = self.config.head_dim
        self.indexer_head_dim = self.config.index_head_dim
        self.cache_dtype = PACKED_KV_STORAGE_DTYPE
        self.win_cache_dtype = PACKED_KV_COMPUTE_DTYPE
        self.block_size = self.infer_config.scheduler_config.block_size
        win_cache_dim = get_kv_cache_dim(self.head_dim)
        cmp_cache_dim = get_kv_cache_dim(self.head_dim, is_compressed=True)
        if self.indexer_head_dim % MXFP4_SCALE_GROUP_SIZE:
            raise ValueError(
                f"MXFP4 index_head_dim must be divisible by {MXFP4_SCALE_GROUP_SIZE}, "
                f"got {self.indexer_head_dim}"
            )
        self.indexer_cache_dim = self.indexer_head_dim // MXFP4_VALUES_PER_BYTE
        self.indexer_cache_dtype = torch.uint8
        self.indexer_cache_scale_dim = self.indexer_head_dim // MXFP4_SCALE_GROUP_SIZE
        self.indexer_cache_scale_dtype = torch.uint8
        candidate_block_size = int(getattr(self.config, "candidate_block_size", 8))
        if self.block_size % candidate_block_size != 0:
            raise ValueError(
                "scheduler block_size must be divisible by candidate_block_size, "
                f"got block_size={self.block_size}, candidate_block_size={candidate_block_size}"
            )
        self.indexer_qsli_candidate_block_size = candidate_block_size
        self.indexer_qsli_cache_dim = candidate_block_size * (
            self.indexer_cache_dim + self.indexer_cache_scale_dim
        )
        self.indexer_qsli_cache_dtype = torch.uint8
        self.state_cache_dtype = torch.float32
        self.block_size = self.infer_config.scheduler_config.block_size
        self.next_n = self.infer_config.speculative_config.num_speculative_tokens
        self.window_size = self.config.sliding_window
        self._entry_specs = (
            ("win_kv", self.win_kv_by_layer, _Scope.PER_LAYER),
            ("cmp_cache", self.cmp_cache_by_layer, _Scope.SHARED),
            ("indexer_cache_scale", self.indexer_cache_scale_by_layer, _Scope.SHARED),
            ("indexer_cache", self.indexer_cache_by_layer, _Scope.SHARED),
            ("indexer_qsli_cache", self.indexer_qsli_cache_by_layer, _Scope.SHARED),
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
                dim=win_cache_dim,
                num_head=1,
                dtype=self.win_cache_dtype,
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

            add_entry(common, layer_idx, f"cmp_cache_src{layer_idx}", cmp_cache_dim, self.cache_dtype)
            add_entry(common, layer_idx, f"indexer_cache_src{layer_idx}", self.indexer_cache_dim,
                      self.indexer_cache_dtype)
            add_entry(common, layer_idx, f"indexer_cache_scale_src{layer_idx}",
                      self.indexer_cache_scale_dim, self.indexer_cache_scale_dtype)

            # QSLI consumes source-layer K/V in candidate-sized combined
            # blocks. QLI-only sources keep the split cache pair above.
            has_qsli_consumer = any(
                consumer > layer_idx and consumer in index_layers
                for consumer in self.source_to_layers.get(layer_idx, [])
            )
            if has_qsli_consumer:
                qsli_common = dict(
                    attn_type="FullAttention",
                    needs_block=True,
                    block_size=(self.block_size * ratio) // self.indexer_qsli_candidate_block_size,
                    manager_key=f"c{ratio}a_qsli_combined_kv",
                    compress_ratio=1,
                )
                add_entry(
                    qsli_common,
                    layer_idx,
                    f"indexer_qsli_cache_src{layer_idx}",
                    self.indexer_qsli_cache_dim,
                    self.indexer_qsli_cache_dtype,
                )

            if ratio == 1:
                continue
            # create state_cache for source layers (cmp_ratio > 1)
            block_size = ratio if self.next_n == 0 else self.next_n + 1
            cache_entries.append(CacheEntry(
                cache_name=f"state_cache_src{layer_idx}",
                attn_type="RingCache",
                dim=self.head_dim,
                num_head=2,
                dtype=torch.float32,
                needs_block=True,
                block_size=block_size,
                manager_key=f"c{ratio}a_cmp_state",
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

    def update_cp_cos_sin(self, attn_metadata, hidden_states, kv_len):
        for zigzag_flag in ["prev", "next"]:
            position_ids_cur = attn_metadata[zigzag_flag]["position_ids_cur"]
            cos_sin = {
                "win": self.rotary_emb(hidden_states, position_ids_cur, kv_len, self.max_position_embeddings),
            }
            cos_sin.update({"win_neg_sin": -cos_sin["win"][1]})

            position_ids_cmp = attn_metadata[zigzag_flag]["position_ids_cmp_for_rope"]
            cos_sin.update({
                "comp": self.compress_rotary_emb(
                    hidden_states, position_ids_cur, kv_len, self.max_position_embeddings),
                "c2a": self.compress_rotary_emb(
                    hidden_states, position_ids_cmp["2"], kv_len, self.max_position_embeddings),
            })
            cos_sin.update({"comp_neg_sin": -cos_sin.get("comp")[1]})
            cos_sin["c1a"] = cos_sin["comp"]
            attn_metadata[zigzag_flag].update({
                "cos_sin": cos_sin,
            })

        # Local tokens are laid out as [prev | next]; token-wise RoPE uses the concatenated tables.
        prev_cos_sin, next_cos_sin = attn_metadata["prev"]["cos_sin"], attn_metadata["next"]["cos_sin"]
        cos_sin = {}
        for key in ("win", "comp"):
            cos_sin[key] = tuple(torch.cat([p, n], dim=0) for p, n in zip(prev_cos_sin[key], next_cos_sin[key]))
            cos_sin[f"{key}_neg_sin"] = torch.cat(
                [prev_cos_sin[f"{key}_neg_sin"], next_cos_sin[f"{key}_neg_sin"]], dim=0)
        cos_sin["c1a"] = cos_sin["comp"]
        attn_metadata["cos_sin"] = cos_sin

    def generate_cos_sin(self, attn_metadata, hidden_states, is_mtp=False):
        # WIN uses base RoPE; C2A/C1A Attention and Indexer use comp.
        # C1A Compressor uses per-token comp; C2A Compressor uses group positions in c2a.
        position_ids = attn_metadata["position_ids"]
        kv_len = attn_metadata["kv_len"]
        cos_sin = {
            "win": self.rotary_emb(hidden_states, position_ids, kv_len, self.max_position_embeddings),
        }
        cos_sin.update({"win_neg_sin": -cos_sin["win"][1]})
        if is_mtp:
            return cos_sin

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

    def merge_visual_embeddings(self, hidden_states, full_input_ids, visual_embeddings, attn_metadata=None):
        image_mask = full_input_ids == self.config.image_token_id
        image_token_count = int(image_mask.sum().item())
        if image_token_count != visual_embeddings.shape[0]:
            raise RuntimeError(
                f"Visual embedding rows ({visual_embeddings.shape[0]}) do not match "
                f"image tokens ({image_token_count})")
        visual_ordinals = image_mask.long().cumsum(0) - 1
        if attn_metadata is not None:
            # CP: locate images on the full input_ids, then keep the positions held by this rank.
            image_mask = select_cp_segments(image_mask, attn_metadata["cp_metadata"])
            visual_ordinals = select_cp_segments(visual_ordinals, attn_metadata["cp_metadata"])
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

    def get_prefill_oproj_max_tokens(self, hidden_states, is_prefill):
        # get global max tokens
        if not is_prefill or self.oproj_tp_size <= 1:
            return None
        local_num_token = hidden_states.shape[0]
        max_num_token = torch.tensor([local_num_token], dtype=torch.int32, device=hidden_states.device)
        dist.all_reduce(max_num_token, op=dist.ReduceOp.MAX, group=self.comm_manager.get_group("oproj_tp_group"))
        return max_num_token

    def compute_engram_hashes(self, input_ids, is_prefill, attn_metadata,
                              prefix_input_ids=None, ngram_shift_mask=None):
        """Hash the full sequence, then keep this rank's segments: n-grams reach across segments."""
        hashes = self.engram_hash(
            input_ids, is_prefill,
            prefix_input_ids=prefix_input_ids,
            ngram_shift_mask=ngram_shift_mask)
        if is_prefill and self.cp_size > 1:
            hashes = select_cp_segments(hashes.squeeze(0), attn_metadata["cp_metadata"]).unsqueeze(0)
        return hashes

    def _schedule_engram_multistream(self, input_ids, is_prefill, exe_mode, attn_metadata,
                                     prefix_input_ids=None, ngram_shift_mask=None):
        """Submit hash (Stage 0) + embedding (Stage 1) for the early engram
        layer to the side stream before the layer loop starts.

        wkv (Stage 2) is submitted later from _run_decoder_layer after
        Layer-0's GroupedMM completes, so wkv (Cube) does not contend
        with GroupedMM (Cube).

        Returns: (pending, side_pending, engram_hashes)
            pending:       {layer_idx: (key,value,weight)} — empty initially;
                           early entry filled after Layer-0.
            side_pending:  mutable dict carrying embeddings
                           from here to _run_decoder_layer; None if disabled.
            engram_hashes: hash tensor computed on the side stream; needed
                           for late-layer submission; None if disabled.
        """
        if not (self.enable_engram_multi_stream and self.with_engram):
            return {}, None, None

        early = self.engram_layer_ids[0] if self.engram_layer_ids else None
        if (early is None or early >= len(self.layers)
                or getattr(self.layers[early], "engram", None) is None):
            return {}, None, None

        # Mutable result table populated by the decoder layer loop.
        pending = {}
        side_pending = {'stage': 'init'}

        with npu_stream_switch(True, self.engram_stream, exe_mode=exe_mode):
            engram_hashes = self.compute_engram_hashes(
                input_ids, is_prefill, attn_metadata,
                prefix_input_ids=prefix_input_ids,
                ngram_shift_mask=ngram_shift_mask)
            early_layer = self.layers[early]
            layer_hash_index = early_layer.engram.layer_hash_index
            layer_engram_hash = engram_hashes[:, :, layer_hash_index, :]
            embeddings = early_layer.engram.precompute_embeddings(
                layer_engram_hash, is_prefill=is_prefill)
            side_pending['embeddings'] = embeddings
            side_pending['stage'] = 'embeddings_done'

        record_stream(True, embeddings, self.engram_stream, exe_mode)

        return pending, side_pending, engram_hashes

    def _submit_late_engram(self, engram_hashes, is_prefill, exe_mode):
        """Submit the late engram layer after the early layer's gating is done.

        Called from the layer loop once the early engram consumer has returned.
        No resource gating is needed — the late precompute overlaps with the
        intermediate decoder layers (2..N-2), which provide enough mixed
        Cube/Vector work to avoid contention.
        """
        if len(self.engram_layer_ids) < 2:
            return None
        late = self.engram_layer_ids[-1]
        if (late >= len(self.layers)
                or getattr(self.layers[late], "engram", None) is None):
            return None
        late_layer = self.layers[late]
        late_idx = len(self.engram_layer_ids) - 1
        with npu_stream_switch(True, self.engram_stream, exe_mode=exe_mode):
            layer_hash_index = late_layer.engram.layer_hash_index
            layer_engram_hash = engram_hashes[:, :, layer_hash_index, :]
            kv = late_layer.engram.precompute(layer_engram_hash, is_prefill=is_prefill)
            record_event(True, self.engram_ready_events, late_idx, exe_mode)
        for tensor in kv:
            record_stream(True, tensor, self.engram_stream, exe_mode)
        return kv

    def _wait_engram_ready(self, decoder_layer, pending_engram, exe_mode):
        """Wait for the side stream to finish before consuming precomputed results."""
        layer_idx = decoder_layer.layer_idx
        precomputed = pending_engram.get(layer_idx)
        if precomputed is None:
            return
        if layer_idx in self.engram_layer_ids:
            wait_event(True, self.engram_ready_events,
                       self.engram_layer_ids.index(layer_idx), exe_mode)

    def _run_decoder_layer(self, decoder_layer, hidden_states, pre_mix, attn_metadata,
                           residual, is_prefill, cur_topk_list, input_ids, image_mask,
                           prefill_moe_global_chunks,
                           engram_hashes, pending_engram, exe_mode,
                           engram_side_pending=None):
        """Invoke one decoder layer with appropriate engram arguments."""
        layer_idx = decoder_layer.layer_idx
        consumes_engram = getattr(decoder_layer, "engram", None) is not None
        use_engram_ms = self.enable_engram_multi_stream and self.with_engram

        current_precomputed = pending_engram.get(layer_idx) if consumes_engram else None
        layer_engram_hashes = None
        if not use_engram_ms and engram_hashes is not None and consumes_engram:
            layer_hash_index = decoder_layer.engram.layer_hash_index
            layer_engram_hashes = engram_hashes[:, :, layer_hash_index, :]

        # Only Layer-0 needs resource events (ev[5] = post-GroupedMM
        # boundary consumed by the wkv wait on the side stream).
        resource_events = self.engram_resource_events if layer_idx == 0 else None

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
            kv_cache=kv_cache,
            engram_hashes=layer_engram_hashes,
            engram_precomputed=current_precomputed,
            engram_resource_events=resource_events,
        )
        self.shared_idx = kv_cache.shared_idx
        self.candidates = kv_cache.candidates

        early = self.engram_layer_ids[0] if self.engram_layer_ids else None

        # After Layer-0 completes: submit wkv (Stage 2) to the side stream.
        # wkv waits on ev[5] (post-GroupedMM) so it only starts after
        # GroupedMM (Cube) finishes, avoiding Cube contention.  The main
        # stream continues to Layer-1 immediately.
        if (use_engram_ms and layer_idx == 0 and engram_side_pending is not None
                and engram_side_pending.get('stage') == 'embeddings_done'
                and early is not None):
            early_layer = self.layers[early]
            with npu_stream_switch(True, self.engram_stream, exe_mode=exe_mode):
                wait_event(self.engram_resource_events is not None,
                           self.engram_resource_events, 5, exe_mode)
                kv = early_layer.engram.precompute_kv(
                    engram_side_pending['embeddings'], is_prefill=is_prefill)
                record_event(True, self.engram_ready_events, 0, exe_mode)
            for tensor in kv:
                record_stream(True, tensor, self.engram_stream, exe_mode)
            pending_engram[early] = kv

        # After the early engram layer's gating completes on the main stream,
        # submit the late engram layer to the same side stream.  By this point
        # the early precompute is already consumed, so the late precompute
        # cannot block the early ready event.
        if use_engram_ms and layer_idx == early and len(self.engram_layer_ids) > 1:
            late_result = self._submit_late_engram(
                engram_hashes, is_prefill, exe_mode)
            if late_result is not None:
                pending_engram[self.engram_layer_ids[-1]] = late_result

        return hidden_states, pre_mix

    @add_start_docstrings_to_model_forward(DEEPSEEKV41_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.Tensor,
        attn_metadata: Optional[Dict] = None,
        is_prefill: Optional[bool] = False,
        cur_topk_list: Optional[torch.Tensor] = None,
        visual_embeddings: Optional[torch.Tensor] = None,
    ):
        full_input_ids = input_ids
        exe_mode = self.infer_config.model_config.exe_mode
        if is_prefill and self.cp_size > 1 and self.embed_tp_size == 1:
            input_ids = select_cp_segments(input_ids, attn_metadata["cp_metadata"])

        inputs_embeds = self.calc_input_embeddings(input_ids, is_prefill)
        hidden_states = inputs_embeds
        image_mask = None
        if visual_embeddings is not None:
            cp_local_hidden = is_prefill and self.cp_size > 1 and self.embed_tp_size == 1
            hidden_states, image_mask = self.merge_visual_embeddings(
                hidden_states, full_input_ids, visual_embeddings,
                attn_metadata if cp_local_hidden else None,
            )
        if is_prefill and self.cp_size > 1 and self.embed_tp_size > 1:
            # Keep full input_ids for embedding TP, then switch hidden states to attention CP layout.
            input_ids = select_cp_segments(input_ids, attn_metadata["cp_metadata"])
            hidden_states = select_cp_segments(inputs_embeds, attn_metadata["cp_metadata"])
            if image_mask is not None:
                image_mask = select_cp_segments(image_mask, attn_metadata["cp_metadata"])
            del inputs_embeds

        kv_len = attn_metadata["kv_len"]
        if is_prefill and self.cp_size > 1:
            self.update_cp_cos_sin(attn_metadata, hidden_states, kv_len)
        else:
            cos_sin = self.generate_cos_sin(attn_metadata, hidden_states)
            attn_metadata.update({'cos_sin': cos_sin})

        residual = None
        prefill_moe_global_chunks = self.get_prefill_moe_global_chunks(hidden_states, is_prefill)
        attn_metadata["oproj_max_tokens"] = self.get_prefill_oproj_max_tokens(hidden_states, is_prefill)

        # Engram hash inputs
        engram_meta = attn_metadata.get("engram", {})
        prefix_input_ids = engram_meta.get("prefix_input_ids")
        ngram_shift_mask = engram_meta.get("shift_mask")

        # mhc init one-hot pre_mix
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        pre_mix = make_identity_pre_mix(hidden_states, self.hc_mult)

        collected_hidden_states = []
        # --- Engram multi-stream scheduling ---
        # Multistream: hash + embedding on side stream; wkv after Layer-0.
        # Non-multistream: hash on main stream.
        pending_engram, side_pending, engram_hashes = self._schedule_engram_multistream(
            full_input_ids, is_prefill, exe_mode, attn_metadata,
            prefix_input_ids=prefix_input_ids,
            ngram_shift_mask=ngram_shift_mask)
        if engram_hashes is None and self.with_engram:
            engram_hashes = self.compute_engram_hashes(
                full_input_ids, is_prefill, attn_metadata,
                prefix_input_ids=prefix_input_ids,
                ngram_shift_mask=ngram_shift_mask)

        for layer_idx, decoder_layer in enumerate(self.layers):
            if layer_idx in self.collect_layer_ids:
                collected_hidden_states.append(hidden_states.mean(dim=1))
            self._wait_engram_ready(decoder_layer, pending_engram, exe_mode)
            hidden_states, pre_mix = self._run_decoder_layer(
                decoder_layer, hidden_states, pre_mix, attn_metadata, residual,
                is_prefill, cur_topk_list, input_ids, image_mask,
                prefill_moe_global_chunks,
                engram_hashes, pending_engram, exe_mode,
                engram_side_pending=side_pending)
            pending_engram.pop(layer_idx, None)
        self.shared_idx = None
        self.candidates = None

        hidden_states = hc_pre_mix(hidden_states, pre_mix)
        hidden_states = self.norm(hidden_states)
        if self.collect_layer_ids:
            return hidden_states, torch.cat(collected_hidden_states, dim=-1)
        return hidden_states


class DeepseekV41ForCausalLM(DeepseekV41PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def set_auxiliary_hidden_layers(self, layer_ids) -> None:
        """Enable generic intermediate hidden-state collection for a consumer."""
        layer_ids = tuple(layer_ids or ())
        if any(layer_id < 0 or layer_id >= self.config.num_hidden_layers for layer_id in layer_ids):
            raise ValueError("Auxiliary hidden layer ids must refer to main-model layers.")
        self.model.collect_layer_ids = frozenset(layer_ids)

    def __init__(
        self,
        config,
        infer_config: InferenceConfig,
        comm_manager: CommManager = None,
        prefix: str = "",
        is_mtp: bool = False
    ):
        super().__init__(config)
        self.config = config
        self.infer_config = infer_config
        self.comm_manager = comm_manager
        self.is_mtp = is_mtp
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
        self.low_latency_tp = bool(
            self.infer_config.model_config.custom_params.get("low_latency_tp", False)
        )
        self.disagg_mode = self.infer_config.disagg_config.disaggregation_mode
        self.enable_mega_moe = (
            self.infer_config.model_config.custom_params.get("enable_mega_moe", False)
            and self.disagg_mode in ("NONE", "PREFILL")
            and not self.is_mtp
        )
        self.li_cache_quant_mode = config.quant_config.li_cache_quant_mode
        self.attention_data = AttnMetaData(self.config, comm_manager, self.infer_config, is_mtp)
        self.enable_cache_compile = self.infer_config.model_config.enable_cache_compile

        self.enable_static_kernel = self.infer_config.model_config.enable_static_kernel
        self.enable_npugraph_ex = self.infer_config.model_config.exe_mode == "npugraph_ex"
        self.enable_multi_streams = self.infer_config.model_config.custom_params.get("enable_multi_streams", False)
        self.engram_tp_size = self.infer_config.model_config.custom_params.get("engram_tp_size", 1)
        self.enable_engram_offload = self.infer_config.model_config.custom_params.get("enable_engram_offload", False)

        self.metadata_event = []
        if self.enable_multi_streams:
            # event[0]: metadata stream may start after the main stream inputs
            # are ready; event[1]: sparse-attention metadata is ready; event[2]:
            # QLI/QSLI metadata is ready for the first Indexer layer.
            self.metadata_event = [torch.npu.Event() for _ in range(3)]

        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.rank_offset = int(os.getenv("RANK_OFFSET", "0"))
        self.global_rank = self.local_rank + self.rank_offset
        self.world_size = self.infer_config.parallel_config.world_size
        kwargs = {"global_rank": self.global_rank}
        if not self.is_mtp:
            self.with_engram = True
        else:
            self.with_engram = False
        self.init_parallel_comm_group()
        if self.low_latency_tp:
            kwargs["separate_mla_stream"] = create_stream("separate_mla")
            if self.global_rank % 2:
                kwargs["separate_mla_compressor_stream"] = create_stream("separate_mla_compressor")
        self.megamoe_context = None
        self.megamoe_backend = None
        self.selected_op_kernels = {}
        if self.enable_mega_moe:
            self.update_op_kernel_dict(("megamoe",))
            self.megamoe_context, self.megamoe_backend = OpKernel.megamoe(
                config,
                get_max_local_prefill_tokens(self.infer_config, config.sliding_window),
                self.comm_manager,
                ep_group_name="dspark_megamoe_ep_group" if is_mtp else "megamoe_ep_group",
            )
            kwargs["megamoe_backend"] = self.megamoe_backend
        self.batch_size_per_rank = self.infer_config.scheduler_config.batch_size_per_dp_rank
        if not is_mtp:
            self.model = DeepseekV41Model(config, self.infer_config, self.comm_manager, prefix, **kwargs)

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

        # Initialize weights and apply final processing
        self.post_init()
        self.block_size = self.infer_config.scheduler_config.block_size
        self.window_size = config.sliding_window
        self.cp_segment_min_len = self.window_size
        # Rows per request that prefill CP hands to a draft model.
        self.cp_prefill_tail_tokens = self.window_size
        self.init_cache_dim()
        self.first_layer_idx = 0
        self.first_layer_ratio = self.config.compress_ratios[self.first_layer_idx]
        self.sparse_attn_metadata_ops = torch.ops.cann_ops_transformer.ds41.mixed_quant_sparse_flash_mla_metadata
        from cann_ops_transformer.ops.ds41 import (
            quant_lightning_indexer_metadata,
            quant_sparse_lightning_indexer_metadata,
        )
        self.quant_lightning_indexer_metadata_ops = quant_lightning_indexer_metadata
        self.quant_sparse_lightning_indexer_metadata_ops = quant_sparse_lightning_indexer_metadata

    @staticmethod
    def check_model_config_before_loading(config, infer_config):
        model_config = infer_config.model_config
        custom_params = model_config.custom_params
        parallel_config = infer_config.parallel_config
        scheduler_config = infer_config.scheduler_config

        kv_cache_quant_mode = config.quant_config.kv_cache_quant_mode
        if kv_cache_quant_mode != "float8":
            raise ValueError(
                f"{kv_cache_quant_mode=} is unsupported; DeepSeek V4.1 requires "
                "quantization_config.kv_cache_scheme with type='float' and num_bits=8."
            )

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

        if (custom_params.get("enable_mega_moe", False)
                and infer_config.disagg_config.disaggregation_mode in ("NONE", "PREFILL")):
            if parallel_config.moe_ep_size <= 1:
                raise ValueError(
                    f"enable_mega_moe=True with moe_ep_size={parallel_config.moe_ep_size} "
                    "is not supported yet."
                )
            if parallel_config.moe_tp_size != 1:
                raise ValueError(
                    f"enable_mega_moe=True with moe_tp_size={parallel_config.moe_tp_size} "
                    "is not supported yet."
                )
            quant_config = getattr(config, "quant_config", None)
            gmm_quant_mode = None if quant_config is None else quant_config.gmm_quant_mode
            # "w4a8float4" is accepted because update_gmm_quant_mode normalizes it
            # to "w4a8mxfloat4" during model construction (MegaMoE only runs there).
            if gmm_quant_mode not in ("w4a8mxfloat4", "w4a8float4"):
                raise ValueError(
                    f"enable_mega_moe=True with gmm_quant_mode={gmm_quant_mode} "
                    "is not supported yet."
                )

        exe_mode = model_config.exe_mode
        enable_cache_compile = model_config.enable_cache_compile
        enable_superkernel = custom_params.get("enable_superkernel", False)
        unsafe_skip_capture_validation = custom_params.get(
            "unsafe_skip_npugraph_capture_validation", False
        )
        moe_chunk_max_len = custom_params.get("moe_chunk_max_len", 65536)
        with_ckpt = model_config.with_ckpt

        if not isinstance(unsafe_skip_capture_validation, bool):
            raise TypeError(
                "custom_params.unsafe_skip_npugraph_capture_validation must be a boolean"
            )
        if unsafe_skip_capture_validation and exe_mode != "npugraph_ex":
            raise ValueError(
                "custom_params.unsafe_skip_npugraph_capture_validation only supports "
                "exe_mode='npugraph_ex'; "
                f"got exe_mode='{exe_mode}'"
            )
        if (
            isinstance(moe_chunk_max_len, bool)
            or not isinstance(moe_chunk_max_len, int)
            or moe_chunk_max_len <= 0
        ):
            raise ValueError(f"{moe_chunk_max_len=} should be a positive integer.")
        if not with_ckpt and not model_config.force_eplb:
            raise ValueError(f"{model_config.force_eplb=} must be True if {with_ckpt =}!")

        low_latency_tp = bool(custom_params.get("low_latency_tp", False))
        if low_latency_tp:
            if parallel_config.world_size % 2:
                raise ValueError("low_latency_tp requires an even world_size for MLA/LI rank pairs")
            if exe_mode not in ("eager", "npugraph_ex"):
                raise ValueError("low_latency_tp uses native streams; select eager or npugraph_ex")
            if parallel_config.attn_tp_size <= 1:
                raise ValueError("low_latency_tp requires attn_tp_size > 1")
            if parallel_config.attn_tp_size != parallel_config.world_size:
                raise ValueError("low_latency_tp requires attn_tp_size == world_size")
            # Low-latency TP keeps one identical token layout through the model.
            for tp_name in (
                "moe_tp_size", "o_proj_tp_size", "lmhead_tp_size",
                "embed_tp_size", "dense_tp_size", "shared_tp_size",
            ):
                setattr(parallel_config, tp_name, parallel_config.world_size)
            parallel_config.attn_dp_size = 1
            parallel_config.embed_dp_size = 1
            parallel_config.moe_ep_size = 1
            if config.num_attention_heads % parallel_config.attn_tp_size != 0:
                raise ValueError("num_attention_heads must be divisible by attn_tp_size")
            if config.o_groups % parallel_config.attn_tp_size != 0:
                raise ValueError("o_groups must be divisible by attn_tp_size")
            if config.moe_intermediate_size % parallel_config.moe_tp_size != 0:
                raise ValueError("moe_intermediate_size must be divisible by moe_tp_size")
            if model_config.force_eplb:
                raise ValueError("Basic MoE TP requires force_eplb=False for identical routing on all ranks")
        elif parallel_config.attn_tp_size > 1:
            raise ValueError(f"{parallel_config.attn_tp_size=} is not supported yet! "
                             "Set low_latency_tp=True to enable the low-latency TP path.")

        dynamo_feat = enable_cache_compile or enable_superkernel
        if exe_mode == "eager" and dynamo_feat:
            raise ValueError(f"{exe_mode=} does not support cache compile or superkernel!")

        if parallel_config.cp_size > 1 and scheduler_config.cp_mini_batch < 1:
            raise ValueError(f"when cp enabled, {scheduler_config.cp_mini_batch=} should be positive")

    def check_model_settings(self):
        self.update_op_kernel_dict(("hc_pre", "hc_post", "gate_topk", "indexer_prolog_qw"))

    def update_op_kernel_dict(self, op_types):
        """Bind requested implementations using kernel_config."""
        auto_import_modules(f"{__package__}.modules.op_impls")
        kernel_config = self.infer_config.model_config.custom_params.get("kernel_config", {})
        platform_version = self.infer_config.model_config.platform_version.value.lower()

        for op_type in op_types:
            if op_type in kernel_config:
                kernel_impl = kernel_config[op_type]
                used_kernel = f"{op_type}_{kernel_impl}_{platform_version}"
            else:
                default_kernel = f"{op_type}_ascendc_{platform_version}"
                used_kernel = (
                    default_kernel
                    if default_kernel in OpKernel.KERNEL_MAP
                    else f"{op_type}_native_{platform_version}"
                )
            if used_kernel not in OpKernel.KERNEL_MAP:
                raise ValueError(
                    f"Unsupported kernel implementation {used_kernel!r} for {op_type!r}."
                )
            OpKernel.op_impl_apply(op_type, used_kernel)
            self.selected_op_kernels[op_type] = used_kernel
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
        indexer_prolog_qw_kernel = getattr(self, "selected_op_kernels", {}).get(
            "indexer_prolog_qw", ""
        )
        use_indexer_prolog_qw = indexer_prolog_qw_kernel.startswith(
            "indexer_prolog_qw_ascendc_"
        )

        indexer_weights_stream = None
        for module_name, module in self.named_modules():
            # Create native Q/W resources after kernel selection, before graph capture.
            if (isinstance(module, Attention) and module.low_latency_tp and module.compress_ratio > 0
                    and not module.is_mla_rank and module.is_index_source and not use_indexer_prolog_qw):
                if indexer_weights_stream is None:
                    indexer_weights_stream = create_stream("separate_mla_weights")
                module.indexer_weights_stream = indexer_weights_stream
            if (isinstance(module, Indexer)
                    and self.low_latency_tp):
                module.use_fused_prolog_qw = use_indexer_prolog_qw
            if module_name.endswith("compressor.norm"):
                module.weight.data = module.weight.data.to(torch.float32)
            if module_name.endswith("attn.indexer.k_norm"):
                module.weight.data = module.weight.data.to(torch.float32)

            if module_name.startswith("vision_model.") and isinstance(module, nn.Linear):
                if enable_weight_nz:
                    module.weight.data = torch_npu.npu_format_cast(module.weight.data, 29)
                continue

            if "wo_a" in module_name:
                config = self.config
                head_dim_per_group = config.num_attention_heads * config.head_dim // config.o_groups
                module.weight.data = module.weight.data.view(-1, config.o_lora_rank, head_dim_per_group) \
                                                   .transpose(1, 2).contiguous()
                if config.quant_config.mm_quant_mode == "w8a8mxfloat8":
                    scale_data = reshape_mx_scale(module.weight_scale.data)
                    module.weight_scale.data = scale_data.view(-1, config.o_lora_rank, *scale_data.shape[1:]) \
                                                   .transpose(1, 2).contiguous()
                if enable_weight_nz:
                    module.weight.data = torch_npu.npu_format_cast(module.weight.data, 29)
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
            is_c1a_compressor = (
                module_name.endswith(".compressor.wkv")
                and self.get_submodule(module_name.rsplit(".", 1)[0]).compress_ratio == 1
            )
            is_weight_nz = enable_weight_nz and not (
                ("compressor" in module_name and not is_c1a_compressor)
                or module_name.endswith(".ffn.gate")
                or (self.enable_mega_moe and ".shared_experts." in module_name)
                or (is_wq_b_transpose and "attn.wq_b" in module_name)
                or ("markov_w2" in module_name)
                or ("confidence_head" in module_name)
            )

            is_transpose = False if "compressor" in module_name else True

            # Indexer prologue kernels consume output-major weights in
            # FRACTAL_NZ.  The K prologue uses the same layout for wk.
            is_indexer_prolog_qw_weight = (
                module_name.endswith("attn.indexer.wk")
                or (
                    use_indexer_prolog_qw
                    and module_name.endswith(("attn.indexer.wq_b", "attn.indexer.weights_proj"))
                )
            )
            is_weight_nz = is_indexer_prolog_qw_weight or is_weight_nz

            is_transpose = is_transpose and not is_indexer_prolog_qw_weight
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

        # Fixed inference weights: gather once, before warm-up and graph capture.
        for module in self.modules():
            if (isinstance(module, Attention) and module.low_latency_tp and module.compress_ratio > 0
                    and module.is_mla_rank and module.attn_tp_size > 1):
                module.full_attn_sink = module.attn_sink.new_empty(module.n_heads)
                dist.all_gather_into_tensor(
                    module.full_attn_sink, module.attn_sink.detach().contiguous(),
                    group=self.comm_manager.get_group(module.attn_group_name),
                )

        if self.megamoe_backend is not None:
            release_layer = (
                getattr(self.megamoe_backend, "release_layer", None)
                if self.disagg_mode == "PREFILL"
                else None
            )
            for module in self.modules():
                if isinstance(module, DeepseekV41MoE):
                    self.megamoe_backend.prepare_layer(module)
                    if release_layer is not None:
                        release_layer(module)

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
            raise ValueError("DeepseekV41ForCausalLM requires comm_manager to initialize communication groups.")

        world_size = self.world_size
        platform_version = self.platform_version
        if self.low_latency_tp:
            self.comm_manager.register_group(
                name="separate_mla_pair",
                subgroups=[[rank, rank + 1] for rank in range(0, world_size, 2)],
                allow_physical_reuse=False,
                platform_version=platform_version,
            )
            self.comm_manager.register_group(
                name="separate_mla_tp",
                subgroups=[list(range(0, world_size, 2))],
                platform_version=platform_version,
            )
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

        if self.engram_tp_size > 1 and self.with_engram:
            for layer_id in self.config.engram_layer_ids:
                self.comm_manager.register_group(
                    name=f"engram_tp_group_{layer_id}",
                    group_num=world_size // self.engram_tp_size,
                    group_size=self.engram_tp_size,
                    platform_version=platform_version,
                    # Keep one physical communication domain per Engram layer.
                    allow_physical_reuse=False,
                )

        if not self.low_latency_tp:
            # The low-latency TP path uses AllReduce, never MC2 dispatch/combine.
            moe_ep_mc2_group_type = None if self.platform_version != PlatformVersion.ASCEND_950 else 3
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
        if self.enable_mega_moe and self.moe_ep_size > 1:
            # MegaMoE path does not support MoE TP yet; check_model_config_before_loading
            # guarantees self.moe_tp_size == 1 before communication groups are built.
            self.comm_manager.register_group(
                name="megamoe_ep_group",
                group_num=self.moe_tp_size,
                group_size=self.moe_ep_size,
                group_stride=self.moe_tp_size,
                return_name=True,
                allow_physical_reuse=False,
                platform_version=platform_version,
            )
        self.comm_manager.register_group(
            name="cp_group",
            group_num=world_size // self.cp_size,
            group_size=self.cp_size,
            group_stride=1,
            platform_version=platform_version,
        )

    def gather_cp_tail_hidden(self, hidden_states, attn_metadata):
        """Collect the last sliding window of every request, which the draft model seeds its cache with."""
        cp_metadata = attn_metadata["cp_metadata"]
        window, row_dim = self.cp_prefill_tail_tokens, hidden_states.shape[-1]
        segment_lens = cp_metadata["segment_lens"]
        tails, offset = [], 0
        for zigzag_flag in ("prev", "next"):
            for segment_len, cur_kv_len in zip(segment_lens, attn_metadata[zigzag_flag]["cur_kv_len"]):
                rows = hidden_states[offset:offset + cur_kv_len][-window:]
                tails.append(F.pad(rows, (0, 0, 0, window - rows.shape[0])))
                offset += segment_len
        all_tails = gather_cp_segments(
            torch.cat(tails, dim=0), cp_metadata, self.comm_manager.get_group("cp_group"))
        all_tails = all_tails.view(-1, len(segment_lens), window, row_dim)

        picked = [
            pick_request_tail(all_tails, cp_metadata, request, window)
            for request in range(len(segment_lens))
        ]
        return torch.cat(picked, dim=0)

    def gather_cp_last_token_hidden(self, outputs, attn_metadata):
        num_tokens, hidden_size = outputs.shape
        cp_metadata = attn_metadata["cp_metadata"]
        batch_size = cp_metadata["batch_size"]
        half_tokens = num_tokens // 2
        # Local tokens are request major within each half, so a request starts after the segments
        # of the requests before it.
        request_offsets, offset = [], 0
        for segment_len in cp_metadata["segment_lens"]:
            request_offsets.append(offset)
            offset += segment_len

        last_hidden = outputs.new_zeros((batch_size, 1, hidden_size))
        for request in range(batch_size):
            if self.global_rank != cp_metadata["last_segment_rank"][request]:
                continue
            local_offset = request_offsets[request] + cp_metadata["last_kv_len"][request] - 1
            if cp_metadata["last_segment_half"][request] == "next":
                local_offset += half_tokens
            last_hidden[request, 0].copy_(outputs[local_offset])
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

    def init_cache_dim(self):
        self.cache_dim = get_kv_cache_dim(self.config.head_dim)

    def generate_kernel_metadata(self, attn_metadata):
        enable_metadata_multi_streams = self.enable_multi_streams
        metadata_stream = attn_metadata.get("metadata_stream")
        main_stream = torch.npu.current_stream() if enable_metadata_multi_streams else None
        if attn_metadata["is_prefill"] and self.cp_size > 1:
            # Each zigzag segment runs its own attention call and needs its own tiling metadata.
            target_metadata = [attn_metadata["prev"], attn_metadata["next"]]
        else:
            target_metadata = [attn_metadata]

        record_event(enable_metadata_multi_streams, self.metadata_event, 0)
        with npu_stream_switch(enable_metadata_multi_streams, metadata_stream):
            wait_event(enable_metadata_multi_streams, self.metadata_event, 0)
            device = attn_metadata["position_ids"].device
            for cur_metadata in target_metadata:
                win_topk_length = cur_metadata["win_topk_length"]
                for ratio in (0, 2, 1):
                    attention_type = "win" if ratio == 0 else f"c{ratio}a"
                    cmp_topk_length = cur_metadata["cmp_topk_length_by_ratio"][str(ratio)]

                    metadata = self.sparse_attn_metadata_ops(
                        ori_topk_length=win_topk_length,
                        cmp_topk_length=cmp_topk_length,
                        cu_seqlens_q=cur_metadata["cu_seq_lens_q"],
                        # TODO: Support TP-sharded query heads in sparse attention metadata.
                        num_heads_q=self.config.num_attention_heads,
                        num_heads_kv=1,
                        head_dim=self.config.head_dim,
                        quant_mode=1,
                        layout_q="TND",
                        layout_kv="PA_BBND",
                        has_ori_kv=True,
                        has_cmp_kv=ratio > 0,
                    )
                    metadata = metadata.to(device, non_blocking=True)
                    cur_metadata["kernel_metadata"][f"{attention_type}_metadata"] = metadata
                    record_stream(enable_metadata_multi_streams, metadata, main_stream)

            # Layer 0 only needs sparse-attention metadata.  Let it start
            # before the independent Indexer metadata is completed.
            record_event(enable_metadata_multi_streams, self.metadata_event, 1)
            # Each zigzag segment scores the cache on its own lengths, so it needs its own
            # Indexer metadata as well.
            for cur_metadata in target_metadata:
                self._generate_li_kernel_metadata(cur_metadata, main_stream)
            record_event(enable_metadata_multi_streams, self.metadata_event, 2)

    def _generate_li_kernel_metadata(self, attn_metadata, main_stream):
        """Generate and publish the fixed V4.1 QLI/QSLI metadata."""
        enable_metadata_multi_streams = self.enable_multi_streams
        position_ids = attn_metadata["position_ids"]
        candidate_topk = self.config.candidate_topk_blocks
        candidate_size = self.config.candidate_block_size
        qli_ratio = 2
        qli_candidate_ratio = 1
        qsli_ratio = 1

        qli_metadata = self.quant_lightning_indexer_metadata_ops(
            cu_seqlens_q=attn_metadata["cu_seq_lens_q"],
            seqused_q=None,
            seqused_k=attn_metadata["compressed_seq_lens"][str(qli_ratio)],
            cmp_residual_k=attn_metadata["compressed_seq_remainders"][str(qli_ratio)],
            batch_size=attn_metadata["actual_seq_k"].shape[0],
            max_seqlen_q=-1,
            max_seqlen_k=-1,
            num_heads_q=self.config.index_n_heads,
            num_heads_k=1,
            head_dim=self.config.index_head_dim,
            topk=self.config.index_topk,
            mask_mode=3,
            cmp_ratio=qli_ratio,
            layout_q="TND",
            layout_k="PA_BBND",
            candidate_topk_blocks=-1,
            candidate_block_size=-1,
        ).to(position_ids.device, non_blocking=True)
        attn_metadata["kernel_metadata"]["c2a_qli_metadata"] = qli_metadata
        record_stream(enable_metadata_multi_streams, qli_metadata, main_stream)

        qli_candidate_metadata = self.quant_lightning_indexer_metadata_ops(
            cu_seqlens_q=attn_metadata["cu_seq_lens_q"],
            seqused_q=None,
            seqused_k=attn_metadata["compressed_seq_lens"][str(qli_candidate_ratio)],
            cmp_residual_k=None,
            batch_size=attn_metadata["actual_seq_k"].shape[0],
            max_seqlen_q=-1,
            max_seqlen_k=-1,
            num_heads_q=self.config.index_n_heads,
            num_heads_k=1,
            head_dim=self.config.index_head_dim,
            topk=self.config.index_topk,
            mask_mode=3,
            cmp_ratio=qli_candidate_ratio,
            layout_q="TND",
            layout_k="PA_BBND",
            candidate_topk_blocks=candidate_topk,
            candidate_block_size=candidate_size,
        ).to(position_ids.device, non_blocking=True)
        attn_metadata["kernel_metadata"]["c1a_qli_candidate_metadata"] = qli_candidate_metadata
        record_stream(enable_metadata_multi_streams, qli_candidate_metadata, main_stream)

        candidate_lengths = torch.div(
            position_ids.reshape(-1) + qsli_ratio * candidate_size,
            qsli_ratio * candidate_size,
            rounding_mode="floor",
        ).clamp_min(1).clamp_max(candidate_topk).to(torch.int32).view(-1, 1)
        qsli_metadata = self.quant_sparse_lightning_indexer_metadata_ops(
            candidate_block_length=candidate_lengths,
            cu_seqlens_q=attn_metadata["cu_seq_lens_q"],
            seqused_q=None,
            seqused_k=attn_metadata["compressed_seq_lens"][str(qsli_ratio)],
            cmp_residual_k=None,
            batch_size=attn_metadata["actual_seq_k"].shape[0],
            max_seqlen_q=-1,
            max_seqlen_k=-1,
            num_heads_q=self.config.index_n_heads,
            num_heads_k=1,
            head_dim=self.config.index_head_dim,
            topk=self.config.index_topk,
            quant_mode=1,
            candidate_block_size=candidate_size,
            mask_mode=3,
            cmp_ratio=qsli_ratio,
            layout_q="TND",
            layout_k="PA_BBND",
        ).to(position_ids.device, non_blocking=True)
        attn_metadata["kernel_metadata"]["c1a_qsli_metadata"] = qsli_metadata
        record_stream(enable_metadata_multi_streams, qsli_metadata, main_stream)

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
        self.generate_kernel_metadata(attn_metadata)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attn_metadata=attn_metadata,
            is_prefill=is_prefill,
            cur_topk_list=cur_topk_list,
            visual_embeddings=visual_embeddings,
        ) # (num_tokens, hidden_size)

        auxiliary_hidden_states = None
        if isinstance(outputs, tuple):
            outputs, auxiliary_hidden_states = outputs
            if is_prefill and self.cp_size > 1:
                auxiliary_hidden_states = self.gather_cp_tail_hidden(auxiliary_hidden_states, attn_metadata)
        prev_hidden_states = outputs
        if auxiliary_hidden_states is not None:
            prev_hidden_states = {
                "prev_hidden_states": prev_hidden_states,
                "auxiliary_hidden_states": auxiliary_hidden_states,
            }

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


class DeepseekV41VisionForCausalLM(DeepseekV41ForCausalLM):
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
