# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional MLA/LI rank pairing for Ascend 950 low-latency TP.

Keep the existing normalization, quantization and cache implementations. Only
the execution owner and the transfer of C4A indices / C128A KV change here.
"""

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch_npu

from executor.core.config import PlatformVersion
from executor.utils.stream_utils import record_event, record_stream, wait_event


# Reserve four tags per layer; offsets 1 and 2 identify indices and compressed KV.
_TAGS_PER_LAYER = 4
_PAYLOAD_TAG_BY_RATIO = {4: 1, 128: 2}


@dataclass(frozen=True)
class SeparateMLATopology:
    world_size: int
    global_rank: int

    @property
    def tp_size(self):
        return self.world_size // 2

    @property
    def tp_rank(self):
        return self.global_rank // 2

    @property
    def is_mla_rank(self):
        return self.global_rank % 2 == 0

    @property
    def peer_rank(self):
        return self.global_rank ^ 1

    @property
    def pairs(self):
        return [[rank, rank + 1] for rank in range(0, self.world_size, 2)]


def validate_separate_mla_config(config, infer_config):
    """Fail before group creation or weight loading; legacy modes are untouched."""
    custom = infer_config.model_config.custom_params
    if not custom.get("low_latency_tp", False):
        return
    parallel = infer_config.parallel_config
    world_size = parallel.world_size
    mla_tp_size = world_size // 2
    checks = (
        (infer_config.model_config.platform_version == PlatformVersion.ASCEND_950,
         "only supports Ascend 950"),
        (world_size > 1 and world_size % 2 == 0, "requires an even world_size > 1"),
        (0 <= parallel.global_rank < world_size, "requires a valid global_rank"),
        (all(getattr(parallel, name) == world_size for name in (
            "attn_tp_size", "o_proj_tp_size", "moe_tp_size", "embed_tp_size", "lmhead_tp_size")),
         "requires attention, output, MoE, embedding and LM-head TP sizes equal to world_size"),
        (parallel.cp_size == 1, "requires cp_size=1"),
        (not custom.get("enable_pypto", False), "does not support enable_pypto=True"),
        (infer_config.disagg_config.disaggregation_mode == "NONE", "only supports offline inference"),
        (mla_tp_size > 0 and config.num_attention_heads % mla_tp_size == 0,
         "requires num_attention_heads divisible by world_size // 2"),
        (mla_tp_size > 0 and config.o_groups % mla_tp_size == 0,
         "requires o_groups divisible by world_size // 2"),
        (all(ratio in (1, 4, 128) for ratio in config.compress_ratios),
         "only supports C1A, C4A and C128A"),
    )
    for valid, message in checks:
        if not valid:
            raise ValueError(f"low_latency_tp {message}.")


def _exchange(attention, tensor, shape, dtype, is_prefill):
    """Broadcast in prefill; send LI's payload to its MLA peer in decode."""
    topology = attention.separate_mla
    group = attention.comm_manager.get_group("separate_mla_group")
    if topology.is_mla_rank:
        tensor = torch.empty(shape, dtype=dtype, device=attention.wq_a.weight.device)
    else:
        tensor = tensor.contiguous()
    if is_prefill:
        source = topology.peer_rank if topology.is_mla_rank else topology.global_rank
        dist.broadcast(tensor, src=source, group=group)
    else:
        layer_idx = attention.layer_idx
        if isinstance(layer_idx, bool) or not isinstance(layer_idx, int) or layer_idx < 0:
            raise ValueError("Separate MLA communication requires a non-negative integer layer_idx.")
        tag = layer_idx * _TAGS_PER_LAYER + _PAYLOAD_TAG_BY_RATIO[attention.compress_ratio]
        if topology.is_mla_rank:
            dist.recv(tensor, src=topology.peer_rank, group=group, tag=tag)
        else:
            dist.send(tensor, dst=topology.peer_rank, group=group, tag=tag)
    return tensor


def reduce_separate_mla_output(attention, output):
    # LI ranks contribute zeros and receive the complete attention output for
    # the residual and full-world MoE TP that follow on every rank.
    dist.all_reduce(output, group=attention.comm_manager.get_group("attn_tp_group"))
    return output


def _li_zero_output(x, dim, payload):
    """Preserve the LI payload dependency while contributing exact zeros."""
    if payload is None:
        return torch.zeros((x.shape[0], dim), dtype=torch.bfloat16, device=x.device)
    # The broadcast is functionalized when compiled. Its result must reach the
    # output, otherwise dead-code elimination can remove the LI-side collective.
    # Mask rather than multiply by zero so NaN/Inf KV values still contribute zero.
    anchor = payload.reshape(-1)[:1].to(torch.bfloat16)
    zero = anchor.masked_fill(torch.ones_like(anchor, dtype=torch.bool), 0).sum()
    return zero.expand(x.shape[0], dim).contiguous()


def _indexer_query(attention, x, is_prefill):
    # Mirror MLA's Q-A input quantization. Decode Linear handles quantization
    # itself; HiFloat8 input is already normalized/quantized by the decoder.
    scale = None
    if is_prefill and attention.mm_quant_mode == "w8a8float8":
        x, scale = torch_npu.npu_dynamic_block_quant(
            x.view(-1, x.shape[-1]), dst_type=torch.float8_e4m3fn,
            row_block_size=1, col_block_size=attention.wq_a.weight_block_size[1],
        )
    elif is_prefill and attention.mm_quant_mode == "w8a8mxfloat8":
        x, scale = torch_npu.npu_dynamic_mx_quant(
            x.view(-1, x.shape[-1]), dst_type=torch.float8_e4m3fn,
        )
    qr = attention.wq_a(x, dynamic_scale=scale).view(x.shape[0], -1)
    return attention.apply_norm_dynamic_quant(qr)


def forward_separate_mla(attention, x, attn_metadata, is_prefill, **kwargs):
    """Execute local work before joining the paired and full-world collectives.

    x uses the unchanged decoder normalization contract. C4A keeps attention
    compression on MLA ranks and indexer compression on LI ranks. C128A moves
    compression to LI, but writes the resulting KV cache on MLA ranks.
    """
    is_mla_rank = attention.separate_mla.is_mla_rank
    ratio = attention.compress_ratio
    if is_mla_rank:
        q, _, _, _ = attention.mla_prolog(x, attn_metadata, is_prefill, **kwargs)

    wait_event(
        attention.enable_multi_streams and not is_prefill and (is_mla_rank or ratio != 1),
        attn_metadata.get("metadata_event"), 1,
    )
    topk = None
    payload = None
    if ratio == 4:
        if is_mla_rank:
            attention.compressor(x, attn_metadata, is_prefill)
        else:
            qr, qr_scale = _indexer_query(attention, x, is_prefill)
            multistream = attention.enable_multi_streams and not is_prefill
            stream = attn_metadata.get("indexer_stream")
            record_stream(multistream, qr, stream)
            if qr_scale is not None:
                record_stream(multistream, qr_scale, stream)
            record_event(multistream, attention.indexer.indexer_events, 0)
            topk = attention.indexer(
                x, qr, qr_scale, attn_metadata,
                cmpr_switch_flag=False, cmpr_event=(), cmpr_event_idx=0, is_prefill=is_prefill,
            )
        topk = _exchange(
            attention, topk, (x.shape[0], 1, attention.indexer.index_topk), torch.int32, is_prefill,
        )
        payload = topk
    elif ratio == 128:
        # compressor_prolog updates LI's persistent state. Its BF16 output is
        # normalized and rotated, but not yet quantized into MLA's KV cache.
        kv = None if is_mla_rank else attention.compressor.compressor_prolog(x, attn_metadata, is_prefill)
        slots = attn_metadata["slot_mapping"]["c128a_cmp_kv"]
        shape = (slots.numel(), attention.head_dim) if is_prefill else (*slots.shape, attention.head_dim)
        kv = _exchange(attention, kv, shape, torch.bfloat16, is_prefill)
        payload = kv
        if is_mla_rank:
            attention.compressor.compressor_epilog(kv, attn_metadata, is_prefill)

    if is_mla_rank:
        output = attention.attn_kernel(q, topk, attn_metadata, is_prefill)
        output = attention.attn_post(output, attn_metadata, is_prefill, reduce_output=False)
    else:
        output = _li_zero_output(x, attention.dim, payload)
    return reduce_separate_mla_output(attention, output)
