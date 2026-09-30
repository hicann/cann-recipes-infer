# coding=utf-8
# This code is copied from vllm implementations.
# (https://github.com/vllm-project/vllm/blob/v0.9.0/vllm/model_executor/layers/quantization/utils/quant_utils.py)
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from types import MappingProxyType

import torch



def is_layer_skipped(
    prefix: str,
    ignored_layers: list[str],
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({})
) -> bool:
    # prefix: model.layers.0.self_attn.q_proj
    # proj_name: q_proj
    proj_name = prefix.split(".")[-1]

    # Fused layers like gate_up_proj or qkv_proj will not be fused
    # in the safetensors checkpoint. So, we convert the name
    # from the fused version to unfused + check to make sure that
    # each shard of the fused layer has the same scheme.
    if proj_name in fused_mapping:
        shard_prefixes = [
            prefix.replace(proj_name, shard_proj_name)
            for shard_proj_name in fused_mapping[proj_name]
        ]

        is_skipped = None
        for shard_prefix in shard_prefixes:
            is_shard_skipped = shard_prefix in ignored_layers

            if is_skipped is None:
                is_skipped = is_shard_skipped
            elif is_shard_skipped != is_skipped:
                raise ValueError(
                    f"Detected some but not all shards of {prefix} "
                    "are quantized. Ensure all shards of fused layers "
                    "use the same precision.")
    else:
        is_skipped = prefix in ignored_layers

    assert is_skipped is not None
    return is_skipped


def swiglu_group_quant(x, *, dst_type, round_scale=False, quant_mode,
                       clamp_limit=None, group_index=None, group_list_type=1,
                       prefer_custom=False):
    """SwiGLU + group quant. prefer_custom selects the repo custom op
    (64-element mx tail, type-2 group lists); otherwise the mainline
    cann_ops_nn kernel is used (count-list group index, 256-aligned tail)."""
    if group_list_type == 2 and not prefer_custom:
        raise RuntimeError(
            "group_list_type=2 requires prefer_custom (the repo custom op)")
    if prefer_custom:
        return torch.ops.custom.npu_swiglu_group_quant(
            x, dst_type=dst_type, round_scale=round_scale,
            quant_mode=quant_mode, clamp_limit=clamp_limit,
            group_index=group_index, output_origin=False,
            group_list_type=group_list_type)
    if clamp_limit is None:
        clamp_limit = -1.0
    return torch.ops.cann_ops_nn.swiglu_group_quant(
        x, dst_type=dst_type, round_scale=round_scale, quant_mode=quant_mode,
        clamp_limit=clamp_limit, group_index=group_index)


def reshape_mx_scale(scale_tensor):
    """
    Reshape the last dimension into pairs for GMM/MM operators; an odd
    group count gets one zero tail slot for the pair view.
    """
    import torch
    num_groups = scale_tensor.size(-1)
    if num_groups % 2:
        pad = torch.zeros((*scale_tensor.shape[:-1], 1), dtype=scale_tensor.dtype,
                          device=scale_tensor.device)
        scale_tensor = torch.cat((scale_tensor, pad), dim=-1)
        num_groups += 1
    return scale_tensor.view(*scale_tensor.shape[:-1], num_groups // 2, 2)
