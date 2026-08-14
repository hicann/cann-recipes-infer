# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Copyright 2026 The Moonshot AI Team. All rights reserved.
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
from __future__ import annotations

import logging
import math
import os
import re
from types import SimpleNamespace
from typing import Iterable, NamedTuple, Optional, Tuple

# mega_moe fused operator (prefill and decode paths, controlled by enable_mega_moe)
import cann_ops_transformer.ops
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe

from executor.utils import calc_moe_hccl_buffer_size, init_comm_group
from executor.utils.stream_utils import (
    npu_stream_switch,
    record_event,
    record_stream,
    wait_event,
    create_event,
    create_stream
)
from module.fuse_moe_gmm import FusedMoEGMM
from module.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from module.quantization import QuantizeMethodBase
from module.quantization.mxfp4 import W4A8MxFp4MoEGMMMethod
from module.quantization.utils.quant_utils import reshape_mx_scale

from .configuration_kimi_k3 import KimiLinearConfig
from .modules import (
    AttnMetaData,
    all_gather_first_dim,
    distributed_argmax,
    dp_to_tp_all_to_all,
    reduce_scatter_first_dim,
    vocab_tp_to_owner,
)

try:
    from ops.cannbot_dsl.flash_kda import flash_kda as _flash_kda_impl
    from ops.cannbot_dsl.fused_recurrent_kda import (
        fused_recurrent_kda_op as _recurrent_kda_impl,
    )
    from ops.cannbot_dsl import (
        block_attn_res_prepare as _block_attn_res_prepare_impl,
        block_attn_res_update as _block_attn_res_update_impl,
    )

except ImportError:
    _flash_kda_impl = None
    _recurrent_kda_impl = None
    _block_attn_res_prepare_impl = None
    _block_attn_res_update_impl = None

logger = logging.getLogger(__name__)

ForwardMetaData = dict
InferenceConfig = SimpleNamespace


def _global_rank() -> int:
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return int(os.getenv("LOCAL_RANK", "0")) + int(os.getenv("RANK_OFFSET", "0"))


def _offline_infer_config(settings):
    """Expose legacy runner settings through the former internal attribute API."""
    model = settings.get("model_config", {})
    parallel = settings.get("parallel_config", {})
    data = settings.get("data_config", {})
    prefill_mini_batch_size = model.get("prefill_mini_batch_size", 0)
    prefill_batch_size = (
        prefill_mini_batch_size
        if prefill_mini_batch_size > 0
        else data.get("batch_size_per_rank", data.get("batch_size", 1))
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            custom_params=model.get("custom_params", {}),
            exe_mode=settings.get("exe_mode", "eager"),
            enable_weight_nz=model.get("enable_weight_nz", False),
            next_n=model.get("next_n", 0),
            draft_model_type=model.get("draft_model_type", "none"),
            prefill_mini_batch_size=prefill_mini_batch_size,
        ),
        parallel_config=SimpleNamespace(
            world_size=settings.get("world_size", int(os.getenv("WORLD_SIZE", "1"))),
            global_rank=_global_rank(),
            attn_tp_size=parallel.get("attn_tp_size", 1),
            attn_dp_size=parallel.get("attn_dp_size", 1),
            moe_tp_size=parallel.get("moe_tp_size", 1),
            moe_dp_size=parallel.get("moe_dp_size", 1),
            moe_ep_size=parallel.get("moe_ep_size", 1),
            shared_tp_size=parallel.get("shared_tp_size", 1),
            dense_tp_size=parallel.get("dense_tp_size", 1),
            embed_tp_size=parallel.get("embed_tp_size", 1),
            embed_dp_size=parallel.get("embed_dp_size", 1),
            lmhead_tp_size=parallel.get("lmhead_tp_size", 1),
            o_proj_tp_size=parallel.get("oproj_tp_size", 1),
        ),
        scheduler_config=SimpleNamespace(
            block_size=model.get("pa_block_size", 128),
            # Total packed tokens before Prefill sequence-parallel sharding.
            # KimiLinearModel derives its rank-local resident buffer by
            # ceil-dividing this value by attn_tp_size.
            max_prefill_tokens=prefill_batch_size * data.get("input_max_len", 128),
            batch_size_per_dp_rank=data.get("batch_size_per_rank", data.get("batch_size", 1)),
        ),
        data_config=SimpleNamespace(
            input_truncated_len=data.get("input_max_len", 128),
            input_max_len=data.get("input_max_len", 128),
            max_new_tokens=data.get("max_new_tokens", 128),
            temperature=data.get("temperature", 1.0),
        ),
    )


class _OfflineCommManager:
    """Small model-local communication registry used by the offline runner."""

    def __init__(self, settings):
        self.settings = settings
        self.world_size = int(os.getenv("WORLD_SIZE", str(settings.get("world_size", 1))))
        self.global_rank = _global_rank()
        self.platform_version = settings.get("model_config", {}).get("platform_version", "950")
        self.groups = {}
        self.group_names = {}
        self.group_sizes = {}

    def register_group(
        self,
        name,
        group_num,
        group_size,
        group_stride=1,
        return_name=False,
        hccl_buffer_size=None,
        group_type=None,
        **kwargs,
    ):
        result = init_comm_group(
            global_rank=self.global_rank,
            group_num=group_num,
            world_size=self.world_size,
            group_stride=group_stride,
            group_name=name,
            hccl_buffer_size=hccl_buffer_size,
            return_name=return_name,
            group_type=group_type,
            platform_version=self.platform_version,
        )
        if return_name:
            group, group_name = result
            self.group_names[name] = group_name
        else:
            group = result
        self.groups[name] = group
        self.group_sizes[name] = group_size

    def get_group(self, name):
        return self.groups.get(name)

    def get_group_name(self, name):
        return self.group_names.get(name)

    def get_rank(self, name):
        if self.group_sizes.get(name, 1) == 1:
            return 0
        return dist.get_rank(self.groups.get(name))


CommManager = _OfflineCommManager

# npu_moe_gating_top_k documents its input as 2D with the expert count
# as the last dim, capped at 2048.
_MOE_GATING_MAX_EXPERTS = 2048

# Inner dimension of the NZ block layout, fixed by the 16-bit cache dtype.
_KV_CACHE_NZ_DIM = 16

_KDA_CHUNK_SIZE = 64

# Default gathered-token ceiling for the MoE prefill routing buffers.
_DEFAULT_MOE_CHUNK_MAX_LEN = 65536


class KdaInputs(NamedTuple):
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    raw_gate: torch.Tensor
    raw_beta: torch.Tensor


class KdaGateParams(NamedTuple):
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    lower_bound: Optional[float]


def _moe_chunk_plan(local_tokens: int, moe_ep_size: int, moe_chunk_max_len: int) -> list[int]:
    """Return per-chunk local-token counts within the routing buffer budget.

    Each rank holds an equal-length SP shard of ``local_tokens``, so chunking
    each shard by the same boundaries keeps double-routing collectives aligned
    across the EP group. The first chunk size is the largest one that respects
    the configured global routing budget, and the remainder is the tail chunk.
    """
    if moe_chunk_max_len <= 0 or moe_ep_size <= 0:
        return [local_tokens]
    gathered_total = local_tokens * moe_ep_size
    if gathered_total <= moe_chunk_max_len:
        return [local_tokens]
    # Every rank uses the same local count so double-routing calls stay aligned.
    max_local_per_chunk = moe_chunk_max_len // moe_ep_size
    full_chunks = local_tokens // max_local_per_chunk
    remainder = local_tokens % max_local_per_chunk
    plan = [max_local_per_chunk] * full_chunks
    if remainder:
        plan.append(remainder)
    return plan


def _softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x) + torch.log1p(torch.exp(-torch.abs(x)))


def _l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x.float() * x.float()).sum(dim=-1, keepdim=True) + eps)


def _torch_chunk_kda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    transition_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    identity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run FLA's Torch chunk KDA on one right-padded request."""
    # Adapted from FLA naive_chunk_kda (MIT), copyright 2023-2026 Songlin Yang,
    # Yu Zhang, Zhiyuan Li:
    # https://github.com/fla-org/flash-linear-attention/blob/0a9b9f222e86b9a895c2447767e9b4cce6c8d530/fla/ops/kda/naive.py#L69
    output_dtype = value.dtype
    batch, tokens, heads, key_dim = query.shape
    value_dim = value.shape[-1]
    pad_len = (-tokens) % _KDA_CHUNK_SIZE
    if pad_len:
        query, key, value, decay = (
            F.pad(tensor, (0, 0, 0, 0, 0, pad_len))
            for tensor in (query, key, value, decay)
        )
        beta = F.pad(beta, (0, 0, 0, pad_len))

    chunk_count = query.shape[1] // _KDA_CHUNK_SIZE

    def chunked(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(
            batch, chunk_count, _KDA_CHUNK_SIZE, heads, *tensor.shape[3:]
        ).permute(0, 3, 1, 2, *range(4, tensor.ndim + 1)).float()

    q = chunked(_l2_normalize(query)) / math.sqrt(key_dim)
    k = chunked(_l2_normalize(key))
    v = chunked(value)
    g = chunked(decay).cumsum(dim=-2)
    b = chunked(beta)

    transition = torch.zeros(
        *g.shape[:-1], _KDA_CHUNK_SIZE, dtype=torch.float32, device=q.device
    )
    for index in range(_KDA_CHUNK_SIZE):
        key_i = k[..., index, :]
        decay_i = g[..., index : index + 1, :]
        transition[..., index] = torch.matmul(
            k * (g - decay_i).exp(), key_i.unsqueeze(-1)
        ).squeeze(-1)
    transition = -(transition * b[..., None]).masked_fill(transition_mask, 0)
    for index in range(1, _KDA_CHUNK_SIZE):
        transition[..., index, :index] = transition[
            ..., index, :index
        ].clone() + (
            transition[..., index, :, None].clone()
            * transition[..., :, :index].clone()
        ).sum(-2)
    transition = (transition + identity) * b[..., None, :]

    corrected_key = transition @ (g.exp() * k)
    corrected_value = transition @ v
    state = initial_state
    output = torch.zeros_like(v)
    for index in range(chunk_count):
        q_i = q[:, :, index]
        k_i = k[:, :, index]
        v_i = corrected_value[:, :, index]
        g_i = g[:, :, index]
        w_i = corrected_key[:, :, index]
        attention = torch.zeros(
            batch,
            heads,
            _KDA_CHUNK_SIZE,
            _KDA_CHUNK_SIZE,
            dtype=torch.float32,
            device=q.device,
        )
        for token_index in range(_KDA_CHUNK_SIZE):
            key_j = k_i[:, :, token_index]
            decay_j = g_i[:, :, token_index : token_index + 1]
            attention[..., token_index] = torch.matmul(
                q_i * (g_i - decay_j).exp(), key_j.unsqueeze(-1)
            ).squeeze(-1)
        attention = attention.masked_fill(attention_mask, 0)
        v_i = v_i - w_i @ state
        output[:, :, index] = (q_i * g_i.exp()) @ state + attention @ v_i
        final_decay = g_i[:, :, -1]
        state = state * final_decay.exp().unsqueeze(-1)
        state = state + (
            (final_decay.unsqueeze(-2) - g_i).exp() * k_i
        ).transpose(-1, -2) @ v_i

    output = output.permute(0, 2, 3, 1, 4).reshape(
        batch, -1, heads, value_dim
    )
    return output[:, :tokens].to(output_dtype), state


class KimiRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gamma = self.weight.to(dtype=hidden_states.dtype)
        return torch_npu.npu_rms_norm(
            hidden_states, gamma, self.variance_epsilon
        )[0]


class SituAndMul(nn.Module):
    """The checkpoint's SiTU gated activation."""

    def __init__(self, beta: float = 1.0, linear_beta: Optional[float] = None, enable_moe_bf16_mode=False) -> None:
        super().__init__()
        self.beta = float(beta)
        self.linear_beta = None if linear_beta is None else float(linear_beta)
        self.enable_moe_bf16_mode = enable_moe_bf16_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = x.chunk(2, dim=-1)
        if not self.enable_moe_bf16_mode:
            up = up.float()
            gate = gate.float()
        gate = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        if not self.enable_moe_bf16_mode:
            return (gate * up).to(x.dtype)
        return (gate * up)


def _activation(config: KimiLinearConfig, enable_moe_bf16_mode):
    if config.hidden_act == "situ":
        return SituAndMul(
            beta=getattr(config, "activation_situ_beta", None) or 1.0,
            linear_beta=getattr(config, "activation_situ_linear_beta", None),
            enable_moe_bf16_mode=enable_moe_bf16_mode,
        )
    return None


def _unpad_kda_input(
    hidden_states: torch.Tensor, pad_len: int
) -> torch.Tensor:
    if not pad_len:
        return hidden_states
    return hidden_states[:-pad_len]


def _pad_kda_output(output: torch.Tensor, pad_len: int) -> torch.Tensor:
    if not pad_len:
        return output
    return torch.cat((output, output.new_zeros(pad_len, *output.shape[1:])))


def _dense_tp(parallel, comm_manager) -> tuple[int, int, object]:
    """Dense TP degree, this rank's position in it, and its process group.

    One field sizes both users of ``KimiMLP``, the layer-0 dense FFN and the MoE
    shared expert; ``shared_tp_size`` is rejected in check_model_settings.
    """
    size = 1 if parallel is None else parallel.dense_tp_size
    if size == 1:
        return 1, 0, None
    return (
        size,
        comm_manager.get_rank("dense_tp_group"),
        comm_manager.get_group("dense_tp_group"),
    )


class KimiMLP(nn.Module):
    """Dense feed-forward network, also used as the MoE shared expert.

    ``tp_size`` splits the intermediate dimension. check_model_settings pins
    dense_tp_size to attn_tp_size and attn_tp > 1 is what turns SP on, so a
    split always comes with a token-sharded caller -- hence the unconditional
    collectives in forward.
    """

    def __init__(
        self,
        config: KimiLinearConfig,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        tp_group=None,
        prefix: str = "",
        enable_moe_bf16_mode: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.tp_size = tp_size
        self.tp_group = tp_group
        hidden_size = hidden_size or config.hidden_size
        intermediate_size = intermediate_size or config.intermediate_size
        quant_config = getattr(config, "quant_config", None)
        # Gate first, then up, matching SituAndMul's chunk order.
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            tp_size=tp_size,
            tp_rank=tp_rank,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.situ = _activation(config, enable_moe_bf16_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Every rank holds a different token slice but the same column shards,
        # so the tokens must be whole before the projections.
        x = all_gather_first_dim(x, self.tp_group, self.tp_size)
        gate_up = self.gate_up_proj(x)
        if self.situ is not None:
            activated = self.situ(gate_up)
        else:
            activated = F.silu(gate_up[..., : gate_up.shape[-1] // 2]) * gate_up[..., gate_up.shape[-1] // 2 :]
        output = self.down_proj(activated)
        # Sum the row-parallel partials and scatter the tokens back.
        return reduce_scatter_first_dim(output, self.tp_group, self.tp_size)


def _mxfp4_expert_quantization(config: KimiLinearConfig) -> bool:
    """True when the checkpoint stores routed experts as MXFP4.

    Matches on the declared scheme -- 4-bit float weights in groups of 32 --
    rather than on the format string, because ``KimiLinearConfig`` rewrites K3's
    vendor spelling into the framework's canonical one (see
    ``normalize_mx_pack_quantization``). The framework's own W4A8 selector is not
    reused: it keys off layer targets, and this model builds its expert method
    directly rather than asking the shared quantization config for a scheme.
    """
    quant = getattr(config, "quantization_config", None)
    if not isinstance(quant, dict):
        return False
    for group in quant.get("config_groups", {}).values():
        weights = group.get("weights") or {}
        if (
            weights.get("num_bits") == 4
            and weights.get("type") == "float"
            and weights.get("group_size") == 32
        ):
            return True
    return False


def _validate_kimi_k3_architecture(config: KimiLinearConfig) -> None:
    # MoE
    if config.routed_expert_hidden_size is None or config.routed_expert_hidden_size <= 0:
        raise ValueError("Kimi K3 requires a positive routed_expert_hidden_size")
    if not config.latent_moe_use_norm:
        raise ValueError("Kimi K3 requires latent MoE normalization")
    if config.hidden_act != "situ":
        raise ValueError("Kimi K3 routed experts require SiTU")
    if not config.moe_renormalize:
        raise ValueError("Kimi K3 requires MoE router renormalization")


class _SituMoEGMMMethod(QuantizeMethodBase):
    """FusedMoEGMM method preserving Kimi K3's exact SiTU activation.

    Both the BF16 and the MXFP4 base methods fuse their activation into the
    span between the two grouped matmuls -- ``npu_swiglu`` for BF16 and
    ``npu_swiglu_mx_quant`` for MXFP4 -- and neither implements SiTU's
    ``beta``/``linear_beta`` double tanh. The activation is therefore always
    unfused here; on the quantized path that also means re-quantizing the
    intermediate explicitly, which the fused operator would otherwise have
    done as part of the activation.
    """

    def __init__(
        self,
        base_method: QuantizeMethodBase,
        situ: SituAndMul,
        quantized: bool = False,
    ) -> None:
        self._base = base_method
        self.situ = situ
        self.quantized = quantized

    def create_weights(self, *args, **kwargs):
        return self._base.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer, **kwargs) -> None:
        self._base.process_weights_after_loading(layer, **kwargs)

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        expert_tokens: torch.Tensor,
        group_list_type: int,
        pertoken_scale: Optional[torch.Tensor] = None,
        final_output_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> torch.Tensor:
        if not self.quantized:
            gate_up = torch_npu.npu_grouped_matmul(
                [x],
                [layer.w13_weight],
                group_list=expert_tokens,
                group_type=0,
                group_list_type=group_list_type,
                split_item=3,
            )[0]
            return torch_npu.npu_grouped_matmul(
                [self.situ(gate_up)],
                [layer.w2_weight],
                group_list=expert_tokens,
                group_type=0,
                group_list_type=group_list_type,
                split_item=3,
            )[0]

        # W4A8: MXFP4 weights, activations quantized to MXFP8 on the fly.
        if pertoken_scale is None:
            x, pertoken_scale = torch_npu.npu_dynamic_mx_quant(
                x, dst_type=torch.float8_e4m3fn
            )
        gate_up = torch_npu.npu_grouped_matmul(
            [x],
            [layer.w13_weight.transpose(1, 2)],
            antiquant_scale=[layer.w13_weight_scale.transpose(1, 2)],
            per_token_scale=[pertoken_scale],
            group_list=expert_tokens,
            group_type=0,
            group_list_type=group_list_type,
            split_item=3,
            output_dtype=torch.bfloat16,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            tuning_config=[0],
        )[0]
        # npu_swiglu_mx_quant would have activated and re-quantized in one op;
        # SiTU has no fused counterpart, so do both steps explicitly.
        activated, pertoken_scale = torch_npu.npu_dynamic_mx_quant(
            self.situ(gate_up), dst_type=torch.float8_e4m3fn
        )
        return torch_npu.npu_grouped_matmul(
            [activated],
            [layer.w2_weight.transpose(1, 2)],
            antiquant_scale=[layer.w2_weight_scale.transpose(1, 2)],
            per_token_scale=[pertoken_scale],
            group_list=expert_tokens,
            group_type=0,
            group_list_type=group_list_type,
            split_item=3,
            output_dtype=final_output_dtype,
            weight_dtype=torch_npu.float4_e2m1fn_x2,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            tuning_config=[0],
        )[0]


class KimiSituMoEGMM(FusedMoEGMM):
    """Packed local experts with the checkpoint-compatible SiTU formula."""

    def __init__(
        self,
        config: KimiLinearConfig,
        hidden_size: int,
        ep_size: int,
        ep_rank: int,
        enable_moe_bf16_mode: bool = False,
    ) -> None:
        self.quantized = _mxfp4_expert_quantization(config)
        super().__init__(
            num_experts=config.num_experts,
            hidden_size=hidden_size,
            intermediate_size=config.moe_intermediate_size,
            bias=False,
            tp_size=1,
            tp_rank=0,
            ep_size=ep_size,
            ep_rank=ep_rank,
            params_dtype=torch.get_default_dtype(),
            quant_config=None,
        )
        self.situ = _activation(config, enable_moe_bf16_mode)
        if self.situ is None:
            raise RuntimeError("Kimi K3 routed experts require the SiTU activation")
        base_method = self.quant_method
        if self.quantized:
            base_method = W4A8MxFp4MoEGMMMethod()
            # FusedMoEGMM already built BF16 parameters; drop them and let the
            # quantized method create the packed ones in their place.
            for name in ("w13_weight", "w2_weight"):
                if name in self._parameters:
                    del self._parameters[name]
            base_method.create_weights(
                layer=self,
                num_experts=self.experts_per_rank,
                hidden_size=hidden_size,
                intermediate_size_per_partition=self.intermediate_size_per_partition,
                params_dtype=torch.get_default_dtype(),
                weight_loader=self.weight_loader,
            )
        self.quant_method = _SituMoEGMMMethod(base_method, self.situ, self.quantized)


class KimiMoEGate(nn.Module):
    def __init__(self, config: KimiLinearConfig) -> None:
        super().__init__()
        self.top_k = config.num_experts_per_token
        self.num_experts = config.num_experts
        if self.num_experts > _MOE_GATING_MAX_EXPERTS:
            raise RuntimeError(
                f"npu_moe_gating_top_k supports at most "
                f"{_MOE_GATING_MAX_EXPERTS} experts, got {self.num_experts}"
            )
        self.routed_scaling_factor = config.routed_scaling_factor
        self.activation = config.moe_router_activation_func
        self.num_expert_group = config.num_expert_group
        self.topk_group = config.topk_group
        # Router logits are computed in FP32. Keep the weight FP32 so loading
        # converts it once instead of casting it on every forward.
        self.weight = nn.Parameter(
            torch.empty(
                self.num_experts, config.hidden_size, dtype=torch.float32
            )
        )
        # The correction bias is also FP32 because it feeds the top-k
        # comparison, where BF16 rounding can reorder experts near a tie.
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(self.num_experts, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = F.linear(hidden_states.float(), self.weight)
        topk_weight, topk_idx, _ = torch_npu.npu_moe_gating_top_k(
            logits,
            k=self.top_k,
            bias=self.e_score_correction_bias.to(logits.dtype),
            k_group=self.topk_group,
            group_count=self.num_expert_group,
            group_select_mode=1,
            renorm=0,
            norm_type=1 if self.activation == "sigmoid" else 0,
            out_flag=False,
            routed_scaling_factor=self.routed_scaling_factor,
            eps=1e-20,
        )
        return topk_idx, topk_weight


class MoEContext:
    """Per-model shared resources for MoE inference.

    Created once at model init; passed through the forward chain so every
    MoE block reads the same shared_stream and mega_sym_buffer instances.
    """
    __slots__ = ("shared_stream", "mega_sym_buffer")

    def __init__(
        self,
        config: KimiLinearConfig,
        infer_config: Optional[InferenceConfig] = None,
        comm_manager: Optional[CommManager] = None,
    ) -> None:
        enable_multi_streams =  infer_config.model_config.custom_params.get("enable_multi_streams", False)
        exe_mode = infer_config.model_config.exe_mode
        # All MoE layers share this stream for multi-stream computation of shared experts.
        self.shared_stream = create_stream('shared', exe_mode) if enable_multi_streams else None
        # mega_moe sym_buffer: allocated immediately after communication-domain registration
        # and reused throughout the entire inference cycle.
        self.mega_sym_buffer = None

        enable_mega_moe = infer_config.model_config.custom_params.get("enable_mega_moe", False)
        moe_ep_size = infer_config.parallel_config.moe_ep_size

        if enable_mega_moe and moe_ep_size > 1:
            max_prefill_tokens = infer_config.scheduler_config.max_prefill_tokens
            moe_chunk_max_len = infer_config.model_config.custom_params.get(
                        "moe_chunk_max_len", _DEFAULT_MOE_CHUNK_MAX_LEN
                    )
            max_token_per_chunk = (
                min(moe_chunk_max_len, max_prefill_tokens)
                if moe_chunk_max_len > 0
                else max_prefill_tokens
            ) // moe_ep_size
            group = comm_manager.get_group("megamoe_ep_group")
            self.mega_sym_buffer = get_symm_buffer_for_mega_moe(
                group,
                num_experts=config.num_experts,
                num_max_tokens_per_rank=max_token_per_chunk,
                num_topk=config.num_experts_per_token,
                hidden=config.routed_expert_hidden_size,
                intermediate_hidden=2 * config.moe_intermediate_size,
                dispatch_quant_mode=4, # 4: MXFP quantization (A8W4)
                dispatch_quant_out_dtype=torch.float8_e4m3fn,
            )



class KimiSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        infer_config: Optional[InferenceConfig] = None,
        comm_manager: Optional[CommManager] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        parallel = None if infer_config is None else infer_config.parallel_config
        self.moe_ep_size = 1 if parallel is None else parallel.moe_ep_size
        self.moe_ep_rank = comm_manager.get_rank("moe_ep_group")
        self.moe_ep_group = comm_manager.get_group("moe_ep_group")
        self.moe_ep_group_mc2_name = comm_manager.get_group_name("moe_ep_group_mc2")
        self.enable_multi_streams =  infer_config.model_config.custom_params.get("enable_multi_streams", False)
        self.exe_mode = infer_config.model_config.exe_mode
        self.moe_chunk_max_len = infer_config.model_config.custom_params.get(
            "moe_chunk_max_len", _DEFAULT_MOE_CHUNK_MAX_LEN
        )
        if self.moe_chunk_max_len > 0 and self.moe_chunk_max_len < self.moe_ep_size:
            raise ValueError(
                f"moe_chunk_max_len ({self.moe_chunk_max_len}) must be >= "
                f"moe_ep_size ({self.moe_ep_size}), otherwise every per-chunk "
                f"double routing would exceed the configured token budget."
            )

        if self.num_experts % self.moe_ep_size:
            raise RuntimeError(
                f"num_experts={self.num_experts} must be divisible by "
                f"moe_ep_size={self.moe_ep_size}"
            )
        self.enable_mega_moe = infer_config.model_config.custom_params.get(
            "enable_mega_moe", False)
        self.enable_moe_bf16_mode = infer_config.model_config.custom_params.get(
            "enable_moe_bf16_mode", False)
        self.prefill_func = (
            self._moe_mega_w4a8
            if self.enable_mega_moe and self.moe_ep_size > 1
            else self.moe_infer_double_routing
        )
        self.local_expert_count = self.num_experts // self.moe_ep_size
        self.local_expert_start = self.moe_ep_rank * self.local_expert_count
        expert_hidden = config.routed_expert_hidden_size
        self.gate = KimiMoEGate(config)
        self.experts = KimiSituMoEGMM(
            config,
            hidden_size=expert_hidden,
            ep_size=self.moe_ep_size,
            ep_rank=self.moe_ep_rank,
            enable_moe_bf16_mode=self.enable_moe_bf16_mode
        )
        self.shared_experts = None
        if config.num_shared_experts > 0:
            # Always active, so it splits the intermediate dimension rather
            # than sharding by expert like the routed branch.
            dense_tp_size, dense_tp_rank, dense_tp_group = _dense_tp(parallel, comm_manager)
            self.shared_experts = KimiMLP(
                config,
                intermediate_size=config.moe_intermediate_size * config.num_shared_experts,
                tp_size=dense_tp_size,
                tp_rank=dense_tp_rank,
                tp_group=dense_tp_group,
                prefix=f"{prefix}.shared_experts",
                enable_moe_bf16_mode=self.enable_moe_bf16_mode,
            )
        self.routed_expert_down_proj = nn.Linear(config.hidden_size, expert_hidden, bias=False)
        self.routed_expert_norm = KimiRMSNorm(
            expert_hidden,
            config.rms_norm_eps,
        )
        self.routed_expert_up_proj = nn.Linear(expert_hidden, config.hidden_size, bias=False)

        self.npu_events = tuple(create_event(self.exe_mode, self.enable_multi_streams) for i in range(2))

        self.top_k = config.num_experts_per_token
        self.moe_intermediate_size = config.moe_intermediate_size


    def _forward_shared_expert(self, switch, main, stream, identity):
        # -- Shared experts on side stream ------------------------------------
        record_event(switch, self.npu_events, 0, self.exe_mode)
        with npu_stream_switch(switch, stream, exe_mode=self.exe_mode):
            wait_event(switch, self.npu_events, 0, self.exe_mode)
            shared_out = self.shared_experts(identity)
            record_event(switch, self.npu_events, 1, self.exe_mode)
        record_stream(switch, shared_out, main, self.exe_mode)
        return shared_out

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        is_prefill: bool = True,
        moe_ctx: Optional[MoEContext] = None,
    ) -> torch.Tensor:

        shared_stream = moe_ctx.shared_stream if moe_ctx is not None else None
        switch = self.enable_multi_streams and not is_prefill
        main_stream = torch.npu.current_stream()
        # make sure hidden_states be reserved for sharedstream
        record_stream(switch, hidden_states, shared_stream, self.exe_mode)

        topk_idx, topk_weight = self.gate(hidden_states)
        routed_states = self.routed_expert_down_proj(hidden_states)

        if self.shared_experts is not None:
            shared_output = self._forward_shared_expert(switch, main_stream, shared_stream, hidden_states)

        if is_prefill:
            # Prefill EP: the supported multi-card path uses MXFP4 experts, so
            # quantize the activation before routing so x and its MX scale are
            # routed together. Otherwise experts would quantize `expanded_x`,
            # whose active_expert_range drop rows are undefined.
            routed_output = self._moe_prefill(routed_states, topk_idx, topk_weight, moe_ctx)
        else:
            # Decode uses MegaMoE when enabled; otherwise MC2 dispatch/combine
            # routes each token precisely to its experts without drop rows.
            if self.enable_mega_moe and self.moe_ep_size > 1:
                routed_output = self._moe_mega_w4a8(
                    routed_states, topk_idx, topk_weight, moe_ctx
                )
            else:
                routed_output = self._moe_mc2_decode(
                    routed_states, topk_idx, topk_weight
                )

        routed_output = self.routed_expert_norm(routed_output)
        routed_output = self.routed_expert_up_proj(routed_output)
        if self.shared_experts is not None:
            wait_event(switch, self.npu_events, 1, self.exe_mode)
            moe_output = routed_output + shared_output
        else:
            moe_output = routed_output
        return moe_output

    def _moe_prefill(self, routed_states, topk_idx, topk_weight, moe_ctx):
        """Prefill EP, MXFP4 experts: double routing or MegaMoE.

        Quantizes the activation to MXFP8 BEFORE routing so x and its per-token
        MX scale route together, and the experts receive that scale rather than
        re-quantizing ``expanded_x`` (whose active_expert_range drop rows are
        undefined and NaN on real weights). ``routed_states`` is this rank's token
        shard of the latent activation.

        When the gathered batch exceeds ``self.moe_chunk_max_len``, the
        pipeline is run in chunks to bound the peak expanded_x and
        finalize-table allocations.  Chunk boundaries are identical on every
        rank in the EP group, so collective ordering is preserved and the
        per-chunk outputs concatenate to the full result.
        """
        plan = _moe_chunk_plan(
            routed_states.shape[0], self.moe_ep_size, self.moe_chunk_max_len
        )
        if len(plan) == 1:
            return self.prefill_func(routed_states, topk_idx, topk_weight, moe_ctx)

        # Verify all EP ranks computed the same chunk count
        # An SP-pad imbalance would give this rank a different plan, causing
        # the per-chunk collectives to deadlock with no error. The MAX-reduce
        # turns that into a clean RuntimeError so the cluster can be reset.
        if self.moe_ep_size > 1:
            plan_len = torch.tensor([len(plan)], dtype=torch.int32,
                                    device=routed_states.device)
            dist.all_reduce(plan_len, op=dist.ReduceOp.MAX,
                            group=self.moe_ep_group)
            if plan_len.item() != len(plan):
                raise RuntimeError(
                    f"MoE chunk plan diverged: rank has {len(plan)} chunks "
                    f"({routed_states.shape[0]} local tokens), EP max is "
                    f"{plan_len.item()}.  Check that SP padding is identical "
                    f"on every rank in this EP group."
                )

        local_tokens, h = routed_states.shape
        output = routed_states.new_empty(local_tokens, h)
        offset = 0
        for chunk_len in plan:
            end = offset + chunk_len
            chunk = self.prefill_func(
                routed_states[offset:end],
                topk_idx[offset:end],
                topk_weight[offset:end],
                moe_ctx
            )
            output[offset:end] = chunk
            offset = end
        return output

    def _moe_mega_w4a8(self, routed_states, topk_idx, topk_weight, moe_ctx):
        """Run routed experts through the MegaMoE dispatch/compute/combine path.

        Dispatch + GMM1 + SiTU + GMM2 + Combine are completed by a single mega_moe
        operator, replacing the double-routing pipeline used by Prefill and the
        MC2 dispatch/combine pipeline used by Decode.

        sym_buffer is allocated once by KimiLinearForCausalLM after communication-domain
        registration, passed through MoEContext, and reused throughout inference.
        """
        if moe_ctx is None or moe_ctx.mega_sym_buffer is None:
            raise RuntimeError("MegaMoE requires an initialized sym_buffer")
        sym_buf = moe_ctx.mega_sym_buffer
        l1 = [self.experts.w13_weight]
        l1_s = [self.experts.w13_weight_scale]
        l2 = [self.experts.w2_weight]
        l2_s = [self.experts.w2_weight_scale]
        y, _ = mega_moe(
            x=routed_states,
            topk_ids=topk_idx.to(torch.int32),
            topk_weights=topk_weight,
            l1_weights=l1,
            l1_weights_sf=l1_s,
            l2_weights=l2,
            l2_weights_sf=l2_s,
            weight1_type=torch_npu.float4_e2m1fn_x2,
            weight2_type=torch_npu.float4_e2m1fn_x2,
            sym_buffer=sym_buf,
            activation="situglu",
            activation_params={
                "beta": self.experts.situ.beta,
                "linear_beta": self.experts.situ.linear_beta,
            },
        )
        return y

    def dispatch_double_routing(self, tokens_per_expert, expanded_x, pertoken_scale):
        """Dispatch expanded tokens and scales to their expert-owner ranks."""
        group = self.moe_ep_group
        owner_counts = torch.empty_like(tokens_per_expert)
        dist.all_to_all_single(owner_counts, tokens_per_expert, group=group)
        count_matrix = torch.stack((owner_counts, tokens_per_expert), dim=0)
        count_matrix = count_matrix.view(2, self.moe_ep_size, -1).sum(-1)
        count_lists = count_matrix.cpu().tolist()
        output_splits = count_lists[0]
        input_splits = count_lists[1]

        owner_x = expanded_x.new_empty(sum(output_splits), expanded_x.shape[-1])
        dist.all_to_all_single(
            owner_x,
            expanded_x,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=group,
        )
        owner_scale = pertoken_scale.new_empty(
            sum(output_splits), *pertoken_scale.shape[1:]
        )
        dist.all_to_all_single(
            owner_scale,
            pertoken_scale,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=group,
        )
        return owner_counts, owner_x, owner_scale, input_splits, output_splits

    def forward_expert(self, owner_x, owner_counts, owner_scale):
        """Re-route owner inputs, run local experts, then restore source order."""
        ordered_x, ordered_scale, unsort_idx, local_counts = torch_npu.npu_moe_re_routing(
            owner_x,
            owner_counts.view(self.moe_ep_size, -1),
            per_token_scales=owner_scale,
        )

        ordered = self.experts(
            ordered_x,
            local_counts,
            group_list_type=1,
            pertoken_scale=ordered_scale,
        )
        owner_output = torch.index_select(
            ordered, 0, unsort_idx.float().argsort().int()
        )
        return owner_output

    def forward_combine_double_routing(
        self, owner_output, expanded_x, input_splits, output_splits
    ):
        """Return expert outputs to the source ranks in expanded-token order."""
        local_output = owner_output.new_empty(expanded_x.shape)
        dist.all_to_all_single(
            local_output,
            owner_output,
            output_split_sizes=input_splits,
            input_split_sizes=output_splits,
            group=self.moe_ep_group,
        )
        return local_output

    def moe_infer_double_routing(
        self, routed_states, topk_idx, topk_weight, moe_ctx
    ):
        """Run one Prefill chunk through the V3.2-style double-routing path."""
        _ = moe_ctx
        if self.enable_moe_bf16_mode:
            topk_weight = topk_weight.bfloat16()
        local_tokens = routed_states.shape[0]
        x_q, scale = torch_npu.npu_dynamic_mx_quant(
            routed_states, dst_type=torch.float8_e4m3fn
        )
        routing_kwargs = dict(
            expert_idx=topk_idx.to(torch.int32),
            active_num=topk_idx.shape[0] * topk_idx.shape[1],
            expert_num=self.num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, self.num_experts],
            quant_mode=-1,
        )
        expanded_x, row_idx, tokens_per_expert, _ = torch_npu.npu_moe_init_routing_v2(
            x_q.view(torch.bfloat16), **routing_kwargs
        )
        expanded_x = expanded_x.view(x_q.dtype)
        exp_scale, _, _, _ = torch_npu.npu_moe_init_routing_v2(
            scale.reshape(local_tokens, -1).to(torch.bfloat16), **routing_kwargs
        )
        pertoken_scale = exp_scale.to(scale.dtype).view(-1, *scale.shape[1:])
        owner_counts, owner_x, owner_scale, input_splits, output_splits = (
            self.dispatch_double_routing(
                tokens_per_expert, expanded_x, pertoken_scale
            )
        )
        owner_output = self.forward_expert(owner_x, owner_counts, owner_scale)
        local_output = self.forward_combine_double_routing(
            owner_output, expanded_x, input_splits, output_splits
        )
        hidden = torch_npu.npu_moe_finalize_routing(
            local_output.float() if not self.enable_moe_bf16_mode else local_output,
            skip1=None, skip2=None, bias=None,
            scales=topk_weight.float() if not self.enable_moe_bf16_mode else topk_weight,
            expanded_src_to_dst_row=row_idx,
            export_for_source_row=None,
            drop_pad_mode=2,
        )
        return hidden.to(routed_states.dtype)

    def _moe_mc2_decode(self, routed_states, topk_idx, topk_weight):
        """Decode EP via MC2 dispatch/combine.

        ``routed_states`` holds this rank's own decode tokens (DP), not a shard
        of a shared sequence, so no all_gather/reduce_scatter is needed. The
        dispatch routes each token precisely to its experts and quantizes the
        latent activation to MXFP8 before communication. The routed MX scale is
        reshaped to the layout required by the MXFP4 expert GMM.
        """
        group_name = self.moe_ep_group_mc2_name
        ids = topk_idx.to(torch.int32)
        common_kwargs = dict(
            moe_expert_num=self.num_experts,
            global_bs=0,
            x_active_mask=None,
            group_ep=group_name,
            group_tp=group_name,
            ep_world_size=self.moe_ep_size,
            ep_rank_id=self.moe_ep_rank,
            tp_world_size=1,
            tp_rank_id=0,
            expert_shard_type=0,
            shared_expert_num=0,
            shared_expert_rank_num=0,
        )
        # quant_mode=4 performs MX dynamic quantization before communicating
        # the MXFP8 activations and E8M0 scales.
        dispatch = torch_npu.npu_moe_distribute_dispatch_v2(
            x=routed_states,
            expert_ids=ids,
            quant_mode=4,
            y_dtype=torch.float8_e4m3fn,
            **common_kwargs,
        )
        expand_x = dispatch[0]
        dynamic_scale = reshape_mx_scale(dispatch[1])
        expand_idx = dispatch[2]
        expert_token_num = dispatch[3]
        ep_recv_counts = dispatch[4]
        tp_recv_counts = dispatch[5] if len(dispatch) > 5 else None

        expert_output = self.experts(
            expand_x,
            expert_token_num,
            group_list_type=1,
            pertoken_scale=dynamic_scale,
        )
        return torch_npu.npu_moe_distribute_combine_v2(
            expert_output, ids, expand_idx, ep_recv_counts,
            topk_weight,
            tp_send_counts=tp_recv_counts,
            expand_scales=None,
            comm_quant_mode=0,
            **common_kwargs,
        )


def _uninitialized(module_cls, *args, **kwargs):
    """Build a module without running its parameter initializer.

    ``nn.Linear`` and ``nn.Embedding`` initialize unconditionally on
    construction, and for the two vocabulary-sized modules that is expensive
    CPU RNG immediately overwritten by the checkpoint. Only those two are built
    this way; the rest are small enough that the saving would not pay for the
    added indirection.

    Safe because ``load_weights`` refuses to finish while any parameter is
    still without a checkpoint tensor, so uninitialized memory cannot reach the
    forward pass.
    """
    return torch.nn.utils.skip_init(module_cls, *args, **kwargs)


def _sp_pad_metadata(metadata: ForwardMetaData, pad_len: int) -> ForwardMetaData:
    """Describe the sequence-parallel pad as one more request segment.

    Attention then runs on the padded stream with the real segments byte for
    byte unchanged: the pad segment writes its keys, values and recurrent state
    to the null block and reads them back from the same offsets, so it never
    touches another request's cache. The caller keeps the original metadata,
    whose cumulative lengths stop at the real tokens, for the tail select that
    drops the pad again.
    """

    # The null block is block 0: the pool pops one id off its free queue and
    # keeps it as the placeholder, so no request is ever given it and a slot
    # below block_size can only belong to the pad. check_model_settings keeps
    # block_size at or above attn_tp, which bounds pad_len.
    padded = dict(metadata)
    padded["actual_seq_lengths_q"] = torch.cat((
        metadata["actual_seq_lengths_q"],
        metadata["actual_seq_lengths_q"].new_full((1,), pad_len),
    ))
    padded["actual_seq_lengths_kv"] = torch.cat((
        metadata["actual_seq_lengths_kv"],
        metadata["actual_seq_lengths_kv"].new_full((1,), pad_len),
    ))
    padded["actual_seq_lengths_cu_q"] = torch.cat((
        metadata["actual_seq_lengths_cu_q"],
        metadata["actual_seq_lengths_cu_q"][-1:] + pad_len,
    ))
    padded["actual_seq_lengths_cu_kv"] = torch.cat((
        metadata["actual_seq_lengths_cu_kv"],
        metadata["actual_seq_lengths_cu_kv"][-1:] + pad_len,
    ))
    cu_list = metadata.get("actual_seq_lengths_cu_list_kv")
    if cu_list is not None:
        padded["actual_seq_lengths_cu_list_kv"] = [*cu_list, cu_list[-1] + pad_len]
    return padded


class KimiShortConvolution(nn.Module):
    """Depthwise causal convolution with an explicit decode cache."""

    def __init__(self, hidden_size: int, kernel_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(
            torch.empty(hidden_size, 1, kernel_size, dtype=torch.bfloat16)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.register_buffer("_conv_weight", None, persistent=False)

    def build_conv_weight(self) -> None:
        with torch.no_grad():
            self._conv_weight = (
                self.weight.squeeze(1).transpose(0, 1).contiguous()
            )

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor],
        block_table: torch.Tensor,
        is_prefill: bool,
        query_start_loc: Optional[torch.Tensor] = None,
        num_accepted_tokens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor]:
        if is_prefill:
            has_initial_state = torch.zeros(
                size=[query_start_loc.shape[0] - 1],
                dtype=torch.int32,
                device=x.device,
            )
            y = torch.ops.cann_ops_transformer.causal_conv1d_fn(
                x=x,
                conv_states=cache,
                cache_indices=block_table,
                weight=self._conv_weight,
                bias=None,
                query_start_loc=query_start_loc,
                has_initial_state=has_initial_state,
            )
        else:
            q_len = x.shape[1] if x.dim() == 3 else 1
            flatten_decode = q_len > 1
            original_shape = x.shape if flatten_decode else None
            if flatten_decode:
                x = x.view(-1, x.shape[-1])
            y = torch.ops.cann_ops_transformer.causal_conv1d_update(
                x=x,
                conv_state=cache,
                conv_state_indices=block_table,
                weight=self._conv_weight,
                bias=None,
                query_start_loc=query_start_loc,
                num_accepted_tokens=num_accepted_tokens,
            )
            if flatten_decode:
                y = y.view(original_shape)
        return y


class KimiDeltaAttention(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        layer_idx: int,
        infer_config: Optional[InferenceConfig] = None,
        comm_manager: Optional[CommManager] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        linear = config.linear_attn_config
        self.layer_idx = layer_idx
        self.head_dim = linear["head_dim"]
        parallel = None if infer_config is None else infer_config.parallel_config
        self.attn_tp_size = 1 if parallel is None else parallel.attn_tp_size
        # The parallel projections take whole-model sizes and divide internally,
        # so they are built from total_num_heads; num_heads is this rank's share
        # and drives the forward pass, the convolutions and the state cache.
        self.total_num_heads = linear["num_heads"]
        self.num_heads = self.total_num_heads // self.attn_tp_size
        self.attn_tp_group = (
            None
            if self.attn_tp_size == 1
            else comm_manager.get_group("attn_tp_group")
        )
        self.attn_tp_rank = (
            0 if self.attn_tp_size == 1 else comm_manager.get_rank("attn_tp_group")
        )
        quant_config = getattr(config, "quant_config", None)
        self.use_flash_kda = infer_config.model_config.custom_params.get("enable_flash_kda", True)
        if self.use_flash_kda and _flash_kda_impl is None:
            raise ImportError("enable_flash_kda=True but ops.cannbot_dsl.flash_kda is not available")
        self.use_fused_recurrent_kda = infer_config.model_config.custom_params.get("enable_fused_recurrent_kda", True)
        if self.use_fused_recurrent_kda and _recurrent_kda_impl is None:
            raise ImportError("enable_fused_recurrent_kda=True but fused_recurrent_kda is not available")
        # Between layers, Prefill holds a token-SP shard while Decode holds a
        # request-DP shard. Attention gathers either layout for head TP and
        # reduce-scatters the projected result back to the owning ranks.
        self.register_buffer(
            "kda_transition_mask",
            torch.triu(torch.ones(_KDA_CHUNK_SIZE, _KDA_CHUNK_SIZE, dtype=torch.bool)),
            persistent=False,
        )
        self.register_buffer(
            "kda_attention_mask",
            torch.triu(
                torch.ones(_KDA_CHUNK_SIZE, _KDA_CHUNK_SIZE, dtype=torch.bool),
                diagonal=1,
            ),
            persistent=False,
        )
        self.register_buffer(
            "kda_identity",
            torch.eye(_KDA_CHUNK_SIZE, dtype=torch.float32),
            persistent=False,
        )
        projection_size = self.head_dim * self.num_heads
        self.projection_size = projection_size
        self.qkv_projection_size = 3 * projection_size
        # The parallel layers take the global output size and shard it
        # themselves, unlike the bare parameters sized per rank above.
        total_projection_size = self.head_dim * self.total_num_heads
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            head_size=self.head_dim,
            total_num_heads=linear["num_heads"],
            total_num_kv_heads=linear["num_heads"],
            bias=False,
            skip_bias_add=False,
            tp_size=self.attn_tp_size,
            tp_rank=self.attn_tp_rank,
            quant_config=None,
            prefix="self_attn.qkv_proj",
            return_bias=False,
        )
        kernel_size = linear["short_conv_kernel_size"]
        self.qkv_conv1d = KimiShortConvolution(self.qkv_projection_size, kernel_size)
        # The checkpoint stores 96 logical per-head values followed by 32 zero
        # padding values. load_weights removes the padding and shards the heads.
        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16))
        )
        # The low-rank legs of the decay and gate projections land in head_dim,
        # which attn_tp does not split, so they stay replicated.
        self.f_a_proj = ReplicatedLinear(
            config.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.f_a_proj",
        )
        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            total_projection_size,
            bias=False,
            tp_size=self.attn_tp_size,
            tp_rank=self.attn_tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
        self.dt_bias = nn.Parameter(torch.zeros(projection_size, dtype=torch.float32))
        self.b_proj = ColumnParallelLinear(
            config.hidden_size,
            self.total_num_heads,
            bias=False,
            tp_size=self.attn_tp_size,
            tp_rank=self.attn_tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.b_proj",
        )
        self.use_full_rank_gate = linear.get("use_full_rank_gate", False)
        self.gate_lower_bound = linear.get("gate_lower_bound")
        if self.use_full_rank_gate:
            self.g_proj = ColumnParallelLinear(
                config.hidden_size,
                total_projection_size,
                bias=False,
                tp_size=self.attn_tp_size,
                tp_rank=self.attn_tp_rank,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )
        else:
            self.g_a_proj = ReplicatedLinear(
                config.hidden_size,
                self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_a_proj",
            )
            self.g_b_proj = ColumnParallelLinear(
                self.head_dim,
                total_projection_size,
                bias=False,
                tp_size=self.attn_tp_size,
                tp_rank=self.attn_tp_rank,
                quant_config=quant_config,
                prefix=f"{prefix}.g_b_proj",
            )
        self.o_norm = KimiRMSNorm(
            self.head_dim,
            config.rms_norm_eps,
            dtype=torch.float32,
        )
        self.o_proj = RowParallelLinear(
            total_projection_size,
            config.hidden_size,
            bias=False,
            tp_size=self.attn_tp_size,
            tp_rank=self.attn_tp_rank,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.attn_type = "Mamba"

    def _state_block_ids(
        self, forward_metadata: ForwardMetaData, cache_kind: str = "KDAConv"
    ) -> torch.Tensor:
        """Resolve the speculative KDA cache blocks for this step."""
        block_table = forward_metadata["block_table"][cache_kind]
        if cache_kind == "KDARecurrent":
            return block_table
        return block_table[:, 0]

    def _chunk_kda_dispatch(
        self,
        inputs: KdaInputs,
        gate_params: KdaGateParams,
        initial_state: torch.Tensor,
        query_boundaries: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Prefill: dispatch to flash_kda or torch reference by use_flash_kda.
        # Returns: output [tokens, H, D] bf16, final_state [batch, H, D, D] fp32.
        if self.use_flash_kda and gate_params.lower_bound is not None:
            return self._prefill_flash_kda(
                inputs, gate_params, initial_state, query_boundaries,
            )
        return self._prefill_torch_kda(
            inputs, gate_params, initial_state, query_boundaries,
        )

    def _slice_request_inputs(
        self,
        inputs: KdaInputs,
        start: int,
        end: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, value, raw_gate, raw_beta = inputs
        return (
            query[start:end].unsqueeze(0),
            key[start:end].unsqueeze(0),
            value[start:end].unsqueeze(0),
            raw_gate[start:end].unsqueeze(0),
            raw_beta[start:end].unsqueeze(0),
        )

    def _prefill_flash_kda(
        self,
        inputs: KdaInputs,
        gate_params: KdaGateParams,
        initial_state: torch.Tensor,
        query_boundaries: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Prefill flash_kda: fuses L2 norm + gate activation + beta sigmoid, per-request chunked.
        # Returns: output [tokens, H, D] bf16, final_state [batch, H, D, D] fp32.
        outputs = []
        final_states = []
        for request, (start, end) in enumerate(
            zip(query_boundaries, query_boundaries[1:])
        ):
            q, k, v, g, b = self._slice_request_inputs(inputs, start, end)
            tokens, key_dim = q.shape[1], q.shape[3]
            pad_len = (-tokens) % _KDA_CHUNK_SIZE
            if pad_len:
                q, k, v = (
                    F.pad(t, (0, 0, 0, 0, 0, pad_len)) for t in (q, k, v)
                )
                g = F.pad(g, (0, 0, 0, 0, 0, pad_len), value=float('-inf'))
                b = F.pad(b, (0, 0, 0, pad_len), value=float('-inf'))
            q, k, v, g, b = (t.contiguous() for t in (q, k, v, g, b))
            output, state = _flash_kda_impl(
                q, k, v, g=g, beta=b,
                scale=1.0 / math.sqrt(key_dim),
                initial_state=initial_state[request:request + 1],
                A_log=gate_params.a_log.data,
                dt_bias=gate_params.dt_bias,
                lower_bound=gate_params.lower_bound,
                layout_qkv="BSND",
            )
            if pad_len:
                output = output[:, :tokens].contiguous()
            outputs.append(output.squeeze(0))
            final_states.append(state.squeeze(0))
        return torch.cat(outputs), torch.stack(final_states)

    def _prefill_torch_kda(
        self,
        inputs: KdaInputs,
        gate_params: KdaGateParams,
        initial_state: torch.Tensor,
        query_boundaries: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Prefill torch reference: Python-side L2 norm + gate/beta activation, pure Python chunk loop.
        # Returns: output [tokens, H, D] bf16, final_state [batch, H, D, D] fp32.
        outputs = []
        final_states = []
        gate_scale = gate_params.a_log.exp().view(-1, 1)
        for request, (start, end) in enumerate(
            zip(query_boundaries, query_boundaries[1:])
        ):
            q_src, k_src, v_src, gate_raw, beta_raw = self._slice_request_inputs(inputs, start, end)
            gate_raw = gate_raw.float()
            beta_raw = beta_raw.float()
            gate_input = gate_raw + gate_params.dt_bias
            if gate_params.lower_bound is not None:
                g = float(gate_params.lower_bound) * torch.sigmoid(gate_scale * gate_input)
            else:
                g = -gate_scale * _softplus(gate_input)
            b = beta_raw.sigmoid()
            torch_initial_state = initial_state[
                request:request + 1
            ].transpose(-1, -2).contiguous()
            output, state = _torch_chunk_kda(
                q_src, k_src, v_src, g, b, torch_initial_state,
                self.kda_transition_mask,
                self.kda_attention_mask,
                self.kda_identity,
            )
            state = state.transpose(-1, -2).contiguous()
            outputs.append(output.squeeze(0))
            final_states.append(state.squeeze(0))
        return torch.cat(outputs), torch.stack(final_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_metadata: ForwardMetaData,
        layer_cache: dict,
        query_start_loc: Optional[torch.Tensor] = None,
        query_boundaries: Optional[list[int]] = None,
    ) -> torch.Tensor:
        # Keep the gathered hidden and KDA projection temporaries inside the
        # core call. Only the narrow per-head gate and KDA output cross this
        # boundary, so the full [tokens, hidden] tensor is released before
        # o_proj allocates another full-width output.
        gate, output = self._forward_core(
            hidden_states,
            forward_metadata,
            layer_cache,
            query_start_loc,
            query_boundaries,
        )
        return self._project_out(gate, output)

    def _forward_core(
        self,
        hidden_states: torch.Tensor,
        forward_metadata: ForwardMetaData,
        layer_cache: dict,
        query_start_loc: Optional[torch.Tensor],
        query_boundaries: Optional[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Prefill reconstructs the token stream; Decode reconstructs the
        # request batch for attention TP. Decode is DP-TP-DP, not SP. The
        # global ForwardMetaData already describes this reconstructed input.
        hidden_states = all_gather_first_dim(
            hidden_states, self.attn_tp_group, self.attn_tp_size
        )
        # g_proj needs the gathered hidden, but its output is only the local
        # head shard. Compute it here so forward need not retain the much wider
        # gathered tensor until output projection.
        gate = self._project_gate(hidden_states)
        tokens = hidden_states.shape[0]
        sp_pad_len = (
            tokens - forward_metadata["prompt_tokens"]
            if self.attn_tp_size > 1 and forward_metadata["is_prefill"]
            else 0
        )
        if forward_metadata["is_prefill"]:
            hidden_states = _unpad_kda_input(hidden_states, sp_pad_len)
            batch = len(query_boundaries) - 1
        else:
            batch = forward_metadata["actual_seq_lengths_q"].shape[0]
        tokens = hidden_states.shape[0]
        state_ids = self._state_block_ids(forward_metadata, "KDAConv")

        input_states = (
            hidden_states
            if forward_metadata["is_prefill"]
            else hidden_states.view(batch, -1, hidden_states.shape[-1])
        )

        fused_qkv = self.qkv_proj(input_states)
        num_accepted_tokens = forward_metadata.get("conv_num_accepted_tokens")
        mixqkv = self.qkv_conv1d(
            fused_qkv,
            layer_cache["conv_state"],
            state_ids,
            forward_metadata["is_prefill"],
            query_start_loc,
            num_accepted_tokens,
        )
        q, k, v = mixqkv.split(self.projection_size, dim=-1)

        shape = (*input_states.shape[:-1], self.num_heads, self.head_dim)
        q, k, v = q.view(shape), k.view(shape), v.view(shape)
        raw_decay = self.f_b_proj(self.f_a_proj(input_states)).view(shape)
        raw_beta = self.b_proj(input_states)
        dt_bias = self.dt_bias.view(self.num_heads, self.head_dim)

        if forward_metadata["is_prefill"]:
            initial_state = torch.zeros(
                len(query_boundaries) - 1, self.num_heads,
                self.head_dim, self.head_dim,
                dtype=torch.float32, device=q.device,
            )
            output, state = self._chunk_kda_dispatch(
                KdaInputs(q, k, v, raw_decay, raw_beta),
                KdaGateParams(self.A_log, dt_bias, self.gate_lower_bound),
                initial_state, query_boundaries,
            )
            recurrent_state_ids = self._state_block_ids(
                forward_metadata, "KDARecurrent"
            )[:, 0]
            self.update_mamba_cache(
                recurrent_state_ids, state, layer_cache["recurrent_state"]
            )
            output = _pad_kda_output(output, sp_pad_len)
        else:
            if self.use_fused_recurrent_kda:
                output = self._decode_fused_kda(
                    KdaInputs(q, k, v, raw_decay, raw_beta),
                    KdaGateParams(self.A_log, dt_bias, self.gate_lower_bound),
                    forward_metadata,
                    layer_cache["recurrent_state"],
                )
            else:
                gate_input = raw_decay + dt_bias
                gate_scale = self.A_log.float().exp().view(self.num_heads, 1)
                use_safe_gate = self.gate_lower_bound is not None
                if use_safe_gate:
                    decay = float(self.gate_lower_bound) * torch.sigmoid(
                        gate_scale * gate_input
                    )
                else:
                    decay = -gate_scale * _softplus(gate_input)
                beta = raw_beta.float().sigmoid()
                output = self._decode_gdr(
                    KdaInputs(q, k, v, decay, beta),
                    forward_metadata,
                    layer_cache["recurrent_state"],
                )
            output = output.view(tokens, *output.shape[2:])
        return gate, output

    def update_mamba_cache(
        self, indices: torch.Tensor, values: torch.Tensor, cache: torch.Tensor
    ) -> None:
        indices = indices.view(-1)
        if values.device != cache.device or values.dtype != cache.dtype:
            values = values.to(device=cache.device, dtype=cache.dtype)
        torch_npu.npu_scatter_nd_update_(cache, indices.view(-1, 1), values)

    def _decode_fused_kda(
        self,
        inputs: KdaInputs,
        gate_params: KdaGateParams,
        forward_metadata: ForwardMetaData,
        recurrent_state_cache: torch.Tensor,
    ) -> torch.Tensor:
        # Decode fused_recurrent_kda: fuses L2 norm + gate activation + beta sigmoid, per-token recurrent.
        # State updates recurrent_state_cache in-place, returns out [B, S, H, D] bf16.
        query, key, value, g, raw_beta = inputs
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        g = g.contiguous()
        batch, seq, _, key_dim = query.shape
        scale = 1 / math.sqrt(key_dim)
        ssm_state_indices = forward_metadata["ssm_state_indices"]
        if ssm_state_indices.numel() != batch * seq:
            raise RuntimeError(
                "KDA recurrent state indices must match the fixed Verify width"
            )
        b = raw_beta.unsqueeze(-1).contiguous()
        out = _recurrent_kda_impl(
            query, key, value,
            state=recurrent_state_cache,
            beta=b,
            g=g,
            scale=scale,
            A_log=gate_params.a_log.data,
            dt_bias=gate_params.dt_bias,
            lower_bound=gate_params.lower_bound,
            layout_qkv="BSND",
            ssm_state_indices=ssm_state_indices.contiguous(),
            num_accepted_tokens=forward_metadata.get("num_accepted_tokens"),
        )
        return out

    def _decode_gdr(
        self,
        inputs: KdaInputs,
        forward_metadata: ForwardMetaData,
        recurrent_state_cache: torch.Tensor,
    ) -> torch.Tensor:
        # Decode gdr: Python-side L2 norm + gate/beta activation, calls npu_recurrent_gated_delta_rule.
        # State updates recurrent_state_cache in-place, returns out [B, S, H, D] bf16.
        # inputs carries already-activated decay and sigmoid'd beta for the gdr path.
        query, key, value, decay, beta = inputs
        batch, seq, num_heads, key_dim = query.shape
        value_dim = value.shape[-1]
        tokens = batch * seq
        scale = 1 / math.sqrt(key_dim)
        ssm_state_indices = forward_metadata["ssm_state_indices"]
        if ssm_state_indices.numel() != batch * seq:
            raise RuntimeError(
                "GDR state indices must match the fixed Verify width"
            )
        q = _l2_normalize(query).reshape(tokens, num_heads, key_dim).to(
            torch.bfloat16
        )
        k = _l2_normalize(key).view(tokens, num_heads, key_dim).to(
            torch.bfloat16
        )
        v = value.view(tokens, num_heads, value_dim).to(torch.bfloat16)
        b = beta.view(tokens, num_heads).to(torch.bfloat16)
        gk = decay.view(tokens, num_heads, key_dim).float()
        core_attn_out = torch_npu.npu_recurrent_gated_delta_rule(
            q,
            k,
            v,
            recurrent_state_cache,
            beta=b,
            scale=scale,
            actual_seq_lengths=forward_metadata["actual_seq_lengths_q"],
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=forward_metadata.get("num_accepted_tokens"),
            g=None,
            gk=gk,
        )
        return core_attn_out.view(batch, seq, num_heads, value_dim)

    def _project_gate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return (
            self.g_proj(hidden_states)
            if self.use_full_rank_gate
            else self.g_b_proj(self.g_a_proj(hidden_states))
        )

    def _project_out(
        self, gate: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        gate = gate.view(output.shape)
        output = self.o_norm(output) * torch.sigmoid(gate.float()).to(output.dtype)
        output = self.o_proj(output.view(output.shape[0], -1))
        return reduce_scatter_first_dim(
            output, self.attn_tp_group, self.attn_tp_size
        )


class KimiMLAAttention(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        layer_idx: int,
        infer_config: Optional[InferenceConfig] = None,
        comm_manager: Optional[CommManager] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        parallel = None if infer_config is None else infer_config.parallel_config
        self.attn_tp_size = 1 if parallel is None else parallel.attn_tp_size
        # The parallel projections take whole-model sizes and divide internally,
        # so they are built from total_num_heads; num_heads is this rank's share
        # and drives the forward pass and the cache geometry.
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // self.attn_tp_size
        self.attn_tp_rank = (
            0 if self.attn_tp_size == 1 else comm_manager.get_rank("attn_tp_group")
        )
        quant_config = getattr(config, "quant_config", None)
        self.attn_tp_group = (
            None
            if self.attn_tp_size == 1
            else comm_manager.get_group("attn_tp_group")
        )
        # Prefill gathers token SP before attention TP. Decode stays request-DP
        # through attention and gathers only at the output-gate TP boundary.
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.o_proj_channel_width = (
            self.total_num_heads * self.v_head_dim // self.attn_tp_size
        )
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scaling = self.q_head_dim ** -0.5
        if self.q_lora_rank is not None:
            self.q_a_proj = ReplicatedLinear(
                config.hidden_size,
                self.q_lora_rank,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.q_a_proj",
            )
            self.q_a_layernorm = KimiRMSNorm(self.q_lora_rank)
            self.q_b_proj = ColumnParallelLinear(
                self.q_lora_rank,
                self.total_num_heads * self.q_head_dim,
                bias=False,
                tp_size=self.attn_tp_size,
                tp_rank=self.attn_tp_rank,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
            self.q_b_proj_decode = ColumnParallelLinear(
                self.q_lora_rank,
                self.total_num_heads * self.q_head_dim,
                bias=False,
                tp_size=1,
                tp_rank=0,
                quant_config=quant_config,
                prefix=f"{prefix}.q_b_proj",
            )
        else:
            self.q_proj = ColumnParallelLinear(
                config.hidden_size,
                self.total_num_heads * self.q_head_dim,
                bias=False,
                tp_size=self.attn_tp_size,
                tp_rank=self.attn_tp_rank,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
            self.q_proj_decode = ColumnParallelLinear(
                config.hidden_size,
                self.total_num_heads * self.q_head_dim,
                bias=False,
                tp_size=1,
                tp_rank=0,
                quant_config=quant_config,
                prefix=f"{prefix}.q_proj",
            )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = KimiRMSNorm(self.kv_lora_rank)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.total_num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            tp_size=self.attn_tp_size,
            tp_rank=self.attn_tp_rank,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.kv_b_proj_decode = ColumnParallelLinear(
            self.kv_lora_rank,
            self.total_num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            tp_size=1,
            tp_rank=0,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_b_proj",
        )
        self.use_output_gate = config.mla_use_output_gate
        if self.use_output_gate:
            self.g_proj = ColumnParallelLinear(
                config.hidden_size,
                self.total_num_heads * self.v_head_dim,
                bias=False,
                tp_size=self.attn_tp_size,
                tp_rank=self.attn_tp_rank,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.v_head_dim,
            config.hidden_size,
            bias=False,
            tp_size=self.attn_tp_size,
            tp_rank=self.attn_tp_rank,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # ---- Framework paged (PageAttention) KV cache ----
        # Caches the compressed latent rather than the expanded per-head K/V:
        #     nope_cache  dim = kv_lora_rank     (512)
        #     rope_cache  dim = qk_rope_head_dim (64)
        # Both carry one KV head (MQA), replicated across attn_tp ranks. Decode
        # absorbs kv_b_proj into the attention; prefill expands on read.
        #
        # K3's MLA is NoPE: FA's query_rope/key_rope parameters carry one more D
        # segment concatenated into the QK dot product, and the cache write is
        # handed cos=1 / sin=0.
        self.attn_type = "FullAttention"
        self.block_size = (
            None if infer_config is None else infer_config.scheduler_config.block_size
        )
        # Split out of kv_b_proj once the checkpoint is loaded; see
        # KimiLinearForCausalLM.process_weights_after_loading.
        self.kv_b_proj_w_k = None
        self.kv_b_proj_w_v = None
        self.kv_b_proj_decode_w_k = None
        self.kv_b_proj_decode_w_v = None
        self.enable_multi_streams = infer_config.model_config.custom_params.get("enable_multi_streams", False)
        self.exe_mode = None if infer_config is None else infer_config.model_config.exe_mode
        self.npu_events_kv = tuple(create_event(self.exe_mode, self.enable_multi_streams) for i in range(2))
        self.npu_events_mla_gate = tuple(create_event(self.exe_mode, self.enable_multi_streams) for i in range(2))

    def _write_latent_cache(
        self,
        compressed: torch.Tensor,
        slot_mapping: torch.Tensor,
        layer_cache: dict,
    ):
        """RMSNorm this step's latent and scatter it into the paged blocks.

        Writes the NZ layout the absorbed decode reads back. K3 is NoPE, so the
        extra 64 key channels are cached without rotation or permutation.
        """
        k_nope, k_rope = torch.split(
            compressed, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_nope = self.kv_a_layernorm(k_nope)
        nope_cache = layer_cache["nope_cache"]
        rope_cache = layer_cache["rope_cache"]
        block_num, block_size = nope_cache.shape[:2]
        torch_npu.npu_scatter_pa_kv_cache(
            k_nope.unsqueeze(1),
            k_rope.unsqueeze(1),
            nope_cache.view(block_num, self.kv_lora_rank // _KV_CACHE_NZ_DIM, block_size, _KV_CACHE_NZ_DIM),
            rope_cache.view(block_num, self.qk_rope_head_dim // _KV_CACHE_NZ_DIM, block_size, _KV_CACHE_NZ_DIM),
            slot_mapping.view(-1),
        )

    def _prepare_query_inputs(
            self, query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split the NoPE query into the two segments consumed by MLA FA."""
        tokens, num_heads = query.shape[:2]
        query_t = query.view(tokens, num_heads, self.q_head_dim)
        query_nope, query_rope = torch.split(query_t, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        return query_nope, query_rope

    def _forward_prefill(
        self,
        query: torch.Tensor,
        compressed: torch.Tensor,
        forward_metadata: ForwardMetaData,
        layer_cache: dict,
    ) -> torch.Tensor:
        """Expand this step's latent and run prefill attention."""
        tokens = query.shape[0]
        query_nope, query_rope = self._prepare_query_inputs(query)
        k_nope, k_rope = torch.split(
            compressed, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_nope = self.kv_a_layernorm(k_nope)
        owner_indices = forward_metadata["mla_owner_token_indices"]
        # A Prefill mini batch is executed by every TP rank, but only the rank
        # that will own a request during Decode stores its replicated latent.
        # Some ranks therefore have no cache write in a given cycle.
        if owner_indices.numel() > 0:
            self._write_latent_cache(
                compressed.index_select(0, owner_indices),
                forward_metadata["slot_mapping"][self.attn_type],
                layer_cache=layer_cache,
            )
        return self._prefill_attention(
            query_nope, query_rope, k_nope, k_rope, tokens, forward_metadata
        )

    def _forward_decode(
            self,
            query: torch.Tensor,
            forward_metadata: ForwardMetaData,
            layer_cache: dict,
            kv_stream: Optional[torch.npu.Stream],
    ) -> torch.Tensor:
        """Attend against the cached latent with kv_b_proj absorbed in."""
        tokens = query.shape[0]
        query_nope, query_rope = self._prepare_query_inputs(query)
        block_table = forward_metadata["block_table"][self.attn_type]

        actual_seq_qlen = forward_metadata["actual_seq_lengths_cu_list_q"]
        actual_seq_kvlen = forward_metadata["actual_seq_lengths_list_kv"]

        return self._decode_attention(
            query_nope,
            query_rope,
            tokens,
            block_table,
            actual_seq_qlen,
            actual_seq_kvlen,
            forward_metadata.get("attention_mask"),
            layer_cache,
            kv_stream,
        )

    def _prefill_attention(
        self,
        query_nope: torch.Tensor,
        query_rope: torch.Tensor,
        k_nope: torch.Tensor,
        k_rope: torch.Tensor,
        tokens: int,
        forward_metadata: ForwardMetaData,
    ) -> torch.Tensor:
        """Expand this step's own latent through kv_b_proj and attend over it.

        One offline Prefill mini cycle carries complete prompt sequences rather
        than token chunks, so attention uses only this cycle's expanded K/V and
        never reads earlier paged blocks back.
        """
        latent = k_nope.view(1, tokens, self.kv_lora_rank)
        # [N, T, qk_nope_head_dim] and [N, T, v_head_dim]
        key_nope = torch.matmul(latent, self.kv_b_proj_w_k.permute(0, 2, 1))
        value = torch.matmul(latent, self.kv_b_proj_w_v)
        key_rope = k_rope.view(1, tokens, self.qk_rope_head_dim).repeat(
            self.num_heads, 1, 1
        )
        cu_kvlen = forward_metadata["actual_seq_lengths_cu_list_kv"]
        output, _ = torch_npu.npu_fused_infer_attention_score_v2(
            query_nope.transpose(0, 1),
            key_nope,
            value,
            query_rope=query_rope.transpose(0, 1),
            key_rope=key_rope,
            num_query_heads=self.num_heads,
            num_key_value_heads=self.num_heads,
            input_layout="NTD_TND",
            atten_mask=forward_metadata["attention_mask"],
            sparse_mode=3,
            actual_seq_qlen=cu_kvlen,
            actual_seq_kvlen=cu_kvlen,
            softmax_scale=self.scaling,
            next_tokens=0,
        )
        return output.reshape(tokens, self.num_heads * self.v_head_dim)

    def _decode_attention(
        self,
        query_nope: torch.Tensor,
        query_rope: torch.Tensor,
        tokens: int,
        block_table: torch.Tensor,
        actual_seq_qlen,
        actual_seq_kvlen,
        attention_mask,
        layer_cache: dict,
        kv_stream: Optional[torch.npu.Stream],
    ) -> torch.Tensor:
        """Attend against the cached latent with kv_b_proj absorbed in."""
        # W_UK folds into the query: [T, N, qk_nope] x [N, qk_nope, kv_lora]
        query_latent = torch_npu.npu_transpose_batchmatmul(
            query_nope,
            self.kv_b_proj_decode_w_k,
            bias=None,
            scale=None,
            perm_x1=(1, 0, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
        ).view(tokens, self.total_num_heads, self.kv_lora_rank)

        nope_nz, rope_nz = self._nz_cache_views(layer_cache)
        wait_event(self.enable_multi_streams, self.npu_events_kv, 1, self.exe_mode)
        batch = block_table.shape[0]
        q_len = tokens // batch
        sparse_mode = 0 if q_len == 1 else 3
        causal_mask = None if q_len == 1 else attention_mask
        output, _ = torch_npu.npu_fused_infer_attention_score_v2(
            query_latent,
            nope_nz,
            nope_nz,
            query_rope=query_rope,
            key_rope=rope_nz,
            num_query_heads=query_latent.shape[1],
            num_key_value_heads=1,
            softmax_scale=self.scaling,
            input_layout="TND_NTD",
            sparse_mode=sparse_mode,
            atten_mask=causal_mask,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
            block_table=block_table,
            block_size=self.block_size,
        )
        # TND_NTD names the output layout: [N, T, kv_lora_rank]. W_UV maps the
        # unpadded heads back to [T, N, v_head_dim].
        output = torch_npu.npu_transpose_batchmatmul(
            output[: self.total_num_heads],
            self.kv_b_proj_decode_w_v,
            bias=None,
            scale=None,
            perm_x1=(0, 1, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
        ).reshape(tokens, self.total_num_heads * self.v_head_dim)
        return output

    def _nz_cache_views(self, layer_cache: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """View the latent blocks in the NZ layout the absorbed FA expects."""
        nope_cache = layer_cache["nope_cache"]
        rope_cache = layer_cache["rope_cache"]
        blocks, block_size = nope_cache.shape[:2]
        nz = _KV_CACHE_NZ_DIM
        return (
            nope_cache.view(blocks, 1, self.kv_lora_rank // nz, block_size, nz),
            rope_cache.view(blocks, 1, self.qk_rope_head_dim // nz, block_size, nz),
        )

    def _forward_kv_attention(
            self,
            tokens: int,
            hidden_states: torch.Tensor,
            forward_metadata: ForwardMetaData,
            layer_cache: dict,
            kv_stream: Optional[torch.npu.Stream] = None,
    ) -> None:
        slot_mapping = forward_metadata["slot_mapping"][self.attn_type]
        record_stream(
            self.enable_multi_streams, slot_mapping, kv_stream, self.exe_mode
        )
        with npu_stream_switch(self.enable_multi_streams, kv_stream, exe_mode=self.exe_mode):
            wait_event(self.enable_multi_streams, self.npu_events_kv, 0, self.exe_mode)
            compressed = self.kv_a_proj_with_mqa(hidden_states)
            self._write_latent_cache(
                compressed, slot_mapping, layer_cache=layer_cache
            )
            record_event(self.enable_multi_streams, self.npu_events_kv, 1, self.exe_mode)

    def forward(
            self,
            hidden_states: torch.Tensor,
            forward_metadata: ForwardMetaData,
            layer_cache: dict,
            kv_stream: Optional[torch.npu.Stream] = None
    ) -> torch.Tensor:
        is_prefill = forward_metadata["is_prefill"]
        if is_prefill:
            hidden_states = all_gather_first_dim(
                hidden_states, self.attn_tp_group, self.attn_tp_size
            )
        tokens = hidden_states.shape[0]

        main_stream = torch.npu.current_stream()
        gate = None
        normalized_q = self.q_a_layernorm(self.q_a_proj(hidden_states))
        if is_prefill:
            query = (
                self.q_b_proj(normalized_q)
                if self.q_lora_rank is not None
                else self.q_proj(hidden_states)
            ).view(tokens, self.num_heads, self.q_head_dim)
            compressed = self.kv_a_proj_with_mqa(hidden_states)
            output = self._forward_prefill(
                query, compressed, forward_metadata, layer_cache
            )
        else:
            query = (
                self.q_b_proj_decode(normalized_q)
                if self.q_lora_rank is not None
                else self.q_proj_decode(hidden_states)
            ).view(tokens, self.total_num_heads, self.q_head_dim)
            record_stream(self.enable_multi_streams, hidden_states, kv_stream, self.exe_mode)
            record_event(self.enable_multi_streams, self.npu_events_kv, 0, self.exe_mode)
            self._forward_kv_attention(
                tokens, hidden_states, forward_metadata, layer_cache, kv_stream
            )
            output = self._forward_decode(
                query, forward_metadata, layer_cache, kv_stream
            )
            output = dp_to_tp_all_to_all(
                output,
                self.attn_tp_group,
                self.attn_tp_size,
                forward_metadata["oproj_output_rows"],
                self.o_proj_channel_width,
            )

        if self.use_output_gate:
            if not is_prefill:
                record_stream(
                    self.enable_multi_streams, hidden_states, kv_stream, self.exe_mode
                )
                record_event(
                    self.enable_multi_streams,
                    self.npu_events_mla_gate,
                    0,
                    self.exe_mode,
                )
                with npu_stream_switch(
                    self.enable_multi_streams, kv_stream, exe_mode=self.exe_mode
                ):
                    wait_event(
                        self.enable_multi_streams,
                        self.npu_events_mla_gate,
                        0,
                        self.exe_mode,
                    )
                    full_hidden = all_gather_first_dim(
                        hidden_states, self.attn_tp_group, self.attn_tp_size
                    )
                    gate = torch.sigmoid(
                        self.g_proj(full_hidden).float()
                    ).to(hidden_states.dtype)
                    record_event(
                        self.enable_multi_streams,
                        self.npu_events_mla_gate,
                        1,
                        self.exe_mode,
                    )
                wait_event(
                    self.enable_multi_streams,
                    self.npu_events_mla_gate,
                    1,
                    self.exe_mode,
                )
                record_stream(
                    self.enable_multi_streams, gate, main_stream, self.exe_mode
                )
            else:
                gate = torch.sigmoid(self.g_proj(hidden_states).float()).to(hidden_states.dtype)
            output = output * gate
        output = self.o_proj(output)
        return reduce_scatter_first_dim(
            output, self.attn_tp_group, self.attn_tp_size
        )


def _apply_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: KimiRMSNorm,
    valid_blocks: Optional[int] = None,
) -> torch.Tensor:
    if valid_blocks is None:
        valid_blocks = block_residual.shape[1]
    if not 0 <= valid_blocks <= block_residual.shape[1]:
        raise ValueError(
            f"valid_blocks={valid_blocks} is outside fixed buffer depth "
            f"{block_residual.shape[1]}"
        )
    values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    values_float = values.float()
    score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
    weighted_keys = torch_npu.npu_rms_norm(values_float, score_weight, norm.variance_epsilon)[0]
    scores = weighted_keys.sum(dim=-1)
    max_blocks = block_residual.shape[1]
    valid_mask = torch.arange(max_blocks, device=values.device) < valid_blocks
    valid_mask = torch.cat(
        (valid_mask, torch.ones(1, dtype=torch.bool, device=values.device))
    )
    scores = scores.masked_fill(~valid_mask.unsqueeze(0), float("-inf"))
    probabilities = scores.softmax(dim=-1).unsqueeze(1)
    return torch.matmul(probabilities, values_float).squeeze(1).to(prefix_sum.dtype)


class AttnResPhase1Stats(NamedTuple):
    """Historical statistics for all slots in one K3 AttnRes block."""

    inter_numerator: torch.Tensor
    inter_max: torch.Tensor
    inter_exp_sum: torch.Tensor


class AttnResPhase2Slot(NamedTuple):
    """Query and historical statistics selected for one AttnRes slot."""

    effective_query: torch.Tensor
    inter_numerator: torch.Tensor
    inter_max: torch.Tensor
    inter_exp_sum: torch.Tensor


def _prepare_attn_res_phase1(
    block_residual: torch.Tensor,
    effective_queries: torch.Tensor,
    valid_blocks: torch.Tensor,
    epsilon: torch.Tensor,
) -> AttnResPhase1Stats:
    """Prepare FP32 Online Softmax statistics for every slot in one block."""
    values_float = block_residual.float()
    inv_rms = torch.rsqrt(values_float.square().mean(dim=-1) + epsilon)
    inter_logits = torch.matmul(
        values_float, effective_queries.transpose(0, 1)
    ).permute(2, 0, 1) * inv_rms.unsqueeze(0)
    valid_mask = (
        torch.arange(block_residual.shape[1], device=block_residual.device)
        < valid_blocks
    )
    inter_logits = inter_logits.masked_fill(
        ~valid_mask.view(1, 1, -1), float("-inf")
    )
    inter_max = inter_logits.max(dim=2).values
    inter_exp = torch.exp(inter_logits - inter_max.unsqueeze(2))
    inter_exp_sum = inter_exp.sum(dim=2)
    inter_numerator = torch.matmul(
        inter_exp.permute(1, 0, 2), values_float
    ).permute(1, 0, 2)
    return AttnResPhase1Stats(
        inter_numerator=inter_numerator,
        inter_max=inter_max,
        inter_exp_sum=inter_exp_sum,
    )


def _update_attn_res_phase2(
    partial_block: torch.Tensor,
    partial_delta: torch.Tensor,
    slot: AttnResPhase2Slot,
    epsilon: torch.Tensor,
) -> torch.Tensor:
    """Update partial in place, then merge one selected slot with Online Softmax."""
    partial_updated = (partial_block.float() + partial_delta.float()).to(
        partial_block.dtype
    )
    partial_block.copy_(partial_updated)
    partial_float = partial_block.float()
    input_logit = (
        torch.matmul(partial_float, slot.effective_query)
        * torch.rsqrt(partial_float.square().mean(dim=-1) + epsilon)
    )

    merged_max = torch.maximum(slot.inter_max, input_logit)
    inter_scale = torch.exp(slot.inter_max - merged_max)
    input_scale = torch.exp(input_logit - merged_max)
    merged_exp_sum = inter_scale * slot.inter_exp_sum + input_scale
    merged_numerator = (
        inter_scale.unsqueeze(-1) * slot.inter_numerator
        + input_scale.unsqueeze(-1) * partial_float
    )
    return (
        merged_numerator / merged_exp_sum.unsqueeze(-1)
    ).to(partial_block.dtype)


class KimiDecoderLayer(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        layer_idx: int,
        infer_config: Optional[InferenceConfig] = None,
        comm_manager: Optional[CommManager] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        parallel = None if infer_config is None else infer_config.parallel_config
        self.is_linear_attn = config.is_kda_layer(layer_idx)
        self.self_attn = (
            KimiDeltaAttention(
                config, layer_idx, infer_config, comm_manager,
                prefix=f"{prefix}.self_attn",
            )
            if self.is_linear_attn
            else KimiMLAAttention(
                config, layer_idx, infer_config, comm_manager,
                prefix=f"{prefix}.self_attn",
            )
        )
        if (
            config.num_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % config.moe_layer_freq == 0
        ):
            self.block_sparse_moe = KimiSparseMoeBlock(
                config, infer_config, comm_manager,
                prefix=f"{prefix}.block_sparse_moe",
            )
        else:
            dense_tp_size, dense_tp_rank, dense_tp_group = _dense_tp(parallel, comm_manager)
            self.mlp = KimiMLP(
                config,
                tp_size=dense_tp_size,
                tp_rank=dense_tp_rank,
                tp_group=dense_tp_group,
                prefix=f"{prefix}.mlp",
            )
        self.input_layernorm = KimiRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_attention_layernorm = KimiRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.attn_res_block_size = config.attn_res_block_size
        self.completed_blocks = (
            layer_idx + self.attn_res_block_size - 1
        ) // self.attn_res_block_size
        self.starts_new_block = layer_idx % self.attn_res_block_size == 0
        self.block_slot = layer_idx // self.attn_res_block_size
        self.self_attention_res_norm = KimiRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.mlp_res_norm = KimiRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.self_attention_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.mlp_res_proj = nn.Linear(config.hidden_size, 1, bias=False)

    def forward_attention(
            self,
            hidden_states: torch.Tensor,
            forward_metadata: ForwardMetaData = None,
            layer_cache: dict = None,
            query_start_loc: Optional[torch.Tensor] = None,
            query_boundaries: Optional[list[int]] = None,
            kv_stream: Optional[torch.npu.Stream] = None
    ) -> torch.Tensor:
        """Run the attention delta for the selected attention type."""
        normalized_states = self.input_layernorm(hidden_states)
        if self.is_linear_attn:
            return self.self_attn(
                normalized_states,
                forward_metadata,
                layer_cache,
                query_start_loc,
                query_boundaries,
            )
        mla_metadata = (
            forward_metadata
            if forward_metadata["is_prefill"]
            else forward_metadata["mla_decode_metadata"]
        )
        return self.self_attn(
            normalized_states, mla_metadata, layer_cache, kv_stream
        )

    def forward_mlp(
        self,
        hidden_states: torch.Tensor,
        forward_metadata: ForwardMetaData = None,
        moe_ctx: Optional[MoEContext] = None,
    ) -> torch.Tensor:
        """Run the original MLP/MoE delta without changing EP or SP behavior."""
        hidden_states = self.post_attention_layernorm(hidden_states)
        if hasattr(self, "block_sparse_moe"):
            return self.block_sparse_moe(
                hidden_states, forward_metadata["is_prefill"], moe_ctx
            )
        return self.mlp(hidden_states)

    def forward(
            self,
            hidden_states: torch.Tensor,
            block_residual: torch.Tensor,
            forward_metadata: ForwardMetaData = None,
            cache_data: tuple[dict, ...] = None,
            query_start_loc: Optional[torch.Tensor] = None,
            query_boundaries: Optional[list[int]] = None,
            moe_ctx: Optional[MoEContext] = None,
            kv_stream: Optional[torch.npu.Stream] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix_sum = hidden_states
        if self.completed_blocks > 0:
            hidden_states = _apply_attn_res(
                prefix_sum,
                block_residual,
                self.self_attention_res_proj,
                self.self_attention_res_norm,
                valid_blocks=self.completed_blocks,
            )
        attention_input = hidden_states
        if self.starts_new_block:
            if self.block_slot >= block_residual.shape[1]:
                raise RuntimeError("AttnRes fixed buffer is smaller than the layer table")
            block_indices = (
                    torch.arange(block_residual.shape[0], device=block_residual.device)
                    * block_residual.shape[1]
                    + self.block_slot
            )
            torch_npu.npu_scatter_nd_update_(
                block_residual.view(-1, block_residual.shape[-1]),
                block_indices.view(-1, 1),
                prefix_sum,
            )
            prefix_sum = None

        attention_output = self.forward_attention(
            hidden_states,
            forward_metadata,
            cache_data[self.layer_idx],
            query_start_loc,
            query_boundaries,
            kv_stream
        )
        prefix_sum = attention_output if prefix_sum is None else prefix_sum + attention_output
        mlp_input = _apply_attn_res(
            prefix_sum,
            block_residual,
            self.mlp_res_proj,
            self.mlp_res_norm,
            valid_blocks=self.completed_blocks + int(self.starts_new_block),
        )
        mlp_output = self.forward_mlp(mlp_input, forward_metadata, moe_ctx)
        return prefix_sum + mlp_output, block_residual, attention_input


class KimiLinearModel(nn.Module):
    def __init__(
        self,
        config: KimiLinearConfig,
        infer_config: Optional[InferenceConfig] = None,
        comm_manager: Optional[CommManager] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.attn_res_mode = (
            config.attn_res_mode
            if config.attn_res_mode in ("original", "fused")
            else "two_phase"
        )
        self.uses_dspark_draft = (
            infer_config is not None
            and infer_config.model_config.draft_model_type == "dspark"
        )
        self.dspark_target_layer_ids = ()
        logger.info("Kimi K3 AttnRes mode: %s", self.attn_res_mode)
        parallel = None if infer_config is None else infer_config.parallel_config
        self.attn_tp_size = 1 if parallel is None else parallel.attn_tp_size
        # Prefill shards packed tokens; Decode shards requests (DP-TP-DP).
        # KDA keeps its existing gather-at-entry TP path. Only MLA stays DP
        # through attention and gathers at g_proj before scattering after o_proj.
        self.attn_tp_group = (
            comm_manager.get_group("attn_tp_group") if self.attn_tp_size > 1 else None
        )
        self.attn_tp_rank = (
            comm_manager.get_rank("attn_tp_group") if self.attn_tp_size > 1 else 0
        )
        self.embed_tp_size = 1 if parallel is None else parallel.embed_tp_size
        self.embed_tp_rank = (
            comm_manager.get_rank("embed_tp_group") if self.embed_tp_size > 1 else 0
        )
        self.embed_tp_group = (
            comm_manager.get_group("embed_tp_group") if self.embed_tp_size > 1 else None
        )
        if self.embed_tp_size > 1:
            # Vocab-parallel embedding: each rank holds vocab/embed_tp rows,
            # embeds only its own id range and all_reduces to reassemble the full
            # hidden (see forward). Saves the replicated vocab*hidden table.
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                config.pad_token_id,
                torch.get_default_dtype(),
                tp_size=self.embed_tp_size,
                tp_rank=self.embed_tp_rank,
            )
        else:
            self.embed_tokens = _uninitialized(
                nn.Embedding, config.vocab_size, config.hidden_size, config.pad_token_id
            )


        enable_multi_streams =  infer_config.model_config.custom_params.get("enable_multi_streams", False)
        exe_mode = infer_config.model_config.exe_mode
        self._kv_stream = create_stream('kv', exe_mode) if enable_multi_streams else None

        # MoEContext creates the shared stream and sym_buffer used by mega_moe.
        # All MoE layers share this MoEContext for multi-stream computation of shared experts.
        # mega_moe sym_buffer: mega_moe buffer allocated immediately after communication-domain
        # registration and allocated only once during the entire inference cycle.
        self.moe_ctx = MoEContext(
            infer_config=infer_config, comm_manager=comm_manager, config=config
        )

        self.layers = nn.ModuleList([
            KimiDecoderLayer(
                config, idx, infer_config, comm_manager,
                prefix=f"{prefix}.layers.{idx}",
            )
            for idx in range(config.num_hidden_layers)
        ])
        self.max_attn_res_blocks = math.ceil(
            config.num_hidden_layers / config.attn_res_block_size
        )
        # AttnRes block residual is a resident buffer: same shape every step,
        # fully overwritten before use.  Graph mode requires such tensors to be
        # created outside the captured region with a stable object id, so it is
        # allocated once at max token count and sliced per step instead of
        # being rebuilt by new_zeros() on every forward.
        #
        # Row count is the rank-local token shard, which differs by phase:
        #   prefill: ceil(max_prefill_tokens / attn_tp_size)
        #   decode : ceil(batch_size_per_dp_rank * (next_n + 1) / attn_tp_size)
        if infer_config is not None:
            scheduler = infer_config.scheduler_config
            prefill_tokens = (
                int(scheduler.max_prefill_tokens) + self.attn_tp_size - 1
            ) // self.attn_tp_size
            decode_tokens = scheduler.batch_size_per_dp_rank * (
                infer_config.model_config.next_n + 1
            )
            decode_tokens = (
                decode_tokens + self.attn_tp_size - 1
            ) // self.attn_tp_size
            self.decode_attn_res_tokens = max(decode_tokens, 1)
            self.max_attn_res_tokens = max(
                prefill_tokens, decode_tokens, 1
            )
        else:
            self.decode_attn_res_tokens = None
            self.max_attn_res_tokens = None
        self.block_residual_buffer: Optional[torch.Tensor] = None
        self.decode_block_residual_buffer: Optional[torch.Tensor] = None
        self.register_buffer(
            "attn_res_effective_queries", None, persistent=False
        )
        self.register_buffer("attn_res_valid_blocks", None, persistent=False)
        self.register_buffer("attn_res_epsilon", None, persistent=False)
        self.output_attn_res_norm = KimiRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.output_attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.norm = KimiRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

    def init_block_residual(self, device, dtype) -> torch.Tensor:
        """Allocate the resident AttnRes buffer outside any captured graph.

        Called from the first (eager) prefill, so by the time decode is
        captured the tensor already exists with a stable object id / address.
        """
        if self.max_attn_res_tokens is None:
            raise RuntimeError(
                "AttnRes buffer needs infer_config to size max_attn_res_tokens"
            )
        self.block_residual_buffer = torch.zeros(
            self.max_attn_res_tokens,
            self.max_attn_res_blocks,
            self.config.hidden_size,
            dtype=dtype,
            device=device,
        )
        torch._dynamo.mark_static(self.block_residual_buffer)
        if self.attn_res_mode == "fused":
            self.decode_block_residual_buffer = torch.zeros(
                self.decode_attn_res_tokens,
                self.max_attn_res_blocks,
                self.config.hidden_size,
                dtype=dtype,
                device=device,
            )
            torch._dynamo.mark_static(self.decode_block_residual_buffer)
        if self.attn_res_mode != "original":
            self.attn_res_valid_blocks = torch.arange(
                1,
                self.max_attn_res_blocks + 1,
                dtype=torch.int64,
                device=device,
            )
            self.attn_res_epsilon = torch.tensor(
                self.config.rms_norm_eps,
                dtype=torch.float32,
                device=device,
            )
        return self.block_residual_buffer

    def prepare_attn_res_effective_queries(self) -> None:
        """Precompute q * RMSNorm gain once after checkpoint loading."""
        if self.attn_res_mode == "original":
            return
        first_weight = self.layers[0].self_attention_res_norm.weight
        effective_queries = torch.empty(
            2 * len(self.layers),
            self.config.hidden_size,
            dtype=torch.float32,
            device=first_weight.device,
        )
        for layer_idx, layer in enumerate(self.layers):
            effective_queries[2 * layer_idx].copy_(
                (
                    layer.self_attention_res_norm.weight.float()
                    * layer.self_attention_res_proj.weight.squeeze(0).float()
                ).detach()
            )
            effective_queries[2 * layer_idx + 1].copy_(
                (
                    layer.mlp_res_norm.weight.float()
                    * layer.mlp_res_proj.weight.squeeze(0).float()
                ).detach()
            )
        self.attn_res_effective_queries = effective_queries

    def _get_block_residual(self, tokens: int, like: torch.Tensor) -> torch.Tensor:
        target_dtype = (
            torch.float32 if self.attn_res_mode == "fused" else like.dtype
        )
        buffer = self.block_residual_buffer
        if buffer is None:
            # Only taken on the first forward (eager prefill), never inside the
            # decode graph.
            buffer = self.init_block_residual(like.device, target_dtype)
        if tokens > buffer.shape[0]:
            raise RuntimeError(
                f"AttnRes buffer holds {buffer.shape[0]} tokens but this step needs "
                f"{tokens}; raise scheduler_config.max_prefill_tokens"
            )
        block_residual = buffer[:tokens]
        if self.attn_res_mode != "fused":
            block_residual.zero_()
        return block_residual

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed the full token stream, vocab-parallel when embed_tp > 1.

        Each rank owns vocab/embed_tp rows: shift ids into its window, zero the
        out-of-range ids, embed, then all_reduce so every rank holds the full
        hidden. Runs before the prefill-SP/decode-DP split.
        """
        if self.embed_tp_size <= 1:
            return self.embed_tokens(input_ids)
        vocab_per_rank = self.config.vocab_size // self.embed_tp_size
        local_ids = input_ids - self.embed_tp_rank * vocab_per_rank
        mask = (local_ids >= 0) & (local_ids < vocab_per_rank)
        embeds = self.embed_tokens(local_ids * mask) * mask.unsqueeze(-1)
        dist.all_reduce(embeds, group=self.embed_tp_group)
        return embeds

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor] = None,
        forward_metadata: ForwardMetaData = None,
        cache_data: tuple[dict, ...] = None,
        query_start_loc: Optional[torch.Tensor] = None,
        query_boundaries: Optional[list[int]] = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be provided")
            hidden_states = self._embed(input_ids)
        else:
            hidden_states = inputs_embeds
        if self.attn_tp_size > 1:
            if forward_metadata["is_prefill"]:
                pad_len = -hidden_states.shape[0] % self.attn_tp_size
                if pad_len:
                    hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))
                    forward_metadata = _sp_pad_metadata(forward_metadata, pad_len)
            local_tokens = hidden_states.shape[0] // self.attn_tp_size
            shard_start = self.attn_tp_rank * local_tokens
            hidden_states = hidden_states[shard_start : shard_start + local_tokens]
            if forward_metadata["is_prefill"] and inputs_embeds is None:
                # The contiguous SP slice still retains the full embedding storage.
                # Materialize the shard so that storage can die before layer 0.
                hidden_states = hidden_states.clone()
        tokens = hidden_states.shape[0]
        if self.attn_res_mode == "fused" and not forward_metadata["is_prefill"]:
            block_residual = self.decode_block_residual_buffer
        else:
            block_residual = self._get_block_residual(tokens, hidden_states)
        collected_target_hidden = []
        if self.attn_res_mode != "original":
            hidden_states, collected_target_hidden = self._forward_attn_res(
                hidden_states,
                block_residual,
                forward_metadata,
                cache_data,
                query_start_loc,
                query_boundaries,
            )
        else:
            for layer_idx, layer in enumerate(self.layers):
                hidden_states, block_residual, attention_input = layer(
                    hidden_states,
                    block_residual,
                    forward_metadata,
                    cache_data,
                    query_start_loc,
                    query_boundaries,
                    self.moe_ctx,
                    self._kv_stream
                )
                target_layer_id = layer_idx - 1
                if (
                    self.uses_dspark_draft
                    and target_layer_id in self.dspark_target_layer_ids
                ):
                    collected_target_hidden.append(
                        (
                            target_layer_id,
                            attention_input,
                        )
                    )
        # AttnRes and final norm run on the rank-owned shard.
        hidden_states = _apply_attn_res(
            hidden_states,
            block_residual,
            self.output_attn_res_proj,
            self.output_attn_res_norm,
            valid_blocks=self.max_attn_res_blocks,
        )
        hidden_states = self.norm(hidden_states)
        target_hidden_states = None
        if self.uses_dspark_draft:
            if len(collected_target_hidden) != len(self.dspark_target_layer_ids):
                raise RuntimeError("not all configured DSpark target layers were collected")
            target_hidden_by_layer = dict(collected_target_hidden)
            target_hidden_states = torch.cat(
                [
                    target_hidden_by_layer[layer_id]
                    for layer_id in self.dspark_target_layer_ids
                ],
                dim=-1,
            )
        if forward_metadata["is_prefill"]:
            segment_ends = forward_metadata["segment_end_indices"]
            if self.attn_tp_size > 1:
                local_tokens = hidden_states.shape[0]
                shard_start = self.attn_tp_rank * local_tokens
                local_mask = (segment_ends >= shard_start) & (
                    segment_ends < shard_start + local_tokens
                )
                local_rows = torch.nonzero(local_mask, as_tuple=False).view(-1)
                local_indices = segment_ends[local_mask] - shard_start
                last_hidden = hidden_states.new_zeros(
                    segment_ends.shape[0], hidden_states.shape[-1]
                )
                if local_rows.numel() > 0:
                    last_hidden.index_copy_(
                        0,
                        local_rows,
                        hidden_states.index_select(0, local_indices),
                    )
                dist.all_reduce(last_hidden, group=self.attn_tp_group)
                hidden_states = last_hidden
            else:
                hidden_states = hidden_states.index_select(0, segment_ends)
        return hidden_states, target_hidden_states

    def _forward_attn_res(
        self,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        forward_metadata: ForwardMetaData,
        cache_data: tuple[dict, ...],
        query_start_loc: Optional[torch.Tensor],
        query_boundaries: Optional[list[int]],
    ) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor]]]:
        """Run AttnRes blocks and collect configured DSpark layer outputs."""
        block_size = self.config.attn_res_block_size
        collected_target_hidden = []
        for block_idx, start in enumerate(range(0, len(self.layers), block_size)):
            hidden_states, block_target_hidden = self._forward_attn_res_block(
                start,
                min(start + block_size, len(self.layers)),
                block_idx,
                hidden_states,
                block_residual,
                forward_metadata,
                cache_data,
                query_start_loc,
                query_boundaries,
            )
            collected_target_hidden.extend(block_target_hidden)
        return hidden_states, collected_target_hidden

    def _run_attn_res_phase1(
        self,
        block_residual: torch.Tensor,
        effective_queries: torch.Tensor,
        valid_blocks: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> AttnResPhase1Stats:
        if self.attn_res_mode == "two_phase":
            return _prepare_attn_res_phase1(
                block_residual,
                effective_queries,
                valid_blocks,
                epsilon,
            )
        inter_numerator, inter_max, inter_exp_sum = _block_attn_res_prepare_impl(
            block_residual,
            effective_queries,
            valid_blocks,
            eps=self.config.rms_norm_eps,
        )
        return AttnResPhase1Stats(
            inter_numerator=inter_numerator,
            inter_max=inter_max,
            inter_exp_sum=inter_exp_sum,
        )

    def _run_attn_res_phase2(
        self,
        partial_block: torch.Tensor,
        partial_delta: torch.Tensor,
        slot: AttnResPhase2Slot,
        epsilon: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.attn_res_mode == "two_phase":
            output = _update_attn_res_phase2(
                partial_block,
                partial_delta,
                slot,
                epsilon,
            )
            return output, partial_block
        return _block_attn_res_update_impl(
            partial_block,
            partial_delta,
            slot.effective_query,
            slot.inter_max,
            slot.inter_exp_sum,
            slot.inter_numerator,
            self.config.rms_norm_eps,
        )

    def _forward_attn_res_block(
        self,
        start_layer_idx: int,
        end_layer_idx: int,
        block_idx: int,
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        forward_metadata: ForwardMetaData,
        cache_data: tuple[dict, ...],
        query_start_loc: Optional[torch.Tensor],
        query_boundaries: Optional[list[int]],
    ) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor]]]:
        """Process one block with an unfused or fused two-phase backend."""
        effective_queries = self.attn_res_effective_queries
        valid_blocks_table = self.attn_res_valid_blocks
        epsilon = self.attn_res_epsilon


        block_indices = (
            torch.arange(
                block_residual.shape[0], device=block_residual.device
            )
            * block_residual.shape[1]
            + block_idx
        )
        block_update = hidden_states
        if block_update.dtype != block_residual.dtype:
            block_update = block_update.to(block_residual.dtype)
        torch_npu.npu_scatter_nd_update_(
            block_residual.view(-1, block_residual.shape[-1]),
            block_indices.view(-1, 1),
            block_update,
        )

        valid_blocks = valid_blocks_table[block_idx]
        block_layers = tuple(
            self.layers[layer_idx]
            for layer_idx in range(start_layer_idx, end_layer_idx)
        )
        block_queries = effective_queries[
            2 * start_layer_idx: 2 * end_layer_idx
        ].contiguous()
        phase1 = self._run_attn_res_phase1(
            block_residual,
            block_queries,
            valid_blocks,
            epsilon,
        )

        partial_dtype = (
            torch.float32
            if self.attn_res_mode == "fused"
            else hidden_states.dtype
        )
        partial_block = torch.zeros_like(hidden_states, dtype=partial_dtype)
        previous_mlp_delta = None
        collected_target_hidden = []
        for layer_offset, layer in enumerate(block_layers):
            attention_slot = 2 * layer_offset
            mlp_slot = attention_slot + 1
            if previous_mlp_delta is None:
                attention_input = (
                    phase1.inter_numerator[attention_slot]
                    / phase1.inter_exp_sum[attention_slot].unsqueeze(-1)
                ).to(hidden_states.dtype)
            else:
                attention_stats = AttnResPhase2Slot(
                    effective_query=block_queries[attention_slot],
                    inter_numerator=phase1.inter_numerator[attention_slot],
                    inter_max=phase1.inter_max[attention_slot],
                    inter_exp_sum=phase1.inter_exp_sum[attention_slot],
                )
                attention_input, partial_block = self._run_attn_res_phase2(
                    partial_block,
                    previous_mlp_delta.contiguous(),
                    attention_stats,
                    epsilon,
                )
                attention_input = attention_input.to(hidden_states.dtype)
            attention_output = layer.forward_attention(
                attention_input,
                forward_metadata,
                cache_data[layer.layer_idx],
                query_start_loc,
                query_boundaries,
                self._kv_stream
            )
            mlp_stats = AttnResPhase2Slot(
                effective_query=block_queries[mlp_slot],
                inter_numerator=phase1.inter_numerator[mlp_slot],
                inter_max=phase1.inter_max[mlp_slot],
                inter_exp_sum=phase1.inter_exp_sum[mlp_slot],
            )
            mlp_input, partial_block = self._run_attn_res_phase2(
                partial_block,
                attention_output.contiguous(),
                mlp_stats,
                epsilon,
            )
            mlp_input = mlp_input.to(hidden_states.dtype)
            previous_mlp_delta = layer.forward_mlp(
                mlp_input,
                forward_metadata,
                self.moe_ctx,
            )
            layer_idx = start_layer_idx + layer_offset
            target_layer_id = layer_idx - 1
            if target_layer_id in self.dspark_target_layer_ids:
                collected_target_hidden.append(
                    (
                        target_layer_id,
                        attention_input,
                    )
                )

        if previous_mlp_delta is not None:
            partial_block.add_(previous_mlp_delta)
        return partial_block.to(hidden_states.dtype), collected_target_hidden


class KimiLinearForCausalLM(nn.Module):
    """Model-local offline Kimi K3 text model."""

    def __init__(
        self,
        config: KimiLinearConfig,
        runner_settings: dict,
        prefix: str = "",
    ) -> None:
        super().__init__()
        _validate_kimi_k3_architecture(config)
        self.config = config
        self.runner_settings = runner_settings
        self.infer_config = _offline_infer_config(runner_settings)
        self.uses_dspark_draft = (
            self.infer_config.model_config.draft_model_type == "dspark"
        )
        self.comm_manager = _OfflineCommManager(runner_settings)
        self._init_parallel_comm_groups()
        # The quantization scheme is routed by module path, so every submodule
        # gets the name it carries in the checkpoint. The registered text entry
        # point uses an empty root prefix.
        self.model = KimiLinearModel(
            config, self.infer_config, self.comm_manager,
            prefix=f"{prefix}.model" if prefix else "model",
        )
        parallel = self.infer_config.parallel_config
        self.lmhead_tp_size = parallel.lmhead_tp_size
        self.lmhead_tp_rank = (
            self.comm_manager.get_rank("lmhead_tp_group") if self.lmhead_tp_size > 1 else 0
        )
        self.lmhead_tp_group = (
            self.comm_manager.get_group("lmhead_tp_group") if self.lmhead_tp_size > 1 else None
        )
        if self.lmhead_tp_size > 1:
            # Vocab-parallel head: each rank produces vocab/lmhead_tp logits; the
            # forward all_gathers them to the full vocab. Saves the replicated
            # hidden*vocab head matrix.
            self.lm_head = ColumnParallelLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                tp_size=self.lmhead_tp_size,
                tp_rank=self.lmhead_tp_rank,
                params_dtype=torch.get_default_dtype(),
            )
        else:
            self.lm_head = _uninitialized(
                nn.Linear, config.hidden_size, config.vocab_size, bias=False
            )
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_token
        self.mxfp4_experts = _mxfp4_expert_quantization(config)
        self.block_size = self.infer_config.scheduler_config.block_size
        self.attn_metadata = AttnMetaData(config, runner_settings)
        self.exe_mode = self.infer_config.model_config.exe_mode
        self.temperature = self.infer_config.data_config.temperature
        self._bound_cache_data: Optional[tuple[dict, ...]] = None

    def bind_cache_data(self, cache_data: tuple[dict, ...]) -> None:
        """Bind mutable inference state outside the compiled user-input tree."""
        if len(cache_data) != self.config.num_hidden_layers:
            raise ValueError(
                f"cache_data must contain {self.config.num_hidden_layers} layers, "
                f"got {len(cache_data)}"
            )
        if self._bound_cache_data is not None:
            mismatch = None
            for layer_idx, (bound, incoming) in enumerate(
                zip(self._bound_cache_data, cache_data)
            ):
                for name, tensor in bound.items():
                    if isinstance(tensor, torch.Tensor) and incoming.get(name) is not tensor:
                        mismatch = (layer_idx, name)
                        break
                if mismatch is not None:
                    break
            if mismatch is None:
                return
            if self.exe_mode != "eager":
                layer_idx, name = mismatch
                raise RuntimeError(
                    "cannot replace main-model cache tensors after graph capture: "
                    f"layer={layer_idx}, cache={name}"
                )
        self._bound_cache_data = cache_data

    def set_draft_config(self, draft_config) -> None:
        # Public IDs name decoder layers. GQA consumes each target layer's
        # complete hidden state, materialized as the following layer's input.
        target_layer_ids = tuple(draft_config.target_layer_ids)
        if len(target_layer_ids) != draft_config.num_target_layers:
            raise ValueError("DSpark target_layer_ids must match num_target_layers")
        if (
            not target_layer_ids
            or min(target_layer_ids) < 0
            or max(target_layer_ids) >= self.config.num_hidden_layers
        ):
            raise ValueError(
                f"DSpark target layers {target_layer_ids} are outside main model "
                f"range [0, {self.config.num_hidden_layers})"
            )
        if draft_config.target_hidden_size != self.config.hidden_size:
            raise ValueError("DSpark target_hidden_size must equal main hidden_size")
        if len(set(target_layer_ids)) != len(target_layer_ids):
            raise ValueError("DSpark target_layer_ids must be unique")
        self.model.dspark_target_layer_ids = target_layer_ids

    def _init_parallel_comm_groups(self) -> None:
        parallel = self.infer_config.parallel_config
        if parallel.attn_tp_size > 1:
            self.comm_manager.register_group(
                name="attn_tp_group",
                group_num=parallel.world_size // parallel.attn_tp_size,
                group_size=parallel.attn_tp_size,
            )
        if parallel.moe_ep_size > 1:
            group_num = parallel.world_size // parallel.moe_ep_size
            self.comm_manager.register_group(
                name="moe_ep_group",
                group_num=group_num,
                group_size=parallel.moe_ep_size,
                group_stride=group_num,
            )
            self.comm_manager.register_group(
                name="megamoe_ep_group",
                group_num=group_num,
                group_size=parallel.moe_ep_size,
                group_stride=group_num,
                return_name=True,
                allow_physical_reuse=False,
            )
            # Separate group for the decode MC2 dispatch/combine ops: they need a
            # dedicated HCCL buffer and cannot physically reuse the default group.
            mc2_buffer_size = calc_moe_hccl_buffer_size(
                self.runner_settings, self.config, is_full_mesh_v2=False
            )
            self.comm_manager.register_group(
                name="moe_ep_group_mc2",
                group_num=group_num,
                group_size=parallel.moe_ep_size,
                group_stride=group_num,
                return_name=True,
                allow_physical_reuse=False,
                hccl_buffer_size=mc2_buffer_size,
                group_type=3,
            )
        if parallel.dense_tp_size > 1:
            self.comm_manager.register_group(
                name="dense_tp_group",
                group_num=parallel.world_size // parallel.dense_tp_size,
                group_size=parallel.dense_tp_size,
            )
        if parallel.embed_tp_size > 1:
            self.comm_manager.register_group(
                name="embed_tp_group",
                group_num=parallel.world_size // parallel.embed_tp_size,
                group_size=parallel.embed_tp_size,
            )
        if parallel.lmhead_tp_size > 1:
            self.comm_manager.register_group(
                name="lmhead_tp_group",
                group_num=parallel.world_size // parallel.lmhead_tp_size,
                group_size=parallel.lmhead_tp_size,
            )

    @staticmethod
    def _to_packed(tensor: torch.Tensor) -> torch.Tensor:
        """Normalize an input to the framework's packed token layout.

        The scheduler already hands over one flat token stream; a 2D input only
        appears from callers that built a rectangular batch themselves, and
        flattening it row-major reproduces the same order.
        """
        if tensor.ndim == 1:
            return tensor
        if tensor.ndim == 2:
            return tensor.view(-1)
        if tensor.ndim == 3:
            return tensor.view(-1, *tensor.shape[2:])
        raise ValueError(f"expected a packed or batched input, got {tuple(tensor.shape)}")

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        input_lens: torch.Tensor,
        kv_len: Optional[torch.Tensor],
        cache_data: tuple[dict, ...],
        is_prefill: bool,
        request_indices: Optional[torch.Tensor] = None,
        num_accepted_tokens: Optional[torch.Tensor] = None,
        first_verify: bool = False,
        active_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Build all step inputs locally without executor metadata objects."""
        self.bind_cache_data(cache_data)
        metadata = self.attn_metadata.get_attn_metadata(
            input_ids=input_ids,
            input_lens=input_lens,
            kv_len=kv_len,
            is_prefill=is_prefill,
            request_indices=request_indices,
            num_accepted_tokens=num_accepted_tokens,
            first_verify=first_verify,
            active_mask=active_mask,
        )
        return {
            "input_ids": input_ids,
            "forward_metadata": metadata,
            "query_start_loc": metadata.get("query_start_loc"),
            "query_boundaries": metadata.get("query_boundaries"),
        }

    def prefill(self, **model_inputs) -> torch.Tensor:
        return self.forward(**model_inputs)

    def decode(self, **model_inputs) -> torch.Tensor:
        return self.forward(**model_inputs)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor],
        position_ids: Optional[torch.LongTensor] = None,
        forward_metadata: ForwardMetaData = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_data: tuple[dict, ...] = None,
        query_start_loc: Optional[torch.Tensor] = None,
        query_boundaries: Optional[list[int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        is_prefill = forward_metadata["is_prefill"]
        if cache_data is None:
            cache_data = self._bound_cache_data
        if cache_data is None:
            raise RuntimeError("main-model cache data must be bound before inference")
        packed_ids = None if input_ids is None else self._to_packed(input_ids)
        if inputs_embeds is not None:
            inputs_embeds = self._to_packed(inputs_embeds)
        hidden_states, target_hidden_states = self.model(
            packed_ids,
            inputs_embeds=inputs_embeds,
            forward_metadata=forward_metadata,
            cache_data=cache_data,
            query_start_loc=query_start_loc,
            query_boundaries=query_boundaries,
        )
        prev_hidden_states = hidden_states
        # The engine samples from [requests, steps, vocab] (execution_engine
        # slices logits[:, -1:, :] on prefill), so the packed layout stops at
        # this boundary. One step per request either way: prefill just reduced
        # to its last token, and decode carries one token per request.
        batch_size = (
            forward_metadata["actual_seq_lengths_q"].shape[0]
            if is_prefill
            else hidden_states.shape[0]
            // (self.infer_config.model_config.next_n + 1)
        )
        hidden_states = hidden_states.view(batch_size, -1, hidden_states.shape[-1])
        if not is_prefill:
            hidden_states = all_gather_first_dim(
                hidden_states, self.lmhead_tp_group, self.lmhead_tp_size
            )
        logits = self.lm_head(hidden_states)
        if self.temperature <= 0:
            token_ids = distributed_argmax(
                logits,
                self.lmhead_tp_group,
                self.lmhead_tp_rank,
                self.lmhead_tp_size,
                owner_local=not is_prefill,
            ).unsqueeze(-1)
            if self.uses_dspark_draft:
                return token_ids, {
                    "prev_hidden_states": prev_hidden_states,
                    "target_hidden_states": target_hidden_states,
                }
            return token_ids
        if self.lmhead_tp_size > 1 and is_prefill:
            # ColumnParallelLinear gives this rank vocab/lmhead_tp logits; gather
            # the shards across the group and concat back to the full vocab.
            gathered = [torch.empty_like(logits) for _ in range(self.lmhead_tp_size)]
            dist.all_gather(gathered, logits.contiguous(), group=self.lmhead_tp_group)
            logits = torch.cat(gathered, dim=-1)
        elif self.lmhead_tp_size > 1:
            logits = vocab_tp_to_owner(
                logits, self.lmhead_tp_group, self.lmhead_tp_size
            )
        if self.uses_dspark_draft:
            return logits, {
                "prev_hidden_states": prev_hidden_states,
                "target_hidden_states": target_hidden_states,
            }
        return logits

    def main_decode(self, **model_inputs):
        return self.forward(**model_inputs)

    # dt_bias is sharded across attn_tp by its flattened head-major dimension.
    # A_log needs separate handling because the checkpoint appends 32 padding
    # entries after the logical heads.
    _ATTN_TP_SHARD_DIM = {
        "self_attn.dt_bias": 0,
    }

    # Checkpoint keeps gate and up unfused; the merged projection takes them as
    # two shards of one weight, which is what its weight_loader indexes by.
    _GATE_UP_SHARD_ID = {"gate_proj": 0, "up_proj": 1}
    # The checkpoint also keeps the KDA projections and convolutions unfused;
    # these name the shard each one occupies in the fused parameter.
    _KDA_QKV_SHARD = {"q_proj": "q", "k_proj": "k", "v_proj": "v"}
    _KDA_CONV_SHARD = {"q_conv1d": 0, "k_conv1d": 1, "v_conv1d": 2}

    # A checkpoint fragment is the last _EXPERT_FRAGMENT_DEPTH dot-separated
    # components of an expert tensor name, e.g. "experts.5.w1.weight_packed".
    _EXPERT_FRAGMENT_DEPTH = 4

    def _expert_param_mapping(self) -> dict[str, tuple[str, int, str]]:
        """checkpoint fragment -> (param suffix, expert id, shard id).

        K3 names its expert projections w1/w2/w3 and, being MXFP4, stores them
        as weight_packed plus weight_scale rather than a single weight. The
        packing itself is what FusedMoEGMM.weight_loader already handles.

        Keyed by fragment rather than scanned: the real checkpoint has 896
        experts, so a list would be 5376 entries scanned once per tensor across
        497220 tensors.
        """
        suffixes = ("weight_packed", "weight_scale") if self.mxfp4_experts else ("weight",)
        mapping = {}
        for expert_id in range(self.num_experts):
            for shard_id, target in (("w1", "w13"), ("w3", "w13"), ("w2", "w2")):
                for suffix in suffixes:
                    # weight_packed feeds w13_weight / w2_weight; weight_scale
                    # feeds w13_weight_scale / w2_weight_scale.
                    param_suffix = (
                        "weight_scale" if suffix.endswith("scale") else "weight"
                    )
                    fragment = f"experts.{expert_id}.{shard_id}.{suffix}"
                    if fragment.count(".") + 1 != self._EXPERT_FRAGMENT_DEPTH:
                        raise RuntimeError(
                            f"expert fragment {fragment!r} is not "
                            f"{self._EXPERT_FRAGMENT_DEPTH} components deep"
                        )
                    mapping[fragment] = (
                        f"experts.{target}_{param_suffix}",
                        expert_id,
                        shard_id,
                    )
        return mapping

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        expert_mapping = self._expert_param_mapping()
        tp_size = self.infer_config.parallel_config.attn_tp_size
        tp_rank = (
            0 if tp_size == 1 else self.comm_manager.get_rank("attn_tp_group")
        )
        fused_qkv_loaded: dict[str, set[str]] = {}
        fused_conv_loaded: dict[str, set[int]] = {}

        def store(param_name: str, tensor: torch.Tensor) -> None:
            param = params[param_name]
            if param.shape != tensor.shape:
                raise ValueError(
                    f"{param_name}: checkpoint gives {tuple(tensor.shape)}, "
                    f"parameter is {tuple(param.shape)}"
                )
            param.data.copy_(tensor.to(dtype=param.dtype))
            loaded.add(param_name)

        for name, tensor in weights:
            # The registered kimi_k3 path is text-only. Release checkpoints can
            # still contain vision tower/projector tensors, which intentionally
            # have no parameter in this model.
            if name.startswith(("vision_tower.", "mm_projector.")):
                continue
            for source_prefix in ("model.language_model.", "language_model."):
                if name.startswith(source_prefix):
                    name = name[len(source_prefix) :]
                    break

            # Routed experts: packed into w13/w2 by the shared loader, which
            # also drops experts owned by other EP ranks.
            parts = name.rsplit(".", self._EXPERT_FRAGMENT_DEPTH)
            fragment = parts[-1] if len(parts) == 1 else ".".join(
                parts[-self._EXPERT_FRAGMENT_DEPTH:]
            )
            expert_entry = expert_mapping.get(fragment)
            if expert_entry is not None:
                param_target, expert_id, shard_id = expert_entry
                param_name = name[: -len(fragment)] + param_target
                if param_name not in params:
                    raise ValueError(
                        f"{name} maps to {param_name}, which is not a parameter"
                    )
                param = params[param_name]
                param.weight_loader(
                    param, tensor, name, shard_id=shard_id, expert_id=expert_id
                )
                loaded.add(param_name)
                continue

            # Dense MLP and shared experts: fold gate/up into gate_up_proj.
            gate_up = re.match(r"(.*)\.(gate_proj|up_proj)\.weight$", name)
            if gate_up is not None:
                param_name = f"{gate_up.group(1)}.gate_up_proj.weight"
                if param_name in params:
                    param = params[param_name]
                    param.weight_loader(
                        param, tensor, self._GATE_UP_SHARD_ID[gate_up.group(2)]
                    )
                    loaded.add(param_name)
                    continue

            qkv_proj = re.match(r"(.*)\.(q_proj|k_proj|v_proj)\.weight$", name)
            if qkv_proj is not None:
                param_name = f"{qkv_proj.group(1)}.qkv_proj.weight"
                if param_name in params:
                    param = params[param_name]
                    shard_id = self._KDA_QKV_SHARD[qkv_proj.group(2)]
                    param.weight_loader(param, tensor, shard_id)
                    shards = fused_qkv_loaded.setdefault(param_name, set())
                    shards.add(shard_id)
                    if shards == set(self._KDA_QKV_SHARD.values()):
                        loaded.add(param_name)
                    continue

            qkv_conv = re.match(r"(.*)\.(q_conv1d|k_conv1d|v_conv1d)\.weight$", name)
            if qkv_conv is not None:
                param_name = f"{qkv_conv.group(1)}.qkv_conv1d.weight"
                if param_name in params:
                    param = params[param_name]
                    local_width = param.shape[0] // 3
                    shard_index = self._KDA_CONV_SHARD[qkv_conv.group(2)]
                    if tp_size > 1:
                        if tensor.shape[0] % tp_size:
                            raise ValueError(
                                f"{name}: dim 0 of size {tensor.shape[0]} is "
                                f"not divisible by attn_tp_size={tp_size}"
                            )
                        tensor = tensor.narrow(0, tp_rank * local_width, local_width)
                    if tensor.shape[0] != local_width:
                        raise ValueError(
                            f"{name}: expected {local_width} rows for shard "
                            f"{qkv_conv.group(2)} of {param_name}, got "
                            f"{tensor.shape[0]}"
                        )
                    start = shard_index * local_width
                    param.data[start : start + local_width].copy_(
                        tensor.to(dtype=param.dtype)
                    )
                    shards = fused_conv_loaded.setdefault(param_name, set())
                    shards.add(shard_index)
                    if shards == set(self._KDA_CONV_SHARD.values()):
                        loaded.add(param_name)
                    continue

            if name not in params:
                raise ValueError(f"checkpoint tensor has no parameter: {name}")

            if name.endswith("self_attn.A_log"):
                num_heads = self.config.linear_attn_config["num_heads"]
                local_heads = num_heads // tp_size
                tensor = tensor.narrow(
                    0, tp_rank * local_heads, local_heads
                )
                store(name, tensor)
                continue

            for source_suffix, decode_suffix in (
                (".q_b_proj.weight", ".q_b_proj_decode.weight"),
                (".q_proj.weight", ".q_proj_decode.weight"),
                (".kv_b_proj.weight", ".kv_b_proj_decode.weight"),
            ):
                if not name.endswith(source_suffix):
                    continue
                decode_name = name[: -len(source_suffix)] + decode_suffix
                if decode_name not in params:
                    continue
                decode_param = params[decode_name]
                decode_loader = getattr(decode_param, "weight_loader", None)
                if decode_loader is None:
                    store(decode_name, tensor)
                else:
                    decode_loader(decode_param, tensor)
                    loaded.add(decode_name)
                break

            param = params[name]
            loader = getattr(param, "weight_loader", None)
            if loader is not None:
                # Every parallel layer -- projections, embedding, head -- takes
                # the whole tensor and keeps its own slice inside weight_loader.
                loader(param, tensor)
                loaded.add(name)
                continue

            shard_dim = next(
                (dim for suffix, dim in self._ATTN_TP_SHARD_DIM.items()
                 if name.endswith(suffix)),
                None,
            )
            if shard_dim is not None and tp_size > 1:
                width = tensor.shape[shard_dim] // tp_size
                if tensor.shape[shard_dim] % tp_size:
                    raise ValueError(
                        f"{name}: dim {shard_dim} of size "
                        f"{tensor.shape[shard_dim]} is not divisible by "
                        f"attn_tp_size={tp_size}"
                    )
                tensor = tensor.narrow(shard_dim, tp_rank * width, width)
            store(name, tensor)

        expected_qkv = set(self._KDA_QKV_SHARD.values())
        incomplete_qkv = {
            name: sorted(expected_qkv - shards)
            for name, shards in fused_qkv_loaded.items()
            if shards != expected_qkv
        }
        if incomplete_qkv:
            raise RuntimeError(
                f"incomplete fused KDA qkv projection shards: {incomplete_qkv}"
            )
        expected_conv = set(self._KDA_CONV_SHARD.values())
        incomplete_conv = {
            name: sorted(expected_conv - shards)
            for name, shards in fused_conv_loaded.items()
            if shards != expected_conv
        }
        if incomplete_conv:
            raise RuntimeError(
                f"incomplete fused KDA qkv convolution shards: {incomplete_conv}"
            )

        missing = sorted(set(params) - loaded)
        if missing:
            raise RuntimeError(
                f"{len(missing)} parameters were never assigned a checkpoint "
                f"tensor and would keep uninitialized memory, starting with: "
                f"{missing[:8]}"
            )
        return loaded

    def process_weights_after_loading(self) -> None:
        is_nz = self.infer_config.model_config.enable_weight_nz
        # kv_b_proj is split first and skipped in the loop below: the split
        # reads the checkpoint's [out, in] layout, which the loop would
        # transpose and cast to NZ out from under it.
        self._split_kv_b_proj()
        for module_name, module in self.named_modules():
            if "kv_b_proj" in module_name:
                continue
            if isinstance(module, KimiShortConvolution):
                module.build_conv_weight()
                continue
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None and hasattr(
                quant_method, "process_weights_after_loading"
            ):
                quant_method.process_weights_after_loading(module, is_nz=is_nz)
        self.model.prepare_attn_res_effective_queries()

    def _split_kv_b_proj(self) -> None:
        """Split Prefill-TP and Decode-DP KV-B layouts for absorbed MLA."""
        for layer in self.model.layers:
            attn = layer.self_attn
            if not hasattr(attn, "kv_b_proj"):
                continue
            for module_name, num_heads, key_attr, value_attr in (
                (
                    "kv_b_proj",
                    attn.num_heads,
                    "kv_b_proj_w_k",
                    "kv_b_proj_w_v",
                ),
                (
                    "kv_b_proj_decode",
                    attn.total_num_heads,
                    "kv_b_proj_decode_w_k",
                    "kv_b_proj_decode_w_v",
                ),
            ):
                module = getattr(attn, module_name)
                weight = module.weight.T.view(
                    attn.kv_lora_rank,
                    num_heads,
                    attn.qk_nope_head_dim + attn.v_head_dim,
                )
                w_k, w_v = weight.split(
                    [attn.qk_nope_head_dim, attn.v_head_dim], dim=-1
                )
                setattr(
                    attn,
                    key_attr,
                    nn.Parameter(w_k.permute(1, 2, 0).contiguous(), requires_grad=False),
                )
                setattr(
                    attn,
                    value_attr,
                    nn.Parameter(w_v.transpose(0, 1).contiguous(), requires_grad=False),
                )

    def check_model_settings(self) -> None:
        parallel = self.infer_config.parallel_config
        next_n = self.infer_config.model_config.next_n
        draft_model_type = self.infer_config.model_config.draft_model_type
        if draft_model_type not in ("none", "dspark"):
            raise RuntimeError(f"unsupported draft_model_type={draft_model_type!r}")
        if (draft_model_type == "none" and next_n != 0) or (
            draft_model_type == "dspark" and next_n <= 0
        ):
            raise RuntimeError(
                "next_n must be 0 without a draft model and positive for DSpark"
            )
        if parallel.moe_tp_size != 1:
            raise RuntimeError("K3 requires moe_tp_size=1")
        if parallel.shared_tp_size != 1:
            raise RuntimeError(
                "K3 sizes the shared expert with dense_tp_size; shared_tp_size must be 1"
            )
        if parallel.dense_tp_size > 1:
            shared_width = (
                0
                if self.config.num_shared_experts is None
                else self.config.moe_intermediate_size * self.config.num_shared_experts
            )
            for label, width in (
                ("intermediate_size", self.config.intermediate_size),
                ("the shared expert intermediate size", shared_width),
            ):
                if width % parallel.dense_tp_size:
                    raise RuntimeError(
                        f"{label}={width} must be divisible by "
                        f"dense_tp_size={parallel.dense_tp_size}"
                    )
        for label, size in (
            ("embed_tp_size", parallel.embed_tp_size),
            ("lmhead_tp_size", parallel.lmhead_tp_size),
        ):
            if self.config.vocab_size % size:
                raise RuntimeError(f"vocab_size must be divisible by {label}={size}")
        if self.config.num_experts % parallel.moe_ep_size:
            raise RuntimeError("num_experts must be divisible by moe_ep_size")
        block_size = self.infer_config.scheduler_config.block_size
        if block_size % _KV_CACHE_NZ_DIM:
            raise RuntimeError(
                f"the NZ latent cache needs block_size divisible by "
                f"{_KV_CACHE_NZ_DIM}, got {block_size}"
            )
        if parallel.moe_ep_size > 1 and not _mxfp4_expert_quantization(self.config):
            raise RuntimeError("MoE expert parallelism requires MXFP4 experts")


__all__ = [
    "KimiLinearForCausalLM",
    "SituAndMul",
]
