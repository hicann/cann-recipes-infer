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

import importlib.util
import logging
import math
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, NamedTuple, Optional, Tuple

# MegaMoE fused operator, enabled independently for Prefill and Decode.
import cann_ops_transformer.ops
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe

from executor.utils import (
    calc_moe_hccl_buffer_size,
    init_comm_group,
    init_comm_group_by_ranks,
)
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
    UnquantizedLinearMethod,
    VocabParallelEmbedding,
)
from module.quantization import QuantizeMethodBase
from module.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
from module.quantization.compressed_tensors.utils import should_ignore_layer
from module.quantization.mxfp4 import W4A8MxFp4MoEGMMMethod
from module.quantization.utils.quant_utils import reshape_mx_scale

from .configuration_kimi_k3 import KimiLinearConfig, mla_uses_mxfp8
from .dspark_registry import DSPARK_DRAFT_MODEL_TYPES
from .eplb import build_rank_local_eplb_topk
from .weight_loader import is_local_expert_weight
from .modules import (
    AttnMetaData,
    all_gather_first_dim,
    distributed_argmax,
    dp_to_tp_all_to_all,
    reduce_scatter_first_dim,
    validate_mega_kda_replayssm_switch,
    vocab_tp_to_owner,
)

_flash_kda_import_error = None
try:
    from ops.flash_kda import flash_kda as _flash_kda_impl
    from ops.flash_kda_metadata import flash_kda_metadata as _flash_kda_metadata_impl
except ImportError as error:
    _flash_kda_import_error = error
    _flash_kda_impl = None
    _flash_kda_metadata_impl = None

_recurrent_kda_import_error = None
try:
    from ops.fused_recurrent_kda_snapshot import (
        fused_recurrent_kda_op as _recurrent_kda_impl,
    )
except ImportError as error:
    _recurrent_kda_import_error = error
    _recurrent_kda_impl = None

_attn_res_import_error = None
try:
    from cann_ops_transformer.ops import (
        block_attn_res_prepare as _block_attn_res_prepare_impl,
        block_attn_res_update as _block_attn_res_update_impl,
    )

except ImportError as error:
    _attn_res_import_error = error
    _block_attn_res_prepare_impl = None
    _block_attn_res_update_impl = None

logger = logging.getLogger(__name__)

try:
    from ops.mega_recurrent_kda import mega_recurrent_kda as _mega_kda_impl
except ImportError:
    _mega_kda_impl = None

_mega_kda_replayssm_import_error = None
try:
    from ops.mega_recurrent_kda_replayssm import (
        mega_recurrent_kda_replayssm as _mega_kda_replayssm_impl,
    )
    from ops.commit_recurrent_kda_replayssm import (
        commit_recurrent_kda_replayssm as _commit_recurrent_kda_replayssm_impl,
    )
except ImportError as error:
    _mega_kda_replayssm_import_error = error
    _mega_kda_replayssm_impl = None
    _commit_recurrent_kda_replayssm_impl = None

try:
    from ops.block_attn_res_update_rms_norm import (
        block_attn_res_update_rms_norm_op as _block_attn_res_update_rms_norm_impl,
    )
except ImportError:
    _block_attn_res_update_rms_norm_impl = None

ForwardMetaData = dict
InferenceConfig = SimpleNamespace


def _load_situ_fusion_op() -> None:
    """Load the installed sparse SiTU Torch extension."""
    package = importlib.util.find_spec("custom_ops")
    if package is None or package.origin is None:
        raise RuntimeError("Kimi K3 requires the installed custom_ops package")
    package_dir = Path(package.origin).parent
    libraries = tuple(package_dir.glob("custom_ops_lib*.so"))
    if not libraries:
        raise RuntimeError(f"custom_ops library is missing from {package_dir}")
    torch.ops.load_library(str(libraries[0]))


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
    prefill_chunk_size = model.get("prefill_chunk_size", 0)
    max_prefill_len = (
        min(data.get("input_max_len", 128), prefill_chunk_size)
        if prefill_chunk_size > 0
        else data.get("input_max_len", 128)
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            custom_params=model.get("custom_params", {}),
            exe_mode=settings.get("exe_mode", "eager"),
             enable_static_kernel=model.get("enable_static_kernel", False),
             enable_weight_nz=model.get("enable_weight_nz", True),
            platform_version=model.get("platform_version", "950"),
            next_n=model.get("next_n", 0),
            draft_model_type=model.get("draft_model_type", "none"),
            dspark_tp_size=model.get("dspark_tp_size", 8),
            force_eplb=model.get("force_eplb", False),
            prefill_mini_batch_size=prefill_mini_batch_size,
            prefill_chunk_size=prefill_chunk_size,
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
            max_prefill_tokens=prefill_batch_size * max_prefill_len,
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
        self.default_hccl_buffer_size = int(os.environ.get("HCCL_BUFFSIZE", 200))
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
        allow_physical_reuse=True,
        **kwargs,
    ):
        if hccl_buffer_size is not None:
            self._has_explicit_hccl_buffer_size = True
        elif getattr(self, "_has_explicit_hccl_buffer_size", False):
            hccl_buffer_size = self.default_hccl_buffer_size
        if group_type not in (None, 0) or not allow_physical_reuse:
            result = None
            for group_id in range(group_num):
                start_rank = group_id * group_size if group_stride == 1 else group_id
                ranks = [start_rank + i * group_stride for i in range(group_size)]
                current = init_comm_group_by_ranks(
                    ranks,
                    global_rank=self.global_rank,
                    group_name=name,
                    hccl_buffer_size=hccl_buffer_size,
                    group_type=group_type,
                    platform_version=self.platform_version,
                    return_name=return_name,
                )
                if self.global_rank in ranks:
                    result = current
        else:
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
_FIA_FP8_CACHE_NZ_DIM = 32
_FIA_FP8_SUPPORTED_HEAD_COUNTS = frozenset(
    (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)
)

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
    lower_bound: float


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


def _super_kernel_scope_begin(name: Optional[str], enabled: bool) -> None:
    """Mark the start of a SuperKernel fusion scope (no-op when disabled)."""
    if enabled:
        torch.npu.super_kernel_scope_begin(name)


def _super_kernel_scope_end(name: Optional[str], enabled: bool) -> None:
    """Mark the end of a SuperKernel fusion scope (no-op when disabled)."""
    if enabled:
        torch.npu.super_kernel_scope_end(name)


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
    """The checkpoint's SiTU activation using the sparse op in BF16 mode."""

    def __init__(
        self,
        beta: float = 1.0,
        linear_beta: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.linear_beta = None if linear_beta is None else float(linear_beta)
        if self.beta <= 0 or self.linear_beta is None or self.linear_beta <= 0:
            raise ValueError("sparse SiTU requires positive beta and linear_beta")

    def forward(
        self,
        x: torch.Tensor,
        expert_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if expert_tokens is None:
            raise ValueError("sparse SiTU requires expert_tokens")
        return torch.ops.custom.npu_situ_and_mul_sparse(
            x,
            expert_tokens,
            beta=self.beta,
            alpha=self.linear_beta,
            high_precision=False,
        )


def _activation(config: KimiLinearConfig):
    if config.hidden_act == "situ":
        return SituAndMul(
            beta=getattr(config, "activation_situ_beta", None) or 1.0,
            linear_beta=getattr(config, "activation_situ_linear_beta", None),
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


def _fused_linear_quant_config(quant_config, prefix):
    """Resolve checkpoint shard ignore entries for a K3 fused Linear."""
    if isinstance(quant_config, CompressedTensorsConfig) and should_ignore_layer(
        layer_name=prefix,
        ignore=quant_config.ignore,
        fused_mapping=quant_config.packed_modules_mapping,
    ):
        return None
    return quant_config


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
    ) -> None:
        super().__init__()
        self.config = config
        self.tp_size = tp_size
        self.tp_group = tp_group
        hidden_size = hidden_size or config.hidden_size
        intermediate_size = intermediate_size or config.intermediate_size
        quant_config = getattr(config, "quant_config", None)
        if quant_config is not None:
            quant_config.packed_modules_mapping["gate_up_proj"] = ["gate_proj", "up_proj"]
        # Gate first, then up, matching SituAndMul's chunk order.
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            tp_size=tp_size,
            tp_rank=tp_rank,
            quant_config=_fused_linear_quant_config(quant_config, f"{prefix}.gate_up_proj"),
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
        self.situ = _activation(config)
        self.register_buffer(
            "_all_rows_expert_tokens",
            torch.tensor([torch.iinfo(torch.int64).max], dtype=torch.int64),
            persistent=False,
        )

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return all_gather_first_dim(x, self.tp_group, self.tp_size)

    def gate_up(
        self, x: torch.Tensor, limit_core_num: bool = False
    ) -> torch.Tensor:
        if limit_core_num:
            with torch_npu.npu.npugraph_ex.scope.limit_core_num(32, 1):
                return self.gate_up_proj(x)
        return self.gate_up_proj(x)

    def activation(self, gate_up: torch.Tensor) -> torch.Tensor:
        if self.situ is not None:
            return self.situ(gate_up, self._all_rows_expert_tokens)
        split = gate_up.shape[-1] // 2
        return F.silu(gate_up[..., :split]) * gate_up[..., split:]

    def project_down(self, activated: torch.Tensor) -> torch.Tensor:
        return self.down_proj(activated)

    def reduce_scatter(self, output: torch.Tensor) -> torch.Tensor:
        return reduce_scatter_first_dim(output, self.tp_group, self.tp_size)

    def forward(self, x: torch.Tensor, limit_core_num: bool = False) -> torch.Tensor:
        # Every rank holds a different token slice but the same column shards,
        # so the tokens must be whole before the projections.
        x = self.all_gather(x)
        gate_up = self.gate_up(x, limit_core_num=limit_core_num)
        activated = self.activation(gate_up)
        output = self.project_down(activated)
        # Sum the row-parallel partials and scatter the tokens back.
        return self.reduce_scatter(output)


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
    """Use BF16-mode SiTU, fused with MXFP8 quantization for MXFP4 experts."""

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

    def gmm1(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        expert_tokens: torch.Tensor,
        group_list_type: int,
        pertoken_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.quantized:
            return torch_npu.npu_grouped_matmul(
                [x],
                [layer.w13_weight],
                group_list=expert_tokens,
                group_type=0,
                group_list_type=group_list_type,
                split_item=3,
            )[0]

        # W4A8: MXFP4 weights, activations quantized to MXFP8 on the fly.
        if pertoken_scale is None:
            x, pertoken_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
        return torch_npu.npu_grouped_matmul(
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

    def activate_and_quant(
        self,
        gate_up: torch.Tensor,
        expert_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.quantized:
            return torch.ops.custom.grouped_situ_mx_quant(
                gate_up,
                expert_tokens,
                beta=self.situ.beta,
                alpha=self.situ.linear_beta,
                high_precision=False,
            )
        return self.situ(gate_up, expert_tokens), None

    def gmm2(
        self,
        layer: nn.Module,
        activated: torch.Tensor,
        expert_tokens: torch.Tensor,
        group_list_type: int,
        pertoken_scale: Optional[torch.Tensor] = None,
        final_output_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        if not self.quantized:
            return torch_npu.npu_grouped_matmul(
                [activated],
                [layer.w2_weight],
                group_list=expert_tokens,
                group_type=0,
                group_list_type=group_list_type,
                split_item=3,
            )[0]

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
        gate_up = self.gmm1(layer, x, expert_tokens, group_list_type, pertoken_scale)
        activated, activated_scale = self.activate_and_quant(gate_up, expert_tokens)
        return self.gmm2(
            layer,
            activated,
            expert_tokens,
            group_list_type,
            activated_scale,
            final_output_dtype,
        )


class KimiSituMoEGMM(FusedMoEGMM):
    """Packed local experts with the checkpoint-compatible SiTU formula."""

    def __init__(
        self,
        config: KimiLinearConfig,
        hidden_size: int,
        ep_size: int,
        ep_rank: int,
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
        self.situ = _activation(config)
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

    def gmm1(
        self,
        x: torch.Tensor,
        expert_tokens: torch.Tensor,
        group_list_type: int = 1,
        pertoken_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.quant_method.gmm1(self, x, expert_tokens, group_list_type, pertoken_scale)

    def activate_and_quant(
        self, gate_up: torch.Tensor, expert_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.quant_method.activate_and_quant(gate_up, expert_tokens)

    def gmm2(
        self,
        activated: torch.Tensor,
        expert_tokens: torch.Tensor,
        group_list_type: int = 1,
        pertoken_scale: Optional[torch.Tensor] = None,
        final_output_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        return self.quant_method.gmm2(
            self,
            activated,
            expert_tokens,
            group_list_type,
            pertoken_scale,
            final_output_dtype,
        )


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
        # Router logits accumulate in FP32 via torch.mm out_dtype in forward;
        # load_weights casts the checkpoint tensor to this bf16 parameter once.
        self.weight = nn.Parameter(
            torch.empty(
                self.num_experts, config.hidden_size, dtype=torch.bfloat16
            )
        )
        # The correction bias is also FP32 because it feeds the top-k
        # comparison, where BF16 rounding can reorder experts near a tie.
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(self.num_experts, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def router_linear(
        self, hidden_states: torch.Tensor, limit_core_num: bool = False
    ) -> torch.Tensor:
        if limit_core_num:
            # For the tuned Decode shapes, limit vec usage so MatMul selects
            # the Cube path and leaves vec resources for concurrent work.
            with torch_npu.npu.npugraph_ex.scope.limit_core_num(32, 1):
                return torch.mm(
                    hidden_states,
                    self.weight.t(),
                    out_dtype=torch.float32,
                )
        return torch.mm(
            hidden_states,
            self.weight.t(),
            out_dtype=torch.float32,
        )

    def topk(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.topk(self.router_linear(hidden_states))


class MLAContext:
    """Share streams/events across sequential MLA layers, outside ModuleList."""

    def __init__(self, infer_config: InferenceConfig) -> None:
        enable = infer_config.model_config.custom_params.get("enable_multi_streams", False)
        exe_mode = infer_config.model_config.exe_mode
        self.gate_stream = create_stream('mla_gate', exe_mode) if enable else None
        # Each MLA joins its gate work before returning. The next MLA may
        # reuse these events; metadata completion spans all layers separately.
        self.gate_events = tuple(create_event(exe_mode, enable) for _ in range(4))
        self.metadata_events = None


class ReplaySSMContext:
    """Share the ReplaySSM Commit stream across sequential KDA layers."""

    def __init__(self, infer_config: InferenceConfig) -> None:
        custom_params = infer_config.model_config.custom_params
        exe_mode = infer_config.model_config.exe_mode
        # The native Event wrappers are no-ops in GE graph mode, so keep that
        # path serial until a graph-native Commit-to-Verify dependency exists.
        self.enabled = bool(
            custom_params.get("enable_multi_streams", False)
            and custom_params.get("enable_mega_kda_replayssm", False)
            and exe_mode != "ge_graph"
        )
        self.exe_mode = exe_mode
        self.commit_stream = (
            create_stream("replayssm_commit", exe_mode) if self.enabled else None
        )
        self.commit_events = tuple(
            create_event(exe_mode, self.enabled) for _ in range(2)
        )


class MoEContext:
    """Per-model shared resources for MoE inference.

    Created once at model init; passed through the forward chain so every
    MoE block reads the same streams, buffers, and current routing indices.
    """
    __slots__ = (
        "shared_stream",
        "shared_comm_stream",
        "router_stream",
        "events",
        "mega_sym_buffer",
        "cur_topk_list",
        "decode_topk_list",
        "force_eplb",
        "attn_tp_size",
        "moe_ep_size",
        "moe_ep_rank",
        "num_experts",
        "top_k",
    )

    def __init__(
        self,
        config: KimiLinearConfig,
        infer_config: Optional[InferenceConfig] = None,
        comm_manager: Optional[CommManager] = None,
    ) -> None:
        custom_params = infer_config.model_config.custom_params
        _load_situ_fusion_op()
        enable_multi_streams = custom_params.get("enable_multi_streams", False)
        exe_mode = infer_config.model_config.exe_mode
        # All MoE layers share this stream for multi-stream computation of shared experts.
        self.shared_stream = create_stream('shared', exe_mode) if enable_multi_streams else None
        # Split Decode uses the same communication stream with or without
        # SuperKernel, keeping shared collectives outside the fusion scope.
        self.shared_comm_stream = (
            create_stream('shared_comm', exe_mode)
            if enable_multi_streams
            else None
        )
        self.router_stream = (
            create_stream('router', exe_mode) if enable_multi_streams else None
        )
        # One shared pool sized for the largest consumer so every decode path
        # reads the same tuple: split Decode uses slots 0-10,
        # prefill uses slots 0-3.
        event_count = 11
        self.events = tuple(
            create_event(exe_mode, enable_multi_streams) for _ in range(event_count)
        )
        # mega_moe sym_buffer: allocated immediately after communication-domain registration
        # and reused throughout the entire inference cycle.
        self.mega_sym_buffer = None
        self.cur_topk_list = None

        parallel = infer_config.parallel_config
        self.force_eplb = getattr(infer_config.model_config, "force_eplb", False)
        self.attn_tp_size = parallel.attn_tp_size
        self.moe_ep_size = parallel.moe_ep_size
        self.moe_ep_rank = comm_manager.get_rank("moe_ep_group") if self.moe_ep_size > 1 else 0
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        scheduler = infer_config.scheduler_config
        decode_tokens = scheduler.batch_size_per_dp_rank * (
            infer_config.model_config.next_n + 1
        )
        decode_local_tokens = (
            decode_tokens + self.attn_tp_size - 1
        ) // self.attn_tp_size
        self.decode_topk_list = None
        if self.force_eplb:
            device = torch.device("npu", torch.npu.current_device())
            self.decode_topk_list = build_rank_local_eplb_topk(
                self.moe_ep_size,
                self.moe_ep_rank,
                decode_local_tokens,
                self.top_k,
                self.num_experts,
                device,
            )

        enable_prefill_mega_moe = custom_params.get("enable_prefill_mega_moe", False)
        moe_ep_size = parallel.moe_ep_size

        if enable_prefill_mega_moe and moe_ep_size > 1:
            moe_chunk_max_len = custom_params.get(
                "moe_chunk_max_len", _DEFAULT_MOE_CHUNK_MAX_LEN
            )
            max_prefill_tokens_per_rank = (
                scheduler.max_prefill_tokens + self.attn_tp_size - 1
            ) // self.attn_tp_size
            max_prefill_tokens_per_chunk = (
                min(max_prefill_tokens_per_rank, moe_chunk_max_len // moe_ep_size)
                if moe_chunk_max_len > 0
                else max_prefill_tokens_per_rank
            )
            group = comm_manager.get_group("megamoe_ep_group")
            self.mega_sym_buffer = get_symm_buffer_for_mega_moe(
                group,
                num_experts=config.num_experts,
                num_max_tokens_per_rank=max(1, max_prefill_tokens_per_chunk),
                num_topk=config.num_experts_per_token,
                hidden=config.routed_expert_hidden_size,
                intermediate_hidden=2 * config.moe_intermediate_size,
                dispatch_quant_mode=4, # 4: MXFP quantization (A8W4)
                dispatch_quant_out_dtype=torch.float8_e4m3fn,
            )

    def prepare_eplb(self, token_count: int, is_prefill: bool, device: torch.device) -> None:
        if not self.force_eplb:
            self.cur_topk_list = None
            return
        if not is_prefill:
            self.cur_topk_list = self.decode_topk_list
            return

        local_tokens = (token_count + self.attn_tp_size - 1) // self.attn_tp_size
        self.cur_topk_list = build_rank_local_eplb_topk(
            self.moe_ep_size,
            self.moe_ep_rank,
            local_tokens,
            self.top_k,
            self.num_experts,
            device,
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
        # Kept for unique per-layer SuperKernel scope names.
        self.prefix = prefix
        parallel = None if infer_config is None else infer_config.parallel_config
        self.moe_ep_size = 1 if parallel is None else parallel.moe_ep_size
        self.moe_ep_rank = comm_manager.get_rank("moe_ep_group")
        self.moe_ep_group = comm_manager.get_group("moe_ep_group")
        self.moe_ep_group_mc2_name = comm_manager.get_group_name("moe_ep_group_mc2")
        custom_params = infer_config.model_config.custom_params
        self.enable_multi_streams = custom_params.get("enable_multi_streams", False)
        self.enable_superkernel = custom_params.get("enable_superkernel", False)
        self.force_eplb = getattr(infer_config.model_config, "force_eplb", False)
        self.exe_mode = infer_config.model_config.exe_mode
        self.moe_chunk_max_len = custom_params.get(
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
        self.enable_prefill_mega_moe = custom_params.get("enable_prefill_mega_moe", False)
        self.local_expert_count = self.num_experts // self.moe_ep_size
        self.local_expert_start = self.moe_ep_rank * self.local_expert_count
        expert_hidden = config.routed_expert_hidden_size
        quant_config = getattr(config, "quant_config", None)
        # K3 latent projections use MXFP8 only when Linear explicitly declares
        # that scheme. Otherwise keep the existing floating-point path, even
        # when a broad Linear target describes MXFP4 routed experts.
        linear_scheme = getattr(quant_config, "target_scheme_map", {}).get("Linear", {})
        weight_quant = linear_scheme.get("weights")
        input_quant = linear_scheme.get("input_activations")
        latent_quant_config = None
        if (
            weight_quant is not None
            and input_quant is not None
            and input_quant.type == "float"
            and input_quant.group_size == 32
            and quant_config.is_dynamic_group_w8a8_mxfp8(weight_quant, input_quant)
        ):
            # Pass the original config so per-layer ignore still takes effect.
            latent_quant_config = quant_config
        self.gate = KimiMoEGate(config)
        self.experts = KimiSituMoEGMM(
            config,
            hidden_size=expert_hidden,
            ep_size=self.moe_ep_size,
            ep_rank=self.moe_ep_rank,
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
            )
        self.routed_expert_down_proj = ReplicatedLinear(
            config.hidden_size,
            expert_hidden,
            bias=False,
            quant_config=latent_quant_config,
            prefix=f"{prefix}.routed_expert_down_proj",
        )
        self.routed_expert_norm = KimiRMSNorm(expert_hidden, config.rms_norm_eps)
        self.routed_expert_up_proj = ReplicatedLinear(
            expert_hidden,
            config.hidden_size,
            bias=False,
            quant_config=latent_quant_config,
            prefix=f"{prefix}.routed_expert_up_proj",
        )

        self.top_k = config.num_experts_per_token
        self.moe_intermediate_size = config.moe_intermediate_size

    def _routed_expert_up(self, routed_output: torch.Tensor) -> torch.Tensor:
        if isinstance(self.routed_expert_up_proj.quant_method, UnquantizedLinearMethod):
            routed_output = self.routed_expert_norm(routed_output)
            return self.routed_expert_up_proj(routed_output)

        gamma = self.routed_expert_norm.weight.to(dtype=routed_output.dtype)
        routed_output, routed_scale, _ = torch_npu.npu_rms_norm_dynamic_mx_quant(
            routed_output,
            gamma,
            beta=None,
            epsilon=self.routed_expert_norm.variance_epsilon,
            round_mode="rint",
            dst_type=torch.float8_e4m3fn,
        )
        return self.routed_expert_up_proj(
            routed_output,
            dynamic_scale=routed_scale,
        )

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        is_prefill: bool = True,
        moe_ctx: Optional[MoEContext] = None,
    ) -> torch.Tensor:
        if is_prefill:
            return self.prefill(hidden_states, moe_ctx)
        return self.decode(hidden_states, moe_ctx)

    def prefill(
        self,
        hidden_states: torch.Tensor,
        moe_ctx: Optional[MoEContext] = None,
    ) -> torch.Tensor:
        shared_stream = moe_ctx.shared_stream if moe_ctx is not None else None
        events = moe_ctx.events if moe_ctx is not None else None
        switch = False # OOM while 128K 32TP, switch on later
        main_stream = torch.npu.current_stream()
        shared_output = None
        # Overlap the shared all-gather with router computation on the main stream.
        if self.shared_experts is not None:
            record_stream(switch, hidden_states, shared_stream, self.exe_mode)
            record_event(switch, events, 0, self.exe_mode)
            with npu_stream_switch(switch, shared_stream, exe_mode=self.exe_mode):
                wait_event(switch, events, 0, self.exe_mode)
                shared_states = self.shared_experts.all_gather(hidden_states)
                record_event(switch, events, 1, self.exe_mode)

        topk_idx, topk_weight = self._route(hidden_states, moe_ctx)

        if self.shared_experts is not None:
            wait_event(switch, events, 1, self.exe_mode)
            with npu_stream_switch(switch, shared_stream, exe_mode=self.exe_mode):
                wait_event(switch, events, 1, self.exe_mode)
                shared_states = self.shared_experts.gate_up(shared_states)
                shared_states = self.shared_experts.activation(shared_states)
                shared_states = self.shared_experts.project_down(shared_states)
                record_event(switch, events, 2, self.exe_mode)

            # Overlap shared reduce-scatter with routed down projection, then join before chunks.
            wait_event(switch, events, 2, self.exe_mode)
            with npu_stream_switch(switch, shared_stream, exe_mode=self.exe_mode):
                shared_output = self.shared_experts.reduce_scatter(shared_states)
                record_event(switch, events, 3, self.exe_mode)

        routed_states = self.routed_expert_down_proj(hidden_states)
        if shared_output is not None:
            wait_event(switch, events, 3, self.exe_mode)
            record_stream(switch, shared_output, main_stream, self.exe_mode)

        plan = _moe_chunk_plan(routed_states.shape[0], self.moe_ep_size, self.moe_chunk_max_len)
        routed_output = torch.empty_like(routed_states) if len(plan) > 1 else None
        offset = 0
        for chunk_len in plan:
            end = offset + chunk_len
            chunk_states = routed_states[offset:end]
            chunk_idx = topk_idx[offset:end]
            chunk_weight = topk_weight[offset:end]
            chunk_output = self.forward_prefill_chunk(chunk_states, chunk_idx, chunk_weight, moe_ctx)
            if routed_output is None:
                routed_output = chunk_output
            else:
                routed_output[offset:end] = chunk_output
            offset = end

        routed_output = self._routed_expert_up(routed_output)
        if shared_output is None:
            return routed_output
        return routed_output + shared_output

    def decode(
        self,
        hidden_states: torch.Tensor,
        moe_ctx: Optional[MoEContext] = None,
    ) -> torch.Tensor:
        """Run split Decode with independent fusion and multi-stream controls."""
        switch = self.enable_superkernel
        multi_streams = self.enable_multi_streams
        main_stream = torch.npu.current_stream()
        shared_stream = moe_ctx.shared_stream if moe_ctx is not None else None
        comm_stream = moe_ctx.shared_comm_stream if moe_ctx is not None else None
        events = moe_ctx.events if moe_ctx is not None else None
        has_shared = self.shared_experts is not None
        shared_output = None

        # AG stays outside the scope and overlaps route/down projection.
        if has_shared:
            record_stream(multi_streams, hidden_states, comm_stream, self.exe_mode)
            record_event(multi_streams, events, 0, self.exe_mode)
            with npu_stream_switch(multi_streams, comm_stream, exe_mode=self.exe_mode):
                wait_event(multi_streams, events, 0, self.exe_mode)
                shared_states = self.shared_experts.all_gather(hidden_states)
                record_stream(multi_streams, shared_states, shared_stream, self.exe_mode)
                record_event(multi_streams, events, 1, self.exe_mode)

        scope = f"{self.prefix}_1.moe"
        # Include router/gating and latent down in the same SuperKernel while
        # keeping the shared-expert AG on the dedicated communication stream.
        _super_kernel_scope_begin(scope, switch)

        topk_idx, topk_weight, routed_states = self._decode_routed_front(
            hidden_states, multi_streams, moe_ctx, router_event_base=2
        )
        ids = topk_idx.to(torch.int32)

        # Shared mm1 must wait the AG result (event 1) but may otherwise start
        # as soon as the AG completes, which can precede the routed front.  To
        # keep dispatch and shared mm1 launched together instead, event 5 fires
        # right before dispatch is issued and mm1 waits it in addition to the
        # AG, while SiTU still waits for dispatch completion (event 6).
        if has_shared:
            record_event(multi_streams, events, 5, self.exe_mode)
        (
            expand_x,
            dynamic_scale,
            expand_idx,
            expert_token_num,
            ep_recv_counts,
            tp_recv_counts,
        ) = self._dispatch_split_moe_decode(routed_states, ids)
        if has_shared:
            record_event(multi_streams, events, 6, self.exe_mode)
            with npu_stream_switch(multi_streams, shared_stream, exe_mode=self.exe_mode):
                wait_event(multi_streams, events, 1, self.exe_mode)
                wait_event(multi_streams, events, 5, self.exe_mode)
                shared_states = self.shared_experts.gate_up(
                    shared_states, limit_core_num=True
                )
                wait_event(multi_streams, events, 6, self.exe_mode)
                shared_states = self.shared_experts.activation(shared_states)
                record_event(multi_streams, events, 7, self.exe_mode)

            # GMM1 also uses AIV, so it starts only after shared SiTU completes.
            wait_event(multi_streams, events, 7, self.exe_mode)
        routed_gate_up = self.experts.gmm1(
            expand_x, expert_token_num, pertoken_scale=dynamic_scale
        )
        if has_shared:
            record_event(multi_streams, events, 8, self.exe_mode)
            with npu_stream_switch(multi_streams, shared_stream, exe_mode=self.exe_mode):
                wait_event(multi_streams, events, 8, self.exe_mode)
                shared_states = self.shared_experts.project_down(shared_states)
                record_stream(multi_streams, shared_states, comm_stream, self.exe_mode)
                record_event(multi_streams, events, 9, self.exe_mode)

        routed_activated, routed_scale = self.experts.activate_and_quant(
            routed_gate_up, expert_token_num
        )
        expert_output = self.experts.gmm2(
            routed_activated, expert_token_num, pertoken_scale=routed_scale
        )

        if has_shared:
            # Exclude shared ReduceScatter from fusion while retaining the
            # named scope around routed compute and combine.
            _super_kernel_scope_begin(None, switch)
            with npu_stream_switch(multi_streams, comm_stream, exe_mode=self.exe_mode):
                wait_event(multi_streams, events, 9, self.exe_mode)
                shared_output = self.shared_experts.reduce_scatter(shared_states)
                record_stream(multi_streams, shared_output, main_stream, self.exe_mode)
                record_event(multi_streams, events, 10, self.exe_mode)
            _super_kernel_scope_end(None, switch)

        routed_output = self._combine_split_moe_decode(
            expert_output,
            ids,
            expand_idx,
            ep_recv_counts,
            tp_recv_counts,
            topk_weight,
        )

        routed_output = self._routed_expert_up(routed_output)

        _super_kernel_scope_end(scope, switch)
        if has_shared:
            wait_event(multi_streams, events, 10, self.exe_mode)
            routed_output = routed_output + shared_output
        return routed_output

    def _route(
        self,
        hidden_states: torch.Tensor,
        moe_ctx: Optional[MoEContext],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk_idx, topk_weight = self._route_topk(
            self.gate.router_linear(hidden_states), moe_ctx
        )
        return topk_idx, topk_weight

    def _route_topk(
        self,
        logits: torch.Tensor,
        moe_ctx: Optional[MoEContext],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk_idx, topk_weight = self.gate.topk(logits)
        if self.force_eplb:
            topk_idx = moe_ctx.cur_topk_list
        return topk_idx, topk_weight

    def _decode_routed_front(
        self,
        hidden_states: torch.Tensor,
        enable_streams: bool,
        moe_ctx: Optional[MoEContext] = None,
        router_event_base: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if not enable_streams:
            topk_idx, topk_weight = self._route(hidden_states, moe_ctx)
            quantized, dynamic_scale = self._routed_expert_down_quant(hidden_states)
            routed_states = self.routed_expert_down_proj(
                quantized, dynamic_scale=dynamic_scale
            )
            return topk_idx, topk_weight, routed_states

        main_stream = torch.npu.current_stream()
        router_stream = moe_ctx.router_stream
        router_events = moe_ctx.events[router_event_base : router_event_base + 3]

        record_stream(True, hidden_states, router_stream, self.exe_mode)
        record_event(True, router_events, 0, self.exe_mode)
        with npu_stream_switch(True, router_stream, exe_mode=self.exe_mode):
            wait_event(True, router_events, 0, self.exe_mode)
            router_logits = self.gate.router_linear(
                hidden_states, limit_core_num=True
            )
            record_event(True, router_events, 1, self.exe_mode)
            topk_idx, topk_weight = self._route_topk(router_logits, moe_ctx)
            record_event(True, router_events, 2, self.exe_mode)

        quantized, dynamic_scale = self._routed_expert_down_quant(hidden_states)
        wait_event(True, router_events, 1, self.exe_mode)
        routed_states = self.routed_expert_down_proj(
            quantized, dynamic_scale=dynamic_scale
        )
        wait_event(True, router_events, 2, self.exe_mode)
        record_stream(True, topk_idx, main_stream, self.exe_mode)
        record_stream(True, topk_weight, main_stream, self.exe_mode)
        return topk_idx, topk_weight, routed_states

    def _routed_expert_down_quant(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Quantize routed input before the latent down projection."""
        if isinstance(
            self.routed_expert_down_proj.quant_method, UnquantizedLinearMethod
        ):
            return hidden_states, None
        quantized, scale = torch_npu.npu_dynamic_mx_quant(
            hidden_states, dst_type=torch.float8_e4m3fn
        )
        return quantized, scale

    def _split_moe_common_kwargs(self) -> dict:
        group_name = self.moe_ep_group_mc2_name
        return dict(
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

    def _dispatch_split_moe_decode(
        self, routed_states: torch.Tensor, expert_ids: torch.Tensor
    ):
        dispatch = torch_npu.npu_moe_distribute_dispatch_v2(
            x=routed_states,
            expert_ids=expert_ids,
            quant_mode=4,
            y_dtype=torch.float8_e4m3fn,
            **self._split_moe_common_kwargs(),
        )
        tp_recv_counts = dispatch[5] if len(dispatch) > 5 else None
        return (
            dispatch[0],
            reshape_mx_scale(dispatch[1]),
            dispatch[2],
            dispatch[3],
            dispatch[4],
            tp_recv_counts,
        )

    def _combine_split_moe_decode(
        self,
        expert_output: torch.Tensor,
        expert_ids: torch.Tensor,
        expand_idx: torch.Tensor,
        ep_recv_counts: torch.Tensor,
        tp_recv_counts: Optional[torch.Tensor],
        topk_weight: torch.Tensor,
    ) -> torch.Tensor:
        return torch_npu.npu_moe_distribute_combine_v2(
            expert_output,
            expert_ids,
            expand_idx,
            ep_recv_counts,
            topk_weight,
            tp_send_counts=tp_recv_counts,
            expand_scales=None,
            comm_quant_mode=0,
            **self._split_moe_common_kwargs(),
        )

    def forward_prefill_chunk(
        self,
        chunk_states: torch.Tensor,
        chunk_idx: torch.Tensor,
        chunk_weight: torch.Tensor,
        moe_ctx: Optional[MoEContext] = None,
    ) -> torch.Tensor:
        if self.enable_prefill_mega_moe and self.moe_ep_size > 1:
            return self._run_megamoe(chunk_states, chunk_idx, chunk_weight, moe_ctx)

        chunk_weight = chunk_weight.bfloat16()
        local_tokens = chunk_states.shape[0]
        x_q, scale = torch_npu.npu_dynamic_mx_quant(chunk_states, dst_type=torch.float8_e4m3fn)
        routing_kwargs = dict(
            expert_idx=chunk_idx.to(torch.int32),
            active_num=chunk_idx.shape[0] * chunk_idx.shape[1],
            expert_num=self.num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, self.num_experts],
            quant_mode=-1,
        )
        expanded_x, row_idx, tokens_per_expert, _ = (
            torch_npu.npu_moe_init_routing_v2(
                x_q.view(torch.bfloat16), **routing_kwargs
            )
        )
        expanded_x = expanded_x.view(x_q.dtype)
        expanded_scale, _, _, _ = torch_npu.npu_moe_init_routing_v2(
            scale.reshape(local_tokens, -1).to(torch.bfloat16),
            **routing_kwargs,
        )
        pertoken_scale = expanded_scale.to(scale.dtype).view(-1, *scale.shape[1:])
        (
            owner_counts,
            owner_x,
            owner_scale,
            input_splits,
            output_splits,
        ) = self.dispatch_double_routing(
            tokens_per_expert, expanded_x, pertoken_scale
        )
        owner_output = self.forward_expert(owner_x, owner_counts, owner_scale)
        local_output = self.forward_combine_double_routing(
            owner_output, expanded_x, input_splits, output_splits
        )
        return torch_npu.npu_moe_finalize_routing(
            local_output,
            skip1=None,
            skip2=None,
            bias=None,
            scales=chunk_weight,
            expanded_src_to_dst_row=row_idx,
            export_for_source_row=None,
            drop_pad_mode=2,
        ).to(chunk_states.dtype)



    def _run_megamoe(self, routed_states, topk_idx, topk_weight, moe_ctx):
        """Run Prefill routed experts through MegaMoE dispatch/compute/combine.

        Dispatch + GMM1 + SiTU + GMM2 + Combine are completed by a single mega_moe
        operator, replacing Prefill's double-routing pipeline.

        sym_buffer is allocated once by KimiLinearForCausalLM after communication-domain
        registration, passed through MoEContext, and reused throughout inference.
        """
        if moe_ctx is None or moe_ctx.mega_sym_buffer is None:
            raise RuntimeError("MegaMoE requires an initialized sym_buffer")
        sym_buf = moe_ctx.mega_sym_buffer
        l1 = [self.experts.w13_weight]
        # The loader stores E8M0 encodings as bytes; MegaMoE requires the
        # actual E8M0 dtype. Reinterpret without converting values or copying.
        l1_s = [self.experts.w13_weight_scale.view(torch.float8_e8m0fnu)]
        l2 = [self.experts.w2_weight]
        l2_s = [self.experts.w2_weight_scale.view(torch.float8_e8m0fnu)]
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
        """Run local experts in expert order and restore owner-token order."""
        ordered_x, ordered_scale, unsort_idx, local_counts = (
            torch_npu.npu_moe_re_routing(
                owner_x,
                owner_counts.view(self.moe_ep_size, -1),
                per_token_scales=owner_scale,
            )
        )
        ordered_output = self.experts(
            ordered_x,
            local_counts,
            group_list_type=1,
            pertoken_scale=ordered_scale,
        )
        return torch.index_select(ordered_output, 0, unsort_idx.float().argsort().int())

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
    if metadata.get("is_chunked_prefill", False):
        return metadata
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
        has_initial_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor]:
        if is_prefill:
            if has_initial_state is None:
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
        self.attn_reduce_scatter_group = (
            None
            if self.attn_tp_size == 1
            else comm_manager.get_group("attn_reduce_scatter_group")
        )
        self.attn_tp_rank = (
            0 if self.attn_tp_size == 1 else comm_manager.get_rank("attn_tp_group")
        )
        quant_config = getattr(config, "quant_config", None)
        if quant_config is not None:
            quant_config.packed_modules_mapping["qkv_proj"] = ["q_proj", "k_proj", "v_proj"]
        if _flash_kda_impl is None:
            raise ImportError(
                "KimiDeltaAttention requires ops.flash_kda and ops.flash_kda_metadata"
            ) from _flash_kda_import_error
        self.use_mega_kda = infer_config.model_config.custom_params.get("enable_mega_kda", False)
        if not isinstance(self.use_mega_kda, bool):
            raise TypeError("enable_mega_kda must be a boolean")
        self.use_mega_kda_replayssm = infer_config.model_config.custom_params.get(
            "enable_mega_kda_replayssm", False
        )
        validate_mega_kda_replayssm_switch(
            infer_config.model_config.custom_params,
            infer_config.model_config.draft_model_type,
            infer_config.model_config.next_n,
        )
        if self.use_mega_kda_replayssm and (
            _mega_kda_replayssm_impl is None
            or _commit_recurrent_kda_replayssm_impl is None
        ):
            raise ImportError(
                "enable_mega_kda_replayssm=True requires both ReplaySSM "
                "MegaKDA operator packages"
            ) from _mega_kda_replayssm_import_error
        if self.use_mega_kda and not self.use_mega_kda_replayssm and _mega_kda_impl is None:
            raise ImportError("enable_mega_kda=True but ops.mega_recurrent_kda is not available")
        if not self.use_mega_kda and _recurrent_kda_impl is None:
            raise ImportError(
                "KimiDeltaAttention requires ops.fused_recurrent_kda_snapshot"
            ) from _recurrent_kda_import_error
        # Between layers, Prefill holds a token-SP shard while Decode holds a
        # request-DP shard. Attention gathers either layout for head TP and
        # reduce-scatters the projected result back to the owning ranks.
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
            quant_config=_fused_linear_quant_config(quant_config, f"{prefix}.qkv_proj"),
            prefix=f"{prefix}.qkv_proj",
            return_bias=False,
        )
        kernel_size = linear["short_conv_kernel_size"]
        self.qkv_conv1d = KimiShortConvolution(self.qkv_projection_size, kernel_size)
        self.register_buffer("_mega_conv_weight", None, persistent=False)
        for name in ("qkv", "fa", "fb", "beta", "output_gate", "output_projection"):
            self.register_buffer(f"_mega_{name}_weight", None, persistent=False)
        for replay_weight_name in (
            "_mega_replayssm_qkv_weight",
            "_mega_replayssm_decay_a_weight",
            "_mega_replayssm_decay_b_weight",
            "_mega_replayssm_beta_weight",
            "_mega_replayssm_output_gate_weight",
            "_mega_replayssm_output_weight",
        ):
            self.register_buffer(replay_weight_name, None, persistent=False)
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
        self.gate_lower_bound = linear["gate_lower_bound"]
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

        if self.use_mega_kda:
            # Legacy snapshot MegaKDA uses the kernel's fixed output RMSNorm
            # epsilon. ReplaySSM receives config.rms_norm_eps explicitly.
            if not self.use_full_rank_gate:
                raise ValueError("MegaKDA requires a full-rank output gate")

    def prepare_mega_kda_weights(self) -> set[nn.Module]:
        """Share NZ roots with MegaKDA and transpose views with Prefill."""
        prepared = set()
        if not self.use_mega_kda:
            return prepared
        for name, linear in (
            ("qkv", self.qkv_proj),
            ("fa", self.f_a_proj),
            ("fb", self.f_b_proj),
            ("beta", self.b_proj),
            ("output_gate", self.g_proj),
            ("output_projection", self.o_proj),
        ):
            if linear.weight.dtype != torch.bfloat16:
                raise ValueError(f"MegaKDA requires BF16 {name} projection weights")
            weight = torch_npu.npu_format_cast(linear.weight.detach().contiguous(), 29)
            setattr(self, f"_mega_{name}_weight", weight)
            linear.weight.data = weight.transpose(0, 1)
            prepared.add(linear)
        return prepared

    def _state_block_ids(
        self, forward_metadata: ForwardMetaData, cache_kind: str = "KDAConv"
    ) -> torch.Tensor:
        """Resolve the speculative KDA cache blocks for this step."""
        block_table = forward_metadata["block_table"][cache_kind]
        if cache_kind == "KDARecurrent":
            return block_table
        return block_table[:, 0]

    def _prefill_flash_kda(
        self,
        inputs: KdaInputs,
        gate_params: KdaGateParams,
        initial_state: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # TND consumes the real packed tokens and existing int32 request offsets.
        # The kernel handles tail chunks; no per-request padding or launch is needed.
        q, k, v, g, b = (tensor.contiguous() for tensor in inputs)
        metadata = _flash_kda_metadata_impl(
            q, v, initial_state, layout_qkv="TND", cu_seqlens=query_start_loc,
        )
        return _flash_kda_impl(
            q, k, v, g=g, beta=b,
            scale=1.0 / math.sqrt(q.shape[-1]),
            initial_state=initial_state,
            A_log=gate_params.a_log.data,
            dt_bias=gate_params.dt_bias,
            lower_bound=gate_params.lower_bound,
            layout_qkv="TND",
            metadata=metadata,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_metadata: ForwardMetaData,
        layer_cache: dict,
        query_start_loc: Optional[torch.Tensor] = None,
        query_boundaries: Optional[list[int]] = None,
        replayssm_ctx: Optional[ReplaySSMContext] = None,
    ) -> torch.Tensor:
        if self.use_mega_kda and not forward_metadata["is_prefill"]:
            return self._decode_mega_kda(
                hidden_states, forward_metadata, layer_cache, replayssm_ctx
            )
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

    def _decode_mega_kda(
        self,
        hidden_states: torch.Tensor,
        forward_metadata: ForwardMetaData,
        layer_cache: dict,
        replayssm_ctx: Optional[ReplaySSMContext] = None,
    ) -> torch.Tensor:
        replayssm_multistream = bool(
            self.use_mega_kda_replayssm
            and replayssm_ctx is not None
            and replayssm_ctx.enabled
        )
        replayssm_stream = (
            replayssm_ctx.commit_stream if replayssm_ctx is not None else None
        )
        replayssm_events = (
            replayssm_ctx.commit_events if replayssm_ctx is not None else None
        )
        replayssm_exe_mode = (
            replayssm_ctx.exe_mode if replayssm_ctx is not None else None
        )
        num_accepted_tokens = None
        if self.use_mega_kda_replayssm:
            num_accepted_tokens = forward_metadata[
                "replay_num_accepted_tokens"
            ].to(dtype=torch.int32).contiguous()
            record_stream(
                replayssm_multistream,
                num_accepted_tokens,
                replayssm_stream,
                replayssm_exe_mode,
            )
            record_event(
                replayssm_multistream,
                replayssm_events,
                0,
                replayssm_exe_mode,
            )
            with npu_stream_switch(
                replayssm_multistream,
                replayssm_stream,
                exe_mode=replayssm_exe_mode,
            ):
                wait_event(
                    replayssm_multistream,
                    replayssm_events,
                    0,
                    replayssm_exe_mode,
                )
                _commit_recurrent_kda_replayssm_impl(
                    layer_cache["recurrent_state"],
                    layer_cache["replay_u"],
                    layer_cache["replay_k"],
                    layer_cache["replay_decay"],
                    num_accepted_tokens,
                )
                record_event(
                    replayssm_multistream,
                    replayssm_events,
                    1,
                    replayssm_exe_mode,
                )

        hidden_states = all_gather_first_dim(
            hidden_states, self.attn_tp_group, self.attn_tp_size
        )
        batch = forward_metadata["actual_seq_lengths_q"].shape[0]
        hidden_states = hidden_states.view(batch, -1, hidden_states.shape[-1])
        if self.use_mega_kda_replayssm:
            wait_event(
                replayssm_multistream,
                replayssm_events,
                1,
                replayssm_exe_mode,
            )
            output = _mega_kda_replayssm_impl(
                hidden_states,
                self._mega_replayssm_qkv_weight,
                self._mega_replayssm_decay_a_weight,
                self._mega_replayssm_decay_b_weight,
                self._mega_replayssm_beta_weight,
                self._mega_replayssm_output_gate_weight,
                self.o_norm.weight.detach(),
                self._mega_replayssm_output_weight,
                self._mega_conv_weight,
                layer_cache["conv_state"],
                layer_cache["recurrent_state"],
                layer_cache["replay_u"],
                layer_cache["replay_k"],
                layer_cache["replay_decay"],
                self._state_block_ids(forward_metadata, "KDAConv"),
                forward_metadata["conv_num_accepted_tokens"],
                self.A_log.detach(),
                self.dt_bias.view(self.num_heads, self.head_dim),
                self.head_dim**-0.5,
                float(self.gate_lower_bound),
                float(self.o_norm.variance_epsilon),
            )
        else:
            output = _mega_kda_impl(
                hidden_states,
                self._mega_qkv_weight,
                self._mega_fa_weight,
                self._mega_fb_weight,
                self._mega_beta_weight,
                self._mega_output_gate_weight,
                self.o_norm.weight.detach(),
                self._mega_output_projection_weight,
                self._mega_conv_weight,
                layer_cache["conv_state"],
                layer_cache["recurrent_state"],
                self._state_block_ids(forward_metadata, "KDAConv"),
                forward_metadata["ssm_state_indices"],
                forward_metadata["conv_num_accepted_tokens"],
                forward_metadata["num_accepted_tokens"],
                self.A_log.detach(),
                self.dt_bias.view(self.num_heads, self.head_dim),
                self.head_dim**-0.5,
                self.gate_lower_bound,
                rms_norm_eps=1e-6,
            )
        return reduce_scatter_first_dim(
            output.view(-1, hidden_states.shape[-1]),
            self.attn_reduce_scatter_group,
            self.attn_tp_size,
        )

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
            forward_metadata.get("prefill_has_initial_state"),
        )


        shape = (*input_states.shape[:-1], self.num_heads, self.head_dim)
        raw_decay = self.f_b_proj(self.f_a_proj(input_states)).view(shape)
        raw_beta = self.b_proj(input_states)
        dt_bias = self.dt_bias.view(self.num_heads, self.head_dim)

        if forward_metadata["is_prefill"]:
            q, k, v = mixqkv.split(self.projection_size, dim=-1)
            q, k, v = q.view(shape), k.view(shape), v.view(shape)
            recurrent_state_ids = self._state_block_ids(
                forward_metadata, "KDARecurrent"
            )[:, 0]
            if forward_metadata.get("is_chunked_prefill", False):
                initial_state = layer_cache["recurrent_state"].index_select(
                    0, recurrent_state_ids.to(torch.long)
                )
            else:
                initial_state = torch.zeros(
                    len(query_boundaries) - 1, self.num_heads,
                    self.head_dim, self.head_dim,
                    dtype=torch.float32, device=q.device,
                )
            output, state = self._prefill_flash_kda(
                KdaInputs(q, k, v, raw_decay, raw_beta),
                KdaGateParams(self.A_log, dt_bias, self.gate_lower_bound),
                initial_state,
                query_start_loc,
            )
            self.update_mamba_cache(
                recurrent_state_ids, state, layer_cache["recurrent_state"]
            )
            output = _pad_kda_output(output, sp_pad_len)
        else:
            output = self._decode_fused_kda(
                mixqkv,
                raw_decay,
                raw_beta,
                KdaGateParams(self.A_log, dt_bias, self.gate_lower_bound),
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

    def prepare_mega_replayssm_weights(self) -> None:
        """Prepare the same KDA weights in ReplaySSM's required ABI format.

        ``prepare_mega_kda_weights`` has already converted the six checkpoint
        weights to FRACTAL_NZ allocation roots and made the Prefill Linear
        parameters transpose views of those roots. Reusing the roots here is
        required: transposing a FRACTAL_NZ Linear view and casting it again
        leaves a view of FRACTAL_NZ storage, which the ReplaySSM graph ABI
        rejects.
        """
        if not self.use_mega_kda_replayssm:
            return

        roots = (
            (
                "qkv",
                self._mega_qkv_weight,
                (self.qkv_projection_size, self.qkv_proj.input_size),
            ),
            (
                "decay-a",
                self._mega_fa_weight,
                (self.head_dim, self.f_a_proj.input_size),
            ),
            (
                "decay-b",
                self._mega_fb_weight,
                (self.projection_size, self.head_dim),
            ),
            (
                "beta",
                self._mega_beta_weight,
                (self.num_heads, self.b_proj.input_size),
            ),
            (
                "output-gate",
                self._mega_output_gate_weight,
                (self.projection_size, self.g_proj.input_size),
            ),
            (
                "output",
                self._mega_output_projection_weight,
                (self.o_proj.output_size, self.projection_size),
            ),
        )
        replay_names = (
            "_mega_replayssm_qkv_weight",
            "_mega_replayssm_decay_a_weight",
            "_mega_replayssm_decay_b_weight",
            "_mega_replayssm_beta_weight",
            "_mega_replayssm_output_gate_weight",
            "_mega_replayssm_output_weight",
        )
        for (label, root, expected_shape), replay_name in zip(roots, replay_names):
            if root is None:
                raise RuntimeError(
                    f"MegaKDA ReplaySSM {label} weight root was not prepared"
                )
            if tuple(root.shape) != expected_shape:
                raise RuntimeError(
                    f"MegaKDA ReplaySSM {label} weight has shape "
                    f"{tuple(root.shape)}, expected {expected_shape}"
                )
            fmt = torch_npu.get_npu_format(root)
            fmt_id = fmt.value if hasattr(fmt, "value") else int(fmt)
            if fmt_id != 29:
                raise RuntimeError(
                    f"MegaKDA ReplaySSM {label} weight must be FRACTAL_NZ, "
                    f"got format={fmt_id}"
                )
            if root.dtype != torch.bfloat16:
                raise TypeError(
                    "MegaKDA ReplaySSM requires BF16 projection weights, got "
                    f"{root.dtype} for {label}"
                )
            setattr(self, replay_name, root)

    def _decode_fused_kda(
        self,
        mixqkv: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
        gate_params: KdaGateParams,
        forward_metadata: ForwardMetaData,
        recurrent_state_cache: torch.Tensor,
    ) -> torch.Tensor:
        # Decode fused_recurrent_kda consumes the ShortConv mixed QKV directly
        # and fuses Q/K split + L2 norm + gate/beta activation + recurrence.
        # State updates recurrent_state_cache in-place, returns out [B, S, H, D] bf16.
        mixqkv = mixqkv.contiguous()
        raw_gate = raw_gate.contiguous()
        scale = 1 / math.sqrt(raw_gate.shape[-1])
        # The packaged Snapshot ABI uses int32 indices and accepted lengths.
        ssm_state_indices = forward_metadata["ssm_state_indices"]
        b = raw_beta.unsqueeze(-1).contiguous()
        num_accepted_tokens = forward_metadata.get("num_accepted_tokens")
        if num_accepted_tokens is not None:
            num_accepted_tokens = num_accepted_tokens.to(torch.int32)
        out = _recurrent_kda_impl(
            mixqkv,
            state=recurrent_state_cache,
            beta=b,
            g=raw_gate,
            scale=scale,
            A_log=gate_params.a_log.data,
            dt_bias=gate_params.dt_bias,
            lower_bound=gate_params.lower_bound,
            layout_qkv="BSND",
            ssm_state_indices=ssm_state_indices.contiguous(),
            num_accepted_tokens=num_accepted_tokens,
        )
        return out

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
        output = self.o_norm(output) * torch.sigmoid(gate)
        output = self.o_proj(output.view(output.shape[0], -1))
        return reduce_scatter_first_dim(
            output, self.attn_reduce_scatter_group, self.attn_tp_size
        )


def view_bf16_nz(weight):
    """Expose Linear's NZ storage as [N/16, K, 16] without copying weights."""
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise ValueError("BF16 MLA Prolog requires a two-dimensional BF16 weight")
    k, output = weight.shape
    if (torch_npu.get_npu_format(weight) != 29 or k % 16 or output % 16
            or weight.storage_offset() != 0 or not weight.is_contiguous()):
        raise ValueError("BF16 MLA Prolog requires an aligned, contiguous NZ weight root")
    # NZ physical [N/16, K/16, 16, 16] is exactly [N/16, K, 16].
    # set_ retains the Storage and its NZ descriptor used by Prefill MatMul.
    return torch.empty(0, dtype=weight.dtype, device=weight.device).set_(
        weight.untyped_storage(), 0, (output // 16, k, 16), (k * 16, 16, 1),
    )


def pack_mxfp8_nz(weight):
    """Checkpoint [O, K] E4M3 -> explicit [O/32, K, 32]."""
    output, k = weight.shape
    return weight.view(torch.uint8).reshape(output // 32, 32, k).transpose(1, 2).contiguous().view(weight.dtype)


def logical_mx_scale(scale, output, k):
    """Keep the actual E8M0 bytes: one exponent for each group of 32 K."""
    return scale.view(torch.uint8).reshape(output, k // 64, 2).contiguous().view(torch.float8_e8m0fnu)


def write_fp8_cache(merged, cache, slots, descale):
    """Write both 512 latent and 64 auxiliary dimensions into FP8 PA_NZ.

    Upstream's qscale_kv is a divisor, despite its name: fp8 = value / scale.
    FA uses that same scale to dequantize. Negative padded slots are skipped.
    Prefill is eager; chunked Prefill's separate scratch cache stays BF16.
    """
    slots = slots.reshape(-1).to(torch.int64)
    valid = slots >= 0
    slots = slots[valid]
    rows = (merged[valid].float() / descale).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    block_size, width = cache.shape[1], cache.shape[-1]
    tiles = width // 32
    indices = ((slots // block_size)[:, None] * tiles * block_size
               + torch.arange(tiles, device=slots.device)[None, :] * block_size
               + (slots % block_size)[:, None]).reshape(-1)
    torch_npu.npu_scatter_nd_update_(cache.view(torch.uint8).view(-1, 32), indices.view(-1, 1), rows.reshape(-1, 32))


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
        if config.q_lora_rank is None or config.q_lora_rank <= 0:
            raise ValueError("Kimi K3 MLA requires a positive q_lora_rank")
        self.config = config
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
        self.attn_reduce_scatter_group = (
            None
            if self.attn_tp_size == 1
            else comm_manager.get_group("attn_reduce_scatter_group")
        )
        custom_params = infer_config.model_config.custom_params
        self.enable_multi_streams = custom_params.get("enable_multi_streams", False)
        self.use_w8a8c8 = mla_uses_mxfp8(config)
        self.mla_decode_tp_group = self.attn_tp_group
        if self.enable_multi_streams and self.attn_tp_size > 1:
            self.mla_decode_tp_group = comm_manager.get_group("mla_decode_tp_group")
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
        # Non-C8 MLA stores the compressed latent and auxiliary key together:
        #     kv_cache  dim = kv_lora_rank + qk_rope_head_dim (576)
        # Decode absorbs kv_b_proj into attention; ordinary Prefill expands
        # K/V on read. MXFP8 uses the same merged geometry in FP8 storage.
        #
        # K3's MLA is NoPE. The merged 576-wide Q/K representation still
        # contains the auxiliary 64-wide segment expected by Flash MLA, but
        # no rotary tensors are supplied to either operator.
        self.attn_type = "FullAttention"
        self.block_size = (
            None if infer_config is None else infer_config.scheduler_config.block_size
        )
        # Split out of kv_b_proj once the checkpoint is loaded; see
        # KimiLinearForCausalLM.process_weights_after_loading.
        self.register_buffer("kv_b_proj_w_k", None)
        self.kv_b_proj_w_v = None
        self.register_buffer("kv_b_proj_decode_w_k", None)
        self.kv_b_proj_decode_w_v = None
        self.exe_mode = None if infer_config is None else infer_config.model_config.exe_mode

    def prepare_prolog_weights(self) -> None:
        """Prepare DSL Prolog weights once, outside graph capture."""
        if self.use_w8a8c8:
            from ops.quant_mla_prolog import quant_mla_prolog_op

            self.mla_prolog_op = quant_mla_prolog_op
        else:
            from ops.mla_prolog_bf16 import mla_prolog_bf16_op

            self.mla_prolog_op = mla_prolog_bf16_op
        for name, linear in (("qa", self.q_a_proj), ("qb", self.q_b_proj_decode),
                             ("kva", self.kv_a_proj_with_mqa)):
            weight = linear.weight.detach()
            if self.use_w8a8c8:
                packed = pack_mxfp8_nz(weight)
                self.register_buffer(f"mla_s_{name}", logical_mx_scale(
                    linear.weight_scale.detach(), *weight.shape,
                ))
            else:
                packed = view_bf16_nz(weight)
            self.register_buffer(f"mla_w_{name}", packed)
        self.register_buffer("mla_gamma_qa", self.q_a_layernorm.weight.detach().float().contiguous())
        self.register_buffer("mla_gamma_kva", self.kv_a_layernorm.weight.detach().float().contiguous())
        if self.use_w8a8c8:
            self.register_buffer("mla_kv_scale", torch.tensor(
                [1.0], dtype=torch.float32, device=self.q_a_proj.weight.device,
            ))

    def _flash_mla_attention(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        q_len: int,
        attention_mask: Optional[torch.Tensor] = None,
        metadata: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run Flash MLA against the combined PA_NZ latent cache."""
        # Prolog's absorbed query uses the latent KV width (512), not the
        # regular per-head qk_nope width used by q_b_proj (for Kimi K3 that
        # width is typically 128).  Flash MLA then consumes the 64-wide rope
        # segment as part of the same 576-wide QK vector.
        merged_qk_dim = self.kv_lora_rank + self.qk_rope_head_dim
        kv_cache_nz = self._pa_nz_cache_view(kv_cache, merged_qk_dim)
        mask_mode = 0 if q_len == 1 else 3
        attn_mask = attention_mask if mask_mode == 3 else None
        if metadata is None:
            metadata = torch.ops.cann_ops_transformer.flash_mla_with_kvcache_metadata(
                cache_seqlens,
                num_heads_q=q.shape[1],
                num_heads_kv=1,
                cu_seqlens_q=cu_seqlens_q,
                seqused_q=None,
                max_seqlen_q=q_len,
                max_seqlen_kv=-1,
                head_dim_qk=576,
                head_dim_v=512,
                mask_mode=mask_mode,
                layout_q="TND",
            )
        output, _ = torch.ops.cann_ops_transformer.flash_mla_with_kvcache(
            q.contiguous(),
            kv_cache_nz,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=None,
            attn_mask=attn_mask,
            metadata=metadata,
            head_dim_v=512,
            softmax_scale=self.scaling,
            mask_mode=mask_mode,
            max_seqlen_q=-1,
            max_seqlen_kv=-1,
            layout_q="TND",
            layout_kv="PA_NZ",
            layout_out="TND",
            return_softmax_lse=False,
        )
        return output

    def _launch_gate_allgather(self, hidden_states, gate_stream, gate_events):
        """Start gate AllGather after the current main-stream work completes."""
        record_stream(
            self.enable_multi_streams, hidden_states, gate_stream, self.exe_mode
        )
        record_event(self.enable_multi_streams, gate_events, 0, self.exe_mode)
        with npu_stream_switch(
            self.enable_multi_streams, gate_stream, exe_mode=self.exe_mode
        ):
            wait_event(self.enable_multi_streams, gate_events, 0, self.exe_mode)
            full_hidden = all_gather_first_dim(
                hidden_states, self.mla_decode_tp_group, self.attn_tp_size
            )
            record_event(self.enable_multi_streams, gate_events, 3, self.exe_mode)
        return full_hidden

    def _forward_decode_flash(
        self,
        hidden_states: torch.Tensor,
        forward_metadata: ForwardMetaData,
        layer_cache: dict,
        metadata_events=None,
        gate_stream=None,
        gate_events=None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run Prolog, then overlap gate AllGather with Flash MLA attention."""
        tokens = hidden_states.shape[0]
        block_table = forward_metadata["block_table"][self.attn_type]
        cache_seqlens = forward_metadata["actual_seq_lengths_kv"]
        # Flash MLA expects the B+1 prefix-sum form with a leading zero.  The
        # legacy actual_seq_lengths_cu_q field stores only segment ends.
        cu_seqlens_q = forward_metadata["query_start_loc"]
        q_len = tokens // block_table.shape[0]
        x = hidden_states.contiguous()
        cache = layer_cache["kv_cache"]
        slots = forward_metadata["slot_mapping"][self.attn_type].reshape(-1).to(torch.int64).contiguous()
        if self.use_w8a8c8:
            x, sx = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
            sx = sx.view(torch.uint8).reshape(tokens, hidden_states.shape[-1] // 64, 2).view(torch.float8_e8m0fnu)
            outs = self.mla_prolog_op(
                x, self.mla_w_qa, self.mla_w_qb, self.mla_w_kva,
                self.kv_b_proj_decode_w_k, sx, self.mla_s_qa, self.mla_s_qb, self.mla_s_kva,
                self.mla_gamma_qa, self.mla_gamma_kva, cache, slots, self.mla_kv_scale,
                norm_eps=1e-6, quant_mode_aw=1, quant_mode_c=1,
            )
            q, q_descale = outs[0], outs[1]
        else:
            q = self.mla_prolog_op(
                x, self.mla_w_qa, self.mla_w_qb, self.mla_w_kva,
                self.kv_b_proj_decode_w_k, self.mla_gamma_qa, self.mla_gamma_kva,
                cache, slots, self.q_a_layernorm.variance_epsilon,
            )
        full_hidden = None
        if self.use_output_gate and self.enable_multi_streams:
            # Wait for Prolog on device before starting AllGather on the gate
            # stream. Flash MLA continues on the main stream without waiting
            # for AllGather, so these two operations can overlap.
            full_hidden = self._launch_gate_allgather(
                hidden_states, gate_stream, gate_events
            )
        if forward_metadata.get("flash_mla_metadata_async", False):
            wait_event(
                self.enable_multi_streams,
                metadata_events,
                1,
                self.exe_mode,
            )
        if self.use_w8a8c8:
            blocks, block_size, _, width = cache.shape
            output, _ = torch.ops.cann_ops_transformer.quant_flash_mla_with_kvcache(
                q, cache.view(blocks, 1, width // 32, block_size, 32),
                q_descale, self.mla_kv_scale, block_table, cache_seqlens, 1,
                cu_seqlens_q=cu_seqlens_q,
                # Constructed once with model inputs; shared by all MLA layers.
                seqused_q=forward_metadata["actual_seq_lengths_q"],
                metadata=forward_metadata["flash_mla_metadata"],
                head_dim_v=self.kv_lora_rank,
                attn_mask=forward_metadata.get("flash_attention_mask") if q_len > 1 else None,
                softmax_scale=self.scaling, mask_mode=3 if q_len > 1 else 0,
                max_seqlen_q=q_len,
                layout_q="TND", layout_kv="PA_NZ", layout_out="TND",
            )
        else:
            output = self._flash_mla_attention(
                q,
                cache,
                block_table,
                cache_seqlens,
                cu_seqlens_q,
                q_len,
                forward_metadata.get("flash_attention_mask"),
                forward_metadata.get("flash_mla_metadata"),
            )
        output = output.to(device=self.kv_b_proj_decode_w_v.device)
        output = torch_npu.npu_transpose_batchmatmul(
            output,
            self.kv_b_proj_decode_w_v,
            bias=None,
            scale=None,
            perm_x1=(1, 0, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
            batch_split_factor=self.attn_tp_size,
        )
        if self.attn_tp_size > 1:
            # Contiguous [TP, local_tokens, heads * v_dim / TP], ready for
            # AllToAll. Keep destination rank first; do not flatten as tokens.
            return output, full_hidden
        return output.reshape(tokens, self.total_num_heads * self.v_head_dim), full_hidden

    def _write_latent_cache(
        self,
        compressed: torch.Tensor,
        slot_mapping: torch.Tensor,
        layer_cache: dict,
        cache_prefix: str = "",
    ):
        """RMSNorm this step's latent and scatter it into the paged blocks.

        Resident caches use the writer's logical 4D PA shape. Non-C8 writes
        scatter individual NZ tiles; the C8 writer uses cache_mode="PA_NZ" to
        update the backing storage in NZ.
        """
        k_nope, k_rope = torch.split(
            compressed, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        merged_cache = layer_cache.get(f"{cache_prefix}kv_cache")
        if merged_cache is not None:
            # Non-C8 MLA keeps ckv and ckr in one PA_NZ cache.
            # Writes from the regular Prefill projection update the same
            # storage that cannbot Prolog uses during Decode.
            k_nope = self.kv_a_layernorm(k_nope)
            merged = torch.cat((k_nope, k_rope), dim=-1).contiguous()
            if self.use_w8a8c8 and not cache_prefix:
                write_fp8_cache(merged, merged_cache, slot_mapping, self.mla_kv_scale)
                return
            block_size = merged_cache.shape[1]
            tile_count = merged.shape[-1] // _KV_CACHE_NZ_DIM
            slots = slot_mapping.reshape(-1).to(torch.int64)
            block_ids = torch.div(slots, block_size, rounding_mode="floor")
            offsets = torch.remainder(slots, block_size)
            tile_indices = torch.arange(
                tile_count, dtype=torch.int64, device=slots.device
            )
            tile_indices = (
                block_ids[:, None] * tile_count * block_size
                + tile_indices[None, :] * block_size
                + offsets[:, None]
            ).reshape(-1)
            nz_cache = merged_cache.view(-1, _KV_CACHE_NZ_DIM)
            torch_npu.npu_scatter_nd_update_(
                nz_cache,
                tile_indices.view(-1, 1),
                merged.view(-1, _KV_CACHE_NZ_DIM),
            )
            return
        nope_cache = layer_cache[f"{cache_prefix}nope_cache"]
        rope_cache = layer_cache[f"{cache_prefix}rope_cache"]
        k_nope = self.kv_a_layernorm(k_nope)
        torch_npu.npu_scatter_pa_kv_cache(
            k_nope.unsqueeze(1),
            k_rope.unsqueeze(1),
            nope_cache,
            rope_cache,
            slot_mapping.reshape(-1),
            cache_mode="Norm",
        )

    @staticmethod
    def _pa_nz_cache_view(
        kv_cache: torch.Tensor, merged_dim: int
    ) -> torch.Tensor:
        """View logical Prolog PA storage as Flash MLA's physical NZ layout."""
        block_num, block_size, num_kv, cache_dim = kv_cache.shape
        return kv_cache.view(
            block_num,
            num_kv,
            cache_dim // _KV_CACHE_NZ_DIM,
            block_size,
            _KV_CACHE_NZ_DIM,
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
        if forward_metadata.get("is_chunked_prefill", False):
            self._write_latent_cache(
                compressed,
                forward_metadata["slot_mapping"]["PrefillFullAttention"],
                layer_cache,
                cache_prefix="prefill_",
            )
            return self._prefill_chunk_attention(
                query_nope, query_rope, tokens, forward_metadata, layer_cache
            )
        return self._prefill_attention(
            query_nope, query_rope, k_nope, k_rope, tokens, forward_metadata
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

    def _prefill_chunk_attention(
        self,
        query_nope: torch.Tensor,
        query_rope: torch.Tensor,
        tokens: int,
        forward_metadata: ForwardMetaData,
        layer_cache: dict,
    ) -> torch.Tensor:
        query_latent = torch_npu.npu_transpose_batchmatmul(
            query_nope,
            self.kv_b_proj_w_k,
            bias=None,
            scale=None,
            perm_x1=(1, 0, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
        ).view(tokens, self.num_heads, self.kv_lora_rank)
        nope_cache, rope_cache = self._nz_cache_inputs(layer_cache, cache_prefix="prefill_")
        output, _ = torch_npu.npu_fused_infer_attention_score_v2(
            query_latent,
            nope_cache,
            nope_cache,
            query_rope=query_rope,
            key_rope=rope_cache,
            num_query_heads=self.num_heads,
            num_key_value_heads=1,
            softmax_scale=self.scaling,
            input_layout="TND_NTD",
            sparse_mode=3,
            atten_mask=forward_metadata["attention_mask"],
            actual_seq_qlen=forward_metadata["actual_seq_lengths_cu_list_q"],
            actual_seq_kvlen=forward_metadata["actual_seq_lengths_list_kv"],
            block_table=forward_metadata["block_table"]["PrefillFullAttention"],
            block_size=self.block_size,
        )
        output = torch_npu.npu_transpose_batchmatmul(
            output[: self.num_heads],
            self.kv_b_proj_w_v,
            bias=None,
            scale=None,
            perm_x1=(0, 1, 2),
            perm_x2=(0, 1, 2),
            perm_y=(1, 0, 2),
        )
        return output.reshape(tokens, self.num_heads * self.v_head_dim)


    def _nz_cache_inputs(
        self, layer_cache: dict, cache_prefix: str = ""
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cache tensors in the layout expected by the attention path."""
        nope_cache = layer_cache[f"{cache_prefix}nope_cache"]
        rope_cache = layer_cache[f"{cache_prefix}rope_cache"]

        block_num, block_size, num_kv_heads, nope_dim = nope_cache.shape
        rope_block_num, rope_block_size, rope_num_kv_heads, rope_dim = (
            rope_cache.shape
        )
        if (
            block_num != rope_block_num
            or block_size != rope_block_size
            or num_kv_heads != 1
            or rope_num_kv_heads != 1
        ):
            raise RuntimeError(
                "resident MLA caches must share [Block, BlockSize, 1] dimensions"
            )
        nope_nz_dim = (
            _KV_CACHE_NZ_DIM
        )
        if nope_dim % nope_nz_dim or rope_dim % _KV_CACHE_NZ_DIM:
            raise RuntimeError("resident MLA cache widths must align to PA_NZ tiles")

        # Writers receive logical [Block, BlockSize, N, D]. With PA_NZ they
        # update the backing storage as [Block, D/tile, BlockSize, N, tile].
        # FIA reads that same storage as [Block, N, D/tile, BlockSize, tile].
        return (
            nope_cache.view(
                block_num,
                num_kv_heads,
                nope_dim // nope_nz_dim,
                block_size,
                nope_nz_dim,
            ),
            rope_cache.view(
                block_num,
                rope_num_kv_heads,
                rope_dim // _KV_CACHE_NZ_DIM,
                block_size,
                _KV_CACHE_NZ_DIM,
            ),
        )


    def forward(
            self,
            hidden_states: torch.Tensor,
            forward_metadata: ForwardMetaData,
            layer_cache: dict,
            mla_ctx: Optional[MLAContext] = None,
    ) -> torch.Tensor:
        is_prefill = forward_metadata["is_prefill"]
        gate_stream = mla_ctx.gate_stream if mla_ctx is not None else None
        gate_events = mla_ctx.gate_events if mla_ctx is not None else None
        chunked_prefill = forward_metadata.get("is_chunked_prefill", False)
        sp_pad_len = 0
        if is_prefill:
            hidden_states = all_gather_first_dim(
                hidden_states, self.attn_tp_group, self.attn_tp_size
            )
            if chunked_prefill:
                sp_pad_len = hidden_states.shape[0] - forward_metadata["prompt_tokens"]
                if sp_pad_len:
                    hidden_states = hidden_states[:-sp_pad_len]
        tokens = hidden_states.shape[0]

        main_stream = torch.npu.current_stream()
        gate = None
        full_hidden = None
        if is_prefill:
            normalized_q = self.q_a_layernorm(self.q_a_proj(hidden_states))
            query = self.q_b_proj(normalized_q).view(
                tokens, self.num_heads, self.q_head_dim
            )
            compressed = self.kv_a_proj_with_mqa(hidden_states)
            output = self._forward_prefill(
                query, compressed, forward_metadata, layer_cache
            )
        else:
            output, full_hidden = self._forward_decode_flash(
                hidden_states, forward_metadata, layer_cache,
                metadata_events=mla_ctx.metadata_events if mla_ctx is not None else None,
                gate_stream=gate_stream, gate_events=gate_events,
            )
            record_event(self.enable_multi_streams, gate_events, 2, self.exe_mode)

        if not is_prefill:
            if self.use_output_gate and self.enable_multi_streams:
                wait_event(
                    self.enable_multi_streams,
                    gate_events,
                    3,
                    self.exe_mode,
                )
            output = dp_to_tp_all_to_all(
                output,
                self.mla_decode_tp_group,
                self.attn_tp_size,
                forward_metadata["oproj_output_rows"],
                self.o_proj_channel_width,
                input_is_tp_packed=True,
            )

        if self.use_output_gate:
            if not is_prefill:
                if self.enable_multi_streams:
                    with npu_stream_switch(
                        self.enable_multi_streams,
                        gate_stream,
                        exe_mode=self.exe_mode,
                    ):
                        wait_event(
                            self.enable_multi_streams,
                            gate_events,
                            2,
                            self.exe_mode,
                        )
                        gate = torch.sigmoid(self.g_proj(full_hidden))
                        record_event(
                            self.enable_multi_streams,
                            gate_events,
                            1,
                            self.exe_mode,
                        )
                    wait_event(
                        self.enable_multi_streams,
                        gate_events,
                        1,
                        self.exe_mode,
                    )
                    record_stream(
                        self.enable_multi_streams, gate, main_stream, self.exe_mode
                    )
                else:
                    full_hidden = all_gather_first_dim(
                        hidden_states, self.attn_tp_group, self.attn_tp_size
                    )
                    gate = torch.sigmoid(self.g_proj(full_hidden))
            else:
                gate = torch.sigmoid(self.g_proj(hidden_states))
            output = output * gate
        output = self.o_proj(output)
        if sp_pad_len:
            output = F.pad(output, (0, 0, 0, sp_pad_len))
        return reduce_scatter_first_dim(
            output, self.attn_reduce_scatter_group, self.attn_tp_size
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
            mla_ctx: Optional[MLAContext] = None,
            replayssm_ctx: Optional[ReplaySSMContext] = None,
            input_normalized: bool = False,
    ) -> torch.Tensor:
        """Run the attention delta for the selected attention type."""
        normalized_states = hidden_states if input_normalized else self.input_layernorm(hidden_states)
        if self.is_linear_attn:
            return self.self_attn(
                normalized_states,
                forward_metadata,
                layer_cache,
                query_start_loc,
                query_boundaries,
                replayssm_ctx,
            )
        mla_metadata = (
            forward_metadata
            if forward_metadata["is_prefill"]
            else forward_metadata["mla_decode_metadata"]
        )
        return self.self_attn(
            normalized_states, mla_metadata, layer_cache, mla_ctx
        )

    def forward_mlp(
        self,
        hidden_states: torch.Tensor,
        forward_metadata: ForwardMetaData = None,
        moe_ctx: Optional[MoEContext] = None,
        input_normalized: bool = False,
    ) -> torch.Tensor:
        """Run the original MLP/MoE delta without changing EP or SP behavior."""
        if not input_normalized:
            hidden_states = self.post_attention_layernorm(hidden_states)
        if hasattr(self, "block_sparse_moe"):
            return self.block_sparse_moe(
                hidden_states, forward_metadata["is_prefill"], moe_ctx
            )
        return self.mlp(hidden_states)


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
        if _block_attn_res_prepare_impl is None:
            raise ImportError(
                "Kimi K3 requires cann_ops_transformer.ops fused AttnRes operators"
            ) from _attn_res_import_error
        self.uses_dspark_draft = (
            infer_config is not None
            and infer_config.model_config.draft_model_type
            in DSPARK_DRAFT_MODEL_TYPES
        )
        self.dspark_target_layer_ids = ()
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
        self.embed_reduce_scatter_group = None
        if self.embed_tp_size == self.attn_tp_size and self.embed_tp_size > 1:
            scatter_group = comm_manager.get_group("attn_reduce_scatter_group")
            # Equal sizes alone do not guarantee identical token ownership.
            if scatter_group is not None and (
                dist.get_process_group_ranks(self.embed_tp_group)
                == dist.get_process_group_ranks(self.attn_tp_group)
                == dist.get_process_group_ranks(scatter_group)
            ):
                self.embed_reduce_scatter_group = scatter_group
        if self.embed_tp_size > 1:
            # Vocab-parallel embedding: each rank holds vocab/embed_tp rows,
            # then sums contributions using ReduceScatter when its rank layout
            # matches attention, or AllReduce before the token split otherwise.
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


        self.mla_ctx = MLAContext(infer_config)
        self.replayssm_ctx = ReplaySSMContext(infer_config)

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
        self.register_buffer("block_residual_buffer", None, persistent=False)
        self.register_buffer("decode_block_residual_buffer", None, persistent=False)
        self.register_buffer("decode_attn_res_block_indices", None, persistent=False)
        self.register_buffer(
            "attn_res_effective_queries", None, persistent=False
        )
        self.register_buffer("attn_res_valid_blocks", None, persistent=False)
        self.output_attn_res_norm = KimiRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.output_attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False)
        self.norm = KimiRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.attn_metadata = None

    def bind_attn_metadata(self, metadata) -> None:
        """Bind request metadata so Flash MLA preparation follows embedding."""
        self.attn_metadata = metadata
        self.mla_ctx.metadata_events = metadata.metadata_events

    def init_block_residual(self, device, dtype) -> torch.Tensor:
        """Allocate the resident AttnRes buffer outside any captured graph.

        Called during post-load runtime initialization, so both Prefill and
        Decode see tensors with stable object ids and addresses.
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
        self.decode_block_residual_buffer = torch.zeros(
            self.decode_attn_res_tokens,
            self.max_attn_res_blocks,
            self.config.hidden_size,
            dtype=dtype,
            device=device,
        )
        torch._dynamo.mark_static(self.decode_block_residual_buffer)
        # Reuse contiguous [T, 1] scatter indices for fused decode.
        self.decode_attn_res_block_indices = (
            torch.arange(self.decode_attn_res_tokens, device=device).view(1, -1, 1)
            * self.max_attn_res_blocks
            + torch.arange(self.max_attn_res_blocks, device=device).view(-1, 1, 1)
        )
        torch._dynamo.mark_static(self.decode_attn_res_block_indices)
        self.attn_res_valid_blocks = torch.arange(
            1,
            self.max_attn_res_blocks + 1,
            dtype=torch.int64,
            device=device,
        ).to(torch.uint64)
        return self.block_residual_buffer

    def initialize_runtime_buffers(self) -> None:
        """Create fixed AttnRes buffers before any model forward is compiled."""
        if self.block_residual_buffer is not None:
            if (
                self.decode_block_residual_buffer is None
                or self.decode_attn_res_block_indices is None
            ):
                raise RuntimeError(
                    "fused AttnRes decode buffers are only partially initialized"
                )
            return

        reference_weight = self.embed_tokens.weight
        self.init_block_residual(reference_weight.device, torch.float32)

    def prepare_attn_res_effective_queries(self) -> None:
        """Precompute q * RMSNorm gain once after checkpoint loading."""
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

    def _get_block_residual(self, tokens: int) -> torch.Tensor:
        buffer = self.block_residual_buffer
        if buffer is None:
            raise RuntimeError(
                "AttnRes buffers must be initialized before model forward"
            )
        if tokens > buffer.shape[0]:
            raise RuntimeError(
                f"AttnRes buffer holds {buffer.shape[0]} tokens but this step needs "
                f"{tokens}; raise scheduler_config.max_prefill_tokens"
            )
        return buffer[:tokens]

    def _embed(
        self, input_ids: torch.Tensor, *, shard_output: bool = False, pad_len: int = 0
    ) -> torch.Tensor:
        """Sum vocab-shard contributions, optionally producing local token rows."""
        if self.embed_tp_size <= 1:
            return self.embed_tokens(input_ids)
        vocab_per_rank = self.config.vocab_size // self.embed_tp_size
        local_ids = input_ids - self.embed_tp_rank * vocab_per_rank
        mask = (local_ids >= 0) & (local_ids < vocab_per_rank)
        embeds = self.embed_tokens(local_ids * mask) * mask.unsqueeze(-1)
        if shard_output:
            # Pad hidden rows with zeros, not token IDs with a real embedding.
            if pad_len:
                embeds = F.pad(embeds, (0, 0, 0, pad_len))
            return reduce_scatter_first_dim(
                embeds, self.embed_reduce_scatter_group, self.attn_tp_size
            )
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
        embed_sharded = False
        embed_pad_len = 0
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be provided")
            embed_sharded = self.embed_reduce_scatter_group is not None and (
                forward_metadata["is_prefill"] or input_ids.shape[0] % self.attn_tp_size == 0
            )
            if embed_sharded and forward_metadata["is_prefill"]:
                embed_pad_len = -input_ids.shape[0] % self.attn_tp_size
            hidden_states = self._embed(
                input_ids, shard_output=embed_sharded, pad_len=embed_pad_len
            )
        else:
            hidden_states = inputs_embeds
        if self.attn_metadata is not None:
            self.attn_metadata.prepare_flash_mla_metadata(forward_metadata)
        if embed_sharded:
            if embed_pad_len:
                forward_metadata = _sp_pad_metadata(forward_metadata, embed_pad_len)
        elif self.attn_tp_size > 1:
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
        if not forward_metadata["is_prefill"]:
            block_residual = self.decode_block_residual_buffer
            if block_residual is None:
                raise RuntimeError(
                    "fused AttnRes decode buffer must be initialized before "
                    "model forward"
                )
        else:
            block_residual = self._get_block_residual(tokens)
        hidden_states, collected_target_hidden = self._forward_attn_res(
            hidden_states,
            block_residual,
            forward_metadata,
            cache_data,
            query_start_loc,
            query_boundaries,
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
    ) -> AttnResPhase1Stats:
        # Keep the Python wrapper's validation lambdas outside Dynamo tracing.
        inter_numerator, inter_max, inter_exp_sum = torch.ops.cann_ops_transformer.block_attn_res_prepare(
            block_residual,
            valid_blocks.reshape(1),
            effective_queries,
            eps=float(self.config.rms_norm_eps),
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
        norm: Optional[KimiRMSNorm] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if norm is not None:
            output = _block_attn_res_update_rms_norm_impl(
                partial_block,
                partial_delta,
                slot.effective_query,
                slot.inter_numerator,
                slot.inter_max,
                slot.inter_exp_sum,
                norm.weight.to(dtype=partial_delta.dtype),
                self.config.rms_norm_eps,
                norm.variance_epsilon,
            )
            return output, partial_block
        output = _block_attn_res_update_impl(
            partial_block,
            partial_delta,
            slot.effective_query,
            slot.inter_numerator,
            slot.inter_max,
            slot.inter_exp_sum,
            eps=float(self.config.rms_norm_eps),
        )

        return output, partial_block

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
        """Process one block with fused two-phase AttnRes operators."""
        effective_queries = self.attn_res_effective_queries
        valid_blocks_table = self.attn_res_valid_blocks
        if not forward_metadata["is_prefill"]:
            block_indices = self.decode_attn_res_block_indices[block_idx]
        else:
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
        )

        partial_block = torch.zeros_like(hidden_states, dtype=torch.float32)
        previous_mlp_delta = None
        collected_target_hidden = []
        fuse_norm = not forward_metadata["is_prefill"]
        for layer_offset, layer in enumerate(block_layers):
            layer_idx = start_layer_idx + layer_offset
            target_layer_id = layer_idx - 1
            # DSpark consumes pre-norm hidden; the fused op only returns normalized y.
            attention_norm_fused = (
                fuse_norm
                and previous_mlp_delta is not None
                and target_layer_id not in self.dspark_target_layer_ids
            )
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
                    norm=layer.input_layernorm if attention_norm_fused else None,
                )
                attention_input = attention_input.to(hidden_states.dtype)
            attention_output = layer.forward_attention(
                attention_input,
                forward_metadata,
                cache_data[layer.layer_idx],
                query_start_loc,
                query_boundaries,
                self.mla_ctx,
                self.replayssm_ctx,
                input_normalized=attention_norm_fused,
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
                norm=layer.post_attention_layernorm if fuse_norm else None,
            )
            mlp_input = mlp_input.to(hidden_states.dtype)
            previous_mlp_delta = layer.forward_mlp(
                mlp_input,
                forward_metadata,
                self.moe_ctx,
                input_normalized=fuse_norm,
            )
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
            self.infer_config.model_config.draft_model_type
            in DSPARK_DRAFT_MODEL_TYPES
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
        moe_ep_size = parallel.moe_ep_size
        moe_ep_rank = (
            0 if moe_ep_size == 1 else self.comm_manager.get_rank("moe_ep_group")
        )
        experts_per_rank = self.num_experts // moe_ep_size
        self.local_expert_start = moe_ep_rank * experts_per_rank
        self.local_expert_end = self.local_expert_start + experts_per_rank
        self.mxfp4_experts = _mxfp4_expert_quantization(config)
        self.block_size = self.infer_config.scheduler_config.block_size
        self.attn_metadata = AttnMetaData(config, runner_settings)
        self.model.bind_attn_metadata(self.attn_metadata)
        self.exe_mode = self.infer_config.model_config.exe_mode
        self.temperature = self.infer_config.data_config.temperature
        self._bound_cache_data: Optional[tuple[dict, ...]] = None

    def should_load_weight(self, name: str) -> bool:
        return is_local_expert_weight(
            name, self.local_expert_start, self.local_expert_end
        )

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
                # MLA and KDA gather/all-reduce operations use CCU-MS.
                group_type=5,
            )
            self.comm_manager.register_group(
                name="attn_reduce_scatter_group",
                group_num=parallel.world_size // parallel.attn_tp_size,
                group_size=parallel.attn_tp_size,
                # MLA and KDA output reduce-scatter uses the AIV domain.
                group_type=3,
            )
            custom_params = self.infer_config.model_config.custom_params
            if custom_params.get("enable_multi_streams", False):
                # Use a dedicated CCU-MS group for overlapping MLA decode collectives.
                self.comm_manager.register_group(
                    name="mla_decode_tp_group",
                    group_num=parallel.world_size // parallel.attn_tp_size,
                    group_size=parallel.attn_tp_size,
                    # Keep MLA decode gather/all-to-all on the CCU-MS path.
                    group_type=5,
                )
        if parallel.moe_ep_size > 1:
            group_num = parallel.world_size // parallel.moe_ep_size
            self.comm_manager.register_group(
                name="moe_ep_group",
                group_num=group_num,
                group_size=parallel.moe_ep_size,
                group_stride=group_num,
                group_type=3,
            )
            if self.infer_config.model_config.custom_params.get("enable_prefill_mega_moe", False):
                self.comm_manager.register_group(
                    name="megamoe_ep_group",
                    group_num=group_num,
                    group_size=parallel.moe_ep_size,
                    group_stride=group_num,
                    return_name=True,
                    allow_physical_reuse=False,
                    group_type=3,
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
                hccl_buffer_size=self.comm_manager.default_hccl_buffer_size,
                group_type=5,
            )
        if parallel.embed_tp_size > 1:
            self.comm_manager.register_group(
                name="embed_tp_group",
                group_num=parallel.world_size // parallel.embed_tp_size,
                group_size=parallel.embed_tp_size,
                hccl_buffer_size=self.comm_manager.default_hccl_buffer_size,
                group_type=3,
            )
        if parallel.lmhead_tp_size > 1:
            self.comm_manager.register_group(
                name="lmhead_tp_group",
                group_num=parallel.world_size // parallel.lmhead_tp_size,
                group_size=parallel.lmhead_tp_size,
                hccl_buffer_size=self.comm_manager.default_hccl_buffer_size,
                group_type=3,
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
        packed_ids = self._to_packed(input_ids)
        self.model.moe_ctx.prepare_eplb(packed_ids.shape[0], is_prefill, packed_ids.device)
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
        if is_prefill and not forward_metadata.get("is_last_prefill_chunk", True):
            output = torch.empty(
                forward_metadata["actual_seq_lengths_q"].shape[0],
                1,
                dtype=torch.long,
                device=hidden_states.device,
            )
            if self.uses_dspark_draft:
                return output, {
                    "prev_hidden_states": prev_hidden_states,
                    "target_hidden_states": target_hidden_states,
                }
            return output
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
        fused_gate_up_loaded: dict[str, set[int]] = {}
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

            # Dense MLP and shared experts: fold gate/up weights and their MXFP8
            # scales into the matching fused gate_up_proj parameter.
            gate_up = re.match(
                r"(.*)\.(gate_proj|up_proj)\.(weight|weight_scale|scale)$", name
            )
            if gate_up is not None:
                suffix = "weight" if gate_up.group(3) == "weight" else "weight_scale"
                param_name = f"{gate_up.group(1)}.gate_up_proj.{suffix}"
                if param_name in params:
                    param = params[param_name]
                    shard_id = self._GATE_UP_SHARD_ID[gate_up.group(2)]
                    shards = fused_gate_up_loaded.setdefault(param_name, set())
                    if shard_id in shards:
                        raise RuntimeError(
                            f"duplicate checkpoint shard {shard_id} for {param_name}"
                        )
                    param.weight_loader(param, tensor, shard_id)
                    shards.add(shard_id)
                    if shards == set(self._GATE_UP_SHARD_ID.values()):
                        loaded.add(param_name)
                    continue

            qkv_proj = re.match(
                r"(.*)\.(q_proj|k_proj|v_proj)\.(weight|weight_scale|scale)$", name
            )
            if qkv_proj is not None:
                suffix = "weight" if qkv_proj.group(3) == "weight" else "weight_scale"
                param_name = f"{qkv_proj.group(1)}.qkv_proj.{suffix}"
                if param_name in params:
                    param = params[param_name]
                    shard_id = self._KDA_QKV_SHARD[qkv_proj.group(2)]
                    shards = fused_qkv_loaded.setdefault(param_name, set())
                    if shard_id in shards:
                        raise RuntimeError(
                            f"duplicate checkpoint shard {shard_id} for {param_name}"
                        )
                    param.weight_loader(param, tensor, shard_id)
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

            # MX checkpoints use either spelling for E8M0 block scales.
            if name.endswith(".scale"):
                scale_name = name[: -len(".scale")] + ".weight_scale"
                if scale_name in params:
                    name = scale_name

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
                (".q_b_proj.weight_scale", ".q_b_proj_decode.weight_scale"),
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

        missing = sorted(set(params) - loaded)
        if missing:
            raise RuntimeError(
                f"{len(missing)} parameters were never assigned a checkpoint "
                f"tensor and would keep uninitialized memory, starting with: "
                f"{missing[:8]}"
            )
        return loaded

    def process_weights_after_loading(self) -> None:
        # Pre-convert every supported framework Linear whose weight is
        # consumed only by MatMul. Native nn.Linear modules keep their layout.
        # kv_b_proj is excluded below because it is split into 3-D weights
        # for the absorbed-attention path.
        nz_linear_module_names = (
            # MLA query/KV projections.
            "q_a_proj",
            "q_b_proj",
            "q_b_proj_decode",
            "kv_a_proj_with_mqa",
            # KDA projections.
            "qkv_proj",
            # "f_a_proj",
            "f_b_proj",
            "b_proj",
            "g_a_proj",
            "g_b_proj",
            # KDA/MLA output gate and output projection.
            "g_proj",
            "o_proj",
            # Dense FFN and the always-active shared experts.
            "gate_up_proj",
            "down_proj",
            # Stable LatentMoE projections around the routed experts.
            "routed_expert_down_proj",
            "routed_expert_up_proj",
            # Vocab-TP output head. The TP=1 native nn.Linear is unaffected.
            "lm_head",
        )
        # kv_b_proj is split first and skipped in the loop below: the split
        # reads the checkpoint's [out, in] layout, which the loop would
        # transpose and cast to NZ out from under it.
        prepared_mega_linears = set()
        for module in list(self.modules()):
            if isinstance(module, KimiMLAAttention) and module.use_w8a8c8:
                module.prepare_prolog_weights()
            elif isinstance(module, KimiDeltaAttention):
                prepared_mega_linears.update(module.prepare_mega_kda_weights())
        self._split_kv_b_proj()
        for module_name, module in self.named_modules():
            if module in prepared_mega_linears:
                continue
            if "kv_b_proj" in module_name:
                continue
            if isinstance(module, KimiShortConvolution):
                module.build_conv_weight()
                continue
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None and hasattr(
                quant_method, "process_weights_after_loading"
            ):
                module_leaf_name = module_name.rsplit(".", 1)[-1]
                is_nz = module_leaf_name in nz_linear_module_names
                quant_method.process_weights_after_loading(module, is_nz=is_nz)
        for module in self.modules():
            if isinstance(module, KimiMLAAttention) and not module.use_w8a8c8:
                module.prepare_prolog_weights()
            if isinstance(module, KimiDeltaAttention) and module.use_mega_kda:
                # Pack before graph capture; keep Prefill's [K, 3*H*D] layout.
                module._mega_conv_weight = (
                    module.qkv_conv1d._conv_weight.view(
                        module.qkv_conv1d.kernel_size, 3, module.num_heads, module.head_dim
                    ).permute(1, 2, 0, 3).contiguous()
                )
                # ReplaySSM gets ABI-specific NZ roots without changing the
                # shared Linear views used by Prefill and snapshot MegaKDA.
                if module.use_mega_kda_replayssm:
                    module.prepare_mega_replayssm_weights()
        self.model.prepare_attn_res_effective_queries()
        self.model.initialize_runtime_buffers()

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
                    w_k.permute(1, 2, 0).contiguous().detach(),
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
        custom_params = self.infer_config.model_config.custom_params
        # ``infer.check_settings`` is the authoritative full ABI/platform
        # validation. Keep only this small guard for alternate runner paths
        # that instantiate the model without going through that entry point.
        validate_mega_kda_replayssm_switch(
            custom_params,
            draft_model_type,
            next_n,
            error_type=RuntimeError,
        )
        if draft_model_type not in ("none", *DSPARK_DRAFT_MODEL_TYPES):
            raise RuntimeError(f"unsupported draft_model_type={draft_model_type!r}")
        if (draft_model_type == "none" and next_n != 0) or (
            draft_model_type in DSPARK_DRAFT_MODEL_TYPES and next_n <= 0
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
        latent_nz_dim = _KV_CACHE_NZ_DIM
        if self.config.kv_lora_rank % latent_nz_dim:
            raise RuntimeError(
                "the MLA latent width must be divisible by the PA_NZ inner "
                f"dimension: kv_lora_rank={self.config.kv_lora_rank}, "
                f"nz_dim={latent_nz_dim}"
            )
        if self.config.qk_rope_head_dim % _KV_CACHE_NZ_DIM:
            raise RuntimeError(
                "the MLA auxiliary-key width must be divisible by "
                f"{_KV_CACHE_NZ_DIM}: "
                f"qk_rope_head_dim={self.config.qk_rope_head_dim}"
            )
        if parallel.moe_ep_size > 1 and not _mxfp4_expert_quantization(self.config):
            raise RuntimeError("MoE expert parallelism requires MXFP4 experts")
        custom_params = self.infer_config.model_config.custom_params
        if custom_params.get("enable_superkernel", False):
            if self.infer_config.model_config.exe_mode != "npugraph_ex":
                raise RuntimeError("enable_superkernel=True requires exe_mode=npugraph_ex")
            if not self.infer_config.model_config.enable_static_kernel:
                raise RuntimeError("enable_superkernel=True requires enable_static_kernel=True")
            if not custom_params.get("enable_multi_streams", False):
                raise RuntimeError("enable_superkernel=True requires enable_multi_streams=True")
            if not self.config.num_shared_experts:
                raise RuntimeError("enable_superkernel=True requires shared experts")
            scope_apis = ("super_kernel_scope_begin", "super_kernel_scope_end")
            if not all(hasattr(torch.npu, name) for name in scope_apis):
                raise RuntimeError(
                    "enable_superkernel=True requires a torch_npu build with "
                    "super_kernel_scope_begin/super_kernel_scope_end support"
                )


__all__ = [
    "KimiLinearForCausalLM",
    "SituAndMul",
]
