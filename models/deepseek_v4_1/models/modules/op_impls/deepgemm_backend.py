# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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



import torch
import torch_npu

from executor.core.config import CommManager
from ..registry import register_op_impl


def _pack_ue8m0_exponents(exponent: torch.Tensor) -> torch.Tensor:
    exponent = exponent.to(torch.int32)
    return (exponent[..., ::2] + exponent[..., 1::2] * 256).to(torch.int16).contiguous()


class DeepGemmContext:
    """Model-level MegaMoE resources shared by all DeepSeek-V4 MoE layers."""

    def __init__(
        self,
        config,
        num_max_tokens_per_rank: int,
        comm_manager: CommManager,
        ep_group_name: str,
    ) -> None:
        try:
            import deep_gemm
        except ImportError as exc:
            raise ImportError(
                "DeepGemm backend requires cann_ops_transformer.ops with "
                ""
            ) from exc

        if comm_manager is None:
            raise RuntimeError("MegaMoE backend requires an initialized CommManager.")
        if not comm_manager.has_group(ep_group_name):
            raise RuntimeError(
                f"MegaMoE backend requires communication group {ep_group_name!r}."
            )

        if num_max_tokens_per_rank <= 0:
            raise ValueError(
                "enable_mega_moe=True requires a positive max_prefill_tokens, "
                f"got {num_max_tokens_per_rank}."
            )

        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.group = comm_manager.get_group(ep_group_name)
        self.deep_gemm = deep_gemm
        num_shared_experts = getattr(config, "n_shared_experts", 0) or 0
        self.sym_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            self.group,
            num_experts=config.n_routed_experts,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_topk=config.num_experts_per_tok,
            hidden=config.hidden_size,
            intermediate_hidden=config.moe_intermediate_size,
            num_shared_experts=num_shared_experts,
        )

    def destroy(self) -> None:
        if self.sym_buffer is not None:
            self.sym_buffer.destroy()
            self.sym_buffer = None


class DeepGemmBackend:
    """DeepSeek-V4 DeepGemm prefill backend."""

    def __init__(self, context: DeepGemmContext) -> None:
        self.context = context
        self.deep_gemm = context.deep_gemm
        self.sym_buffer = context.sym_buffer

    def prefill(
        self,
        moe,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor,
    ) -> torch.Tensor:
        if self.sym_buffer is None:
            raise RuntimeError("MegaMoE backend requires an initialized sym_buffer.")

        num_tokens, hidden = x.shape
        if num_tokens > self.context.num_max_tokens_per_rank:
            raise RuntimeError(
                f"MegaMoE got {num_tokens} tokens on this rank, above the symmetric "
                f"buffer bound {self.context.num_max_tokens_per_rank}."
            )

        if num_tokens == 0:
            dummy_x, dummy_ids, dummy_weight = moe._megamoe_dummy_chunk
            self._run_mega_moe(moe, dummy_x, dummy_ids, dummy_weight)
            return x.new_empty(0, hidden)

        routed_output = self._run_mega_moe(moe, x, topk_ids, topk_weight)
        return routed_output.view(num_tokens, hidden)

    @staticmethod
    def build_dummy_chunk(moe) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = int(moe.hidden_dim)
        topk = int(moe.top_k)

        dummy_x = moe.gate.weight.new_zeros((1, hidden))
        dummy_ids = torch.arange(
            topk, dtype=torch.int32, device=dummy_x.device
        ).unsqueeze(0)
        dummy_weight = torch.zeros(
            (1, topk), dtype=torch.float32, device=dummy_x.device
        )
        return dummy_x, dummy_ids, dummy_weight

    def build_shared_args(self, moe) -> tuple:
        n_shared = getattr(moe, "n_shared_experts", None) or 0
        if n_shared != 1:
            raise RuntimeError(
                "MegaMoE shared expert path only supports n_shared_experts=1"
            )
        shared = moe.shared_experts
        hidden = moe.hidden_dim
        int_h = moe.intermediate_size

        w1 = shared.gate_up_proj.weight
        if tuple(w1.shape) != (hidden, 2 * int_h):
            raise RuntimeError(
                f"unexpected shared gate_up_proj layout: {tuple(w1.shape)}"
            )
        w1 = w1.transpose(-2, -1).contiguous()

        w2 = shared.down_proj.weight
        if tuple(w2.shape) != (int_h, hidden):
            raise RuntimeError(
                f"unexpected shared down_proj layout: {tuple(w2.shape)}"
            )
        w2 = w2.transpose(-2, -1).contiguous()

        s1 = shared.gate_up_proj.weight_scale
        if tuple(s1.shape) != (hidden // 64, 2 * int_h, 2):
            raise RuntimeError(
                f"unexpected shared gate_up_proj scale layout: {tuple(s1.shape)}"
            )
        s1_raw = s1.transpose(0, 1).contiguous().reshape(2 * int_h, hidden // 32)
        s1_packed = _pack_ue8m0_exponents(s1_raw.view(torch.uint8))

        s2 = shared.down_proj.weight_scale
        if tuple(s2.shape) != (int_h // 64, hidden, 2):
            raise RuntimeError(
                f"unexpected shared down_proj scale layout: {tuple(s2.shape)}"
            )
        s2_raw = s2.transpose(0, 1).contiguous().reshape(hidden, int_h // 32)
        s2_packed = _pack_ue8m0_exponents(s2_raw.view(torch.uint8))

        return (w1, s1_packed), (w2, s2_packed)

    def prepare_layer(self, moe) -> None:
        n_shared = getattr(moe, "n_shared_experts", 0) or 0
        if n_shared > 0:
            shared_l1, shared_l2 = self.build_shared_args(moe)
        else:
            shared_l1, shared_l2 = None, None

        moe._megamoe_dummy_chunk = self.build_dummy_chunk(moe)

        experts = moe.experts
        w13 = experts.w13_weight.data
        w2 = experts.w2_weight.data
        w13_sf = experts.w13_weight_scale.data
        w2_sf = experts.w2_weight_scale.data

        w13_nd = torch_npu.npu_format_cast(w13.contiguous(), 2).view(torch.uint8).contiguous()
        w2_nd = torch_npu.npu_format_cast(w2.contiguous(), 2).view(torch.uint8).contiguous()
        w13_int8 = w13_nd.view(torch.int8)
        w2_int8 = w2_nd.view(torch.int8)

        w13_sf_flat = w13_sf.reshape(w13_sf.shape[0], w13_sf.shape[1], -1)
        w2_sf_flat = w2_sf.reshape(w2_sf.shape[0], w2_sf.shape[1], -1)
        w13_sf_packed = _pack_ue8m0_exponents(w13_sf_flat)
        w2_sf_packed = _pack_ue8m0_exponents(w2_sf_flat)

        (fused_w13, fused_w13_sf), (fused_w2, fused_w2_sf) = \
            self.deep_gemm.transform_weights_for_mega_moe(
                (w13_int8, w13_sf_packed), (w2_int8, w2_sf_packed))

        del w13_nd, w2_nd, w13_int8, w2_int8
        del w13_sf_flat, w2_sf_flat, w13_sf_packed, w2_sf_packed
        torch.npu.empty_cache()

        if shared_l1 is not None:
            shared_l1, shared_l2 = self.deep_gemm.transform_weights_for_mega_moe(
                shared_l1, shared_l2)

        moe._megamoe_static_kwargs = {
            "l1_weights": (fused_w13, fused_w13_sf),
            "l2_weights": (fused_w2, fused_w2_sf),
            "shared_l1_weights": shared_l1,
            "shared_l2_weights": shared_l2,
            "activation_clamp": moe.swiglu_limit,
        }

    def release_layer(self, moe) -> None:
        experts = moe.experts
        experts.w13_weight = None
        experts.w2_weight = None
        experts.w13_weight_scale = None
        experts.w2_weight_scale = None
        if getattr(moe, "n_shared_experts", 0) or 0:
            shared = moe.shared_experts
            shared.gate_up_proj.weight = None
            shared.gate_up_proj.weight_scale = None
            shared.down_proj.weight = None
            shared.down_proj.weight_scale = None
        torch.npu.empty_cache()

    def _run_mega_moe(
        self,
        moe,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor,
    ) -> torch.Tensor:
        buffer = self.sym_buffer
        num_tokens = x.shape[0]
        hidden = x.shape[1]

        x_fp8, x_scale = torch_npu.npu_dynamic_mx_quant(
            x.contiguous(), dst_type=torch.float8_e4m3fn)
        x_scale_u8 = x_scale.view(torch.uint8)
        if x_scale_u8.dim() > 2:
            x_scale_u8 = x_scale_u8.flatten(1)
        x_sf_packed = _pack_ue8m0_exponents(x_scale_u8)

        buffer.x[:num_tokens].copy_(x_fp8)
        buffer.x_sf[:num_tokens].copy_(x_sf_packed)
        buffer.topk_idx[:num_tokens].copy_(topk_ids.to(torch.int64))
        buffer.topk_weights[:num_tokens].copy_(topk_weight)

        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device=x.device)

        kwargs = moe._megamoe_static_kwargs
        self.deep_gemm.fp8_fp4_mega_moe(
            y,
            kwargs["l1_weights"],
            kwargs["l2_weights"],
            buffer,
            shared_l1_weights=kwargs["shared_l1_weights"],
            shared_l2_weights=kwargs["shared_l2_weights"],
            activation='swiglu',
            activation_clamp=kwargs["activation_clamp"],
        )
        return y


@register_op_impl(op_type="megamoe")
def megamoe_deepgemm(config, num_max_tokens_per_rank: int,
                     comm_manager: CommManager, ep_group_name: str):
    context = DeepGemmContext(config, num_max_tokens_per_rank, comm_manager, ep_group_name)
    return context, DeepGemmBackend(context)
