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


def _reinterpret_scale_as_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """Reinterpret uint8-packed MX scales as float8_e8m0fnu (same byte width)."""
    if scale.dtype == torch.uint8:
        return scale.view(dtype=torch.float8_e8m0fnu)
    return scale


class MegaMoEContext:
    """Model-level MegaMoE resources shared by all DeepSeek-V4 MoE layers."""

    def __init__(
        self,
        config,
        num_max_tokens_per_rank: int,
        comm_manager: CommManager,
        ep_group_name: str,
    ) -> None:
        try:
            from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe
        except ImportError as exc:
            raise ImportError(
                "MegaMoE backend requires cann_ops_transformer.ops with "
                "get_symm_buffer_for_mega_moe and mega_moe."
            ) from exc

        if comm_manager is None:
            raise RuntimeError("MegaMoE backend requires an initialized CommManager.")
        if not comm_manager.has_group(ep_group_name):
            raise RuntimeError(
                f"MegaMoE backend requires communication group {ep_group_name!r}."
            )

        # Per-rank upper bound for one mega_moe call, which feeds the whole
        # local prefill batch in a single call (no chunking).
        if num_max_tokens_per_rank <= 0:
            raise ValueError(
                "enable_mega_moe=True requires a positive max_prefill_tokens, "
                f"got {num_max_tokens_per_rank}."
            )

        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.group = comm_manager.get_group(ep_group_name)
        self.mega_moe = mega_moe
        self.sym_buffer = get_symm_buffer_for_mega_moe(
            self.group,
            num_experts=config.n_routed_experts,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_topk=config.num_experts_per_tok,
            hidden=config.hidden_size,
            intermediate_hidden=config.moe_intermediate_size,
            max_recv_token_num=0,
            dispatch_quant_mode=4,
            dispatch_quant_out_dtype=torch.float8_e4m3fn,
            combine_quant_mode=0,
            comm_alg="",
            topk_weights_type=0,
        )
    def destroy(self) -> None:
        if self.sym_buffer is not None:
            self.sym_buffer.destroy()
            self.sym_buffer = None


class MegaMoEBackend:
    """DeepSeek-V4 MegaMoE prefill backend."""

    def __init__(self, context: MegaMoEContext) -> None:
        self.context = context
        self.mega_moe = context.mega_moe
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
            # mega_moe is a collective op and documents num_tokens >= 1, so an
            # empty prefill batch is padded with a 1-token dummy (ids = arange,
            # distinct experts) whose output is discarded. The call itself must
            # still happen to keep per-chunk collectives aligned across ranks.
            # Shared experts run inside the op on the dummy token as well.
            dummy_x, dummy_ids, dummy_weight = moe._megamoe_dummy_chunk
            self._run_mega_moe(moe, dummy_x, dummy_ids, dummy_weight)
            return x.new_empty(0, hidden)

        # mega_moe computes routed AND shared experts: the returned Y already
        # contains the shared-expert contribution, so no model-side add is
        # needed (the former independent-stream shared-expert path is removed).
        routed_output = self._run_mega_moe(moe, x, topk_ids, topk_weight)
        return routed_output.view(num_tokens, hidden)

    @staticmethod
    def build_dummy_chunk(moe) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = int(moe.hidden_dim)
        topk = int(moe.top_k)

        # dummy_x follows the gate weight device/dtype after weight loading.
        # topk ids/weights use the gate_topk/MegaMoE contract.
        # Empty chunks have no real gate output; synthetic ids/weights only
        # satisfy MegaMoE's collective input contract, and the dummy output
        # is discarded.
        dummy_x = moe.gate.weight.new_zeros((1, hidden))
        dummy_ids = torch.arange(
            topk, dtype=torch.int32, device=dummy_x.device
        ).unsqueeze(0)
        dummy_weight = torch.zeros(
            (1, topk), dtype=torch.float32, device=dummy_x.device
        )
        return dummy_x, dummy_ids, dummy_weight

    @staticmethod
    def build_shared_args(moe) -> tuple:
        """Build mega_moe shared-expert inputs from this layer's shared expert.

        Only n_shared_experts == 1 is supported (the sole guard lives here,
        not in check_model_config_before_loading). Model-side layout (950
        post-loading) serves npu_quant_matmul: the weight is a transposed
        [in, out] view with uint8 MX scales, while mega_moe expects:
            shared_l1_weights (1, 2*int_h, hidden)
            shared_l2_weights (1, hidden, int_h)
        Derived tensors are prepared once after weight loading.
        """
        n_shared = getattr(moe, "n_shared_experts", None) or 0
        if n_shared != 1:
            raise RuntimeError(
                "MegaMoE shared expert path only supports n_shared_experts=1"
            )
        shared = moe.shared_experts
        hidden = moe.hidden_dim
        int_h = moe.intermediate_size

        # Post-loading stores weights as transposed [in, out] views; verify the
        # expected layout, then transpose back to logical [out, in].
        w1 = shared.gate_up_proj.weight
        if tuple(w1.shape) != (hidden, 2 * int_h):
            raise RuntimeError(
                f"unexpected shared gate_up_proj layout: {tuple(w1.shape)}"
            )
        w1 = w1.transpose(-2, -1).reshape(1, 2 * int_h, hidden)

        w2 = shared.down_proj.weight
        if tuple(w2.shape) != (int_h, hidden):
            raise RuntimeError(
                f"unexpected shared down_proj layout: {tuple(w2.shape)}"
            )
        w2 = w2.transpose(-2, -1).reshape(1, hidden, int_h)

        # MX scales are stored as (in/64, out, 2) transposed views of the
        # logical (out, in/64, 2) layout.
        s1 = shared.gate_up_proj.weight_scale
        if tuple(s1.shape) != (hidden // 64, 2 * int_h, 2):
            raise RuntimeError(
                f"unexpected shared gate_up_proj scale layout: {tuple(s1.shape)}"
            )
        s1 = s1.transpose(0, 1).reshape(1, 2 * int_h, hidden // 64, 2)

        s2 = shared.down_proj.weight_scale
        if tuple(s2.shape) != (int_h // 64, hidden, 2):
            raise RuntimeError(
                f"unexpected shared down_proj scale layout: {tuple(s2.shape)}"
            )
        s2 = s2.transpose(0, 1).reshape(1, hidden, int_h // 64, 2)

        return (
            [w1],
            [w2],
            [_reinterpret_scale_as_e8m0(s1)],
            [_reinterpret_scale_as_e8m0(s2)],
        )

    def prepare_layer(self, moe) -> None:
        s_w1, s_w2, s_s1, s_s2 = self.build_shared_args(moe)
        moe._megamoe_dummy_chunk = self.build_dummy_chunk(moe)
        moe._megamoe_static_kwargs = {
            "l1_weights": [moe.experts.w13_weight],
            "l2_weights": [moe.experts.w2_weight],
            "weight1_type": torch_npu.float4_e2m1fn_x2,
            "weight2_type": torch_npu.float4_e2m1fn_x2,
            "l1_weights_sf": [
                _reinterpret_scale_as_e8m0(moe.experts.w13_weight_scale)
            ],
            "l2_weights_sf": [
                _reinterpret_scale_as_e8m0(moe.experts.w2_weight_scale)
            ],
            "activation": "swiglu",
            "activation_clamp": moe.swiglu_limit,
            "shared_l1_weights": s_w1,
            "shared_l2_weights": s_w2,
            "shared_l1_weights_sf": s_s1,
            "shared_l2_weights_sf": s_s2,
            "shared_weight1_type": torch.float8_e4m3fn,
            "shared_weight2_type": torch.float8_e4m3fn,
            "shared_expert_quant_out_dtype": torch.float8_e4m3fn,
        }

    def _run_mega_moe(
        self,
        moe,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor,
    ) -> torch.Tensor:
        static_kwargs = moe._megamoe_static_kwargs
        mega_moe_kwargs = {
            "x": x,
            "topk_ids": topk_ids,
            "topk_weights": topk_weight,
            "sym_buffer": self.sym_buffer,
            **static_kwargs,
        }
        routed_output, _ = self.mega_moe(**mega_moe_kwargs)
        return routed_output


@register_op_impl(op_type="megamoe")
def megamoe_ascendc(config, num_max_tokens_per_rank: int,
                    comm_manager: CommManager, ep_group_name: str):
    context = MegaMoEContext(config, num_max_tokens_per_rank, comm_manager, ep_group_name)
    return context, MegaMoEBackend(context)
