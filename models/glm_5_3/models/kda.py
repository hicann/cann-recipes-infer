# coding=utf-8
# KDA (Kimi Delta Attention) linear attention for GLM-5.3.
# Adapted from
# https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/modeling_glm5_next.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Copyright 2026 The GLM & ZhipuAI team and HuggingFace Inc. team. All rights reserved.
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

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

import torch_npu

import cann_ops_transformer
from ops.cannbot_dsl.flash_kda import flash_kda as _flash_kda_impl
from ops.cannbot_dsl.fused_recurrent_kda import (
    fused_recurrent_kda_op as _recurrent_kda_impl,
)

from executor.core.config import InferenceConfig, CommManager
from executor.core.kv_cache.cache_info import MambaCacheEntry
from executor.utils.forward_metadata import ForwardMetaData
from module.linear import ColumnParallelLinear, ReplicatedLinear, RowParallelLinear

from .configuration_glm53 import Glm53Config
from .modules import Glm53RMSNormGated


@dataclass
class Glm53StepMetaData:
    """Per-step helpers built once at the model level, shared by all layers.

    token_batch_idx:   [T] request index of each packed token.
    n_pools_total:     P, the pool-axis width.
    pool_slots:        [B, P] physical slots of each request's complete pools.
                       Prefill only; decode reads the pooled cache through
                       npu_lightning_indexer's own block table.
    pool_valid:        [B, P] validity mask paired with pool_slots.
    pool_commit_slots: [T] pooled-cache slot each new token commits into
                       (slot 0 = null block when the token completes no pool).
    query_boundaries / query_start_loc / has_initial_state: prefill only.
    """
    token_batch_idx: torch.Tensor
    n_pools_total: int
    pool_slots: Optional[torch.Tensor]
    pool_valid: Optional[torch.Tensor]
    pool_commit_slots: torch.Tensor
    n_pools_per_req: torch.Tensor
    query_boundaries: Optional[List[int]] = None
    query_start_loc: Optional[torch.Tensor] = None
    has_initial_state: Optional[torch.Tensor] = None


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv_norm = torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x / inv_norm


class Glm53KdaAttention(nn.Module):
    """KDA linear attention layer with framework-managed Mamba state caches."""

    CONV_PROJ = ("q", "k", "v")
    CHUNK_SIZE = 64

    def __init__(self, config: Glm53Config, infer_config: InferenceConfig,
                 comm_manager: CommManager, layer_idx: int, prefix: str = "", **kwargs):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = config.linear_head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.lower_bound = config.linear_lower_bound
        self.rms_norm_eps = config.rms_norm_eps
        self.comm_manager = comm_manager

        self.attn_tp_size = infer_config.parallel_config.attn_tp_size
        if self.attn_tp_size > 1:
            raise NotImplementedError(
                "glm_5_3 supports attn_tp_size == 1 only (attention runs DP)")
        self.attn_tp_rank = 0

        self.total_num_heads = config.linear_num_heads
        self.num_heads = self.total_num_heads // self.attn_tp_size
        self.qkv_dim = self.num_heads * self.head_dim
        total_qkv_dim = self.total_num_heads * self.head_dim

        self.use_fused_recurrent_kda = self.lower_bound is not None

        quant_config = getattr(config, "quant_config", None)
        linear_kwargs = dict(bias=False, tp_size=self.attn_tp_size,
                             tp_rank=self.attn_tp_rank, quant_config=quant_config)
        self.q_proj = ColumnParallelLinear(self.hidden_size, total_qkv_dim,
                                           prefix=f"{prefix}.q_proj", **linear_kwargs)
        self.k_proj = ColumnParallelLinear(self.hidden_size, total_qkv_dim,
                                           prefix=f"{prefix}.k_proj", **linear_kwargs)
        self.v_proj = ColumnParallelLinear(self.hidden_size, total_qkv_dim,
                                           prefix=f"{prefix}.v_proj", **linear_kwargs)

        # Depth-wise causal conv weights [C_local, kernel] per q/k/v; loader
        # accepts nn.Conv1d layout [C, 1, kernel] too.
        for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
            param = nn.Parameter(
                torch.empty(self.qkv_dim, self.conv_kernel_size, dtype=torch.bfloat16),
                requires_grad=False)
            param.weight_loader = self._conv_weight_loader
            self.register_parameter(name, param)

        for proj in self.CONV_PROJ:
            self.register_buffer(f"conv_weight_t_{proj}", None, persistent=False)

        self.f_a_proj = ReplicatedLinear(self.hidden_size, self.head_dim, bias=False,
                                         quant_config=quant_config,
                                         prefix=f"{prefix}.f_a_proj")
        self.f_b_proj = ColumnParallelLinear(self.head_dim, total_qkv_dim,
                                             prefix=f"{prefix}.f_b_proj", **linear_kwargs)
        self.g_a_proj = ReplicatedLinear(self.hidden_size, self.head_dim, bias=False,
                                         quant_config=quant_config,
                                         prefix=f"{prefix}.g_a_proj")
        self.g_b_proj = ColumnParallelLinear(self.head_dim, total_qkv_dim,
                                             prefix=f"{prefix}.g_b_proj", **linear_kwargs)
        self.b_proj = ColumnParallelLinear(self.hidden_size, self.total_num_heads,
                                           prefix=f"{prefix}.b_proj", **linear_kwargs)

        self.a_log = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32),
                                  requires_grad=False)
        self.a_log.weight_loader = self._fp32_loader
        self.dt_bias = nn.Parameter(torch.zeros(self.qkv_dim, dtype=torch.float32),
                                    requires_grad=False)
        self.dt_bias.weight_loader = self._fp32_loader

        self.o_norm = Glm53RMSNormGated(self.head_dim, eps=self.rms_norm_eps)
        self.o_proj = RowParallelLinear(total_qkv_dim, self.hidden_size, bias=False,
                                        tp_size=self.attn_tp_size, tp_rank=self.attn_tp_rank,
                                        input_is_parallel=True, quant_config=quant_config,
                                        prefix=f"{prefix}.o_proj")

        self._accepted_tokens_cache = None

        self.attn_type = "Mamba"
        for proj in self.CONV_PROJ:
            setattr(self, f"conv_state_cache_{proj}", torch.Tensor([]))
        self.recurrent_state_cache = torch.Tensor([])
        self.cache_entries = [
            *[
                MambaCacheEntry(
                    cache_name=f"conv_state_cache_{proj}",
                    dtype=torch.bfloat16,
                    needs_block=True,
                    shape=[self.conv_kernel_size - 1, self.qkv_dim],
                    tensor_setter=(
                        lambda tensor, layer=self, name=f"conv_state_cache_{proj}":
                        setattr(layer, name, tensor)),
                )
                for proj in self.CONV_PROJ
            ],
            # fused-kernel layout [blocks, H, D, D]; the torch reference paths
            # transpose the last two dims to/from HF's [.., Dk, Dv]
            MambaCacheEntry(
                cache_name="recurrent_state_cache",
                dtype=(torch.float32 if self.use_fused_recurrent_kda
                       else torch.bfloat16),
                needs_block=True,
                shape=[self.num_heads, self.head_dim, self.head_dim],
                tensor_setter=lambda tensor, layer=self: setattr(
                    layer, "recurrent_state_cache", tensor),
            ),
        ]

    # ---- weight loading / post-processing -------------------------------
    @staticmethod
    def _conv_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
        w = loaded_weight
        if w.dim() == 3:  # nn.Conv1d layout [C, 1, kernel]
            w = w.squeeze(1)
        param.data.copy_(w.to(param.dtype))

    @staticmethod
    def _fp32_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight.to(torch.float32))

    def _conv_inputs(self, qkv):
        """(x, conv state cache, transposed weight) per q/k/v projection."""
        return [
            (x, getattr(self, f"conv_state_cache_{p}"),
             getattr(self, f"conv_weight_t_{p}"))
            for x, p in zip(qkv, self.CONV_PROJ)
        ]

    def build_conv_weight(self):
        with torch.no_grad():
            for proj in self.CONV_PROJ:
                w = getattr(self, f"{proj}_conv1d")
                setattr(self, f"conv_weight_t_{proj}", w.transpose(0, 1).contiguous())

    # ---- gates ----------------------------------------------------------
    def _raw_decay(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """[T, H, D] raw projection, bf16, before dt_bias / a_log activation.

        Stays bf16: the fused kernels take the raw gate and assert bf16; only
        the gated-delta-rule path activates it, in _activate_decay.
        """
        raw = self.f_b_proj(self.f_a_proj(hidden_states))
        return raw.view(-1, self.num_heads, self.head_dim)

    def _activate_decay(self, raw_decay: torch.Tensor) -> torch.Tensor:
        gate_input = raw_decay.float() + \
            self.dt_bias.float().view(self.num_heads, self.head_dim)
        decay_rate = torch.exp(self.a_log.float()).view(self.num_heads, 1)
        if self.lower_bound is not None:
            return float(self.lower_bound) * torch.sigmoid(decay_rate * gate_input)
        softplus = torch.where(gate_input > 20.0, gate_input,
                               torch.log1p(torch.exp(gate_input)))
        return -decay_rate * softplus

    def _accepted_tokens(self, batch_size: int) -> torch.Tensor:
        """Cached all-ones `num_accepted_tokens` for fused_recurrent_kda.

        Passing None makes the kernel allocate an int64 tensor per call, which
        falls back to AI_CPU. While next_n == 0 the value is constant, so build
        it once. MTP would need the real per-request accepted count here.
        """
        cache = self._accepted_tokens_cache
        device = self.recurrent_state_cache.device
        if cache is None or cache.shape[0] != batch_size or cache.device != device:
            cache = torch.ones(batch_size, dtype=torch.int32, device=device)
            self._accepted_tokens_cache = cache
        return cache

    def _state_block_ids(self, forward_metadata: ForwardMetaData,
                         batch_size: int) -> torch.Tensor:
        block_table = forward_metadata.block_table[self.attn_type]
        if block_table.shape[0] < batch_size:
            raise RuntimeError(
                f"layer {self.layer_idx}: Mamba block_table covers "
                f"{block_table.shape[0]} requests but this step runs {batch_size}.")
        return block_table[:batch_size, 0].to(torch.int32)

    # ---- main forward ----------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,       # [T, hidden] packed TND
        forward_metadata: ForwardMetaData,
        step_metadata: Optional[Glm53StepMetaData] = None,
        **kwargs,
    ) -> torch.Tensor:
        tokens = hidden_states.shape[0]

        qkv = [self.q_proj(hidden_states),
               self.k_proj(hidden_states),
               self.v_proj(hidden_states)]
        raw_decay = self._raw_decay(hidden_states)         # [T, H, D] bf16 (raw)
        raw_beta = self.b_proj(hidden_states)              # [T, H] pre-sigmoid
        gate = self.g_b_proj(self.g_a_proj(hidden_states))  # [T, C]

        if forward_metadata.is_prefill:
            core_out = self._forward_prefill(qkv, raw_decay, raw_beta,
                                             forward_metadata, step_metadata)
        else:
            core_out = self._forward_decode(qkv, raw_decay, raw_beta, forward_metadata)

        shape = (tokens, self.num_heads, self.head_dim)
        output = self.o_norm(core_out.view(shape), gate.view(shape))
        output = output.to(hidden_states.dtype).reshape(tokens, -1)
        return self.o_proj(output)

    # ---- prefill ---------------------------------------------------------
    def _forward_prefill(self, qkv, raw_decay, raw_beta, forward_metadata,
                         step_metadata) -> torch.Tensor:
        batch_size = len(step_metadata.query_boundaries) - 1
        state_ids = self._state_block_ids(forward_metadata, batch_size)

        q, k, v = self._conv_prefill(qkv, state_ids, step_metadata)  # 3 x [T, C] silu'd

        outputs = []
        final_states = []
        for r, (start, end) in enumerate(zip(step_metadata.query_boundaries,
                                             step_metadata.query_boundaries[1:])):
            shape = (1, end - start, self.num_heads, self.head_dim)
            q_r = q[start:end].view(shape)
            k_r = k[start:end].view(shape)
            v_r = v[start:end].view(shape)
            out, state = self._prefill_flash_kda(
                q_r, k_r, v_r,
                raw_decay[start:end].unsqueeze(0),
                raw_beta[start:end].unsqueeze(0))
            outputs.append(out[0])
            final_states.append(state[0])  # flash_kda emits the fused layout

        states = torch.stack(final_states).to(self.recurrent_state_cache.dtype)
        torch_npu.npu_scatter_nd_update_(
            self.recurrent_state_cache, state_ids.view(-1, 1).long(), states)
        return torch.cat(outputs, dim=0)  # [T, H, D]

    def _prefill_flash_kda(self, q, k, v, raw_decay, raw_beta):
        """One-request flash_kda call (fuses l2norm + gate activation + beta
        sigmoid; consumes RAW decay [1,L,H,D] / beta [1,L,H])."""
        tokens = q.shape[1]
        pad_len = (-tokens) % self.CHUNK_SIZE
        if pad_len:
            q, k, v = (F.pad(t, (0, 0, 0, 0, 0, pad_len)) for t in (q, k, v))
            raw_decay = F.pad(raw_decay, (0, 0, 0, 0, 0, pad_len), value=float("-inf"))
            raw_beta = F.pad(raw_beta, (0, 0, 0, pad_len), value=float("-inf"))
        q, k, v, g, b = (t.contiguous() for t in (q, k, v, raw_decay, raw_beta))
        initial_state = torch.zeros(
            1, self.num_heads, self.head_dim, self.head_dim,
            dtype=torch.float32, device=q.device)
        out, state = _flash_kda_impl(
            q, k, v, g=g, beta=b,
            scale=1.0 / math.sqrt(self.head_dim),
            initial_state=initial_state,
            A_log=self.a_log.data,
            dt_bias=self.dt_bias.view(self.num_heads, self.head_dim),
            lower_bound=self.lower_bound,
            layout_qkv="BSND",
        )
        if pad_len:
            out = out[:, :tokens].contiguous()
        return out, state

    def _conv_prefill(self, qkv, state_ids, step_metadata) -> list:
        """qkv: list of three [T, C] raw projections -> three silu'd [T, C]."""
        return [
            torch.ops.cann_ops_transformer.causal_conv1d_fn(
                x=x,
                conv_states=cache,
                cache_indices=state_ids,
                weight=weight_t,
                bias=None,
                query_start_loc=step_metadata.query_start_loc,
                has_initial_state=step_metadata.has_initial_state,
                activation="silu",
            )
            for x, cache, weight_t in self._conv_inputs(qkv)
        ]

    def _conv_decode(self, qkv, state_ids) -> list:
        """qkv: list of three [B, C] raw projections; rolls the conv windows."""
        batch_size = qkv[0].shape[0]
        return [
            torch.ops.cann_ops_transformer.causal_conv1d_update(
                x=x.contiguous().view(batch_size, 1, self.qkv_dim),
                conv_state=cache,
                conv_state_indices=state_ids,
                weight=weight_t,
                bias=None,
                activation="silu",
            ).view(batch_size, self.qkv_dim)
            for x, cache, weight_t in self._conv_inputs(qkv)
        ]

    # ---- decode ----------------------------------------------------------
    def _forward_decode(self, qkv, raw_decay, raw_beta, forward_metadata) -> torch.Tensor:
        batch_size = qkv[0].shape[0]
        state_ids = self._state_block_ids(forward_metadata, batch_size)

        q, k, v = self._conv_decode(qkv, state_ids)   # 3 x [B, C] silu'd
        shape = (batch_size, self.num_heads, self.head_dim)
        q, k, v = q.view(shape), k.view(shape), v.view(shape)

        if self.use_fused_recurrent_kda:
            out = _recurrent_kda_impl(
                q.unsqueeze(1).contiguous(),
                k.unsqueeze(1).contiguous(),
                v.unsqueeze(1).contiguous(),
                state=self.recurrent_state_cache,
                beta=raw_beta.unsqueeze(1).unsqueeze(-1).contiguous(),
                g=raw_decay.unsqueeze(1).contiguous(),
                scale=1.0 / math.sqrt(self.head_dim),
                A_log=self.a_log.data,
                dt_bias=self.dt_bias.view(self.num_heads, self.head_dim),
                lower_bound=self.lower_bound,
                layout_qkv="BSND",
                ssm_state_indices=state_ids.contiguous(),
                num_accepted_tokens=self._accepted_tokens(batch_size),
            )
            return out.view(batch_size, self.num_heads, self.head_dim)

        g = self._activate_decay(raw_decay)          # [B, H, D] fp32
        beta = raw_beta.float().sigmoid()            # [B, H]
        q_n = l2norm(q.float()).to(torch.bfloat16)
        k_n = l2norm(k.float()).to(torch.bfloat16)
        core_out = torch_npu.npu_recurrent_gated_delta_rule(
            q_n, k_n, v.to(torch.bfloat16), self.recurrent_state_cache,
            beta=beta.to(torch.bfloat16),
            scale=1.0 / math.sqrt(self.head_dim),
            actual_seq_lengths=forward_metadata.actual_seq_lengths_q.to(torch.int32),
            ssm_state_indices=state_ids,
            num_accepted_tokens=None,
            g=None,
            gk=g,
        )
        return core_out.view(batch_size, self.num_heads, self.head_dim)
