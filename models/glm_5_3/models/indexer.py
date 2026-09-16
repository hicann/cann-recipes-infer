# coding=utf-8
# GLM-5.3 DSA indexer with k-pool compression.
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

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

import torch_npu

from .configuration_glm53 import Glm53Config


def build_pool_gather(
    pool_block_table: torch.Tensor,     # [B, max_pool_blocks] int32 ("DsaPool")
    n_pools_per_req: torch.Tensor,      # [B] complete pools visible to the request
    storage_block_size: int,            # pools per physical block (= block_size)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(pool_slots [B, P] slots into the pooled-key cache, valid [B, P]).

    Prefill only: it reads `.max()` on the host, which graph mode forbids.
    """
    p_max = max(int(n_pools_per_req.max().item()), 1)
    device = pool_block_table.device
    pools = torch.arange(p_max, device=device)
    blk = (pools // storage_block_size).clamp(max=pool_block_table.shape[1] - 1)
    slots = (pool_block_table[:, blk].long() * storage_block_size
             + (pools % storage_block_size))
    valid = pools.unsqueeze(0) < n_pools_per_req.view(-1, 1).to(device)
    return slots, valid


def build_pool_commit_slots(
    pool_block_table: torch.Tensor,   # [B, max_pool_blocks]
    position_ids: torch.Tensor,       # [T] logical position of each new token
    token_batch_idx: torch.Tensor,    # [T] request index of each new token
    storage_block_size: int,
    kpool: int,
) -> torch.Tensor:
    """[T] pooled-cache slot each new token should commit its pool into.

    Tokens that complete no pool are steered to slot 0 (the null block, owned by
    no request) so the commit stays one unconditional scatter of static shape.
    """
    pool_idx = position_ids.long() // kpool
    blk = (pool_idx // storage_block_size).clamp(max=pool_block_table.shape[1] - 1)
    slots = (pool_block_table[token_batch_idx.long(), blk].long() * storage_block_size
             + (pool_idx % storage_block_size))
    completes = (position_ids.long() % kpool) == (kpool - 1)
    return torch.where(completes, slots, torch.zeros_like(slots))


class Glm53Indexer(nn.Module):
    """Scores and selects the top-k compressed key pools for a DSA layer."""

    # Query-block size for scoring chunks, cut inside one request, so the score
    # memory alive at a time is QUERY_BLOCK x n_heads x num_pools x 4B.
    QUERY_BLOCK = 1024

    def __init__(self, config: Glm53Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.kpool = config.index_kpool
        self.always_select_tail = config.index_kpool_always_select_tail
        self.softmax_scale = self.head_dim ** -0.5
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False,
                              dtype=torch.bfloat16)
        self.wk = nn.Linear(config.hidden_size, self.head_dim, bias=False, dtype=torch.bfloat16)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6, dtype=torch.bfloat16)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False,
                                      dtype=torch.bfloat16)
        self.index_kpool_compress_ape = nn.Parameter(
            torch.zeros(self.kpool, self.head_dim, dtype=torch.bfloat16), requires_grad=False)
        self.index_kpool_compress_gate = nn.Parameter(
            torch.zeros(self.head_dim, config.hidden_size, dtype=torch.bfloat16),
            requires_grad=False)

    @property
    def topk_width(self) -> int:
        """Fixed output width: expanded pools + the (kpool-1)-token tail max."""
        return self.index_topk + (self.kpool - 1 if self.always_select_tail else 0)

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,          # [T, hidden] packed
        q_resid: torch.Tensor,                 # [T, q_lora_rank]
        position_ids: torch.Tensor,            # [T] query positions
        key_cache: torch.Tensor,               # [nb, bs, 1, 128] paged
        gate_cache: torch.Tensor,              # [nb, bs, 1, 128] paged
        slot_mapping: torch.Tensor,            # [T] physical slots for new tokens
        pool_cache: torch.Tensor,              # [nb, bs, 1, 128] paged pooled keys
        pool_slot_of_token: torch.Tensor,      # [T] pooled slot this token completes
        num_pools: int,                        # P, the pool-axis width
        pool_slots: Optional[torch.Tensor],    # [B, P] pooled-key slots (prefill)
        pool_valid: Optional[torch.Tensor],    # [B, P] (prefill)
        pool_block_table: Optional[torch.Tensor] = None,   # lightning backend
        n_pools_per_req: Optional[torch.Tensor] = None,
        cu_q: Optional[torch.Tensor] = None,
        is_decode: bool = False,
        query_boundaries: Optional[List[int]] = None,
    ) -> torch.Tensor:
        # 1) compute + cache this step's per-token key / gate
        k_new = self.k_norm(self.wk(hidden_states))                        # [T, 128]
        gate_new = F.linear(hidden_states, self.index_kpool_compress_gate)  # [T, 128]
        torch_npu.npu_scatter_nd_update_(
            key_cache.view(-1, self.head_dim),
            slot_mapping.view(-1, 1),
            k_new.to(key_cache.dtype))
        torch_npu.npu_scatter_nd_update_(
            gate_cache.view(-1, self.head_dim),
            slot_mapping.view(-1, 1),
            gate_new.to(gate_cache.dtype))
        # 2) commit the pools completed by this step
        self._commit_pools(key_cache, gate_cache, slot_mapping, pool_cache,
                           pool_slot_of_token)

        # 3) score pools per packed query, 4) select + expand (+ tail)
        tokens = hidden_states.shape[0]
        q_all = self.wq_b(q_resid).view(tokens, self.n_heads, self.head_dim)
        w_all = self.weights_proj(hidden_states).float() * (self.n_heads ** -0.5)
        select_k = min(self.index_topk // self.kpool, num_pools)

        if is_decode:
            selected = self._select_pools_li(
                q_all, (w_all * self.softmax_scale).to(q_all.dtype), pool_cache,
                pool_block_table, n_pools_per_req, cu_q, select_k)
            return self._expand_pools(selected.clamp(min=0), selected.ge(0),
                                      position_ids)

        pool_keys = pool_cache.view(-1, self.head_dim)[
            pool_slots.reshape(-1)].view(*pool_slots.shape, self.head_dim)
        pool_end = (torch.arange(num_pools, device=pool_keys.device) * self.kpool
                    + self.kpool - 1)
        pool_within_kv = pool_valid

        # Chunk along request boundaries first, then by QUERY_BLOCK inside a
        # request, so every query in a chunk shares one [P, D] pool-key matrix
        # and scoring broadcasts it instead of gathering a [t, P, D] copy.
        if query_boundaries is None:
            raise ValueError(
                "prefill indexer needs step_metadata.query_boundaries to slice "
                "the packed batch into requests")
        outputs = []
        for req_idx, (req_start, req_end) in enumerate(
                zip(query_boundaries, query_boundaries[1:])):
            keys_r = pool_keys[req_idx]                  # [P, D] view, no copy
            valid_r = pool_within_kv[req_idx]            # [P]
            for start in range(req_start, req_end, self.QUERY_BLOCK):
                end = min(start + self.QUERY_BLOCK, req_end)
                outputs.append(self._select_block(
                    q_all[start:end], w_all[start:end], position_ids[start:end],
                    keys_r, pool_end, valid_r, num_pools))
        if not outputs:
            return q_all.new_zeros((0, self.topk_width), dtype=torch.int32)
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)

    def _commit_pools(self, key_cache, gate_cache, slot_mapping, pool_cache,
                      pool_slot_of_token):
        """Write the pooled key of every pool this step completed.

        A pool becomes selectable only once all `kpool` tokens exist and never
        changes afterwards, so it is pooled once and read back from the cache
        from then on -- that is what makes decode O(1) per DSA layer.

        `block_size % kpool == 0` (checked at cache-declaration time) means a
        pool never straddles a block, so its token slots are the `kpool`
        consecutive slots ending at the last token's own slot.
        """
        offs = torch.arange(self.kpool, device=slot_mapping.device)
        tok_slots = (slot_mapping.view(-1, 1).long() - (self.kpool - 1) + offs).clamp_(min=0)
        flat_k = key_cache.view(-1, self.head_dim)
        flat_g = gate_cache.view(-1, self.head_dim)
        k_grp = flat_k[tok_slots.reshape(-1)].view(-1, self.kpool, self.head_dim)
        g_grp = flat_g[tok_slots.reshape(-1)].view(-1, self.kpool, self.head_dim)
        logits = g_grp.float() + self.index_kpool_compress_ape.float()[None]
        probs = logits.softmax(dim=1).to(k_grp.dtype)
        pooled = (probs * k_grp).sum(dim=1)                              # [T, 128]
        torch_npu.npu_scatter_nd_update_(
            pool_cache.view(-1, self.head_dim),
            pool_slot_of_token.view(-1, 1),
            pooled.to(pool_cache.dtype))

    def _select_pools_li(self, q, w, pool_cache, pool_block_table,
                         n_pools_per_req, cu_q, select_k):
        """Pool-level top-k via npu_lightning_indexer -> [t, select_k] pool
        indices, dense prefix + -1 padding.

        Decode only: the op derives causality from a 1:1 query<->key mapping,
        which our kpool:1 pools satisfy only at one query per request. Prefill
        has no equivalent on stock torch_npu and stays on the torch path.
        """
        topk, _ = torch_npu.npu_lightning_indexer(
            query=q,
            key=pool_cache,
            weights=w,
            actual_seq_lengths_query=cu_q.to(torch.int32),
            actual_seq_lengths_key=n_pools_per_req.to(torch.int32),
            block_table=pool_block_table,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=select_k,
            sparse_mode=3,
        )
        selected = topk.reshape(q.shape[0], -1)[:, :select_k].long()
        selected_valid = selected.ge(0)
        order = torch.argsort((~selected_valid).to(torch.float32), dim=-1, stable=True)
        return selected.gather(-1, order)

    def _select_block(self, q, w, positions, pool_keys_r, pool_end,
                      pool_valid_r, num_pools) -> torch.Tensor:
        """One request's query slice.

        q: [t, H, D]; w: [t, H]; positions: [t]; pool_keys_r: [P, D];
        pool_valid_r: [P]  ->  int32 [t, W].
        """
        scores = torch.matmul(q.float(), pool_keys_r.float().transpose(0, 1))
        scores = scores.mul_(self.softmax_scale).relu_()
        index_scores = torch.matmul(w.unsqueeze(1), scores).squeeze(1)   # [t, P]

        visible = pool_end[None, :] <= positions.view(-1, 1)             # causality
        valid_candidates = visible & pool_valid_r[None, :]               # + kv validity
        index_scores = index_scores.masked_fill(
            ~valid_candidates, torch.finfo(index_scores.dtype).min)

        select_k = min(self.index_topk // self.kpool, num_pools)
        selected = index_scores.topk(select_k, dim=-1).indices           # [t, K]
        selected_valid = valid_candidates.gather(-1, selected)

        # sparse_indices must be a dense prefix -- an interior -1 truncates it.
        order = torch.argsort((~selected_valid).to(torch.float32), dim=-1, stable=True)
        selected = selected.gather(-1, order)
        selected_valid = selected_valid.gather(-1, order)

        return self._expand_pools(selected, selected_valid, positions)

    def _expand_pools(self, selected, selected_valid, positions) -> torch.Tensor:
        """Pool indices (dense prefix) -> int32 token indices [t, W], plus tail."""
        device = selected.device
        offsets = torch.arange(self.kpool, device=device)
        token_indices = (selected.unsqueeze(-1) * self.kpool + offsets).flatten(-2)
        token_indices = token_indices.masked_fill(
            ~selected_valid.unsqueeze(-1).expand(-1, -1, self.kpool).flatten(-2), -1)

        width = self.topk_width
        if token_indices.shape[-1] < width:
            token_indices = F.pad(token_indices, (0, width - token_indices.shape[-1]), value=-1)

        if self.always_select_tail:
            # Incomplete tail pool of each query's visible prefix, at most
            # kpool-1 tokens. Scattered right after the last valid pool token
            # (not appended at the fixed width) to keep the prefix contiguous.
            tail_offsets = torch.arange(self.kpool - 1, device=device)
            n_visible = positions.view(-1, 1) + 1
            tail_start = (n_visible // self.kpool) * self.kpool
            tail_indices = tail_start + tail_offsets[None, :]            # [t, kpool-1]
            tail_valid = tail_offsets[None, :] < n_visible.remainder(self.kpool)
            tail_indices = tail_indices.masked_fill(~tail_valid, -1)
            dest = selected_valid.sum(-1, keepdim=True) * self.kpool + tail_offsets[None, :]
            token_indices.scatter_(-1, dest.clamp(max=width - 1), tail_indices)

        return token_indices[..., :width].to(torch.int32)
