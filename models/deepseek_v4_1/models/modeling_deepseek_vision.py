# coding=utf-8
# Copyright (c) 2023 DeepSeek
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Upstream portions are licensed under MIT; see ../LICENSE_VISION.txt.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""DeepSeek-V4-Flash-Vision encoder and image-level parallel packing."""

from bisect import bisect_left
from functools import lru_cache

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from cann_ops_transformer.ops import flash_attn_metadata


IMAGE = 1


@lru_cache(8)
def get_vision_cos_sin(n_h: int, n_w: int, dim: int, theta: float):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class PatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(3 * config.vision_patch_size ** 2, config.vision_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(1))


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.vision_n_heads
        self.head_dim = config.vision_dim // config.vision_n_heads
        self.scale = self.head_dim**-0.5
        self.wqkv = nn.Linear(config.vision_dim, 3 * config.vision_dim)
        self.wo = nn.Linear(config.vision_dim, config.vision_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        total_tokens = hidden_states.shape[0]
        qkv = self.wqkv(hidden_states).reshape(total_tokens, 3, self.n_heads, self.head_dim)
        query, key, value = [part.contiguous() for part in qkv.unbind(dim=1)]
        query = apply_rotary(query, cos, sin)
        key = apply_rotary(key, cos, sin)

        output, _ = torch.ops.cann_ops_transformer.flash_attn(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            metadata=metadata,
            softmax_scale=self.scale,
            mask_mode=0,
            win_left=-1,
            win_right=-1,
            layout_q="TND",
            layout_kv="TND",
            layout_out="TND",
            return_softmax_lse=False,
        )
        return self.wo(output.reshape(total_tokens, -1))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.vision_dim, 2 * config.vision_inter_dim, bias=False)
        self.w2 = nn.Linear(config.vision_inter_dim, config.vision_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = self.w1(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        intermediate_hidden_states = F.silu(gate) * up
        hidden_states = self.w2(intermediate_hidden_states)
        return hidden_states


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.vision_dim)
        self.attn = Attention(config)
        self.norm2 = RMSNorm(config.vision_dim)
        self.mlp = MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(hidden_states, cos, sin, cu_seqlens, max_seqlen, metadata)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class ViT(nn.Module):
    """Packed full-attention ViT with one logical TND sequence per image."""

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.vision_n_heads
        self.head_dim = config.vision_dim // self.n_heads
        self.rope_dim = self.head_dim // 2
        self.rope_theta = config.vision_rope_theta
        self.patch_embed = PatchEmbed(config)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.vision_n_layers))
        self.norm = RMSNorm(config.vision_dim)

    def forward(self, patches: torch.Tensor, image_shapes: list[tuple[int, int]]) -> torch.Tensor:
        hidden_states = self.patch_embed(patches)
        cos_parts = []
        sin_parts = []
        actual_seq_end = []
        sequence_end = 0
        for n_h, n_w in image_shapes:
            cos, sin = get_vision_cos_sin(n_h, n_w, self.rope_dim, self.rope_theta)
            cos_parts.append(cos)
            sin_parts.append(sin)
            sequence_end += n_h * n_w
            actual_seq_end.append(sequence_end)
        cos = torch.cat(cos_parts).to(hidden_states.device)
        sin = torch.cat(sin_parts).to(hidden_states.device)
        cu_seqlens = torch.tensor([0, *actual_seq_end], dtype=torch.int32, device=hidden_states.device)
        max_seqlen = max(n_h * n_w for n_h, n_w in image_shapes)
        metadata = flash_attn_metadata(
            self.n_heads,
            self.n_heads,
            self.head_dim,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            batch_size=len(image_shapes),
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            mask_mode=0,
            win_left=-1,
            win_right=-1,
            layout_q="TND",
            layout_kv="TND",
            layout_out="TND",
        )
        for block in self.blocks:
            hidden_states = block(hidden_states, cos, sin, cu_seqlens, max_seqlen, metadata)
        return self.norm(hidden_states)


class Aligner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.downsample_ratio = config.vision_downsample_ratio
        input_dim = config.vision_dim * self.downsample_ratio**2
        self.w1 = nn.Linear(input_dim, config.hidden_size)
        self.w2 = nn.Linear(config.hidden_size, config.hidden_size)

    def merge(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        r = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)
        return x

    def project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.w1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.w2(hidden_states)
        return hidden_states


def _contiguous_image_cuts(images: list[dict], world_size: int) -> list[int]:
    prefix = [0]
    for image in images:
        prefix.append(prefix[-1] + image["types"].numel())
    total_rows = prefix[-1]
    cuts = [0]
    for rank in range(1, world_size):
        target = (rank * total_rows + world_size - 1) // world_size
        cuts.append(bisect_left(prefix, target))
    cuts.append(len(images))
    return cuts


class DeepseekV41VisionModel(nn.Module):
    """Vision, Aligner, sentinel layout, and image-level view parallelism."""

    def __init__(self, config, view_parallel_group, view_parallel_rank: int, view_parallel_size: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.view_parallel_group = view_parallel_group
        self.view_parallel_rank = view_parallel_rank
        self.view_parallel_size = view_parallel_size
        self.vision = ViT(config)
        self.aligner = Aligner(config)
        self.image_start = nn.Parameter(torch.empty(config.hidden_size))
        self.image_end = nn.Parameter(torch.empty(config.hidden_size))
        self.image_newline = nn.Parameter(torch.empty(config.hidden_size))

    def _encode_local_images(self, images: list[dict]) -> torch.Tensor:
        if not images:
            return self.image_start.new_empty((0, self.hidden_size))

        vision_weight = self.vision.patch_embed.proj.weight
        patches = torch.cat([image["patches"] for image in images]).to(
            device=vision_weight.device, dtype=vision_weight.dtype
        )
        image_shapes = [(image["n_vit_h"], image["n_vit_w"]) for image in images]
        patch_lengths = [n_h * n_w for n_h, n_w in image_shapes]
        hidden_states = self.vision(patches, image_shapes)

        merged_parts = []
        patch_start = 0
        for (n_h, n_w), patch_length in zip(image_shapes, patch_lengths):
            patch_end = patch_start + patch_length
            merged_parts.append(self.aligner.merge(hidden_states[patch_start:patch_end], n_h, n_w))
            patch_start = patch_end

        merged_lengths = [part.shape[0] for part in merged_parts]
        aligned = torch.cat(merged_parts)
        aligned = self.aligner.project(aligned)
        return self._layout_image_embeddings(aligned, images, merged_lengths)

    def _layout_image_embeddings(
        self,
        aligned: torch.Tensor,
        images: list[dict],
        merged_lengths: list[int],
    ) -> torch.Tensor:
        dummy = self.image_start.new_zeros(self.hidden_size)
        sentinel_embeddings = torch.stack(
            (self.image_start, dummy, self.image_newline, self.image_end)
        )
        blocks = []
        merged_start = 0
        for image, merged_length in zip(images, merged_lengths):
            merged_end = merged_start + merged_length
            types = image["types"].to(sentinel_embeddings.device)
            block = sentinel_embeddings.index_select(0, types)
            image_mask = types == IMAGE
            block[image_mask] = aligned[merged_start:merged_end]
            blocks.append(block)
            merged_start = merged_end
        return torch.cat(blocks)

    def _gather_blocks(self, local_blocks: torch.Tensor, rows_per_rank: list[int]) -> torch.Tensor:
        if self.view_parallel_size == 1:
            return local_blocks
        max_rows = max(rows_per_rank)
        padded = local_blocks.new_zeros((max_rows, self.hidden_size))
        padded[: local_blocks.shape[0]] = local_blocks
        gathered = local_blocks.new_empty((self.view_parallel_size * max_rows, self.hidden_size))
        dist.all_gather_into_tensor(gathered, padded.contiguous(), group=self.view_parallel_group)
        gathered = gathered.view(self.view_parallel_size, max_rows, self.hidden_size)
        return torch.cat([gathered[rank, :rows] for rank, rows in enumerate(rows_per_rank)])

    def forward(self, mm_inputs_list: list[dict | None]):
        request_rows = []
        images = []
        for mm_inputs in mm_inputs_list:
            request_images = [] if mm_inputs is None else mm_inputs["images"]
            images.extend(request_images)
            request_rows.append(sum(image["types"].numel() for image in request_images))

        request_embeddings = [None] * len(mm_inputs_list)
        if not images:
            return request_embeddings

        cuts = _contiguous_image_cuts(images, self.view_parallel_size)
        rows_per_rank = [
            sum(image["types"].numel() for image in images[cuts[rank] : cuts[rank + 1]])
            for rank in range(self.view_parallel_size)
        ]
        local_images = images[cuts[self.view_parallel_rank] : cuts[self.view_parallel_rank + 1]]
        local_blocks = self._encode_local_images(local_images)
        all_blocks = self._gather_blocks(local_blocks, rows_per_rank)

        request_start = 0
        for request_index, request_row_count in enumerate(request_rows):
            if request_row_count:
                request_end = request_start + request_row_count
                request_embeddings[request_index] = all_blocks[request_start:request_end]
                request_start = request_end
        return request_embeddings
