# coding=utf-8
# Adapted from
# https://github.com/Wan-Video/Wan2.2/blob/main/wan/distributed/sequence_parallel.py
# Copyright (c) Huawei Technologies Co., Ltd. 2025.
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
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
import logging
import types
import torch
import torch_npu
import torch.cuda.amp as amp
from .parallel_mgr import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)
from ..modules.attn_layer import xFuserLongContextAttention
from ..modules.model import sinusoidal_embedding_1d


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=torch.float32,
        device=original_tensor.device
        ).to(original_tensor.dtype)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor



@torch.amp.autocast('npu', enabled=False)
def rope_apply(x, grid_sizes, freqs_list):
    s, n, c = x.size(1), x.size(2), x.size(3)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        x_i = x[i, :s].reshape(1, s, n, c)
        if not x_i.is_contiguous():
            x_i = x_i.contiguous()
        
        cos, sin = freqs_list[i]
        cos = cos.to(dtype=x_i.dtype, device=x_i.device)
        sin = sin.to(dtype=x_i.dtype, device=x_i.device)

        x_i = torch_npu.npu_rotary_mul(
            input=x_i,
            r1=cos,
            r2=sin,
            rotary_mode="interleave"
        )

        output.append(x_i)

    return torch.cat(output).float()


def sp_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    y=None,
    t_idx=None,
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """

    # if self.model_type == 'i2v':
    #     assert y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    # assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    # time embeddings
    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    with torch.amp.autocast('npu', dtype=torch.bfloat16):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim,
                                    t).unflatten(0, (bt, seq_len)).float())
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    # Context Parallel
    x = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    e = torch.chunk(e, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
    e0 = torch.chunk(e0, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]


    if self.freqs_list is None:
        c = (self.dim // self.num_heads) // 2
        s = x.shape[1]
        freqs = self.freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        freqs_list = []

        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = f * h * w
            freqs_i = torch.cat([
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ],
            dim=-1).reshape(seq_len, 1, -1)

            # apply rotary embedding
            sp_size = get_sequence_parallel_world_size()
            sp_rank = get_sequence_parallel_rank()
            freqs_i = pad_freqs(freqs_i, s * sp_size)
            s_per_rank = s
            freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) * s_per_rank), :, :]
            cos, sin = torch.chunk(torch.view_as_real(freqs_i_rank.to(torch.complex64)), 2, dim=-1)
            cos = cos.unsqueeze(0).expand(-1, -1, -1, -1, 2).flatten(-2)
            sin = sin.unsqueeze(0).expand(-1, -1, -1, -1, 2).flatten(-2)
            freqs_i_rank = (cos, sin)
            freqs_list.append(freqs_i_rank)
        self.freqs_list = freqs_list

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs_list,
        context=context,
        context_lens=context_lens,
    )

    for block in self.blocks:
        x = block(x, **kwargs)

    # head
    x = self.head(x, e)

    # Context Parallel
    x = get_sp_group().all_gather(x, dim=1)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


def sp_attn_forward(self, x, seq_lens, grid_sizes, freqs, dtype=torch.bfloat16, args=None):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    q = rope_apply(q, grid_sizes, freqs)
    k = rope_apply(k, grid_sizes, freqs)

    if args is None:
        if hasattr(self, 'size'):
            fake_args = types.SimpleNamespace(size=self.size)
        else:
            fake_args = types.SimpleNamespace(size='1280*720')
    else:
        fake_args = args

    x = xFuserLongContextAttention(args=fake_args)(
        None,
        query=half(q),
        key=half(k),
        value=half(v),
        seq_lens=seq_lens,
        window_size=self.window_size,
    )

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x
