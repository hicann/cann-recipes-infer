# Copyright (c) Huawei Technologies Co., Ltd. 2026.
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
# ==============================================================================
#
# Modified from diffusers==0.29.2
#
# ==============================================================================
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.distributed as dist

from .vae import DecoderCausal3D, BaseOutput, DecoderOutput, DiagonalGaussianDistribution, EncoderCausal3D


def decode(
    self, z: torch.FloatTensor, return_dict: bool = True, generator=None
) -> Union[DecoderOutput, torch.FloatTensor]:
    """
    Decode a batch of images/videos.

    Args:
        z (`torch.FloatTensor`): Input batch of latent vectors.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

    Returns:
        [`~models.vae.DecoderOutput`] or `tuple`:
            If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
            returned.

    """
    if not dist.is_initialized():
        raise RuntimeError("""The distributed environment is not initialized! 
        VAE Parallel only supports distributed environment.""")
    up_rate = 8
    overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor)) // 2
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    wid_size = z.shape[-1]
    act_chunk_len = wid_size // world_size

    # Padding the sequence of adjacent devices.
    z_chunks = []
    for i in range(0, wid_size, act_chunk_len):
        if i == 0:
            start_idx = i
        else:
            start_idx = i - overlap_size
        end_idx = min(i + act_chunk_len + overlap_size, wid_size)
        z_chunks.append(z[..., start_idx:end_idx])

    z_local = z_chunks[rank].contiguous()
    decoded_local = self._decode(z_local).sample

    # Remove pads.
    if rank == 0:
        start_idx = 0
    else:
        start_idx = overlap_size * up_rate
    end_idx = start_idx + act_chunk_len * up_rate
    decoded_local = decoded_local[..., start_idx:end_idx].contiguous()

    gather_tensor = torch.empty((world_size,) + decoded_local.shape, 
                            dtype=decoded_local.dtype, 
                            device=decoded_local.device)
    gather_list = list(torch.chunk(gather_tensor, world_size, dim=0))
    dist.all_gather(gather_list, decoded_local)
    decoded = torch.cat(gather_list, dim=-1).contiguous()
    decoded = decoded.squeeze(0)

    if not return_dict:
        return (decoded,)

    return DecoderOutput(sample=decoded)