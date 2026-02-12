# coding=utf-8
# Adapted from  
# https://github.com/Tencent-Hunyuan/HunyuanVideo,
# Copyright (c) Huawei Technologies Co., Ltd. 2025.
# Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
#
# This code is based on Tencent-Hunyuan's HunyuanVideo library and the HunyuanVideo
# implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to HunyuanVideo used by Tencent-Hunyuan team that trained the model.
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
import os
import json
from pathlib import Path
from einops import rearrange

import torch
import torchvision
import numpy as np
import imageio

CODE_SUFFIXES = {
    ".py",  # Python codes
    ".sh",  # Shell scripts
    ".yaml",
    ".yml",  # Configuration files
}


def safe_dir(path):
    """
    Create a directory (or the parent directory of a file) if it does not exist.

    Args:
        path (str or Path): Path to the directory.

    Returns:
        path (Path): Path object of the directory.
    """
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    return path


def safe_file(path):
    """
    Create the parent directory of a file if it does not exist.

    Args:
        path (str or Path): Path to the file.

    Returns:
        path (Path): Path object of the file.
    """
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    return path


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=1, fps=24):
    """save videos by video tensor
       copy from 
       https://github.com/guoyww/AnimateDiff/blob/e92bd5671ba62c0d774a32951453e328018b7c5b/animatediff/utils/util.py#L61

    Args:
        videos (torch.Tensor): video tensor predicted by the model
        path (str): path to save video
        rescale (bool, optional): rescale the video tensor from [-1, 1] to  . Defaults to False.
        n_rows (int, optional): Defaults to 1.
        fps (int, optional): video save fps. Defaults to 8.
    """
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = torch.clamp(x, 0, 1)
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)