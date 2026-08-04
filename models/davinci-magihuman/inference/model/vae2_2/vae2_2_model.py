# coding=utf-8
# Adapted from
# https://github.com/GAIR-NLP/daVinci-MagiHuman,
# Copyright (c) Huawei Technologies Co., Ltd. 2026.

__all__ = ["Wan2_2_VAE", "get_vae2_2"]

import gc
import torch

from .vae2_2_module import Wan2_2_VAE


def get_vae2_2(model_path, weight_dtype=torch.float32) -> Wan2_2_VAE:
    vae = Wan2_2_VAE(vae_pth=model_path, dtype=weight_dtype, device="cuda")
    gc.collect()
    torch.cuda.empty_cache()
    return vae
