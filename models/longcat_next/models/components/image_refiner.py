# coding=utf-8
# Adapted from
# https://huggingface.co/meituan-longcat/LongCat-Next  (image_refiner.py)
# Copyright (c) 2026 Meituan
# The LongCat-Next portion above is released under the MIT License (see that repository's LICENSE file).
# Local changes: constructor-parameter config filtering, identifier-case, assertion-style
# and formatting compliance fixes, licensed under the Apache License, Version 2.0,
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

"""Image refiner: refiner pipeline, refiner container, and utilities.

Contains:
- RefinerImageProcessor: Image pre/post-processing for the diffusion pipeline
- RefinerPipeline: DiffusionPipeline for image refinement
- ImageRefinerContainer: nn.Module container for refiner sub-modules
- IdentityWithArgs: Placeholder module for cond_proj
- de_transform / tensor2pil: Tensor-to-PIL conversion utilities
"""

import inspect
import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from safetensors.torch import load_file
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from diffusers import DiffusionPipeline
from diffusers.configuration_utils import register_to_config
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor, is_valid_image_imagelist
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from .refiner_modules import FlowMatchEulerDiscreteScheduler

from .refiner_modules import Transformer2DModel, RotaryPosEmbed

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_config_dict(cfg, cls) -> dict:
    """Keep only the target constructor's explicitly declared keyword arguments."""
    if hasattr(cfg, "to_dict"):
        d = cfg.to_dict()
    elif isinstance(cfg, dict):
        d = dict(cfg)
    else:
        d = {k: v for k, v in vars(cfg).items()}

    # Local adaptation: derive accepted keys from the constructor, not a metadata blocklist.
    parameters = inspect.signature(cls.__init__).parameters
    valid_keys = set()
    for name, parameter in parameters.items():
        if name == "self":
            continue
        if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            valid_keys.add(name)
    return {k: v for k, v in d.items() if k in valid_keys}


# ---------------------------------------------------------------------------
# Image Refiner Container (nn.Module for state_dict loading)
# ---------------------------------------------------------------------------


class ImageRefinerContainer(nn.Module):
    """Container for refiner components.

    Holds base_transformer, vae, cond_proj as nn.Module children so their
    parameters appear in the parent model's state_dict and are loaded
    automatically via from_pretrained.
    """

    def __init__(self, visual_decoder_config):
        super().__init__()

        tc = visual_decoder_config.transformer_config
        vc = visual_decoder_config.vae_config

        self.base_transformer = Transformer2DModel(**_clean_config_dict(tc, Transformer2DModel))

        self.vae = AutoencoderKL(**_clean_config_dict(vc, AutoencoderKL))
        self.vae.requires_grad_(False)

        text_feat_dim = getattr(tc, "text_feat_dim", 3584)
        codebook_dim = getattr(visual_decoder_config, "codebook_dim", text_feat_dim)
        if codebook_dim != text_feat_dim:
            self.cond_proj = nn.Linear(codebook_dim, text_feat_dim)
        else:
            self.cond_proj = IdentityWithArgs()

    @classmethod
    def from_pretrained(cls, config, model_path: str):
        model = cls(config)
        weight_dict = load_file(model_path, device="cpu")
        model.load_state_dict({k.removeprefix("image_refiner."): v for k, v in weight_dict.items()
                              if k.startswith("image_refiner.")}, strict=True)
        model.eval()
        return model

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


class RefinerImageProcessor(VaeImageProcessor):
    """Image processor for refiner - extends diffusers' VaeImageProcessor."""

    @register_to_config
    def __init__(
        self,
        do_resize: bool = True,
        vae_scale_factor: int = 16,
        resample: str = "lanczos",
        max_pixels: Optional[int] = None,
        max_side_length: Optional[int] = None,
        do_normalize: bool = True,
        do_binarize: bool = False,
        do_convert_grayscale: bool = False,
    ):
        super().__init__(
            do_resize=do_resize,
            vae_scale_factor=vae_scale_factor,
            resample=resample,
            do_normalize=do_normalize,
            do_binarize=do_binarize,
            do_convert_grayscale=do_convert_grayscale,
        )
        self.max_pixels = max_pixels
        self.max_side_length = max_side_length

    def get_new_height_width(
        self,
        image: Union["Image.Image", np.ndarray, torch.Tensor],
        height: Optional[int] = None,
        width: Optional[int] = None,
        max_pixels: Optional[int] = None,
        max_side_length: Optional[int] = None,
    ) -> Tuple[int, int]:
        import PIL.Image

        if height is None:
            if isinstance(image, PIL.Image.Image):
                height = image.height
            elif isinstance(image, torch.Tensor):
                height = image.shape[2]
            else:
                height = image.shape[1]

        if width is None:
            if isinstance(image, PIL.Image.Image):
                width = image.width
            elif isinstance(image, torch.Tensor):
                width = image.shape[3]
            else:
                width = image.shape[2]

        if max_side_length is None:
            max_side_length = self.max_side_length
        if max_pixels is None:
            max_pixels = self.max_pixels

        ratio = 1.0
        if max_side_length is not None:
            max_side_length_ratio = max_side_length / max(height, width)
        else:
            max_side_length_ratio = 1.0

        cur_pixels = height * width
        max_pixels_ratio = (max_pixels / cur_pixels) ** 0.5 if max_pixels is not None else 1.0
        ratio = min(max_pixels_ratio, max_side_length_ratio, 1.0)

        sf = self.config.vae_scale_factor
        new_height = int(height * ratio) // sf * sf
        new_width = int(width * ratio) // sf * sf
        return new_height, new_width

    def preprocess(  # pylint: disable=huawei-too-many-arguments
        self,
        image: PipelineImageInput,
        height: Optional[int] = None,
        width: Optional[int] = None,
        max_pixels: Optional[int] = None,
        max_side_length: Optional[int] = None,
        resize_mode: str = "default",
        crops_coords: Optional[Tuple[int, int, int, int]] = None,
    ) -> torch.Tensor:
        import PIL.Image

        supported_formats = (PIL.Image.Image, np.ndarray, torch.Tensor)

        if self.config.do_convert_grayscale and isinstance(image, (torch.Tensor, np.ndarray)) and image.ndim == 3:
            if isinstance(image, torch.Tensor):
                image = image.unsqueeze(1)
            else:
                if image.shape[-1] == 1:
                    image = np.expand_dims(image, axis=0)
                else:
                    image = np.expand_dims(image, axis=-1)

        if isinstance(image, list) and isinstance(image[0], np.ndarray) and image[0].ndim == 4:
            warnings.warn(
                "Passing `image` as a list of 4d np.ndarray is deprecated. "
                "Please concatenate the list along the batch dimension and pass it as a single 4d np.ndarray",
                FutureWarning,
            )
            image = np.concatenate(image, axis=0)
        if isinstance(image, list) and isinstance(image[0], torch.Tensor) and image[0].ndim == 4:
            warnings.warn(
                "Passing `image` as a list of 4d torch.Tensor is deprecated. "
                "Please concatenate the list along the batch dimension and pass it as a single 4d torch.Tensor",
                FutureWarning,
            )
            image = torch.cat(image, axis=0)

        if not is_valid_image_imagelist(image):
            raise ValueError(
                f"Input is in incorrect format. Currently, we only support "
                f"{', '.join(str(x) for x in supported_formats)}"
            )
        if not isinstance(image, list):
            image = [image]

        if isinstance(image[0], PIL.Image.Image):
            if crops_coords is not None:
                image = [i.crop(crops_coords) for i in image]
            if self.config.do_resize:
                height, width = self.get_new_height_width(image[0], height, width, max_pixels, max_side_length)
                image = [self.resize(i, height, width, resize_mode=resize_mode) for i in image]
            if self.config.do_convert_grayscale:
                image = [self.convert_to_grayscale(i) for i in image]
            image = self.pil_to_numpy(image)
            image = self.numpy_to_pt(image)
        elif isinstance(image[0], np.ndarray):
            image = np.concatenate(image, axis=0) if image[0].ndim == 4 else np.stack(image, axis=0)
            image = self.numpy_to_pt(image)
            height, width = self.get_new_height_width(image, height, width, max_pixels, max_side_length)
            if self.config.do_resize:
                image = self.resize(image, height, width)
        elif isinstance(image[0], torch.Tensor):
            image = torch.cat(image, axis=0) if image[0].ndim == 4 else torch.stack(image, axis=0)
            if self.config.do_convert_grayscale and image.ndim == 3:
                image = image.unsqueeze(1)
            channel = image.shape[1]
            if channel == self.config.vae_latent_channels:
                return image
            height, width = self.get_new_height_width(image, height, width, max_pixels, max_side_length)
            if self.config.do_resize:
                image = self.resize(image, height, width)

        do_normalize = self.config.do_normalize
        if do_normalize and image.min() < 0:
            warnings.warn(
                "Passing `image` as torch tensor with value range in [-1,1] is deprecated. "
                f"The expected value range for image tensor is [0,1] when passing as pytorch tensor or numpy Array. "
                f"You passed `image` with value range [{image.min()},{image.max()}]",
                FutureWarning,
            )
            do_normalize = False
        if do_normalize:
            image = self.normalize(image)

        if self.config.do_binarize:
            image = self.binarize(image)

        return image


@dataclass
class RefinerOutput:
    images: Union[List[Image.Image], torch.Tensor]


class IdentityWithArgs(nn.Module):
    """Placeholder Identity module for cond_proj."""

    def __init__(self, dtype=torch.float32, device=None):
        super().__init__()
        self.register_buffer("_dummy", torch.zeros((), dtype=dtype, device=device))

    @property
    def dtype(self):
        return self._dummy.dtype

    @property
    def device(self):
        return self._dummy.device

    def forward(self, x, *args, **kwargs):
        return x


def _retrieve_timesteps(
    scheduler: FlowMatchEulerDiscreteScheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    # If scheduler uses dynamic shifting and caller passed num_tokens, compute mu
    # (same as training code refiner pipeline)
    num_tokens = kwargs.pop("num_tokens", None)
    if num_tokens is not None and getattr(scheduler.config, "use_dynamic_shifting", False):
        # Compute mu from num_tokens using scheduler's linear interpolation
        base_shift = getattr(scheduler.config, "base_shift", 0.5)
        max_shift = getattr(scheduler.config, "max_shift", 1.15)
        base_seq_len = getattr(scheduler.config, "base_image_seq_len", 256)
        max_seq_len = getattr(scheduler.config, "max_image_seq_len", 4096)
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = num_tokens * m + b
        kwargs["mu"] = mu

    accepted = set(inspect.signature(scheduler.set_timesteps).parameters.keys())
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted}

    if timesteps is not None:
        if "timesteps" not in accepted:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **filtered_kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **filtered_kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class RefinerPipeline(DiffusionPipeline):
    """
    Image refiner evaluation pipeline.

    - cond comes from upstream model: encoder_hidden_states (quants / last_latent)
    - grid_thw_list is used to split cond (consistent with training)
    - image as ref image
    - Supports FlowMatchEulerDiscreteScheduler + velocity model
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        transformer: Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        cond_proj: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            cond_proj=cond_proj if cond_proj is not None else IdentityWithArgs(),
        )

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if hasattr(self.vae.config, "block_out_channels")
            else 8
        )
        self.image_processor = RefinerImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2, do_resize=True
        )
        self.patch_size = int(getattr(self.transformer.config, "patch_size", 16))

        self._num_timesteps: int = 0
        self._current_timestep: Optional[torch.Tensor] = None
        self._interrupt: bool = False
        self._freqs_cis: Optional[torch.Tensor] = None
        self._text_guidance_scale: float = 1.0
        self._image_guidance_scale: float = 1.0
        self._cfg_range: Tuple[float, float] = (0.0, 1.0)

    @torch.no_grad()
    def _get_freqs_cis(self, device, dtype):
        if self._freqs_cis is None:
            self._freqs_cis = RotaryPosEmbed.get_freqs_cis(
                self.transformer.config.axes_dim_rope,
                self.transformer.config.axes_lens,
                theta=10000,
            )
        return self._freqs_cis

    @staticmethod
    def _split_tokens(
        encoder_hidden_states: torch.Tensor,
        grid_thw_list: List[Tuple[int, int, int]],
    ) -> List[torch.Tensor]:
        splits = [int(h) * int(w) // 4 for (_, h, w) in grid_thw_list]
        return list(torch.split(encoder_hidden_states, splits, dim=1))

    @staticmethod
    def _looks_like_latents(x: Union[torch.Tensor, Image.Image], latent_ch_hint: int = 16) -> bool:
        if not isinstance(x, torch.Tensor):
            return False
        if x.ndim not in (3, 4):
            return False
        c = int(x.shape[-3])
        if c == 3:
            return False
        if c == latent_ch_hint:
            return True
        if c > 3 and c <= 32:
            return True
        return False

    @torch.no_grad()
    def _preprocess_to_vae_range(self, img: torch.Tensor) -> torch.Tensor:
        if img.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            img = img.float()
        if img.max() > 1.5:
            img = img / 255.0
        if img.min() >= 0.0 and img.max() <= 1.0:
            img = img * 2.0 - 1.0
        return img.clamp(-1, 1)

    @torch.no_grad()
    def _encode_image_to_latents(
        self,
        img_any: Union[Image.Image, torch.Tensor],
        device,
        dtype,
    ) -> Tuple[torch.Tensor, int, int]:
        latent_ch_hint = int(getattr(getattr(self.vae, "config", None), "latent_channels", 16))

        if self._looks_like_latents(img_any, latent_ch_hint=latent_ch_hint):
            z = img_any
            if z.ndim == 3:
                z = z.unsqueeze(0)
            z = z.to(device=device, dtype=dtype)
            h_lat, w_lat = z.shape[-2], z.shape[-1]
            return z, h_lat, w_lat

        if isinstance(img_any, Image.Image):
            img = torch.from_numpy(
                np.array(img_any).astype("float32") / 255.0
            ).permute(2, 0, 1).unsqueeze(0)
        elif isinstance(img_any, torch.Tensor):
            img = img_any
            if img.ndim == 3:
                img = img.unsqueeze(0)
        else:
            raise TypeError("Unsupported image type. Use PIL.Image or torch.Tensor or latent Tensor.")

        img = self._preprocess_to_vae_range(img)

        h_dim, w_dim = img.shape[-2:]
        base = self.patch_size * self.vae_scale_factor
        target_h = max(base, math.ceil(h_dim / base) * base)
        target_w = max(base, math.ceil(w_dim / base) * base)
        if (h_dim != target_h) or (w_dim != target_w):
            img = F.interpolate(img, size=(target_h, target_w), mode="bilinear", align_corners=False)

        img = img.to(device=device, dtype=self.vae.dtype)

        posterior = self.vae.encode(img).latent_dist
        z0 = posterior.sample()
        if getattr(self.vae.config, "shift_factor", None) is not None:
            z0 = z0 - self.vae.config.shift_factor
        if getattr(self.vae.config, "scaling_factor", None) is not None:
            z0 = z0 * self.vae.config.scaling_factor

        z0 = z0.to(device=device, dtype=dtype)
        h_lat, w_lat = z0.shape[-2], z0.shape[-1]
        return z0, h_lat, w_lat

    @staticmethod
    def _expand_to_list(x, n):
        if x is None:
            return [None] * n
        if isinstance(x, (Image.Image, torch.Tensor)):
            return [x] * n
        if not (isinstance(x, list)):
            raise AssertionError("`image` must be PIL / Tensor or list of them.")
        if not (len(x) == n):
            raise AssertionError("`len(image)` must equal number of image chunks")
        return x

    @torch.no_grad()
    def _denoise_once(  # pylint: disable=huawei-too-many-arguments
        self,
        cond_tokens: torch.Tensor,
        ref_img: Optional[Union[Image.Image, torch.Tensor]],
        num_inference_steps: int = 28,
        timesteps: Optional[List[int]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        text_guidance_scale: float = 1.0,
        image_guidance_scale: float = 1.0,
        cfg_range: Tuple[float, float] = (0.0, 1.0),
        enable_processor_bar: bool = True,
    ):
        device = cond_tokens.device
        weight_dtype = self.transformer.dtype

        self._text_guidance_scale = text_guidance_scale
        self._image_guidance_scale = image_guidance_scale
        self._cfg_range = cfg_range

        cond_tokens = cond_tokens.to(device=device, dtype=weight_dtype)
        text_feats = self.cond_proj(cond_tokens)
        batch, seq_len, _ = text_feats.shape
        text_mask = torch.ones(batch, seq_len, device=device, dtype=torch.bool)

        ref_image_hidden_states = None
        h_lat: int
        w_lat: int

        if ref_img is not None:
            if isinstance(ref_img, torch.Tensor) and ref_img.ndim == 4 and ref_img.shape[0] == batch:
                z_ref, h_lat, w_lat = self._encode_image_to_latents(ref_img, device=device, dtype=weight_dtype)
                ref_image_hidden_states = [[z_ref[b]] for b in range(batch)]
            else:
                z_ref, h_lat, w_lat = self._encode_image_to_latents(ref_img, device=device, dtype=weight_dtype)
                z_single = z_ref[0]
                ref_image_hidden_states = [[z_single] for _ in range(batch)]
        else:
            h_lat = w_lat = 128 // self.vae_scale_factor

        c_lat = getattr(self.transformer.config, "in_channels", None)
        if c_lat is None:
            if ref_image_hidden_states is not None:
                c_lat = ref_image_hidden_states[0][0].shape[0]
            else:
                raise ValueError("transformer.config.in_channels is None and no ref_img was provided.")
        latents_shape = (batch, c_lat, h_lat, w_lat)

        if isinstance(generator, list):
            if len(generator) != batch:
                raise ValueError(
                    f"len(generator)={len(generator)} must equal B={batch} when passing list of generators."
                )
            latents = torch.stack(
                [
                    torch.randn(
                        (1, c_lat, h_lat, w_lat),
                        generator=generator[i],
                        device=device,
                        dtype=weight_dtype,
                    ).squeeze(0)
                    for i in range(batch)
                ],
                dim=0,
            )
        else:
            latents = torch.randn(latents_shape, generator=generator, device=device, dtype=weight_dtype)

        num_tokens = h_lat * w_lat
        timesteps_sched, num_inference_steps = _retrieve_timesteps(
            self.scheduler,
            num_inference_steps=num_inference_steps,
            device=device,
            timesteps=timesteps,
            num_tokens=num_tokens,
        )
        num_warmup_steps = max(len(timesteps_sched) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps_sched)

        freqs_cis = self._get_freqs_cis(device=device, dtype=weight_dtype)

        progress_bar = self.progress_bar(total=num_inference_steps) if enable_processor_bar else None
        for i, t in enumerate(timesteps_sched):
            if self._interrupt:
                continue
            self._current_timestep = t

            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            step_frac = i / max(len(timesteps_sched) - 1, 1)
            use_cfg = (cfg_range[0] <= step_frac <= cfg_range[1]) and (
                text_guidance_scale > 1.0 or image_guidance_scale > 1.0
            )

            if not use_cfg:
                optional_kwargs: Dict[str, Any] = {}
                if "ref_image_hidden_states" in inspect.signature(self.transformer.forward).parameters:
                    optional_kwargs["ref_image_hidden_states"] = ref_image_hidden_states
                model_pred = self.transformer(
                    latents, timestep, text_feats, freqs_cis, text_mask, **optional_kwargs
                )
            else:
                text_uncond = torch.zeros_like(text_feats)

                opt_kwargs_text: Dict[str, Any] = {}
                if "ref_image_hidden_states" in inspect.signature(self.transformer.forward).parameters:
                    opt_kwargs_text["ref_image_hidden_states"] = ref_image_hidden_states

                model_pred_text = self.transformer(
                    latents, timestep, text_feats, freqs_cis, text_mask, **opt_kwargs_text
                )

                opt_kwargs_ref: Dict[str, Any] = {}
                if "ref_image_hidden_states" in inspect.signature(self.transformer.forward).parameters:
                    opt_kwargs_ref["ref_image_hidden_states"] = ref_image_hidden_states

                model_pred_ref = self.transformer(
                    latents, timestep, text_uncond, freqs_cis, text_mask, **opt_kwargs_ref
                )

                opt_kwargs_uncond: Dict[str, Any] = {}
                if "ref_image_hidden_states" in inspect.signature(self.transformer.forward).parameters:
                    opt_kwargs_uncond["ref_image_hidden_states"] = None

                model_pred_uncond = self.transformer(
                    latents, timestep, text_uncond, freqs_cis, text_mask, **opt_kwargs_uncond
                )

                if text_guidance_scale > 1.0 and image_guidance_scale > 1.0:
                    model_pred = (
                        model_pred_uncond
                        + image_guidance_scale * (model_pred_ref - model_pred_uncond)
                        + text_guidance_scale * (model_pred_text - model_pred_ref)
                    )
                elif text_guidance_scale > 1.0:
                    model_pred = model_pred_uncond + text_guidance_scale * (model_pred_text - model_pred_uncond)
                elif image_guidance_scale > 1.0:
                    model_pred = model_pred_uncond + image_guidance_scale * (model_pred_ref - model_pred_uncond)
                else:
                    model_pred = model_pred_text

            latents = self.scheduler.step(model_pred, t, latents, return_dict=False)[0]
            latents = latents.to(dtype=weight_dtype)

            if progress_bar is not None:
                if i == len(timesteps_sched) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        if progress_bar is not None:
            progress_bar.close()

        self._current_timestep = None

        latents = latents.to(dtype=self.vae.dtype)
        if getattr(self.vae.config, "scaling_factor", None) is not None:
            latents = latents / self.vae.config.scaling_factor
        if getattr(self.vae.config, "shift_factor", None) is not None:
            latents = latents + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]

        images = self.image_processor.postprocess(image, output_type=output_type)
        return images

    @torch.no_grad()
    def __call__(
        self,
        *,
        encoder_hidden_states: torch.Tensor,
        grid_thw_list: List[Tuple[int, int, int]],
        image: Union[Image.Image, torch.Tensor, List[Union[Image.Image, torch.Tensor]], None] = None,
        num_inference_steps: int = 28,
        timesteps: Optional[List[int]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        text_guidance_scale: float = 1.5,
        image_guidance_scale: float = 1.5,
        cfg_range: Tuple[float, float] = (0.0, 1.0),
        enable_processor_bar: bool = True,
        **kwargs,
    ) -> Union[RefinerOutput, List[Image.Image], torch.Tensor]:
        self._interrupt = False

        token_chunks = self._split_tokens(encoder_hidden_states, grid_thw_list)
        ref_list = self._expand_to_list(image, len(token_chunks))

        results_pil: List[Image.Image] = []
        results_pt: Optional[torch.Tensor] = None

        for tok, _, img_any in zip(token_chunks, grid_thw_list, ref_list):
            imgs = self._denoise_once(
                cond_tokens=tok,
                ref_img=img_any,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                generator=generator,
                output_type=output_type,
                text_guidance_scale=text_guidance_scale,
                image_guidance_scale=image_guidance_scale,
                cfg_range=cfg_range,
                enable_processor_bar=enable_processor_bar,
            )

            if output_type == "pil":
                results_pil += imgs
            else:
                results_pt = imgs if results_pt is None else torch.cat([results_pt, imgs], dim=0)

        if not return_dict:
            return results_pil if output_type == "pil" else results_pt
        return RefinerOutput(images=results_pil if output_type == "pil" else results_pt)


def de_transform(
    tensor: torch.Tensor,
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711),
    rescale_factor: float = 1 / 255,
) -> torch.Tensor:
    """De-normalize and de-rescale, suitable for images processed by Qwen2VLImageProcessor."""
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    mean_t = torch.tensor(mean).view(1, -1, 1, 1).to(tensor.device)
    std_t = torch.tensor(std).view(1, -1, 1, 1).to(tensor.device)
    tensor = tensor * std_t + mean_t
    tensor = tensor / rescale_factor
    tensor = torch.clamp(tensor / 255.0, 0, 1)
    return tensor


def tensor2pil(image_t: torch.Tensor, image_mean, image_std) -> Image.Image:
    """Convert a tensor to a PIL Image."""
    image_t = image_t.detach().cpu()
    rescale_factor = 1 / 255
    sample = de_transform(
        image_t,
        mean=image_mean,
        std=image_std,
        rescale_factor=rescale_factor,
    )[0]
    ndarr = sample.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return Image.fromarray(ndarr)
