# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
"""Standalone Step-3.7-Flash image processor."""

from itertools import product
from math import ceil
from typing import List, Optional, Tuple

import numpy as np
import torch

from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

# Special token strings (ids resolved from the tokenizer at runtime so this
# stays robust if the vocab is ever extended). Verified against
# .original_ref/tokenizer_config.json: <im_start>=128000, <im_patch>=128001,
# <im_end>=128002, <patch_start>=128003, <patch_newline>=128004,
# <patch_end>=128005.
IMAGE_TOKEN = "<im_patch>"
IM_START = "<im_start>"
IM_END = "<im_end>"
PATCH_START = "<patch_start>"
PATCH_END = "<patch_end>"
PATCH_NEWLINE = "<patch_newline>"

MAX_IMAGE_SIZE = 3024

# Feature-count constants (== vision-tower output grids).
NUM_IMAGE_FEATURE = 169   # 13x13 from the 728x728 whole image
NUM_PATCH_FEATURE = 81    # 9x9 from each 504x504 sub-image

# Vision preprocessor sizes.
WHOLE_IMAGE_SIZE = 728
PATCH_IMAGE_SIZE = 504

_MEAN = [0.48145466, 0.4578275, 0.40821073]
_STD = [0.26862954, 0.26130258, 0.27577711]


# ---------------------------------------------------------------------------
# Pixel preprocessing (faithful port of Step3VisionProcessor, torchvision-only).
# ---------------------------------------------------------------------------
class Step3VisionProcessor:
    """Whole / patch image -> normalized, resized pixel_values tensor."""

    def __init__(self, size=WHOLE_IMAGE_SIZE, interpolation_mode="bilinear",
                 patch_size=PATCH_IMAGE_SIZE):
        interp = (InterpolationMode.BICUBIC if interpolation_mode == "bicubic"
                  else InterpolationMode.BILINEAR)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
            transforms.Resize((size, size), interpolation=interp, antialias=True),
        ])
        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
            transforms.Resize((patch_size, patch_size), interpolation=interp,
                              antialias=True),
        ]) if patch_size is not None else None

    def __call__(self, image: Image.Image, is_patch: bool = False) -> dict:
        if is_patch:
            return {"pixel_values": self.patch_transform(image).unsqueeze(0)}
        return {"pixel_values": self.transform(image).unsqueeze(0)}


# ---------------------------------------------------------------------------
# Image patcher (faithful port of ImagePatcher).
# ---------------------------------------------------------------------------
class ImagePatcher:
    @staticmethod
    def determine_window_size(long: int, short: int) -> int:
        if long <= 728:
            return short if long / short > 1.5 else 0
        return min(short, 504) if long / short > 4 else 504

    @staticmethod
    def slide_window(width, height, sizes, steps, img_rate_thr=0.6):
        if not 0 <= img_rate_thr <= 1:
            raise ValueError("`img_rate_thr` should lie in 0~1")
        windows = []
        for size, step in zip(sizes, steps):
            size_w, size_h = size
            step_w, step_h = step
            x_num = 1 if width <= size_w else ceil((width - size_w) / step_w + 1)
            x_start = [step_w * i for i in range(x_num)]
            if len(x_start) > 1 and x_start[-1] + size_w > width:
                x_start[-1] = width - size_w
            y_num = 1 if height <= size_h else ceil((height - size_h) / step_h + 1)
            y_start = [step_h * i for i in range(y_num)]
            if len(y_start) > 1 and y_start[-1] + size_h > height:
                y_start[-1] = height - size_h
            start = np.array(list(product(y_start, x_start)), dtype=int)
            start[:, [0, 1]] = start[:, [1, 0]]
            windows.append(np.concatenate([start, start + size], axis=1))
        windows = np.concatenate(windows, axis=0)
        return [(int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1]))
                for b in windows], (x_num, y_num)

    @staticmethod
    def square_pad(img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        size = max(w, h)
        padded = Image.new(img.mode, (size, size), 0)
        padded.paste(img, (0, 0))
        return padded

    @staticmethod
    def get_image_size_for_padding(w: int, h: int) -> Tuple[int, int]:
        ratio = w / h
        if min(h, w) < 32 and (ratio > 4 or ratio < 1 / 4):
            new_size = max(h, w)
            return new_size, new_size
        return w, h

    @staticmethod
    def get_image_size_for_preprocess(w: int, h: int) -> Tuple[int, int]:
        if max(h, w) > MAX_IMAGE_SIZE:
            scale = MAX_IMAGE_SIZE / max(h, w)
            w = int(w * scale)
            h = int(h * scale)
        return w, h

    @staticmethod
    def get_image_size_for_crop(w: int, h: int, window: int):
        w_ratio = w / window
        h_ratio = h / window
        if w_ratio < 1:
            width_new = w
        else:
            decimal_w = w_ratio - w // window
            w_ratio = int(w_ratio) + 1 if decimal_w > 0.2 else int(w_ratio)
            width_new = window * w_ratio
        if h_ratio < 1:
            height_new = h
        else:
            decimal_h = h_ratio - h // window
            h_ratio = int(h_ratio) + 1 if decimal_h > 0.2 else int(h_ratio)
            height_new = window * h_ratio
        return int(width_new), int(height_new)

    @staticmethod
    def patch_crop(img, i, j, th, tw):
        return img.crop((j, i, j + tw, i + th))

    def __call__(self, img: Image.Image):
        img_w, img_h = img.size
        new_w, new_h = self.get_image_size_for_padding(img_w, img_h)
        if new_w != img_w or new_h != img_h:
            img = self.square_pad(img)
            img_w, img_h = img.size
        new_w, new_h = self.get_image_size_for_preprocess(img_w, img_h)
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        window = self.determine_window_size(max(new_h, new_w), min(new_h, new_w))
        if window == 0:
            return img, [], None
        new_w, new_h = self.get_image_size_for_crop(new_w, new_h, window)
        if (new_w, new_h) != (img_w, img_h):
            img_for_crop = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        else:
            img_for_crop = img
        patches, newlines = [], []
        center_list, (x_num, _) = self.slide_window(
            new_w, new_h, [(window, window)], [(window, window)])
        for pid, (x, y, pw, ph) in enumerate(center_list):
            patches.append(self.patch_crop(img_for_crop, y, x, ph, pw))
            if (pid + 1) % x_num == 0:
                newlines.append(pid)
        if newlines and newlines[-1] == len(patches) - 1:
            newlines.pop()
        mask = ([i in newlines for i in range(len(patches))]
                if len(patches) > 0 else None)
        return img, patches, mask


# ---------------------------------------------------------------------------
# Top-level processor: text+images -> {input_ids, pixel_values,
# patch_pixel_values, num_patches}. Dependency-light, AutoTokenizer-driven.
# ---------------------------------------------------------------------------
class Step3p7Processor:
    """Framework-free port of Step3VLProcessor."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.image_preprocessor = Step3VisionProcessor(
            WHOLE_IMAGE_SIZE, "bilinear", PATCH_IMAGE_SIZE)
        self.patcher = ImagePatcher()
        self.num_image_feature_size = NUM_IMAGE_FEATURE
        self.num_patch_feature_size = NUM_PATCH_FEATURE
        self.image_token = IMAGE_TOKEN

    # ---- token-id helpers -------------------------------------------------
    def _tid(self, token: str) -> int:
        return self.tokenizer.convert_tokens_to_ids(token)

    @property
    def image_token_id(self) -> int:
        return self._tid(self.image_token)

    # ---- placeholder id sequences (mirror Step3VLProcessor) ----------------
    def _get_image_repl_ids(self) -> List[int]:
        return ([self._tid(IM_START)]
                + [self.image_token_id] * self.num_image_feature_size
                + [self._tid(IM_END)])

    def _get_patch_repl_ids(self, num_patches: int,
                            newline_mask: Optional[List[bool]]) -> List[int]:
        ids: List[int] = []
        for i in range(num_patches):
            ids += ([self._tid(PATCH_START)]
                    + [self.image_token_id] * self.num_patch_feature_size
                    + [self._tid(PATCH_END)])
            if newline_mask and newline_mask[i]:
                ids.append(self._tid(PATCH_NEWLINE))
        return ids

    def _get_image_repl_features_ids(self, num_patches: int,
                                     newline_mask) -> List[int]:
        patch_ids = (self._get_patch_repl_ids(num_patches, newline_mask)
                     if num_patches > 0 else [])
        return patch_ids + self._get_image_repl_ids()

    # ----------------------------------------------------------------------
    def __call__(self, text: str, images: Optional[List[Image.Image]] = None,
                 device="cpu", dtype=torch.bfloat16) -> dict:
        """Build model inputs for one image+text prompt.

        Args:
            text: chat-template rendered string containing one `<im_patch>`
                placeholder per image (as the chat template emits).
            images: list of PIL images, one per `<im_patch>` placeholder in
                `text`. None / [] -> pure text (no vision fields).
            device / dtype: where/what to place pixel tensors.

        Returns dict with:
            input_ids: LongTensor [T]   (1-D, ready for the unified-flow model)
            pixel_values: FloatTensor [N_img, 3, 728, 728]   (or absent)
            patch_pixel_values: FloatTensor [N_sub, 3, 504, 504]  (or absent)
            num_patches: list[int]  (#sub-images per image)
            image_token_id: int
        """
        images = images or []
        if not images:
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            return {
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "num_patches": [],
                "image_token_id": self.image_token_id,
            }

        # ----- per-image patching + pixel tensors -----
        pixel_values_lst = []
        patch_pixel_values_lst = []
        num_patches = []
        repl_ids_per_image = []
        for img in images:
            img = img.convert("RGB")
            whole_img, sub_patches, newline_mask = self.patcher(img)
            pixel_values_lst.append(
                self.image_preprocessor(whole_img, is_patch=False)["pixel_values"])
            for sub in sub_patches:
                patch_pixel_values_lst.append(
                    self.image_preprocessor(sub, is_patch=True)["pixel_values"])
            num_patches.append(len(sub_patches))
            repl_ids_per_image.append(
                self._get_image_repl_features_ids(len(sub_patches), newline_mask))

        pixel_values = torch.cat(pixel_values_lst).to(device).to(dtype)
        patch_pixel_values = (torch.cat(patch_pixel_values_lst).to(device).to(dtype)
                              if patch_pixel_values_lst else None)

        # ----- text-side placeholder expansion (id-level) -----
        # Tokenize the chat-template text, then splice each single image_token id
        # with the full per-image replacement id sequence (preserving order).
        base_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        img_tid = self.image_token_id
        n_placeholders = base_ids.count(img_tid)
        if n_placeholders != len(images):
            raise ValueError(
                f"text has {n_placeholders} `{self.image_token}` placeholders "
                f"but {len(images)} images were provided")
        out_ids: List[int] = []
        img_idx = 0
        for tid in base_ids:
            if tid == img_tid:
                out_ids.extend(repl_ids_per_image[img_idx])
                img_idx += 1
            else:
                out_ids.append(tid)

        result = {
            "input_ids": torch.tensor(out_ids, dtype=torch.long),
            "pixel_values": pixel_values,
            "num_patches": num_patches,
            "image_token_id": img_tid,
        }
        if patch_pixel_values is not None:
            result["patch_pixel_values"] = patch_pixel_values
        return result


__all__ = ["Step3p7Processor", "Step3VisionProcessor", "ImagePatcher",
           "IMAGE_TOKEN", "NUM_IMAGE_FEATURE", "NUM_PATCH_FEATURE"]
