# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Copyright (c) 2023 DeepSeek
# Upstream portions are licensed under MIT; see ../LICENSE_VISION.txt.
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

"""Image preprocessing.

An image becomes a `n_vit_h x n_vit_w` patch grid for the ViT and a `n_llm_h x n_llm_w` token grid
after the 3x3 aligner downsample, which the LLM sees as

    [IMAGE_START] + ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_END]

Every one of those positions carries `image_token_id` in `input_ids`; only the token type tells them
apart. The IMAGE slots are filled with aligner rows in reading order.
"""

import base64
import io
import math
import json
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
import numpy as np
import torch
from PIL import Image, ImageOps

from executor.core.mm_processor import BaseMMProcessor
from .encoding_dsv4 import IMAGE_PLACEHOLDER, process_image_messages

TEXT = -1
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(best_height: int, best_width: int, patch_size: int, downsample_ratio: int):
    """Token grid the aligner produces from a patch grid of this pixel size."""
    return math.ceil((best_height // patch_size) / downsample_ratio), math.ceil(
        (best_width // patch_size) / downsample_ratio
    )


def solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    """Largest aspect-preserving pixel size whose token grid still fits in max_n_token."""
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:  # very tall: collapse to a single column
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:  # very wide: collapse to a single row
        return cell, (max_n_token - 3) * cell
    beta = min(math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height)
    return math.floor(height * beta / patch_size) * patch_size, math.floor(width * beta / patch_size) * patch_size


def safe_resize(height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token):
    """Shrink the pixel size until the image costs at most max_n_token LLM tokens."""
    n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token)
        n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch_size, downsample_ratio)
        assert num_image_tokens(n_llm_h, n_llm_w) <= max_n_token
    return n_llm_h, n_llm_w, best_height, best_width


def image_token_types(n_llm_h: int, n_llm_w: int) -> torch.Tensor:
    """Default layout: the aligner grid in reading order, one IMAGE_NEW_LINE per row."""
    types = [IMAGE_START]
    types += ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
    types.append(IMAGE_END)
    return torch.tensor(types, dtype=torch.int64)

def plan_image_grid(
        width: int,
        height: int,
        patch_size: int,
        downsample_ratio: int,
        max_n_token: int,
        min_pixels: int,
        max_wh_ratio: int | None,
        ):
    """Resize plan for an image of the given original size; a pure function of its arguments."""
    if max_wh_ratio is not None and width > height * max_wh_ratio:
        width = height * max_wh_ratio
    if 0 < width * height < min_pixels:
        ratio = (min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / patch_size) * patch_size
    best_height = math.ceil(height / patch_size) * patch_size
    return safe_resize(height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token)

class DeepseekV41ImageProcessor(BaseMMProcessor):
    """Expand image placeholders and materialize vision inputs."""

    def __init__(self, model_path, tokenizer):
        super().__init__(model_path, tokenizer)
        with (Path(model_path) / "config.json").open(encoding="utf-8") as file:
            config = json.load(file)
        self.image_token_id = int(config["image_token_id"])

        vision = config["vision_config"]
        self.vision_patch_size = int(vision["patch_size"])
        self.vision_downsample_ratio = int(vision["downsample_ratio"])
        self.vision_max_n_token = int(vision["max_num_tokens"])
        self.vision_min_pixels = int(vision["min_pixels"])
        self.vision_max_wh_ratio = vision.get("max_wh_ratio")
        self.vision_enabled = int(vision["num_hidden_layers"]) > 0

        placeholder_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
        if placeholder_id is not None and placeholder_id != tokenizer.unk_token_id:
            assert placeholder_id == self.image_token_id, (placeholder_id, self.image_token_id)
        self.input_truncated_len = None

    def should_defer_truncation(self, prompt) -> bool:
        return isinstance(prompt, list) and any(
            part.get("type") in ("image", "image_url")
            for message in prompt
            if isinstance(message.get("content"), list)
            for part in message["content"]
        )

    @staticmethod
    def _load_image(record):
        """Load image bytes from raw/base64 data, an Anthropic source, URL, or path."""

        def load_bytes(record) -> bytes:
            data = record.get("data")
            if isinstance(data, bytes):
                return data
            if isinstance(data, str):
                return base64.b64decode(data)

            source = record.get("source")
            if isinstance(source, dict):
                if source.get("data") is not None:
                    return base64.b64decode(source["data"])
                if source.get("url"):
                    return load_bytes({"url": source["url"]})

            url = record.get("url")
            if isinstance(url, str) and url:
                if url.startswith("data:"):
                    header, _, payload = url.partition(",")
                    if ";base64" not in header:
                        raise ValueError(f"Unsupported data URL encoding: {header}")
                    return base64.b64decode(payload)
                if url.startswith(("http://", "https://")):
                    with urlopen(url, timeout=30) as response:
                        return response.read()
                if url.startswith("file://"):
                    parsed = urlparse(url)
                    if parsed.netloc not in ("", "localhost"):
                        raise ValueError(f"Unsupported file URL host: {parsed.netloc}")
                    with open(unquote(parsed.path), "rb") as file:
                        return file.read()
                with open(url, "rb") as file:
                    return file.read()

            raise ValueError(f"Cannot load image from record: {list(record.keys())}")
        with Image.open(io.BytesIO(load_bytes(record))) as source:
            return source.convert("RGB")

    def _process_image(self, record):
        """Load and transform one image record into ViT patches."""

        p = self.vision_patch_size
        image = self._load_image(record)
        n_llm_h, n_llm_w, best_height, best_width = plan_image_grid(
            image.width,image.height, p,
            self.vision_downsample_ratio, self.vision_max_n_token,
            self.vision_min_pixels, self.vision_max_wh_ratio,
        )
        n_vit_h, n_vit_w = best_height // p, best_width // p
        if self.vision_max_wh_ratio is not None and image.width >= self.vision_max_wh_ratio * image.height:
            image = image.resize((best_width, best_height))
        else:
            image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
        x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
        x = ((x - 0.5) / 0.5).to(torch.bfloat16)
        patches = x.reshape(3, n_vit_h, p, n_vit_w, p).permute(1, 3, 0, 2, 4).reshape(n_vit_h * n_vit_w, 3, p, p)
        return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w

    def process(self, request) -> None:
        _, image_records = (
            process_image_messages(request.prompt)
            if isinstance(request.prompt, list)
            else (None, [])
        )
        if not image_records:
            request.mm_inputs = None
            request.mm_token_count = 0
            return

        input_ids = request.input_ids.reshape(-1).tolist()
        placeholder_count = sum(token == self.image_token_id for token in input_ids)
        if placeholder_count != len(image_records):
            raise ValueError(
                f"Found {placeholder_count} image tokens but got {len(image_records)} images"
            )
        if not self.vision_enabled:
            raise ValueError("The model config has no vision tower (vision_n_layers == 0) but the prompt contains images")

        tokens = []
        images = []
        records = iter(image_records)
        max_length = self.input_truncated_len
        for token in input_ids:
            if max_length is not None and len(tokens) >= max_length:
                break
            if token != self.image_token_id:
                tokens.append(token)
                continue

            patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = self._process_image(
                next(records)
            )
            types = image_token_types(n_llm_h, n_llm_w)
            if max_length is not None and len(tokens) + types.numel() > max_length:
                break
            images.append(
                {
                    "patches": patches,
                    "n_vit_h": n_vit_h,
                    "n_vit_w": n_vit_w,
                    "types": types,
                }
            )
            tokens.extend([self.image_token_id] * types.numel())

        request.input_ids = torch.tensor(tokens, dtype=torch.long)
        request.mm_inputs = {"images": images} if images else None
        request.mm_token_count = sum(image["types"].numel() for image in images)
