# coding=utf-8
# Adapted from
# https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/weights.py
#
# Copyright (c) 2025 OpenAI. All rights reserved.
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

import logging
import math
import os
import json
import shutil
import sys
import argparse  # <-- added back

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------------------------------------------------------
# This Checkpoint class, particularly the __init__, get_tensor, and
# get_mxfp4_tensor methods, is adapted from:
# https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/weights.py
# which is distributed under the Apache License, Version 2.0.
#
# Modifications made:
# - Added tensor transpose (permute) at the end of get_mxfp4_tensor
#   to adapt shape for the current framework's weight layout.
# - Integrated complete shard saving and index update logic.
# - Converted private methods to public (removed leading underscore)
#   and replaced asserts with explicit exception raising.
# --------------------------------------------------------------------


class Checkpoint:
    def __init__(self, path: str, device: torch.device):
        device_str = (
            device.type
            if device.index is None
            else device.type + ":" + str(device.index)
        )
        self.device_str = device_str

        safetensor_files = [
            os.path.join(path, fname)
            for fname in os.listdir(path)
            if fname.endswith(".safetensors")
        ]
        tensor_name_to_file = {}
        for safetensor_file in safetensor_files:
            with safe_open(safetensor_file, framework="pt", device=device_str) as f:
                for key in f.keys():
                    tensor_name_to_file[key] = safetensor_file
        self.tensor_name_to_file = tensor_name_to_file

    def get_tensor(self, name: str) -> torch.Tensor:
        """Retrieve a tensor by name from the checkpoint."""
        if name not in self.tensor_name_to_file:
            raise KeyError(f"Tensor {name} not found.")
        with safe_open(
            self.tensor_name_to_file[name], framework="pt", device=self.device_str
        ) as f:
            return f.get_tensor(name)

    def get_mxfp4_tensor(
        self,
        blocks_name: str,
        scales_name: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        # After flattening the 4D tensor into [rows, block_size], memory usage
        # doubles during conversion, so we process in chunks for safety.
        # rows_per_chunk is the number of rows processed per chunk;
        # 16384*512 is an empirical value from openai/gpt-oss.
        rows_per_chunk: int = 16384 * 512,
    ) -> torch.Tensor:
        """Dequantize MXFP4 blocks and scales to a full-precision tensor."""
        if blocks_name not in self.tensor_name_to_file:
            raise KeyError(f"Blocks {blocks_name} not found.")
        if scales_name not in self.tensor_name_to_file:
            raise KeyError(f"Scales {scales_name} not found.")

        blocks = self.get_tensor(blocks_name)
        scales = self.get_tensor(scales_name).to(torch.int32) - 127

        if blocks.shape[:-1] != scales.shape:
            raise RuntimeError(f"Shape mismatch: {blocks.shape=} != {scales.shape=}")

        lut = torch.tensor([
            +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ], dtype=dtype, device=blocks.device)

        # Use descriptive variable names
        *prefix_shape, num_groups, block_size = blocks.shape
        rows_total = math.prod(prefix_shape) * num_groups

        blocks = blocks.reshape(rows_total, block_size)
        scales = scales.reshape(rows_total, 1)

        out = torch.empty(rows_total, block_size * 2, dtype=dtype, device=blocks.device)

        for r0 in range(0, rows_total, rows_per_chunk):
            r1 = min(r0 + rows_per_chunk, rows_total)
            blk = blocks[r0:r1]
            exp = scales[r0:r1]
            idx_lo = (blk & 0x0F).to(torch.long)
            idx_hi = (blk >> 4).to(torch.long)
            sub = out[r0:r1]
            sub[:, 0::2] = lut[idx_lo]
            sub[:, 1::2] = lut[idx_hi]
            torch.ldexp(sub, exp, out=sub)
            del idx_lo, idx_hi, blk, exp

        # Transpose to match target framework's weight shape requirement.
        out = out.reshape(*prefix_shape, num_groups, block_size * 2).view(*prefix_shape, num_groups * block_size * 2)
        out = out.permute(0, 2, 1).contiguous()
        return out.to(dtype)


# Find quantized weight pairs: block data and corresponding shared scale tensors.
def build_param_map_from_index(index_path: str):
    with open(index_path, 'r') as f:
        index_data = json.load(f)
    weight_map = index_data['weight_map']
    param_map = {}
    for name in weight_map.keys():
        if name.endswith('_blocks'):
            base_name = name[:-7]
            scales_name = base_name + '_scales'
            if scales_name in weight_map:
                param_map[base_name] = (name, scales_name)
                continue
            else:
                raise ValueError(
                    f"Found _blocks tensor '{name}' but corresponding "
                    f"_scales tensor '{scales_name}' is missing in weight_map. "
                    "Please check if the checkpoint is complete or if the naming "
                    "convention has changed."
                )
        elif name.endswith('_scales'):
            continue
        if name not in param_map:
            param_map[name] = name
    return param_map


# Clean config by removing quantization-related fields and setting dtype to bfloat16.
def clean_config(config_path: str, output_path: str):
    with open(config_path, 'r') as f:
        config = json.load(f)
    config.pop('quantization_config', None)
    config.pop('_quantization_config', None)
    config['torch_dtype'] = 'bfloat16'
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)


def convert_and_save(checkpoint_dir: str, output_dir: str, device: torch.device = torch.device("cpu")):
    os.makedirs(output_dir, exist_ok=True)

    # Copy and clean config
    src_config = os.path.join(checkpoint_dir, "config.json")
    if not os.path.exists(src_config):
        raise FileNotFoundError(f"config.json not found in {checkpoint_dir}")
    dst_config = os.path.join(output_dir, "config.json")
    clean_config(src_config, dst_config)

    # Copy index file
    src_index = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if not os.path.exists(src_index):
        raise FileNotFoundError(f"model.safetensors.index.json not found in {checkpoint_dir}")
    shutil.copy(src_index, os.path.join(output_dir, "model.safetensors.index.json"))

    # Copy tokenizer and other aux files
    for fname in [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "generation_config.json",
        "chat_template.jinja",
    ]:
        src = os.path.join(checkpoint_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_dir, fname))

    ckpt = Checkpoint(checkpoint_dir, device)
    param_map = build_param_map_from_index(src_index)
    logging.info(f"Total parameters to process: {len(param_map)}")

    max_shard_size = 5 * 1024**3  # 5 GiB per shard
    shard_idx = 1
    current_shard = {}
    current_size = 0
    new_weight_map = {}

    param_names = sorted(param_map.keys())

    for param_name in tqdm(param_names, desc="Processing tensors"):
        mapping = param_map[param_name]
        if isinstance(mapping, tuple):
            blocks_name, scales_name = mapping
            tensor = ckpt.get_mxfp4_tensor(blocks_name, scales_name, dtype=torch.bfloat16)
        else:
            tensor = ckpt.get_tensor(mapping)
        # Non-quantized tensors are not type-casted.

        tensor_size = tensor.numel() * tensor.element_size()

        if current_size + tensor_size > max_shard_size and current_shard:
            tmp_name = f"model-{shard_idx:05d}-tmp.safetensors"
            save_file(current_shard, os.path.join(output_dir, tmp_name), metadata={"format": "pt"})
            for name in current_shard:
                new_weight_map[name] = tmp_name
            shard_idx += 1
            current_shard = {}
            current_size = 0

        current_shard[param_name] = tensor
        current_size += tensor_size

    if current_shard:
        tmp_name = f"model-{shard_idx:05d}-tmp.safetensors"
        save_file(current_shard, os.path.join(output_dir, tmp_name), metadata={"format": "pt"})
        for name in current_shard:
            new_weight_map[name] = tmp_name

        # Rename based on new_weight_map to avoid including leftover files.
        saved_tmp_files = sorted(set(new_weight_map.values()))
        total_shards = len(saved_tmp_files)
        final_weight_map = {}
        for idx, tmp_name in enumerate(saved_tmp_files):
            new_name = f"model-{idx:05d}-of-{total_shards:05d}.safetensors"
            os.rename(os.path.join(output_dir, tmp_name), os.path.join(output_dir, new_name))
            for key, val in new_weight_map.items():
                if val == tmp_name:
                    final_weight_map[key] = new_name

        # Update index with new weight map and total size
        new_index_path = os.path.join(output_dir, "model.safetensors.index.json")
        with open(new_index_path, 'r') as f:
            index_data = json.load(f)
        index_data['weight_map'] = final_weight_map
        total_size = 0
        for shard_file in set(final_weight_map.values()):
            shard_path = os.path.join(output_dir, shard_file)
            total_size += os.path.getsize(shard_path)
        index_data['metadata']['total_size'] = total_size
        with open(new_index_path, 'w') as f:
            json.dump(index_data, f, indent=2)

        logging.info(f"Conversion complete! Model saved to {output_dir}")
        logging.info(f"Total size: {total_size / 1024**3:.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MXFP4 quantized weights to BF16 safetensors.")
    parser.add_argument("--input", type=str, required=True,
                        help="Input directory where mxfp4 weights will be saved")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory where converted BF16 weights will be saved")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use for conversion (default: cpu)")
    args = parser.parse_args()

    device = torch.device(args.device)
    convert_and_save(args.input, args.output, device)