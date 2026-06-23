# coding=utf-8
# Adapted from
# https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/MindIE/LLM/DeepSeek/DeepSeek-V2/NPU_inference/fp8_cast_bf16.py
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

import math
import os
import json
import shutil
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file


def int_weight_quant(tensor: torch.Tensor, bits=8, weight_clip_factor=None):
    assert tensor.dim() == 2
    qmax = 2 ** (bits - 1) - 1
    abs_max = torch.abs(tensor).max(dim=1, keepdim=True)[0]
    if weight_clip_factor is not None:
        abs_max = abs_max * weight_clip_factor
    scale = abs_max / qmax
    assert scale.shape == (tensor.shape[0], 1)
    quantized = torch.round(tensor / scale)
    quantized = torch.clamp(quantized, -qmax, qmax)
    return quantized.to(torch.int8), scale.to(torch.float32)


def is_ignore_quant(weight_name, quant_ignore_layers_name):
    """
    Check if a layer should be ignored during quantization based on its name.
    """
    is_ignore = False
    for layer in quant_ignore_layers_name:
        if layer in weight_name:
            is_ignore = True
            print(f'Ignore quantization {weight_name}')
            break
    return is_ignore


def generate_ignore_item(num_layers):
    """
    Generate a list of layer names to be ignored during quantization.
    """
    ignore = []
    for i in range(0, num_layers):
        for j in range(2):
            ignore.append(f'model.layers.{i}.self_attn.{j}.kv_b_proj')
            ignore.append(f'model.layers.{i}.self_attn.{j}.kv_a_proj_with_mqa')
            ignore.append(f'model.layers.{i}.self_attn.{j}.o_proj')
            ignore.append(f'model.layers.{i}.self_attn.{j}.q_a_proj')
            ignore.append(f'model.layers.{i}.self_attn.{j}.q_b_proj')
        ignore.append(f'model.layers.{i}.mlp.router.classifier')
    ignore.append(f'model.mtp.layers.0.eh_proj')
    ignore.append(f'model.mtp.layers.0.self_attn.kv_a_proj_with_mqa')
    ignore.append(f'model.mtp.layers.0.self_attn.kv_b_proj')
    ignore.append(f'model.mtp.layers.0.self_attn.o_proj')
    ignore.append(f'model.mtp.layers.0.self_attn.q_a_proj')
    ignore.append(f'model.mtp.layers.0.self_attn.q_b_proj')
    ignore.append('lm_head')
    return ignore


def generate_quant_group(a_num_bits=8, w_num_bits=8, targets=None):
    quant_group = {"input_activations": {"actorder": None, "block_structure": None, "dynamic": True,
                                         "group_size": None, "num_bits": a_num_bits,
                                         "observer": "memoryless", "observer_kwargs": {},
                                         "strategy": "token", "symmetric": True, "type": "int"},
                   "output_activations": None,
                   "targets": targets,
                   "weights": {"actorder": None, "block_structure": None, "dynamic": False,
                               "group_size": None, "num_bits": w_num_bits,
                               "observer": "minmax", "observer_kwargs": {},
                               "strategy": "channel", "symmetric": True, "type": "int"}}
    return quant_group


def generate_quant_config(num_layers):
    ignores = generate_ignore_item(num_layers=num_layers)
    kv_cache_scheme = None
    quant_config = {"config_groups": {"group_0": {}}, "format": "int-quantized",
                    "global_compression_ratio": 1, "ignore": ignores, "kv_cache_scheme": kv_cache_scheme,
                    "quant_method": "compressed-tensors", "quantization_status": "compressed"}
    targets = ["Linear"]
    quant_config["config_groups"]["group_0"] = generate_quant_group(a_num_bits=8, w_num_bits=8, targets=targets)
    return quant_config


def copy_py_json(src, target):
    for root, _, files in os.walk(src):
        for file in files:
            if file.endswith(('.py', '.json')):
                src_path = os.path.join(root, file)
                rel_dir = os.path.relpath(root, src)
                dst_dir = os.path.join(target, rel_dir)
                os.makedirs(dst_dir, exist_ok=True)
                dst_path = os.path.join(dst_dir, file)
                shutil.copy2(src_path, dst_path)


def main(bf16_path, output_path):
    """
    Quantize the model to INT8 (moe and mlps).

    This function reads BF16 weights from the specified directory, converts them to a8w8,
    and saves the converted weights to another specified directory. It also updates the
    model index file to reflect the changes.

    Args:
    bf16_path (str): The path to the directory containing the BF16 weights and model index file.
    output_path (str): The path to the directory where the converted BF16/INT8 weights will be saved.

    Notes:
    - The function assumes that the BF16 weights are stored in safetensor files.
    - The function caches loaded safetensor files to optimize memory usage.
    - The function updates the model index file to remove references to scale_inv tensors.
    """
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(output_path, exist_ok=True)
    config_file = os.path.join(bf16_path, 'config.json')
    with open(config_file, "r") as f:
        config = json.load(f)
    if 'quantization_config' in config:
        config.pop('quantization_config')

    new_weight_map = {}
    num_layers = config['num_layers']
    quantization_config = generate_quant_config(num_layers)
    config['quantization_config'] = quantization_config

    # Cache for loaded safetensor files
    loaded_files = {}

    total_quant_layers = 0
    safetensor_files = list(glob(os.path.join(bf16_path, "*.safetensors")))
    safetensor_files.sort()
    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cpu")
        loaded_files[file_name] = current_state_dict
        new_state_dict = {}
        for weight_name, weight in current_state_dict.items():
            if ".mlp" not in weight_name:
                new_state_dict[weight_name] = weight
                new_weight_map[weight_name] = file_name
            else:
                if "router" in weight_name:
                    new_state_dict[weight_name] = weight
                    new_weight_map[weight_name] = file_name
                else:
                    total_quant_layers += 1
                    scale_name = f"{weight_name}_scale"
                    int8_weight, scale_inv = int_weight_quant(weight, weight_clip_factor=None)
                    new_state_dict[weight_name] = int8_weight
                    new_state_dict[scale_name] = scale_inv

                    new_weight_map[weight_name] = file_name
                    new_weight_map[scale_name] = file_name

        new_safetensor_file = os.path.join(output_path, file_name)
        save_file(new_state_dict, new_safetensor_file, metadata={'format': 'pt'})

        # Memory management: keep only the 2 most recently used files
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
    print(f"total_quant_layers is {total_quant_layers}")

    copy_py_json(bf16_path, output_path)

    # Update model index
    new_model_index_file = os.path.join(
        output_path, "model.safetensors.index.json")
    new_config_file = os.path.join(output_path, "config.json")
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": new_weight_map}, f, indent=2)

    with open(new_config_file, "w") as f:
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_bf16_hf_path", type=str, required=True)
    parser.add_argument("--output_hf_path", type=str, required=True)
    args = parser.parse_args()

    main(args.input_bf16_hf_path, args.output_hf_path)