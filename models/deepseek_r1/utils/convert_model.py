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
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file
import shutil


def weight_dequant(weight: torch.Tensor, scale: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor, efficiently handling cases where
    `weight` is not a multiple of `block_size` by broadcasting `scale`.

    Args:
        weight (torch.Tensor): The quantized weight tensor of shape(M, N).
        scale (torch.Tensor): The scale tensor of shape (M // block_size, N // block_size).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `weight`, converted to the default dtype.

    Raises:
        AssertionError: If `scale` dimensions do not align with `weight` shape after scaling.
    """

    # Get the original dimensions of weight
    M, N = weight.shape

    # Compute the effective block dimensions for scale
    scale_m, scale_n = scale.shape
    assert scale_m == (
        M + block_size - 1) // block_size, "Mismatch in scale rows and weight rows."
    assert scale_n == (
        N + block_size - 1) // block_size, "Mismatch in scale columns and weight columns."

    # Convert weight to float32 for calculations
    weight = weight.to(torch.float32)

    # Expand scale to match the weight tensor's shape
    scale_expanded = scale.repeat_interleave(
        block_size, dim=0).repeat_interleave(block_size, dim=1)

    # Trim scale_expanded to match weight's shape if necessary
    scale_expanded = scale_expanded[:M, :N]

    # Perform element-wise multiplication
    dequantized_weight = weight * scale_expanded

    # Convert the output to the default dtype
    dequantized_weight = dequantized_weight.to(torch.get_default_dtype())

    return dequantized_weight


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


def generate_ignore_item(num_layers, num_hidden_layers):
    """
    Generate a list of layer names to be ignored during quantization.
    """
    ignore = []
    for i in range(0, num_layers):
        ignore.append(f'model.layers.{i}.self_attn.kv_b_proj')
        if i >= 3:
            ignore.append(f'model.layers.{i}.mlp.gate')
        if i >= num_hidden_layers:
            ignore.append(f'model.layers.{i}.eh_proj')
            ignore.append(f'model.layers.{i}.shared_head.head')
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


def generate_quant_config(c8, num_layers, num_hidden_layers):
    ignores = generate_ignore_item(num_layers=num_layers, num_hidden_layers=num_hidden_layers)
    if c8:
        kv_cache_scheme = {"num_bits": 8, "type": "int", "strategy": "tensor", "dynamic": False, "symmetric": True}
    else:
        kv_cache_scheme = None
    quant_config = {"config_groups": {"group_0": {}}, "format": "int-quantized",
                    "global_compression_ratio": 1, "ignore": ignores, "kv_cache_scheme": kv_cache_scheme,
                    "quant_method": "compressed-tensors", "quantization_status": "compressed"}
    targets = ["Linear"]
    quant_config["config_groups"]["group_0"] = generate_quant_group(
        a_num_bits=8, w_num_bits=8, targets=targets)
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


def load_c8_quant_params(num_hidden_layers, clip_param_path):
    kv_quant_params = {}
    quant_param_files = list(glob(os.path.join(clip_param_path, "*.pth")))
    quant_param_files.sort()
    for layer_idx in range(0, num_hidden_layers):
        expected_file = os.path.join(
            clip_param_path, f'quant_parameters_{layer_idx}.pth')
        if not os.path.exists(expected_file):
            raise ValueError(
                f"{expected_file} not found, please check the {clip_param_path}")
        else:
            quant_params = torch.load(expected_file)
        for name, factor in quant_params.items():
            complete_name = f"model.layers.{layer_idx}.{name}"
            if complete_name.endswith("_scale"):
                kv_quant_params[complete_name] = factor
    return kv_quant_params


def main(fp8_path, output_path, w8a8, c8=False, quant_param_path=None):
    """
    Converts FP8 weights to BF16 and saves the converted weights.

    This function reads FP8 weights from the specified directory, converts them to BF16,
    and saves the converted weights to another specified directory. It also updates the
    model index file to reflect the changes.

    Args:
    fp8_path (str): The path to the directory containing the FP8 weights and model index file.
    output_path (str): The path to the directory where the converted BF16/INT8 weights will be saved.
    w8a8 (bool): Quantize the model to INT8.
    c8 (bool): Use W8A8C8 quantization scheme if True, otherwise use W8A8C16.
    clip (bool): Whether to use clipping parameters during quantization.
    quant_param_path (str, optional): The path to the directory containing quantization parameters.

    Raises:
    KeyError: If a required scale_inv tensor is missing for a weight.

    Notes:
    - The function assumes that the FP8 weights are stored in safetensor files.
    - The function caches loaded safetensor files to optimize memory usage.
    - The function updates the model index file to remove references to scale_inv tensors.
    """
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(output_path, exist_ok=True)
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    config_file = os.path.join(fp8_path, 'config.json')
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    with open(config_file, "r") as f:
        config = json.load(f)
    if 'quantization_config' in config:
        config.pop('quantization_config')

    weight_map = model_index["weight_map"]
    new_weight_map = {}
    num_hidden_layers = config['num_hidden_layers']
    num_nextn_predict_layers = config['num_nextn_predict_layers']
    num_layers = num_hidden_layers + num_nextn_predict_layers
    quant_ignore_layers = []
    if w8a8:
        quantization_config = generate_quant_config(c8, num_layers, num_hidden_layers)
        config['quantization_config'] = quantization_config
        quant_ignore_layers = generate_ignore_item(num_layers, num_hidden_layers)

    # Cache for loaded safetensor files
    loaded_files = {}

    # Helper function to get tensor from the correct file

    def get_tensor(tensor_name):
        """
        Retrieves a tensor from the cached safetensor files or loads it from disk if not cached.

        Args:
            tensor_name (str): The name of the tensor to retrieve.

        Returns:
            torch.Tensor: The retrieved tensor.

        Raises:
            KeyError: If the tensor does not exist in the safetensor file.
        """
        file_name = weight_map[tensor_name]
        if file_name not in loaded_files:
            file_path = os.path.join(fp8_path, file_name)
            loaded_files[file_name] = load_file(file_path, device="cpu")
        return loaded_files[file_name][tensor_name]

    if c8:
        assert quant_param_path is not None, "Please pass the quant_param_path"
        kv_quant_params = load_c8_quant_params(num_layers, quant_param_path)

    safetensor_files = list(glob(os.path.join(fp8_path, "*.safetensors")))
    safetensor_files.sort()
    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cpu")
        loaded_files[file_name] = current_state_dict
        new_state_dict = {}
        for weight_name, weight in current_state_dict.items():
            if weight_name.endswith("_scale_inv"):
                continue
            elif weight.element_size() == 1:
                # FP8 weight
                scale_inv_name = f"{weight_name}_scale_inv"
                try:
                    # Get scale_inv from the correct file
                    scale_inv = get_tensor(scale_inv_name)
                    bf16_weight = weight_dequant(weight, scale_inv)
                    if w8a8:
                        is_ignore_layer = is_ignore_quant(
                            weight_name, quant_ignore_layers)
                        if not is_ignore_layer:
                            weight_clip_factor = None
                            int8_weight, scale_inv = int_weight_quant(
                                bf16_weight, weight_clip_factor=weight_clip_factor)
                            new_scale_name = scale_inv_name.replace(
                                '_scale_inv', '_scale')

                            new_state_dict[weight_name] = int8_weight
                            new_state_dict[new_scale_name] = scale_inv

                            new_weight_map[weight_name] = file_name
                            new_weight_map[new_scale_name] = file_name
                        else:
                            new_state_dict[weight_name] = bf16_weight
                            new_weight_map[weight_name] = file_name
                    else:
                        new_state_dict[weight_name] = bf16_weight
                        new_weight_map[weight_name] = file_name
                except KeyError:
                    print(
                        f"Warning: Missing scale_inv tensor for {weight_name}, skipping conversion")
                    new_state_dict[weight_name] = weight
                    new_weight_map[weight_name] = file_name
            else:
                new_state_dict[weight_name] = weight
                new_weight_map[weight_name] = file_name

        new_safetensor_file = os.path.join(output_path, file_name)
        save_file(new_state_dict, new_safetensor_file,
                  metadata={'format': 'pt'})

        # Memory management: keep only the 2 most recently used files
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]

    if c8:
        safetensor_files = list(
            glob(os.path.join(output_path, "*.safetensors")))
        safetensor_files.sort()
        last_safetensor_file = safetensor_files[-1]
        last_safetensor_dict = load_file(last_safetensor_file, device="cpu")
        last_safetensor_dict.update(kv_quant_params)
        last_file_name = os.path.basename(safetensor_file)
        for weight_name in kv_quant_params.keys():
            new_weight_map[weight_name] = last_file_name

        new_safetensor_file = os.path.join(output_path, last_file_name)
        save_file(last_safetensor_dict, new_safetensor_file,
                  metadata={'format': 'pt'})

    copy_py_json(fp8_path, output_path)

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
    parser.add_argument("--input_fp8_hf_path", type=str, required=True)
    parser.add_argument("--output_hf_path", type=str, required=True)
    parser.add_argument("--w8a8", action='store_true')
    parser.add_argument("--c8", action='store_true')
    parser.add_argument("--quant_param_path", type=str, default=None)
    args = parser.parse_args()

    main(args.input_fp8_hf_path, args.output_hf_path,
         args.w8a8, args.c8, args.quant_param_path)
