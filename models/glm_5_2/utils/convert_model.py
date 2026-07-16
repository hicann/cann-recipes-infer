# coding=utf-8
# Adapted from
# https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/MindIE/LLM/DeepSeek/DeepSeek-V2/NPU_inference/fp8_cast_bf16.py
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

import json
import math
import os
import shutil
from argparse import ArgumentParser
from glob import glob

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from mx_quantize import quantize_mx, pack_uint4, unpack_uint4, f4_unpacked_to_f32, f32_to_f4_unpacked


NUM_BITS_4 = 4
NUM_BITS_8 = 8


def get_had_pow2(n, norm=True):
    """
    Generate Hadamard matrix of size n x n, where n is a power of 2.
    If norm is True, the matrix is normalized by dividing each element by sqrt(2), had @ had.T = I
    """
    if not ((n & (n - 1) == 0) and (n > 0)):
        raise ValueError(f"n must be a positive power of 2, got{n}")
    had = torch.ones(1, 1)
    while had.shape[0] != n:
        had = torch.cat((torch.cat([had, had], 1),
                        torch.cat([had, -had], 1)), 0)
        if norm:
            had /= math.sqrt(2)
    return had


def weight_dequant(weight: torch.Tensor, scale: torch.Tensor, 
                   block_size: int = 128, is_mx: bool = False) -> torch.Tensor:
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
    weight = weight.to(torch.float32)
    scale = scale.to(torch.float32)

    if is_mx:
        scale_expanded = scale.repeat_interleave(block_size, dim=1)
    else:
        # Compute the effective block dimensions for scale
        scale_m, scale_n = scale.shape
        assert scale_m == (
            M + block_size - 1) // block_size, "Mismatch in scale rows and weight rows."
        assert scale_n == (
            N + block_size - 1) // block_size, "Mismatch in scale columns and weight columns."

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


def unpack_mxfloat4_to_fp32(packed_tensor):
    e2m1_values = torch.tensor([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
    ], dtype=torch.float32, device=packed_tensor.device)

    low_4bits = packed_tensor & 0x0F
    high_4bits = (packed_tensor // 16) & 0x0F

    unpacked = torch.stack([low_4bits, high_4bits], dim=-1)

    fp32_tensor = e2m1_values[unpacked.long()]
    new_shape = list(packed_tensor.shape)
    new_shape[-1] = new_shape[-1] * 2

    return fp32_tensor.view(*new_shape)


def scale_fp32_to_u64(weight_scale):
    """
    Convert FP32 scale to UINT64 scale for W4A8 MoEGMM.
    """
    k, n = weight_scale.shape
    scale_np = weight_scale.float().cpu().numpy()
    scale_uint32 = scale_np.astype(np.float32)
    scale_uint32.dtype = np.uint32
    scale_uint64 = np.zeros((k, n * 2), dtype=np.uint32)
    scale_uint64[..., ::2] = scale_uint32
    scale_uint64.dtype = np.uint64
    scale_uint64 = torch.from_numpy(scale_uint64).to(torch.uint64)
    return scale_uint64


def pack_4bit(x: torch.Tensor):
    """
    Pack int4 weight for W4A8 MoEGMM. Each two int4 numbers are packed into one byte.
    """
    assert x.dtype == torch.int8
    x = x.T.contiguous()  # pack along output channel dim.
    shape = x.shape
    x = x.view(-1, 2)
    # for example, 5(0b00000101) << 4 -> 0b01010000, -7 (0b11111001) & 0b00001111 -> 0b00001001,
    # then 0b01010000 | 0b00001001 -> 0b01011001
    x1 = x[:, 0]
    x2 = x[:, 1]
    y_x2 = torch.bitwise_left_shift(x2, 4)
    y_x1 = x1 & 0b00001111
    y = torch.bitwise_or(y_x1, y_x2)
    y = y.view(shape[0], shape[1] // 2)
    return y.T.contiguous()


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
    if bits == 4:   # not mxquant
        # pack 4bit for W4A8 MoEGMM
        quantized = quantized.to(torch.int8)
        bias = int4_assistance_bias(quantized, scale)
        quantized = pack_4bit(quantized)
        scale = scale_fp32_to_u64(scale)
        return quantized, scale, bias
    else:
        return quantized.to(torch.int8), scale.to(torch.float32), None


def int4_assistance_bias(weight, weight_scale):
    """
    Calculate the int4 weight assistance bias matrix for W4A8 MoEGEMM.
    """
    repeat_times = weight.shape[1] // weight_scale.shape[1]
    expanded_scale = weight_scale.repeat_interleave(repeat_times, dim=1)
    # 8 is the max value of INT4, for normalizing the quantization range of assistance bias.
    weight_assistant_matrix = (expanded_scale * weight * 8).sum(dim=1).float()
    return weight_assistant_matrix


def is_match_layer_name(weight_name, layer_names):
    is_match_weight_name = False
    for layer in layer_names:
        if layer in weight_name:
            is_match_weight_name = True
            break
    return is_match_weight_name


def is_indexer_layer(layer_idx, num_hidden_layers, indexer_types):
    """
    GLM-5.2 IndexShare: only "full" layers carry their own Lightning Indexer weights
    (self_attn.indexer.*); "shared" layers reuse the previous full layer's top-k and have
    NO indexer tensors. MTP / next-n layers (layer_idx >= num_hidden_layers) always carry a
    full indexer. When `indexer_types` is absent (e.g. GLM-5, no IndexShare) every layer has
    an indexer, so we fall back to True to preserve the legacy behaviour.
    """
    if layer_idx >= num_hidden_layers:
        return True
    if not indexer_types:
        return True
    return indexer_types[layer_idx] == "full"


def generate_ignore_item(num_layers, num_hidden_layers, first_k_dense_replace,
                         indexer_types=None, is_mx=False):
    """
    Generate a list of layer names to be ignored during quantization.

    GLM-5.2 IndexShare: indexer.* names are only emitted for layers that actually carry an
    indexer ("full" + MTP); "shared" layers have no indexer tensors so emitting them would
    produce dangling ignore entries that never match a real module.
    """
    ignore = []
    for i in range(0, num_layers):
        if is_indexer_layer(i, num_hidden_layers, indexer_types):
            if not is_mx:
                ignore.append(f'model.layers.{i}.self_attn.indexer.wk')
            ignore.append(f'model.layers.{i}.self_attn.indexer.weights_proj')
        if not is_mx:
            ignore.append(f'model.layers.{i}.self_attn.kv_a_proj_with_mqa')
            ignore.append(f'model.layers.{i}.self_attn.q_a_proj')
        ignore.append(f'model.layers.{i}.self_attn.kv_b_proj')
        if i >= first_k_dense_replace:
            ignore.append(f'model.layers.{i}.mlp.gate')
        if i >= num_hidden_layers:
            ignore.append(f'model.layers.{i}.eh_proj')
            ignore.append(f'model.layers.{i}.shared_head.head')
    ignore.append('lm_head')
    return ignore


def generate_w4a8_quant(num_layers, first_k_dense_replace):
    quant_w4a8_layers = []
    for i in range(0, num_layers):
        if i >= first_k_dense_replace:
            quant_w4a8_layers.append(f'model.layers.{i}.mlp.experts')
    return quant_w4a8_layers


def generate_quant_group(a_num_bits=8, w_num_bits=8, targets=None, activation_use_clip=False, qtype="float"):
    quant_group = {"input_activations": {"actorder": None, "block_structure": None, "dynamic": True,
                                         "group_size": None, "num_bits": a_num_bits,
                                         "observer": "memoryless", "observer_kwargs": {},
                                         "strategy": "token", "symmetric": True, "type": qtype},
                   "activation_use_clip": activation_use_clip,
                   "output_activations": None,
                   "targets": targets,
                   "weights": {"actorder": None, "block_structure": None, "dynamic": False,
                               "group_size": None, "num_bits": w_num_bits,
                               "observer": "minmax", "observer_kwargs": {},
                               "strategy": "channel", "symmetric": True, "type": qtype}}
    return quant_group


def generate_quant_config(c8, ignores, w4a8=False, clip=False, is_mx=False):
    """
    Generate a quantization configuration dictionary based on the specified parameters.
    """
    kv_cache_scheme = {"num_bits": NUM_BITS_8,
                       "type": 'float' if is_mx else "int",
                       "strategy": 'group',
                       "group_size": 128,
                       "dynamic": 'true',
                       "symmetric": 'true'} if c8 or is_mx else None
    config_groups = {"group_0": {}}
    if w4a8:
        config_groups.update({"group_1": {}})
    quant_config = {"config_groups": config_groups,
                    "format": "float-quantized" if is_mx else "int-quantized",
                    "global_compression_ratio": 1,
                    "ignore": ignores,
                    "kv_cache_scheme": kv_cache_scheme,
                    "li_cache_scheme": {
                        "type": 'float' if is_mx else "int",
                        "num_bits": NUM_BITS_8,
                    },
                    "quant_method": "mxfp8" if is_mx and not w4a8 else "compressed-tensors",
                    "quantization_status": "compressed"}
    if is_mx:
        quant_config["activation_scheme"] = "dynamic"
    qtype = "float" if is_mx else "int"
    quant_config["config_groups"]["group_0"] = generate_quant_group(
        a_num_bits=NUM_BITS_8, w_num_bits=NUM_BITS_8, targets=["Linear"], qtype=qtype)
    if w4a8:
        quant_config["config_groups"]["group_1"] = generate_quant_group(
            a_num_bits=NUM_BITS_8, w_num_bits=NUM_BITS_4, targets=["MoEGMM"], activation_use_clip=clip, qtype=qtype)
    if is_mx:
        quant_config["weight_block_size"] = [1, 32]
    return quant_config


def generate_li_hadamard_matrix(quant_param_path, num_layers, num_hidden_layers=None,
                                indexer_types=None, dim=128):
    hadamard_matrixs = {}
    if num_hidden_layers is None:
        num_hidden_layers = num_layers
    for layer_idx in range(0, num_layers):
        # GLM-5.2 IndexShare: only "full" / MTP layers own an indexer hadamard matrix.
        if not is_indexer_layer(layer_idx, num_hidden_layers, indexer_types):
            continue
        key = f'model.layers.{layer_idx}.self_attn.indexer.hadamard_matrix'
        if quant_param_path is None:
            hadamard_matrixs[key] = get_had_pow2(
                dim, norm=True).to(torch.bfloat16)
        else:
            hadamard_path = os.path.join(
                quant_param_path, f'quant_parameters_{layer_idx}.pth')
            if not os.path.exists(hadamard_path):
                hadamard_matrix = get_had_pow2(
                    dim, norm=True).to(torch.bfloat16)
            else:
                quant_params = torch.load(hadamard_path)
                if key not in quant_params:
                    hadamard_matrix = get_had_pow2(
                        dim, norm=True).to(torch.bfloat16)
                else:
                    hadamard_matrix = quant_params[key].bfloat16()
            hadamard_matrixs[key] = hadamard_matrix
    return hadamard_matrixs


def copy_py_json_jinja(src, target):
    for root, _, files in os.walk(src):
        for file in files:
            if file.endswith(('.py', '.json', '.jinja')):
                src_path = os.path.join(root, file)
                rel_dir = os.path.relpath(root, src)
                dst_dir = os.path.join(target, rel_dir)
                os.makedirs(dst_dir, exist_ok=True)
                dst_path = os.path.join(dst_dir, file)
                shutil.copy2(src_path, dst_path)


def load_clip_params(num_hidden_layers, num_nextn_predict_layers, clip_param_path):
    num_layers = num_hidden_layers + num_nextn_predict_layers
    kv_clip_params = {}
    act_clip_params = {}
    weight_clip_params = {}
    clip_param_files = list(glob(os.path.join(clip_param_path, "*.pth")))
    clip_param_files.sort()
    for layer_idx in range(0, num_layers):
        expected_file = os.path.join(
            clip_param_path, f'quant_parameters_{layer_idx}.pth')
        if not os.path.exists(expected_file):
            if layer_idx < num_hidden_layers:
                raise ValueError(
                    f"{expected_file} not found, please check the {clip_param_path}")
            else:
                # For layer >= num_hidden_layers, if not found, use num_hidden_layers-1's quant params with factor 1.0
                expected_file = os.path.join(
                    clip_param_path, f'quant_parameters_{num_hidden_layers - 1}.pth')
                old_quant_params = torch.load(expected_file)
                quant_params = {
                    k.replace(f'layers.{num_hidden_layers - 1}', f'layers.{layer_idx}'): torch.tensor(1.0).to(v.dtype)
                    for k, v in old_quant_params.items()}
        else:
            quant_params = torch.load(expected_file)
        for name, factor in quant_params.items():
            complete_name = f"model.layers.{layer_idx}.{name}"
            if complete_name.endswith("w_alpha"):
                weight_clip_params[complete_name] = factor
            elif complete_name.endswith("ckv_a_alpha"):
                kv_clip_params[complete_name] = factor
            elif complete_name.endswith("alpha"):
                act_clip_params[complete_name] = factor
    return kv_clip_params, act_clip_params, weight_clip_params


def generate_quant_layers(num_layers, num_experts, first_k_dense_replace=0,
                          indexer_types=None, num_hidden_layers=None, w4a8=False, is_mx=False):
    quant_layers = {}
    if num_hidden_layers is None:
        num_hidden_layers = num_layers
    moe_bit = NUM_BITS_4 if w4a8 else NUM_BITS_8
    se_bit = NUM_BITS_8
    attn_bit = NUM_BITS_8
    mlp_linears = ["gate_proj", "down_proj", "up_proj"]
    for i in range(num_layers):
        if i >= first_k_dense_replace:
            for j in range(num_experts):
                for n in mlp_linears:
                    quant_layers[f"model.layers.{i}.mlp.experts.{j}.{n}"] = moe_bit
            for n in mlp_linears:
                quant_layers[f"model.layers.{i}.mlp.shared_experts.{n}"] = se_bit
        else:
            for n in mlp_linears:
                quant_layers[f"model.layers.{i}.mlp.{n}"] = se_bit
        if is_mx:
            quant_layers[f"model.layers.{i}.self_attn.q_a_proj"] = attn_bit
            quant_layers[f"model.layers.{i}.self_attn.kv_a_proj_with_mqa"] = attn_bit
        quant_layers[f"model.layers.{i}.self_attn.o_proj"] = attn_bit
        quant_layers[f"model.layers.{i}.self_attn.q_b_proj"] = attn_bit
        # GLM-5.2 IndexShare: only "full" / MTP layers own indexer Linear weights.
        if is_indexer_layer(i, num_hidden_layers, indexer_types):
            quant_layers[f"model.layers.{i}.self_attn.indexer.wq_b"] = attn_bit
            if is_mx:
                quant_layers[f"model.layers.{i}.self_attn.indexer.wk"] = attn_bit

    return quant_layers


def main(fp8_path, output_path, quant_type, clip=False, quant_param_path=None):
    """
    Converts FP8 weights to BF16 and saves the converted weights.

    This function reads FP8 weights from the specified directory, converts them to BF16,
    and saves the converted weights to another specified directory. It also updates the
    model index file to reflect the changes.

    Args:
    fp8_path (str): The path to the directory containing the FP8 weights and model index file.
    output_path (str): The path to the directory where the converted BF16/INT8 weights will be saved.
    quant_type (str): The type of quantization to apply. Supported values are "bfloat16",
    "w8a8c16", "w8a8c8", "w4a8c16", and "w4a8c8".
    clip (bool, optional): Whether to apply clipping during quantization. Defaults to False.
    quant_param_path (str, optional): The path to the directory containing quantization parameters.
    w4a8 (bool): Quantize the MoE to W4A8.

    Raises:
    KeyError: If a required scale_inv tensor is missing for a weight.

    Notes:
    - The function assumes that the FP8 weights are stored in safetensor files.
    - The function caches loaded safetensor files to optimize memory usage.
    - The function updates the model index file to remove references to scale_inv tensors.
    """
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(output_path, exist_ok=True)
    assert quant_type in [
        "bfloat16", "w8a8c16", "w8a8c8", "w4a8c16", "w4a8c8", "w8a8-mx"], f"Unsupported quant_type: {quant_type}"
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    config_file = os.path.join(fp8_path, 'config.json')
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    with open(config_file, "r") as f:
        config = json.load(f)

    weight_map = model_index["weight_map"]
    new_weight_map = {}
    num_hidden_layers = config['num_hidden_layers']
    num_nextn_predict_layers = config['num_nextn_predict_layers']
    first_k_dense_replace = config['first_k_dense_replace']
    num_layers = num_hidden_layers + num_nextn_predict_layers
    num_experts = config['n_routed_experts']
    # GLM-5.2 IndexShare: per-layer "full"/"shared" indexer schedule. Absent for GLM-5 (every
    # layer has an indexer) -> is_indexer_layer() falls back to True.
    indexer_types = config.get('indexer_types')
    quant_ignore_layers = []
    quant_w4a8_layers = []
    c8 = quant_type.endswith('c8')
    w4a8 = quant_type.startswith("w4a8")
    w8a8 = quant_type.startswith("w8a8")
    is_mx = quant_type.endswith('mx')
    if w8a8 or w4a8:
        quant_ignore_layers = generate_ignore_item(
            num_layers, num_hidden_layers, first_k_dense_replace,
            indexer_types=indexer_types, is_mx=is_mx)
        quantization_config = generate_quant_config(
            c8, quant_ignore_layers, w4a8=w4a8, clip=clip, is_mx=is_mx)
        config['quantization_config'] = quantization_config

    quant_layers = generate_quant_layers(num_layers, num_experts,
        first_k_dense_replace=first_k_dense_replace,
        indexer_types=indexer_types, num_hidden_layers=num_hidden_layers,
        w4a8=w4a8, is_mx=is_mx
    )

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

    if c8 and clip:
        assert quant_param_path is not None, "Please pass the quant_param_path"
        kv_clip_params, act_clip_params, weight_clip_params = load_clip_params(
            num_hidden_layers, num_nextn_predict_layers, quant_param_path)

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
                    if weight.dtype == torch.int8:
                        weight = unpack_mxfloat4_to_fp32(weight.view(torch.uint8))
                        weight = weight_dequant(weight, scale_inv, block_size=32, is_mx=True)
                    else:
                        weight = weight_dequant(weight, scale_inv)
                except KeyError:
                    print(
                        f"Warning: Missing scale_inv tensor for {weight_name}, skipping conversion")
                new_state_dict[weight_name] = weight
                new_weight_map[weight_name] = file_name
            else:
                new_state_dict[weight_name] = weight
                new_weight_map[weight_name] = file_name
            
            if w4a8 or w8a8:
                new_weight_name = weight_name.rsplit(".", 1)[0]
                if new_weight_name in list(quant_layers.keys()):
                    if clip:
                        weight_clip_name = weight_name.replace(
                            "weight", "w_alpha")
                        weight_clip_factor = weight_clip_params.get(
                            weight_clip_name, None)
                    else:
                        weight_clip_factor = None

                    bit = quant_layers[new_weight_name]
                    if is_mx:
                        quant_weight, scale_inv = quantize_mx(weight, bit, real_quant=True)
                        if bit == NUM_BITS_4:
                            quant_weight = f32_to_f4_unpacked(quant_weight.float())
                            quant_weight = pack_uint4(quant_weight)
                    else:
                        quant_weight, scale_inv, bias = \
                            int_weight_quant(weight, bits=bit, weight_clip_factor=weight_clip_factor)
                    new_scale_name = weight_name.replace('.weight', '.weight_scale')

                    new_state_dict[weight_name] = quant_weight
                    new_state_dict[new_scale_name] = scale_inv

                    new_weight_map[weight_name] = file_name
                    new_weight_map[new_scale_name] = file_name

        new_safetensor_file = os.path.join(output_path, file_name)
        save_file(new_state_dict, new_safetensor_file,
                  metadata={'format': 'pt'})

        # Memory management: keep only the 2 most recently used files
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]

    if c8 or clip:
        safetensor_files = list(
            glob(os.path.join(output_path, "*.safetensors")))
        safetensor_files.sort()
        first_safetensor_file = safetensor_files[-1]
        file_name = os.path.basename(safetensor_file)
        first_safetensor_dict = load_file(first_safetensor_file, device="cpu")

        if c8:
            # Add Hadamard matrix to the first safetensor file
            hadamard_matrixs = generate_li_hadamard_matrix(
                quant_param_path, num_layers, num_hidden_layers=num_hidden_layers,
                indexer_types=indexer_types, dim=128)
            first_safetensor_dict.update(hadamard_matrixs)

            # Update weight map
            for weight_name in hadamard_matrixs.keys():
                new_weight_map[weight_name] = file_name

        if clip:
            first_safetensor_dict.update(act_clip_params)
            for weight_name in act_clip_params.keys():
                new_weight_map[weight_name] = file_name

        first_safetensor_dict.update(kv_clip_params)
        for weight_name in kv_clip_params.keys():
            new_weight_map[weight_name] = file_name

        new_safetensor_file = os.path.join(output_path, file_name)
        save_file(first_safetensor_dict, new_safetensor_file,
                  metadata={'format': 'pt'})

    copy_py_json_jinja(fp8_path, output_path)

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
    parser.add_argument("--input_hf_path", type=str, required=True)
    parser.add_argument("--output_hf_path", type=str, required=True)
    parser.add_argument("--quant_type", type=str, default="bfloat16",
                        choices=["bfloat16", "w8a8c16", "w8a8c8", "w4a8c16", "w4a8c8", "w8a8-mx"])
    parser.add_argument("--clip", action='store_true')
    parser.add_argument("--quant_param_path", type=str, default=None)
    args = parser.parse_args()

    main(args.input_hf_path, args.output_hf_path,
         args.quant_type, args.clip, args.quant_param_path)
