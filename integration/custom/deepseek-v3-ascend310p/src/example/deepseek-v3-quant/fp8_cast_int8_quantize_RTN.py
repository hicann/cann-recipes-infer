# coding=utf-8
# Copyright (c) 2025 QINGMAO INTELLIGENCE TECHNOLOGY (BEIJING) CO., LTD. and Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm
import shutil

import torch
from safetensors.torch import load_file, save_file

#from kernel import weight_dequant

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

    #Get the original dimensions of weight
    M, N = weight.shape

    # Compute the effective block dimensions for scale
    scale_m, scale_n = scale.shape
    assert scale_m == (M + block_size - 1) // block_size, "Mismatch in scale rows and weight rows."
    assert scale_n == (N + block_size - 1) // block_size, "Mismatch in scale columns and weight columns."

    # Convert weight to float32 for calculations
    weight = weight.to(torch.float32)

    # Expand scale to match the weight tensor's shape
    scale_expanded = scale.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)

    # Trim scale_expanded to match weight's shape if necessary
    scale_expanded = scale_expanded[:M, :N]

    # Perform element-wise multiplication
    dequantized_weight = weight * scale_expanded

    # Convert the output to the default dtype
    dequantized_weight = dequantized_weight.to(torch.get_default_dtype())

    return dequantized_weight


# Symmetric, per-channel INT8 weight quantization copied from meituan bf16_cast_channel_int8.py
def weight_quant(tensor: torch.Tensor):
    tensor = tensor.float()                 # ensure full-precision math
    assert tensor.dim() == 2
    qmax = 127.0
    abs_max = torch.abs(tensor).max(dim=1, keepdim=True)[0]  # [rows, 1]
    scale = abs_max / qmax                                  # [rows, 1]
    assert scale.shape == (tensor.shape[0], 1)
    quantized = torch.round(tensor / scale)
    quantized = torch.clamp(quantized, -qmax, qmax)
    return quantized.to(torch.int8), scale.to(torch.float32)


def copy_file(fp8_path, int8_path):
    files_to_copy = [
        'config.json',
        'configuration_deepseek.py',
        'tokenizer_config.json',
        'tokenizer.json'
    ]

    os.makedirs(int8_path, exist_ok=True)

    # copy file
    for file_name in files_to_copy:
        src_path = os.path.join(fp8_path, file_name)
        dst_path = os.path.join(int8_path, file_name)

        # check source file
        if not os.path.exists(src_path):
            print(f"[WARN]: source file is not exist: {src_path}")
            continue

        try:
            # copy file
            shutil.copy2(src_path, dst_path)
            print(f"successed: {file_name}")
        except Exception as e:
            print(f"failed {file_name}: {e}")

def main(fp8_path, int8_path):
    """
    Converts FP8 weights to INT8 (via BF16 de-quantization) and saves the converted weights.

    This function reads FP8 weights from the specified directory, de-quantizes them to BF16,
    optionally splits special tensors, then quantizes them to INT8 using symmetric per-channel
    weight quantization.  Quantized weights are saved alongside their FP32 per-channel scales,
    whose tensor names are formed by replacing the trailing ".weight" with ".w_scale".

    Args:
    fp8_path (str): The path to the directory containing the FP8 weights and model index file.
    int8_path (str): The path to the directory where the converted INT8 weights will be saved.

    Raises:
    KeyError: If a required scale_inv tensor is missing for a weight.

    Notes:
    - The function assumes that the FP8 weights are stored in safetensor files.
    - The function caches loaded safetensor files to optimize memory usage.
    - The function updates the model index file to remove references to scale_inv tensors.
    """
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(int8_path, exist_ok=True)
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    print(model_index_file)
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

    # Load config.json for splitting tensors
    config_file = os.path.join(fp8_path, "config.json")
    with open(config_file, "r") as f:
        config = json.load(f)

    # Cache for loaded safetensor files
    loaded_files = {}
    fp8_weight_names = []
    scale_suffix = ".w_scale"

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
            # Ignore layer 61
            elif weight_name.startswith("model.layers.61"):
                continue
            #FP8 weight
            elif weight.element_size() == 1:
                scale_inv_name = f"{weight_name}_scale_inv"
                try:
                    # Get scale_inv from the correct file
                    scale_inv = get_tensor(scale_inv_name)
                    fp8_weight_names.append(weight_name)
                    # BF16 tensor
                    weight = weight_dequant(weight, scale_inv)

                    # Split tensors if necessary
                    split_items = {}
                    if 'q_b_proj' in weight_name:
                        weight = weight.view(config['num_attention_heads'], config['qk_nope_head_dim'] + config['qk_rope_head_dim'], config['q_lora_rank'])
                        q_b1_proj, q_b2_proj = torch.split(weight, [config['qk_nope_head_dim'], config['qk_rope_head_dim']], dim=1)
                        q_b1_proj = q_b1_proj.reshape(config['num_attention_heads']*config['qk_nope_head_dim'], config['q_lora_rank'])
                        q_b2_proj = q_b2_proj.reshape(config['num_attention_heads']*config['qk_rope_head_dim'], config['q_lora_rank'])
                        split_items[weight_name.replace('q_b_proj', 'q_b1_proj')] = q_b1_proj
                        split_items[weight_name.replace('q_b_proj', 'q_b2_proj')] = q_b2_proj
                    elif 'kv_b_proj' in weight_name:
                        weight = weight.view(config['num_attention_heads'], config['qk_nope_head_dim'] + config['v_head_dim'], config['kv_lora_rank'])
                        kv_b2_proj, kv_b1_proj = torch.split(weight, [config['qk_nope_head_dim'], config['v_head_dim']], dim=1)
                        kv_b2_proj = torch.split(kv_b2_proj, 1, dim=0)
                        kv_b1_proj = torch.split(kv_b1_proj, 1, dim=0)
                        for i in range(config['num_attention_heads']):
                            split_items[weight_name.replace('kv_b_proj', f'kv_b2_proj.{i}')] = kv_b2_proj[i].squeeze(0).t().contiguous()
                            split_items[weight_name.replace('kv_b_proj', f'kv_b1_proj.{i}')] = kv_b1_proj[i].squeeze(0)
                    else:
                        split_items[weight_name] = weight

                    # Quantize each (possibly split) tensor and store weight + scale
                    for split_name, split_tensor in split_items.items():
                        q_tensor, q_scale = weight_quant(split_tensor)   # INT8 + FP32 scale
                        new_state_dict[split_name] = q_tensor
                        scale_name = split_name.replace(".weight", scale_suffix)
                        new_state_dict[scale_name] = q_scale
                except KeyError:
                    raise Exception("KeyError")
            else:
                new_state_dict[weight_name] = weight

        new_safetensor_file = os.path.join(int8_path, file_name)
        save_file(new_state_dict, new_safetensor_file)

        # Memory management: keep only the 2 most recently used files
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
            # torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Update model index
    # ------------------------------------------------------------------
    new_model_index_file = os.path.join(int8_path, "model.safetensors.index.json")

    # Drop every *_scale_inv entry that belonged to an FP8 weight
    for weight_name in fp8_weight_names:
        scale_inv_name = f"{weight_name}_scale_inv"
        if scale_inv_name in weight_map:
            weight_map.pop(scale_inv_name)
    new_weight_map = {}
    scale_suffix = ".w_scale"

    for weight_name, filename in weight_map.items():
        # Ignore layer 61
        if weight_name.startswith("model.layers.61"):
            continue
        # -----------------------------------------------
        # Build a small "split_items" dict for this entry
        # key   -> bool  (True if the tensor was quantized
        #                 and therefore needs a .w_scale)
        # -----------------------------------------------
        split_items = {}

        if 'q_b_proj' in weight_name:
            split_items[weight_name.replace('q_b_proj', 'q_b1_proj')] = True
            split_items[weight_name.replace('q_b_proj', 'q_b2_proj')] = True

        elif 'kv_b_proj' in weight_name:
            for i in range(config['num_attention_heads']):
                split_items[weight_name.replace('kv_b_proj', f'kv_b2_proj.{i}')] = True
                split_items[weight_name.replace('kv_b_proj', f'kv_b1_proj.{i}')] = True

        else:
            # Was this original tensor one we quantized?
            split_items[weight_name] = weight_name in fp8_weight_names

        # ------------------------------------------------
        # Write each weight (and, if needed, its scale)
        # into new_weight_map in the desired order.
        # ------------------------------------------------
        for wname, needs_scale in split_items.items():
            new_weight_map[wname] = filename
            if needs_scale:
                scale_key = wname.replace(".weight", scale_suffix)
                new_weight_map[scale_key] = filename

    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": new_weight_map}, f, indent=2)

    copy_file(fp8_path, int8_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-fp8-hf-path", type=str, required=True)
    parser.add_argument("--output-int8-hf-path", type=str, required=True)
    args = parser.parse_args()
    main(args.input_fp8_hf_path, args.output_int8_hf_path)
