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

import json
import logging
import os
import shutil
from argparse import ArgumentParser
from dataclasses import dataclass
from glob import glob

from tqdm import tqdm
import torch
import torch_npu
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)


def copy_aux_files(src_dir: str, dst_dir: str) -> None:
    """Copy tokenizer/config/code assets needed by HF `from_pretrained`.

    This converter rewrites only *.safetensors + index/config, so we must also
    carry over tokenizer files and optional remote-code python modules.
    """
    os.makedirs(dst_dir, exist_ok=True)

    # Common HF tokenizer / config artifacts
    candidates = [
        # tokenizer
        "tokenizer.model",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        # generation / misc
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
    ]

    for name in candidates:
        s = os.path.join(src_dir, name)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(dst_dir, name))

    # Carry over any custom code files when trust_remote_code=True
    for pattern in ("tokenization_*.py", "configuration_*.py", "modeling_*.py"):
        for s in glob(os.path.join(src_dir, pattern)):
            if os.path.isfile(s):
                shutil.copy2(s, os.path.join(dst_dir, os.path.basename(s)))

    # If the repo uses SentencePiece but file name differs, copy any *.model
    # (small files; safe) but avoid huge weights
    for s in glob(os.path.join(src_dir, "*.model")):
        bn = os.path.basename(s)
        if bn.endswith(".model") and os.path.getsize(s) < 50 * 1024 * 1024:
            shutil.copy2(s, os.path.join(dst_dir, bn))


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


def hif8_weight_quant(tensor: torch.Tensor, bits=8, weight_clip_factor=None):
    assert tensor.dim() == 2
    tensor = tensor.npu()
    quantized_tensor, scale = torch_npu.npu_dynamic_quant(tensor, dst_type=torch_npu.hifloat8)
    quantized_tensor, scale = quantized_tensor.cpu(), scale.cpu()
    scale = scale.unsqueeze(-1)
    assert scale.shape == (tensor.shape[0], 1)  # per-row(per-channel)
    return quantized_tensor, scale.to(torch.float32)


def fp8_weight_quant(tensor: torch.Tensor, bits=8, weight_clip_factor=None):
    assert tensor.dim() == 2
    tensor = tensor.npu()
    quantized_tensor, scale = torch_npu.npu_dynamic_quant(tensor, dst_type=torch_npu.float8_e4m3fn)
    quantized_tensor, scale = quantized_tensor.cpu(), scale.cpu()
    scale = scale.unsqueeze(-1)
    assert scale.shape == (tensor.shape[0], 1)
    return quantized_tensor, scale.to(torch.float32)


def mxfp8_weight_quant(tensor: torch.Tensor, bits=8, weight_clip_factor=None):
    assert tensor.dim() == 2
    tensor = tensor.npu()
    quantized_tensor, scale = torch_npu.npu_dynamic_mx_quant(tensor, dst_type=torch_npu.float8_e4m3fn)
    # For mxfp8 @ mxfp4, the scale is represented as 2D instead of 3D.
    (dim0, dim1, dim2) = scale.shape
    scale = scale.reshape(dim0, dim1 * dim2)
    quantized_tensor, scale = quantized_tensor.cpu(), scale.cpu()
    return quantized_tensor, scale


def mxfp4_weight_quant(tensor: torch.Tensor, bits=8, weight_clip_factor=None):
    assert tensor.dim() == 2
    tensor = tensor.npu()
    quantized_tensor, scale = torch_npu.npu_dynamic_mx_quant(tensor, dst_type=torch_npu.float4_e2m1fn_x2)
    # For mxfp8 @ mxfp4, the scale is represented as 2D instead of 3D.
    (dim0, dim1, dim2) = scale.shape
    scale = scale.reshape(dim0, dim1 * dim2)
    quantized_tensor, scale = quantized_tensor.cpu(), scale.cpu()
    return quantized_tensor, scale


def quantize_to_mxfp4_e2m1(input_tensor, block_size=32):
    """
    Perform row-wise MXFP4 quantization on a 2D tensor.
    input_tensor: [Rows, Cols] bf16/fp32 tensor
    block_size: Number of elements per row sharing a scale (default 32)

    Returns:
    quantized_float: [Rows, Cols] float tensor (storing fp4 values)
    scales_2d: [Rows, Cols // block_size] uint8 tensor (storing e8m0 offset values)
    """
    assert input_tensor.dim() == 2, "Input must be a 2D matrix"
    rows, cols = input_tensor.shape
    assert cols % block_size == 0, f"Cols {cols} must be divisible by block_size {block_size}"

    device = input_tensor.device
    # 1. Reshape for block-wise processing: [Rows, Num_Blocks, Block_Size]
    input_reshaped = input_tensor.view(rows, -1, block_size).to(torch.float32)

    # 2. Compute max absolute value per block
    # max_val shape: [rows, num_blocks, 1]
    max_val_per_block = torch.max(torch.abs(input_reshaped), dim=2, keepdim=True)[0]

    # 3. Compute E8M0 Scale (shared exponent)
    # E2M1 maximum representable value is 6.0
    scale_exp = torch.ceil(torch.log2(max_val_per_block / 6.0 + 1e-12))

    # Map to uint8 (E8M0 offset 127)
    scale_uint8 = (scale_exp + 127).clamp(0, 255).to(torch.uint8)
    # Compress to 2D: [Rows, Cols // block_size]
    scales_2d = scale_uint8.squeeze(-1)

    # 4. Compute actual scale value for dequantization simulation
    actual_scale = torch.pow(2.0, scale_exp)

    # 5. Scale and map to FP4 discrete values
    rescaled_input = input_reshaped / actual_scale

    sign = torch.sign(rescaled_input)
    abs_input = torch.abs(rescaled_input)

    # FP4 E2M1 standard values
    fp4_v_list = torch.tensor([0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=device)

    # Find nearest neighbor (Vectorized)
    # shape: [rows, num_blocks, block_size, 1] - [7]
    dist = torch.abs(abs_input.unsqueeze(-1) - fp4_v_list)
    idx = torch.argmin(dist, dim=-1)
    quantized_abs = fp4_v_list[idx]

    # 6. Restore to original 2D shape
    quantized_float = (sign * quantized_abs).view(rows, cols)

    return quantized_float, scales_2d


def mxfp4_weight_quant_gmm(input_tensor):
    quantized_float, scale = quantize_to_mxfp4_e2m1(input_tensor, block_size=32)
    weight = quantized_float.npu()
    weight = torch_npu.npu_format_cast(weight, 29, torch.float8_e4m3fn)  # ND -> NZ, torch.float16
    weight = torch_npu.npu_convert_weight_to_int4pack(weight)  # compress
    weight = weight.to(torch.float).cpu()
    return weight, scale


def is_ignore_quant(weight_name, quant_ignore_layers_name):
    """
    Check if a layer should be ignored during quantization based on its name.
    """
    is_ignore = False
    for layer in quant_ignore_layers_name:
        if layer in weight_name:
            is_ignore = True
            logger.info("Ignore quantization %s", weight_name)
            break
    return is_ignore


# Ignore layers that should not be quantized
def generate_ignore_item(num_layers, num_hidden_layers):
    """
    Generate a list of layer names to be ignored during quantization.
    """
    del num_hidden_layers
    ignore = [None] * (4 * num_layers + 3)
    for i in range(0, num_layers):
        # ignore.append(f'model.layers.{i}.mlp.gate') ## Pangu
        ignore.append(f'model.layers.{i}.input_layernorm')
        ignore.append(f'model.layers.{i}.post_attention_layernorm')
        ignore.append(f'model.layers.{i}.self_attn.q_norm')
        ignore.append(f'model.layers.{i}.self_attn.k_norm')
    ignore.append('lm_head')
    ignore.append('model.norm')
    ignore.append('model.embed_tokens')
    return ignore


def generate_mxfp_quant_config(c8, num_layers, num_hidden_layers, quant_method="mxfp8"):
    ignored_layers = generate_ignore_item(num_layers=num_layers, num_hidden_layers=num_hidden_layers)
    if c8:
        kv_cache_scheme = {"num_bits": 8, "type": "float", "strategy": "tensor", "dynamic": False, "symmetric": True}
    else:
        kv_cache_scheme = None
    targets = ["Linear"]
    quant_config = {
        "activation_scheme": "dynamic",
        "output_activations": None,
        "targets": targets,
        "weights": {"actorder": None, "block_structure": None, "dynamic": False,
                    "group_size": 32, "num_bits": 8,
                    "observer": "minmax", "observer_kwargs": {},
                    "strategy": "group", "symmetric": True, "type": "float"},
        "format": "float-quantized",
        "global_compression_ratio": 1,
        "ignored_layers": ignored_layers,
        "kv_cache_scheme": kv_cache_scheme,
        "quant_method": quant_method,
        "quantization_status": "compressed"
    }
    return quant_config


def generate_hifp_quant_group(a_num_bits=8, w_num_bits=8, targets=None):
    del a_num_bits
    return {
        "output_activations": None,
        "targets": targets,
        "weights": {
            "actorder": None,
            "block_structure": None,
            "dynamic": False,
            "group_size": None,
            "num_bits": w_num_bits,
            "observer": "minmax",
            "observer_kwargs": {},
            "strategy": "channel",
            "symmetric": True,
            "type": "int",
        },
    }


def generate_hifp_quant_config(c8, num_layers, num_hidden_layers, quant_method="hif8"):
    ignores = generate_ignore_item(num_layers=num_layers, num_hidden_layers=num_hidden_layers)
    if c8:
        kv_cache_scheme = {"num_bits": 8, "type": "int", "strategy": "tensor", "dynamic": False, "symmetric": True}
    else:
        kv_cache_scheme = None
    quant_config = {
        "activation_scheme": "dynamic",
        "format": "int-quantized",
        "global_compression_ratio": 1,
        "ignored_layers": ignores,
        "kv_cache_scheme": kv_cache_scheme,
        "quant_method": quant_method,
        "quantization_status": "compressed",
    }
    quant_config.update(generate_hifp_quant_group(a_num_bits=8, w_num_bits=8, targets=["Linear"]))
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


@dataclass(frozen=True)
class ConversionOptions:
    w8a8: bool
    c8: bool = False
    quant_param_path: str | None = None
    w_quant: str = "mxfp4"


def _load_model_config(bf16_path: str) -> dict:
    """Load config file and remove existing quantization config."""
    config_file = os.path.join(bf16_path, 'config.json')
    with open(config_file, "r") as f:
        config = json.load(f)
    if 'quantization_config' in config:
        config.pop('quantization_config')
    return config


def _setup_quantization_config(
    config: dict, options: ConversionOptions, num_layers: int, num_hidden_layers: int
) -> tuple[dict, list]:
    """Setup quantization config and return ignore layers list."""
    quant_ignore_layers = []
    if not options.w8a8:
        return config, quant_ignore_layers

    if options.w_quant == "hif8":
        quantization_config = generate_hifp_quant_config(
            options.c8, num_layers, num_hidden_layers, quant_method=options.w_quant
        )
    else:
        quantization_config = generate_mxfp_quant_config(
            options.c8, num_layers, num_hidden_layers, quant_method=options.w_quant
        )
    config['quantization_config'] = quantization_config
    quant_ignore_layers = generate_ignore_item(num_layers, num_hidden_layers)
    return config, quant_ignore_layers


def _process_single_weight(
    weight_name: str,
    weight: torch.Tensor,
    file_name: str,
    options: ConversionOptions,
    quant_ignore_layers: list,
    weight_quant_fn,
) -> tuple[dict, dict]:
    """Process a single weight tensor and return state_dict and weight_map entries."""
    state_entries = {weight_name: weight}
    map_entries = {weight_name: file_name}

    if weight.element_size() != 2:
        return state_entries, map_entries

    if not options.w8a8:
        return state_entries, map_entries

    if is_ignore_quant(weight_name, quant_ignore_layers):
        return state_entries, map_entries

    # Only quantize 2D Linear weights. Keep bias/1D and other tensors in BF16/FP16.
    if weight.dim() != 2 or (not weight_name.endswith(".weight")):
        return state_entries, map_entries

    logger.info("weight_name: %s", weight_name)
    quantized_weight, scale = weight_quant_fn(weight, weight_clip_factor=None)
    scale_name = f"{weight_name}_scale"

    state_entries = {weight_name: quantized_weight, scale_name: scale}
    map_entries = {weight_name: file_name, scale_name: file_name}
    return state_entries, map_entries


def _handle_c8_kv_cache(
    output_path: str, kv_quant_params: dict, new_weight_map: dict, last_safetensor_file: str
) -> None:
    """Handle c8 kv cache post-processing."""
    safetensor_files = list(glob(os.path.join(output_path, "*.safetensors")))
    safetensor_files.sort()
    last_file = safetensor_files[-1]
    last_dict = load_file(last_file, device="cpu")
    last_dict.update(kv_quant_params)
    last_file_name = os.path.basename(last_safetensor_file)
    for weight_name in kv_quant_params.keys():
        new_weight_map[weight_name] = last_file_name
    save_file(last_dict, os.path.join(output_path, last_file_name), metadata={'format': 'pt'})


def _save_output_files(output_path: str, new_weight_map: dict, config: dict) -> None:
    """Save model index and config files."""
    new_model_index_file = os.path.join(output_path, "model.safetensors.index.json")
    new_config_file = os.path.join(output_path, "config.json")
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": new_weight_map}, f, indent=2)
    with open(new_config_file, "w") as f:
        json.dump(config, f, indent=2)


def main(
    bf16_path,
    output_path,
    options: ConversionOptions,
):
    """
    Converts BF16 weights to quantized format and saves the converted weights.

    This function reads weights from the specified directory, applies quantization
    if enabled, and saves the converted weights to another specified directory.

    Args:
        bf16_path (str): Path to the directory containing the BF16 weights.
        output_path (str): Path to the directory where converted weights will be saved.
        options (ConversionOptions): Conversion options including quantization settings.
    """
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(output_path, exist_ok=True)
    copy_aux_files(bf16_path, output_path)

    # Load model config
    config = _load_model_config(bf16_path)
    new_weight_map = {}

    # Get layer counts
    num_hidden_layers = int(config.get('num_hidden_layers', 48))
    num_nextn_predict_layers = config.get('num_nextn_predict_layers', 0)
    num_layers = num_hidden_layers + num_nextn_predict_layers

    # Setup quantization config
    config, quant_ignore_layers = _setup_quantization_config(
        config, options, num_layers, num_hidden_layers
    )

    # Get quantization function
    quant_fns = {
        "hif8": hif8_weight_quant,
        "fp8": fp8_weight_quant,
        "mxfp8": mxfp8_weight_quant,
        "mxfp4": mxfp4_weight_quant,
    }
    if options.w_quant not in quant_fns:
        raise ValueError(
            f"Unsupported --w_quant={options.w_quant}. Supported: {', '.join(sorted(quant_fns.keys()))}"
        )
    weight_quant_fn = quant_fns[options.w_quant]

    # Load c8 kv cache params if needed
    kv_quant_params = None
    if options.c8:
        assert options.quant_param_path is not None, "Please pass the quant_param_path"
        kv_quant_params = load_c8_quant_params(num_layers, options.quant_param_path)

    # Process safetensor files
    loaded_files = {}
    safetensor_files = list(glob(os.path.join(bf16_path, "*.safetensors")))
    safetensor_files.sort()

    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cpu")
        loaded_files[file_name] = current_state_dict
        new_state_dict = {}

        for weight_name, weight in current_state_dict.items():
            try:
                state_entries, map_entries = _process_single_weight(
                    weight_name, weight, file_name, options,
                    quant_ignore_layers, weight_quant_fn
                )
                new_state_dict.update(state_entries)
                new_weight_map.update(map_entries)
            except KeyError:
                logger.warning("Missing scale_inv tensor for %s, skipping conversion", weight_name)
                new_state_dict[weight_name] = weight
                new_weight_map[weight_name] = file_name

        save_file(new_state_dict, os.path.join(output_path, file_name), metadata={'format': 'pt'})

        # Memory management: keep only the 2 most recently used files
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]

    # Handle c8 kv cache post-processing
    if options.c8 and kv_quant_params:
        _handle_c8_kv_cache(output_path, kv_quant_params, new_weight_map, safetensor_file)

    copy_py_json(bf16_path, output_path)
    _save_output_files(output_path, new_weight_map, config)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_bf16_hf_path", type=str, required=True)
    parser.add_argument("--output_hf_path", type=str, required=True)
    parser.add_argument("--w8a8", action='store_true')
    parser.add_argument(
        "--w_quant",
        type=str,
        default="mxfp4",
        choices=["hif8", "fp8", "mxfp8", "mxfp4"],
        help="Weight quantization strategy used when --w8a8 is enabled.",
    )
    parser.add_argument("--c8", action='store_true')
    parser.add_argument("--quant_param_path", type=str, default=None)
    args = parser.parse_args()

    main(
        args.input_bf16_hf_path,
        args.output_hf_path,
        ConversionOptions(
            w8a8=args.w8a8,
            c8=args.c8,
            quant_param_path=args.quant_param_path,
            w_quant=args.w_quant,
        ),
    )
