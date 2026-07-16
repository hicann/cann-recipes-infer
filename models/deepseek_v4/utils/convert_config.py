# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
import sys
import re
import shutil
from argparse import ArgumentParser


NUM_BITS_4 = 4
NUM_BITS_8 = 8

def generate_ignore_item(num_layers, compress_ratios, is_fp=False):
    """
    Generate a list of layer names to be ignored during quantization.
    """
    ignore = []
    for i in range(0, num_layers):
        ratio = compress_ratios[i]
        if not is_fp:
            ignore.append(f"layers.{i}.attn.wq_a")
            ignore.append(f"layers.{i}.attn.wkv")
            ignore.append(f"layers.{i}.attn.wo_a")
        if ratio == 4: # model have compress ratios [1, 4, 128]
            ignore.append(f"layers.{i}.attn.indexer.weights_proj")
            ignore.append(f"layers.{i}.attn.indexer.compressor.wgate")
            ignore.append(f"layers.{i}.attn.indexer.compressor.wkv")
            ignore.append(f"layers.{i}.attn.compressor.wgate")
            ignore.append(f"layers.{i}.attn.compressor.wkv")
        if ratio == 128: # model have compress ratios [1, 4, 128]
            ignore.append(f"layers.{i}.attn.compressor.wgate")
            ignore.append(f"layers.{i}.attn.compressor.wkv")
    if not is_fp:
        ignore.append("mtp.0.attn.wq_a")
        ignore.append("mtp.0.attn.wkv")
        ignore.append('mtp.0.attn.wo_a')
    ignore.append('mtp.0.head')
    ignore.append('head')
    return ignore


def generate_quant_group(a_num_bits=8, w_num_bits=8, qtype="float", activation_use_clip=False, is_mx=False):
    input_group_size = None
    weight_group_size = None
    weight_block_size = None
    if qtype == "int":
        activation_quant_strategy = "token"
        weight_quant_strategy = "channel"
    elif qtype == "float":
        activation_quant_strategy = "group"
        weight_quant_strategy = "group" if is_mx else "block"
        # 32: 32 elements has same scale element when mxfp for dsv4
        # 128: 128 elements has same scale element when fp for dsv4
        input_group_size = 32 if is_mx else 128
        weight_group_size = 32 if is_mx else None
        weight_block_size = "128x128" if not is_mx else None
    else:
        raise ValueError(f"Unsupported quant dtype {qtype}")
    quant_group = {"input_activations": {"actorder": None, "block_structure": None, "dynamic": True,
                                         "group_size": input_group_size, "num_bits": a_num_bits,
                                         "observer": "memoryless", "observer_kwargs": {},
                                         "strategy": activation_quant_strategy, "symmetric": True, "type": qtype},
                   "activation_use_clip": activation_use_clip,
                   "output_activations": None,
                   "weights": {"actorder": None, "block_structure": weight_block_size, "dynamic": False,
                               "group_size": weight_group_size, "num_bits": w_num_bits,
                               "observer": "minmax", "observer_kwargs": {},
                               "strategy": weight_quant_strategy, "symmetric": True, "type": qtype}}
    return quant_group


def generate_quant_config(cache_scheme, ignores, w4a8=False, is_fp=False, is_mx=False):
    """
    Generate a quantization configuration dictionary based on the specified parameters.
    """
    config_groups = {"group_0": {"targets": ["Linear"]}}
    if w4a8:
        config_groups.update({"group_1": {"targets": ["MoEGMM"]}})
    quant_config = {"config_groups": config_groups,
                    "format": "float-quantized" if is_fp else "int-quantized",
                    "global_compression_ratio": 1,
                    "ignore": ignores,
                    "quant_method": "compressed-tensors",
                    "quantization_status": "compressed"}
    quant_config.update(cache_scheme)
    qtype = "float" if is_fp else "int"
    quant_config["config_groups"]["group_0"].update(
        generate_quant_group(a_num_bits=NUM_BITS_8, w_num_bits=NUM_BITS_8, qtype=qtype, is_mx=is_mx)
        )
    if w4a8:
        quant_config["config_groups"]["group_1"].update(
            generate_quant_group(
                a_num_bits=NUM_BITS_8, 
                w_num_bits=NUM_BITS_4, 
                qtype=qtype, 
                is_mx=is_fp,  # only support mxfp4 for w4a8 with float type
                )
            )
    return quant_config


def main(fp8_path):
    config_file = os.path.join(fp8_path, 'config.json')
    with open(config_file, "r") as f:
        config = json.load(f)
    num_layers = config['num_hidden_layers']
    compress_ratios = config['compress_ratios']
    cache_scheme = {"kv_cache_scheme": {"num_bits": NUM_BITS_8, "type": "float"},
                    "li_cache_scheme": {
                        "type": "float",
                        "num_bits": NUM_BITS_8,
                    }}
    if 'quantization_config' in config:
        config.pop('quantization_config')

    quant_ignore_layers = generate_ignore_item(num_layers, compress_ratios, is_fp=True)
    quantization_config = generate_quant_config(
        cache_scheme, quant_ignore_layers, w4a8=True, is_fp=True)
    config['quantization_config'] = quantization_config
    config['quantization_config']["quant_method"] = "compressed-tensors"
    config['quantization_config']["quantization_status"] = "compressed"

    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_fp8_hf_path", type=str, required=True)
    args = parser.parse_args()
    main(args.input_fp8_hf_path)