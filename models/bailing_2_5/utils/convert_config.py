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
import logging
from argparse import ArgumentParser

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

BLOCK_K = 32


def is_mla_layer(layer_idx, num_hidden_layers, layer_group_size):
    """
    True for full-attention (MLA) layers, False for linear-attention layers.

    Mirrors BailingMoeV25DecoderLayer's attention_layer_type.
    """
    return ((layer_idx + 1) % layer_group_size == 0
            or layer_idx >= (num_hidden_layers // layer_group_size) * layer_group_size)


def generate_ignore(config):
    """Build the minimal, correct ignore list (runtime names) from the model config."""
    n = config["num_hidden_layers"]
    gs = config.get("layer_group_size", 5)
    first_k_dense = config.get("first_k_dense_replace", 0)

    ignore = ["lm_head", "model.embed_tokens"]
    for i in range(n):
        if is_mla_layer(i, n, gs):
            # kv_b_proj is a ColumnParallelLinear that the MLA path consumes specially; keep bf16.
            ignore.append(f"model.layers.{i}.self_attn.kv_b_proj")
        if i >= first_k_dense:
            # the routed-expert gate (ReplicatedLinear) must stay full precision.
            ignore.append(f"model.layers.{i}.mlp.router.classifier")
    return ignore


def _quant_group(targets, weight_num_bits, input_observer):
    """
    One compressed-tensors config_group for MX: activations and weights both quantized
    per-group (group_size=BLOCK_K), matching the runtime's is_dynamic_group_w8a8/w4a8_mxfp8
    scheme gates in compressed_tensors.py (weight strategy must be 'group' or 'block').
    """
    return {
        "targets": targets,
        "input_activations": {
            "actorder": None, "block_structure": None, "dynamic": True, "group_size": BLOCK_K,
            "num_bits": 8, "observer": input_observer, "observer_kwargs": {},
            "strategy": "group", "symmetric": True, "type": "float",
        },
        "activation_use_clip": False,
        "output_activations": None,
        "weights": {
            "actorder": None, "block_structure": None, "dynamic": False, "group_size": BLOCK_K,
            "num_bits": weight_num_bits, "observer": "minmax", "observer_kwargs": {},
            "strategy": "group", "symmetric": True, "type": "float",
        },
    }


def generate_quant_config(config):
    """Full Ling MXFP quantization_config: Linear -> w8a8 mxfp8, MoEGMM -> w4a8 mxfp4."""
    return {
        "config_groups": {
            "group_0": _quant_group(["Linear"], 8, "memoryless"),
            "group_1": _quant_group(["MoEGMM"], 4, "minmax"),
        },
        "format": "float-quantized",
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "weight_block_size": [1, BLOCK_K],
        "ignore": generate_ignore(config),
    }


def main(input_config, output_config=None, full=False):
    output_config = output_config or input_config
    with open(input_config, "r") as f:
        config = json.load(f)

    existing = config.get("quantization_config")
    if existing is not None and not full:
        # only regularize the ignore list, leaving the group/format definitions unchanged
        old_n = len(existing.get("ignore", []))
        existing["ignore"] = generate_ignore(config)
        action = f"regularized ignore: {old_n} -> {len(existing['ignore'])} entries"
    else:
        config["quantization_config"] = generate_quant_config(config)
        action = f"generated full quantization_config (ignore: {len(config['quantization_config']['ignore'])} entries)"

    with open(output_config, "w") as f:
        json.dump(config, f, indent=2)
    logging.info("%s. Written: %s", action, output_config)


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate/regularize lingv25 quantization_config.ignore.")
    parser.add_argument("--config", type=str, required=True,
                        help="Input config.json (an existing runtime config, or the bare HF config).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output config.json. Defaults to --config (in-place).")
    parser.add_argument("--full", action="store_true",
                        help="Regenerate the entire quantization_config (not just the ignore list). "
                             "Use when bootstrapping from a bare HF config that has none.")
    args = parser.parse_args()
    main(args.config, args.output, args.full)
