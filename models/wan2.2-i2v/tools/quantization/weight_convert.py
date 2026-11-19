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
"""Convert Wan Attention/FFN weights to symmetric channel-wise INT8."""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from quantization.loader import EXPERTS, INDEX, TARGET, read_json, resolve_ignored


def quantize_weight(weight):
    if weight.ndim != 2 or not weight.is_floating_point() or not bool(torch.isfinite(weight).all()):
        raise ValueError('Weight conversion requires a finite floating [N,K] tensor')
    matrix = weight.detach().to(device='cpu', dtype=torch.float32)
    scale = matrix.abs().amax(dim=1, keepdim=True) / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = torch.round(matrix / scale).clamp(-127, 127).to(torch.int8).contiguous()
    return quantized, scale.contiguous()


def compressed_config(ignored):
    # Weights use fixed per-channel scales; activations are quantized per token.
    weight_config = dict(num_bits=8, type='int', symmetric=True, strategy='channel', dynamic=False)
    activation_config = dict(num_bits=8, type='int', symmetric=True, strategy='token', dynamic=True)
    scheme = {
        'weights': weight_config,
        'input_activations': activation_config,
        'targets': ['Linear'],
    }
    return {
        'format': 'int-quantized',
        'ignore': sorted(ignored),
        'config_groups': {'wan_w8a8': scheme},
        'quant_method': 'compressed-tensors',
        'quantization_status': 'compressed',
    }


def write_json(path, value):
    temporary = Path(path).with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def convert_expert(source_root, output_root, expert, ignored_patterns):
    source, destination = Path(source_root) / expert, Path(output_root) / expert
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f'Output expert directory already contains files: {destination}')
    destination.mkdir(parents=True, exist_ok=True)
    config = read_json(source / 'config.json')
    index = read_json(source / INDEX)['weight_map']
    targets = {key[:-7] for key in index if key.endswith('.weight') and TARGET.fullmatch(key[:-7])}
    ignored = resolve_ignored(targets, expert, ignored_patterns)
    quantized = targets - ignored
    output_index, floating_linears, seen = {}, set(ignored), set()
    for shard in sorted(set(index.values())):
        exported = {}
        with safe_open(str(source / shard), framework='pt', device='cpu') as reader:
            for key in reader.keys():
                if key in seen or index.get(key) != shard:
                    raise ValueError(f'{expert}.{key}: duplicate tensor or source index mismatch')
                seen.add(key)
                value = reader.get_tensor(key)
                name = key[:-7] if key.endswith('.weight') else None
                if name in quantized:
                    exported[key], exported[name + '.weight_scale'] = quantize_weight(value)
                else:
                    exported[key] = value.contiguous()
                    if name and value.ndim == 2:
                        floating_linears.add(name)
                if key.endswith('.bias') and key[:-5] in quantized and value.dtype != torch.bfloat16:
                    raise ValueError(f'{expert}.{key}: expected original BF16 bias')
        temporary = destination / (shard + '.tmp')
        save_file(exported, str(temporary), metadata={'format': 'pt'})
        temporary.replace(destination / shard)
        output_index.update({key: shard for key in exported})
        print(f'{expert}: converted {shard}', flush=True)
    if seen != set(index):
        raise ValueError(f'{expert}: source checkpoint tensor set is incomplete')
    config['quantization_config'] = compressed_config(floating_linears)
    write_json(destination / 'config.json', config)
    write_json(destination / INDEX, {'weight_map': output_index})
    return config['quantization_config']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--ignored-layers', help='JSON list of exact layer names or glob patterns')
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Use a separate sibling directory for INT8 artifacts')
    ignored = read_json(args.ignored_layers) if args.ignored_layers else []
    if not isinstance(ignored, list) or not all(isinstance(name, str) for name in ignored):
        raise ValueError('ignored-layers must be a JSON list of strings')
    configs = {expert: convert_expert(source, output, expert, ignored) for expert in EXPERTS}
    write_json(output / 'config.json', {'quantization_config': compressed_config(
        [expert + '.' + layer for expert, config in configs.items() for layer in config['ignore']])})


if __name__ == '__main__':
    main()
