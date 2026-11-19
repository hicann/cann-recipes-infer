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
"""Load Wan W8A8 experts from their config and safetensors index."""
import fnmatch
import inspect
import json
import logging
import re
from pathlib import Path

import torch
from safetensors import safe_open

from .int8 import Int8Linear

EXPERTS = ('high_noise_model', 'low_noise_model')
INDEX = 'diffusion_pytorch_model.safetensors.index.json'
TARGET = re.compile(r'^blocks[.]([0-9]+)[.](?:(self_attn|cross_attn)[.](q|k|v|o)|ffn[.](0|2))$')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def resolve_ignored(names, expert, patterns):
    result = set()
    for pattern in patterns:
        if any(pattern.startswith(other + '.') for other in EXPERTS if other != expert):
            continue
        matched = {
            name
            for name in names
            if fnmatch.fnmatchcase(name, pattern) or fnmatch.fnmatchcase(expert + '.' + name, pattern)
        }
        if not matched:
            raise ValueError(f'Ignored layer matches no {expert} Attention/FFN Linear: {pattern}')
        result.update(matched)
    return result


def validate_artifact_root(artifact_root):
    config = read_json(Path(artifact_root) / 'config.json')
    if config.get('quantization_config', {}).get('quant_method') != 'compressed-tensors':
        raise ValueError('Artifact config.json must declare quant_method=compressed-tensors')
    return config


def load_quantized_expert(model_cls, artifact_root, expert, source_root=None,
                          ignored_patterns=(), layout='nd_view'):
    folder = Path(artifact_root) / expert
    config = read_json(folder / 'config.json')
    qconfig = config.get('quantization_config', {})
    if qconfig.get('quant_method') != 'compressed-tensors':
        raise ValueError(f'{expert}: missing compressed-tensors configuration')
    groups = qconfig.get('config_groups', {})
    if not groups:
        raise ValueError(f'{expert}: quantization config_groups is empty')
    for group in groups.values():
        weights, inputs = group.get('weights', {}), group.get('input_activations', {})
        if (weights.get('num_bits') != 8 or weights.get('type') != 'int'
                or weights.get('strategy') != 'channel' or not weights.get('symmetric')
                or weights.get('dynamic') or inputs.get('num_bits') != 8
                or inputs.get('type') != 'int' or inputs.get('strategy') != 'token'
                or not inputs.get('symmetric') or not inputs.get('dynamic')):
            raise ValueError(f'{expert}: expected channel-weight / dynamic-token W8A8')
    index = read_json(folder / INDEX)['weight_map']
    parameters = inspect.signature(model_cls.__init__).parameters
    kwargs = {name: value for name, value in config.items() if name in parameters}
    with torch.device('meta'):
        model = model_cls(**kwargs)
        targets = {
            name
            for name, layer in model.named_modules()
            if TARGET.fullmatch(name) and isinstance(layer, torch.nn.Linear)
        }
        quantized = {name[:-13] for name in index if name.endswith('.weight_scale')}
        if not quantized or not quantized <= targets:
            raise ValueError(f'{expert}: INT8 scales must belong to Attention/FFN Linear layers')
        config_ignored = {
            name
            for name in targets
            if any(fnmatch.fnmatchcase(name, pattern) for pattern in qconfig.get('ignore', []))
        }
        if quantized & config_ignored:
            raise ValueError(f'{expert}: ignored layers must contain original BF16 tensors')
        overridden = resolve_ignored(targets, expert, ignored_patterns)
        fallback = quantized & overridden
        active = quantized - overridden
        for name in sorted(active):
            parent, attribute = name.rsplit('.', 1)
            old = model.get_submodule(name)
            setattr(model.get_submodule(parent), attribute,
                    Int8Linear(old.in_features, old.out_features, old.bias is not None,
                               layout, device='meta', prefix=expert + '.' + name))
    expected = model.state_dict()
    state = {}
    for shard in sorted(set(index.values())):
        with safe_open(str(folder / shard), framework='pt', device='cpu') as reader:
            for key in reader.keys():
                if index.get(key) != shard or key in state:
                    raise ValueError(f'{expert}: tensor index mismatch: {key}')
                if any(key == name + '.' + suffix for name in fallback
                       for suffix in ('weight', 'weight_scale', 'bias')):
                    continue
                state[key] = reader.get_tensor(key)
    if fallback:
        if source_root is None:
            raise ValueError('Runtime BF16 fallback requires the original checkpoint')
        source = Path(source_root) / expert
        source_index = read_json(source / INDEX)['weight_map']
        for name in sorted(fallback):
            for suffix in ('weight', 'bias'):
                key = name + '.' + suffix
                if key in expected:
                    with safe_open(str(source / source_index[key]), framework='pt', device='cpu') as reader:
                        state[key] = reader.get_tensor(key)
    if set(state) != set(expected):
        raise ValueError(f'{expert}: checkpoint tensor names differ from the model')
    for key, value in state.items():
        if value.shape != expected[key].shape:
            raise ValueError(f'{expert}.{key}: incompatible tensor shape')
        if value.dtype != expected[key].dtype:
            if value.dtype != torch.bfloat16 or expected[key].dtype != torch.float32:
                raise ValueError(f'{expert}.{key}: incompatible tensor dtype')
    model.load_state_dict(state, strict=True, assign=True)
    if hasattr(model, 'freqs'):
        from wan.modules.model import rope_params
        dimension = model.dim // model.num_heads
        with torch.device('cpu'):
            model.freqs = torch.cat([rope_params(1024, dimension - 4 * (dimension // 6)),
                                    rope_params(1024, 2 * (dimension // 6)),
                                    rope_params(1024, 2 * (dimension // 6))], dim=1)
        model.freqs_list = None
    logging.info('%s W8A8: %d INT8 Linear, %d BF16 target layers', expert, len(active), len(targets - active))
    return model.eval().requires_grad_(False)
