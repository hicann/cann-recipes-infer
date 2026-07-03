# coding=utf-8
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/tokenizers/registry.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

"""Lazy tokenizer registry."""

import importlib
import logging
from dataclasses import dataclass, field

from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

_TOKENIZER_SPECS = {
    "deepseek_v4": ("models.deepseek_v4.utils.tokenizer", "DeepseekV4Tokenizer"),
}


@dataclass
class _TokenizerRegistry:
    tokenizers: dict[str, tuple[str, str]] = field(default_factory=dict)

    def register(self, model_name: str, module: str, class_name: str) -> None:
        # This mutates the module-level singleton registry. Register custom
        # tokenizers before forking worker processes to avoid per-process drift.
        if model_name in self.tokenizers:
            logger.warning(
                "%s.%s is already registered for model_name=%r and will be overwritten.",
                module,
                class_name,
                model_name,
            )
        self.tokenizers[model_name] = (module, class_name)

    def load_tokenizer_cls(self, model_name: str):
        if model_name not in self.tokenizers:
            return None
        module, class_name = self.tokenizers[model_name]
        return getattr(importlib.import_module(module), class_name)

    def load_tokenizer(self, model_name: str, *args, **kwargs):
        tokenizer_cls = self.load_tokenizer_cls(model_name)
        if tokenizer_cls is None:
            return AutoTokenizer.from_pretrained(*args, **kwargs)
        return tokenizer_cls.from_pretrained(*args, **kwargs)


# Intentional singleton pattern: tokenizer specs are shared through this
# module-level registry and may be extended by register().
TokenizerRegistry = _TokenizerRegistry(_TOKENIZER_SPECS.copy())


def get_tokenizer(model_name: str, *args, **kwargs):
    return TokenizerRegistry.load_tokenizer(model_name, *args, **kwargs)
