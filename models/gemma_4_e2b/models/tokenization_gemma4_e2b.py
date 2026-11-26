# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Adapted from HuggingFace Transformers v5.5.0 Gemma4 (Apache-2.0).
# Copyright 2026 the HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Keep base E2B raw completion semantics through the chat-oriented scheduler."""
from transformers import AutoTokenizer

RAW_COMPLETION_TEMPLATE = (
    "{% if messages|length != 1 or messages[0]['role'] != 'user' %}"
    "{{ raise_exception('Base E2B accepts one plain-text user prompt, not multi-turn chat') }}"
    "{% endif %}"
    "{% if messages[0]['content'] is not string %}"
    "{{ raise_exception('Base E2B accepts text content only') }}{% endif %}"
    "{% if enable_thinking|default(false) %}"
    "{{ raise_exception('Base E2B has no instruction thinking template') }}{% endif %}"
    "{{ messages[0]['content'] }}"
)


class Gemma4E2BTokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(*args, **kwargs)
        if not tokenizer.chat_template:
            # No extra BOS: the original tokenizer's postprocessor owns special tokens.
            tokenizer.chat_template = RAW_COMPLETION_TEMPLATE
        return tokenizer
