# coding=utf-8
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/tokenizers/deepseek_v4.py
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

"""DeepSeek-V4 tokenizer wrapper."""

import copy
from typing import Any

from transformers import PreTrainedTokenizerFast

from .encoding_dsv4 import encode_messages


def get_deepseek_v4_tokenizer(tokenizer):
    dsv4_tokenizer = copy.copy(tokenizer)
    added_vocab = tokenizer.get_added_vocab()
    added_vocab_size = len(added_vocab)
    tokenizer_vocab_size = tokenizer.vocab_size

    class _DeepseekV4Tokenizer(tokenizer.__class__):
        def apply_chat_template(
            self,
            conversation: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            tokenize: bool = True,
            **kwargs,
        ):
            thinking = kwargs.get("thinking", False) or kwargs.get("enable_thinking", False)
            thinking_mode = "thinking" if thinking else "chat"

            messages = copy.deepcopy(conversation)
            if tools:
                messages.insert(0, {"role": "system", "tools": tools})

            reasoning_effort = kwargs.get("reasoning_effort")
            if not isinstance(reasoning_effort, str):
                reasoning_effort = None
            elif reasoning_effort == "none":
                thinking_mode = "chat"
                reasoning_effort = None
            elif reasoning_effort in ("max", "xhigh"):
                reasoning_effort = "max"
            else:
                reasoning_effort = "high"

            prompt = encode_messages(
                messages,
                thinking_mode=thinking_mode,
                drop_thinking=kwargs.get("drop_thinking", True),
                reasoning_effort=reasoning_effort,
            )

            if not tokenize:
                return prompt

            tokenizer_kwargs = {
                key: kwargs[key] for key in ("truncation", "max_length") if key in kwargs
            }
            return self.encode(prompt, add_special_tokens=False, **tokenizer_kwargs)

        def num_special_tokens_to_add(self, *args, **kwargs) -> int:
            return len(self.encode(""))

        def __len__(self) -> int:
            return tokenizer_vocab_size + added_vocab_size

        def get_added_vocab(self) -> dict[str, int]:
            return added_vocab.copy()

        def __reduce__(self):
            return get_deepseek_v4_tokenizer, (tokenizer,)

    _DeepseekV4Tokenizer.__name__ = f"DSV4{tokenizer.__class__.__name__}"
    dsv4_tokenizer.__class__ = _DeepseekV4Tokenizer
    return dsv4_tokenizer


class DeepseekV4Tokenizer:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        tokenizer = PreTrainedTokenizerFast.from_pretrained(*args, **kwargs)
        return get_deepseek_v4_tokenizer(tokenizer)
