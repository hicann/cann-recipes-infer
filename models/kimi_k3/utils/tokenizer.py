# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2026 The Moonshot AI Team. All rights reserved.
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

"""Compatibility loader for the tokenizer shipped with the Kimi K3 checkpoint."""

from transformers import AutoTokenizer


class KimiK3Tokenizer:
    """Load the checkpoint tokenizer and normalize its Transformers 5 behavior."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(*args, **kwargs)

        # K3 encodes its structural markers in encode/apply_chat_template.  It
        # has no CLS/SEP tokens, so the generic Transformers 5 default would
        # otherwise prepend and append None to every plain-text request.
        if (
            getattr(tokenizer, "special_tokens_pattern", None) == "cls_sep"
            and tokenizer.cls_token_id is None
            and tokenizer.sep_token_id is None
        ):
            tokenizer.special_tokens_pattern = "none"
        return tokenizer
