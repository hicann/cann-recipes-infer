# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

"""Compatibility loader for the tokenizer shipped with LongCat-Next."""

from transformers import AutoTokenizer


class LongcatNextTokenizer:
    """Load the checkpoint tokenizer with the required Mistral regex fix."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        kwargs.setdefault("fix_mistral_regex", True)
        return AutoTokenizer.from_pretrained(*args, **kwargs)
