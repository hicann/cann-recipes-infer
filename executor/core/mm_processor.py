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

"""Base contract for model-specific multimodal request processors."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from executor.core.forward_data_info import Request


class BaseMMProcessor(ABC):
    """Convert a tokenized request into its final multimodal representation."""

    def __init__(
        self,
        model_path: str,
        tokenizer,
    ):
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.input_truncated_len = None

    def should_defer_truncation(self, prompt) -> bool:
        """Whether this prompt must be truncated after multimodal expansion."""
        return False

    @abstractmethod
    def process(self, request: "Request") -> None:
        """Update input IDs, MM inputs, and MM token count in place."""
