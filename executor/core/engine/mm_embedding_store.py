# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Short-lived visual embeddings between MM encode and language prefill."""

from typing import Iterable, Optional, Sequence

import torch


class MMEmbeddingStore:
    def __init__(self):
        self._entries: dict[int, Optional[torch.Tensor]] = {}

    def put_many(
        self,
        request_ids: Sequence[int],
        embeddings: Sequence[Optional[torch.Tensor]],
    ) -> None:
        if len(request_ids) != len(embeddings):
            raise ValueError("MM encoder outputs must match requests")
        self._entries.update(zip(request_ids, embeddings))

    def pop_many(self, request_ids: Iterable[int]) -> Optional[torch.Tensor]:
        embeddings = [self._entries.pop(request_id) for request_id in request_ids]
        embeddings = [embedding for embedding in embeddings if embedding is not None]
        if not embeddings:
            return None
        return embeddings[0] if len(embeddings) == 1 else torch.cat(embeddings, dim=0)

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)
