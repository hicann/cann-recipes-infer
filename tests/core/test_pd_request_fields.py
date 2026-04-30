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

from executor.core.types_.types import Request


def test_request_has_pd_fields():
    req = Request(request_id=1, prompt="hello", input_ids=[1, 2, 3])

    assert req.bootstrap_room == -1
    assert req.bootstrap_addr == ""
    assert req.disagg_prefill_dp_rank == -1
    assert req.metadata_buffer_index == -1
    assert req.disagg_kv_sender is None
