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

from executor.online.kv_transfer import MetadataBufferPool


def test_metadata_buffer_alloc_and_free():
    pool = MetadataBufferPool(capacity=2)
    idx = pool.alloc()
    assert idx == 0
    idx2 = pool.alloc()
    assert idx2 == 1
    pool.free(idx)
    # After free, the freed slot returns to the pool (order depends on deque ops).
    assert pool.alloc() in (0, 1)


def test_metadata_buffer_read_and_write():
    pool = MetadataBufferPool(capacity=1)
    idx = pool.alloc()
    pool.write(idx, {
        "request_id": 3,
        "output_bootstrap_room": 7,
        "output_id": 9,
        "kv_len": 11,
    })
    result = pool.read(idx)
    assert result == {
        "request_id": 3,
        "output_bootstrap_room": 7,
        "output_id": 9,
        "kv_len": 11,
    }


def test_metadata_buffer_zero_slot_returns_none():
    pool = MetadataBufferPool(capacity=1)
    idx = pool.alloc()
    # Slot is zero-initialized; read() returns None on all-zero (not yet written).
    assert pool.read(idx) is None


def test_metadata_buffer_with_spec_tokens():
    pool = MetadataBufferPool(capacity=1)
    idx = pool.alloc()
    pool.write(idx, {
        "request_id": 1,
        "output_bootstrap_room": 42,
        "output_id": 100,
        "kv_len": 16,
        "mtp_spec_tokens": [101, 102, 103],
    })
    result = pool.read(idx)
    assert result["mtp_spec_tokens"] == [101, 102, 103]
    assert result["request_id"] == 1
    assert result["output_bootstrap_room"] == 42
