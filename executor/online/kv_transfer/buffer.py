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

"""Metadata buffer for RDMA direct write from prefill to decode.

Layout: one contiguous NPU HBM byte buffer of `capacity * SLOT_SIZE` bytes.
Each request owns a slot identified by `metadata_buffer_index`. Prefill, after
completing KV RDMA transfer, does a small RDMA write into the decode peer's
slot via memfabric (same device-to-device channel). Decode reads/unpacks from
the slot locally — no HTTP, no bootstrap server involvement.

NPU device required for memfabric DEVICE_RDMA — CPU pinned memory is NOT
reachable by the transfer engine's remote-write path.

Binary slot layout (STRUCT_FMT):
  - request_id: uint64 (8B)
  - bootstrap_room: uint64 (8B)
  - output_id: uint64 (8B)  (first decoded token id, or 0)
  - kv_len: uint64 (8B)
  - mtp_spec_count: uint32 (4B)         (number of valid spec tokens; 0 = MTP disabled)
  - mtp_spec_tokens: 7 x uint32 (28B)   (zero-padded — vocab fits in uint32)
Total = 64 bytes (single RDMA op, single registered region — keeps the
metadata RDMA path off memfabric's batch-many-small-regions corner case
that mis-fires on Ascend a2).
"""

from __future__ import annotations

import struct
import threading
from collections import deque
from typing import List, Optional

import torch


# Max number of MTP spec tokens transferred per request — fits in the 32B
# padding alongside a uint32 count.  Bump only by reshuffling the layout.
MAX_SPEC_TOKENS = 7
STRUCT_FMT = f"QQQQI{MAX_SPEC_TOKENS}I"          # 32 + 4 + 28 = 64B
SLOT_SIZE = struct.calcsize(STRUCT_FMT)
if SLOT_SIZE != 64:
    raise RuntimeError(f"SLOT_SIZE must be 64 bytes, got {SLOT_SIZE}")


def pack_metadata(
    request_id: int,
    bootstrap_room: int,
    output_id: int,
    kv_len: int,
    mtp_spec_tokens: Optional[List[int]] = None,
) -> bytes:
    spec = list(mtp_spec_tokens) if mtp_spec_tokens else []
    if len(spec) > MAX_SPEC_TOKENS:
        raise ValueError(
            f"mtp_spec_tokens length {len(spec)} exceeds MAX_SPEC_TOKENS={MAX_SPEC_TOKENS}"
        )
    count = len(spec)
    padded = spec + [0] * (MAX_SPEC_TOKENS - count)
    return struct.pack(
        STRUCT_FMT,
        request_id, bootstrap_room, output_id, kv_len,
        count, *padded,
    )


def unpack_metadata(raw: bytes) -> dict:
    fields = struct.unpack(STRUCT_FMT, raw)
    request_id, bootstrap_room, output_id, kv_len = fields[:4]
    count = int(fields[4])
    out = {
        "request_id": int(request_id),
        "output_bootstrap_room": int(bootstrap_room),
        "output_id": int(output_id),
        "kv_len": int(kv_len),
    }
    if count > 0:
        out["mtp_spec_tokens"] = [int(t) for t in fields[5:5 + count]]
    return out


class MetadataBufferPool:
    """Slot allocator over a contiguous pinned-CPU byte buffer.

    The buffer is registered with the transfer engine so prefill can RDMA-write
    directly into a slot. Decode reads the slot and unpacks the 32-byte layout.
    """

    def __init__(self, capacity: int):
        self.capacity = max(1, capacity)
        self._free = deque(range(self.capacity))
        self._lock = threading.Lock()
        self._on_device = False
        # Allocate on NPU HBM — memfabric DEVICE_RDMA requires device memory
        # on both ends (CPU pinned is not addressable by the transfer engine).
        try:
            self._buffer = torch.zeros(
                self.capacity * SLOT_SIZE, dtype=torch.uint8, device="npu",
            )
            self._on_device = True
        except Exception:
            # Fallback to CPU for tests / non-NPU environments.
            self._buffer = torch.zeros(self.capacity * SLOT_SIZE, dtype=torch.uint8)
        self._base_ptr = self._buffer.data_ptr()

    @property
    def base_ptr(self) -> int:
        return self._base_ptr

    @property
    def total_bytes(self) -> int:
        return self.capacity * SLOT_SIZE

    @property
    def slot_size(self) -> int:
        return SLOT_SIZE

    def slot_ptr(self, idx: int) -> int:
        return self._base_ptr + idx * SLOT_SIZE

    def alloc(self) -> int:
        with self._lock:
            if not self._free:
                raise RuntimeError("metadata buffer exhausted")
            return self._free.popleft()

    def available_size(self) -> int:
        with self._lock:
            return len(self._free)

    def free(self, idx: int) -> None:
        # Zero under the same lock that guards the freelist so a concurrent
        # alloc cannot hand this slot to another caller while we're still
        # wiping it (otherwise their write would be partially overwritten).
        with self._lock:
            if idx not in self._free:
                self._write_raw(idx, b"\x00" * SLOT_SIZE)
                self._free.append(idx)

    def read(self, idx: int) -> Optional[dict]:
        """Unpack the slot bytes. Returns None if the slot is all-zero (uninitialized)."""
        raw = self._read_raw(idx)
        if raw == b"\x00" * SLOT_SIZE:
            return None
        return unpack_metadata(raw)

    def write(self, idx: int, metadata: dict) -> None:
        """Pack metadata into the slot — used on the prefill side for local test mode."""
        raw = pack_metadata(
            int(metadata.get("request_id", 0)),
            int(metadata.get("output_bootstrap_room", 0)),
            int(metadata.get("output_id", 0)),
            int(metadata.get("kv_len", 0)),
            metadata.get("mtp_spec_tokens"),
        )
        self._write_raw(idx, raw)

    # ---- low-level byte IO ----
    def _write_raw(self, idx: int, raw: bytes) -> None:
        start = idx * SLOT_SIZE
        end = start + SLOT_SIZE
        # torch handles the host→device copy automatically if buffer is on NPU.
        src = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
        self._buffer[start:end] = src

    def _read_raw(self, idx: int) -> bytes:
        start = idx * SLOT_SIZE
        end = start + SLOT_SIZE
        slot = self._buffer[start:end]
        if self._on_device:
            slot = slot.cpu()
        return bytes(slot.tolist())
