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

"""KV-cache transfer subsystem for PD disaggregation.

Modules:
- transfer_engine: KVPoll + data structures + TransferEngine ABC + AscendTransferEngine
- buffer: MetadataBufferPool (NPU-registered slots for RDMA metadata writes)
- transfer_manager: KVTransferManager (ZMQ listener + transfer executor + bootstrap client)
- conn: AscendKVSender / AscendKVReceiver (per-request)
"""

from .buffer import MetadataBufferPool, pack_metadata, unpack_metadata
from .conn import AscendKVReceiver, AscendKVSender
from .transfer_manager import KVTransferManager
from .transfer_engine import (
    AscendTransferEngine,
    KVArgsRegisterInfo,
    KVPoll,
    PrefillRankInfo,
    PrefillServerInfo,
    TargetRankMapping,
    TransferEngine,
    all_reduce_poll,
)

__all__ = [
    "AscendKVReceiver",
    "AscendKVSender",
    "AscendTransferEngine",
    "KVArgsRegisterInfo",
    "KVTransferManager",
    "KVPoll",
    "MetadataBufferPool",
    "PrefillRankInfo",
    "PrefillServerInfo",
    "TargetRankMapping",
    "TransferEngine",
    "all_reduce_poll",
    "pack_metadata",
    "unpack_metadata",
]
