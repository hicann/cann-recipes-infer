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

import unittest

from executor.online.kv_transfer import KVTransferManager
from executor.online.kv_transfer import (
    PrefillServerInfo,
)


class TestKVManagerValidation(unittest.TestCase):
    def test_validate_prefill_info_rejects_block_size_mismatch(self):
        kv_transfer_manager = KVTransferManager.__new__(KVTransferManager)
        kv_transfer_manager.block_size = 32
        kv_transfer_manager.attn_tp_size = 1
        kv_transfer_manager.attn_cp_size = 1
        info = PrefillServerInfo(
            attn_tp_size=1,
            attn_cp_size=1,
            dp_size=1,
            block_size=64,
            kv_cache_dtype="bfloat16",
        )

        with self.assertRaises(ValueError):
            kv_transfer_manager.validate_prefill_info(info)

    def test_validate_prefill_info_rejects_kv_dtype_mismatch(self):
        kv_transfer_manager = KVTransferManager.__new__(KVTransferManager)
        kv_transfer_manager.block_size = 32
        kv_transfer_manager.kv_cache_dtype = "bfloat16"
        kv_transfer_manager.attn_tp_size = 1
        kv_transfer_manager.attn_cp_size = 1
        info = PrefillServerInfo(
            attn_tp_size=1,
            attn_cp_size=1,
            dp_size=1,
            block_size=32,
            kv_cache_dtype="float16",
        )

        with self.assertRaises(ValueError):
            kv_transfer_manager.validate_prefill_info(info)

    def test_validate_prefill_info_accepts_matching_geometry(self):
        kv_transfer_manager = KVTransferManager.__new__(KVTransferManager)
        kv_transfer_manager.block_size = 32
        kv_transfer_manager.kv_cache_dtype = "bfloat16"
        kv_transfer_manager.attn_tp_size = 1
        kv_transfer_manager.attn_cp_size = 1
        info = PrefillServerInfo(
            attn_tp_size=1,
            attn_cp_size=1,
            dp_size=1,
            block_size=32,
            kv_cache_dtype="bfloat16",
        )

        kv_transfer_manager.validate_prefill_info(info)


if __name__ == "__main__":
    unittest.main()
