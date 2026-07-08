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
    def _decode_manager(self, **attrs):
        # validate_prefill_info reads a handful of instance attrs; construct via
        # __new__ and set only what the method touches so we don't need the full
        # KVTransferManager init (transfer engine, sockets, ...).
        mgr = KVTransferManager.__new__(KVTransferManager)
        mgr.kv_block_sizes = [32]
        mgr.kv_cache_dtype = "bfloat16"
        mgr.attn_tp_size = 1
        mgr.attn_cp_size = 1
        mgr.max_prefill_tokens = 4096
        for key, value in attrs.items():
            setattr(mgr, key, value)
        return mgr

    def test_validate_prefill_info_rejects_block_size_mismatch(self):
        mgr = self._decode_manager(kv_block_sizes=[32])
        info = PrefillServerInfo(
            attn_tp_size=1,
            attn_cp_size=1,
            dp_size=1,
            block_sizes=[64],
            kv_cache_dtype="bfloat16",
        )

        with self.assertRaises(ValueError):
            mgr.validate_prefill_info(info)

    def test_validate_prefill_info_rejects_kv_dtype_mismatch(self):
        mgr = self._decode_manager()
        info = PrefillServerInfo(
            attn_tp_size=1,
            attn_cp_size=1,
            dp_size=1,
            block_sizes=[32],
            kv_cache_dtype="float16",
        )

        with self.assertRaises(ValueError):
            mgr.validate_prefill_info(info)

    def test_validate_prefill_info_accepts_matching_geometry(self):
        mgr = self._decode_manager()
        info = PrefillServerInfo(
            attn_tp_size=1,
            attn_cp_size=1,
            dp_size=1,
            block_sizes=[32],
            kv_cache_dtype="bfloat16",
            max_prefill_tokens=4096,
        )

        mgr.validate_prefill_info(info)

    def test_validate_prefill_info_rejects_decode_cap_below_prefill(self):
        # Decode's admission cap (2048) is stricter than prefill's (4096): a
        # prompt of length 3000 would be admitted by prefill but rejected by
        # decode, and prefill has no way to learn of the decode-side reject ->
        # it would RDMA KV to a room decode never set up a receiver for. Must
        # fail fast at bootstrap-info validation.
        mgr = self._decode_manager(max_prefill_tokens=2048)
        info = PrefillServerInfo(
            attn_tp_size=1,
            attn_cp_size=1,
            dp_size=1,
            block_sizes=[32],
            kv_cache_dtype="bfloat16",
            max_prefill_tokens=4096,
        )

        with self.assertRaises(ValueError):
            mgr.validate_prefill_info(info)

    def test_validate_prefill_info_accepts_decode_cap_above_prefill(self):
        # Decode more lenient than prefill is safe: anything prefill admits,
        # decode also admits; the reverse (prefill rejects) is covered by the
        # existing prefill->decode KVPoll.Failed notify.
        mgr = self._decode_manager(max_prefill_tokens=8192)
        info = PrefillServerInfo(
            attn_tp_size=1,
            attn_cp_size=1,
            dp_size=1,
            block_sizes=[32],
            kv_cache_dtype="bfloat16",
            max_prefill_tokens=4096,
        )

        mgr.validate_prefill_info(info)

    def test_validate_prefill_info_skips_cap_check_when_prefill_predates_field(self):
        # Rolling deploy: an older prefill that doesn't publish max_prefill_tokens
        # sends 0. The cap check must be skipped rather than false-alarm.
        mgr = self._decode_manager(max_prefill_tokens=2048)
        info = PrefillServerInfo(
            attn_tp_size=1,
            attn_cp_size=1,
            dp_size=1,
            block_sizes=[32],
            kv_cache_dtype="bfloat16",
            max_prefill_tokens=0,
        )

        mgr.validate_prefill_info(info)


if __name__ == "__main__":
    unittest.main()
