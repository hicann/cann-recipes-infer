# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import threading
import unittest

from executor.online.kv_transfer import KVTransferManager


class TestKVManagerCleanup(unittest.TestCase):
    def test_cleanup_room_clears_all_room_state(self):
        kv_transfer_manager = KVTransferManager.__new__(KVTransferManager)
        kv_transfer_manager.state_lock = threading.RLock()
        kv_transfer_manager.request_status = {7: 4}
        kv_transfer_manager.transfer_infos = {7: [{"dst_block_ids": {"FullAttention": [1]}}]}
        kv_transfer_manager.transfer_metadata = {7: {"output_id": 3}}
        kv_transfer_manager.prefill_response_tracker = {7: {0, 1}}
        kv_transfer_manager.room_to_addr = {7: "127.0.0.1:18800"}
        kv_transfer_manager.failure_records = {7: "boom"}
        kv_transfer_manager.addr_to_rooms_tracker = {"127.0.0.1:18800": {7, 8}}

        kv_transfer_manager.cleanup_room(7, bootstrap_addr="127.0.0.1:18800")

        self.assertNotIn(7, kv_transfer_manager.request_status)
        self.assertNotIn(7, kv_transfer_manager.transfer_infos)
        self.assertNotIn(7, kv_transfer_manager.transfer_metadata)
        self.assertNotIn(7, kv_transfer_manager.prefill_response_tracker)
        self.assertNotIn(7, kv_transfer_manager.room_to_addr)
        self.assertNotIn(7, kv_transfer_manager.failure_records)
        self.assertEqual(kv_transfer_manager.addr_to_rooms_tracker["127.0.0.1:18800"], {8})


if __name__ == "__main__":
    unittest.main()
