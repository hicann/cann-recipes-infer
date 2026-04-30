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
from unittest.mock import Mock

from executor.core.config import DisaggConfig
from executor.online.kv_transfer import KVTransferManager


class TestKVManagerThreading(unittest.TestCase):
    def test_kv_manager_initializes_locks(self):
        kv_transfer_manager = KVTransferManager(
            disagg_config=DisaggConfig(
                disaggregation_mode="DECODE",
                store_url="tcp://127.0.0.1:10002",
                local_ip="127.0.0.1",
            ),
        )
        self.assertTrue(hasattr(kv_transfer_manager, "state_lock"))
        kv_transfer_manager.shutdown()

    def test_shutdown_closes_socket_and_stops_threads(self):
        kv_transfer_manager = KVTransferManager(
            disagg_config=DisaggConfig(
                disaggregation_mode="DECODE",
                store_url="tcp://127.0.0.1:10003",
                local_ip="127.0.0.1",
            ),
        )
        # Swap the real socket/context with mocks so we can assert
        # shutdown calls close()/term(). Tear the real ones down first
        # so they don't leak.
        real_socket = kv_transfer_manager.server_socket
        real_context = kv_transfer_manager.context
        if real_socket is not None:
            real_socket.close()
        if real_context is not None:
            real_context.term()
        kv_transfer_manager.server_socket = Mock()
        kv_transfer_manager.context = Mock()
        kv_transfer_manager.shutdown()
        kv_transfer_manager.server_socket.close.assert_called_once()
        kv_transfer_manager.context.term.assert_called_once()


if __name__ == "__main__":
    unittest.main()
