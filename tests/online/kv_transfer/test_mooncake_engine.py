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

"""Tests for MooncakeAscendTransferEngine via mocks (no real Mooncake install required)."""

import os
import sys
import types
import unittest
from unittest import mock

from executor.online.kv_transfer import transfer_engine as te


def _install_fake_mooncake(*, init_returns=0, rpc_port=23456):
    """Replace mooncake.engine.TransferEngine with a recording stub."""
    captured = {}

    class _FakeEngine:
        @staticmethod
        def initialize(*args, **kwargs):
            # 4 positional args expected: (local_hostname, "P2PHANDSHAKE", "ascend", device_name)
            captured["init_args"] = args
            captured["init_kwargs"] = kwargs
            return init_returns

        @staticmethod
        def get_rpc_port():
            return rpc_port

        @staticmethod
        def register_memory(ptr, length):
            captured.setdefault("registered", []).append((int(ptr), int(length)))
            return 0

        @staticmethod
        def batch_transfer_sync_write(target, srcs, dsts, lengths):
            captured.setdefault("transfers", []).append(
                (target, list(srcs), list(dsts), list(lengths))
            )
            return 0

    engine_mod = types.ModuleType("mooncake.engine")
    engine_mod.TransferEngine = _FakeEngine
    pkg = types.ModuleType("mooncake")
    pkg.engine = engine_mod
    return pkg, engine_mod, captured


class TestMooncakeEngineInit(unittest.TestCase):

    def setUp(self):
        # Stash + restore env around tests.
        self._saved_env = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")

    def tearDown(self):
        if self._saved_env is None:
            os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)
        else:
            os.environ["ASCEND_RT_VISIBLE_DEVICES"] = self._saved_env

    def test_initialize_passes_p2phandshake_and_ascend_literals(self):
        pkg, eng, captured = _install_fake_mooncake()
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "2"
        with mock.patch.dict(sys.modules, {"mooncake": pkg, "mooncake.engine": eng}):
            engine = te.MooncakeAscendTransferEngine(
                hostname="10.0.0.1",
                disaggregation_mode="PREFILL",
                local_rank=0,
                device_name="",
            )
            engine.initialize()

        args = captured["init_args"]
        self.assertEqual(len(args), 4)
        local_hostname, mode_flag, protocol, device_name = args
        # 1-segment local_hostname (just the IP) — matches HIXL wiki demo and
        # vllm-ascend; Mooncake picks the physical NPU id from current thread's
        # Ascend context, not from local_hostname.
        self.assertEqual(local_hostname, "10.0.0.1")
        # Literal flags — no metadata URL.
        self.assertEqual(mode_flag, "P2PHANDSHAKE")
        self.assertEqual(protocol, "ascend")
        self.assertEqual(device_name, "")

    def test_initialize_failure_raises_with_helpful_message(self):
        pkg, eng, _ = _install_fake_mooncake(init_returns=-1)
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
        with mock.patch.dict(sys.modules, {"mooncake": pkg, "mooncake.engine": eng}):
            engine = te.MooncakeAscendTransferEngine(
                hostname="10.0.0.1", disaggregation_mode="PREFILL", local_rank=0,
            )
            with self.assertRaises(RuntimeError) as ctx:
                engine.initialize()
        self.assertIn("Mooncake Ascend backend not available", str(ctx.exception))
        self.assertIn("USE_ASCEND_DIRECT", str(ctx.exception))

    def test_rank_port_consumed_from_get_rpc_port_not_placeholder(self):
        pkg, eng, captured = _install_fake_mooncake(rpc_port=98765)
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "1"
        with mock.patch.dict(sys.modules, {"mooncake": pkg, "mooncake.engine": eng}):
            engine = te.MooncakeAscendTransferEngine(
                hostname="10.0.0.1", disaggregation_mode="PREFILL", local_rank=0,
            )
            engine.initialize()
        # engine.rank_port must be the value Mooncake returned from get_rpc_port
        # (Mooncake assigns the real RPC port via findAvailableTcpPort).
        self.assertEqual(engine.rank_port, 98765)

    def test_session_id_set_after_initialize(self):
        """
        conn.py:send_metadata unconditionally reads transfer_engine.session_id;
        memfabric's AscendTransferEngine sets it as "<hostname>:<rpc_port>".
        Mooncake backend must do the same or decode-side AttributeError at runtime.
        """
        pkg, eng, _ = _install_fake_mooncake(rpc_port=54321)
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
        with mock.patch.dict(sys.modules, {"mooncake": pkg, "mooncake.engine": eng}):
            engine = te.MooncakeAscendTransferEngine(
                hostname="10.0.0.1", disaggregation_mode="PREFILL", local_rank=0,
            )
            engine.initialize()
        self.assertTrue(hasattr(engine, "session_id"))
        self.assertEqual(engine.session_id, "10.0.0.1:54321")

    def test_batch_transfer_sync_forwards_to_mooncake_write(self):
        pkg, eng, captured = _install_fake_mooncake()
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
        with mock.patch.dict(sys.modules, {"mooncake": pkg, "mooncake.engine": eng}):
            engine = te.MooncakeAscendTransferEngine(
                hostname="10.0.0.1", disaggregation_mode="PREFILL", local_rank=0,
            )
            engine.initialize()
            engine.batch_transfer_sync(
                "10.0.0.2:23456", [0xAA], [0xBB], [128],
            )
        self.assertEqual(captured["transfers"], [("10.0.0.2:23456", [0xAA], [0xBB], [128])])


if __name__ == "__main__":
    unittest.main()
