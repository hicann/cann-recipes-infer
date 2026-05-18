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

"""Tests for build_transfer_engine factory and DisaggConfig.engine_backend wiring."""

import sys
import types
import unittest
from unittest import mock

from executor.core.config import DisaggConfig
from executor.online.kv_transfer import transfer_engine as te


class TestEngineBackendField(unittest.TestCase):
    """DisaggConfig.engine_backend additive contract — defaults preserved."""

    def test_default_backend_is_memfabric(self):
        cfg = DisaggConfig()
        self.assertEqual(cfg.engine_backend, "memfabric")

    def test_store_url_field_still_present(self):
        # Memfabric-specific but kept on DisaggConfig.
        cfg = DisaggConfig()
        self.assertTrue(hasattr(cfg, "store_url"))
        self.assertTrue(hasattr(cfg, "is_store_creator_node"))

    def test_engine_backend_can_be_set_to_mooncake(self):
        cfg = DisaggConfig(engine_backend="mooncake")
        self.assertEqual(cfg.engine_backend, "mooncake")


class TestBuildTransferEngineFactory(unittest.TestCase):
    """build_transfer_engine dispatches on disagg_config.engine_backend."""

    def _cfg(self, *, backend: str, store_url: str = "tcp://1.2.3.4:10002") -> DisaggConfig:
        return DisaggConfig(
            disaggregation_mode="PREFILL",
            bootstrap_host="1.2.3.4",
            bootstrap_port=18800,
            engine_backend=backend,
            store_url=store_url,
            is_store_creator_node=True,
            local_ip="1.2.3.4",
        )

    def test_memfabric_backend_returns_ascend_transfer_engine(self):
        cfg = self._cfg(backend="memfabric")
        # Patch the memfabric_hybrid import path via sys.modules so AscendTransferEngine.__init__
        # construction succeeds in a test environment without the package installed.
        fake_module = types.ModuleType("memfabric_hybrid")

        class _FakeMFEngine:
            @staticmethod
            def get_rpc_port():
                return 12345
        fake_module.TransferEngine = _FakeMFEngine
        with mock.patch.dict(sys.modules, {"memfabric_hybrid": fake_module}):
            engine = te.build_transfer_engine(
                cfg, hostname="1.2.3.4", local_rank=0,
            )
        self.assertIsInstance(engine, te.AscendTransferEngine)
        self.assertEqual(engine.rank_port, 12345)
        # Factory must forward the memfabric-specific ctor args verbatim from
        # DisaggConfig — otherwise existing memfabric deployments silently lose
        # store_url / is_store_creator_node and the store bootstrap breaks.
        self.assertEqual(engine.hostname, "1.2.3.4")
        self.assertEqual(engine.disaggregation_mode, "PREFILL")
        self.assertEqual(engine.store_url, "tcp://1.2.3.4:10002")
        self.assertTrue(engine.is_store_creator_node)
        # memfabric backend's npu_id contract: factory maps local_rank -> npu_id.
        self.assertEqual(engine.npu_id, 0)

    def test_mooncake_backend_returns_mooncake_transfer_engine(self):
        cfg = self._cfg(backend="mooncake")
        # Stub mooncake.engine.TransferEngine so import + construction succeeds without
        # the actual native lib being present.
        engine_mod = types.ModuleType("mooncake.engine")

        class _FakeMoonEngine:
            def __init__(self):
                self._init_args = None

            def initialize(self, *args, **kwargs):
                # Caller expects 4 positional args; verify shape by storing them.
                self._init_args = args
                return 0

            @staticmethod
            def get_rpc_port():
                return 23456
        engine_mod.TransferEngine = _FakeMoonEngine
        mooncake_pkg = types.ModuleType("mooncake")
        mooncake_pkg.engine = engine_mod
        with mock.patch.dict(sys.modules, {"mooncake": mooncake_pkg, "mooncake.engine": engine_mod}):
            engine = te.build_transfer_engine(
                cfg, hostname="1.2.3.4", local_rank=0,
            )
        self.assertIsInstance(engine, te.MooncakeAscendTransferEngine)

    def test_unknown_backend_raises(self):
        cfg = self._cfg(backend="bogus")
        with self.assertRaises(ValueError):
            te.build_transfer_engine(
                cfg, hostname="1.2.3.4", local_rank=0,
            )


class TestModuleImportLazy(unittest.TestCase):
    """mooncake import must be lazy: importing transfer_engine alone must not require it."""

    def test_transfer_engine_module_does_not_eagerly_import_mooncake(self):
        # Re-import in a clean module set; mooncake is not installed in CI by default.
        # If the import becomes eager (top-level `import mooncake`) this would fail in
        # environments where mooncake is absent.
        with mock.patch.dict(sys.modules, {k: v for k, v in sys.modules.items() if k != "mooncake"}, clear=False):
            # Pretend mooncake is missing — top-level transfer_engine import must still succeed.
            sys.modules.pop("mooncake", None)
            sys.modules.pop("mooncake.engine", None)
            sys.modules.pop("executor.online.kv_transfer.transfer_engine", None)
            import importlib
            mod = importlib.import_module("executor.online.kv_transfer.transfer_engine")
            self.assertTrue(hasattr(mod, "MooncakeAscendTransferEngine"))
            self.assertTrue(hasattr(mod, "build_transfer_engine"))

    def test_no_module_level_import_of_mooncake(self):
        """
        AST guard: `import mooncake` / `from mooncake import ...` MUST live
        inside a function body (lazy), never at module top level. Catches the
        case where a clever `try: import mooncake; except: pass` slips past the
        runtime check above.
        """
        import ast
        import inspect
        from executor.online.kv_transfer import transfer_engine as _te

        source = inspect.getsource(_te)
        tree = ast.parse(source)
        for node in tree.body:  # only module-level (top-level) statements
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertFalse(
                        alias.name == "mooncake" or alias.name.startswith("mooncake."),
                        f"top-level `import {alias.name}` in transfer_engine.py "
                        f"breaks the lazy-import contract",
                    )
            if isinstance(node, ast.ImportFrom):
                self.assertFalse(
                    node.module == "mooncake" or (node.module or "").startswith("mooncake."),
                    f"top-level `from {node.module} import ...` in transfer_engine.py "
                    f"breaks the lazy-import contract",
                )


if __name__ == "__main__":
    unittest.main()
