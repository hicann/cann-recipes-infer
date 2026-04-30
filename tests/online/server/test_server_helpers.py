# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import importlib
import unittest

from fastapi import HTTPException

from executor.online.server import (
    build_online_request_dict,
    load_server_app_config,
)


class TestServerHelpers(unittest.TestCase):
    def test_server_module_imports(self):
        module = importlib.import_module("executor.online.server")
        self.assertTrue(hasattr(module, "create_app"))

    def test_load_server_app_config_reads_parallel_config(self):
        cfg = load_server_app_config(
            {
                "parallel_config": {
                    "world_size": 8,
                    "attn_tp_size": 2,
                    "cp_size": 2,
                },
            },
            disaggregation_mode="PREFILL",
        )
        self.assertEqual(cfg.disaggregation_mode, "PREFILL")
        self.assertEqual(cfg.world_size, 8)
        self.assertEqual(cfg.dp_size, 2)

    def test_build_online_request_dict_carries_pd_fields(self):
        request_dict = build_online_request_dict(
            prompt="hello",
            sampling_params={"max_tokens": 4},
            bootstrap_room=7,
            bootstrap_host="127.0.0.1",
            bootstrap_port=18800,
            require_bootstrap=True,
        )
        self.assertEqual(request_dict["prompt"], "hello")
        self.assertEqual(request_dict["sampling_params"], {"max_tokens": 4})
        self.assertEqual(request_dict["bootstrap_room"], 7)
        self.assertEqual(request_dict["bootstrap_host"], "127.0.0.1")
        self.assertEqual(request_dict["bootstrap_port"], 18800)

    def test_build_online_request_dict_validates_prefill_fields(self):
        with self.assertRaises(HTTPException):
            build_online_request_dict(
                prompt="hello",
                sampling_params={},
                bootstrap_room=None,
                bootstrap_host=None,
                bootstrap_port=None,
                require_bootstrap=True,
            )


if __name__ == "__main__":
    unittest.main()
