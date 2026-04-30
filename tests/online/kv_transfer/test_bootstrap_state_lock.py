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

import asyncio
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from executor.online.bootstrap import init_bootstrap


class TestBootstrapStateLock(unittest.TestCase):
    def test_bootstrap_state_has_async_lock(self):
        app = FastAPI()
        state = init_bootstrap(
            app,
            {"parallel_config": {"world_size": 1, "attn_tp_size": 1}},
        )
        self.assertIsInstance(state.lock, asyncio.Lock)

    def test_register_dp_rank_and_query_still_work(self):
        app = FastAPI()
        init_bootstrap(
            app,
            {"parallel_config": {"world_size": 1, "attn_tp_size": 1}},
        )
        client = TestClient(app)
        resp = client.post(
            "/bootstrap/register_dp_rank", json={"bootstrap_room": 1, "dp_rank": 3}
        )
        self.assertEqual(resp.status_code, 200)
        query = client.post("/bootstrap/query_dp_ranks", json={"bootstrap_rooms": [1]})
        self.assertEqual(query.json(), {"1": 3})


if __name__ == "__main__":
    unittest.main()
