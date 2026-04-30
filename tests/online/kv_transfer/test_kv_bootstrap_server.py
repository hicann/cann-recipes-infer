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

from fastapi import FastAPI
from fastapi.testclient import TestClient

from executor.online.bootstrap import init_bootstrap


def _register_payload(dp=0, cp=0, tp=0, ip="127.0.0.1", port=34567):
    return {
        "attn_dp_rank": dp,
        "attn_cp_rank": cp,
        "attn_tp_rank": tp,
        "rank_ip": ip,
        "rank_port": port,
    }


def test_route_returns_503_before_ready():
    app = FastAPI()
    init_bootstrap(
        app, {"parallel_config": {"world_size": 2, "attn_tp_size": 1}}
    )
    client = TestClient(app)

    resp = client.get("/bootstrap/route")
    assert resp.status_code == 503


def test_register_rank_then_route_ready():
    app = FastAPI()
    init_bootstrap(
        app, {"parallel_config": {"world_size": 1, "attn_tp_size": 1}}
    )
    client = TestClient(app)

    put_resp = client.put("/bootstrap/register_rank", json=_register_payload())
    assert put_resp.status_code == 200

    get_resp = client.get("/bootstrap/route")
    assert get_resp.status_code == 200
    data = get_resp.json()
    assert data["ranks"]["0,0,0"] == {
        "rank_ip": "127.0.0.1",
        "rank_port": 34567,
    }


def test_register_and_query_dp_rank():
    app = FastAPI()
    init_bootstrap(
        app, {"parallel_config": {"world_size": 1, "attn_tp_size": 1}}
    )
    client = TestClient(app)

    register_resp = client.post(
        "/bootstrap/register_dp_rank",
        json={"bootstrap_room": 99, "dp_rank": 3},
    )
    assert register_resp.status_code == 200

    query_resp = client.post("/bootstrap/query_dp_ranks", json=[99, 100])
    assert query_resp.status_code == 200
    assert query_resp.json() == {"99": 3, "100": -1}
