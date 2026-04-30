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

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, Tuple

from fastapi import FastAPI, HTTPException, Request

logger = logging.getLogger(__name__)


@dataclass
class BootstrapState:
    """Bootstrap server state.

    Responsibilities (and ONLY these):
    - Prefill rank registration (rank_table)
    - Topology query (attn/pp sizes, ranks)
    - Per-request prefill DP rank routing (room_to_dp_rank)

    Transfer state (room_to_transfer_info / status / metadata) is NOT handled
    here; it flows directly prefill → decode via ZMQ (sync_status) and RDMA.
    """
    yaml_dict: dict
    attn_tp_size: int = field(init=False)
    attn_cp_size: int = field(init=False)
    attn_dp_size: int = field(init=False)
    block_size: int = field(init=False, default=1)
    kv_cache_dtype: str = field(init=False, default="bfloat16")
    rank_table: Dict[Tuple[int, int, int], Dict[str, int | str]] = field(
        default_factory=dict
    )
    room_to_dp_rank: Dict[int, int] = field(default_factory=dict)
    expected_count: int = field(init=False)

    def __post_init__(self) -> None:
        # Python 3.10+: asyncio.Lock() defers loop binding until first
        # acquire, so creating here in the parent thread and acquiring in
        # uvicorn's daemon-thread loop is safe.
        self.lock = asyncio.Lock()
        parallel_config = self.yaml_dict.get("parallel_config", {})
        self.attn_tp_size = int(parallel_config.get("attn_tp_size", 1))
        self.attn_cp_size = int(parallel_config.get("cp_size", 1))
        world_size = int(parallel_config.get("world_size", 1))
        self.attn_dp_size = max(
            1,
            world_size // max(1, self.attn_tp_size * self.attn_cp_size),
        )
        self.expected_count = (
            self.attn_tp_size * self.attn_cp_size * self.attn_dp_size
        )
        logger.info(
            "BootstrapState init: world_size=%d attn_tp_size=%d cp_size=%d "
            "attn_dp_size=%d expected_count=%d",
            world_size, self.attn_tp_size, self.attn_cp_size,
            self.attn_dp_size, self.expected_count,
        )

    @property
    def is_ready(self) -> bool:
        return len(self.rank_table) >= self.expected_count


def init_bootstrap(app: FastAPI, yaml_dict: dict) -> BootstrapState:
    state = BootstrapState(yaml_dict)

    @app.put("/bootstrap/register_rank")
    async def register_rank(request: Request):
        payload = await request.json()
        try:
            dp = int(payload.get("attn_dp_rank", payload.get("system_dp_rank", 0)))
            cp = int(payload.get("attn_cp_rank", 0))
            tp = int(payload.get("attn_tp_rank", 0))
            ip = str(payload["rank_ip"])
            port = int(payload["rank_port"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Missing rank registration fields") from exc
        state.block_size = int(payload.get("block_size", state.block_size))
        state.kv_cache_dtype = str(payload.get("kv_cache_dtype", state.kv_cache_dtype))
        async with state.lock:
            state.rank_table[(dp, cp, tp)] = {"rank_ip": ip, "rank_port": port}
            size = len(state.rank_table)
        logger.info(
            "register_rank (dp=%d, cp=%d, tp=%d) %s:%d -> table size %d/%d",
            dp, cp, tp, ip, port, size, state.expected_count,
        )
        return {"status": "ok"}

    @app.get("/bootstrap/route")
    async def route():
        async with state.lock:
            if not state.is_ready:
                raise HTTPException(status_code=503, detail="bootstrap ranks not ready")
            return {
                "attn_tp_size": state.attn_tp_size,
                "attn_cp_size": state.attn_cp_size,
                "dp_size": state.attn_dp_size,
                "block_size": state.block_size,
                "kv_cache_dtype": state.kv_cache_dtype,
                "ranks": {
                    f"{dp},{cp},{tp}": endpoint
                    for (dp, cp, tp), endpoint in state.rank_table.items()
                },
            }

    @app.post("/bootstrap/register_dp_rank")
    async def register_dp_rank(request: Request):
        payload = await request.json()
        try:
            bootstrap_room = int(payload["bootstrap_room"])
            dp_rank = int(payload["dp_rank"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Missing bootstrap_room or dp_rank") from exc
        async with state.lock:
            state.room_to_dp_rank[bootstrap_room] = dp_rank
        return {"status": "ok"}

    @app.post("/bootstrap/query_dp_ranks")
    async def query_dp_ranks(request: Request):
        payload = await request.json()
        bootstrap_rooms = (
            payload.get("bootstrap_rooms", []) if isinstance(payload, dict) else payload
        ) or []
        async with state.lock:
            return {
                str(room): state.room_to_dp_rank.get(room, -1)
                for room in bootstrap_rooms
            }

    @app.get("/bootstrap/health")
    async def health():
        return {"status": "ok", "ready": state.is_ready}

    return state
