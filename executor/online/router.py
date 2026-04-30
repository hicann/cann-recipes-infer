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

"""Minimal PD Router.

For each request:
  1. Inject bootstrap_room/host/port into the payload.
  2. Concurrently POST to one Prefill instance and one Decode instance.
  3. Return the Decode response verbatim (status + body) to the client.

No prefill-response business validation, no fail-over, no custom exceptions.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import List, Optional
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from executor.online.constants import BOOTSTRAP_PORT


def _normalize_base_url(addr: str) -> str:
    if "://" not in addr:
        addr = f"http://{addr}"
    return addr.rstrip("/")


def _request_url(addr: str, path: str) -> str:
    base = _normalize_base_url(addr)
    return base if base.endswith(path) else f"{base}{path}"


def _prefill_host(addr: str) -> str:
    """Hostname of a Prefill address (URL port is HTTP port, not bootstrap)."""
    return urlsplit(_normalize_base_url(addr)).hostname or "127.0.0.1"


class _PDRequest(BaseModel):
    bootstrap_room: Optional[int] = None
    bootstrap_host: Optional[str] = None
    bootstrap_port: Optional[int] = None


class GenerateRequest(_PDRequest):
    prompt: str


class CompletionRequest(_PDRequest):
    model: str
    prompt: str | List[str]
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stop: List[str] = []


class ChatCompletionRequest(_PDRequest):
    model: str
    messages: List[dict]
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stop: List[str] = []


class PDDispatcher:
    def __init__(
        self,
        prefill_addrs: List[str],
        decode_addrs: List[str],
        timeout: float = 1800.0,
        bootstrap_ports: Optional[List[int]] = None,
    ):
        ports = bootstrap_ports or [BOOTSTRAP_PORT] * len(prefill_addrs)
        self.prefill_targets = [
            {
                "url": addr,
                "bootstrap_host": _prefill_host(addr),
                "bootstrap_port": port,
            }
            for addr, port in zip(prefill_addrs, ports)
        ]
        self.decode_addrs = decode_addrs
        self.timeout = timeout

    @property
    def prefill_addrs(self) -> List[str]:
        return [t["url"] for t in self.prefill_targets]

    def _inject_pd_fields(self, request_dict: dict) -> dict:
        payload = dict(request_dict)
        if payload.get("bootstrap_room") is None:
            payload["bootstrap_room"] = uuid.uuid4().int & ((1 << 63) - 1)
        room = int(payload["bootstrap_room"])
        target_idx = room % len(self.prefill_targets)
        target = self.prefill_targets[target_idx]
        if payload.get("bootstrap_host") is None:
            payload["bootstrap_host"] = target["bootstrap_host"]
        if payload.get("bootstrap_port") is None:
            payload["bootstrap_port"] = target["bootstrap_port"]
        # Pre-compute prefill DP rank so decode side can skip the HTTP query.
        # When dp_size > 1, the DPDispatcher round-robins internally and the
        # exact dp_rank is unknown to the router. In that case we leave the
        # field unset and decode falls back to querying the bootstrap server.
        # For the common dp_size==1 case this eliminates the query entirely.
        payload.setdefault("disagg_prefill_dp_rank", 0)
        return payload

    def _select_decode(self, room: int) -> str:
        return self.decode_addrs[room % len(self.decode_addrs)]

    async def dispatch(self, request_dict: dict, path: str):
        payload = self._inject_pd_fields(request_dict)
        room = int(payload["bootstrap_room"])
        prefill_url = _request_url(
            self.prefill_targets[room % len(self.prefill_targets)]["url"], path,
        )
        decode_url = _request_url(self._select_decode(room), path)

        async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
            try:
                _, decode_resp = await asyncio.gather(
                    client.post(prefill_url, json=payload),
                    client.post(decode_url, json=payload),
                )
            except httpx.TransportError as exc:
                # Downstream prefill/decode not listening yet or connection
                # dropped mid-flight. Surface as 503 so clients retry instead
                # of seeing an opaque 500 "internal error".
                raise HTTPException(
                    status_code=503,
                    detail=f"backend unavailable: {exc}",
                ) from exc

        if decode_resp.status_code != 200:
            raise HTTPException(
                status_code=decode_resp.status_code, detail=decode_resp.text
            )
        try:
            result = decode_resp.json()
        except ValueError as exc:
            raise HTTPException(
                status_code=502, detail="backend returned non-JSON response"
            ) from exc
        if isinstance(result, dict) and "result" in result:
            return result["result"]
        return result


def create_router_app(
    prefill_addrs: List[str],
    decode_addrs: List[str],
    bootstrap_ports: Optional[List[int]] = None,
) -> FastAPI:
    app = FastAPI(title="PD Router")
    dispatcher = PDDispatcher(
        prefill_addrs=prefill_addrs,
        decode_addrs=decode_addrs,
        bootstrap_ports=bootstrap_ports,
    )

    @app.post("/generate")
    async def generate(request: GenerateRequest):
        return await dispatcher.dispatch(request.model_dump(), path="/generate")

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        if isinstance(request.prompt, list) and len(request.prompt) != 1:
            raise HTTPException(
                status_code=400, detail="PD Router currently supports a single prompt"
            )
        return await dispatcher.dispatch(
            request.model_dump(), path="/v1/completions"
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        return await dispatcher.dispatch(
            request.model_dump(), path="/v1/chat/completions"
        )

    @app.get("/stats")
    async def stats():
        return {
            "status": "running",
            "prefill_backends": len(dispatcher.prefill_addrs),
            "decode_backends": len(dispatcher.decode_addrs),
        }

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "prefill_addrs": dispatcher.prefill_addrs,
            "decode_addrs": dispatcher.decode_addrs,
            "time": int(time.time()),
        }

    return app
