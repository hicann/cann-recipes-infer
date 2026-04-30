# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import asyncio
import unittest
from unittest.mock import patch

from executor.online.router import PDDispatcher


class _Resp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class _RecordingClient:
    def __init__(self, responses, seen_urls):
        self._responses = list(responses)
        self._seen_urls = seen_urls

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json):
        del json
        self._seen_urls.append(url)
        return self._responses.pop(0)


class TestPDRouterSelection(unittest.TestCase):
    def test_router_selects_backend_by_bootstrap_room(self):
        responses = [
            _Resp(200, {"accepted": True, "request_id": 1, "bootstrap_room": 5}),
            _Resp(200, {"result": {"text": "ok"}}),
        ]
        seen_urls = []

        async def run_case():
            with patch(
                "executor.online.router.httpx.AsyncClient",
                side_effect=lambda *args, **kwargs: _RecordingClient(
                    responses, seen_urls
                ),
            ):
                router = PDDispatcher(
                    prefill_addrs=["prefill-0", "prefill-1"],
                    decode_addrs=["decode-0", "decode-1"],
                )
                await router.dispatch({"prompt": "hi", "bootstrap_room": 5}, path="/generate")

        asyncio.run(run_case())
        self.assertEqual(
            sorted(seen_urls),
            ["http://decode-1/generate", "http://prefill-1/generate"],
        )


if __name__ == "__main__":
    unittest.main()
