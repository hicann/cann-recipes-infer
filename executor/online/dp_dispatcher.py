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

"""DP Dispatcher for distributed inference request routing."""

import asyncio
import enum
import logging
import pickle
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional

import zmq

from executor.online.constants import zmq_ports_for_role

logger = logging.getLogger(__name__)


class ServerStatus(str, enum.Enum):
    """Server lifecycle states surfaced via /health."""
    Starting = "Starting"
    Up = "Up"
    UnHealthy = "UnHealthy"


class BackendUnavailableError(RuntimeError):
    """Dispatcher cannot enqueue to a worker (not ready / ZMQ send timeout).

    HTTP layer translates this to 503 via an app.exception_handler so clients
    see "service unavailable, retry" instead of 500 "internal error".
    """


@dataclass
class PendingRequest:
    request_id: int
    event: threading.Event
    result: Any = None
    error: Optional[str] = None


class DPDispatcher:
    """Distributed inference request dispatcher.

    Used on the online server process to:
    1. Distribute HTTP requests to DP leaders via ROUTER socket
    2. Collect finished results from DP leaders via PULL socket
    3. Manage pending requests and notify waiters
    """

    MAX_TERMINAL_REQUEST_CACHE = 1024
    REQUEST_TIMEOUT_SECS = 1800.0
    ENQUEUE_TIMEOUT_MS = 60000

    def __init__(
        self,
        dp_size: int,
        router_port: int,
        pull_port: int,
        disaggregation_mode: str = "NONE",
    ):
        self.dp_size = dp_size
        self.router_port = router_port
        self.pull_port = pull_port
        self.disaggregation_mode = disaggregation_mode

        self.ctx: Optional[zmq.Context] = None
        self.router: Optional[zmq.Socket] = None
        self.pull: Optional[zmq.Socket] = None

        self.pending_requests: Dict[int, PendingRequest] = {}
        self.terminal_requests: "OrderedDict[int, str]" = OrderedDict()
        self._request_id_counter = 0
        self._dp_counter = 0
        self._lock = threading.Lock()

        self._result_listener: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.server_status: ServerStatus = ServerStatus.Starting

    def start(self):
        """Start dispatcher sockets and the background result listener."""
        if zmq is None:
            raise RuntimeError("zmq is required to start DPDispatcher")
        self.ctx = zmq.Context()

        self.router = self.ctx.socket(zmq.ROUTER)
        self.router.setsockopt(zmq.SNDTIMEO, self.ENQUEUE_TIMEOUT_MS)
        self.router.bind(f"tcp://0.0.0.0:{self.router_port}")

        self.pull = self.ctx.socket(zmq.PULL)
        self.pull.bind(f"tcp://0.0.0.0:{self.pull_port}")

        self._stop_event.clear()
        self._result_listener = threading.Thread(
            target=self._result_listener_loop, daemon=True
        )
        self._result_listener.start()

        self.server_status = ServerStatus.Up
        logger.info(
            f"DPDispatcher started: dp_size={self.dp_size}, "
            f"ROUTER on 0.0.0.0:{self.router_port}, "
            f"PULL on 0.0.0.0:{self.pull_port}"
        )

    def stop(self):
        """Stop dispatcher resources and notify pending waiters."""
        self.server_status = ServerStatus.UnHealthy
        self._stop_event.set()

        if self._result_listener:
            self._result_listener.join(timeout=2.0)

        if self.router:
            self.router.close(linger=0)
        if self.pull:
            self.pull.close(linger=0)
        if self.ctx:
            self.ctx.term()

        with self._lock:
            for pending in self.pending_requests.values():
                pending.error = "Server shutdown"
                pending.event.set()

        logger.info("DPDispatcher stopped")

    def _get_next_dp_rank(self) -> int:
        """Round-robin selection of the next DP leader."""
        if self.dp_size == 1:
            return 0
        with self._lock:
            dp_rank = self._dp_counter
            self._dp_counter = (self._dp_counter + 1) % self.dp_size
            return dp_rank

    def _record_terminal_request(self, request_id: int, reason: str) -> None:
        """Cache recently completed or expired requests for late-result handling."""
        self.terminal_requests[request_id] = reason
        self.terminal_requests.move_to_end(request_id)
        while len(self.terminal_requests) > self.MAX_TERMINAL_REQUEST_CACHE:
            self.terminal_requests.popitem(last=False)

    async def submit_request(self, request_dict: dict) -> int:
        """Submit a request dictionary and return its request_id."""
        with self._lock:
            if "request_id" not in request_dict:
                request_id = self._request_id_counter
                self._request_id_counter += 1
            else:
                self._request_id_counter = max(self._request_id_counter, request_dict["request_id"] + 1)
            request_dict["request_id"] = request_id

            event = threading.Event()
            self.pending_requests[request_id] = PendingRequest(
                request_id=request_id,
                event=event,
            )

        dp_rank = self._get_next_dp_rank()
        data = pickle.dumps(request_dict)
        identity = str(dp_rank).encode()

        try:
            self.router.send_multipart([identity, data])
        except zmq.Again as exc:
            with self._lock:
                self.pending_requests.pop(request_id, None)
            raise BackendUnavailableError(
                f"Worker DP rank {dp_rank} not ready (enqueue timeout)"
            ) from exc

        return request_id

    async def wait_result(
        self, request_id: int, timeout: float = REQUEST_TIMEOUT_SECS
    ) -> Any:
        """Wait for a specific request result with timeout."""
        with self._lock:
            if request_id not in self.pending_requests:
                raise ValueError(f"Unknown request_id: {request_id}")
            pending = self.pending_requests[request_id]

        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(None, lambda: pending.event.wait(timeout))

        if not success:
            with self._lock:
                self.pending_requests.pop(request_id, None)
                self._record_terminal_request(request_id, "timeout")
            logger.warning(
                "request %s: timed out after %.1fs waiting for backend result",
                request_id, timeout,
            )
            raise asyncio.TimeoutError()

        with self._lock:
            result = pending.result
            error = pending.error
            self.pending_requests.pop(request_id, None)
            self._record_terminal_request(request_id, "error" if error else "completed")

        if error:
            logger.warning("request %s: failed with error: %s", request_id, error)
            raise RuntimeError(error)

        return result

    def get_pending_count(self) -> int:
        """Return the current number of pending requests."""
        with self._lock:
            return len(self.pending_requests)

    def _process_result_event(self, request_id: int, result: Any) -> None:
        with self._lock:
            pending = self.pending_requests.get(request_id)
            if pending is None:
                return
            pending.result = result
            pending.event.set()

    def _result_listener_loop(self):
        """Background thread polling results from the PULL socket."""
        poller = zmq.Poller()
        poller.register(self.pull, zmq.POLLIN)
        while not self._stop_event.is_set():
            try:
                if not poller.poll(100):  # 100ms timeout
                    continue
                parts = self.pull.recv_multipart(zmq.NOBLOCK)

                if len(parts) >= 3:
                    dp_rank = int(parts[0].decode())
                    request_id = pickle.loads(parts[1])
                    result = pickle.loads(parts[2])
                    self._process_result_event(request_id, result)
                    with self._lock:
                        if request_id not in self.pending_requests:
                            terminal_reason = self.terminal_requests.get(request_id)
                            if terminal_reason:
                                logger.info(
                                    "Ignoring late result for request_id %s from DP rank %s: %s",
                                    request_id, dp_rank, terminal_reason,
                                )
                            else:
                                logger.warning("Unknown request_id: %s", request_id)
                    logger.debug(
                        "Result for request %s from DP rank %s",
                        request_id, dp_rank,
                    )

            except zmq.Again:
                continue
            except Exception as e:
                if not self._stop_event.is_set():
                    logger.error(f"Error in result listener: {e}")
                    self.server_status = ServerStatus.UnHealthy
