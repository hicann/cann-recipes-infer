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

import argparse
import asyncio
from dataclasses import dataclass
from contextlib import asynccontextmanager
import logging
import multiprocessing as mp
import multiprocessing.connection  # noqa: F401 — mp.connection.Connection
import os
import sys
import threading
import time
import uuid
from typing import List, Optional

import yaml
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from executor.online.bootstrap import init_bootstrap
from executor.online.dp_dispatcher import BackendUnavailableError, ServerStatus
from executor.core.config.inference_config import DisaggConfig
from executor.online.constants import (
    BOOTSTRAP_PORT,
    DECODE_HTTP_PORT,
    PREFILL_HTTP_PORT,
    ROLE_DECODE,
    ROLE_NONE,
    ROLE_PREFILL,
    ROUTER_HTTP_PORT,
    zmq_ports_for_role,
)

from executor.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


WATCHDOG_INTERVAL_SECS = 5

mp.set_start_method("spawn", force=True)


@dataclass
class ServerAppConfig:
    disaggregation_mode: str = ROLE_NONE
    world_size: int = 1
    dp_size: int = 1


def load_server_app_config(
    yaml_dict: dict, disaggregation_mode: str = ROLE_NONE
) -> ServerAppConfig:
    parallel_config = yaml_dict.get("parallel_config", {})
    world_size = int(parallel_config.get("world_size", 1))
    tp_size = int(parallel_config.get("attn_tp_size", 1))
    cp_size = int(parallel_config.get("cp_size", 1))

    return ServerAppConfig(
        disaggregation_mode=disaggregation_mode,
        world_size=world_size,
        dp_size=world_size // (tp_size * cp_size),
    )



def build_online_request_dict(  # pylint: disable=too-many-arguments
    prompt,
    sampling_params: Optional[dict] = None,
    bootstrap_room: Optional[int] = None,
    bootstrap_host: Optional[str] = None,
    bootstrap_port: Optional[int] = None,
    require_bootstrap: bool = False,
) -> dict:
    request_dict = {"prompt": prompt, "sampling_params": sampling_params or {}}
    if bootstrap_room is not None:
        request_dict["bootstrap_room"] = bootstrap_room
    if bootstrap_host is not None:
        request_dict["bootstrap_host"] = bootstrap_host
    if bootstrap_port is not None:
        request_dict["bootstrap_port"] = bootstrap_port

    if require_bootstrap and request_dict.get("bootstrap_room") is None:
        raise HTTPException(
            status_code=400, detail="bootstrap_room is required in PREFILL mode"
        )
    if require_bootstrap and request_dict.get("bootstrap_host") is None:
        raise HTTPException(
            status_code=400, detail="bootstrap_host is required in PREFILL mode"
        )
    if require_bootstrap and request_dict.get("bootstrap_port") is None:
        raise HTTPException(
            status_code=400, detail="bootstrap_port is required in PREFILL mode"
        )

    return request_dict


def _redirect_worker_stdio(log_file: str) -> None:
    """Redirect worker stdout/stderr (Python and underlying fds) to log_file.

    Uses os.dup2 so writes from C extensions also land in the worker
    log instead of the parent's terminal. The opened
    fd is closed in the same scope after dup2 — the file lives on through
    fd 1 / fd 2 and is released when the worker process exits.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.dup2(fd, sys.stdout.fileno())
        os.dup2(fd, sys.stderr.fileno())
    finally:
        os.close(fd)
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    # Re-attach handler to the new stdio stream after fork.
    setup_logging()


def worker_main(  # pylint: disable=too-many-arguments
    global_rank: int,
    local_rank: int,
    world_size: int,
    local_world_size: int,
    yaml_file_path: str,
    leader_addr: str,
    pipe_writer: mp.connection.Connection,
    log_file: Optional[str] = None,
    disagg_config: Optional[DisaggConfig] = None,
) -> None:
    """Worker entrypoint used by the server parent process.

    leader_addr: instance leader IP for ZMQ connect (independent of torch.distributed).
    """
    if log_file:
        _redirect_worker_stdio(log_file)

    # Pin each worker to a disjoint slice of CPU cores — matches offline
    # function.sh's `taskset -c`.  Without this, N workers all inherit the
    # parent's full-machine affinity and PyTorch/OMP in each worker spawns
    # num_cpus threads, producing N× oversubscription that devastates weight
    # loading throughput.  Must run before any torch import so OMP picks up
    # the restricted affinity at init.
    num_cpus = os.cpu_count() or 1
    cores_per_rank = max(1, num_cpus // max(1, local_world_size))
    start = local_rank * cores_per_rank
    end = start + cores_per_rank
    os.sched_setaffinity(0, range(start, end))
    import torch
    torch.set_num_threads(cores_per_rank)
    os.environ.setdefault("OMP_NUM_THREADS", str(cores_per_rank))

    os.environ["RANK"] = str(global_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    logger.info(
        f"Worker global_rank={global_rank}, local_rank={local_rank}/{world_size} "
        f"starting, leader={leader_addr}"
    )

    from executor.core.config import InferenceConfig
    from executor.online.online_inference import OnlineInference

    with open(yaml_file_path, "r") as f:
        yaml_dict = yaml.safe_load(f)
    config = InferenceConfig.from_dict(
        yaml_dict,
        global_rank=global_rank,
        local_rank=local_rank,
        disagg_config=disagg_config,
    )
    llm = OnlineInference(config, leader_addr)

    pipe_writer.send("ready")
    pipe_writer.close()

    logger.info(f"Worker global_rank={global_rank} ready, entering inference loop")
    llm.run_continuous_loop()
    logger.info(f"Worker global_rank={global_rank} exiting")


class WorkerManager:
    """Manage online worker subprocess lifecycle."""

    def __init__(
        self,
        yaml_file_path: str,
        leader_addr: str,
        log_dir: str,
        disagg_config: DisaggConfig | None = None,
    ):
        self.yaml_file_path = yaml_file_path
        self.leader_addr = leader_addr
        self.log_dir = log_dir
        self.disagg_config = disagg_config
        self.workers: List[mp.Process] = []
        self._pipes: List[mp.connection.Connection] = []
        self._stop_event = threading.Event()
        self._watchdog_thread: Optional[threading.Thread] = None

    def spawn_workers(
        self, world_size: int, local_world_size: int, rank_offset: int
    ) -> None:
        """Spawn local workers and wait until all report ready."""
        for local_rank in range(local_world_size):
            global_rank = rank_offset + local_rank
            pipe_parent, pipe_child = mp.Pipe(duplex=False)
            p = mp.Process(
                target=worker_main,
                args=(
                    global_rank,
                    local_rank,
                    world_size,
                    local_world_size,
                    self.yaml_file_path,
                    self.leader_addr,
                    pipe_child,
                    os.path.join(self.log_dir, f"log_{global_rank}.log"),
                    self.disagg_config,
                ),
                daemon=True,
                name=f"Worker-rank{global_rank}",
            )
            p.start()
            pipe_child.close()
            self.workers.append(p)
            self._pipes.append(pipe_parent)
            logger.info(
                f"Worker global_rank={global_rank}, local_rank={local_rank} spawned (pid={p.pid})"
            )

        for local_rank, pipe in enumerate(self._pipes):
            global_rank = rank_offset + local_rank
            msg = pipe.recv()
            if msg != "ready":
                raise RuntimeError(
                    f"Worker global_rank={global_rank} sent unexpected message: {msg!r}"
                )
            logger.info(f"Worker global_rank={global_rank} ready")

        logger.info(
            f"All {local_world_size} local workers ready (rank_offset={rank_offset})"
        )

    def start_watchdog(self) -> None:
        """Start the watchdog daemon thread."""
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="WorkerWatchdog"
        )
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        while not self._stop_event.is_set():
            for proc in self.workers:
                if not proc.is_alive():
                    logger.error(
                        f"Worker '{proc.name}' (pid={proc.pid}) crashed, "
                        f"exitcode={proc.exitcode}"
                    )
            time.sleep(WATCHDOG_INTERVAL_SECS)

    def shutdown(self) -> None:
        """Terminate all workers and stop watchdog."""
        self._stop_event.set()
        for proc in self.workers:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10)
                if proc.is_alive():
                    proc.kill()
        logger.info("All workers stopped")

    def wait(self) -> None:
        """Wait for all worker subprocesses to exit."""
        try:
            for proc in self.workers:
                proc.join()
        finally:
            self.shutdown()


class GenerateRequest(BaseModel):
    prompt: str
    bootstrap_room: Optional[int] = None
    bootstrap_host: Optional[str] = None
    bootstrap_port: Optional[int] = None


class CompletionRequest(BaseModel):
    model: str
    prompt: str | List[str]
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stop: List[str] = []
    bootstrap_room: Optional[int] = None
    bootstrap_host: Optional[str] = None
    bootstrap_port: Optional[int] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[dict]
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stop: List[str] = []
    bootstrap_room: Optional[int] = None
    bootstrap_host: Optional[str] = None
    bootstrap_port: Optional[int] = None


REQUEST_TIMEOUT_SECS = 1800.0


def create_app(dispatcher, server_config: ServerAppConfig):
    """Create FastAPI app (server process only)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        del app
        try:
            yield
        finally:
            dispatcher.stop()

    app = FastAPI(title="Model Inference Server", lifespan=lifespan)

    @app.exception_handler(BackendUnavailableError)
    async def _backend_unavailable_handler(request, exc: BackendUnavailableError):
        del request
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    async def _require_server_ready():
        status = dispatcher.server_status
        if status == ServerStatus.Starting:
            raise HTTPException(status_code=503, detail="server starting")
        if status == ServerStatus.UnHealthy:
            raise HTTPException(status_code=503, detail="server unhealthy")

    async def _wait_result(request_id: int):
        try:
            result = await dispatcher.wait_result(request_id, REQUEST_TIMEOUT_SECS)
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=504, detail="Request timeout") from exc
        if result.get("finish_reason") == "prompt_too_long":
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Prompt too long: {result.get('prompt_tokens')} tokens "
                    f"exceeds the configured max_prefill_tokens limit"
                ),
            )
        return result

    def _build_req(prompt, sampling_params, src) -> dict:
        return build_online_request_dict(
            prompt=prompt,
            sampling_params=sampling_params,
            bootstrap_room=src.bootstrap_room,
            bootstrap_host=src.bootstrap_host,
            bootstrap_port=src.bootstrap_port,
            require_bootstrap=disagg_mode == ROLE_PREFILL,
        )

    def _usage(result: dict) -> dict:
        pt = result.get("prompt_tokens", 0)
        ct = result.get("completion_tokens", 0)
        return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}

    disagg_mode = server_config.disaggregation_mode

    @app.post("/generate", dependencies=[Depends(_require_server_ready)])
    async def generate(request: GenerateRequest):
        request_id = await dispatcher.submit_request(
            _build_req(request.prompt, {}, request)
        )
        result = await _wait_result(request_id)
        if disagg_mode == ROLE_PREFILL:
            return result
        return {"status": "success", "result": result}

    @app.post("/v1/completions", dependencies=[Depends(_require_server_ready)])
    async def completions(request: CompletionRequest):
        # temperature/top_p/stop are accepted at the API surface for OpenAI
        # client compatibility but not yet wired into the sampling path.
        sampling_params = {
            "max_tokens": request.max_tokens,
        }
        prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
        if disagg_mode == ROLE_PREFILL and len(prompts) != 1:
            raise HTTPException(400, "PREFILL mode currently supports a single prompt")

        request_ids = [
            await dispatcher.submit_request(_build_req(p, sampling_params, request))
            for p in prompts
        ]

        if disagg_mode == ROLE_PREFILL:
            for rid in request_ids:
                result = await _wait_result(rid)
            return result

        choices = []
        totals = {"prompt_tokens": 0, "completion_tokens": 0}
        for idx, rid in enumerate(request_ids):
            result = await _wait_result(rid)
            choices.append({
                "index": idx,
                "text": result["output"],
                "finish_reason": result.get("finish_reason", "stop"),
            })
            totals["prompt_tokens"] += result["prompt_tokens"]
            totals["completion_tokens"] += result["completion_tokens"]
        totals["total_tokens"] = totals["prompt_tokens"] + totals["completion_tokens"]
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:8]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": choices,
            "usage": totals,
        }

    @app.post("/v1/chat/completions", dependencies=[Depends(_require_server_ready)])
    async def chat_completions(request: ChatCompletionRequest):
        # temperature/top_p/stop are accepted at the API surface for OpenAI
        # client compatibility but not yet wired into the sampling path.
        sampling_params = {
            "max_tokens": request.max_tokens,
        }
        request_id = await dispatcher.submit_request(
            _build_req(request.messages, sampling_params, request)
        )

        result = await _wait_result(request_id)
        if disagg_mode == ROLE_PREFILL:
            return result
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": result["output"]},
                "finish_reason": result.get("finish_reason", "stop"),
            }],
            "usage": _usage(result),
        }

    @app.post("/stop")
    async def stop():
        dispatcher.stop()
        return {"status": "stopped", "message": "Server stopped"}

    @app.get("/health")
    async def health():
        status = dispatcher.server_status
        http_code = 200 if status == ServerStatus.Up else 503
        return JSONResponse(
            status_code=http_code,
            content={
                "status": status.value,
                "disaggregation_mode": disagg_mode,
            },
        )

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": "default",
                    "object": "model",
                    "owned_by": "cann-recipes-infer3",
                }
            ],
        }

    @app.get("/stats")
    async def stats():
        return {
            "status": "running",
            "pending_requests": dispatcher.get_pending_count(),
            "num_workers": server_config.world_size,
        }

    return app


def start_bootstrap_server(yaml_dict: dict) -> threading.Thread:
    """Run the PD bootstrap HTTP server on BOOTSTRAP_PORT in a daemon thread.

    Bootstrap is a standalone HTTP service, separate from the prefill
    inference API, so prefill ranks and decode clients can reach it at a
    fixed, well-known port regardless of the main server's port.
    """
    import uvicorn

    bootstrap_app = FastAPI(title="PD Bootstrap Server")
    init_bootstrap(bootstrap_app, yaml_dict)
    config = uvicorn.Config(
        bootstrap_app, host="0.0.0.0", port=BOOTSTRAP_PORT, log_level="info"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(
        target=server.run, daemon=True, name="BootstrapServer"
    )
    thread.start()
    logger.info(f"Bootstrap server listening on 0.0.0.0:{BOOTSTRAP_PORT}")
    return thread


def parse_args():
    """Parse server startup arguments."""
    parser = argparse.ArgumentParser(description="llm inference server")
    parser.add_argument(
        "--role", choices=["prefill", "decode", "router"], required=True,
        help="PD role: prefill, decode, or router",
    )
    parser.add_argument("--yaml-file-path", type=str, help="role-specific YAML")
    parser.add_argument("--node-index", type=int, help="node index within this role")
    parser.add_argument("--devices-per-node", type=int, help="accelerator count per node")
    parser.add_argument("--ips", nargs="+", help="role-local IPs (prefill/decode/router)")
    parser.add_argument("--bootstrap-port", type=int, default=BOOTSTRAP_PORT)
    # Both prefill and decode workers receive the full PREFILL_IPS and DECODE_IPS
    # lists so the memfabric store host (PREFILL_IPS[0]) is inferrable on either
    # role without extra CLI / YAML / env plumbing.
    parser.add_argument("--prefill-ips", nargs="+", help="all prefill node IPs (flat, cross-instance)")
    parser.add_argument("--decode-ips", nargs="+", help="all decode node IPs (flat, cross-instance)")
    # Router-only args:
    parser.add_argument("--prefill-addrs", nargs="+", help="prefill leader IPs (router mode)")
    parser.add_argument("--decode-addrs", nargs="+", help="decode leader IPs (router mode)")

    args = parser.parse_args()
    if args.role == "router":
        if not args.prefill_addrs or not args.decode_addrs:
            parser.error("--role router requires --prefill-addrs and --decode-addrs")
    else:
        if not args.yaml_file_path:
            parser.error("--yaml-file-path is required for prefill/decode")
        if args.node_index is None or args.devices_per_node is None:
            parser.error("--node-index and --devices-per-node are required")
        if not args.ips:
            parser.error("--ips is required for prefill/decode")
        if not args.prefill_ips:
            parser.error("--prefill-ips is required for prefill/decode")
    return args


def main():
    """Start the online HTTP server process."""
    import uvicorn

    args = parse_args()
    role = args.role.upper()

    # ── Router (early return) ──
    if role == "ROUTER":
        from executor.online.router import create_router_app

        prefill_targets = [f"http://{ip}:{PREFILL_HTTP_PORT}" for ip in args.prefill_addrs]
        decode_targets = [f"http://{ip}:{DECODE_HTTP_PORT}" for ip in args.decode_addrs]
        bootstrap_ports = [args.bootstrap_port] * len(args.prefill_addrs)
        app = create_router_app(prefill_targets, decode_targets, bootstrap_ports=bootstrap_ports)
        uvicorn.run(app, host="0.0.0.0", port=ROUTER_HTTP_PORT)
        return

    # ── Prefill / Decode ──
    with open(args.yaml_file_path, "r") as f:
        yaml_dict = yaml.safe_load(f)
    server_config = load_server_app_config(yaml_dict, disaggregation_mode=role)
    service_port = PREFILL_HTTP_PORT if role == ROLE_PREFILL else DECODE_HTTP_PORT

    # Instance boundary: derive leader IP and rank offset from IPs + world_size.
    if server_config.world_size % args.devices_per_node != 0:
        raise ValueError(
            f"world_size={server_config.world_size} must be divisible by "
            f"devices_per_node={args.devices_per_node}; otherwise nodes_per_inst "
            "silently floors and instance boundaries will be wrong."
        )
    nodes_per_inst = max(1, server_config.world_size // args.devices_per_node)
    inst_idx = args.node_index // nodes_per_inst
    rank_in_inst = args.node_index % nodes_per_inst
    is_leader = (rank_in_inst == 0)
    leader_addr = args.ips[inst_idx * nodes_per_inst]
    rank_offset = rank_in_inst * args.devices_per_node

    # Prefill workers register to bootstrap on the instance leader.
    # Decode gets bootstrap_host per-request from the router.
    bootstrap_host = leader_addr if role == ROLE_PREFILL else ""
    # MemFabric config store is a service-global singleton: it lives on the
    # leader node of the FIRST prefill instance and is shared by every rank
    # of every prefill AND decode instance. Under our flat cross-instance
    # PREFILL_IPS convention that node's IP is at args.prefill_ips[0]. Port
    # is a fixed convention, hardcoded.
    store_port = 10002
    first_prefill_leader = args.prefill_ips[0]
    store_url = f"tcp://{first_prefill_leader}:{store_port}"
    # Store creator = first prefill instance's leader node + its rank-0 worker.
    # Decomposed (inst_idx == 0 and rank_in_inst == 0) rather than the
    # equivalent (args.node_index == 0) so the "first instance's leader"
    # semantics don't rely on the reader knowing the flat-list encoding.
    is_store_creator_node = (
        role == ROLE_PREFILL and inst_idx == 0 and rank_in_inst == 0
    )
    disagg_config = DisaggConfig(
        disaggregation_mode=role,
        bootstrap_host=bootstrap_host,
        bootstrap_port=args.bootstrap_port,
        store_url=store_url,
        is_store_creator_node=is_store_creator_node,
        local_ip=args.ips[args.node_index],
    )

    # Prefill leader: start bootstrap HTTP server before spawning workers so
    # worker KV managers can register synchronously at init.
    # Backend-specific setup (e.g. memfabric create_config_store) runs inside
    # each worker's AscendTransferEngine.initialize() — main process does not
    # hold any backend resource.
    if is_leader and role == ROLE_PREFILL:
        logger.info(
            "Starting Prefill bootstrap server at %s:%d",
            bootstrap_host, args.bootstrap_port,
        )
        start_bootstrap_server(yaml_dict)

    worker_mgr = WorkerManager(
        args.yaml_file_path,
        leader_addr,
        os.path.join(os.getcwd(), os.environ.get("RES_PATH", "./")),
        disagg_config=disagg_config,
    )
    local_world_size = min(args.devices_per_node, server_config.world_size - rank_offset)
    worker_mgr.spawn_workers(server_config.world_size, local_world_size, rank_offset)
    worker_mgr.start_watchdog()

    if not is_leader:
        logger.info(
            f"Non-leader node (instance {inst_idx}): managing local ranks "
            f"[{rank_offset}, {rank_offset + args.devices_per_node - 1}]"
        )
        worker_mgr.wait()
        return

    from executor.online.dp_dispatcher import DPDispatcher
    _router_port, _pull_port = zmq_ports_for_role(role)
    dispatcher = DPDispatcher(
        dp_size=server_config.dp_size,
        disaggregation_mode=role,
        router_port=_router_port,
        pull_port=_pull_port,
    )
    dispatcher.start()
    logger.info(
        f"Server started: role={role}, dp_size={server_config.dp_size}, "
        f"num_workers={server_config.world_size}"
    )
    app = create_app(dispatcher, server_config)

    try:
        uvicorn.run(app, host="0.0.0.0", port=service_port)
    finally:
        dispatcher.stop()
        worker_mgr.shutdown()


if __name__ == "__main__":
    main()
