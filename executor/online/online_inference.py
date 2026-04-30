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

"""Online inference with distributed inference support."""

import logging
import json
import pickle
from dataclasses import dataclass
from typing import List, Optional, Tuple
from urllib import request as urllib_request

import torch
import torch.distributed as dist
import zmq

from executor.core.config import InferenceConfig
from executor.core.types_.types import SamplingParams
from executor.offline.offline_inference import OfflineInference
from executor.online.constants import zmq_ports_for_role
from executor.online.kv_transfer import KVTransferManager, MetadataBufferPool
from executor.online.scheduler import (
    DecodeDisaggScheduler,
    PrefillDisaggScheduler,
)

logger = logging.getLogger(__name__)


@dataclass
class InferenceSockets:
    """Container for ZMQ sockets used in inference loop.

    Only DP Leaders need sockets (for server communication).
    TP broadcast uses torch.distributed instead of ZMQ.
    """

    leader_socket: Optional[zmq.Socket] = None
    output_socket: Optional[zmq.Socket] = None


class OnlineInference(OfflineInference):
    """Online inference with distributed inference support.

    All ranks execute the same main loop:
    1. Receive requests (DP Leader via ZMQ, TP Workers via torch broadcast)
    2. Add requests to scheduler
    3. Run inference step
    4. DP Leader dispatches results
    """

    def __init__(self, infer_config: InferenceConfig, leader_addr: str = "127.0.0.1"):
        super().__init__(infer_config)
        self.leader_addr = leader_addr
        self.disaggregation_mode = infer_config.disagg_config.disaggregation_mode
        self.batch_size = infer_config.scheduler_config.batch_size_per_dp_rank
        self.router_port, self.pull_port = zmq_ports_for_role(self.disaggregation_mode)

        self._compute_parallel_ranks()

        self.kv_transfer_manager = None
        if self.disaggregation_mode in ("PREFILL", "DECODE"):
            self.kv_transfer_manager = self._build_kv_manager()
            self.scheduler = self._build_disagg_scheduler()
            self.kv_transfer_manager.transfer_engine.initialize()
            self.kv_transfer_manager.register_memory(self.engine.kvcache_manager)

        logger.info(
            f"Rank {self.global_rank}: dp_rank={self.dp_rank}, tp_rank={self.tp_rank}, "
            f"is_dp_leader={self.is_dp_leader}"
        )

    def _compute_parallel_ranks(self) -> None:
        pc = self.infer_config.parallel_config
        self.global_rank = pc.global_rank
        self.local_rank = pc.local_rank
        self.world_size = pc.world_size
        self.dp_size = pc.attn_dp_size
        self.tp_size = pc.attn_tp_size
        self.cp_size = pc.cp_size
        self.group_size = self.tp_size * self.cp_size

        self.dp_rank = self.global_rank // self.group_size
        self.cp_rank = (self.global_rank % self.group_size) // self.tp_size
        self.tp_rank = self.global_rank % self.tp_size
        self.is_dp_leader = self.global_rank % self.group_size == 0

    def _build_kv_manager(self) -> KVTransferManager:
        # Only DECODE needs a metadata_pool — it's the NPU HBM target buffer
        # that prefill RDMA-writes into. PREFILL's RDMA source is the
        # per-thread _metadata_scratch pool inside KVTransferManager itself.
        metadata_pool = (
            MetadataBufferPool(capacity=max(1, self.batch_size))
            if self.disaggregation_mode == "DECODE"
            else None
        )
        return KVTransferManager(
            disagg_config=self.infer_config.disagg_config,
            attn_tp_size=self.tp_size, attn_tp_rank=self.tp_rank,
            attn_cp_size=self.cp_size, attn_cp_rank=self.cp_rank,
            attn_dp_size=self.dp_size, attn_dp_rank=self.dp_rank,
            npu_id=self.local_rank,
            kv_cache_dtype=self.infer_config.model_config.dtype,
            is_mla_backend=self.engine.kvcache_manager.is_mla_backend,
            metadata_pool=metadata_pool,
        )

    def _build_disagg_scheduler(self):
        tp_cpu_group = (
            self.engine.comm_manager.get_group("tp_cpu_group") if self.group_size > 1 else None
        )
        if self.disaggregation_mode == "PREFILL":
            return PrefillDisaggScheduler(
                tokenizer=self.engine.tokenizer,
                config=self.infer_config.scheduler_config,
                kv_transfer_manager=self.kv_transfer_manager,
                kv_cache_manager=self.engine.kvcache_manager,
                dp_rank=self.dp_rank,
                is_dp_leader=self.is_dp_leader,
                tp_cpu_group=tp_cpu_group,
                input_truncated_len=self.infer_config.data_config.input_truncated_len,
            )
        return DecodeDisaggScheduler(
            tokenizer=self.engine.tokenizer,
            config=self.infer_config.scheduler_config,
            kv_transfer_manager=self.kv_transfer_manager,
            kv_cache_manager=self.engine.kvcache_manager,
            tp_cpu_group=tp_cpu_group,
            input_truncated_len=self.infer_config.data_config.input_truncated_len,
        )

    def run_continuous_loop(self):
        """Main online inference loop.  All ranks run the same flow.

        Per-iteration contract (every rank, in order):
          ① receive: leader drains ZMQ + broadcasts to TP followers
          ② advance_queues_consensus: all-rank Gloo-backed queue advancement
             (PREFILL: bootstrap_queue; DECODE: prealloc + transfer_queue).
             May surface failed requests as a StepOutput.
          ③ negotiate phase: DP-level all_gather → TP broadcast
          ④ run_step: forward when phase is set; inflight drain when None
          ⑤ idle_wait: only when globally idle (no work anywhere)

        Workers' StepOutput is silently dropped by _dispatch_results (which
        checks is_dp_leader + output_socket), so scheduler methods can return
        StepOutput on every rank.
        """
        ctx = zmq.Context()
        sockets = None
        try:
            sockets = self._init_sockets(ctx)
            while True:
                for request in self.get_receive_requests(sockets):
                    self._add_request_pd(request)

                pre_output = self.scheduler.advance_queues_consensus(self.engine)
                if pre_output:
                    self._dispatch_results(pre_output, sockets.output_socket)

                phase = self._negotiate_phase()

                step_output = self.scheduler.run_step(self.engine, phase=phase)
                if step_output:
                    self._dispatch_results(step_output, sockets.output_socket)

                if phase is None and not self.scheduler.has_work():
                    self._idle_wait(sockets)
        except Exception as e:
            logger.error(f"Rank {self.global_rank} error: {e}", exc_info=True)
        finally:
            self._cleanup_sockets(sockets)
            ctx.term()

    def _add_request_pd(self, request: dict) -> None:
        """Add a synchronized real request to scheduler with PD fields when present."""
        input_ids = request.get("input_ids")
        if request.get("prompt") is None:
            return

        sampling_params = request.get("sampling_params", {})
        params = SamplingParams(
            max_tokens=sampling_params.get("max_tokens", 32),
        )
        if hasattr(self.scheduler, "set_pd_request_context"):
            self.scheduler.set_pd_request_context(request)
        self.scheduler.add_request(
            prompt=request.get("prompt"),
            request_id=request.get("request_id"),
            sampling_params=params,
            input_ids=input_ids,
        )

    def _init_sockets(self, ctx: zmq.Context) -> InferenceSockets:
        """Initialize ZMQ sockets for DP Leader only.

        DP Leader:
            - leader_socket: DEALER (receive requests from server)
            - output_socket: PUSH (send results to server)

        TP Workers have no ZMQ sockets; they receive data via torch broadcast.
        """
        if self.is_dp_leader:
            leader_socket = ctx.socket(zmq.DEALER)
            try:
                leader_socket.setsockopt(zmq.IDENTITY, str(self.dp_rank).encode())
                leader_socket.connect(f"tcp://{self.leader_addr}:{self.router_port}")
                output_socket = ctx.socket(zmq.PUSH)
                try:
                    output_socket.connect(f"tcp://{self.leader_addr}:{self.pull_port}")
                except Exception:
                    output_socket.close(linger=0)
                    raise
            except Exception:
                leader_socket.close(linger=0)
                raise

            logger.info(
                f"DP Leader {self.dp_rank}: "
                f"DEALER -> {self.leader_addr}:{self.router_port}, "
                f"PUSH -> {self.leader_addr}:{self.pull_port}"
            )

            return InferenceSockets(
                leader_socket=leader_socket,
                output_socket=output_socket,
            )
        else:
            logger.info(
                f"TP Worker {self.tp_rank}: using torch broadcast for coordination"
            )
            return InferenceSockets()

    def _cleanup_sockets(self, sockets: Optional[InferenceSockets]) -> None:
        """Close all ZMQ sockets."""
        if not sockets:
            return
        for sock in [sockets.leader_socket, sockets.output_socket]:
            if sock:
                sock.close(linger=0)

    def _broadcast_group(self, obj):
        """Broadcast a Python object from DP leader to all TP/CP/PP followers
        within one DP group (size = tp_size * cp_size). No-op when
        group_size == 1 (single rank per DP group — no followers)."""
        if self.group_size == 1:
            return obj
        tp_cpu_group = self.engine.comm_manager.get_group("tp_cpu_group")
        obj_list = [obj]
        dist.broadcast_object_list(obj_list, src=0, group=tp_cpu_group)
        return obj_list[0]

    def _idle_wait(self, sockets: InferenceSockets) -> None:
        """Block-wait when all DP ranks have no work.

        Only DP leader polls ZMQ; TP workers will block on the torch
        broadcast at the start of the next get_receive_requests call.
        """
        if self.is_dp_leader and sockets.leader_socket is not None:
            poller = zmq.Poller()
            poller.register(sockets.leader_socket, zmq.POLLIN)
            poller.poll(self.IDLE_POLL_TIMEOUT_MS)

    def _sync_dp_local_state(
        self,
        local_has_prefill: bool,
        local_has_decode: bool,
    ) -> List[Tuple[bool, bool]]:
        """Synchronize local scheduler state across DP leaders."""
        dp_leader_group = self.engine.comm_manager.get_group("dp_leader_group")
        if self.dp_size <= 1 or dp_leader_group is None:
            return [(local_has_prefill, local_has_decode)]

        local_state = torch.tensor(
            [int(local_has_prefill), int(local_has_decode)],
            dtype=torch.long,
        )
        gathered = [torch.zeros_like(local_state) for _ in range(self.dp_size)]
        dist.all_gather(gathered, local_state, group=dp_leader_group)
        return [(bool(item[0].item()), bool(item[1].item())) for item in gathered]

    def _decide_phase(self, states: List[Tuple[bool, bool]]) -> Optional[str]:
        """Determine the global phase from local scheduler states."""
        if any(has_prefill for has_prefill, _ in states):
            return "prefill"
        if any(has_decode for _, has_decode in states):
            return "decode"
        return None

    # Polling timeout (ms) when globally idle (all DP ranks have no work).
    # DP leader blocks on zmq.poll; TP workers block on the torch broadcast
    # in the next get_receive_requests call.
    IDLE_POLL_TIMEOUT_MS = 1000

    def get_receive_requests(self, sockets: InferenceSockets) -> List[dict]:
        """Receive all available request messages for the current loop.

        DP Leader drains ZMQ socket in non-blocking mode, tokenizes, then
        broadcasts payloads to TP workers via torch. TP workers just receive.
        """
        if not self.is_dp_leader:
            return self._broadcast_group(None)
        payloads = self._drain_and_tokenize(sockets)
        return self._broadcast_group(payloads)

    def _drain_and_tokenize(self, sockets: InferenceSockets) -> List[dict]:
        """Drain the ZMQ socket (non-blocking) and tokenize each request."""
        requests = []
        if sockets.leader_socket is not None:
            while True:
                try:
                    data = sockets.leader_socket.recv(zmq.NOBLOCK)
                    request = pickle.loads(data)
                    logger.debug(
                        "DP Leader %s received request %s",
                        self.dp_rank, request.get('request_id'),
                    )
                    requests.append(request)
                except zmq.Again:
                    break
                except Exception as e:
                    logger.error(f"Error receiving request: {e}")
                    break
        return [
            {**r, "input_ids": self.scheduler.tokenize_request(r.get("prompt"))}
            for r in requests
        ]

    def _negotiate_phase(self) -> Optional[str]:
        """Pick the phase to run on all ranks this iteration.

        DP leader decides, then broadcasts to all TP/CP followers in the group.
        Queue advancement is handled upstream by scheduler.advance_queues_consensus()
        (Gloo consensus over tp_cpu_group), so by the time this runs every TP
        rank already has the same waiting_queue / running_requests —
        has_work() is a safe local check.
        """
        mode = self.disaggregation_mode
        if mode == "PREFILL":
            local_prefill, local_decode = self.scheduler.has_work(), False
        elif mode == "DECODE":
            local_prefill, local_decode = False, self.scheduler.has_work()
        else:
            raise ValueError(f"Unexpected disaggregation_mode: {mode}")

        # TP workers always follow the DP leader's decision.
        if not self.is_dp_leader:
            return self._broadcast_group(None)

        if self.dp_size > 1:
            states = self._sync_dp_local_state(local_prefill, local_decode)
        else:
            states = [(local_prefill, local_decode)]
        phase = self._decide_phase(states)
        return self._broadcast_group(phase)

    def _dispatch_results(self, step_output, output_socket: Optional[zmq.Socket]):
        """Dispatch finished results. Only DP leader does this."""
        if not self.is_dp_leader or not output_socket:
            return

        for req_id in step_output.finished_requests:
            request = self.scheduler.get_finished_request(req_id)
            if request:
                result = {
                    "output": self.engine.tokenizer.decode(request.output_id_list),
                    "finish_reason": request.finish_reason,
                    "prompt_tokens": request.prompt_tokens,
                    "completion_tokens": len(request.output_id_list),
                }
                output_socket.send_multipart(
                    [
                        str(self.dp_rank).encode(),
                        pickle.dumps(req_id),
                        pickle.dumps(result),
                    ]
                )
                logger.info(
                    f"DP Leader {self.dp_rank} completed request {req_id}: "
                    f"{len(request.output_id_list)} tokens"
                )
            self.scheduler.finished_requests.pop(req_id, None)
