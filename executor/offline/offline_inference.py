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

"""Offline batch inference entrypoint using Scheduler and ExecutionEngine."""

from typing import List, Optional, Tuple
import logging

import torch

from executor.core.config import InferenceConfig
from executor.core.forward_data_info import SamplingParams
from executor.core.engine import ExecutionEngine
from executor.core.scheduler import Scheduler
from executor.core.forward_data_info import GenerationOutput, Request
from executor.core.support_models import load_model_classes, load_speculative_model_classes
from executor.utils.common_utils import process_infer_time

logger = logging.getLogger(__name__)


class OfflineInference:
    """Batch inference entry point using Scheduler and ExecutionEngine.

    This class provides a simple interface for offline batch inference
    with support for batching multiple requests.

    Usage:
        config = InferenceConfig.from_yaml("config.yaml")
        llm = OfflineInference(config)
        results = llm.generate(["Hello", "How are you?", "What's new?"])

    Attributes:
        config: Inference configuration.
        engine: ExecutionEngine for model inference.
        scheduler: Scheduler for request management.
    """

    def __init__(
        self,
        infer_config: InferenceConfig,
    ):
        """Initialize offline inference.

        Args:
            infer_config: Inference configuration including scheduler_config.
        """
        self.infer_config = infer_config

        # Initialize execution engine
        self.engine = ExecutionEngine(self.infer_config)
        self._load_model()
        self.enable_afd = self.infer_config.model_config.custom_params.get("enable_afd", False)
        self.enable_offline_afd_phase_sync = (
            self.enable_afd
            and self.infer_config.disagg_config.disaggregation_mode == "NONE"
        )
        self._compute_parallel_ranks()

        # Match LongCat AFD's 1:1 FFN/Attention rank layout.
        self.engine.comm_manager.init_cpu_groups(
            enable_afd=self.enable_offline_afd_phase_sync
        )

        # Initialize scheduler with engine's tokenizer
        self.scheduler = Scheduler(
            tokenizer=self.engine.tokenizer,
            config=self.infer_config.scheduler_config,
            input_truncated_len=self.infer_config.data_config.input_truncated_len,
            mm_processor=self.engine.mm_processor,
            enable_mm_encode=self.engine.enable_mm_encode,
        )

    def _compute_parallel_ranks(self) -> None:
        pc = self.infer_config.parallel_config
        self.global_rank = pc.global_rank
        self.local_rank = pc.local_rank
        self.world_size = pc.world_size
        self.tp_size = pc.attn_tp_size
        self.cp_size = pc.cp_size
        self.group_size = self.tp_size * self.cp_size
        self.dp_size = pc.cp_prefill_dp_size if self.cp_size > 1 else pc.attn_dp_size

        logical_rank = self.global_rank
        if self.enable_offline_afd_phase_sync:
            logical_rank %= self.world_size // 2
        self.dp_rank = logical_rank // self.group_size
        self.cp_rank = (logical_rank % self.group_size) // self.tp_size
        self.tp_rank = logical_rank % self.tp_size
        self.is_dp_leader = logical_rank % self.group_size == 0
        self.is_phase_leader = (
            self.is_dp_leader
            and (
                not self.enable_offline_afd_phase_sync
                or not self.engine.is_afd_ffn_rank
            )
        )

    def _load_model(self) -> None:
        """Load model based on configuration."""
        model_name = self.infer_config.model_config.model_name
        # AFD FFN-only ranks load the FFN variant registered as "<model_name>_ffn".
        if self.engine.is_afd_ffn_rank:
            model_name = f"{model_name}_ffn"

        model_config_cls = load_model_classes(model_name)
        if len(model_config_cls) == 2:
            model_class, config_class = model_config_cls
        else:
            model_class, _, config_class = model_config_cls
        speculative_config = self.infer_config.speculative_config
        if not speculative_config.enabled:
            draft_model_class = None
            draft_config_class = None
        else:
            draft_model_class, draft_config_class = load_speculative_model_classes(
                model_name,
                speculative_config.method,
            )
        self.engine.init(
            config_class,
            model_class,
            draft_model_class,
            draft_config_class,
        )
        self.engine.warm_up()

    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
        batch_size: Optional[int] = None,
    ) -> tuple[List[GenerationOutput], Optional[dict], List[float]]:
        """Generate text for a batch of prompts.

        This method processes prompts using batching:
        1. Add all requests to scheduler
        2. Run scheduling loop until all requests complete
        3. Collect and return results

        Args:
            prompts: List of input text prompts.

        Returns:
            A tuple containing:
            - List of GenerationOutput objects, one per prompt.
            - Aggregated MTP statistics dict (None if MTP not enabled).
            - Batch-level inference time list: [prefill, decode...] for text-only models,
              or [encode, prefill, decode...] when MM Encode is enabled.
        """
        if not prompts:
            logger.warning(
                "no prompts on this rank; returning early. If other DP ranks have work, "
                "phase negotiation will stall.")
            return [], None, []
        if self.engine.is_afd_ffn_rank:
            return self._generate_afd_ffn(prompts)

        # Reset scheduler for new batch
        self.scheduler.reset()
        if self.engine.mm_embedding_store is not None:
            self.engine.mm_embedding_store.clear()

        parallel_config = self.infer_config.parallel_config
        enable_cp = parallel_config.cp_size > 1

        batch_prompt = prompts[0]
        if isinstance(batch_prompt, str):
            batch_prompt = [{"role": "user", "content": batch_prompt}]
        request_ids = self.scheduler.add_batch_requests(
            batch_prompt, batch_size, sampling_params
        )
        prompt_map = {request_id: batch_prompt for request_id in request_ids}

        # Run scheduling loop until all requests complete
        while True:
            phase = self._negotiate_phase()
            if phase is None:
                break
            local_has_work = self._has_phase_work(phase)
            # Every rank executes the negotiated phase; ranks without local
            # work run a dummy batch so model collectives remain aligned.
            did_run = self.scheduler.run_step(self.engine, phase=phase)
            if self._sync_phase_stall(phase, local_has_work and not did_run):
                break

        # Collect results (only for original requests, not padded ones)
        results = []
        # Store backend-neutral speculative decoding statistics.
        speculative_stats = {
            "spec_num_accepted_tokens": [],
            "spec_num_draft_tokens": [],
            "spec_num_forward_ct": [],
            "valid_output_len": [],
            "decode_execution_time": [],
            "full_decode_execution_time": [],
            "decode_output_tokens": [],
            "timed_spec_num_forward_ct": [],
            "timed_spec_num_accepted_tokens": [],
            "timed_spec_num_draft_tokens": [],
            "timed_decode_output_tokens": [],
        }
        if enable_cp:
            current_cp_rank = parallel_config.global_rank % parallel_config.cp_size
            request_ids = [
                request_id
                for request_id in request_ids
                if self.scheduler.finished_requests.get(request_id, None) is not None
                and self.scheduler.finished_requests[request_id].cp_rank == current_cp_rank
            ]

        use_mm_batch_infer_time = self.engine.enable_mm_encode
        needs_cp_infer_time_fallback = enable_cp and not request_ids
        batch_infer_time = []
        if use_mm_batch_infer_time or needs_cp_infer_time_fallback:
            for request_id in request_ids:
                finished_request = self.scheduler.finished_requests.get(request_id)
                if finished_request is not None:
                    batch_infer_time = finished_request.infer_time
                    if batch_infer_time and use_mm_batch_infer_time:
                        batch_infer_time = [
                            self.scheduler.mm_encode_infer_time,
                            self.scheduler.prefill_infer_time,
                            *batch_infer_time[1:],
                        ]
                    break
        for request_id in request_ids:
            request = self.scheduler.pop_finished_request(request_id)
            if request is None:
                logger.warning(
                    "request %s: not found in finished_requests after scheduler "
                    "loop exited — returning empty result", request_id,
                )
                results.append(GenerationOutput(
                    prompt=prompt_map[request_id],
                    output_text="",
                    finish_reason="error",
                ))
                continue

            if not use_mm_batch_infer_time:
                batch_infer_time = request.infer_time

            # Decode only the valid output segment truncated by max output length or EOS.
            valid_output_id_list = self.get_valid_output(request)
            output_text = self.engine.tokenizer.decode(
                torch.tensor(valid_output_id_list), skip_special_tokens=True)
            if request.draft_info:
                # Match durations to counted request verification rounds, excluding
                # trailing static-batch execution after this request has finished.
                request_decode_times = request.infer_time[1:request.spec_num_forward_ct + 1]
                speculative_stats["full_decode_execution_time"].append(
                    sum(request_decode_times)
                )
                speculative_stats["decode_execution_time"].append(sum(request.timed_infer_time))
                speculative_stats["timed_spec_num_forward_ct"].append(request.timed_spec_num_forward_ct)
                speculative_stats["timed_spec_num_accepted_tokens"].append(
                    request.timed_spec_num_accepted_tokens
                )
                speculative_stats["timed_spec_num_draft_tokens"].append(
                    request.timed_spec_num_draft_tokens
                )
                speculative_stats["timed_decode_output_tokens"].append(
                    request.timed_decode_output_tokens
                )
                speculative_stats["decode_output_tokens"].append(max(len(valid_output_id_list) - 1, 0))
                speculative_stats["spec_num_accepted_tokens"].append(request.spec_num_accepted_tokens)
                speculative_stats["spec_num_draft_tokens"].append(request.spec_num_draft_tokens)
                speculative_stats["spec_num_forward_ct"].append(request.spec_num_forward_ct)
                speculative_stats["valid_output_len"].append(request.valid_output_len)

            results.append(GenerationOutput(
                prompt=prompt_map[request_id],
                output_text=output_text,
                finish_reason=request.finish_reason,
            ))

        return results, speculative_stats, batch_infer_time

    def _negotiate_phase(self) -> Optional[str]:
        """Choose one phase for all DP/TP ranks in this offline step."""
        local_mm = self.scheduler.has_mm_pending()
        local_prefill = self.scheduler.has_prefill_ready()
        local_decode = bool(self.scheduler.running_requests)

        if self.is_phase_leader:
            states = self._sync_dp_local_state(local_mm, local_prefill, local_decode)
            phase = self._decide_phase(states)
        else:
            phase = None
        return self._broadcast_group(phase)

    def _has_phase_work(self, phase: str) -> bool:
        if phase == "mm_encode":
            return self.scheduler.has_mm_pending()
        if phase == "prefill":
            return self.scheduler.has_prefill_ready()
        if phase == "decode":
            return bool(self.scheduler.running_requests)
        return False

    def _sync_phase_stall(self, phase: str, local_stalled: bool) -> bool:
        """Detect a real stall and stop the offline batch consistently.

        ``run_step()`` returns ``False`` for both an expected dummy batch and
        a real scheduling failure.  ``local_stalled`` is set only when this
        rank has work but cannot build a batch.  Reduce that state within the
        TP/CP group and across DP leaders so every rank exits together.  This
        is a fail-fast guard for resource failures such as KV cache exhaustion;
        it does not recover the pending request.
        """
        tp_group = self.engine.comm_manager.get_group("tp_cpu_group")
        if tp_group is not None:
            stall_tensor = torch.tensor([int(local_stalled)], dtype=torch.long)
            torch.distributed.all_reduce(
                stall_tensor,
                op=torch.distributed.ReduceOp.MAX,
                group=tp_group,
            )
            local_stalled = bool(stall_tensor.item())

        if self.is_phase_leader:
            group = self.engine.comm_manager.get_group("dp_leader_group")
            if self.dp_size <= 1 or group is None:
                stalled = local_stalled
            else:
                local_state = torch.tensor([int(local_stalled)], dtype=torch.long)
                gathered = [torch.zeros_like(local_state) for _ in range(self.dp_size)]
                torch.distributed.all_gather(gathered, local_state, group=group)
                stalled = any(bool(item[0].item()) for item in gathered)
        else:
            stalled = None
        stalled = bool(self._broadcast_group(stalled))
        if stalled and self.is_phase_leader:
            logger.warning(
                "Offline %s phase stalled with local work pending; stopping the scheduling loop.",
                phase,
            )
        return stalled

    def _sync_dp_local_state(
        self,
        local_has_mm: bool,
        local_has_prefill: bool,
        local_has_decode: bool,
    ) -> List[Tuple[bool, bool, bool]]:
        """Gather MM Encode, Prefill and Decode state across DP leaders."""
        group = self.engine.comm_manager.get_group("dp_leader_group")
        if self.dp_size <= 1 or group is None:
            return [(local_has_mm, local_has_prefill, local_has_decode)]

        local_state = torch.tensor(
            [int(local_has_mm), int(local_has_prefill), int(local_has_decode)],
            dtype=torch.long,
        )
        gathered = [torch.zeros_like(local_state) for _ in range(self.dp_size)]
        torch.distributed.all_gather(gathered, local_state, group=group)
        return [
            (
                bool(item[0].item()),
                bool(item[1].item()),
                bool(item[2].item()),
            )
            for item in gathered
        ]

    @staticmethod
    def _decide_phase(
        states: List[Tuple[bool, bool, bool]],
    ) -> Optional[str]:
        """Apply the global phase priority used by offline scheduling."""
        if any(has_prefill for _, has_prefill, _ in states):
            return "prefill"
        if any(has_mm for has_mm, _, _ in states):
            return "mm_encode"
        if any(has_decode for _, _, has_decode in states):
            return "decode"
        return None

    def _broadcast_group(self, obj):
        """Broadcast a phase or request payload within the local CPU group."""
        if self.group_size == 1 and not self.enable_offline_afd_phase_sync:
            return obj

        obj_list = [obj]
        source = self.dp_rank * self.group_size
        if self.enable_offline_afd_phase_sync:
            source += self.world_size // 2
        torch.distributed.broadcast_object_list(
            obj_list,
            src=source,
            group=self.engine.comm_manager.get_group("tp_cpu_group"),
        )
        return obj_list[0]

    def _generate_afd_ffn(self, prompts: List[str]) -> tuple[List[GenerationOutput], Optional[dict], List[float]]:
        self.scheduler.reset()
        if self.engine.mm_embedding_store is not None:
            self.engine.mm_embedding_store.clear()
        prompts = [
            [{"role": "user", "content": p}] if isinstance(p, str) else p
            for p in prompts
        ]
        request_ids = self.scheduler.add_batch_requests(prompts[0], len(prompts))
        if not request_ids:
            return [], None, []

        while True:
            phase = self._negotiate_phase()
            if phase is None:
                break

            local_has_work = self._has_phase_work(phase)
            did_run = self.scheduler.run_step(self.engine, phase=phase)
            if self._sync_phase_stall(phase, local_has_work and not did_run):
                break

        request = None if not request_ids else (
            self.scheduler.running_requests.get(request_ids[0])
            or self.scheduler.finished_requests.get(request_ids[0])
        )
        infer_time = request.infer_time if request is not None else []

        decode_infer_time = infer_time[1:] if infer_time and len(infer_time) > 1 else []
        avg_decode_time = process_infer_time(decode_infer_time, len(decode_infer_time))
        logger.info(
            "%s ffn average inference time cost is %.2f ms",
            self.engine.main_worker.model_name,
            avg_decode_time * 1000,
        )
        return [], None, infer_time

    def get_valid_output(self, request: Request) -> List[int]:
        if request.valid_output_len is not None:
            return request.output_id_list[:request.valid_output_len]
        else:
            # Return full output when generation hasn't reached valid stop condition
            return request.output_id_list
