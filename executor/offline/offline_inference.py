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

from typing import Dict, List, Optional
import logging

import torch

from executor.core.config import InferenceConfig
from executor.core.engine import ExecutionEngine
from executor.core.scheduler import Scheduler
from executor.core.forward_data_info import GenerationOutput, Request
from executor.core.support_models import load_model_classes
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

        # Initialize scheduler with engine's tokenizer
        self.scheduler = Scheduler(
            tokenizer=self.engine.tokenizer,
            config=self.infer_config.scheduler_config,
            input_truncated_len=self.infer_config.data_config.input_truncated_len,
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
            model_mtp_class = None
        else:
            model_class, model_mtp_class, config_class = model_config_cls
            model_mtp_class = None if self.engine.next_n == 0 else model_mtp_class
        self.engine.init(config_class, model_class, model_mtp_class)
        self.engine.warm_up()

    def generate(
        self,
        prompts: List[str],
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
            - Batch-level inference time list. Index 0 is prefill time and the rest are decode batches.
        """
        if not prompts:
            return [], None, []
        if self.engine.is_afd_ffn_rank:
            return self._generate_afd_ffn(prompts)

        # Reset scheduler for new batch
        self.scheduler.reset()

        parallel_config = self.infer_config.parallel_config
        enable_cp = parallel_config.cp_size > 1
        # CP prefill uses a global compute batch on every CP rank; decode still
        # returns only this rank's owner requests.
        batch_size = self.scheduler.config.batch_size if enable_cp else self.scheduler.config.batch_size_per_dp_rank

        # Convert str prompts to chat message format for chat template tokenization
        prompts = [
            [{"role": "user", "content": p}] if isinstance(p, str) else p
            for p in prompts
        ]

        # Add all requests to scheduler
        request_ids = []
        prompt_map = {}
        for prompt in prompts:
            request_id = self.scheduler.add_request(prompt)
            request_ids.append(request_id)
            # simplified process, to be optimized
            prompt_map[request_id] = prompt
            if len(request_ids) >= batch_size:
                break

        original_request_count = len(request_ids)
        if len(request_ids) < batch_size:
            # Pad requests to batch_size if needed
            while len(request_ids) < batch_size:
                request_id = self.scheduler.add_request(prompts[-1])
                request_ids.append(request_id)

        # Run scheduling loop until all requests complete
        while self.scheduler.has_work():
            if not self.scheduler.run_step(self.engine):
                logger.warning("Scheduler has work but no batch was scheduled. Breaking loop to avoid infinite wait.")
                break

        # Collect results (only for original requests, not padded ones)
        results = []
        # Store raw MTP statistics
        mtp_stats = {
            "spec_num_accepted_tokens": [],
            "spec_num_forward_ct": [],
            "valid_output_len": []
        }
        result_request_ids = request_ids[:original_request_count]
        if enable_cp:
            current_cp_rank = parallel_config.global_rank % parallel_config.cp_size
            result_request_ids = [
                request_id for request_id in result_request_ids
                if request_id % parallel_config.cp_size == current_cp_rank
            ]
        for request_id in result_request_ids:
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

            # Decode only the valid output segment truncated by max output length or EOS.
            valid_output_id_list = self.get_valid_output(request)
            output_text = self.engine.tokenizer.decode(
                torch.tensor(valid_output_id_list), skip_special_tokens=True)
            # Calculate MTP accept rate
            if request.mtp_info:
                mtp_stats["spec_num_accepted_tokens"].append(request.spec_num_accepted_tokens)
                mtp_stats["spec_num_forward_ct"].append(request.spec_num_forward_ct)
                mtp_stats["valid_output_len"].append(request.valid_output_len)

            results.append(GenerationOutput(
                prompt=prompt_map[request_id],
                output_text=output_text,
                finish_reason=request.finish_reason,
            ))

        return results, mtp_stats, request.infer_time

    def _generate_afd_ffn(self, prompts: List[str]) -> tuple[List[GenerationOutput], Optional[dict], List[float]]:
        self.scheduler.reset()
        batch_size = self.scheduler.config.batch_size_per_dp_rank
        prompts = [
            [{"role": "user", "content": p}] if isinstance(p, str) else p
            for p in prompts
        ]
        request_ids = []
        for prompt in prompts:
            request_ids.append(self.scheduler.add_request(prompt))
            if len(request_ids) >= batch_size:
                break
        if not request_ids:
            return [], None, []
        # AFD FFN ranks build dummy inputs to receive hidden states from Attention ranks.
        # Keep the FFN-side batch shape fixed by padding dummy requests when needed.
        while len(request_ids) < batch_size:
            request_ids.append(self.scheduler.add_request(prompts[-1]))

        infer_time = []
        while self.scheduler.has_work():
            if not self.scheduler.run_step(self.engine):
                logger.warning("AFD FFN scheduler has work but no batch was scheduled.")
                break
            request = self.scheduler.running_requests.get(request_ids[0])
            if request is None:
                request = self.scheduler.finished_requests.get(request_ids[0])
            if request is not None:
                infer_time = request.infer_time

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
