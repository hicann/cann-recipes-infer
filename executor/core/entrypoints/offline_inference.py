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

from typing import List, Optional
import logging

import torch

from executor.core.config import InferenceConfig
from executor.core.engine import ExecutionEngine
from executor.core.scheduler import Scheduler
from executor.core.types_ import GenerationOutput
from .support_models import model_dict

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


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
        )

    def _load_model(self) -> None:
        """Load model based on configuration."""
        model_name = self.infer_config.model_config.model_name

        if model_name in model_dict:
            model_config_cls = model_dict[model_name]
            if len(model_config_cls) == 2:
                model_class, config_class = model_config_cls
                model_mtp_class = None
            else:
                model_class, model_mtp_class, config_class = model_config_cls
                model_mtp_class = None if self.engine.next_n == 0 else model_mtp_class
            self.engine.load_model(config_class, model_class, model_mtp_class)
            self.engine.warm_up()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def generate(
        self,
        prompts: List[str],
    ) -> tuple[List[GenerationOutput], List[Optional[dict]]]:
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
            - List of MTP statistics dicts for each request (None if MTP not enabled).
              Each dict contains 'spec_num_accepted_tokens' and 'spec_num_forward_ct'.
        """
        if not prompts:
            return []

        # Reset scheduler for new batch
        self.scheduler.reset()

        # Use batch_size_per_dp_rank for distributed inference
        batch_size = self.scheduler.config.batch_size_per_dp_rank

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
            step_output = self.scheduler.run_step(self.engine)
            if step_output is None:
                logging.warning("Scheduler has work but no batch was scheduled. Breaking loop to avoid infinite wait.")
                break

        # Collect results (only for original requests, not padded ones)
        results = []
        # Store raw MTP statistics
        mtp_stats = {
            "spec_num_accepted_tokens": [],
            "spec_num_forward_ct": 0
        }
        for request_id in request_ids[:original_request_count]:
            request = self.scheduler.get_finished_request(request_id)
            if request is None:
                results.append(GenerationOutput(
                    prompt=prompt_map[request_id],
                    output_text="",
                    finish_reason="error",
                ))
                continue

            # Decode output
            output_text = self.engine.tokenizer.decode(torch.tensor(request.output_id_list), skip_special_tokens=True)
            # Caculate mtp accept rate
            if request.mtp_info:
                mtp_stats["spec_num_accepted_tokens"].append(request.spec_num_accepted_tokens)

            results.append(GenerationOutput(
                prompt=prompt_map[request_id],
                output_text=output_text,
                finish_reason=request.finish_reason,
            ))
        mtp_stats["spec_num_forward_ct"] = request.spec_num_forward_ct
        return results, mtp_stats, request.infer_time
