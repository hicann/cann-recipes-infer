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

"""ExecutionEngine for model inference."""

import os
import logging
from typing import Dict

import torch
from transformers import AutoTokenizer

from executor.core.config import InferenceConfig, CommManager
from executor.utils import get_default_group
from executor.utils.forward_metadata import set_forward_metadata, get_forward_metadata
from executor.core.model_worker import ModelWorker
from executor.utils.profiler_context import create_profiler

torch.npu.config.allow_internal_format = True
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


class ExecutionEngine:

    def __init__(self, infer_config: InferenceConfig):
        """Initialize engine with configuration.

        Args:
            config: Inference configuration.
        """
        self.infer_config = infer_config
        self.device = None
        self.tokenizer = None
        self.hf_config = None
        self.max_new_tokens = self.infer_config.scheduler_config.max_new_tokens
        self.input_max_len = self.infer_config.scheduler_config.input_max_len
        self.exe_mode = self.infer_config.model_config.exe_mode

        # Distributed info
        self.local_rank = self.infer_config.parallel_config.local_rank
        self.global_rank = self.infer_config.parallel_config.global_rank
        self.world_size = self.infer_config.parallel_config.world_size
        self._init_device()

        # Initialize workers
        self.main_worker = ModelWorker(self.infer_config, self.device)

        # Profiling configuration
        self.enable_profiler = self.infer_config.model_config.enable_profiler
        self.output_path = self.infer_config.model_config.output_path
        self.profiler = create_profiler(self.enable_profiler, os.path.join(self.output_path, "prof"))

    def _init_device(self):
        """Initialize NPU device and communication."""
        logging.info("Set execution using npu index: %s, global: %s", self.local_rank, self.global_rank)
        self.device = torch.device("%s:%s" % ("npu", self.local_rank))
        torch.npu.set_device(self.device)

        # Initialize distributed process group for multi-card scenarios
        if torch.npu.is_available() and self.world_size > 1:
            default_pg = get_default_group()
            if default_pg is None:
                master_addr = os.environ["MASTER_ADDR"]
                master_port = int(os.environ["MASTER_PORT"])
                torch.distributed.init_process_group(
                    backend="hccl", world_size=self.world_size, rank=self.global_rank)

        # Initialize communication manager after process group is set up
        self.comm_manager = CommManager(self.infer_config.parallel_config)
        self.comm_manager.initialize()

    def load_model(
        self,
        config_cls,
        main_model_cls
    ):
        """Load model with optional MTP model for speculative decoding."""
        logging.info("Loading model...")
        self.main_worker.load_model(main_model_cls, config_cls, self.comm_manager)

        # Initialize tokenizer (from main worker)
        self.hf_config = self.main_worker.hf_config
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.infer_config.model_config.model_path,
            padding_side="right",
            truncation_side='right',
            trust_remote_code=True
        )

    @property
    def model(self):
        """Get the main model."""
        return self.main_worker.model if self.main_worker else None

    def _build_model_inputs(self, input_ids: torch.Tensor, is_prefill: bool):
        """Build model inputs and set forward metadata."""
        batch_size, seq_len = input_ids.size()
        attention_mask = ~torch.tril(torch.ones((2048, 2048), dtype=torch.bool, device=self.device))

        if is_prefill:
            mask = input_ids != self.tokenizer.pad_token_id
            position_ids = mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(mask == 0, 1)
            kv_len = torch.zeros((batch_size), dtype=torch.long, device=input_ids.device)
            actual_seq_lengths_kv = torch.tensor(
                [seq_len] * batch_size, dtype=torch.long, device=self.device
            )
            if self.infer_config.model_config.packed_sequence:
                actual_seq_lengths_kv = actual_seq_lengths_kv.cumsum(0)
        else:
            # Decode: use kv_len from previous step
            kv_len = get_forward_metadata().kv_len + 1
            position_ids = kv_len.unsqueeze(1)
            actual_seq_lengths_kv = kv_len + 1

            # acl_graph special handling
            if self.exe_mode == "acl_graph":
                actual_seq_lengths_kv = actual_seq_lengths_kv.detach().cpu().numpy().tolist()

        # Set forward metadata
        set_forward_metadata(
            is_prefill=is_prefill,
            kv_len=kv_len,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            attention_mask=attention_mask,
        )

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "forward_metadata": get_forward_metadata(),
        }

    def _get_warmup_shape(self):
        """Calculate warm-up input shapes based on batch_size_per_dp_rank and config."""
        batch_size = self.infer_config.scheduler_config.batch_size_per_dp_rank
        seq_len = self.infer_config.scheduler_config.input_max_len

        return batch_size, seq_len

    def warm_up(self):
        """Execute warm-up by running exactly one prefill and one decode step.
        Triggers graph compilation during decode if graph mode enabled.
        """
        logging.info("Starting warm-up...")

        # 1. Get warm-up shape info from config
        batch_size, seq_len = self._get_warmup_shape()

        # 2. Execute ONE prefill step with dummy inputs
        logging.info("Warm-up: executing prefill step...")
        dummy_input_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=self.device)
        model_inputs = self._build_model_inputs(dummy_input_ids, is_prefill=True)
        _ = self.main_worker.inference(model_inputs, is_prefill=True)

        # 3. Execute ONE decode step with graph compilation
        logging.info("Warm-up: executing decode step...")
        dummy_input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
        model_inputs = self._build_model_inputs(dummy_input_ids, is_prefill=False)

        # Trigger graph compilation if graph mode enabled and not yet compiled
        if "graph" in self.exe_mode:
            logging.info("Warm-up: triggering graph compilation...")
            self.main_worker.compile_model()

        _ = self.main_worker.inference(model_inputs, is_prefill=False)

        logging.info("Warm-up completed successfully.")

    def forward_batch(self, batch) -> Dict[str, torch.Tensor]:
        """Execute forward pass for a batch of requests.

        This is the primary interface for Scheduler to execute both
        prefill and decode phases through a unified interface.

        Args:
            batch: Batch object containing requests, input_ids, position_ids, etc.

        Returns:
            Dict containing:
                - "next_tokens": Dict mapping request_id to generated token
                - "logits": [batch_size, seq_len, vocab_size] model outputs
        """
        # Create profiler context
        with self.profiler as prof:
            # Prepare model inputs
            inputs_ids = batch.input_ids.to(self.device)
            model_inputs = self._build_model_inputs(inputs_ids, is_prefill=batch.is_prefill)

            # Run inference
            logits, _ = self.main_worker.inference(model_inputs, is_prefill=batch.is_prefill)
            # Step the profiler

            # Update KV cache length after prefill
            if batch.is_prefill:
                kv_lens = torch.max(model_inputs["position_ids"], dim=1)[0]
                set_forward_metadata(kv_len=kv_lens)

            # Sample next tokens for each request
            next_tokens = self._sample_tokens_for_batch(batch, logits)

            prof.step()

        return {
            "next_tokens": next_tokens,
            "logits": logits,
        }

    def _sample_tokens_for_batch(
        self,
        batch,
        logits: torch.Tensor
    ) -> Dict[int, int]:
        """Sample next tokens for each request in batch.

        Args:
            batch: Batch containing request metadata.
            logits: [batch_size, seq_len, vocab_size] model outputs.

        Returns:
            Dict mapping request_id to sampled token.
        """
        next_tokens = {}

        for i, request in enumerate(batch.requests):
            # Prefill and Decode: single position
            last_logits = logits[i, -1, :]

            # Greedy sampling
            next_token = torch.argmax(last_logits).item()
            next_tokens[request.request_id] = next_token

        return next_tokens
