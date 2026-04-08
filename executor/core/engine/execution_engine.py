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
from typing import Dict, Optional, Any

import torch
from transformers import AutoTokenizer

from executor.core.config import InferenceConfig, CommManager
from executor.utils import get_default_group
from executor.utils.forward_metadata import set_forward_metadata, get_forward_metadata
from executor.utils.profiler_context import ProfilerManager
from executor.core.model_worker import ModelWorker, MTPWorker
from ..types_.types import MTPInfo, Batch, StepOutput

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
        self.next_n = self.infer_config.model_config.next_n
        self.prefill_mini_batch = self.infer_config.scheduler_config.prefill_mini_batch

        # Distributed info
        self.local_rank = self.infer_config.parallel_config.local_rank
        self.global_rank = self.infer_config.parallel_config.global_rank
        self.world_size = self.infer_config.parallel_config.world_size
        self._init_device()

        # Initialize workers
        self.main_worker = ModelWorker(self.infer_config, self.device)
        self.mtp_worker = MTPWorker(self.infer_config, self.device) if self.next_n > 0 else None

        # Profiling configuration
        self.enable_profiler = self.infer_config.model_config.enable_profiler
        self.output_path = os.getenv("RES_PATH", "./")
        self.profiler = ProfilerManager(self.enable_profiler, self.output_path)

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
        main_model_cls,
        mtp_model_cls=None,
    ):
        """Load model with optional MTP model for speculative decoding."""
        logging.info("Loading main model...")
        self.main_worker.load_model(main_model_cls, config_cls, self.comm_manager)

        if self.mtp_worker is not None:
            if mtp_model_cls is not None:
                logging.info("Loading mtp model...")
                self.mtp_worker.mtp_model_worker.load_model(mtp_model_cls, config_cls, self.comm_manager)
                self.mtp_worker.share_weights_from_main_model(self.main_worker.model)
            else:
                model_name = self.infer_config.model_config.model_name
                raise ValueError(f"next_n > 0 enables speculative inference, but {model_name} dosen't" + \
                                 "contain an MTP model and doesn't support speculative inference; set next_n to 0")

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

    @property
    def mtp_model(self):
        """Get the mtp model."""
        return self.mtp_worker.mtp_model_worker.model if self.mtp_worker else None

    def _build_model_inputs(
            self,
            input_ids: torch.Tensor,
            is_prefill: bool,
            batch: Batch = None,
            cycle_idx: int = 0,
        ):
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
            actual_seq_lengths_q = torch.arange(1, batch_size + 1, dtype=torch.long, device=self.device) * seq_len
            if self.infer_config.model_config.packed_sequence:
                actual_seq_lengths_kv = actual_seq_lengths_kv.cumsum(0)
        else:
            if self.mtp_worker:
                input_ids, kv_len, position_ids = self.mtp_worker.get_main_model_inputs(input_ids, batch)
                seq_len = input_ids.shape[-1]
            else:
                kv_len = get_forward_metadata().kv_len.to(self.device) + 1
                position_ids = kv_len.view(-1, seq_len)
            actual_seq_lengths_kv = kv_len + 1
            actual_seq_lengths_q = torch.arange(1, batch_size + 1, dtype=torch.long, device=self.device) * seq_len
            # npugraph_ex special handling
            if self.exe_mode == "npugraph_ex":
                actual_seq_lengths_kv = actual_seq_lengths_kv.detach().cpu().numpy().tolist()
                actual_seq_lengths_q = actual_seq_lengths_q.detach().cpu().numpy().tolist()

        # Set forward metadata
        set_forward_metadata(
            is_prefill=is_prefill,
            kv_len=kv_len,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            actual_seq_lengths_q=actual_seq_lengths_q,
            attention_mask=attention_mask,
        )
        model_inputs = {
            "input_ids": input_ids.contiguous(),
            "position_ids": position_ids,
            "forward_metadata": get_forward_metadata(),
        }
        if is_prefill:
            model_inputs.update({"cycle_idx": cycle_idx})
        return model_inputs

    def _get_warmup_shape(self):
        """Calculate warm-up input shapes based on batch_size_per_dp_rank/prefill_mini_batch and config."""
        prefill_batch_size = self.prefill_mini_batch
        decode_batch_size = self.infer_config.scheduler_config.batch_size_per_dp_rank
        seq_len = self.infer_config.scheduler_config.input_max_len

        return prefill_batch_size, decode_batch_size, seq_len

    def warm_up(self):
        """Execute warm-up by running exactly one prefill and one decode step of main and mtp model.
        Triggers graph compilation during decode if graph mode enabled.
        """
        logging.info("Starting warm-up...")

        # 1. Get warm-up shape info from config
        prefill_batch_size, decode_batch_size, seq_len = self._get_warmup_shape()

        # 2. Execute ONE prefill step with dummy inputs
        logging.info("Warm-up [Main]: executing model prefill step...")
        dummy_input_ids = torch.zeros((prefill_batch_size, seq_len), dtype=torch.long, device=self.device)
        model_inputs = self._build_model_inputs(dummy_input_ids, is_prefill=True)
        output, _ = self.main_worker.inference(model_inputs, is_prefill=True)
        if self.mtp_worker:
            logging.info("Warm-up [MTP]: executing model prefill step...")
            prev_hidden_states = output[1].view(prefill_batch_size, -1, output[1].shape[-1])
            model_inputs['prev_hidden_states'] = prev_hidden_states
            output, _ = self.mtp_worker.mtp_model_worker.inference(model_inputs, is_prefill=True, is_mtp=True)

        seq_len = 1 if self.next_n == 0 else self.next_n + 1
        kv_len_int = torch.max(model_inputs["position_ids"][0]).item()
        dummy_kv_len = torch.tensor([kv_len_int] * decode_batch_size, dtype=torch.long, device=self.device).contiguous()
        set_forward_metadata(kv_len=dummy_kv_len)

        # 3. Execute ONE decode step with graph compilation
        logging.info("Warm-up [Main]: executing model decode step...")
        dummy_input_ids = torch.zeros((decode_batch_size, seq_len), dtype=torch.long, device=self.device)
        model_inputs = self._build_model_inputs(dummy_input_ids, is_prefill=False)

        # Trigger graph compilation if graph mode enabled and not yet compiled 
        if self.exe_mode in ["ge_graph", "npugraph_ex"]:
            logging.info("Warm-up: triggering graph compilation...")
            self.main_worker.compile_model()
        output, _ = self.main_worker.inference(model_inputs, is_prefill=False)

        if self.mtp_worker:
            logging.info("Warm-up [MTP]: executing mtp model decode step...")
            model_inputs['prev_hidden_states'] = output[1]
            if self.exe_mode in ["ge_graph", "npugraph_ex"]:
                logging.info("Warm-up: triggering graph compilation...")
                self.mtp_worker.mtp_model_worker.compile_model()
            _ = self.mtp_worker.mtp_model_worker.inference(model_inputs, is_prefill=False, is_mtp=True)

        logging.info("Warm-up completed successfully.")

    def forward_batch(self, batch: Batch) -> Dict[str, Any]:
        """Execute forward pass for a batch of requests.

        This is the primary interface for Scheduler to execute both
        prefill and decode phases through a unified interface.

        Args:
            batch: Batch object containing requests, input_ids, position_ids, etc.

        Returns:
            Dict containing:
                - "next_tokens": Dict mapping request_id to generated token IDs
                - "logits": [batch_size, seq_len, vocab_size] model outputs
        """
        # Materialize request states into batch tensors before profiling and model input construction.
        batch.build_tensors_from_requests()
        inputs_ids = batch.input_ids.to(self.device)
        batch.input_ids = inputs_ids

        # Set profiler status based on prefill or decode
        self.profiler.set_status(batch.is_prefill)
        # Prepare model inputs
        model_inputs = self._build_model_inputs(
                inputs_ids,
                is_prefill=batch.is_prefill,
                batch=batch,
                cycle_idx=batch.cycle_idx,
                )

        # Run inference
        output, infer_time = self.main_worker.inference(model_inputs, is_prefill=batch.is_prefill)

        # Handle different output formats: tuple (logits, prev_hidden_states) or tensor (logits)
        if isinstance(output, tuple):
            logits, prev_hidden_states = output
        else:
            logits = output
            prev_hidden_states = None

        # Update KV cache length after prefill
        if batch.is_prefill:
            kv_lens = torch.max(model_inputs["position_ids"], dim=1)[0]
            set_forward_metadata(kv_len=kv_lens)

        # Sample next tokens
        next_tokens = self._sample_tokens(batch, logits)

        if self.mtp_worker:
            accepted_num = self.verify_spec_tokens(batch, next_tokens)
            infer_time_mtp = self.mtp_worker.inference(batch, next_tokens, accepted_num,
                                                       model_inputs, prev_hidden_states)
            infer_time += infer_time_mtp

        self.profiler.step()
        next_tokens_by_request = batch.update_requests_from_batch(next_tokens, infer_time)
        return {
            "next_tokens": next_tokens_by_request,
            "logits": logits,
        }

    def _sample_tokens(
        self,
        batch: Batch,
        logits: torch.Tensor
    ) -> Dict[int, int]:
        """Sample next tokens for each request in batch."""
        if batch.is_prefill:
            logits = logits[:, -1:, :]
        next_tokens = torch.argmax(logits, dim=-1).to(self.device)
        return next_tokens

    def verify_spec_tokens(self, batch, main_next_tokens):
        '''
        Verify spec tokens with main model's output, stop accepting tokens if rejection occurs in a batch.
        Each batch would process verification seperately.
        '''
        if batch.is_prefill:
            return torch.zeros([main_next_tokens.shape[0]], dtype=torch.int64, device=self.device) # shape: (Batch,)
        else: # after main decode
            batch_size = batch.input_ids.shape[0]
            token_mask = batch.mtp_infos.spec_tokens == main_next_tokens[:batch_size, :self.next_n]
            has_invalid = (token_mask == False).any(dim=-1)
            invalid_pos = (token_mask == False).int().argmax(dim=-1)
            accepted_num = torch.where(has_invalid, invalid_pos, token_mask.shape[-1]).to(self.device)
        return accepted_num
