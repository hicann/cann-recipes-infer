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

from executor.core.config import InferenceConfig
from executor.utils import get_default_group
from executor.utils.forward_metadata import set_forward_metadata, get_forward_metadata
from executor.utils.profiler_context import ProfilerManager
from executor.core.model_worker import ModelWorker, MTPWorker
from executor.core.kv_cache import KVCacheManager, ModelCacheInfo, create_single_type_managers
from executor.core.kv_cache.cache_utils import allocate_cache_tensors, calculate_block_num, \
    prepare_block_tables, prepare_slot_mapping, validate_cache_info
from ..types_.types import MTPInfo, Batch, StepOutput

torch.npu.config.allow_internal_format = True
logger = logging.getLogger(__name__)


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
        self.kvcache_manager = None
        self.comm_manager = None
        self.max_new_tokens = self.infer_config.scheduler_config.max_new_tokens
        self.input_truncated_len = self.infer_config.data_config.input_truncated_len
        self.block_size = self.infer_config.scheduler_config.block_size
        self.next_n = self.infer_config.model_config.next_n
        # PD (prefill or decode) roles hold only their own KV and do not need the MTP draft-token buffer
        self.is_online = (
            infer_config.disagg_config.disaggregation_mode in ("PREFILL", "DECODE")
        )
        if self.is_online:
            # no chunk so max_prefill_tokens is the max len of kv
            self.max_total_len = self.infer_config.scheduler_config.max_prefill_tokens + self.max_new_tokens
        else:
            # In offline MTP mode, reserve extra KV cache for the draft model's speculative forwards.
            self.max_total_len = self.input_truncated_len + self.max_new_tokens * (self.next_n + 1) + self.next_n
        self.block_table_max_len = int((self.max_total_len + self.block_size - 1) / self.block_size)
        self.exe_mode = self.infer_config.model_config.exe_mode

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
        """Initialize NPU device and the dist process group.

        CommManager is built later by ModelWorker.init() — it needs the
        loaded hf_config to size the moe_ep buffer correctly.
        """
        logger.info("Set execution using npu index: %s, global: %s", self.local_rank, self.global_rank)
        self.device = torch.device("%s:%s" % ("npu", self.local_rank))
        torch.npu.set_device(self.device)

        if torch.npu.is_available() and self.world_size > 1:
            default_pg = get_default_group()
            if default_pg is None:
                torch.distributed.init_process_group(
                    backend="hccl", world_size=self.world_size, rank=self.global_rank)

    def init(self, config_cls, main_model_cls, mtp_model_cls=None):
        """Bring the engine to ready: load the model (and MTP draft if any),
        build the comm_manager, set up tokenizer + KV cache.
        """
        logger.info("Loading main model...")
        # Primary worker creates the process-wide comm_manager (sized from
        # hf_config); secondary workers (e.g. MTP) reuse it.
        self.main_worker.init(main_model_cls, config_cls)
        self.comm_manager = self.main_worker.comm_manager
        cache_info = self.main_worker.get_cache_info()

        if self.mtp_worker is not None:
            if mtp_model_cls is not None:
                logger.info("Loading mtp model...")
                self.mtp_worker.mtp_model_worker.init(
                    mtp_model_cls, config_cls, comm_manager=self.comm_manager,
                )
                cache_info_mtp = self.mtp_worker.mtp_model_worker.get_cache_info()
                if cache_info and cache_info_mtp:
                    # Support page attention
                    cache_info = cache_info.merge(cache_info_mtp)
                self.mtp_worker.share_weights_from_main_model(self.main_worker.model)
            else:
                model_name = self.infer_config.model_config.model_name
                raise ValueError(f"next_n > 0 enables speculative inference, but {model_name} dosen't" +
                                 "contain an MTP model and doesn't support speculative inference; set next_n to 0")

        # Initialize tokenizer (from main worker)
        self.hf_config = self.main_worker.hf_config
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.infer_config.model_config.model_path,
            padding_side="right",
            truncation_side='right',
            trust_remote_code=True
        )

        # Initialize KV cache
        if cache_info is not None:
            # Support page attention management
            logging.info("Initializing KV cache with PageAttention management.")
            self._init_cache_manager(cache_info)
        else:
            # Not support page attention
            self.main_worker.init_kvcache()
            if self.mtp_worker is not None:
                # Support MTP
                self.mtp_worker.mtp_model_worker.init_kvcache()

    def _init_cache_manager(self, cache_info: ModelCacheInfo):
        validate_cache_info(cache_info)
        block_num_by_type = calculate_block_num(
            infer_config=self.infer_config,
            cache_info=cache_info,
            offline_max_len=self.max_total_len if not self.is_online else None,
            tp_group=self.comm_manager.get_group("attn_tp_group"),
        )
        allocate_cache_tensors(
            device=self.device,
            cache_info=cache_info,
            block_num_by_type=block_num_by_type,
        )

        single_type_managers = create_single_type_managers(
            cache_info=cache_info,
            block_num_by_type=block_num_by_type,
            max_model_len=self.max_total_len,
        )
        self.kvcache_manager = KVCacheManager(
            max_model_len=self.max_total_len,
            single_type_managers=single_type_managers,
            cache_info=cache_info,
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
        seq_lens: torch.Tensor = None,
        batch: Batch = None,
        request_offset: int = 0,
    ):
        """Build model inputs and set forward metadata."""
        attention_mask = ~torch.tril(torch.ones((2048, 2048), dtype=torch.bool, device=self.device))
        prompt_tokens = int(seq_lens.sum().item()) if is_prefill and seq_lens is not None else 0

        if is_prefill:
            position_ids = torch.cat(
                [torch.arange(seq_len, dtype=torch.long, device=self.device) for seq_len in seq_lens.tolist()]
            )
            actual_seq_lengths_kv = seq_lens.to(self.device)
            actual_seq_lengths_q = actual_seq_lengths_kv.clone()
            actual_seq_lengths_cu_kv = actual_seq_lengths_kv.cumsum(0)
            actual_seq_lengths_cu_q = actual_seq_lengths_cu_kv.clone()
            actual_seq_lengths_cu_list_kv = None
            actual_seq_lengths_cu_list_q = None
            kv_len_list = []
            cache_seq_len = self.input_truncated_len + self.max_new_tokens
            for req_idx, seq_len in enumerate(seq_lens.tolist()):
                slot_idx = request_offset + req_idx
                start = slot_idx * cache_seq_len
                kv_len_list.append(
                    torch.arange(start, start + seq_len, dtype=torch.long, device=self.device)
                )
            kv_len = torch.cat(kv_len_list) if kv_len_list else torch.empty(0, dtype=torch.long, device=self.device)
        else:
            if self.mtp_worker:
                input_ids, kv_len, position_ids = self.mtp_worker.get_main_model_inputs(input_ids, batch)
                seq_len = self.mtp_worker.next_n + 1
                # Pad to batch_size_per_dp_rank. kv_len is [actual_bs];
                # input_ids / position_ids are [actual_bs * seq_len].
                # _pad_batch works on dim-0, so only kv_len can use it directly;
                # the others need a seq_len multiplier.
                actual_bs = kv_len.shape[0]
                target_bs = self.infer_config.scheduler_config.batch_size_per_dp_rank
                if actual_bs < target_bs:
                    pad_bs = target_bs - actual_bs
                    pad_tokens = pad_bs * seq_len
                    input_ids = torch.cat([
                        input_ids,
                        torch.zeros(pad_tokens, dtype=input_ids.dtype, device=input_ids.device),
                    ])
                    position_ids = torch.cat([  # [batch, next_n + 1]
                        position_ids,
                        torch.zeros(pad_tokens, dtype=position_ids.dtype, device=position_ids.device),
                    ])
                kv_len = self._pad_batch(kv_len)
                batch_size = kv_len.shape[0]
            else:
                input_ids = self._pad_batch(input_ids)
                kv_len = self._pad_batch(get_forward_metadata().kv_len.to(self.device) + 1)
                position_ids = kv_len.clone()
                batch_size, seq_len = input_ids.shape[0], 1
            actual_seq_lengths_kv = kv_len + 1
            actual_seq_lengths_q = torch.full((batch_size,), seq_len, dtype=torch.long, device=self.device)
            actual_seq_lengths_cu_q = actual_seq_lengths_q.cumsum(0)
            actual_seq_lengths_cu_kv = actual_seq_lengths_kv.cumsum(0)
            actual_seq_lengths_cu_list_kv = None
            actual_seq_lengths_cu_list_q = None
            actual_seq_lengths_list_kv = None
            actual_seq_lengths_list_q = None
            # npugraph_ex special handling
            if self.exe_mode == "npugraph_ex":
                actual_seq_lengths_cu_list_kv = actual_seq_lengths_cu_kv.detach().cpu().numpy().tolist()
                actual_seq_lengths_cu_list_q = actual_seq_lengths_cu_q.detach().cpu().numpy().tolist()
                actual_seq_lengths_list_kv = actual_seq_lengths_kv.detach().cpu().numpy().tolist()
                actual_seq_lengths_list_q = actual_seq_lengths_q.detach().cpu().numpy().tolist()
        if self.kvcache_manager:
            # Get blcok_table and slot_mapping
            batch_size = actual_seq_lengths_cu_q.shape[0]
            block_tables = prepare_block_tables(batch.requests if batch else None, self.kvcache_manager,
                                                self.block_table_max_len, device=self.device, batch_size=batch_size)
            slot_mapping = prepare_slot_mapping(position_ids, actual_seq_lengths_cu_q, block_tables, self.block_size)
        else:
            block_tables = None
            slot_mapping = None
        set_forward_metadata(
            is_prefill=is_prefill,
            kv_len=kv_len,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            actual_seq_lengths_q=actual_seq_lengths_q,
            actual_seq_lengths_cu_kv=actual_seq_lengths_cu_kv,
            actual_seq_lengths_cu_q=actual_seq_lengths_cu_q,
            actual_seq_lengths_cu_list_kv=actual_seq_lengths_cu_list_kv if not is_prefill else None,
            actual_seq_lengths_cu_list_q=actual_seq_lengths_cu_list_q if not is_prefill else None,
            actual_seq_lengths_list_kv=actual_seq_lengths_list_kv if not is_prefill else None,
            actual_seq_lengths_list_q=actual_seq_lengths_list_q if not is_prefill else None,
            attention_mask=attention_mask,
            prompt_tokens=prompt_tokens,
            block_table=block_tables,
            slot_mapping=slot_mapping,
        )
        model_inputs = {
            "input_ids": input_ids.contiguous(),
            "position_ids": position_ids,
            "forward_metadata": get_forward_metadata(),
        }
        if is_prefill and not self.kvcache_manager:
            model_inputs.update({
                "request_offset": request_offset,
            })
        return model_inputs

    def _pad_batch(self, tensor: torch.Tensor, pad_value: int = 0) -> torch.Tensor:
        """Pad the batch dimension of a tensor to configured decode batch size."""
        target_batch = self.infer_config.scheduler_config.batch_size_per_dp_rank
        if tensor.size(0) >= target_batch:
            return tensor
        pad_rows = target_batch - tensor.size(0)
        pad_shape = (pad_rows, *tensor.shape[1:])
        return torch.cat([
            tensor,
            torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device),
        ], dim=0)

    def _get_warmup_shape(self):
        """Calculate warm-up input shapes based on current packed prefill/decode config."""
        prefill_batch_size = self.infer_config.scheduler_config.batch_size_per_dp_rank
        decode_batch_size = self.infer_config.scheduler_config.batch_size_per_dp_rank
        seq_len = self.input_truncated_len

        max_prefill_tokens = self.infer_config.scheduler_config.max_prefill_tokens
        if max_prefill_tokens > 0:
            seq_len = min(seq_len, max_prefill_tokens)
            prefill_batch_size = max(1, max_prefill_tokens // seq_len)

        return prefill_batch_size, decode_batch_size, seq_len

    def warm_up(self):
        """Execute warm-up by running exactly one prefill and one decode step of main and mtp model.
        Triggers graph compilation during decode if graph mode enabled.
        """
        logger.info("Starting warm-up...")

        # 1. Get warm-up shape info from config
        prefill_batch_size, decode_batch_size, seq_len = self._get_warmup_shape()

        if self.infer_config.disagg_config.disaggregation_mode in ["NONE", "PREFILL"]:
            # 2. Execute ONE prefill step with dummy inputs (packed sequence format)
            logger.info("Warm-up [Main]: executing model prefill step...")
            dummy_seq_lens = torch.tensor([seq_len] * prefill_batch_size, dtype=torch.long, device=self.device)
            dummy_input_ids = torch.zeros(prefill_batch_size * seq_len, dtype=torch.long, device=self.device)
            model_inputs = self._build_model_inputs(dummy_input_ids, is_prefill=True, seq_lens=dummy_seq_lens)
            output, _ = self.main_worker.inference(model_inputs, is_prefill=True)
            if self.mtp_worker:
                logger.info("Warm-up [MTP]: executing model prefill step...")
                prev_hidden_states = output[1]
                model_inputs['prev_hidden_states'] = prev_hidden_states
                output, _ = self.mtp_worker.mtp_model_worker.inference(model_inputs, is_prefill=True, is_mtp=True)


        if self.infer_config.disagg_config.disaggregation_mode in ["NONE", "DECODE"]:
            dummy_kv_len = torch.full(
                (decode_batch_size,),
                seq_len - 1,
                dtype=torch.long,
                device=self.device,
            )
            set_forward_metadata(kv_len=dummy_kv_len)
            # 3. Execute ONE decode step with graph compilation
            seq_len = 1 if self.next_n == 0 else self.next_n + 1
            logger.info("Warm-up [Main]: executing model decode step...")
            if self.mtp_worker:
                dummy_input_ids = torch.zeros(decode_batch_size * seq_len, dtype=torch.long, device=self.device)
            else:
                dummy_input_ids = torch.zeros(decode_batch_size, dtype=torch.long, device=self.device)
            model_inputs = self._build_model_inputs(dummy_input_ids, is_prefill=False)

            # Trigger graph compilation if graph mode enabled and not yet compiled
            if self.exe_mode in ["ge_graph", "npugraph_ex"]:
                logger.info("Warm-up: triggering graph compilation...")
                self.main_worker.compile_model()
            output, _ = self.main_worker.inference(model_inputs, is_prefill=False)

            if self.mtp_worker:
                logger.info("Warm-up [MTP]: executing mtp model decode step...")
                model_inputs['prev_hidden_states'] = output[1]
                if self.exe_mode in ["ge_graph", "npugraph_ex"]:
                    logger.info("Warm-up: triggering graph compilation...")
                    self.mtp_worker.mtp_model_worker.compile_model()
                _ = self.mtp_worker.mtp_model_worker.inference(model_inputs, is_prefill=False, is_mtp=True)

        logger.info("Warm-up completed successfully.")

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
        batch.build_tensors_from_requests()

        self.profiler.set_status(batch.is_prefill)
        inputs_ids = batch.input_ids.to(self.device)
        seq_lens = batch.seq_lens.to(self.device) if batch.seq_lens is not None else None
        model_inputs = self._build_model_inputs(
            inputs_ids,
            is_prefill=batch.is_prefill,
            seq_lens=seq_lens,
            batch=batch,
            request_offset=batch.request_offset,
        )

        # Run inference
        output, infer_time_main = self.main_worker.inference(model_inputs, is_prefill=batch.is_prefill)

        # Handle different output formats: tuple (logits, prev_hidden_states) or tensor (logits)
        if isinstance(output, tuple):
            logits, prev_hidden_states = output
        else:
            logits = output
            prev_hidden_states = None

        if batch.is_prefill:
            set_forward_metadata(kv_len=batch.seq_lens.to(self.device) - 1)

        next_tokens = self._sample_tokens(batch, logits)

        infer_times_mtp: list[float] = []
        if self.mtp_worker:
            accepted_num = self.verify_spec_tokens(batch, next_tokens)
            infer_times_mtp = self.mtp_worker.inference(batch, next_tokens, accepted_num,
                                                        model_inputs, prev_hidden_states)

        self.profiler.step()

        # Trim padded results to actual request count for request tracking
        actual_batch = len(batch.requests)
        if next_tokens.shape[0] > actual_batch:
            next_tokens = next_tokens[:actual_batch]
        kv_len = get_forward_metadata().kv_len
        if kv_len is not None and kv_len.shape[0] > actual_batch:
            set_forward_metadata(kv_len=kv_len[:actual_batch])

        infer_time_total = infer_time_main + sum(infer_times_mtp)
        next_tokens_by_request = batch.update_requests_from_batch(
            batch.is_prefill,
            next_tokens,
            infer_time_total,
        )
        return {
            "next_tokens": next_tokens_by_request,
            "logits": logits,
            "inference_time": infer_time_total,
            "inference_time_main": infer_time_main,
            "inference_times_mtp": infer_times_mtp,
        }

    def _sample_tokens(
        self,
        batch: Batch,
        logits: torch.Tensor
    ) -> torch.Tensor:
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
