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
import torch_npu
from transformers import AutoTokenizer

from executor.core.config import InferenceConfig
from executor.utils import get_default_group
from executor.utils.forward_metadata import PrefillCPMetaData, set_forward_metadata, get_forward_metadata
from executor.utils.profiler_context import ProfilerManager
from executor.core.model_worker import ModelWorker, MTPWorker
from executor.core.kv_cache import KVCacheManager, ModelCacheInfo, create_single_type_managers
from executor.core.kv_cache.cache_utils import allocate_cache_tensors, calculate_block_num, \
    prepare_block_tables, prepare_slot_mapping, validate_cache_info
from ..forward_data_info import Batch, StepOutput

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
        self.eos_token_id = None
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
        custom_params = self.infer_config.model_config.custom_params
        enable_afd = custom_params.get("enable_afd", False)
        self.is_afd_ffn_rank = enable_afd and self.global_rank < self.world_size // 2
        self._init_device()

        # Initialize workers
        self.main_worker = ModelWorker(self.infer_config, self.device)
        self.mtp_worker = MTPWorker(self.infer_config, self.device) if self.next_n > 0 \
            and not self.is_afd_ffn_rank else None

        # Profiling configuration
        self.enable_profiler = self.infer_config.model_config.enable_profiler
        self.output_path = os.path.join(os.getenv("WORK_DIR", "."), os.getenv("RES_PATH", ""))
        self.profiler = ProfilerManager(self.enable_profiler, self.output_path)

    def _init_device(self):
        """Initialize NPU device and the dist process group.

        CommManager is built later by ModelWorker.init(); model constructors
        register their own business communication groups after it is available.
        """
        logger.info("Set execution using npu index: %s, global: %s", self.local_rank, self.global_rank)
        self.device = torch.device("%s:%s" % ("npu", self.local_rank))
        torch.npu.set_device(self.device)

        if torch.npu.is_available() and self.world_size > 1:
            default_pg = get_default_group()
            if default_pg is None:
                pg_options = None
                if self.infer_config.model_config.platform_version.is_ascend_950():
                    pg_options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
                    pg_options.hccl_config = {"hccl_op_expansion_mode": 5}
                torch.distributed.init_process_group(
                    backend="hccl",
                    world_size=self.world_size,
                    rank=self.global_rank,
                    pg_options=pg_options,
                )

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
                raise ValueError(f"next_n > 0 enables speculative inference, but {model_name} doesn't " +
                                 "contain an MTP model and doesn't support speculative inference; set next_n to 0")

        # Initialize tokenizer (from main worker)
        self.hf_config = self.main_worker.hf_config
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.infer_config.model_config.model_path,
            padding_side="right",
            truncation_side='right',
            trust_remote_code=True
        )
        if getattr(self.hf_config, "eos_token_id", None) is not None:
            self.tokenizer.eos_token_id = self.hf_config.eos_token_id
        self.eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

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
        # Offload models may still need model-owned HBM workspace, for example
        # selected KV buffers. Reserve that budget before sizing framework KV
        # blocks, then initialize the workspace after KV cache tensors exist.
        offload_workspace_npu_bytes = self._get_offload_workspace_npu_bytes()
        block_num_by_type = calculate_block_num(
            infer_config=self.infer_config,
            cache_info=cache_info,
            offline_max_len=self.max_total_len if not self.is_online else None,
            tp_group=self.comm_manager.get_group("attn_tp_group"),
            extra_reserved_memory_bytes=offload_workspace_npu_bytes,
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
        if self.mtp_worker is not None:
            self.mtp_worker.kvcache_manager = self.kvcache_manager
        self._init_offload_workspace()

    def _get_offload_workspace_npu_bytes(self) -> int:
        memory_infos = [self.main_worker.get_offload_workspace_memory_info()]
        if self.mtp_worker is not None:
            memory_infos.append(self.mtp_worker.mtp_model_worker.get_offload_workspace_memory_info())

        total_npu_bytes = 0
        for memory_info in memory_infos:
            if memory_info is None:
                continue
            total_npu_bytes += memory_info.npu_bytes
            for item in memory_info.items:
                logger.info(
                    "Offload workspace memory: name=%s location=%s bytes=%s",
                    item.name,
                    item.location,
                    item.bytes,
                )
        if total_npu_bytes > 0:
            logger.info("Reserve offload workspace NPU memory: %s bytes", total_npu_bytes)
        return total_npu_bytes

    def _init_offload_workspace(self):
        self.main_worker.init_offload_workspace()
        if self.mtp_worker is not None:
            self.mtp_worker.mtp_model_worker.init_offload_workspace()

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
        cp_metadata = None

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
            # Get block_table and slot_mapping
            batch_size = actual_seq_lengths_cu_q.shape[0]
            block_table_max_lens = self.kvcache_manager.get_block_table_max_lens(self.max_total_len)
            block_tables = prepare_block_tables(batch.requests if batch else None, self.kvcache_manager,
                                                block_table_max_lens, device=self.device, batch_size=batch_size)
            slot_mapping = prepare_slot_mapping(
                position_ids,
                actual_seq_lengths_cu_q,
                self.kvcache_manager,
                block_tables,
            )
        else:
            block_tables = None
            slot_mapping = None
        if is_prefill and self.infer_config.parallel_config.cp_size > 1:
            input_ids, position_ids, cp_metadata = self._apply_prefill_cp(
                input_ids=input_ids,
                position_ids=position_ids,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                block_table=block_tables,
                slot_mapping=slot_mapping,
                batch=batch,
            )
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
            cp_metadata=cp_metadata,
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

    def _apply_prefill_cp(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        actual_seq_lengths_kv: torch.Tensor,
        block_table: Optional[Dict[str, torch.Tensor]],
        slot_mapping: Optional[Dict[str, torch.Tensor]],
        batch: Optional[Batch] = None,
    ):
        """Pad packed prefill TND input and slice it in PCP head-tail style."""
        parallel_config = self.infer_config.parallel_config
        cp_size = parallel_config.cp_size
        if self.exe_mode not in ["eager", "ge_graph", "npugraph_ex"]:
            raise ValueError(f"Prefill CP does not support exe_mode={self.exe_mode}.")
        if self.world_size != cp_size:
            raise ValueError(
                f"Prefill CP currently follows the legacy constraint world_size == cp_size, "
                f"got world_size={self.world_size}, cp_size={cp_size}."
            )
        if parallel_config.attn_tp_size != 1:
            raise ValueError(
                f"Phase1 prefill CP does not support attn_tp_size={parallel_config.attn_tp_size}."
            )

        actual_seq_lengths_kv = actual_seq_lengths_kv.to(self.device)
        batch_size = actual_seq_lengths_kv.numel()
        if batch_size == 0:
            raise ValueError("Prefill CP requires at least one local request per rank in Phase1.")

        cp_segment_num = cp_size * 2
        cp_rank = self.global_rank % cp_size
        seq_lens = [int(seq_len) for seq_len in actual_seq_lengths_kv.detach().cpu().tolist()]
        starts = [0]
        padded_starts = [0]
        padded_seq_lens = []
        padded_input_ids = []
        padded_position_ids = []
        for seq_len in seq_lens:
            starts.append(starts[-1] + seq_len)
            segment_len = (seq_len + cp_segment_num - 1) // cp_segment_num
            padded_seq_len = segment_len * cp_segment_num
            padded_seq_lens.append(padded_seq_len)
            padded_starts.append(padded_starts[-1] + padded_seq_len)
        compute_slot_mapping = {} if slot_mapping is not None else None
        full_block_table_len = max(
            (padded_seq_len + self.block_size - 1) // self.block_size
            for padded_seq_len in padded_seq_lens
        )
        if slot_mapping is not None:
            compute_cache_len = full_block_table_len * self.block_size
            for cache_type in slot_mapping:
                mapping_list = []
                for req_idx, padded_seq_len in enumerate(padded_seq_lens):
                    start = req_idx * compute_cache_len
                    mapping_list.append(
                        torch.arange(
                            start,
                            start + padded_seq_len,
                            dtype=slot_mapping[cache_type].dtype,
                            device=self.device,
                        )
                    )
                compute_slot_mapping[cache_type] = (
                    torch.cat(mapping_list) if mapping_list else slot_mapping[cache_type].new_empty(0)
                )

        for seq_len, padded_seq_len, seq_start in zip(seq_lens, padded_seq_lens, starts[:-1]):
            real_ids = input_ids[seq_start:seq_start + seq_len]
            pad_len = padded_seq_len - seq_len
            if pad_len > 0:
                real_ids = torch.cat([
                    real_ids,
                    torch.zeros(pad_len, dtype=input_ids.dtype, device=input_ids.device),
                ])
            padded_input_ids.append(real_ids)
            padded_position_ids.append(
                torch.arange(padded_seq_len, dtype=position_ids.dtype, device=position_ids.device)
            )

        input_ids = torch.cat(padded_input_ids) if padded_input_ids else input_ids.new_empty(0)
        position_ids = torch.cat(padded_position_ids) if padded_position_ids else position_ids.new_empty(0)

        if block_table is not None:
            block_table_for_compute = {
                cache_type: torch.arange(
                    0,
                    len(seq_lens) * full_block_table_len,
                    dtype=table.dtype,
                    device=self.device,
                ).reshape(len(seq_lens), full_block_table_len)
                for cache_type, table in block_table.items()
            }
        else:
            block_table_for_compute = None

        prev_lens = []
        next_lens = []
        kv_len_prev = []
        kv_len_next = []
        global_valid_indices_list = []

        def get_rank_local_indices(rank):
            # Zigzag CP assigns each rank one head segment and one mirrored
            # tail segment from every padded request.
            rank_prev_indices = []
            rank_next_indices = []
            for padded_seq_start, padded_seq_len in zip(padded_starts[:-1], padded_seq_lens):
                segment_len = padded_seq_len // cp_segment_num if padded_seq_len > 0 else 0
                prev_start = rank * segment_len
                next_segment = cp_segment_num - rank - 1
                next_start = next_segment * segment_len
                rank_prev_indices.extend(
                    range(padded_seq_start + prev_start, padded_seq_start + prev_start + segment_len)
                )
                rank_next_indices.extend(
                    range(padded_seq_start + next_start, padded_seq_start + next_start + segment_len)
                )
            return rank_prev_indices, rank_next_indices

        for seq_len, padded_seq_len, padded_seq_start in zip(seq_lens, padded_seq_lens, padded_starts[:-1]):
            segment_len = padded_seq_len // cp_segment_num if padded_seq_len > 0 else 0

            prev_lens.append(segment_len)
            next_lens.append(segment_len)

            kv_len_prev.append((cp_rank + 1) * segment_len)
            kv_len_next.append((cp_segment_num - cp_rank) * segment_len)

            global_valid_indices_list.extend(range(padded_seq_start, padded_seq_start + seq_len))

        local_owner_valid_indices = []
        owner_request_indices = []
        local_owner_valid_padded = []
        requests = batch.requests if batch is not None else []
        for req_idx in range(batch_size):
            request_cp_rank = requests[req_idx].cp_rank if req_idx < len(requests) else req_idx % cp_size
            if request_cp_rank == cp_rank:
                owner_request_indices.append(req_idx)
                local_owner_valid_indices.extend(range(starts[req_idx], starts[req_idx + 1]))
                local_owner_valid_padded.extend(
                    range(padded_starts[req_idx], padded_starts[req_idx] + seq_lens[req_idx])
                )
        local_owner_request_indices = torch.tensor(owner_request_indices, dtype=torch.long, device=self.device)
        local_owner_valid_indices = torch.tensor(local_owner_valid_indices, dtype=torch.long, device=self.device)
        local_owner_valid_padded = torch.tensor(local_owner_valid_padded, dtype=torch.long, device=self.device)
        if slot_mapping is not None:
            local_owner_slot_mapping = {
                cache_type: torch.index_select(mapping, 0, local_owner_valid_indices)
                for cache_type, mapping in slot_mapping.items()
            }
        else:
            local_owner_slot_mapping = None
        if self.infer_config.disagg_config.disaggregation_mode == "PREFILL":
            persistent_valid_indices = torch.tensor(global_valid_indices_list, dtype=torch.long, device=self.device)
            persistent_slot_mapping = slot_mapping
            output_request_indices = (
                torch.arange(batch_size, dtype=torch.long, device=self.device)
                if cp_rank == 0 else torch.empty(0, dtype=torch.long, device=self.device)
            )
        else:
            persistent_valid_indices = local_owner_valid_padded
            persistent_slot_mapping = local_owner_slot_mapping
            output_request_indices = local_owner_request_indices

        local_prev_indices, local_next_indices = get_rank_local_indices(cp_rank)

        all_rank_local_indices = []
        for rank in range(cp_size):
            rank_prev_indices, rank_next_indices = get_rank_local_indices(rank)
            all_rank_local_indices.extend(rank_prev_indices + rank_next_indices)

        local_indices = torch.tensor(
            local_prev_indices + local_next_indices,
            dtype=torch.long,
            device=self.device,
        )
        restore_indices = torch.tensor(
            sorted(range(len(all_rank_local_indices)), key=all_rank_local_indices.__getitem__),
            dtype=torch.long,
            device=self.device,
        )
        prev_lens_tensor = torch.tensor(prev_lens, dtype=actual_seq_lengths_kv.dtype, device=self.device)
        next_lens_tensor = torch.tensor(next_lens, dtype=actual_seq_lengths_kv.dtype, device=self.device)
        kv_len_prev_tensor = torch.tensor(kv_len_prev, dtype=actual_seq_lengths_kv.dtype, device=self.device)
        kv_len_next_tensor = torch.tensor(kv_len_next, dtype=actual_seq_lengths_kv.dtype, device=self.device)
        cp_metadata = PrefillCPMetaData(
            enabled=True,
            cp_size=cp_size,
            global_padded_token_num=padded_starts[-1],
            local_token_num=local_indices.numel(),
            local_indices=local_indices,
            restore_indices=restore_indices,
            global_valid_indices=torch.tensor(global_valid_indices_list, dtype=torch.long, device=self.device),
            global_slot_mapping=compute_slot_mapping,
            global_block_table=block_table_for_compute,
            persistent_valid_indices=persistent_valid_indices,
            persistent_slot_mapping=persistent_slot_mapping,
            output_request_indices=output_request_indices,
            actual_seq_q_prev=prev_lens_tensor.cumsum(dim=0),
            actual_seq_q_next=next_lens_tensor.cumsum(dim=0),
            kv_len_prev=kv_len_prev_tensor,
            kv_len_next=kv_len_next_tensor,
            local_prev_token_num=sum(prev_lens),
            local_next_token_num=sum(next_lens),
        )
        return input_ids, position_ids, cp_metadata

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

    def _prepare_mtp_next_tokens(self, next_tokens: torch.Tensor, model_inputs: Dict[str, Any],
                                 is_prefill: bool) -> torch.Tensor:
        """Expand owner-local prefill CP samples to the global request order for MTP."""
        if not is_prefill:
            return next_tokens
        forward_metadata = model_inputs.get("forward_metadata")
        cp_metadata = getattr(forward_metadata, "cp_metadata", None)
        if cp_metadata is None or not cp_metadata.enabled:
            return next_tokens

        global_batch_size = forward_metadata.actual_seq_lengths_kv.numel()
        local_next_tokens = next_tokens.to(self.device).contiguous()
        output_indices = cp_metadata.output_request_indices.to(self.device)
        if local_next_tokens.shape[0] != output_indices.numel():
            raise RuntimeError(
                "CP MTP next_tokens must match output_request_indices before global restore, "
                f"got next_tokens={local_next_tokens.shape[0]}, indices={output_indices.numel()}."
            )
        mtp_next_tokens = local_next_tokens.new_zeros(
            (global_batch_size,) + tuple(local_next_tokens.shape[1:])
        )
        if output_indices.numel() > 0:
            mtp_next_tokens.index_copy_(0, output_indices, local_next_tokens)
        torch.distributed.all_reduce(
            mtp_next_tokens,
            op=torch.distributed.ReduceOp.SUM,
            group=self.comm_manager.get_group("cp_group"),
        )
        return mtp_next_tokens

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
        warmup_role = "AFD FFN" if self.is_afd_ffn_rank else "Main"

        if self.infer_config.disagg_config.disaggregation_mode in ["NONE", "PREFILL"]:
            # 2. Execute ONE prefill step with dummy inputs (packed sequence format)
            logger.info("Warm-up [%s]: executing model prefill step...", warmup_role)
            dummy_seq_lens = torch.tensor([seq_len] * prefill_batch_size, dtype=torch.long, device=self.device)
            if not hasattr(self.hf_config, "vocab_size") or self.hf_config.vocab_size is None:
                raise ValueError(
                    "hf_config is missing 'vocab_size', which is required to generate dummy input_ids "
                    "for the warmup phase. Please check that the model's config.json contains a valid "
                    "vocab_size field."
                )
            dummy_input_ids = torch.randint(
                0, self.hf_config.vocab_size,
                (prefill_batch_size * seq_len,), dtype=torch.long, device=self.device,
            )
            if self.is_afd_ffn_rank:
                num_tokens = prefill_batch_size * seq_len
                model_inputs = self._build_afd_ffn_inputs(num_tokens, is_prefill=True)
            else:
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
            logger.info("Warm-up [%s]: executing model decode step...", warmup_role)
            dummy_input_ids = torch.randint(
                0, self.hf_config.vocab_size,
                (decode_batch_size * seq_len,), dtype=torch.long, device=self.device,
            )
            if self.is_afd_ffn_rank:
                model_inputs = self._build_afd_ffn_inputs(decode_batch_size * seq_len, is_prefill=False)
            else:
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

    def _build_afd_ffn_inputs(self, num_tokens: int, is_prefill: bool) -> Dict:
        return self.main_worker.model.build_afd_inputs(
            num_tokens,
            is_prefill,
            dtype=self.main_worker.dtype,
            device=self.device,
        )

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
        if self.is_afd_ffn_rank:
            return self._forward_afd_ffn_batch(batch)

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

        selected_logits = logits
        cp_metadata = None
        if batch.is_prefill:
            kv_len = batch.seq_lens.to(self.device) - 1
            cp_metadata = getattr(model_inputs["forward_metadata"], "cp_metadata", None)
            if cp_metadata is not None and cp_metadata.enabled:
                output_indices = cp_metadata.output_request_indices.to(self.device)
                kv_len = torch.index_select(kv_len, 0, output_indices)
                # The model returns global CP prefill logits; only selected
                # request rows are consumed by sampling and request updates.
                selected_logits = torch.index_select(logits, 0, output_indices)
            set_forward_metadata(kv_len=kv_len)

        next_tokens = self._sample_tokens(batch, selected_logits)

        infer_times_mtp: list[float] = []
        if self.mtp_worker:
            accepted_num = self.verify_spec_tokens(batch, next_tokens)
            mtp_next_tokens = self._prepare_mtp_next_tokens(next_tokens, model_inputs, batch.is_prefill)
            infer_times_mtp = self.mtp_worker.inference(batch, mtp_next_tokens, accepted_num,
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
            eos_token_id=self.eos_token_id,
        )
        return {
            "next_tokens": next_tokens_by_request,
            "logits": selected_logits,
            "inference_time": infer_time_total,
            "inference_time_main": infer_time_main,
            "inference_times_mtp": infer_times_mtp,
        }

    def _forward_afd_ffn_batch(self, batch: Batch) -> Dict[str, Any]:
        self.profiler.set_status(batch.is_prefill)
        decode_q_len = 1 if self.next_n == 0 else self.next_n + 1
        if batch.is_prefill:
            input_num_tokens = batch.total_tokens
        else:
            # Match the Attention side's padded decode send shape.
            input_num_tokens = self.infer_config.scheduler_config.batch_size_per_dp_rank * decode_q_len

        model_inputs = self._build_afd_ffn_inputs(
            input_num_tokens,
            batch.is_prefill,
        )
        _, infer_time_main = self.main_worker.inference(model_inputs, is_prefill=batch.is_prefill)

        actual_batch = len(batch.requests)
        if batch.is_prefill:
            set_forward_metadata(kv_len=batch.seq_lens.to(self.device) - 1)
            next_tokens = torch.zeros((actual_batch, 1), dtype=torch.long, device=self.device)
        else:
            kv_len = torch.tensor(
                [request.computed_len + 1 for request in batch.requests],
                dtype=torch.long,
                device=self.device,
            )
            set_forward_metadata(kv_len=kv_len)
            next_tokens = torch.zeros((actual_batch, 1), dtype=torch.long, device=self.device)

        next_tokens_by_request = batch.update_requests_from_batch(
            batch.is_prefill,
            next_tokens,
            infer_time_main,
        )
        self.profiler.step()
        return {
            "next_tokens": next_tokens_by_request,
            "logits": None,
            "inference_time": infer_time_main,
            "inference_time_main": infer_time_main,
            "inference_times_mtp": [],
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
        Each batch processes verification separately.
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
