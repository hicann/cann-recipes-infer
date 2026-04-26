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

"""MTP Worker for speculative inference with multi-token prediction.

This module provides the MTPWorker class that handles multi-token prediction
for speculative decoding. It orchestrates multiple steps of MTP model inference
to draft tokens that will be verified by the main model.
"""

import logging
from typing import Dict, Optional, Tuple

import torch

from executor.core.config import InferenceConfig
from executor.core.model_worker.model_worker import ModelWorker
from executor.utils.forward_metadata import set_forward_metadata, get_forward_metadata
from ..types_.types import MTPInfo, Batch
from executor.core.kv_cache.cache_utils import prepare_slot_mapping


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


class MTPWorker:
    """Worker class for executing MTP (Multi-Token Prediction) model inference.

    This class encapsulates the MTP model worker and provides methods for:
    - Getting MTP model inputs
    - Processing MTP model outputs
    - Executing multi-step speculative inference (propose)

    The MTPWorker is responsible for the small draft model that predicts
    multiple tokens in speculative decoding.

    Member Variables:
        Config:
            infer_config: Inference configuration containing all runtime settings.
            next_n: Number of speculative tokens to predict per step.
            exe_mode: Execution mode (eager, npugraph_ex, etc.).
            batch_size_per_dp_rank: Batch size for each dp rank.

        Model Components:
            device: NPU device for computation.
            mtp_model_worker: ModelWorker instance wrapping the MTP model.
    """

    def __init__(self, infer_config: InferenceConfig, device):
        """Initialize MTPWorker with inference configuration and device."""
        # Config
        self.infer_config = infer_config
        self.next_n = self.infer_config.model_config.next_n
        self.exe_mode = self.infer_config.model_config.exe_mode
        self.batch_size_per_dp_rank = self.infer_config.scheduler_config.batch_size_per_dp_rank
        self.block_size = self.infer_config.scheduler_config.block_size

        # Model Components
        self.device = device
        self.mtp_model_worker = ModelWorker(self.infer_config, self.device)

    def share_weights_from_main_model(self, main_model):
        """Share reusable weights from the main model to the MTP model when missing."""
        mtp_model = self.mtp_model_worker.model

        # Share lm_head if mtp_model's lm_head is None
        if hasattr(mtp_model, 'lm_head') and mtp_model.lm_head is None:
            if hasattr(main_model, 'lm_head'):
                mtp_model.lm_head = main_model.lm_head
            else:
                raise ValueError(
                    f"Current MTP model lacks and needs to reuse the main model's lm_head, "
                    f"but lm_head cannot be found in {main_model.__class__.__name__}. "
                    "Please check the main model's lm_head."
                )

        # Share embed_tokens if mtp_model's embed_tokens is None
        if hasattr(mtp_model.model, 'embed_tokens') and mtp_model.model.embed_tokens is None:
            if hasattr(main_model.model, 'embed_tokens'):
                mtp_model.model.embed_tokens = main_model.model.embed_tokens
            else:
                raise ValueError(
                    "Current MTP model lacks and needs to reuse the main model's embed_tokens, "
                    f"but embed_tokens cannot be found in {main_model.__class__.__name__}. "
                    "Please check the main model's embed_tokens."
                )

    @staticmethod
    def _pad_seq_len_to_size(tensor, size):
        """Pad or truncate tensor to the specified size along sequence dimension."""
        if tensor.dim() < 2:
            raise ValueError("Tensor must have at least 2 dimensions, where dim 1 is seq_len")
        if tensor.shape[1] >= size:
            # Truncate if tensor is already larger than target size
            return tensor[:, :size]
        else:
            # Pad with zeros if tensor is smaller than target size
            return torch.cat(
                [
                    tensor,
                    torch.zeros((tensor.shape[0], size - tensor.shape[1], *tensor.shape[2:]),
                                dtype=tensor.dtype, device=tensor.device),
                ],
                dim=1,
            )

    def get_main_model_inputs(self, input_ids, batch):
        """Build main model decode inputs. Returns 1D packed input_ids and position_ids."""
        kv_len = get_forward_metadata().kv_len.to(self.device)
        q_len = self.next_n + 1
        if batch:
            spec_tokens = batch.mtp_infos.spec_tokens
            # input_ids is [B] 1D, unsqueeze to [B, 1] for concat with spec_tokens [B, next_n]
            input_ids = torch.cat([input_ids.unsqueeze(1), spec_tokens], dim=1)
            input_ids = input_ids[:, -q_len:].reshape(-1).clone()
            # Rewind kv_len to the accepted prefix before the next main-model verification step.
            # The draft and main models share forward_metadata, and draft inference
            # increments kv_len by (self.next_n - 1), so we subtract it here and add
            # 1 back to align with the current main-model verification step.
            kv_len = kv_len - (self.next_n - 1) + 1
        indices = torch.arange(q_len - 1, -1, -1, device=self.mtp_model_worker.device)
        position_ids = (kv_len.unsqueeze(1) - indices).clamp(min=0).reshape(-1)

        return input_ids, kv_len, position_ids

    def get_mtp_model_inputs(
        self,
        batch: Batch,
        main_next_tokens: torch.Tensor,
        model_inputs_main: Dict,
        prev_hidden_states: torch.Tensor,
    ) -> Dict:
        """Prepare inputs for MTP model inference based on current phase."""
        if batch.is_prefill:
            if batch.seq_lens is None:
                raise RuntimeError("seq_lens is required for packed MTP prefill.")

            seq_lens = batch.seq_lens.to(device=self.device, dtype=torch.long)
            packed_input_ids = batch.input_ids.to(self.device).view(-1)
            packed_hidden_states = prev_hidden_states.to(self.device).view(-1, prev_hidden_states.shape[-1])
            total_tokens = int(seq_lens.sum().item())
            if packed_hidden_states.shape[0] < total_tokens:
                raise RuntimeError("Packed prev_hidden_states is shorter than seq_lens in MTP prefill.")

            cu_seq_lens = seq_lens.cumsum(0)
            input_ids_mtp = packed_input_ids.roll(-1)
            input_ids_mtp[cu_seq_lens - 1] = main_next_tokens[:, 0].to(self.device)
            input_ids_mtp = input_ids_mtp[:total_tokens]
            prev_hidden_states = packed_hidden_states[:total_tokens]
            position_ids_mtp = model_inputs_main["position_ids"]
            forward_metadata_mtp = model_inputs_main["forward_metadata"]
        else:
            input_ids_mtp = main_next_tokens.reshape(-1).clone()
            position_ids_mtp = model_inputs_main["position_ids"]
            forward_metadata_mtp = get_forward_metadata()
        model_inputs = {
            "input_ids": input_ids_mtp.contiguous(),
            "position_ids": position_ids_mtp,
            "prev_hidden_states": prev_hidden_states,
            "forward_metadata": forward_metadata_mtp,
        }
        if batch.is_prefill:
            model_inputs.update({
                "request_offset": model_inputs_main.get("request_offset", 0),
            })

        return model_inputs

    def mtp_model_output_postprocess(
        self,
        model_inputs: Dict,
        logits: torch.Tensor,
        mtp_infos: MTPInfo,
    ) -> Dict:
        """Process MTP model output and update state for the next inference step."""
        forward_metadata = get_forward_metadata()
        next_tokens = torch.argmax(logits, dim=-1)
        batch_size = next_tokens.shape[0]
        q_len = self.next_n + 1
        # is_prefill_step = mtp_info.is_prefill
        if mtp_infos.is_prefill:
            # Prefill branch: skip MTP decode, advance kv_len directly
            kv_len = forward_metadata.kv_len + q_len - 1
            kv_len += self.next_n - 1
            spec_token = MTPWorker._pad_seq_len_to_size(next_tokens[:, -1:], self.next_n)
            mtp_infos.is_prefill = False
        else:
            # Decode branch: update state based on accepted tokens
            cur_idx = (mtp_infos.accepted_num.view(-1, 1) + 1).long()
            spec_token_idx = (cur_idx - 1).clamp(0, self.next_n)
            spec_token = torch.gather(next_tokens, dim=1, index=spec_token_idx)

        if mtp_infos.spec_tokens is None:
            # First speculative token: initialize accumulator
            mtp_infos.spec_tokens = spec_token
        else:
            # Subsequent tokens: append to accumulator
            mtp_infos.spec_tokens = torch.cat([mtp_infos.spec_tokens, spec_token], dim=-1)

    def get_next_mtp_model_inputs(
        self,
        model_inputs: Dict,
        mtp_infos: MTPInfo,
        prev_hidden_states: torch.Tensor,
    ) -> Dict:
        """Build model inputs for the next MTP iteration from processed outputs."""
        input_ids = model_inputs["input_ids"]
        q_len = self.next_n + 1
        batch_size = input_ids.shape[0] // q_len
        forward_metadata = get_forward_metadata()
        kv_len = forward_metadata.kv_len + 1
        actual_seq_lengths_kv = kv_len + 1
        actual_seq_lengths_q = torch.full((batch_size,), q_len, dtype=torch.long, device=self.device)
        actual_seq_lengths_cu_q = actual_seq_lengths_q.cumsum(0)
        actual_seq_lengths_cu_kv = actual_seq_lengths_kv.cumsum(0)
        actual_seq_lengths_cu_list_q = None
        actual_seq_lengths_cu_list_kv = None
        actual_seq_lengths_list_q = None
        actual_seq_lengths_list_kv = None
        if self.mtp_model_worker.exe_mode == "npugraph_ex":
            actual_seq_lengths_cu_list_q = actual_seq_lengths_cu_q.detach().cpu().numpy().tolist()
            actual_seq_lengths_cu_list_kv = actual_seq_lengths_cu_kv.detach().cpu().numpy().tolist()
            actual_seq_lengths_list_q = actual_seq_lengths_q.detach().cpu().numpy().tolist()
            actual_seq_lengths_list_kv = actual_seq_lengths_kv.detach().cpu().numpy().tolist()
        set_forward_metadata(kv_len=kv_len, is_prefill=False,
                             actual_seq_lengths_kv=actual_seq_lengths_kv,
                             actual_seq_lengths_cu_q=actual_seq_lengths_cu_q,
                             actual_seq_lengths_cu_kv=actual_seq_lengths_cu_kv,
                             actual_seq_lengths_cu_list_q=actual_seq_lengths_cu_list_q,
                             actual_seq_lengths_cu_list_kv=actual_seq_lengths_cu_list_kv,
                             actual_seq_lengths_list_q=actual_seq_lengths_list_q,
                             actual_seq_lengths_list_kv=actual_seq_lengths_list_kv)
        indices = torch.arange(q_len - 1, -1, -1, device=self.mtp_model_worker.device)
        position_ids = (kv_len.unsqueeze(1) - indices).reshape(-1)
        slot_mapping = prepare_slot_mapping(position_ids, actual_seq_lengths_cu_q,
                                             forward_metadata.block_table, self.block_size)
        set_forward_metadata(slot_mapping=slot_mapping)
        input_ids_2d = input_ids.view(batch_size, q_len)
        cur_tokens = MTPWorker._pad_seq_len_to_size(input_ids_2d, q_len + 1)
        cur_idx = (mtp_infos.accepted_num.view(-1, 1) + 1).long()
        last_spec_token = mtp_infos.spec_tokens[:, -1:]
        input_ids = cur_tokens.scatter_(dim=1, index=cur_idx, src=last_spec_token)[:, 1:].reshape(-1).clone()

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "prev_hidden_states": prev_hidden_states,
            "forward_metadata": get_forward_metadata(),
        }

    def inference(
        self,
        batch: Batch,
        main_next_tokens: torch.Tensor,
        accepted_num: torch.Tensor,
        model_inputs_main: Dict,
        prev_hidden_states: torch.Tensor,
    ):
        """Execute multi-step speculative inference to generate draft tokens."""
        # Determine number of MTP steps: single for mini-batch prefill, else next_n steps
        loop_mtp = 1 if batch.is_prefill else self.next_n
        infer_time_mtp = 0
        if not batch.mtp_infos:
            batch.mtp_infos = MTPInfo()
        batch.mtp_infos.set_mtp_info(
            accepted_num=accepted_num, is_prefill=batch.is_prefill, next_n=self.next_n, spec_tokens=None)

        # Step 1: Get initial MTP model inputs
        model_inputs_mtp = self.get_mtp_model_inputs(
            batch,
            main_next_tokens,
            model_inputs_main,
            prev_hidden_states)

        # Step 2: Loop through MTP inference steps
        for step_idx in range(loop_mtp):
            if step_idx > 0:
                model_inputs_mtp = self.get_next_mtp_model_inputs(model_inputs_mtp, batch.mtp_infos, prev_hidden_states)

            # Execute MTP model inference
            output, infer_time = self.mtp_model_worker.inference(
                model_inputs_mtp,
                is_prefill=batch.mtp_infos.is_prefill,
                is_mtp=True
            )
            infer_time_mtp += infer_time
            logits, prev_hidden_states = output

            # Process output after inference
            self.mtp_model_output_postprocess(
                model_inputs_mtp,
                logits,
                mtp_infos=batch.mtp_infos,
            )
        return infer_time_mtp
