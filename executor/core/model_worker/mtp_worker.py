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

"""Native multi-token-prediction backend for speculative decoding.

MTP reuses the target model's cache timeline and predicts one proposal token
per draft invocation. The worker accumulates ``next_n`` proposal tokens, then
builds one fixed-width ``next_n + 1`` target verification bucket. Framework
lifecycle, shared weights, and verification orchestration live in
``BaseSpeculativeWorker``; this module owns only MTP-specific state updates and
model-input construction.
"""

from typing import Dict

import torch

from executor.core.config import InferenceConfig
from executor.core.kv_cache.cache_utils import prepare_slot_mapping
from executor.core.model_worker.model_worker import ModelWorker
from executor.utils.forward_metadata import get_forward_metadata, set_forward_metadata
from .base_speculative_worker import BaseSpeculativeWorker
from ..forward_data_info import Batch, MTPInfo


class MTPWorker(BaseSpeculativeWorker):
    """Generate sequential MTP proposals against the target-model cache."""

    method = "mtp"
    state_cls = MTPInfo

    def __init__(
        self,
        infer_config: InferenceConfig,
        device,
    ):
        super().__init__(infer_config, device)
        self.mtp_model_worker = ModelWorker(
            infer_config,
            device,
            is_draft_model=True,
            model_path=infer_config.speculative_config.draft_model_path,
        )

    @property
    def model_worker(self) -> ModelWorker:
        return self.mtp_model_worker

    def warm_up_prefill(
        self,
        input_ids: torch.Tensor,
        seq_lens: torch.Tensor,
        model_inputs: Dict,
        prev_hidden_states: torch.Tensor,
    ) -> None:
        draft_inputs = dict(model_inputs)
        draft_inputs["prev_hidden_states"] = prev_hidden_states
        self.mtp_model_worker.inference(draft_inputs, is_prefill=True, is_mtp=True)

    def build_main_warmup_batch(self, input_ids: torch.Tensor) -> Batch:
        batch_size = input_ids.shape[0]
        batch = Batch(
            is_prefill=False,
            input_ids=input_ids,
        )
        batch.mtp_infos = MTPInfo(
            spec_tokens=input_ids.new_zeros((batch_size, self.next_n)),
        )
        return batch

    def warm_up_decode(self, model_inputs: Dict, main_output) -> None:
        draft_inputs = dict(model_inputs)
        draft_inputs["prev_hidden_states"] = main_output[1]
        if self.exe_mode in ["ge_graph", "npugraph_ex"]:
            self.mtp_model_worker.compile_model()
        self.mtp_model_worker.inference(draft_inputs, is_prefill=False, is_mtp=True)

    def get_main_model_inputs(self, input_ids, batch):
        """Build main model decode inputs. Returns packed inputs and verification width."""
        mtp_infos = batch.mtp_infos if batch is not None else None
        if mtp_infos is None or mtp_infos.spec_tokens is None:
            return None

        q_len = self.next_n + 1
        # Speculative tokens may come from CPU-side PD transfer metadata or a
        # dummy warm-up batch. Keep the batch state on the verification device.
        mtp_infos.spec_tokens = mtp_infos.spec_tokens.to(self.device)
        return self._build_main_verification_inputs(
            input_ids,
            mtp_infos.spec_tokens,
            q_len,
        )

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

            position_ids_mtp = model_inputs_main["position_ids"]
            forward_metadata_mtp = model_inputs_main["forward_metadata"]
            cp_metadata = getattr(forward_metadata_mtp, "cp_metadata", None)
            enable_cp = cp_metadata is not None and cp_metadata.enabled
            if enable_cp:
                seq_lens = forward_metadata_mtp.actual_seq_lengths_kv.to(device=self.device, dtype=torch.long)
                packed_input_ids = torch.index_select(
                    model_inputs_main["input_ids"].to(self.device).view(-1),
                    0,
                    cp_metadata.global_valid_indices,
                )
            else:
                seq_lens = batch.seq_lens.to(device=self.device, dtype=torch.long)
                packed_input_ids = batch.input_ids.to(self.device).view(-1)

            packed_hidden_states = prev_hidden_states.to(self.device).view(-1, prev_hidden_states.shape[-1])
            total_tokens = int(seq_lens.sum().item())
            expected_hidden_tokens = cp_metadata.local_token_num if enable_cp else total_tokens
            if packed_hidden_states.shape[0] < expected_hidden_tokens:
                raise RuntimeError("Packed prev_hidden_states is shorter than seq_lens in MTP prefill.")

            next_tokens = main_next_tokens.to(self.device)
            if enable_cp and next_tokens.shape[0] != seq_lens.numel():
                raise RuntimeError("CP MTP prefill requires global next tokens prepared by execution engine.")

            cu_seq_lens = seq_lens.cumsum(0)
            input_ids_mtp = packed_input_ids.roll(-1)
            input_ids_mtp[cu_seq_lens - 1] = next_tokens[:, 0]
            input_ids_mtp = input_ids_mtp[:total_tokens]
            prev_hidden_states = packed_hidden_states[:expected_hidden_tokens]
            if enable_cp:
                global_valid_indices = cp_metadata.global_valid_indices
                input_ids_mtp_padded = input_ids_mtp.new_zeros(cp_metadata.global_padded_token_num)
                input_ids_mtp_padded.index_copy_(0, global_valid_indices, input_ids_mtp)
                input_ids_mtp = input_ids_mtp_padded
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
    ) -> None:
        """Process MTP model output and update state for the next inference step."""
        forward_metadata = get_forward_metadata()
        cp_metadata = getattr(forward_metadata, "cp_metadata", None)
        if mtp_infos.is_prefill and cp_metadata is not None and cp_metadata.enabled:
            output_indices = cp_metadata.output_request_indices.to(logits.device)
            # MTP model also returns global CP prefill logits; postprocess only
            # updates speculative state for rows selected by the current rank.
            logits = torch.index_select(logits, 0, output_indices)
        next_tokens = torch.argmax(logits, dim=-1)
        q_len = self.next_n + 1
        if mtp_infos.is_prefill:
            # Prefill branch: skip MTP decode, advance kv_len directly
            kv_len = forward_metadata.kv_len + q_len - 1
            kv_len += self.next_n - 1
            spec_token = MTPWorker._pad_seq_len_to_size(next_tokens[:, -1:], self.next_n)
            set_forward_metadata(kv_len=kv_len, is_prefill=False)
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
            # NPU graph metadata keeps device tensors for operators and host
            # lists for static graph guards; both describe the same lengths.
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
        position_ids = (kv_len.unsqueeze(1) - indices).clamp(min=0).reshape(-1)
        slot_mapping = prepare_slot_mapping(
            position_ids,
            actual_seq_lengths_cu_q,
            self.kvcache_manager,
            forward_metadata.block_table,
        )
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
        """Execute multi-step speculative inference to generate draft tokens.

        Returns per-step inference times so the scheduler can log each step
        individually instead of only seeing the summed total.
        """
        # Prefill produces the first proposal token; decode recursively advances
        # the MTP model until the fixed-width verification bucket is complete.
        loop_mtp = 1 if batch.is_prefill else self.next_n
        infer_times: list[float] = []
        if not batch.mtp_infos:
            batch.mtp_infos = MTPInfo()
        batch.mtp_infos.set_mtp_info(
            accepted_num=accepted_num, is_prefill=batch.is_prefill,
            spec_tokens=None)

        model_inputs_mtp = self.get_mtp_model_inputs(
            batch,
            main_next_tokens,
            model_inputs_main,
            prev_hidden_states)

        for step_idx in range(loop_mtp):
            if step_idx > 0:
                model_inputs_mtp = self.get_next_mtp_model_inputs(model_inputs_mtp, batch.mtp_infos,
                                                                  prev_hidden_states)

            output, infer_time = self.mtp_model_worker.inference(
                model_inputs_mtp,
                is_prefill=batch.mtp_infos.is_prefill,
                is_mtp=True
            )
            infer_times.append(infer_time)
            logits, prev_hidden_states = output

            self.mtp_model_output_postprocess(
                model_inputs_mtp,
                logits,
                mtp_infos=batch.mtp_infos,
            )
        return infer_times
