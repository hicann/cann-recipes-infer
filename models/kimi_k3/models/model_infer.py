# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Fixed-batch offline generation loop for Kimi K3.

Prefill may be split into fixed-size mini batches.  Every mini-batch cycle is
still executed by the complete attention-TP group (token sequence parallel),
while cache writes are addressed with the request's position in the eventual
decode batch.  Decode therefore starts from the same resident cache tensors
without a cache merge or device-to-device copy.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .modules import CacheData, gather_sp_shards_to_owner


@dataclass
class DSparkRuntimeState:
    spec_tokens: torch.Tensor | None
    draft_probs: torch.Tensor | None
    num_accepted_tokens: torch.Tensor


@dataclass
class DSparkAcceptanceStats:
    total_accepted_tokens: int
    verify_count: int

    @classmethod
    def create(cls) -> "DSparkAcceptanceStats":
        return cls(0, 0)


class KimiK3Infer:
    def __init__(self, runner_settings: dict, model_runner, draft_model_runner=None):
        self.runner_settings = runner_settings
        self.model_runner = model_runner
        self.model = model_runner.model
        self.draft_model_runner = draft_model_runner
        self.draft_model = None if draft_model_runner is None else draft_model_runner.model
        self.tokenizer = model_runner.tokenizer
        self.device = model_runner.device
        data_config = runner_settings.get("data_config", {})
        self.batch_size = data_config.get(
            "batch_size_per_rank", data_config.get("batch_size", 1)
        )
        self.input_max_len = data_config.get("input_max_len", 128)
        self.max_new_tokens = data_config.get("max_new_tokens", 128)
        self.temperature = data_config.get("temperature", 1.0)
        model_config = runner_settings.get("model_config", {})
        self.draft_model_type = model_config.get("draft_model_type", "none")
        self.next_n = int(model_config.get("next_n", 0))
        self.uses_dspark = self.draft_model_type == "dspark"
        self.prefill_mini_batch_size = model_runner.prefill_mini_batch_size
        self.mini_batch = (
            self.prefill_mini_batch_size
            if self.prefill_mini_batch_size > 0
            else self.batch_size
        )
        self.prefill_cycles = model_runner.prefill_cycles
        self.cache_manager = CacheData(self.model.config, runner_settings, self.device)
        self.attn_metadata = self.model.attn_metadata
        self.attn_tp_size = self.model.infer_config.parallel_config.attn_tp_size
        self.attn_tp_rank = self.model.model.attn_tp_rank
        self.attn_tp_group = self.model.model.attn_tp_group
        self.local_batch = self.batch_size // self.attn_tp_size
        self.eos_ids = self._collect_eos_token_ids()
        self.pad_token_id = int(self.tokenizer.pad_token_id)
        self.enable_profiler = bool(self.model_runner.enable_profiler)
        self._decode_profiler_context = None
        self._decode_profiler = None
        empty_chat_ids = self._apply_chat_template("")
        self.chat_template_overhead = len(empty_chat_ids)
        if self.chat_template_overhead > self.input_max_len:
            raise ValueError(
                "input_max_len is shorter than the Kimi K3 chat template: "
                f"input_max_len={self.input_max_len}, "
                f"template_tokens={self.chat_template_overhead}"
            )

    @staticmethod
    def _normalize_eos_token_ids(eos_token_id):
        if eos_token_id is None:
            return set()
        if isinstance(eos_token_id, int):
            return {eos_token_id}
        if isinstance(eos_token_id, list):
            return {int(token_id) for token_id in eos_token_id}
        raise TypeError(f"unsupported eos_token_id type: {type(eos_token_id)}")

    def _collect_eos_token_ids(self):
        generation_config = getattr(self.model_runner, "hf_generation_config", None)
        return set().union(
            self._normalize_eos_token_ids(
                getattr(self.model.config, "eos_token_id", None)
            ),
            self._normalize_eos_token_ids(
                getattr(generation_config, "eos_token_id", None)
            ),
            self._normalize_eos_token_ids(
                getattr(self.tokenizer, "eos_token_id", None)
            ),
        )

    def _apply_chat_template(self, content):
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        if isinstance(input_ids, torch.Tensor):
            return input_ids.view(-1).tolist()
        if hasattr(input_ids, "input_ids"):
            input_ids = input_ids.input_ids
        return list(input_ids)

    def _encode_prompt(self, prompt):
        # The checkpoint tokenizer warns whenever its specialized encode method
        # receives generic HF kwargs. Encode plainly, truncate the user content,
        # then apply the model's structural chat template.
        content_ids = self.tokenizer.encode(prompt)
        content_budget = self.input_max_len - self.chat_template_overhead
        content_ids = content_ids[:content_budget]
        while True:
            content = self.tokenizer.decode(content_ids)
            input_ids = self._apply_chat_template(content)
            overflow = len(input_ids) - self.input_max_len
            if overflow <= 0:
                if not input_ids:
                    raise ValueError("chat template produced an empty prompt sequence")
                return torch.tensor(input_ids, dtype=torch.long, device=self.device)
            if not content_ids:
                raise ValueError(
                    "Kimi K3 chat template exceeds input_max_len without user content"
                )
            content_ids = content_ids[:-min(overflow, len(content_ids))]

    def sample(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits[:, -1, :]
        if self.temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        probs = torch.softmax(logits / max(self.temperature, 1e-5), dim=-1, dtype=torch.float32)
        return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1, keepdim=True)

    def _tokenize(self, prompts):
        if len(prompts) != self.batch_size:
            if len(prompts) == 1:
                prompts = prompts * self.batch_size
            else:
                raise ValueError(
                    f"offline Kimi K3 requires exactly batch_size={self.batch_size} prompts, "
                    f"got {len(prompts)}"
                )
        token_rows = []
        for prompt in prompts:
            token_rows.append(self._encode_prompt(prompt))
        input_lens = torch.tensor(
            [row.numel() for row in token_rows], dtype=torch.int32, device=self.device
        )
        return torch.cat(token_rows), input_lens, token_rows

    def get_inputs(self, prompts, cache_data=None, draft_cache_data=None):
        input_ids, input_lens, token_rows = self._tokenize(prompts)
        if cache_data is None:
            cache_data = self.cache_manager.init_cache_data()
        if self.uses_dspark and draft_cache_data is None:
            draft_cache_data = self.cache_manager.init_dspark_cache_data(
                self.draft_model.config
            )
        return {
            "input_ids": input_ids,
            "generate_ids": list(token_rows),
            "prompt_token_rows": tuple(token_rows),
            "input_lens": input_lens,
            "kv_len": None,
            "cache_data": cache_data,
            "draft_cache_data": draft_cache_data,
            "is_prefill": True,
        }

    def _model_inputs(self, input_dict, first_verify=False):
        return self.model.prepare_inputs_for_generation(
            input_ids=input_dict["input_ids"],
            input_lens=input_dict["input_lens"],
            kv_len=input_dict["kv_len"],
            cache_data=input_dict["cache_data"],
            is_prefill=input_dict["is_prefill"],
            request_indices=input_dict.get("request_indices"),
            num_accepted_tokens=input_dict.get("num_accepted_tokens"),
            first_verify=first_verify,
            active_mask=input_dict.get("active_mask"),
        )

    def _profile_path(self, phase):
        return os.path.join(self.model_runner.res_path, "prof", phase)

    def _start_decode_profiler(self, warm_up):
        if warm_up or not self.enable_profiler or self._decode_profiler is not None:
            return
        self._decode_profiler_context = self.model_runner.define_profiler(
            enable_profiler=True,
            profile_save_path=self._profile_path("decode"),
            active=10,
            skip_first=10,
        )
        self._decode_profiler = self._decode_profiler_context.__enter__()

    def _close_decode_profiler(self):
        if self._decode_profiler_context is None:
            return
        try:
            # Advance once more so a generation ending on the active-window
            # boundary can trigger the trace handler before the context closes.
            self._decode_profiler.step()
        finally:
            self._decode_profiler_context.__exit__(*sys.exc_info())
            self._decode_profiler_context = None
            self._decode_profiler = None

    def _step_decode_profiler(self):
        if self._decode_profiler is not None:
            self._decode_profiler.step()

    def _run_model(self, model_inputs, is_prefill, warm_up):
        if not is_prefill:
            self._start_decode_profiler(warm_up)
        if dist.is_initialized():
            dist.barrier()
        torch.npu.synchronize()
        start = time.time()
        with torch.no_grad():
            result = (
                self.model.prefill(**model_inputs)
                if is_prefill
                else self.model.decode(**model_inputs)
            )
        if isinstance(result, tuple):
            logits, aux = result
        else:
            logits, aux = result, None
        torch.npu.synchronize()
        elapsed = time.time() - start
        stage = "prefill" if is_prefill else "decode"
        cycle_idx = model_inputs["forward_metadata"].get("prefill_cycle_idx")
        if is_prefill and self.prefill_cycles > 1:
            stage = f"prefill minibatch {cycle_idx}"
        warm_prefix = "[warm up] " if warm_up else ""
        model_prefix = "[Verify] " if self.uses_dspark and not is_prefill else ""
        logging.info(
            "%s%s%s [%s] inference time cost %.2f ms",
            model_prefix,
            self.model_runner.model_name,
            warm_prefix,
            stage,
            elapsed * 1000,
        )
        return logits, aux, elapsed

    def _append_tokens(self, input_dict, next_tokens, request_offset=0):
        for local_idx, token in enumerate(next_tokens):
            request_idx = request_offset + local_idx
            input_dict["generate_ids"][request_idx] = torch.cat(
                (input_dict["generate_ids"][request_idx], token.view(1))
            )

    def _post_process(self, input_dict, logits):
        next_tokens = self.sample(logits)
        self._append_tokens(input_dict, next_tokens)
        input_dict["input_ids"] = next_tokens.view(-1)
        if input_dict["is_prefill"]:
            input_dict["kv_len"] = input_dict["input_lens"].clone()
            input_dict["is_prefill"] = False
        else:
            input_dict["kv_len"] = input_dict["kv_len"] + 1
        return input_dict

    def process_mini_batch_inputs(self, input_dict, cycle_idx):
        """Build one Prefill cycle while retaining the full resident cache.

        Unlike the old DeepSeek implementation, Kimi K3 cannot narrow every
        cache along dimension 0: KDA is replicated by request across TP ranks,
        whereas MLA is request-DP owned.  ``request_indices`` lets the metadata
        builder address both layouts directly in their final Decode storage.
        """
        request_start = cycle_idx * self.mini_batch
        request_end = request_start + self.mini_batch
        token_rows = input_dict["prompt_token_rows"][request_start:request_end]
        return {
            "input_ids": torch.cat(token_rows),
            "input_lens": input_dict["input_lens"][request_start:request_end],
            "kv_len": None,
            "cache_data": input_dict["cache_data"],
            "is_prefill": True,
            "request_indices": torch.arange(
                request_start,
                request_end,
                dtype=torch.long,
                device=self.device,
            ),
            "prefill_cycle_idx": cycle_idx,
        }

    def prefill_infer_single_cycle(self, input_dict, cycle_idx, warm_up=False):
        cycle_input = self.process_mini_batch_inputs(input_dict, cycle_idx)
        model_inputs = self._model_inputs(cycle_input)
        model_inputs["forward_metadata"]["prefill_cycle_idx"] = cycle_idx
        profile_this_cycle = (
            self.enable_profiler
            and not warm_up
            and cycle_idx == self.prefill_cycles // 2
        )
        if profile_this_cycle:
            with self.model_runner.define_profiler(
                enable_profiler=True,
                profile_save_path=self._profile_path("prefill"),
                active=1,
                skip_first=0,
            ) as profiler:
                logits, aux, elapsed = self._run_model(
                    model_inputs, is_prefill=True, warm_up=warm_up
                )
                profiler.step()
        else:
            logits, aux, elapsed = self._run_model(
                model_inputs, is_prefill=True, warm_up=warm_up
            )
        next_tokens = self.sample(logits)
        self._append_tokens(
            input_dict,
            next_tokens,
            request_offset=cycle_idx * self.mini_batch,
        )
        return next_tokens.view(-1), aux, elapsed

    def merge_multi_cycle_res(self, input_dict, cycle_next_tokens):
        """Switch from mini-batch Prefill SP to full-batch request-DP Decode."""
        input_dict["input_ids"] = torch.cat(cycle_next_tokens, dim=0)
        input_dict["kv_len"] = input_dict["input_lens"].clone()
        input_dict["is_prefill"] = False
        return input_dict

    def _log_outputs(self, input_dict):
        fallback_rank = int(os.getenv("LOCAL_RANK", "0")) + int(
            os.getenv("RANK_OFFSET", "0")
        )
        global_rank = int(os.getenv("RANK", os.getenv("RANK_ID", str(fallback_rank))))
        if global_rank != 0:
            return
        prompt_lens = input_dict["input_lens"].tolist()
        for request_idx, token_ids in enumerate(input_dict["generate_ids"]):
            ids = token_ids.tolist()
            prompt_len = int(prompt_lens[request_idx])
            completion = ids[prompt_len:]
            for idx, token_id in enumerate(completion):
                if token_id in self.eos_ids:
                    completion = completion[:idx]
                    break
            logging.info(
                "Inference decode result:\n%s",
                self.tokenizer.decode(completion, skip_special_tokens=False),
            )

    def _log_acceptance_stats(
        self, stats: DSparkAcceptanceStats, average_round_time=None
    ):
        fallback_rank = int(os.getenv("LOCAL_RANK", "0")) + int(
            os.getenv("RANK_OFFSET", "0")
        )
        global_rank = int(os.getenv("RANK", os.getenv("RANK_ID", str(fallback_rank))))
        if global_rank != 0:
            return
        total_spec_tokens = stats.verify_count * self.next_n
        accept_length = (
            stats.total_accepted_tokens / stats.verify_count + 1
            if stats.verify_count
            else 0.0
        )
        accept_rate = (
            stats.total_accepted_tokens / total_spec_tokens
            if total_spec_tokens
            else 0.0
        )
        logging.info("The speculation accept length: %.4f", accept_length)
        logging.info("The speculation accept rate: %.4f", accept_rate)
        if average_round_time is not None and accept_length > 0:
            logging.info(
                "%s model average equivalent latency with actual acceptance "
                "length %.4f is %.2f ms",
                self.model_runner.model_name,
                accept_length,
                average_round_time / accept_length * 1000,
            )
            official_accept_length = 3.85
            logging.info(
                "%s model average equivalent latency with official acceptance "
                "length %.2f is %.2f ms",
                self.model_runner.model_name,
                official_accept_length,
                average_round_time / official_accept_length * 1000,
            )

    def _model_generate_legacy(self, prompts, cache_data=None, warm_up=False):
        input_dict = self.get_inputs(prompts, cache_data)
        cycle_next_tokens = []
        prefill_times = []
        for cycle_idx in range(self.prefill_cycles):
            next_tokens, _, elapsed = self.prefill_infer_single_cycle(
                input_dict, cycle_idx, warm_up=warm_up
            )
            cycle_next_tokens.append(next_tokens)
            prefill_times.append(elapsed)
        self.merge_multi_cycle_res(input_dict, cycle_next_tokens)

        if not warm_up and prefill_times:
            logging.info(
                "%s prefill average inference time cost is %.2f ms over %d cycle(s)",
                self.model_runner.model_name,
                sum(prefill_times) / len(prefill_times) * 1000,
                self.prefill_cycles,
            )

        decode_steps = min(2, self.max_new_tokens - 1) if warm_up else self.max_new_tokens - 1
        decode_times = []
        for _ in range(max(decode_steps, 0)):
            decode_inputs = self._model_inputs(input_dict)
            logits, _, elapsed = self._run_model(decode_inputs, is_prefill=False, warm_up=warm_up)
            decode_times.append(elapsed)
            self._post_process(input_dict, logits)
            self._step_decode_profiler()

        if not warm_up:
            if decode_times:
                logging.info(
                    "%s decode average inference time cost is %.2f ms",
                    self.model_runner.model_name,
                    sum(decode_times) / len(decode_times) * 1000,
                )
            self._log_outputs(input_dict)
        return input_dict

    def _generated_count(self, input_dict, request_idx):
        return int(input_dict["generate_ids"][request_idx].numel()) - int(
            input_dict["input_lens"][request_idx].item()
        )

    def _active_mask(self, input_dict, ignore_eos=False):
        active = []
        for request_idx, row in enumerate(input_dict["generate_ids"]):
            last_token = int(row[-1].item())
            active.append(
                (ignore_eos or last_token not in self.eos_ids)
                and self._generated_count(input_dict, request_idx) < self.max_new_tokens
            )
        return torch.tensor(active, dtype=torch.bool, device=self.device)

    def _route_prefill_target_hidden(
        self, target_hidden_states, request_ids, input_lens
    ):
        context_states = self.draft_model.prepare_target_hidden_states(
            target_hidden_states
        )
        total_tokens = sum(input_lens)
        offsets = [0]
        for length in input_lens:
            offsets.append(offsets[-1] + length)
        owners = sorted({request_id // self.local_batch for request_id in request_ids})
        for owner_rank in owners:
            gathered = gather_sp_shards_to_owner(
                context_states,
                total_tokens,
                owner_rank,
                self.attn_tp_rank,
                self.attn_tp_size,
                self.attn_tp_group,
            )
            if gathered is None:
                continue
            for local_idx, request_id in enumerate(request_ids):
                if request_id // self.local_batch != owner_rank:
                    continue
                input_len = input_lens[local_idx]
                owner_row = request_id % self.local_batch
                prompt_context = gathered[
                    offsets[local_idx] : offsets[local_idx + 1]
                ]
                positions = torch.arange(
                    input_len, dtype=torch.long, device=self.device
                ).view(1, -1)
                self.draft_model.propose(
                    {
                        "is_prefill": True,
                        "target_hidden_positions": positions,
                        "block_table": self.attn_metadata.mla_block_table[
                            owner_row : owner_row + 1
                        ],
                        "slot_block_table": self.attn_metadata.mla_slot_block_table[
                            owner_row : owner_row + 1
                        ],
                        "cache_data": self._dspark_input["draft_cache_data"],
                    },
                    self._dspark_input["input_ids"].new_zeros((1, 1)),
                    prompt_context.unsqueeze(0),
                )

    def _dspark_propose(
        self, input_dict, anchor_tokens, target_hidden, context_positions, warm_up=False
    ):
        owner_start = self.attn_tp_rank * self.local_batch
        owner_end = owner_start + self.local_batch
        if target_hidden.shape[0] != self.local_batch:
            raise RuntimeError("DSpark target hidden must remain owner-local")
        if context_positions.shape[0] != self.local_batch:
            raise RuntimeError("DSpark context positions must remain owner-local")
        if dist.is_initialized():
            dist.barrier()
        torch.npu.synchronize()
        start = time.time()
        with torch.no_grad():
            proposal = self.draft_model.propose(
                {
                    "is_prefill": False,
                    "target_hidden_positions": context_positions,
                    "block_table": self.attn_metadata.mla_block_table,
                    "slot_block_table": self.attn_metadata.mla_slot_block_table,
                    "cache_data": input_dict["draft_cache_data"],
                },
                anchor_tokens[owner_start:owner_end],
                target_hidden,
            )
        torch.npu.synchronize()
        elapsed = time.time() - start
        warm_prefix = "[warm up] " if warm_up else ""
        logging.info(
            "[DSpark] %s %s[decode] inference time cost %.2f ms",
            self.draft_model_runner.model_name,
            warm_prefix,
            elapsed * 1000,
        )
        spec_tokens = proposal["spec_tokens"]
        draft_probs = torch.softmax(
            proposal["logits"].float()
            / (max(self.temperature, 1e-5) if self.temperature > 0 else 1.0),
            dim=-1,
        )
        for request_idx in range(self.batch_size):
            eos_seen = False
            for draft_idx in range(self.next_n):
                if eos_seen:
                    spec_tokens[request_idx, draft_idx] = self.pad_token_id
                    draft_probs[request_idx, draft_idx].zero_()
                elif int(spec_tokens[request_idx, draft_idx].item()) in self.eos_ids:
                    eos_seen = True
        return spec_tokens, draft_probs, elapsed

    def _sample_distribution(self, probs):
        if self.temperature <= 0:
            return int(probs.argmax().item())
        total = probs.sum()
        if float(total.item()) <= 0:
            return int(probs.argmax().item())
        return int(torch.multinomial(probs / total, 1).item())

    def _verify_and_commit(
        self,
        input_dict,
        target_logits,
        spec_tokens,
        draft_probs,
        acceptance_stats=None,
    ):
        target_probs = torch.softmax(
            target_logits.float()
            / (max(self.temperature, 1e-5) if self.temperature > 0 else 1.0),
            dim=-1,
        )
        active = input_dict["active_mask"]
        counts = torch.ones(self.batch_size, dtype=torch.int32, device=self.device)
        anchors = target_logits.new_full(
            (self.batch_size, 1), self.pad_token_id, dtype=torch.long
        )
        for request_idx in range(self.batch_size):
            if not bool(active[request_idx].item()):
                continue
            remaining = self.max_new_tokens - self._generated_count(input_dict, request_idx)
            max_drafts = min(self.next_n, max(remaining - 1, 0))
            committed = []
            accepted_drafts = 0
            stopped = False
            for draft_idx in range(max_drafts):
                token = int(spec_tokens[request_idx, draft_idx].item())
                target = target_probs[request_idx, draft_idx]
                if self.temperature <= 0:
                    accepted = token == int(target.argmax().item())
                else:
                    draft_p = float(draft_probs[request_idx, draft_idx, token].item())
                    target_p = float(target[token].item())
                    ratio = min(1.0, target_p / max(draft_p, 1e-12))
                    accepted = float(torch.rand((), device=self.device).item()) < ratio
                if accepted:
                    committed.append(token)
                    accepted_drafts += 1
                    if token in self.eos_ids:
                        stopped = True
                        break
                    continue
                if self.temperature <= 0:
                    committed.append(int(target.argmax().item()))
                    stopped = True
                    break
                replacement_probs = torch.clamp(
                    target - draft_probs[request_idx, draft_idx], min=0
                )
                if float(replacement_probs.sum().item()) <= 0:
                    replacement_probs = target
                committed.append(self._sample_distribution(replacement_probs))
                stopped = True
                break

            if acceptance_stats is not None:
                acceptance_stats.total_accepted_tokens += accepted_drafts
                acceptance_stats.verify_count += 1
            counts[request_idx] = accepted_drafts + 1
            if not stopped and remaining > len(committed):
                bonus = self._sample_distribution(
                    target_probs[request_idx, accepted_drafts]
                )
                committed.append(bonus)
            if committed:
                token_tensor = torch.tensor(
                    committed, dtype=torch.long, device=self.device
                )
                input_dict["generate_ids"][request_idx] = torch.cat(
                    (input_dict["generate_ids"][request_idx], token_tensor)
                )
                anchors[request_idx, 0] = token_tensor[-1]
            else:
                anchors[request_idx, 0] = input_dict["generate_ids"][request_idx][-1]
        return anchors, counts

    def _committed_context(self, input_dict, target_hidden, counts, old_kv_len):
        local_start = self.attn_tp_rank * self.local_batch
        hidden = target_hidden.view(
            self.local_batch, self.next_n + 1, target_hidden.shape[-1]
        )
        positions = torch.full(
            (self.local_batch, self.next_n + 1),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        for local_idx in range(self.local_batch):
            request_idx = local_start + local_idx
            if not bool(input_dict["active_mask"][request_idx].item()):
                hidden[local_idx].zero_()
                continue
            count = int(counts[request_idx].item())
            positions[local_idx, :count] = old_kv_len[request_idx] + torch.arange(
                count, dtype=torch.long, device=self.device
            )
            if count < self.next_n + 1:
                hidden[local_idx, count:].zero_()
        return hidden, positions

    def _mask_finished_context(self, input_dict, hidden, positions):
        local_start = self.attn_tp_rank * self.local_batch
        local_active = input_dict["active_mask"][
            local_start : local_start + self.local_batch
        ]
        hidden[~local_active] = 0
        positions[~local_active] = -1
        return hidden, positions

    def _model_generate_dspark(
        self, prompts, cache_data=None, draft_cache_data=None, warm_up=False
    ):
        input_dict = self.get_inputs(prompts, cache_data, draft_cache_data)
        acceptance_stats = DSparkAcceptanceStats.create()
        self._dspark_input = input_dict
        cycle_next_tokens = []
        prefill_times = []
        for cycle_idx in range(self.prefill_cycles):
            next_tokens, aux, elapsed = self.prefill_infer_single_cycle(
                input_dict, cycle_idx, warm_up=warm_up
            )
            if aux is None or aux.get("target_hidden_states") is None:
                raise RuntimeError("DSpark requires target hidden states from Main Prefill")
            request_start = cycle_idx * self.mini_batch
            request_ids = list(range(request_start, request_start + self.mini_batch))
            self._route_prefill_target_hidden(
                aux["target_hidden_states"],
                request_ids,
                [int(input_dict["input_lens"][idx].item()) for idx in request_ids],
            )
            cycle_next_tokens.append(next_tokens)
            prefill_times.append(elapsed)
        self.merge_multi_cycle_res(input_dict, cycle_next_tokens)
        input_dict["num_accepted_tokens"] = torch.ones(
            self.batch_size, dtype=torch.int32, device=self.device
        )
        input_dict["active_mask"] = self._active_mask(
            input_dict, ignore_eos=warm_up
        )

        if not bool(input_dict["active_mask"].any().item()):
            if not warm_up:
                self._log_acceptance_stats(acceptance_stats)
                self._log_outputs(input_dict)
            return input_dict

        verify_width = self.next_n + 1
        first_inputs = torch.full(
            (self.batch_size, verify_width),
            self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        first_inputs[:, 0] = input_dict["input_ids"]
        first_inputs[~input_dict["active_mask"]] = self.pad_token_id
        input_dict["input_ids"] = first_inputs
        model_inputs = self._model_inputs(input_dict, first_verify=True)
        logits, aux, elapsed = self._run_model(
            model_inputs, is_prefill=False, warm_up=warm_up
        )
        decode_times = [elapsed]
        sampled_anchors = self.sample(logits[:, :1])
        anchors = sampled_anchors.new_full(
            sampled_anchors.shape, self.pad_token_id
        )
        for request_idx in range(self.batch_size):
            if bool(input_dict["active_mask"][request_idx].item()):
                anchors[request_idx] = sampled_anchors[request_idx]
                self._append_tokens(
                    input_dict, sampled_anchors[request_idx : request_idx + 1], request_idx
                )
        old_kv_len = input_dict["kv_len"].clone()
        input_dict["kv_len"].add_(input_dict["active_mask"].to(torch.int32))
        local_hidden, context_positions = self._committed_context(
            input_dict, aux["target_hidden_states"],
            input_dict["num_accepted_tokens"], old_kv_len,
        )
        input_dict["active_mask"] = self._active_mask(
            input_dict, ignore_eos=warm_up
        )
        local_hidden, context_positions = self._mask_finished_context(
            input_dict, local_hidden, context_positions
        )
        if not bool(input_dict["active_mask"].any().item()):
            self._step_decode_profiler()
            if not warm_up:
                self._log_acceptance_stats(acceptance_stats)
                self._log_outputs(input_dict)
            return input_dict
        spec_tokens, draft_probs, elapsed = self._dspark_propose(
            input_dict, anchors, local_hidden, context_positions, warm_up=warm_up
        )
        self._step_decode_profiler()
        draft_times = [elapsed]
        state = DSparkRuntimeState(
            spec_tokens=spec_tokens,
            draft_probs=draft_probs,
            num_accepted_tokens=input_dict["num_accepted_tokens"],
        )
        max_rounds = 1 if warm_up else self.max_new_tokens
        for _ in range(max_rounds):
            input_dict["active_mask"] = self._active_mask(
                input_dict, ignore_eos=warm_up
            )
            if not bool(input_dict["active_mask"].any().item()):
                break
            verify_inputs = torch.cat((anchors, state.spec_tokens), dim=1)
            verify_inputs[~input_dict["active_mask"]] = self.pad_token_id
            input_dict["input_ids"] = verify_inputs
            input_dict["num_accepted_tokens"] = state.num_accepted_tokens
            old_kv_len = input_dict["kv_len"].clone()
            model_inputs = self._model_inputs(input_dict)
            target_logits, aux, elapsed = self._run_model(
                model_inputs, is_prefill=False, warm_up=warm_up
            )
            decode_times.append(elapsed)
            anchors, counts = self._verify_and_commit(
                input_dict,
                target_logits,
                state.spec_tokens,
                state.draft_probs,
                None if warm_up else acceptance_stats,
            )
            input_dict["kv_len"].add_(
                counts * input_dict["active_mask"].to(torch.int32)
            )
            local_hidden, context_positions = self._committed_context(
                input_dict, aux["target_hidden_states"], counts, old_kv_len
            )
            input_dict["active_mask"] = self._active_mask(
                input_dict, ignore_eos=warm_up
            )
            state.num_accepted_tokens = counts
            if not bool(input_dict["active_mask"].any().item()):
                self._step_decode_profiler()
                break
            local_hidden, context_positions = self._mask_finished_context(
                input_dict, local_hidden, context_positions
            )
            state.spec_tokens, state.draft_probs, elapsed = self._dspark_propose(
                input_dict,
                anchors,
                local_hidden,
                context_positions,
                warm_up=warm_up,
            )
            self._step_decode_profiler()
            draft_times.append(elapsed)

        if not warm_up:
            if prefill_times:
                logging.info(
                    "%s prefill average inference time cost is %.2f ms over %d cycle(s)",
                    self.model_runner.model_name,
                    sum(prefill_times) / len(prefill_times) * 1000,
                    self.prefill_cycles,
                )
            if decode_times:
                logging.info(
                    "[Verify] %s model average inference time cost is %.2f ms",
                    self.model_runner.model_name,
                    sum(decode_times) / len(decode_times) * 1000,
                )
            if draft_times:
                logging.info(
                    "[DSpark] %s model average inference time cost is %.2f ms",
                    self.draft_model_runner.model_name,
                    sum(draft_times) / len(draft_times) * 1000,
                )
            average_round_time = (
                (sum(decode_times) + sum(draft_times)) / len(decode_times)
                if decode_times
                else None
            )
            self._log_acceptance_stats(acceptance_stats, average_round_time)
            self._log_outputs(input_dict)
        return input_dict

    def model_generate(
        self, prompts, cache_data=None, draft_cache_data=None, warm_up=False
    ):
        try:
            if self.uses_dspark:
                return self._model_generate_dspark(
                    prompts, cache_data, draft_cache_data, warm_up
                )
            return self._model_generate_legacy(prompts, cache_data, warm_up)
        finally:
            self._close_decode_profiler()
