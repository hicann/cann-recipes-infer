# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import sys
import time
import math
import argparse
import logging
import copy
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from executor.utils import get_init_attn_mask, process_infer_time, build_dataset_input, remove_padding_left


class Infer(nn.Module):
    def __init__(self, runner_settings, main_model, mtp_model=None):
        super().__init__()
        self.runner_settings = runner_settings
        self.next_n = runner_settings.get("model_config").get("next_n", 0)
        self.use_dataset = runner_settings.get("data_config").get("dataset", "default") != "default"
        self.cp_size = self.runner_settings.get("parallel_config").get("cp_size", 1)
        self.spec_len = self.next_n + 1 # speculative len is one more than num of mtp modules

        self.main_model = main_model
        self.mtp_model = mtp_model

        self.mini_batch = self.main_model.prefill_mini_batch_size \
            if self.main_model.prefill_mini_batch_size > 0 else self.main_model.batch_size_per_rank

        self.tmp_kv = None
        self.tmp_indexer_kv = None

    def inference(self, model_runner, model_inputs, cycle_idx=None, is_prefill=False, warm_up=False, prefix=''):
        torch.npu.synchronize()
        start_time = time.time()
        with torch.no_grad():
            if is_prefill:
                logits, prev_hidden_states = model_runner.model.prefill(**model_inputs)
            else:
                logits, prev_hidden_states = model_runner.model.decode(**model_inputs)

        torch.npu.synchronize()
        inference_time = time.time() - start_time

        enable_profiler = self.main_model.enable_profiler and not warm_up
        prof_prefix_str = "[Prof On]" if enable_profiler else ""
        if not is_prefill:
            mode_prefix_str = "decode"
        else:
            if self.main_model.prefill_mini_batch_size > 0:
                mode_prefix_str = "prefill minibatch " + str(cycle_idx)
            else:
                mode_prefix_str = "prefill"
        if not warm_up:
            logging.info(f"{prefix} {prof_prefix_str} {model_runner.model_name} " + 
                         f"[{mode_prefix_str}] inference time cost {(inference_time)*1000:.2f} ms")
        return logits, prev_hidden_states, inference_time

    def model_input_prepare(self, model, input_dict):
        input_ids = input_dict.get("input_ids")
        attention_mask = input_dict.get("attention_mask")
        past_key_values = input_dict.get("past_key_values")
        past_key_values_indexer = input_dict.get("past_key_values_indexer")

        is_prefill = input_dict.get("is_prefill")
        kv_len = input_dict.get("kv_len")
        share_mask_tril = input_dict.get("share_mask_tril")
        prev_hidden_states = input_dict.get("prev_hidden_states")

        model_inputs = model.prepare_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            past_key_values_indexer=past_key_values_indexer,
            prev_hidden_states=prev_hidden_states,
            is_prefill=is_prefill,
            kv_len=kv_len,
            input_lens=input_dict.get("input_lens"),
            share_mask_tril=share_mask_tril)
        if model.perfect_eplb:
            b, s = model_inputs["input_ids"].shape
            if is_prefill and self.cp_size > 1:
                s = s // self.cp_size
            model_inputs["cur_topk_list"] = model.gen_cur_topk_idx(is_prefill, b, s)

        return model_inputs
    
    def model_output_process(self, model_inputs, outputs, input_dict):
        input_dict['is_prefill'] = False
        input_dict['input_lens'] = input_dict['input_lens'] + 1
        input_dict['kv_len'] = model_inputs['kv_len'] + 1
        past_key_values = model_inputs.get("past_key_values")
        input_dict["past_key_values"] = past_key_values
        past_key_values_indexer = model_inputs.get("past_key_values_indexer")
        input_dict["past_key_values_indexer"] = past_key_values_indexer
        next_tokens = torch.argmax(outputs, dim=-1)
        input_dict['input_ids'] = next_tokens
        input_dict['generate_ids'] = torch.cat([input_dict['generate_ids'], next_tokens], dim=-1)

    def model_inference(self, input_dict, cycle_idx=None, warm_up=False):
        model_inputs = self.model_input_prepare(self.main_model.model, input_dict)
        outputs, _, inference_time = self.inference(self.main_model, model_inputs,
                                                    is_prefill=input_dict['is_prefill'],
                                                    cycle_idx=cycle_idx, warm_up=warm_up)
        self.model_output_process(model_inputs, outputs, input_dict)

        return input_dict, inference_time

    def model_inference_mtp(self, input_dict_main, input_dict_mtp=None,
                            mtp_argdict=None, cycle_idx=None, warm_up=False):
        # main model
        model_inputs = self.model_input_prepare(self.main_model.model, input_dict_main)
        outputs, prev_hidden_states, infer_time_main = self.inference(self.main_model, model_inputs,
                                                                      is_prefill=input_dict_main['is_prefill'],
                                                                      cycle_idx=cycle_idx,
                                                                      warm_up=warm_up, prefix='[MainModel]')
        mtp_argdict['main_next_tokens'] = torch.argmax(outputs, dim=-1)
        mtp_argdict['accepted_num'] = self.verify_spec_tokens(
            input_dict_main, input_dict_mtp, mtp_argdict['main_next_tokens'])
        if input_dict_main['is_prefill']:
            self.update_model_inputs_prefill(model_inputs, input_dict_main, input_dict_mtp,
                                             mtp_argdict['main_next_tokens'], prev_hidden_states)
        else:
            self.update_model_inputs_decode(input_dict_main, input_dict_mtp,
                                            mtp_argdict['main_next_tokens'],
                                            prev_hidden_states, mtp_argdict['accepted_num'])
        mtp_argdict['generate_tokens'] = mtp_argdict['generate_tokens'] + 1 + mtp_argdict['accepted_num']
        mtp_argdict['total_accepted_num'] = mtp_argdict['total_accepted_num'] + mtp_argdict['accepted_num']

        input_dict_mtp, infer_time_mtp = self.mtp_post_process(input_dict_mtp, cycle_idx, warm_up)

        # update inputs for main model to verify in the next round
        mtp_spec_tokens = input_dict_mtp['spec_tokens']
        input_dict_main['input_ids'] = torch.cat([input_dict_main['input_ids'], mtp_spec_tokens], dim=1)

        return input_dict_main, input_dict_mtp, infer_time_main, infer_time_mtp

    def verify_spec_tokens(self, input_dict_main, input_dict_mtp, main_next_tokens):
        '''
        Verify spec tokens with main model's output, stop accepting tokens if rejection occurs in a batch.
        Each batch would process verification seperately.
        '''
        accepted_num = torch.zeros([main_next_tokens.shape[0]], dtype=torch.int64, device="npu") # shape: (Batch,)

        if input_dict_main['is_prefill']:
            return accepted_num
        else: # after main decode
            token_mask = input_dict_mtp['spec_tokens'] == main_next_tokens[:, :self.next_n]
            has_invalid = (token_mask == False).any(dim=-1)
            invalid_pos = (token_mask == False).int().argmax(dim=-1)
            accepted_num = torch.where(has_invalid, invalid_pos, token_mask.shape[-1])
        return accepted_num

    def update_model_inputs_prefill(self, model_inputs, input_dict_main, input_dict_mtp,
                                    main_next_tokens, main_hidden):
        batch_size, _ = main_next_tokens.shape
        input_dict_main['is_prefill'] = False
        cur_kv_len_prefill = model_inputs['kv_len']
        indices = torch.arange(self.spec_len, device="npu")
        kv_len = (cur_kv_len_prefill).unsqueeze(1) + indices # kv_len increase by one after prefill

        # update main inputs for next round
        input_dict_main['input_ids'] = main_next_tokens
        input_dict_main['kv_len'] = kv_len
        input_dict_main['input_lens'] += self.spec_len
        past_key_values = model_inputs.get("past_key_values")
        input_dict_main['past_key_values'] = past_key_values
        past_key_values_indexer = model_inputs.get("past_key_values_indexer")
        input_dict_main['past_key_values_indexer'] = past_key_values_indexer

        input_dict_main['generate_ids'] = torch.cat([input_dict_main['generate_ids'], main_next_tokens], dim=-1)
        input_dict_main['prev_hidden_states'] = torch.chunk(main_hidden, chunks=batch_size, dim=0)

        # update mtp inputs for mtp_prefill, append main model's output to mtp's input with slicing window
        input_dict_mtp['input_ids'] = input_dict_main['generate_ids'][:, 1:]
        input_dict_mtp['prev_hidden_states'] = main_hidden # (B, S, H)
        return input_dict_main, input_dict_mtp

    def update_model_inputs_decode(self, input_dict_main, input_dict_mtp, main_next_tokens,
                                   main_hidden, accepted_num):
        batch_size = main_next_tokens.shape[0]
        accepted_num = accepted_num.view(-1, 1)
        last_step_hidden = input_dict_main['prev_hidden_states']

        # update main inputs for next round
        input_dict_main['kv_len'] = input_dict_main['kv_len'] + 1 + accepted_num

        # each batch accepts different length of tokens, need to pad generate_ids
        generate_ids = []
        for i in range(batch_size):
            cur_len = accepted_num[i] + 1 # total tokens obtained for this batch
            cur_ids = torch.cat([input_dict_main['generate_ids'][i, :],
                                    main_next_tokens[i, :cur_len]], dim=-1).flip(0)
            generate_ids.append(cur_ids)
        generate_ids_pad_left = pad_sequence(generate_ids, batch_first=True,
                                                    padding_value=self.main_model.tokenizer.pad_token_id)
        input_dict_main['generate_ids'] = generate_ids_pad_left.flip(1)
        input_dict_main['input_ids'] = input_dict_main['generate_ids'][:, -1:] # append spec tokens after mtp
        input_dict_main['input_lens'] = input_dict_main['input_lens'] + 1 + accepted_num

        # update mtp inputs for spec infer
        input_dict_mtp['input_ids'] = input_dict_main['generate_ids'][:, -self.spec_len:]
        input_dict_mtp['kv_len'] = input_dict_mtp['kv_len_cached'] + 1 + accepted_num
        input_dict_mtp['input_lens'] = input_dict_mtp['input_lens'] + 1 + accepted_num
        input_dict_mtp['generate_ids'] = input_dict_main['generate_ids']
        input_dict_mtp['spec_tokens'] = None
        input_dict_mtp['kv_len_cached'] = input_dict_mtp['kv_len']

        # process prev hidden states for main and mtp, only keep accepted main hidden
        input_dict_main['prev_hidden_states'] = []
        mtp_prev_hid_tmp = []
        for j in range(batch_size):
            cur_len = accepted_num[j] + 1 # total tokens obtained for this batch
            cur_accepted_hidden = main_hidden[j, :cur_len, :].unsqueeze(0) # B,S,H
            mtp_prev_hid = torch.cat([last_step_hidden[j], cur_accepted_hidden], dim=1)[:, -self.spec_len:, :]
            mtp_prev_hid_tmp.append(mtp_prev_hid)
            input_dict_main['prev_hidden_states'].append(cur_accepted_hidden)
        input_dict_mtp['prev_hidden_states'] = torch.cat(mtp_prev_hid_tmp, dim=0)

        return input_dict_main, input_dict_mtp

    def mtp_post_process(self, input_dict_mtp, cycle_idx, warm_up):
        # mtp model
        for _ in range(self.next_n):
            model_inputs = self.model_input_prepare(self.mtp_model.model, input_dict_mtp)
            outputs, prev_hidden_states, infer_time_spec = self.inference(self.mtp_model, model_inputs,
                                                                          is_prefill=input_dict_mtp['is_prefill'],
                                                                          cycle_idx=cycle_idx,
                                                                          warm_up=warm_up,
                                                                          prefix='[MTP]')
            # mtp model output process
            input_dict_mtp = self.mtp_model_output_process(model_inputs, input_dict_mtp,
                                                            outputs, prev_hidden_states)

        return input_dict_mtp, infer_time_spec

    def mtp_model_output_process(self, model_inputs, input_dict, outputs, prev_hidden_states):
        if input_dict['is_prefill']:
            # kv_len increase by one after prefill
            kv_len_prefill = model_inputs['kv_len']
            indices = torch.arange(self.spec_len, device="npu")
            kv_len = kv_len_prefill.unsqueeze(1) + indices
        else:
            kv_len = input_dict['kv_len'] + 1

        next_tokens = torch.argmax(outputs, dim=-1)
        spec_token = next_tokens[:, -1:]

        # keep record of spec tokens for main model verification
        if input_dict['spec_tokens'] is None:
            input_dict['spec_tokens'] = spec_token
        else:
            input_dict['spec_tokens'] = torch.cat([input_dict['spec_tokens'], spec_token], dim=-1)

        past_key_values = model_inputs.get("past_key_values")
        input_dict["past_key_values"] = past_key_values
        past_key_values_indexer = model_inputs.get("past_key_values_indexer")
        input_dict['past_key_values_indexer'] = past_key_values_indexer
        input_dict['kv_len'] = kv_len
        input_dict['input_lens'] = input_dict['input_lens'] + 1
        input_dict['prev_hidden_states'] = prev_hidden_states[:, -self.spec_len:, :]

        # for next_n > 1, need to pad inputs for the first decode step
        if (next_tokens.shape[1] < self.spec_len) and (self.next_n > 1):
            pad_len = self.spec_len - next_tokens.shape[1]
            next_tokens = torch.cat([input_dict['generate_ids'][:, -pad_len:], next_tokens], dim=-1)
            input_dict['kv_len'] -= pad_len
        input_dict['input_ids'] = next_tokens

        # cache kv_len for the first mtp module
        if input_dict['is_prefill']:
            input_dict['kv_len_cached'] = kv_len
            input_dict['is_prefill'] = False

        return input_dict

    def obtain_mtp_stats(self, total_accepted_num, cnt, infer_time_rec):
        avg_accpeted_num = torch.mean(total_accepted_num)
        logging.info("Finish inference, cnt is %d, average accepted num per batch is %d", cnt, avg_accpeted_num)

        total_tokens = total_accepted_num + cnt
        avg_infer_time = process_infer_time(infer_time_rec, total_tokens[0]) # logging for the first batch
        logging.info(
            f"{self.main_model.model_name} decode average inference time cost is {(avg_infer_time)*1000:.2f} ms")

        return avg_infer_time

    def get_prefill_profile_cycle(self, enable_profiler):
        if not enable_profiler:
            return 1
        prefill_profile_cycle = 10
        if self.main_model.prefill_mini_batch_size == 0:
            return prefill_profile_cycle
        if self.prefill_cycles > prefill_profile_cycle:
            return 1
        return math.ceil(prefill_profile_cycle / self.prefill_cycles)
    
    def init_cache(self, model, inputs, input_dict):
        if input_dict['past_key_values'] is None:
            input_dict['past_key_values'] = model.init_cache(
                inputs.input_ids,
                num_hidden_layers=self.next_n if model.is_mtp else model.config.num_hidden_layers
            )
        
        if input_dict['past_key_values_indexer'] is None:
            input_dict['past_key_values_indexer'] = model.init_cache_for_indexer(
                inputs.input_ids,
                num_hidden_layers=self.next_n if model.is_mtp else model.config.num_hidden_layers
            )

    def get_inputs(self, prompts, past_key_values, past_key_values_indexer,
                   past_key_values_mtp, past_key_values_indexer_mtp, warm_up):
        tokenizer = self.main_model.tokenizer
        kwargs = {
            "return_tensors": "pt", "truncation": True, "padding": "max_length",
            "max_length": self.main_model.input_max_len,
            "add_generation_prompt": True, "return_dict": True
        }
        if self.use_dataset:
            prompts = build_dataset_input(tokenizer, prompts, self.main_model.input_max_len)
        input_prompts = []
        for prompt in prompts:
            input_prompts.append([{"role":"user", "content": prompt}])
        inputs = tokenizer.apply_chat_template(input_prompts, **kwargs).to(self.main_model.device)

        # 2048: fixed shape of mask, used in PFA
        share_mask_tril_main = get_init_attn_mask(2048, self.main_model.device)
        share_mask_tril_mtp = get_init_attn_mask(2048, self.main_model.device)
        input_lens = copy.deepcopy(inputs.input_ids.size()[1])
        if not warm_up:
            logging.info(f"Prompt lens is: {input_lens}")
        input_dict_main = {
            "input_ids": inputs.input_ids, "generate_ids": inputs.input_ids,
            "input_lens": input_lens, "kv_len": None,
            "past_key_values": past_key_values, "past_key_values_indexer": past_key_values_indexer,
            "attention_mask": inputs.attention_mask,
            "share_mask_tril": share_mask_tril_main,
            "prev_hidden_states": None,
            "is_prefill": True,
        }
        self.init_cache(self.main_model.model, inputs, input_dict_main)

        input_dict_mtp = None
        if self.mtp_model is not None:
            mtp_attn_mask = inputs.attention_mask
            input_dict_mtp = {
                "input_ids": inputs.input_ids, "generate_ids": inputs.input_ids,
                "input_lens": input_lens, "kv_len": None,
                "past_key_values": past_key_values_mtp,
                "past_key_values_indexer": past_key_values_indexer_mtp,
                "attention_mask": mtp_attn_mask,
                "share_mask_tril": share_mask_tril_mtp,
                "prev_hidden_states": None,
                "is_prefill": True,
                "spec_tokens": None,
                "kv_len_cached": None
            }
            self.init_cache(self.mtp_model.model, inputs, input_dict_mtp)

        return input_dict_main, input_dict_mtp, inputs, input_lens

    def get_jump_flag(self, cnt, warm_up):
        default_decode_dump = 2
        # warm up only perform for 5 times(decode)
        jump_flag_warm = warm_up and cnt >= default_decode_dump
        # do not generate after max_token
        jump_flag_oversize = cnt >= self.main_model.max_new_tokens
        jump_flag = jump_flag_oversize or jump_flag_warm
        return jump_flag

    def process_mini_batch_inputs(self, input_dict, cycle_idx, is_mtp=False):
        is_batch_required = self.main_model.cp_size == 1 or \
                (self.main_model.global_rank * self.main_model.batch_size_per_rank <= cycle_idx * self.mini_batch \
                < (self.main_model.global_rank + 1) * self.main_model.batch_size_per_rank)
        input_dict_single_cycle = {}
        # inputs should be sliced to mini batch
        for key in ['input_ids', 'generate_ids', 'kv_len', 'attention_mask', 'prev_hidden_states']:
            if input_dict[key] is None:
                input_dict_single_cycle[key] = input_dict[key]
            else:
                input_dict_single_cycle[key] = input_dict[key][cycle_idx * self.mini_batch: \
                                                               (cycle_idx + 1) * self.mini_batch]
        # inputs should be copied
        for key in ['input_lens', 'share_mask_tril']:
            input_dict_single_cycle[key] = input_dict[key]

        input_dict_single_cycle['is_prefill'] = True

        # kv cache should be narrowed
        input_dict_single_cycle['past_key_values'] = ()
        j = cycle_idx % (self.main_model.batch_size_per_rank // self.mini_batch)
        batch_len = self.main_model.model.cache_len * self.mini_batch
        if is_batch_required:
            for past_key_value in input_dict['past_key_values']:
                k, v = past_key_value
                input_dict_single_cycle['past_key_values'] += (
                    (
                        k.narrow(0, j * batch_len, batch_len),
                        v.narrow(0, j * batch_len, batch_len),
                    ),
                )
            input_dict_single_cycle['past_key_values_indexer'] = ()
            for past_key_value in input_dict['past_key_values_indexer']:
                input_dict_single_cycle['past_key_values_indexer'] += (
                    (
                        past_key_value[0].narrow(0, j * batch_len, batch_len),
                    ),
                )
        else:
            k, v = self.tmp_kv[0]
            input_dict_single_cycle['past_key_values'] = \
                tuple([(k.narrow(0, j * batch_len, batch_len), v.narrow(0, j * batch_len, batch_len), ), ] \
                    * self.main_model.model.config.num_hidden_layers)
            input_dict_single_cycle['past_key_values_indexer'] = \
                tuple([(self.tmp_indexer_kv[0][0].narrow(0, j * batch_len, batch_len), )] 
                    * self.main_model.model.config.num_hidden_layers)
        if is_mtp:
            input_dict_single_cycle['spec_tokens'] = input_dict['spec_tokens']
            input_dict_single_cycle['kv_len_cached'] = input_dict['kv_len_cached']

        return input_dict_single_cycle

    def prefill_infer_single_cycle(self, cycle_idx, input_dict_main, input_dict_mtp, mtp_argdict,
                                   infer_time_rec, prof, warm_up):
        
        input_dict_main_single_cycle = self.process_mini_batch_inputs(input_dict_main, cycle_idx)

        input_dict_mtp_single_cycle = None
        mtp_argdict_single_cycle = copy.deepcopy(mtp_argdict)

        if self.mtp_model is None:
            input_dict_main_single_cycle, infer_time_main = self.model_inference(input_dict_main_single_cycle,
                                                                                 cycle_idx, warm_up=warm_up)
            infer_time_rec.append(infer_time_main)
        else:
            input_dict_mtp_single_cycle = self.process_mini_batch_inputs(input_dict_mtp, cycle_idx, True)
            input_dict_main_single_cycle, \
            input_dict_mtp_single_cycle, \
            infer_time_main, \
            infer_time_mtp = self.model_inference_mtp(input_dict_main_single_cycle,
                                                        input_dict_mtp_single_cycle,
                                                        mtp_argdict_single_cycle,
                                                        cycle_idx=cycle_idx,
                                                        warm_up=warm_up)
            infer_time_rec.append(infer_time_main)
            infer_time_rec.append(infer_time_mtp)

        prof.step()
        return input_dict_main_single_cycle, input_dict_mtp_single_cycle, mtp_argdict_single_cycle

    def allocate_cp_prefill_multi_cycle_tmpkv(self, input_ids):
        if self.main_model.cp_size > 1:
            if self.tmp_kv is None:
                self.tmp_kv = self.main_model.model.init_cache(input_ids, num_hidden_layers=1)
            if self.tmp_indexer_kv is None:
                self.tmp_indexer_kv = self.main_model.model.init_cache_for_indexer(input_ids, num_hidden_layers=1)

    def cat_mini_batch_res(self, input_dict_cycles_res):
        input_dict = {}
        input_dict['is_prefill'] = input_dict_cycles_res[0]['is_prefill']
        input_dict['input_lens'] = input_dict_cycles_res[0]['input_lens']
        kv_len_list = []
        input_ids_list = []
        generate_ids_list = []
        for res in input_dict_cycles_res:
            kv_len_list.append(res['kv_len'])
            input_ids_list.append(res['input_ids'])
            generate_ids_list.append(res['generate_ids'])

        input_dict['kv_len'] = torch.cat(kv_len_list, dim=0)
        input_dict['input_ids'] = torch.cat(input_ids_list, dim=0)
        input_dict['generate_ids'] = torch.cat(generate_ids_list, dim=0)

        if self.mtp_model is not None:
            spec_tokens = []
            prev_hidden_states_list = []
            kv_len_cached = []
            for res in input_dict_cycles_res:
                if 'spec_tokens' in res and res['spec_tokens'] is not None:
                    spec_tokens.append(res['spec_tokens'])
                if 'prev_hidden_states' in res and res['prev_hidden_states'] is not None:
                    prev_hidden_states_list.append(res['prev_hidden_states'])
                if 'kv_len_cached' in res and res['kv_len_cached'] is not None:
                    kv_len_cached.append(res['kv_len_cached'])
            if len(spec_tokens) > 0:
                input_dict['spec_tokens'] = torch.cat(spec_tokens, dim=0)
            if len(kv_len_cached) > 0:
                input_dict['kv_len_cached'] = torch.cat(kv_len_cached, dim=0)
            if len(prev_hidden_states_list) > 0:
                if isinstance(prev_hidden_states_list[0], tuple):
                    result = ()
                    for tpl in prev_hidden_states_list:
                        result += tpl
                    input_dict['prev_hidden_states'] = result
                else:
                    input_dict['prev_hidden_states'] = torch.cat(prev_hidden_states_list, dim=0)

        return input_dict

    def merge_multi_cycle_res(self, input_dict_main_cycles_res, input_dict_mtp_cycles_res,
                              mtp_argdict_cycles_res, input_dict_main_ori, input_dict_mtp_ori):
        if len(input_dict_main_cycles_res) == 1:
            return input_dict_main_cycles_res[0], input_dict_mtp_cycles_res[0], mtp_argdict_cycles_res[0]
        input_dict_main = self.cat_mini_batch_res(input_dict_main_cycles_res)
        input_dict_main['past_key_values'] = input_dict_main_ori['past_key_values']
        input_dict_main['past_key_values_indexer'] = input_dict_main_ori['past_key_values_indexer']

        input_dict_mtp, mtp_argdict = None, None
        if self.mtp_model is not None:
            input_dict_mtp = self.cat_mini_batch_res(input_dict_mtp_cycles_res)
            input_dict_mtp['past_key_values'] = input_dict_mtp_ori['past_key_values']
            input_dict_mtp['past_key_values_indexer'] = input_dict_mtp_ori['past_key_values_indexer']

            mtp_argdict = {}
            generate_tokens = []
            total_accepted_num = []
            main_next_tokens = []
            accepted_num = []

            for res in mtp_argdict_cycles_res:
                generate_tokens.append(res['generate_tokens'])
                total_accepted_num.append(res['total_accepted_num'])
                main_next_tokens.append(res['main_next_tokens'])
                accepted_num.append(res['accepted_num'])

            mtp_argdict['generate_tokens'] = torch.cat(generate_tokens, dim=0)
            mtp_argdict['total_accepted_num'] = torch.cat(total_accepted_num, dim=0)
            mtp_argdict['main_next_tokens'] = torch.cat(main_next_tokens, dim=0)
            mtp_argdict['accepted_num'] = torch.cat(accepted_num, dim=0)
        return input_dict_main, input_dict_mtp, mtp_argdict

    def model_generate(self, prompts, past_key_values=None, past_key_values_indexer=None,
                       past_key_values_mtp=None, past_key_values_indexer_mtp=None, warm_up=False):
        # init input_dict and count
        input_dict_main, input_dict_mtp, inputs, input_lens = self.get_inputs(prompts,
                                                                              past_key_values,
                                                                              past_key_values_indexer,
                                                                              past_key_values_mtp,
                                                                              past_key_values_indexer_mtp,
                                                                              warm_up)

        if self.main_model.cp_size > 1:
            self.allocate_cp_prefill_multi_cycle_tmpkv(input_dict_main['input_ids'])

        batch_size, seq_len = inputs.input_ids.shape
        mtp_argdict = {}
        mtp_argdict['generate_tokens'] = torch.zeros([self.mini_batch], device="npu")
        mtp_argdict['total_accepted_num'] = torch.zeros([self.mini_batch], device="npu")

        if self.main_model.prefill_mini_batch_size > 0:
            self.prefill_cycles = batch_size // self.main_model.prefill_mini_batch_size
        else:
            self.prefill_cycles = 1

        generate_tokens = 0
        prefill_cnt = 0
        prefill_infer_time_rec = []

        enable_profiler = self.main_model.enable_profiler and not warm_up
        prefill_prof_cycles = self.get_prefill_profile_cycle(enable_profiler)

        prefill_profiler = self.main_model.define_profiler(
                            enable_profiler=enable_profiler,
                            profile_save_path=os.path.join(self.main_model.res_path, "prefill", "prof"),
                            active=1, skip_first=5, repeat=1)
        with prefill_profiler as prof:
            for _ in range(prefill_prof_cycles):
                # clone because model inference will do postprocess to input_dict after prefill
                input_dict_main_clone = input_dict_main.copy()
                input_dict_mtp_clone = input_dict_mtp.copy() if input_dict_mtp is not None else None
                input_dict_main_cycles_res = []
                input_dict_mtp_cycles_res = []
                mtp_argdict_cycles_res = []
                for cycle_idx in range(self.prefill_cycles):  
                    input_dict_main_single_cycle, \
                    input_dict_mtp_single_cycle, \
                    mtp_argdict_single_cycle = self.prefill_infer_single_cycle(cycle_idx,
                                                                               input_dict_main_clone,
                                                                               input_dict_mtp_clone,
                                                                               mtp_argdict,
                                                                               prefill_infer_time_rec,
                                                                               prof, warm_up)
                    is_batch_required = self.main_model.cp_size == 1 or \
                        (self.main_model.global_rank * self.main_model.batch_size_per_rank
                         <= cycle_idx * self.mini_batch
                        < (self.main_model.global_rank + 1) * self.main_model.batch_size_per_rank)
                    if is_batch_required:
                        input_dict_main_cycles_res.append(input_dict_main_single_cycle)
                        input_dict_mtp_cycles_res.append(input_dict_mtp_single_cycle)
                        mtp_argdict_cycles_res.append(mtp_argdict_single_cycle)
                    prefill_cnt += 1

        if not warm_up:
            avg_infer_time = process_infer_time(prefill_infer_time_rec, prefill_cnt)
            logging.info(f"{self.main_model.model_name} prefill " +
                         f" average inference time cost is {(avg_infer_time)*1000:.2f} ms")

        # merge multi cycle res
        input_dict_main, input_dict_mtp, mtp_argdict = self.merge_multi_cycle_res(input_dict_main_cycles_res,
                                                                                  input_dict_mtp_cycles_res,
                                                                                  mtp_argdict_cycles_res,
                                                                                  input_dict_main,
                                                                                  input_dict_mtp)
        # prefill output process
        generate_tokens += 1

        decode_cnt = 0
        decode_infer_time_rec = []

        # run decode profile
        decode_profiler = self.main_model.define_profiler(enable_profiler=enable_profiler,
                                    profile_save_path=os.path.join(self.main_model.res_path,'decode', 'prof'))
        with decode_profiler as prof:
            while True:
                jump_flag = self.get_jump_flag(decode_cnt, warm_up)
                if jump_flag:
                    break
                if self.mtp_model is None:
                    _, infer_time_main = self.model_inference(input_dict_main, warm_up=warm_up)
                    decode_infer_time_rec.append(infer_time_main)
                else:
                    input_dict_main, \
                    input_dict_mtp, \
                    infer_time_main, \
                    infer_time_mtp = self.model_inference_mtp(input_dict_main, input_dict_mtp,
                                                              mtp_argdict, warm_up=warm_up)

                    decode_infer_time_rec.append(infer_time_main)
                    decode_infer_time_rec.append(infer_time_mtp)
                prof.step()
                generate_tokens += 1
                decode_cnt += 1

        if not warm_up:
            if self.mtp_model is None:
                avg_infer_time = process_infer_time(decode_infer_time_rec, decode_cnt)
                logging.info(f"{self.main_model.model_name} decode " + \
                             f" average inference time cost is {(avg_infer_time)*1000:.2f} ms")
            else:
                avg_infer_time = self.obtain_mtp_stats(mtp_argdict['total_accepted_num'],
                                                       decode_cnt, decode_infer_time_rec)

            # detokenize outputs
            generate_ids_list = remove_padding_left(
                input_dict_main["generate_ids"].clip(0, self.main_model.model.config.vocab_size - 1),
                self.main_model.tokenizer.pad_token_id
                )
            
            for generate_ids in generate_ids_list:
                res = self.main_model.tokenizer.decode(generate_ids[input_lens:], skip_special_tokens=False)
                if self.main_model.tokenizer.eos_token in res:
                    res = res.split(self.main_model.tokenizer.eos_token)[0]
                logging.info("Inference decode result: \n%s", res)

        return input_dict_main, input_dict_mtp
