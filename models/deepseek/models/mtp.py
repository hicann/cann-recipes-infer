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
import argparse
import logging
import copy
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from executor.utils import get_init_attn_mask, process_infer_time


class InferMTP(nn.Module):
    def __init__(self, runner_settings, main_model, mtp_model):
        super().__init__()
        self.runner_settings = runner_settings
        self.enable_pa = runner_settings.get("model_config").get("enable_pa", False)
        self.next_n = runner_settings.get("model_config").get("next_n", "0")
        self.spec_len = self.next_n + 1 # speculative len is one more than num of mtp modules

        self.main_model = main_model
        self.mtp_model = mtp_model

    def model_generate_mtp(self, prompts, warm_up=False):
        # init input_dict and count
        input_dict_main, input_dict_mtp, inputs, input_lens = self.get_mtp_inputs(prompts)
        batch_size, _ = inputs.input_ids.shape
        cnt = 0
        infer_time_rec = []
        generate_tokens = torch.zeros([batch_size], device="npu")
        total_accepted_num = torch.zeros([batch_size], device="npu")

        profiler = self.main_model.define_profiler(
            enable_profiler=self.main_model.enable_profiler and not warm_up,
            profile_save_path=f"{self.main_model.res_path}/prof",
        )
        with profiler as prof:
            while True:
                jump_flag = self.main_model.get_jump_flag(cnt, warm_up)
                if jump_flag:
                    break
                step_time = 0
                # main model
                model_inputs = self.main_model.model_input_prepare(input_dict_main)
                outputs = self.main_model.model_inference(model_inputs,
                                                            is_prefill=input_dict_main['is_prefill'], warm_up=warm_up)
                # The outputs is a tuple containing logits, inference_time and prev_hidden_states.
                logits = outputs[0]
                infer_time_main = outputs[1]
                prev_hidden_states = outputs[2]
                step_time += infer_time_main
                main_next_tokens = torch.argmax(logits, dim=-1)
                accepted_num = self.verify_spec_tokens(input_dict_main, input_dict_mtp, main_next_tokens)
                if input_dict_main['is_prefill']:
                    self.update_model_inputs_prefill(model_inputs, input_dict_main, input_dict_mtp,
                                                     main_next_tokens, prev_hidden_states)
                else:
                    self.update_model_inputs_decode(input_dict_main, input_dict_mtp,
                                                    main_next_tokens, prev_hidden_states, accepted_num)

                generate_tokens = generate_tokens + 1 + accepted_num
                total_accepted_num = total_accepted_num + accepted_num

                # mtp model
                for _ in range(self.next_n):
                    model_inputs = self.mtp_model.model_input_prepare(input_dict_mtp)
                    outputs = self.mtp_model.model_inference(model_inputs,
                                                            is_prefill=input_dict_mtp['is_prefill'], warm_up=warm_up)
                    # The outputs is a tuple containing logits, inference_time and prev_hidden_states.
                    logits = outputs[0]
                    infer_time_spec = outputs[1]
                    prev_hidden_states = outputs[2]
                    # mtp model output process
                    input_dict_mtp = self.mtp_model_output_process(model_inputs, input_dict_mtp,
                                                                    logits, prev_hidden_states)
                    step_time += infer_time_spec

                # update inputs for main model to verify in the next round
                mtp_spec_tokens = input_dict_mtp['spec_tokens']
                input_dict_main['input_ids'] = torch.cat([input_dict_main['input_ids'], mtp_spec_tokens], dim=1)
                prof.step()
                cnt += 1
                infer_time_rec.append(step_time)

        if not warm_up:
            avg_infer_time = self.obtain_mtp_stats(total_accepted_num, cnt, infer_time_rec)

        # to modify to print for each batch
        logging.info("Finish inference, cnt = %d, total_accepted_num = %d", cnt, total_accepted_num[0])
        generate_ids = input_dict_main["generate_ids"][:, input_lens:].clip(0,\
                            self.main_model.model.config.vocab_size - 1)
        return self.main_model.tokenizer_decode(generate_ids)

    def get_mtp_inputs(self, prompts):
        inputs = self.main_model.tokenize_prompts(prompts)
        # 2048: fixed shape of mask, used in PFA
        share_mask_tril_main = get_init_attn_mask(2048, self.main_model.device)
        share_mask_tril_mtp = get_init_attn_mask(2048, self.mtp_model.device)
        input_lens = copy.deepcopy(inputs.input_ids.size()[1])
        logging.info(f"Prompt lens is: {input_lens}")
        input_dict_main = {
            "input_ids": inputs.input_ids, "generate_ids": inputs.input_ids,
            "input_lens": input_lens, "kv_len": None,
            "past_key_values": self.main_model.past_key_values,
            "attention_mask": inputs.attention_mask,
            "share_mask_tril": share_mask_tril_main,
            "prev_hidden_states": None,
            "is_prefill": True,
        }
        mtp_attn_mask = inputs.attention_mask
        input_dict_mtp = {
            "input_ids": inputs.input_ids, "generate_ids": inputs.input_ids,
            "input_lens": input_lens, "kv_len": None,
            "past_key_values": self.mtp_model.past_key_values,
            "attention_mask": mtp_attn_mask,
            "share_mask_tril": share_mask_tril_mtp,
            "prev_hidden_states": None,
            "is_prefill": True,
            "spec_tokens": None,
            "kv_len_cached": None
        }
        return input_dict_main, input_dict_mtp, inputs, input_lens


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


    def update_model_inputs_prefill(self, model_inputs, input_dict_main, input_dict_mtp, main_next_tokens, main_hidden):
        batch_size, _ = main_next_tokens.shape
        input_dict_main['is_prefill'] = False
        cur_kv_len_prefill = torch.max(model_inputs.get('position_ids'), axis=1)[0]
        indices = torch.arange(self.spec_len, device="npu")
        kv_len = (cur_kv_len_prefill + 1).unsqueeze(1) + indices # kv_len increase by one after prefill

        # update main inputs for next round
        input_dict_main['input_ids'] = main_next_tokens
        input_dict_main['kv_len'] = kv_len
        input_dict_main['input_lens'] += self.spec_len
        past_key_values = model_inputs.get("past_key_values")
        input_dict_main['past_key_values'] = past_key_values

        input_dict_main['generate_ids'] = torch.cat([input_dict_main['generate_ids'], main_next_tokens], dim=-1)
        seq_len_list = tuple((cur_kv_len_prefill + 1).tolist())
        main_hidden = main_hidden[:, :sum(seq_len_list), :]
        input_dict_main['prev_hidden_states'] = torch.split(main_hidden, seq_len_list, dim=1)

        # update mtp inputs for mtp_prefill, append main model's output to mtp's input with slicing window
        input_dict_mtp['input_ids'] = input_dict_main['generate_ids'][:, 1:]
        input_dict_mtp['prev_hidden_states'] = main_hidden # (B, S, H)
        return input_dict_main, input_dict_mtp


    def update_model_inputs_decode(self, input_dict_main, input_dict_mtp, main_next_tokens, main_hidden, accepted_num):
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


    def mtp_model_output_process(self, model_inputs, input_dict, outputs, prev_hidden_states):
        if input_dict['is_prefill']:
            # kv_len increase by one after prefill
            kv_len_prefill = torch.max(model_inputs.get('position_ids'), axis=1)[0] + 1
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
        logging.info(f"{self.main_model.model_name} average inference time cost is {(avg_infer_time)*1000:.2f} ms")

        return avg_infer_time
