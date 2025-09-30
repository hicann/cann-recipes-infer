# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging
import copy
from operator import attrgetter
from functools import wraps
import numpy as np
import torch
import torch_npu
from executor.model_runner import ModelRunner
from models.modeling_gpt_oss import GptOssForCausalLM
from models.configuration_gpt_oss import GptOssConfig

root_logger = logging.getLogger()
root_logger.handlers.clear()
logging.basicConfig(format='%(asctime)s - %(levelname)s - [LLM](%(filename)s:%(lineno)d): %(message)s',
                    level=logging.INFO)
logging.getLogger("paramiko").setLevel(logging.ERROR)

torch.manual_seed(42)
torch.npu.manual_seed_all(42)


def override(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def create_init_attn_mask(mask_length, device):
    mask = torch.tril(
        torch.ones((mask_length, mask_length),
                   dtype=torch.int, device=device))
    mask = (1 - mask) * -3.3895e+38
    return mask


def create_sliding_window_attention_mask(mask, window_size, device=None):
    seq_len = mask.shape[-1]
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        if start > 0:
            mask[i, :start] = -3.3895e+38

    return mask


class GptOssRunner(ModelRunner):
    def __init__(self, runner_settings):
        super().__init__(runner_settings)
        self.tp_size = runner_settings.get("parallel_config").get("tp_size", 1)

    def init_model(self):
        self.use_pretrained_model = True
        config = GptOssConfig
        super().init_model(GptOssForCausalLM, config)

    @override
    def _process_weight_after_loading(self):
        self.to_device()
        self.cast_format()

    @override
    def cast_format(self):
        def for_each_to_transpose(weights, parent, layer_idx=""):
            prefix = "transpose"
            for name in weights:
                try:
                    getter = attrgetter(name)
                    tensor = getter(parent)
                except AttributeError:
                    logging.info(f"[WARN] weightName {name} not exist in parent layer.{layer_idx}")
                    continue
                logging.debug("Before %s; size: %s, dtype %s, format %d",
                             f"{prefix} layer.{layer_idx}.{name}", tensor.size(),
                             tensor.dtype, torch_npu.get_npu_format(tensor))

                setattr(tensor, "data", tensor.data.transpose(-2, -1).contiguous())

                logging.debug("After %s; size: %s, dtype %s, format %d",
                             f"{prefix} layer.{layer_idx}.{name}", getter(parent).size(),
                             getter(parent).dtype, torch_npu.get_npu_format(tensor))

        weights = [
            "self_attn.merged_qkv_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.router.weight"
        ]
        for_each_to_transpose(["lm_head.weight"], self.model)
        # cast format for model layers
        for layer_idx, layer in enumerate(self.model.model.layers):
            for_each_to_transpose(weights, layer, layer_idx)

    @override
    def model_input_prepare(self, input_dict):
        input_ids = input_dict.get("input_ids")
        attention_mask = input_dict.get("attention_mask")
        past_key_values = input_dict.get("past_key_values")
        position_ids = input_dict.get("position_ids")
        kv_len = input_dict.get("kv_len")
        cache_position = input_dict.get("cache_position")
        is_prefill = input_dict.get("is_prefill")
        generate_ids = input_dict.get("generate_ids")
        curr_position = input_dict.get("curr_position")
        causal_mask_mapping = input_dict.get("causal_mask_mapping")

        model_inputs = self.model.prepare_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            kv_len=kv_len,
            position_ids=position_ids,
            cache_position=cache_position,
            is_prefill=is_prefill,
            generate_ids=generate_ids,
            curr_position=curr_position,
            causal_mask_mapping=causal_mask_mapping,
            )
        return model_inputs

    @override
    def model_output_process(self, model_inputs, outputs, input_dict):
        logits = outputs
        attention_mask = None
        input_dict['attention_mask'] = attention_mask
        input_dict['past_key_values'] = model_inputs.get("past_key_values")
        kv_len = torch.max(model_inputs.get("position_ids"), axis=1)[0] + 1
        input_dict['kv_len'] = kv_len
        input_dict["is_prefill"] = False
        next_tokens = torch.argmax(logits, dim=-1)[:, -1:]
        if next_tokens.dim() > 2:
            next_tokens = next_tokens.squeeze(-1)
        input_dict['generate_ids'] = torch.cat([input_dict['generate_ids'], next_tokens], dim=-1)
        input_dict['curr_position'] += 1

    @override
    def model_generate(self, prompts, warm_up=False, **kwargs):
        inputs = self.tokenize_prompts(prompts)
        input_lens = copy.deepcopy(inputs.input_ids.size()[1])
        logging.info("Prompt lens is : %d", input_lens)
        profiler = self.define_profiler(enable_profiler=self.enable_profiler, profile_save_path=f"{self.res_path}/prof")

        mask = create_init_attn_mask(
            mask_length=self.runner_settings.get("data_config").get("max_position_embeddings", 4196),
            device=self.device
            )
        mask_4d = mask.unsqueeze(0).unsqueeze(0).expand(inputs.input_ids.shape[0], 1, -1, -1).to(torch.bfloat16)
        sliding_mask = create_sliding_window_attention_mask(mask, self.model.config.sliding_window, self.device)
        sliding_mask_4d = sliding_mask.unsqueeze(0).unsqueeze(0).expand(inputs.input_ids.shape[0], \
            1, -1, -1).to(torch.bfloat16)
        causal_mask_mapping = {
                "full_attention": mask_4d,
                "sliding_attention": sliding_mask_4d,
            }
        input_dict = {
            "input_ids": inputs.input_ids, "generate_ids": inputs.input_ids.clone(),
            "past_key_values": None, "attention_mask": inputs.attention_mask,
            "position_ids": None,
            "kv_len": None,
            "cache_position": None,
            "is_prefill": True,
            "curr_position": input_lens - 1,
            "causal_mask_mapping": causal_mask_mapping,
        }

        generate_tokens = 0
        cnt = 0
        with profiler as prof:
            while True:
                jump_flag = self.get_jump_flag(cnt, warm_up, generate_tokens)
                if jump_flag:
                    break

                model_inputs = self.model_input_prepare(input_dict)
                outputs = self.model_inference(model_inputs, warm_up=warm_up)
                self.model_output_process(model_inputs, outputs, input_dict)
                prof.step()
                generate_tokens += 1
                cnt += 1

        generate_ids = input_dict["generate_ids"][0:1, input_lens:].clip(0, self.model.config.vocab_size - 1)
        return self.tokenizer_decode(generate_ids)

    def get_jump_flag(self, cnt, warm_up, generate_tokens):
        default_decode_dump = 2
        # warm up only perform for 5 times(decode)
        jump_flag_warm = warm_up and cnt >= default_decode_dump
        # do not generate after max_token
        jump_flag_oversize = generate_tokens >= self.max_new_tokens
        jump_flag = jump_flag_oversize or jump_flag_warm
        return jump_flag
