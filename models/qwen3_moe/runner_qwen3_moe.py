# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import time
import argparse
import logging
import copy
from functools import wraps
from operator import attrgetter
import numpy as np
import torch
import torch_npu
from executor.model_runner import ModelRunner
from models.modeling_qwen3_moe import Qwen3MoeForCausalLM
from module.quantization import QuantizeMethodBase
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


def get_init_attn_mask(mask_length, device, valid_len=None):
    share_mask_tril = ~torch.tril(
        torch.ones((mask_length, mask_length),
                   dtype=torch.bool, device=device))
    if valid_len is not None:
        share_mask_tril[-valid_len:, :] = torch.zeros(valid_len, mask_length)
    return share_mask_tril


def get_decode_mask(mask_length, device, position):
    decode_mask = torch.zeros((1, mask_length), device=device)
    decode_mask[0, :position] = 1
    return decode_mask


class Qwen3MoeRunner(ModelRunner):
    def __init__(self, runner_settings):
        super().__init__(runner_settings)
        self.with_ckpt = runner_settings.get("model_config").get("with_ckpt", True)
        self.attn_dp_size = runner_settings.get("parallel_config").get("attn_dp_size", 1)
        self.enable_cache_compile = runner_settings.get("model_config").get("enable_cache_compile", False)

    @staticmethod
    def repeat_batch(tensor, repeat_num):
        if repeat_num == 1:
            return tensor
        return tensor.repeat(repeat_num, *[1] * (tensor.dim() - 1))

    def init_model(self):
        if self.with_ckpt:
            self.use_pretrained_model = True
            config = None
        else:
            self.use_pretrained_model = False
        from models.configuration_qwen3_moe import Qwen3MoeConfig as config
        super().init_model(Qwen3MoeForCausalLM, config)

    @override
    def _process_weight_after_loading(self):
        '''
        Doing weight transpose, format cast to nz after loading weights from files.
        '''
        self.to_device()

        for module_name, module in self.model.named_modules():
            quant_method = getattr(module, "quant_method", None)

            is_nz = False if ("mlp.gate" in module_name and "proj" not in module_name) else True
            if isinstance(quant_method, QuantizeMethodBase):
                quant_method.process_weights_after_loading(module, is_nz=is_nz)

    @override
    def graph_compile(self):
        import torchair as tng
        tng.patch_for_hcom()
        from torchair.configs.compiler_config import CompilerConfig

        compiler_config = CompilerConfig()
        compiler_config.experimental_config.frozen_parameter = True
        compiler_config.experimental_config.tiling_schedule_optimize = True
        compiler_config.experimental_config.topology_sorting_strategy = "StableRDFS"
        if self.enable_cache_compile:
            case_name = "compile_cache/" + os.getenv("CASE_NAME")
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), case_name)
            self.model.decode = tng.inference.cache_compile(self.model.decode, cache_dir=cache_dir,
                                config=compiler_config, dynamic=True, fullgraph=True, ge_cache=True)
        else:
            npu_backend = tng.get_npu_backend(compiler_config=compiler_config)
            self.model.decode = torch.compile(self.model.decode, dynamic=True, fullgraph=True, backend=npu_backend)

    @override
    def model_inference(self, model_inputs, is_prefill=False, warm_up=False):
        torch.npu.synchronize()
        if warm_up and self.execute_mode == "ge_graph":
            self.mark_inputs(model_inputs)
        start_time = time.time()
        with torch.no_grad():
            if is_prefill:
                logits = self.model.prefill(**model_inputs)
            else:
                logits = self.model.decode(**model_inputs)

        torch.npu.synchronize()
        end_time = time.time()
        inference_time = end_time - start_time
        inference_stage = "prefill" if is_prefill else "decode"
        logging.info(f"{self.model_name} inference time cost of {inference_stage} is {(inference_time)*1000:.2f} ms")
        return logits

    def mark_detail(self, model_inputs, item_key, is_cache=False):
        from torch._dynamo import mark_static
        item = model_inputs.get(item_key, None)
        if item is None:
            return
        if is_cache:
            for item_sub in item:
                for sub_item in item_sub:
                    mark_static(sub_item)
        else:
            mark_static(item)

    @override
    def mark_inputs(self, model_inputs):
        self.mark_detail(model_inputs, "input_ids")
        self.mark_detail(model_inputs, "kv_len")
        self.mark_detail(model_inputs, "past_key_values", is_cache=True)
        self.mark_detail(model_inputs, "position_ids")
        self.mark_detail(model_inputs, "attention_mask")
        self.mark_detail(model_inputs, "cur_topk_list")

    @override
    def model_input_prepare(self, input_dict):
        input_ids = input_dict.get("input_ids")
        attention_mask = input_dict.get("attention_mask")
        past_key_values = input_dict.get("past_key_values")
        is_prefill = input_dict.get("is_prefill")
        kv_len = input_dict.get("kv_len")
        share_mask_tril = input_dict.get("share_mask_tril")
        model_inputs = self.model.prepare_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            is_prefill=is_prefill,
            kv_len=kv_len,
            input_lens=input_dict.get("input_lens"),
            share_mask_tril=share_mask_tril
            )
        if self.model.perfect_eplb:
            input_ids_update = model_inputs.get("input_ids")
            model_inputs["cur_topk_list"] = \
                self.model.gen_cur_topk_idx(is_prefill, input_ids_update.shape[0], input_ids_update.shape[1])

        return model_inputs

    @override
    def model_output_process(self, model_inputs, outputs, input_dict):
        next_batch = self.batch_size if input_dict["is_prefill"] else 1
        next_batch_dp = next_batch // self.attn_dp_size if input_dict["is_prefill"] else 1
        input_dict['is_prefill'] = False
        input_dict['input_lens'] = input_dict['input_lens'] + 1

        kv_len = torch.max(model_inputs.get("position_ids"), axis=1)[0] + 1
        input_dict['kv_len'] = kv_len

        logits = outputs
        past_key_values = model_inputs.get("past_key_values")
        past_key_values_batch = ()
        for past_key_values_layer_i in past_key_values:
            cache_new_i = ()
            for cache_j in past_key_values_layer_i:
                cache_j_new = cache_j
                cache_new_i += (cache_j_new, )
            past_key_values_batch += (cache_new_i, )
        input_dict["past_key_values"] = past_key_values_batch

        attention_mask = None

        share_mask_tril = get_decode_mask(mask_length=self.max_position_embeddings,
                                            device=self.device,
                                            position=input_dict["input_lens"])
        share_mask_tril = share_mask_tril[None, None, ...]

        input_dict['attention_mask'] = attention_mask
        input_dict['share_mask_tril'] = share_mask_tril

        next_tokens = torch.argmax(logits, dim=-1)[:, -1:]
        input_dict['input_ids'] = next_tokens
        input_dict['generate_ids'] = torch.cat([input_dict['generate_ids'], next_tokens], dim=-1)

    @override
    def model_generate(self, prompts, warm_up=False):
        inputs = self.tokenize_prompts(prompts)
        profiler = self.define_profiler(
            enable_profiler=self.enable_profiler and not warm_up,
            profile_save_path=f"{self.res_path}/prof_{self.model_name}",
        )

        # get init input_dict
        share_mask_tril = get_init_attn_mask(2048, self.device)

        input_lens = copy.deepcopy(inputs.input_ids.size()[1])
        logging.info("Prompt lens is : %d", input_lens)
        input_dict = {
            "input_ids": inputs.input_ids, "generate_ids": inputs.input_ids,
            "input_lens": input_lens, "kv_len": None,
            "past_key_values": None, "attention_mask": inputs.attention_mask, "share_mask_tril": share_mask_tril,
            "is_prefill": True,
        }

        generate_tokens = 0
        cnt = 0
        with profiler as prof:
            while True:
                jump_flag = self.get_jump_flag(cnt, warm_up, generate_tokens)
                if jump_flag:
                    break

                model_inputs = self.model_input_prepare(input_dict)
                outputs = self.model_inference(model_inputs, is_prefill=input_dict["is_prefill"], warm_up=warm_up)
                self.model_output_process(model_inputs, outputs, input_dict)
                generate_tokens += 1
                cnt += 1
                prof.step()

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
