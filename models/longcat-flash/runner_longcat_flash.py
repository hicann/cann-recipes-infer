# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

import os
import time
import logging
import copy
import gc
from operator import attrgetter
from functools import wraps
import numpy as np
import torch
import torch.distributed as dist
import torch_npu
import torch.nn as nn
from executor.model_runner import ModelRunner
from models.model_setting import check_is_attn_rank
from models.modeling_longcat_flash import LongcatFlashForCausalLM, LongcatFlashModelMTP
from models.ffn import FFNForCausalLM
from models.configuration_longcat_flash import LongcatFlashConfig
from module.quantization import QuantizeMethodBase
from module.quantization.compressed_tensors.compressed_tensors_moe_gmm import CompressedTensorW8A8Int8MoEGMMMethod

from executor.utils import (
    override, get_init_attn_mask, process_infer_time)

root_logger = logging.getLogger()
root_logger.handlers.clear()
logging.basicConfig(format='%(asctime)s - %(levelname)s - [LLM](%(filename)s:%(lineno)d): %(message)s',
                    level=logging.INFO)
logging.getLogger("paramiko").setLevel(logging.ERROR)

torch.manual_seed(42)
torch.npu.manual_seed_all(42)


def create_init_attn_mask(mask_length, device):
    mask = torch.tril(
        torch.ones((mask_length, mask_length),
                   dtype=torch.int, device=device))
    mask = (1 - mask) * -3.3895e+38
    return mask


class LongcatFlashRunner(ModelRunner):
    def __init__(self, runner_settings):
        super().__init__(runner_settings)
        self.tp_size = runner_settings.get("parallel_config").get("tp_size", 1) # dense tp
        self.ep_size = runner_settings.get("parallel_config").get("ep_size", 1)
        self.batch_size = runner_settings.get("data_config").get("batch_size")
        self.with_ckpt = runner_settings.get("model_config").get("with_ckpt", True)
        self.enable_multi_stream = runner_settings.get("model_config").get("enable_multi_stream", 0)
        self.enable_cache_compile = runner_settings.get("model_config").get("enable_cache_compile", False)
        self.batch_size_per_rank = runner_settings.get("data_config").get("batch_size_per_rank")
        self.decode_only = runner_settings.get("model_config").get("decode_only", False)
        self.input_max_len = runner_settings.get("data_config").get("input_max_len")
        self.enable_afd = runner_settings.get("model_config").get("enable_afd", False)
        self.next_n = runner_settings.get("model_config").get("next_n", "0")
        self.spec_len = self.next_n + 1 # speculative len is one more than num of mtp modules
        self.is_attn_rank = check_is_attn_rank(runner_settings)

    @override
    def init_model(self, is_mtp=False):
        self.is_mtp = is_mtp
        if self.with_ckpt:
            self.use_pretrained_model = True
            config = None
        else:
            self.use_pretrained_model = False
        logging.info(f"use_pretrained_model: {self.use_pretrained_model}")
        if self.is_attn_rank:
            if is_mtp:
                super().init_model(LongcatFlashModelMTP, LongcatFlashConfig)
            else:
                super().init_model(LongcatFlashForCausalLM, LongcatFlashConfig)
        else:
            # Given that the MTP module in the Longcat model is a Dense MLP,
            # therefore, in the AFD scenario there is no MTP module on the FFN Rank.
            super().init_model(FFNForCausalLM, LongcatFlashConfig)

    @override
    def graph_compile(self):
        import torchair as tng
        tng.patch_for_hcom()
        from torchair.configs.compiler_config import CompilerConfig
        torch._dynamo.config.inline_inbuilt_nn_modules = False
        compiler_config = CompilerConfig()
        compiler_config.experimental_config.frozen_parameter = True
        compiler_config.experimental_config.tiling_schedule_optimize = True
        compiler_config.experimental_config.topology_sorting_strategy = "StableRDFS"
        compiler_config.ge_config.enable_single_stream = False if self.enable_multi_stream else True

        npu_backend = tng.get_npu_backend(compiler_config=compiler_config)
        if self.enable_cache_compile:
            case_name = "compile_cache/" + os.getenv("CASE_NAME")
            cache_model = self.model.decode
            if self.is_mtp:
                case_name += "_spec"
                cache_model = self.model.mtp_compile_decode
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), case_name)
            self.model.decode = tng.inference.cache_compile(
                cache_model,
                cache_dir=cache_dir,
                config=compiler_config,
                dynamic=False,
                fullgraph=True,
                ge_cache=True
            )
        else:
            self.model.decode = torch.compile(self.model.decode, dynamic=False, fullgraph=True, backend=npu_backend)

    @override
    def init_splited_kv_b_weight(self):
        def for_each_to_init_splited_k_b_weight(layer):
            try:
                if hasattr(self.model.model, 'mtp'):
                    data_getter = attrgetter(f"kv_b_proj_w_k_data")
                    data_tensor = data_getter(layer.self_attn)
                    layer.self_attn.kv_b_proj_w_k = nn.Parameter(data_tensor.contiguous(), requires_grad=False)
                else:
                    for attn in layer.self_attn:
                        data_getter = attrgetter(f"kv_b_proj_w_k_data")
                        data_tensor = data_getter(attn)
                        attn.kv_b_proj_w_k = nn.Parameter(data_tensor.contiguous(), requires_grad=False)
            except AttributeError:
                pass

        def for_each_to_init_splited_v_b_weight(layer):
            try:
                if hasattr(self.model.model, 'mtp'):
                    data_getter = attrgetter(f"kv_b_proj_w_v_data")
                    data_tensor = data_getter(layer.self_attn)
                    layer.self_attn.kv_b_proj_w_v = nn.Parameter(data_tensor.contiguous(), requires_grad=False)
                else:
                    for attn in layer.self_attn:
                        data_getter = attrgetter(f"kv_b_proj_w_v_data")
                        data_tensor = data_getter(attn)
                        attn.kv_b_proj_w_v = nn.Parameter(data_tensor.contiguous(), requires_grad=False)
            except AttributeError:
                pass

        def for_each_to_offload_kv_b_weight(layer):
            try:
                if not hasattr(self.model.model, 'mtp'):
                    for attn in layer.self_attn:
                        attn.kv_b_proj.weight = None
                else:
                    layer.self_attn.kv_b_proj.weight = None
            except AttributeError:
                pass

        if self.is_mtp:
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'mtp'):
                for layer in self.model.model.mtp.layers:
                    for_each_to_init_splited_k_b_weight(layer)
                    for_each_to_init_splited_v_b_weight(layer)
                    for_each_to_offload_kv_b_weight(layer)
            else:
                logging.info("INFO: MTP layers not found. Skipping")
        else:
            if self.is_attn_rank:
                for layer in self.model.model.layers:
                    for_each_to_init_splited_k_b_weight(layer)
                    for_each_to_init_splited_v_b_weight(layer)
                    for_each_to_offload_kv_b_weight(layer)
        gc.collect()

    @override
    def _process_weight_after_loading(self):
        self.init_splited_kv_b_weight()
        self.to_device()
        # map for scales need to cast to float when apply w8a8 quant method
        float_scales_map = [
            "gate_up_proj",
        ]
        # map for smooth scales need to cast to float when apply w8a8 quant method
        float_smooth_scales_map = [
            "down_proj"
        ]
        if self.enable_online_split_weight:
            for module_name, module in self.model.named_modules():
                if "kv_b_proj" in module_name:
                    continue
                quant_method = getattr(module, "quant_method", None)
                scales_dtype = {}
                for scale_name in float_scales_map:
                    # if scale in module need type cast, add target dtype to dict
                    if scale_name in module_name:
                        scales_dtype['scale_dtype'] = torch.float
                        break
                for smooth_scale_name in float_smooth_scales_map:
                    # if smooth scale in module need type cast, add target dtype to dict
                    if smooth_scale_name in module_name:
                        scales_dtype['smooth_scale_dtype'] = torch.float
                        break

                is_nz = False if ("classifier" in module_name) else True
                is_transpose = False if ("classifier" in module_name) else True
                if isinstance(quant_method, QuantizeMethodBase):
                    quant_method.process_weights_after_loading(module, is_nz=is_nz, is_transpose=is_transpose, \
                                                               scales_dtype=scales_dtype)
                # Dynamic quant for input_avtivation of first grouped matmul requies complete smooth scale.
                # When applying expert parallel, each device only reserves smooth scales of mapping experts.
                # Need to do all gather to obtain complete smooth scale.
                if isinstance(quant_method, CompressedTensorW8A8Int8MoEGMMMethod):
                    moe_ep_size = self.runner_settings.get("parallel_config").get("moe_ep_size", 1)
                    if moe_ep_size > 1:
                        all_experts_smooth_scale = module.smooth_scale_1.data.new_empty(
                            module.smooth_scale_1.data.shape[0] * moe_ep_size, module.smooth_scale_1.data.shape[1])
                        dist.all_gather_into_tensor(all_experts_smooth_scale, module.smooth_scale_1.data,
                                                    group=self.model.hccl_comm_dict.get("moe_ep_group", None))
                        module.smooth_scale_1.data = all_experts_smooth_scale
        else:
            self.scale_dtype_adapter()
            self.cast_format()

    @override
    def model_input_prepare(self, input_dict):
        input_ids = input_dict.get("input_ids")
        attention_mask = input_dict.get("attention_mask")
        past_key_values = input_dict.get("past_key_values")
        if past_key_values is None:
            self.past_key_values = self.model.init_cache(input_ids)
            input_dict["past_key_values"] = self.past_key_values
        prev_hidden_states = input_dict.get("prev_hidden_states")
        is_prefill = input_dict.get("is_prefill")
        kv_len = input_dict.get("kv_len")
        share_mask_tril = input_dict.get("share_mask_tril")

        model_inputs = self.model.prepare_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=self.past_key_values,
            prev_hidden_states=prev_hidden_states,
            is_prefill=is_prefill,
            kv_len=kv_len,
            input_lens=input_dict.get("input_lens"),
            share_mask_tril=share_mask_tril,
        )
        if self.model.perfect_eplb:
            input_ids_update = model_inputs.get("input_ids")
            model_inputs["cur_topk_list"] = \
                self.model.gen_cur_topk_idx(is_prefill, input_ids_update.shape[0], input_ids_update.shape[1])
        return model_inputs

    def ffn_input_prepare(self, is_prefill):
        seq_len = self.input_max_len if is_prefill else self.spec_len
        dumy_hid_shape = [self.batch_size_per_rank, seq_len, self.model.config.hidden_size]
        dumy_hid = torch.ones(dumy_hid_shape, dtype=torch.bfloat16).npu()
        model_inputs = dict()
        model_inputs["hidden_states"] = dumy_hid

        if self.model.perfect_eplb:
            model_inputs["cur_topk_list"] = \
                self.model.gen_cur_topk_idx(is_prefill, self.batch_size_per_rank, seq_len)
        return model_inputs

    @override
    def model_inference(self, model_inputs, is_prefill=False, warm_up=False):
        dist.barrier() if not self.enable_afd else None
        torch.npu.synchronize()
        if warm_up and self.execute_mode == "ge_graph":
            self.mark_inputs(model_inputs)
        start_time = time.time()
        with torch.no_grad():
            if is_prefill:
                logits, prev_hidden_states = self.model.prefill(**model_inputs)
            else:
                logits, prev_hidden_states = self.model.decode(**model_inputs)

        torch.npu.synchronize()
        end_time = time.time()
        inference_time = end_time - start_time
        inference_stage = "prefill" if is_prefill else "decode"
        logging.info(f"{self.model_name} inference time cost of {inference_stage} is {(inference_time)*1000:.2f} ms")
        return (logits, inference_time, prev_hidden_states)

    def ffn_inference(self, model_inputs, is_prefill=False, warm_up=False):
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
        logging.info(
            f"{self.model_name} ffn inference time cost of {inference_stage} is {(inference_time) * 1000:.2f} ms")
        return logits, inference_time

    @override
    def model_output_process(self, model_inputs, outputs, input_dict):
        input_dict['is_prefill'] = False
        input_dict['input_lens'] = input_dict['input_lens'] + 1
        kv_len = torch.max(model_inputs.get("position_ids"), axis=1)[0] + 1
        input_dict['kv_len'] = kv_len
        past_key_values = model_inputs.get("past_key_values")
        input_dict["past_key_values"] = past_key_values
        next_tokens = torch.argmax(outputs, dim=-1)
        input_dict['input_ids'] = next_tokens
        input_dict['generate_ids'] = torch.cat([input_dict['generate_ids'], next_tokens], dim=-1)

    @override
    def model_generate(self, prompts, warm_up=False):
        inputs = self.tokenize_prompts(prompts)
        share_mask_tril = get_init_attn_mask(2048, self.device)
        input_lens = copy.deepcopy(inputs.input_ids.size()[1])
        position_ids = inputs.attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(inputs.attention_mask == 0, 1)
        if not self.decode_only:
            input_dict = {
                "input_ids": inputs.input_ids, 
                "generate_ids": inputs.input_ids,
                "input_lens": input_lens, 
                "kv_len": None,
                "past_key_values": None, 
                "attention_mask": inputs.attention_mask, 
                "share_mask_tril": share_mask_tril,
                "is_prefill": True
            }
        else:
            # default padding to input_max_len by tokenize_prompts
            input_dict = {
                "input_ids": torch.full((self.batch_size_per_rank, 1), self.tokenizer.pad_token_id, device='npu'), 
                "generate_ids": inputs.input_ids, 
                "input_lens": input_lens, 
                "kv_len": torch.full((self.batch_size_per_rank,), self.input_max_len, device='npu'),
                "past_key_values": None, 
                "attention_mask": inputs.attention_mask, 
                "share_mask_tril": share_mask_tril, 
                "is_prefill": False
            }
        super().model_generate(input_dict, input_lens, warm_up)

    def ffn_server(self, warm_up=False):
        cnt = 0
        infer_time_rec = []
        is_prefill = True

        max_step = self.max_new_tokens
        profiler = self.define_profiler(
            enable_profiler=self.enable_profiler and not warm_up,
            profile_save_path=f"{self.res_path}/prof_ffn",
        )
        with profiler as prof:
            while cnt < max_step:
                jump_flag = self.get_jump_flag(cnt, warm_up)
                if jump_flag:
                    break

                model_inputs = self.ffn_input_prepare(is_prefill)
                _, infer_time = self.ffn_inference(model_inputs, is_prefill=is_prefill, warm_up=warm_up)
                prof.step()
                if is_prefill:
                    is_prefill = False
                cnt += 1
                infer_time_rec.append(infer_time)

        if not warm_up:
            avg_infer_time = process_infer_time(infer_time_rec, cnt)
            logging.info(f"{self.model_name} ffn average inference time cost is {(avg_infer_time) * 1000:.2f} ms")
