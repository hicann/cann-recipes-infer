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
import gc
from operator import attrgetter
from typing import get_args
import numpy as np
import torch
import torch.distributed as dist
import torch_npu
import torch.nn as nn
from models.modeling_deepseek import DeepseekV3ForCausalLM, DeepseekV3ModelMTP
from executor.utils import override, get_init_attn_mask, process_infer_time
from executor.model_runner import ModelRunner
from executor.model_loader.default_loader import DefaultModelLoader
from executor.model_loader.dummy_loader import DummyModelLoader
from module.utils import to_transpose_nz
from module.quantization import QuantizeMethodBase
from module.quantization.compressed_tensors.compressed_tensors_moe_gmm import CompressedTensorW8A8Int8MoEGMMMethod

root_logger = logging.getLogger()
root_logger.handlers.clear()
logging.basicConfig(format='%(asctime)s - %(levelname)s - [LLM](%(filename)s:%(lineno)d): %(message)s',
                    level=logging.INFO)
logging.getLogger("paramiko").setLevel(logging.ERROR)

torch.manual_seed(42)
torch.npu.manual_seed_all(42)


class DeepSeekRunner(ModelRunner):
    def __init__(self, runner_settings):
        super().__init__(runner_settings)
        self.batch_size = runner_settings.get("data_config").get("batch_size")
        self.with_ckpt = runner_settings.get("model_config").get("with_ckpt", True)
        self.enable_weight_nz = runner_settings.get("model_config").get("enable_weight_nz", True)
        self.enable_cache_compile = runner_settings.get("model_config").get("enable_cache_compile", False)
        self.enable_mla_prolog = runner_settings.get("model_config").get("enable_mla_prolog", False)
        self.share_mask_tril = get_init_attn_mask(2048, self.device)  # 2048: fixed shape of mask, used in PFA
        self.enable_prefill_multi_cycle = runner_settings.get("model_config").get("enable_prefill_multi_cycle", False)
        self.enable_o_proj_alltoall = runner_settings.get("parallel_config").get("enable_o_proj_alltoall", False)
        if self.enable_prefill_multi_cycle and not runner_settings.get("model_config").get("enable_pa", False):
            raise ValueError("not support prefill_multi_cycle when disable paged attention!")
        self.prefill_cycles = 0
        self.query_id_list = []
        self.past_key_values = None

    @override
    def init_model(self, is_mtp=False):
        self.is_mtp = is_mtp
        if self.with_ckpt:
            self.use_pretrained_model = True
            config = None
        else:
            self.use_pretrained_model = False
        from models.configuration_deepseek import DeepseekV3Config as config
        logging.info(f"use_pretrained_model: {self.use_pretrained_model}")
        if is_mtp:
            model = DeepseekV3ModelMTP
            super().init_model(DeepseekV3ModelMTP, config)
        else:
            model = DeepseekV3ForCausalLM
            super().init_model(DeepseekV3ForCausalLM, config)

    @override
    def _process_weight_after_loading(self):
        '''
        Doing weight transpose, format cast to nz, and scale type cast after loading weights from files.
        '''
        self.init_splited_kv_b_weight()
        self.to_device()
        # map for scales need to cast to float when apply w8a8 quant method
        float_scales_map = [
            "merge_up_gate_proj",
        ]
        # map for smooth scales need to cast to float when apply w8a8 quant method
        float_smooth_scales_map = [
            "down_proj"
        ]
        if self.enable_mla_prolog:
            float_scales_map += [
                "q_a_proj",
                "q_b_proj",
                "kv_a_proj_with_mqa"
            ]
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
                # if smootj scale in module need type cast, add target dtype to dict
                if smooth_scale_name in module_name:
                    scales_dtype['smooth_scale_dtype'] = torch.float
                    break

            is_nz = False if ("mlp.gate" in module_name and "proj" not in module_name) else True
            if isinstance(quant_method, QuantizeMethodBase):
                quant_method.process_weights_after_loading(module, is_nz=is_nz, scales_dtype=scales_dtype)
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

    @override
    def init_splited_kv_b_weight(self):
        def for_each_to_init_splited_k_b_weight(layer, layer_idx=""):
            try:
                data_getter = attrgetter("self_attn.kv_b_proj_w_k_data")
                data_tensor = data_getter(layer)
                layer.self_attn.kv_b_proj_w_k = nn.Parameter(data_tensor.contiguous(), requires_grad=False)
            except AttributeError:
                pass

        def for_each_to_init_splited_v_b_weight(layer, layer_idx=""):
            try:
                data_getter = attrgetter("self_attn.kv_b_proj_w_v_data")
                data_tensor = data_getter(layer)
                layer.self_attn.kv_b_proj_w_v = nn.Parameter(data_tensor.contiguous(), requires_grad=False)
            except AttributeError:
                pass

        def for_each_to_offload_kv_b_weight(layer, layer_idx=""):
            try:
                layer.self_attn.kv_b_proj.weight = None
            except AttributeError:
                pass

        if self.is_mtp:
            for _, layer in self.model.model.layers.items():
                for_each_to_init_splited_k_b_weight(layer, self.model.config.num_hidden_layers)
                for_each_to_init_splited_v_b_weight(layer, self.model.config.num_hidden_layers)
                for_each_to_offload_kv_b_weight(layer, self.model.config.num_hidden_layers)
        else:
            for layer_idx, layer in enumerate(self.model.model.layers):
                for_each_to_init_splited_k_b_weight(layer, layer_idx)
                for_each_to_init_splited_v_b_weight(layer, layer_idx)
                for_each_to_offload_kv_b_weight(layer, layer_idx)
        gc.collect()

    @override
    def graph_compile(self):
        import torchair as tng
        tng.patch_for_hcom()
        from torchair.configs.compiler_config import CompilerConfig
        use_aclgraph = self.execute_mode == "acl_graph"

        compiler_config = CompilerConfig()
        compiler_config.experimental_config.frozen_parameter = True
        compiler_config.experimental_config.tiling_schedule_optimize = True
        compiler_config.experimental_config.topology_sorting_strategy = "StableRDFS"
        if use_aclgraph:
            compiler_config.mode = "reduce-overhead"
            if torch.__version__ < "2.5.0":
                compiler_config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = (
                    True
                )

        npu_backend = tng.get_npu_backend(compiler_config=compiler_config)
        if self.enable_cache_compile:
            case_name = "compile_cache/" + os.getenv("CASE_NAME")
            if self.is_mtp:
                case_name += "_spec"
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), case_name)
            self.model.decode = tng.inference.cache_compile(
                self.model.decode,
                cache_dir=cache_dir,
                config=compiler_config,
                dynamic=True if use_aclgraph else False,
                fullgraph=True,
                ge_cache=not use_aclgraph,
            )
        else:
            self.model.decode = torch.compile(self.model.decode,
                                              dynamic=True if use_aclgraph else False,
                                              fullgraph=True,
                                              backend=npu_backend)

    def prefill_multi_cycle(self, model_inputs, mini_batch: int = 1):
        self.prefill_cycles = 0
        self.processed_query_id_list = []
        logits = []
        prev_hidden_states = []
        kv_len = model_inputs['kv_len'].clone().detach().cpu().tolist()
        kv_len_cumsum = [0] + np.cumsum(kv_len).tolist()
        bs = len(kv_len)

        self.prefill_cycles = bs // mini_batch
        for i in range(self.prefill_cycles):
            self.processed_query_id_list += list(range(i * mini_batch, (i + 1) * mini_batch))
            model_inputs_prefill = {}
            model_inputs_prefill['input_ids'] = \
                model_inputs['input_ids'][0][kv_len_cumsum[i * mini_batch]: kv_len_cumsum[(i + 1) * mini_batch]]\
                .unsqueeze(0)
            model_inputs_prefill['position_ids'] = model_inputs['position_ids'][
                i * mini_batch: (i + 1) * mini_batch]
            model_inputs_prefill['past_key_values'] = ()
            for past_key_value in model_inputs['past_key_values']:
                k, v = past_key_value
                model_inputs_prefill['past_key_values'] += (
                    (
                        k.narrow(0, i * self.model.cache_len, self.model.cache_len),
                        v.narrow(0, i * self.model.cache_len, self.model.cache_len),
                    ),
                )
            model_inputs_prefill['kv_len'] = model_inputs['kv_len'][i * mini_batch: (i + 1) * mini_batch]
            model_inputs_prefill['attention_mask'] = model_inputs['attention_mask']
            if model_inputs_prefill['attention_mask'] is not None and not self.model.enable_pa:
                model_inputs_prefill['attention_mask'] = \
                    model_inputs_prefill['attention_mask'][i * mini_batch: (i + 1) * mini_batch]
            model_inputs_prefill['actual_seq_lengths_kv'] = torch.cumsum(model_inputs_prefill['kv_len'], dim=0).tolist()
            model_inputs_prefill['prev_hidden_states'] = model_inputs['prev_hidden_states']
            if (model_inputs_prefill['prev_hidden_states'] is not None) and self.is_mtp:
                model_inputs_prefill['prev_hidden_states'] = \
                    model_inputs_prefill['prev_hidden_states'][0][kv_len_cumsum[i * mini_batch]:\
                        kv_len_cumsum[(i + 1) * mini_batch]].unsqueeze(0)
            if self.model.perfect_eplb:
                # The batch and seq axis of prefill have been merged, so batch_size is 1
                model_inputs_prefill["cur_topk_list"] = self.model.gen_cur_topk_idx(True, 1,
                    kv_len_cumsum[(i + 1) * mini_batch] - kv_len_cumsum[i * mini_batch])

            logit, prev_hidden_state = self.model.prefill(**model_inputs_prefill)
            logits.append(logit)
            prev_hidden_states.append(prev_hidden_state)

        logits = torch.cat(logits, dim=0)
        # shape is BSH, concat S dim
        prev_hidden_states = torch.cat(prev_hidden_states, dim=1)
        return logits, prev_hidden_states

    @override
    def model_input_prepare(self, input_dict):
        input_ids = input_dict.get("input_ids")
        attention_mask = input_dict.get("attention_mask")
        past_key_values = input_dict.get("past_key_values")
        is_prefill = input_dict.get("is_prefill")
        kv_len = input_dict.get("kv_len")
        share_mask_tril = input_dict.get("share_mask_tril")
        prev_hidden_states = input_dict.get("prev_hidden_states")
        if past_key_values is None:
            self.past_key_values = self.model.init_cache(input_ids)
            input_dict["past_key_values"] = self.past_key_values
        model_inputs = self.model.prepare_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=self.past_key_values,
            prev_hidden_states=prev_hidden_states,
            is_prefill=is_prefill,
            kv_len=kv_len,
            input_lens=input_dict.get("input_lens"),
            share_mask_tril=share_mask_tril)
        if self.model.perfect_eplb:
            input_ids_update = model_inputs.get("input_ids")
            model_inputs["cur_topk_list"] = \
                self.model.gen_cur_topk_idx(is_prefill, input_ids_update.shape[0], input_ids_update.shape[1])

        return model_inputs

    @override
    def model_inference(self, model_inputs, is_prefill=False, warm_up=False):
        torch.npu.synchronize()
        start_time = time.time()
        with torch.no_grad():
            if is_prefill:
                if self.enable_prefill_multi_cycle:
                    logits, prev_hidden_states = self.prefill_multi_cycle(model_inputs, 1)
                else:
                    logits, prev_hidden_states = self.model.prefill(**model_inputs)
            else:
                logits, prev_hidden_states = self.model.decode(**model_inputs)

        torch.npu.synchronize()
        end_time = time.time()
        inference_time = end_time - start_time
        inference_stage = "prefill" if is_prefill else "decode"
        logging.info(f"{self.model_name} inference time cost of {inference_stage} is {(inference_time)*1000:.2f} ms")
        return (logits, inference_time, prev_hidden_states)

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
        input_lens = copy.deepcopy(inputs.input_ids.size()[1])
        input_dict = {
            "input_ids": inputs.input_ids, "generate_ids": inputs.input_ids,
            "input_lens": input_lens, "kv_len": None,
            "past_key_values": self.past_key_values,
            "attention_mask": inputs.attention_mask,
            "share_mask_tril": self.share_mask_tril,
            "is_prefill": True,
        }
        super().model_generate(input_dict, input_lens, warm_up)