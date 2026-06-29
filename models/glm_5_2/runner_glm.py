# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
import sys
import math
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
from models.modeling_glm import GlmMoeDsaForCausalLM, GlmMoeDsaModelMTP
from executor.utils import override, get_init_attn_mask
from executor.model_runner import ModelRunner
from module.quantization import QuantizeMethodBase
from module.quantization.compressed_tensors.compressed_tensors_moe_gmm import (
    CompressedTensorW8A8Int8MoEGMMMethod,
    CompressedTensorW4A8Int8MoEGMMMethod
)

root_logger = logging.getLogger()
root_logger.handlers.clear()
logging.basicConfig(format='%(asctime)s - %(levelname)s - [LLM](%(filename)s:%(lineno)d): %(message)s',
                    level=logging.INFO)
logging.getLogger("paramiko").setLevel(logging.ERROR)

torch.manual_seed(42)
torch.npu.manual_seed_all(42)


class GlmRunner(ModelRunner):
    def __init__(self, runner_settings):
        super().__init__(runner_settings)
        self.batch_size = runner_settings.get("data_config").get("batch_size")
        self.with_ckpt = runner_settings.get("model_config").get("with_ckpt", True)
        self.enable_weight_nz = runner_settings.get("model_config").get("enable_weight_nz", True)
        self.enable_cache_compile = runner_settings.get("model_config").get("enable_cache_compile", False)
        self.share_mask_tril = get_init_attn_mask(2048, self.device)  # 2048: fixed shape of mask, used in PFA
        self.prefill_mini_batch_size = runner_settings.get("model_config").get("prefill_mini_batch_size", 0)
        self.batch_size_per_rank = runner_settings.get("data_config").get("batch_size_per_rank", 1)
        self.cp_size = self.runner_settings.get("parallel_config").get("cp_size", 1)
        self.use_dataset = runner_settings.get("data_config").get("dataset", "default") != "default"
        self.platform_version = self.runner_settings.get("model_config").get("platform_version", "A3")
        self.prefill_cycles = 0
        self.query_id_list = []
        self.max_new_tokens = runner_settings.get("data_config").get("max_new_tokens", 32)
        self.enable_static_kernel = self.runner_settings.get("model_config").get("enable_static_kernel", False)

    @override
    def init_model(self, is_mtp=False):
        self.is_mtp = is_mtp
        if self.with_ckpt:
            self.use_pretrained_model = True
            config = None
        else:
            self.use_pretrained_model = False
        from models.configuration_glm import GlmMoeDsaConfig as config
        logging.info(f"use_pretrained_model: {self.use_pretrained_model}")
        if is_mtp:
            model = GlmMoeDsaModelMTP
            super().init_model(GlmMoeDsaModelMTP, config)
        else:
            model = GlmMoeDsaForCausalLM
            super().init_model(GlmMoeDsaForCausalLM, config)

    @override
    def _process_weight_after_loading(self):
        '''
        Doing weight transpose, format cast to nz, and scale type cast after loading weights from files.
        '''
        self.init_splited_kv_b_weight()
        self.to_device()
        # map for scales need to cast to float when apply w8a8 quant method
        float_scales_map = [
            "gate_up_proj",
            "q_b_proj",
            "wq_b",
        ]
        # map for smooth scales need to cast to float when apply w8a8 quant method
        float_smooth_scales_map = [
            "down_proj"
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
                # if smooth scale in module need type cast, add target dtype to dict
                if smooth_scale_name in module_name:
                    scales_dtype['smooth_scale_dtype'] = torch.float
                    break
            enable_weight_nz = self.enable_weight_nz
            if self.platform_version == "950":
                if any(
                    attn_proj_name in module_name
                    for attn_proj_name in ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa"]
                ):
                    enable_weight_nz = True

            if isinstance(quant_method, QuantizeMethodBase):
                quant_method.process_weights_after_loading(
                    module, is_nz=enable_weight_nz, scales_dtype=scales_dtype)

            if self.platform_version == "950" and self.model.config.quant_config.mm_quant_mode == "w8a8mxfloat8":
                if any(
                    attn_proj_name in module_name
                    for attn_proj_name in ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa"]
                ):
                    module.weight_scale = nn.Parameter(
                        module.weight_scale.transpose(0, 1).flatten(1).view(dtype=torch.float8_e8m0fnu),
                        requires_grad=False
                    )

            # Dynamic quant for input_avtivation of first grouped matmul requies complete smooth scale.
            # When applying expert parallel, each device only reserves smooth scales of mapping experts.
            # Need to do all gather to obtain complete smooth scale.
            if isinstance(quant_method, CompressedTensorW8A8Int8MoEGMMMethod) or \
                isinstance(quant_method, CompressedTensorW4A8Int8MoEGMMMethod):
                moe_ep_size = self.runner_settings.get("parallel_config").get("moe_ep_size", 1)
                if moe_ep_size > 1:
                    all_experts_smooth_scale = module.smooth_scale_1.data.new_empty(
                        module.smooth_scale_1.data.shape[0] * moe_ep_size, module.smooth_scale_1.data.shape[1])
                    dist.all_gather_into_tensor(all_experts_smooth_scale, module.smooth_scale_1.data,
                                                group=self.model.hccl_comm_dict.get("moe_ep_group", None))
                    module.smooth_scale_1.data = all_experts_smooth_scale

    @override
    def graph_compile(self):
        if not self.enable_cache_compile:
            torch._dynamo.config.inline_inbuilt_nn_modules = False
            if self.execute_mode == "npugraph_ex":
                compile_options = {
                    "frozen_parameter": True,
                    "static_kernel_compile": self.enable_static_kernel,
                }
                self.model.decode = torch.compile(self.model.decode, dynamic=False, fullgraph=True,
                                                backend="npugraph_ex", options=compile_options)
            else:
                import torchair as tng
                from torchair.configs.compiler_config import CompilerConfig

                compiler_config = CompilerConfig()
                compiler_config.experimental_config.frozen_parameter = True
                compiler_config.experimental_config.tiling_schedule_optimize = True
                compiler_config.experimental_config.topology_sorting_strategy = "StableRDFS"
                npu_backend = tng.get_npu_backend(compiler_config=compiler_config)
                self.model.decode = torch.compile(self.model.decode, dynamic=False, fullgraph=True, backend=npu_backend)

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