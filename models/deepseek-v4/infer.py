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
import argparse
import logging
import torch

from runner_deepseek import DeepSeekRunner
from models.model_infer import Infer
from executor.utils import update_settings, align_up, read_yaml
from executor.utils.data_utils import generate_prompt
from models.modules.registry import auto_import_modules, OpKernel

root_logger = logging.getLogger()
root_logger.handlers.clear()
logging.basicConfig(format='%(asctime)s - %(levelname)s - [LLM](%(filename)s:%(lineno)d): %(message)s',
                    level=logging.INFO)
logging.getLogger("paramiko").setLevel(logging.ERROR)
torch.manual_seed(42)
torch.npu.manual_seed_all(42)


def parse_args():
    parser = argparse.ArgumentParser(description="llm run parameters")
    parser.add_argument('--yaml_file_path', type=str, help="inference configurations")
    parser.add_argument('--local_rank', type=int, default=0, help="Local rank id for torch distributed launch")
    parser_args = parser.parse_args()
    return parser_args


def run_deepseek(runner_settings):
    preset_prompts, query_id_list = generate_prompt(runner_settings)
    model_runner_main = DeepSeekRunner(runner_settings)
    # to accelerate the compiling process for torch dynamo
    torch.npu.set_compile_mode(jit_compile=False)
    model_runner_main.init_model()

    model_runner_mtp = None
    next_n = runner_settings.get("model_config").get("next_n", 0)
    if next_n > 0:
        model_runner_mtp = DeepSeekRunner(runner_settings)
        model_runner_mtp.init_model(is_mtp=True)
        # the mtp modules share embed, lm_head, rotary_emb with the main model
        model_runner_mtp.model.model.embed_tokens = model_runner_main.model.model.embed_tokens
        model_runner_mtp.model.lm_head = model_runner_main.model.lm_head
        model_runner_mtp.model.rotary_emb = model_runner_main.model.model.rotary_emb

    # init mtp infer process
    infer = Infer(runner_settings, model_runner_main, model_runner_mtp)
    # warmup
    input_dict_main, input_dict_mtp = infer.model_generate(preset_prompts, warm_up=True)
    cache_data_mtp = input_dict_mtp['cache_data'] if next_n > 0 else None
    infer.model_generate(preset_prompts, cache_data=input_dict_main['cache_data'], cache_data_mtp=cache_data_mtp)


def check_parallel_settings(world_size, runner_settings):
    attn_tp_size = runner_settings.get("parallel_config").get("attn_tp_size")
    oproj_tp_size = runner_settings.get("parallel_config").get("oproj_tp_size", 1)
    moe_tp_size = runner_settings.get("parallel_config").get("moe_tp_size")
    embed_tp_size = runner_settings.get("parallel_config").get("embed_tp_size")
    lmhead_tp_size = runner_settings.get("parallel_config").get("lmhead_tp_size", embed_tp_size)
    cp_size = runner_settings.get("parallel_config").get("cp_size", 1)
    dense_tp_size = runner_settings.get("parallel_config").get("dense_tp_size", 1)
    attn_dp_size = world_size // attn_tp_size
    batch_size = runner_settings.get("data_config").get("batch_size", 1)

    if world_size <= 0:
        raise ValueError(f"{world_size=} must greater than 0")
    if world_size % attn_tp_size != 0:
        raise ValueError(f"{world_size=} is not divisible by {attn_tp_size=}")
    if world_size % oproj_tp_size != 0:
        raise ValueError(f"{world_size=} is not divisible by {oproj_tp_size=}")
    if oproj_tp_size > 8:
        raise ValueError(f"{oproj_tp_size=} should not be greater than 8")
    if moe_tp_size != 1:
        raise ValueError(f"{moe_tp_size=} is invalid. The moe_tp_size supports only 1.")
    if attn_tp_size > 1 and oproj_tp_size > 1:
        raise ValueError(f"oproj_tp_size does not support when attn_tp_size is lager than 1")
    if world_size % moe_tp_size != 0:
        raise ValueError(f"{world_size=} is not divisible by {moe_tp_size=}")
    if world_size % embed_tp_size != 0 or world_size % lmhead_tp_size != 0:
        raise ValueError(f"{world_size=} is not divisible by {embed_tp_size=} or {lmhead_tp_size=}")
    if embed_tp_size < attn_tp_size:
        raise ValueError(f"{embed_tp_size=} should not be smaller than {attn_tp_size=}")
    elif embed_tp_size % attn_tp_size != 0:
        raise ValueError(f"{embed_tp_size=} should be a multiple of {attn_tp_size=}")
    if batch_size % attn_dp_size != 0:
        raise ValueError(f"{batch_size=} is not divisible by {attn_dp_size=}")
    if cp_size > 1 and world_size != cp_size:
        raise ValueError(f"when cp enabled, {world_size=} should equal to {cp_size=}")
    if world_size % dense_tp_size != 0:
        raise ValueError(f"{world_size=} is not divisible by {dense_tp_size=}")


def check_model_settings(world_size, runner_settings):
    exe_mode = runner_settings.get("exe_mode")
    enable_cache_compile = runner_settings.get("model_config").get("enable_cache_compile", False)
    enable_multi_streams = runner_settings.get("model_config").get("enable_multi_streams", False)
    enable_limit_core = runner_settings.get("model_config").get("enable_limit_core", False)
    platform_version = runner_settings.get("model_config").get("platform_version", "A3")
    enable_superkernel = runner_settings.get("model_config").get("enable_superkernel", False)
    enable_pypto = runner_settings.get("model_config").get("enable_pypto", False)
    next_n = runner_settings.get("model_config").get("next_n", 0)
    prefill_mini_batch_size = runner_settings.get("model_config").get("prefill_mini_batch_size", 0)
    with_ckpt = runner_settings.get("model_config").get("with_ckpt", True)
    perfect_eplb = runner_settings.get("model_config").get("perfect_eplb", False)

    if not with_ckpt and not perfect_eplb:
        raise ValueError(f"{perfect_eplb=} must be True if {with_ckpt =}!")

    if exe_mode not in ["ge_graph", "eager", "npugraph_ex"]:
        raise ValueError(f"{exe_mode=} does not supported!")

    dynamo_feat = (enable_cache_compile or enable_superkernel)
    if exe_mode == "eager" and dynamo_feat:
        raise ValueError(f"{exe_mode=} does not support cache compile or superkernel!")
    if exe_mode == "eager" and enable_multi_streams:
        logging.info("When using eager execution mode, enable-multi-streams only takes effect during the prefill phase.")
    if enable_limit_core and not enable_multi_streams:
        raise ValueError(f"{enable_limit_core=} only if enable_multi_streams!")
    if enable_limit_core and platform_version != "A3":
        raise ValueError(f"{enable_limit_core=} only supports platform A3!")
    if enable_limit_core and enable_pypto:
        raise ValueError(f"{enable_pypto=} does not support {enable_limit_core=}!")
    pa_max_length = runner_settings.get("model_config").get("pa_max_length", 2048)
    block_size = runner_settings.get("model_config").get("pa_block_size", 128)
    if pa_max_length % block_size != 0:
        raise ValueError(f"{pa_max_length=} should be a multiple of {block_size=}")
    if next_n > 3:
        raise ValueError(f"{next_n=} must equal or smaller than 3")
    if prefill_mini_batch_size > 0:
        batch_size = runner_settings.get("data_config").get("batch_size", 1)
        if prefill_mini_batch_size > batch_size or batch_size % prefill_mini_batch_size != 0:
            raise ValueError(f"{batch_size=} should be divided by {prefill_mini_batch_size=}")
        attn_tp_size = runner_settings.get("parallel_config").get("attn_tp_size")
        attn_dp_size = world_size // attn_tp_size
        batch_size_per_rank = batch_size // attn_dp_size
        if prefill_mini_batch_size > batch_size_per_rank or batch_size_per_rank % prefill_mini_batch_size != 0:
            raise ValueError(f"{batch_size_per_rank=} should be divided by {prefill_mini_batch_size=}")
    cp_size = runner_settings.get("parallel_config").get("cp_size", 1)
    if cp_size > 1 and prefill_mini_batch_size != 1:
        raise ValueError(f"when cp enabled, {prefill_mini_batch_size=} should be 1")


def check_vars(world_size, runner_settings):
    check_parallel_settings(world_size, runner_settings)
    check_model_settings(world_size, runner_settings)
    temperature = runner_settings.get("data_config").get("temperature", 1.0)
    if temperature < 0.0:
        raise ValueError(f"{temperature=} should be greater than or equal to 0.0")


def update_op_kernel_dict(runner_settings):
    """
    kernel_config: Dict, op impls defined by user, {op_type: op_impls}
    OpKernel.OP_TYPE: List, contain ops with different impls
    OpKernel.KERNEL_MAP: Dict, contain op_impls {op_impl_name: op_impl}
    """
    # import all mudules under models.modules.op_impls files
    auto_import_modules("models.modules.op_impls")
    kernel_config = runner_settings.get("kernel_config", {})
    platform_version = runner_settings.get("model_config").get("platform_version", "A3").lower()
    enable_pypto = runner_settings.get("model_config").get("enable_pypto", False)
    if enable_pypto and platform_version != "a3":
        raise (f"PYPTO kernel for this model on {platform_version=} is not supported yet.")
    for op_type in OpKernel.OP_TYPE:
        if op_type in kernel_config:
            kernel_impl = kernel_config[op_type]
            used_kernel = op_type + "_" + kernel_impl + "_" + platform_version
        else:
            default_kernel = op_type + "_ascendc" + "_" + platform_version
            if default_kernel in OpKernel.KERNEL_MAP:
                used_kernel = default_kernel
            else:
                used_kernel = op_type + "_native" + "_" + platform_version
        if enable_pypto:
            pypto_kernel = op_type + "_pypto_a3"
            if pypto_kernel in OpKernel.KERNEL_MAP:
                used_kernel = pypto_kernel
        OpKernel.op_impl_apply(op_type, used_kernel)
        logging.info(f"{op_type} use impl {used_kernel}")


def update_vars(world_size, runner_settings):
    attn_tp_size = runner_settings.get("parallel_config").get("attn_tp_size")
    attn_dp_size = world_size // attn_tp_size
    cp_size = runner_settings.get("parallel_config").get("cp_size", 1)
    moe_dp_size = world_size // runner_settings.get("parallel_config").get("moe_tp_size")
    moe_ep_size = moe_dp_size
    embed_dp_size = world_size // runner_settings.get("parallel_config").get("embed_tp_size")

    batch_size = runner_settings.get("data_config").get("batch_size", 1)
    batch_size_per_rank = batch_size // attn_dp_size
    if cp_size > 1:
        prefill_dp_size = world_size // cp_size // attn_tp_size
        bs_per_cp_group = batch_size // prefill_dp_size
        runner_settings = update_settings(runner_settings, "data_config", "bs_per_cp_group", bs_per_cp_group)

    runner_settings = update_settings(runner_settings, "data_config", "batch_size_per_rank", batch_size_per_rank)
    runner_settings = update_settings(runner_settings, "parallel_config", "attn_dp_size", attn_dp_size)
    runner_settings = update_settings(runner_settings, "parallel_config", "moe_dp_size", moe_dp_size)
    runner_settings = update_settings(runner_settings, "parallel_config", "moe_ep_size", moe_ep_size)
    runner_settings = update_settings(runner_settings, "parallel_config", "embed_dp_size", embed_dp_size)

    input_max_len = runner_settings.get("data_config").get("input_max_len", 32)
    max_new_tokens = runner_settings.get("data_config").get("max_new_tokens", 32)
    if cp_size > 1:
        input_max_len = align_up(input_max_len, cp_size * 2)
        runner_settings = update_settings(runner_settings, "data_config", "input_max_len", input_max_len)
        if input_max_len // (cp_size * 2) < 128:
            input_max_len = cp_size * 2 * 128
            runner_settings = update_settings(runner_settings, "data_config", "input_max_len", input_max_len)
            logging.warning(f"Sequence length is too short. To enable CP, \
                            it has been padded to cp_size * 2 * 128 (model window size).")
    next_n = runner_settings.get("model_config").get("next_n", 0)
    if next_n == 0:
        max_position_embeddings = max_new_tokens + input_max_len
    else: # MTP
        max_position_embeddings = max_new_tokens * (next_n + 1) + input_max_len + next_n - 1
    runner_settings = update_settings(
        runner_settings, "data_config", "max_position_embeddings", max_position_embeddings)

    pa_block_size = runner_settings.get("model_config").get("pa_block_size", 128)
    pa_max_length = align_up(max_position_embeddings, pa_block_size)
    runner_settings = update_settings(runner_settings, "model_config", "pa_max_length", pa_max_length)
    platform_version = runner_settings.get("model_config").get("platform_version", "A3")
    enable_weight_nz = not platform_version == "950"
    runner_settings = update_settings(runner_settings, "model_config", "enable_weight_nz",
                                      enable_weight_nz)
    return runner_settings


if __name__ == "__main__":
    args = parse_args()
    yaml_file_path = args.yaml_file_path
    runner_settings = read_yaml(yaml_file_path)
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    check_vars(world_size, runner_settings)
    update_op_kernel_dict(runner_settings)
    runner_settings = update_vars(world_size, runner_settings)
    logging.info(f"runner_settings is: {runner_settings}")

    run_deepseek(runner_settings)
    logging.info("model run success")
