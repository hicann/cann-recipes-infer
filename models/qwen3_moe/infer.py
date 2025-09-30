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
import json
import torch

CUR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.realpath(os.path.join(CUR_DIR, ".."))
sys.path.append(ROOT_DIR)
from runner_qwen3_moe import Qwen3MoeRunner
from executor.utils import update_settings, align_up, read_yaml
from executor.utils.data_utils import generate_prompt

root_logger = logging.getLogger()
root_logger.handlers.clear()
logging.basicConfig(format='%(asctime)s - %(levelname)s - [LLM](%(filename)s:%(lineno)d): %(message)s',
                    level=logging.INFO)
logging.getLogger("paramiko").setLevel(logging.ERROR)
torch.manual_seed(42)
torch.npu.manual_seed_all(42)


def parse_args():
    parser = argparse.ArgumentParser(description="llm run parameters")
    parser.add_argument('--local_rank', type=int, default=0, help="Local rank id for torch distributed launch")
    parser.add_argument('--yaml_file_path', type=str, help="inference configurations")
    parser_args = parser.parse_args()
    return parser_args


def run_qwen3_moe(runner_settings):
    attn_tp_size = runner_settings.get("parallel_config").get("attn_tp_size", 1)
    preset_prompts, _ = generate_prompt(runner_settings, attn_tp_size)
    model_runner = Qwen3MoeRunner(runner_settings)
    # 表示在图模式下开启算子二进制复用，提高图模式下编译阶段性能
    torch.npu.set_compile_mode(jit_compile=False)
    model_runner.init_model()
    # warmup
    model_runner.model_generate(preset_prompts, warm_up=True)
    logging.info(f"Warm up finishes.")

    # generate perf data
    model_runner.model_generate(preset_prompts)


def check_parallel_settings(world_size, runner_settings):
    attn_tp_size = runner_settings.get("parallel_config").get("attn_tp_size")
    moe_tp_size = runner_settings.get("parallel_config").get("moe_tp_size")
    embed_tp_size = runner_settings.get("parallel_config").get("embed_tp_size")
    lmhead_tp_size = runner_settings.get("parallel_config").get("lmhead_tp_size", embed_tp_size)

    if world_size <= 0:
        raise ValueError(f"{world_size=} must greater than 0")
    if not (attn_tp_size == embed_tp_size == lmhead_tp_size):
        raise ValueError(f"{attn_tp_size=} must be equal to {embed_tp_size=} and {lmhead_tp_size=}")
    if world_size % attn_tp_size != 0:
        raise ValueError(f"{world_size=} is not divisible by {attn_tp_size=}")
    if world_size % moe_tp_size != 0:
        raise ValueError(f"{world_size=} is not divisible by {moe_tp_size=}")
    if world_size % embed_tp_size != 0 or world_size % lmhead_tp_size != 0:
        raise ValueError(f"{world_size=} is not divisible by {embed_tp_size=} or {lmhead_tp_size=}")
    if embed_tp_size < attn_tp_size:
        raise ValueError(f"{embed_tp_size=} should not be smaller then {attn_tp_size=}")
    elif embed_tp_size % attn_tp_size != 0:
        raise ValueError(f"{embed_tp_size=} should be a multiple of {attn_tp_size=}")


def check_model_settings(runner_settings):
    exe_mode = runner_settings.get("exe_mode")
    enable_cache_compile = runner_settings.get("model_config").get("enable_cache_compile", False)

    if exe_mode not in ["ge_graph", "eager"]:
        raise ValueError(f"{exe_mode=} does not supported! Only the eager and ge_graph mode are supported！")

    dynamo_feat = enable_cache_compile
    if exe_mode == "eager" and dynamo_feat:
        logging.info(f"{exe_mode=} does not support cache compile!")


def check_vars(world_size, runner_settings):
    check_parallel_settings(world_size, runner_settings)
    check_model_settings(runner_settings)


def update_vars(world_size, runner_settings):
    attn_dp_size = world_size // runner_settings.get("parallel_config").get("attn_tp_size")
    moe_dp_size = world_size // runner_settings.get("parallel_config").get("moe_tp_size")
    moe_ep_size = moe_dp_size
    embed_dp_size = world_size // runner_settings.get("parallel_config").get("embed_tp_size")

    batch_size = runner_settings.get("data_config").get("batch_size", 1)
    batch_size_per_rank = batch_size // attn_dp_size

    runner_settings = update_settings(runner_settings, "data_config", "batch_size_per_rank", batch_size_per_rank)
    runner_settings = update_settings(runner_settings, "parallel_config", "attn_dp_size", attn_dp_size)
    runner_settings = update_settings(runner_settings, "parallel_config", "moe_dp_size", moe_dp_size)
    runner_settings = update_settings(runner_settings, "parallel_config", "moe_ep_size", moe_ep_size)
    runner_settings = update_settings(runner_settings, "parallel_config", "embed_dp_size", embed_dp_size)

    input_max_len = runner_settings.get("data_config").get("input_max_len", 32)
    max_new_tokens = runner_settings.get("data_config").get("max_new_tokens", 32)

    max_len_bound = max_new_tokens + input_max_len
    runner_settings = update_settings(runner_settings, "data_config", "max_position_embeddings", max_len_bound)

    return runner_settings


if __name__ == "__main__":
    args = parse_args()
    yaml_file_path = args.yaml_file_path
    runner_settings = read_yaml(yaml_file_path)
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    check_vars(world_size, runner_settings)
    logging.info(f"runner_settings is: {runner_settings}")
    update_vars(world_size, runner_settings)
    run_qwen3_moe(runner_settings)
    logging.info("model run success")
