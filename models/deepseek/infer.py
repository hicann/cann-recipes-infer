# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import argparse
import logging
import torch

from runner_deepseek import DeepSeekRunner
from models.mtp import InferMTP
from models.model_setting import update_vars, check_vars
from executor.utils import read_yaml
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
    parser.add_argument('--yaml_file_path', type=str, help="inference configurations")
    parser.add_argument('--local_rank', type=int, default=0, help="Local rank id for torch distributed launch")
    parser_args = parser.parse_args()
    return parser_args


def run_deepseek(runner_settings):
    preset_prompts, _ = generate_prompt(runner_settings)
    model_runner = DeepSeekRunner(runner_settings)
    # 表示在图模式下开启算子二进制复用，提高图模式下编译阶段性能
    torch.npu.set_compile_mode(jit_compile=False)
    model_runner.init_model()
    # warmup
    model_runner.model_generate(preset_prompts, warm_up=True)
    logging.info(f"Warm up finishes.")

    # generate perf data
    model_runner.model_generate(preset_prompts)


def run_deepseek_mtp(runner_settings):
    preset_prompts, query_id_list = generate_prompt(runner_settings)
    model_runner_main = DeepSeekRunner(runner_settings)
    model_runner_mtp = DeepSeekRunner(runner_settings)
    # to accelerate the compiling process for torch dynamo
    torch.npu.set_compile_mode(jit_compile=False)
    model_runner_main.init_model()
    model_runner_mtp.init_model(is_mtp=True)
    # the mtp modules share embed, lm_head, rotary_emb with the main model
    model_runner_mtp.model.model.embed_tokens = model_runner_main.model.model.embed_tokens
    model_runner_mtp.model.lm_head = model_runner_main.model.lm_head
    model_runner_mtp.model.rotary_emb = model_runner_main.model.model.rotary_emb

    # init mtp infer process
    infer_mtp = InferMTP(runner_settings, model_runner_main, model_runner_mtp)
    # warmup
    infer_mtp.model_generate_mtp(preset_prompts, warm_up=True)
    logging.info(f"Warm up finishes.")

    # generate perf data
    infer_mtp.model_generate_mtp(preset_prompts)


if __name__ == "__main__":
    args = parse_args()
    yaml_file_path = args.yaml_file_path
    runner_settings = read_yaml(yaml_file_path)
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    check_vars(world_size, runner_settings)
    update_vars(world_size, runner_settings)
    logging.info(f"runner_settings is: {runner_settings}")

    next_n = runner_settings.get("model_config").get("next_n", 0)
    if next_n > 0:
        run_deepseek_mtp(runner_settings)
    else:
        run_deepseek(runner_settings)
    logging.info("model run success")
