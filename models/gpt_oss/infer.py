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
from runner_gpt_oss import GptOssRunner
from executor.utils import read_yaml, update_settings
from executor.utils.data_utils import generate_prompt

root_logger = logging.getLogger()
root_logger.handlers.clear()
logging.basicConfig(format='%(asctime)s - %(levelname)s - [LLM](%(filename)s:%(lineno)d): %(message)s',
                    level=logging.INFO)
logging.getLogger("paramiko").setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser(description="llm run parameters")
    parser.add_argument('--yaml_file_path', type=str, help="inference configurations")
    parser_args = parser.parse_args()
    return parser_args


def check_vars(world_size, runner_settings):
    tp_size = runner_settings.get("parallel_config").get("tp_size")
    if world_size <= 0:
        raise ValueError(f"{world_size=} must greater than 0")
    if not (world_size == tp_size):
        raise ValueError(f"{world_size=} must be equal to {tp_size=}")


def update_vars(world_size, runner_settings):
    batch_size = runner_settings.get("data_config").get("batch_size", 1)
    input_max_len = runner_settings.get("data_config").get("input_max_len", 32)
    max_new_tokens = runner_settings.get("data_config").get("max_new_tokens", 32)
    max_len_bound = max_new_tokens + input_max_len
    runner_settings = update_settings(runner_settings, "data_config", "max_position_embeddings", max_len_bound)

    return runner_settings


def run_gpt_oss(runner_settings):
    tp_size = runner_settings.get("parallel_config").get("tp_size", 1)
    preset_prompts, _ = generate_prompt(runner_settings, tp_size)
    model_runner = GptOssRunner(runner_settings)
    model_runner.init_model()
    # generate perf data
    model_runner.model_generate(preset_prompts)


if __name__ == "__main__":
    args = parse_args()
    yaml_file_path = args.yaml_file_path
    runner_settings = read_yaml(yaml_file_path)
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    check_vars(world_size, runner_settings)
    logging.info(f"runner_settings is: {runner_settings}")
    update_vars(world_size, runner_settings)
    run_gpt_oss(runner_settings)
    logging.info("model run success")
