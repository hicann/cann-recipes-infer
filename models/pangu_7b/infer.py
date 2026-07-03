# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import logging
import os
import sys
import torch

from runner_openpangu_dense import PanguEmbeddedRunner

from executor.utils import read_yaml
from executor.utils.data_utils import generate_prompt
from models.model_setting import check_vars, update_vars

CUR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.realpath(os.path.join(CUR_DIR, ".."))
sys.path.append(ROOT_DIR)

root_logger = logging.getLogger()
root_logger.handlers.clear()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - [LLM](%(filename)s:%(lineno)d): %(message)s",
    level=logging.INFO,
)
logging.getLogger("paramiko").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

torch.manual_seed(42)
torch.npu.manual_seed_all(42)


def parse_args():
    parser = argparse.ArgumentParser(description="llm run parameters")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank id for torch distributed launch")
    parser.add_argument("--yaml_file_path", type=str, help="inference configurations")
    parser_args = parser.parse_args()
    return parser_args


def run_pangu_embedded(runner_settings):
    preset_prompts, _ = generate_prompt(runner_settings)
    model_runner = PanguEmbeddedRunner(runner_settings)
    torch.npu.set_compile_mode(jit_compile=False)
    model_runner.init_model()
    # warmup
    model_runner.model_generate(preset_prompts, warm_up=True)
    logger.info("Warm up finishes.")
    # generate perf data
    model_runner.model_generate(preset_prompts)


if __name__ == "__main__":
    args = parse_args()
    yaml_file_path = args.yaml_file_path
    runner_settings = read_yaml(yaml_file_path)
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    check_vars(world_size, runner_settings)
    update_vars(world_size, runner_settings)
    logger.info("runner_settings is: %s", runner_settings)
    run_pangu_embedded(runner_settings)
    logger.info("model run success")
