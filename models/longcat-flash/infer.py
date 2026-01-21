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
import sys
import argparse
import logging
import torch

CUR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.realpath(os.path.join(CUR_DIR, ".."))
sys.path.append(ROOT_DIR)
from runner_longcat_flash import LongcatFlashRunner
from models.mtp import InferMTP
from models.model_setting import check_is_attn_rank, check_and_update_vars
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
    parser.add_argument('--local_rank', type=int, default=0, help="Local rank id for torch distributed launch")
    parser.add_argument('--yaml_file_path', type=str, help="inference configurations")
    parser_args = parser.parse_args()
    return parser_args


def run_longcat_flash(runner_settings):
    attn_tp_size = runner_settings.get("parallel_config").get("attn_tp_size", 1)
    preset_prompts, query_id_list = generate_prompt(runner_settings)
    model_runner = LongcatFlashRunner(runner_settings)
    model_runner.init_model()
    # warmup
    model_runner.model_generate(preset_prompts, warm_up=True)
    # generate perf data
    model_runner.model_generate(preset_prompts)


def run_longcat_flash_mtp(runner_settings):
    preset_prompts, _ = generate_prompt(runner_settings)
    model_runner_main = LongcatFlashRunner(runner_settings)
    model_runner_mtp = LongcatFlashRunner(runner_settings)
    model_runner_main.init_model()
    model_runner_mtp.init_model(is_mtp=True)
    # the mtp modules share lm_head, rotary_emb with the main model
    model_runner_mtp.model.lm_head = model_runner_main.model.lm_head
    model_runner_mtp.model.rotary_emb = model_runner_main.model.model.rotary_emb

    # init mtp infer process
    infer_mtp = InferMTP(runner_settings, model_runner_main, model_runner_mtp)
    # warmup
    infer_mtp.model_generate_mtp(preset_prompts, warm_up=True)
    # generate perf data
    infer_mtp.model_generate_mtp(preset_prompts)


def run_longcat_flash_ffn(runner_settings):
    model_runner_ffn = LongcatFlashRunner(runner_settings)
    model_runner_ffn.init_model()
    # warmup
    model_runner_ffn.ffn_server(warm_up=True)
    # generate perf data
    model_runner_ffn.ffn_server()


if __name__ == "__main__":
    args = parse_args()
    yaml_file_path = args.yaml_file_path
    runner_settings = read_yaml(yaml_file_path)

    check_and_update_vars(runner_settings)
    is_attn_rank = check_is_attn_rank(runner_settings)
    logging.info(f"runner_settings is: {runner_settings}")

    next_n = runner_settings.get("model_config").get("next_n", 0)
    if next_n > 0:
        run_longcat_flash_mtp(runner_settings) if is_attn_rank else run_longcat_flash_ffn(runner_settings)
    else:
        run_longcat_flash(runner_settings) if is_attn_rank else run_longcat_flash_ffn(runner_settings)
    logging.info("model run success")

