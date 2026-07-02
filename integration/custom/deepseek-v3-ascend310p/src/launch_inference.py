# coding=utf-8
# Copyright (c) 2025 QINGMAO INTELLIGENCE TECHNOLOGY (BEIJING) CO., LTD. and Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import json
import signal
import subprocess
import sys
import yaml
from http.server import BaseHTTPRequestHandler, HTTPServer
import random
import socket
import multiprocessing as mp
from transformers import PreTrainedTokenizer
from transformers import AutoTokenizer
from typing import List
import json
from subprocess import PIPE
import time

class InputPreprocessor:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def _token_prompt(self, prompt: str):
        return self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)

def run(config, prompts: List[str]):
    devices = ','.join([str(i) for i in config["devices"]])
    os.environ["ASCEND_VISIBLE_DEVICES"] = devices

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    inputpreprocess = InputPreprocessor(tokenizer)

    token_ids = []
    for prompt in prompts:
        try:
            tokens = inputpreprocess._token_prompt(prompt)
            token_ids.append(tokens)
        except Exception as e:
            print(f"Failed to process prompt '{prompt[:50]}...': {str(e)}", file=sys.stderr)
            token_ids.append([])

    json_data = json.dumps(token_ids)

    try:
        dp_size = str(config["data_parallel"])
    except KeyError:
        dp_size = "1"

    command = [
        'mpiexec', '--allow-run-as-root', '--np', str(config["world_size"]),
        '-x', 'MILLM_LOG_FIRST_RANK_ONLY=ON',
        '-x', 'MILLM_LOG_LEVEL=DEBUG',
        '-x', 'ASCEND_VISIBLE_DEVICES',
        '-x', 'LD_LIBRARY_PATH',
        '-x', 'ASCEND_AICPU_PATH',
        '-x', 'ASCEND_OPP_PATH',
        '-x', 'ASCEND_HOME_PATH',
        '-x', 'HCCL_BUFFSIZE=1024',
        os.path.abspath(config["exe"]),
        '--model_path', config["model_path"],
        '--max_num_batched_tokens', str(config["max_num_batched_tokens"]),
        '--block_size', str(config["block_size"]),
        '--tp', str(config["tensor_parallel"]),
        '--ep', str(config["expert_parallel"]),
        '--pp', str(config["pipeline_parallel"]),
        '--dp', dp_size,
        '--num_gpu_blocks', str(config['num_gpu_blocks']),
        '--world_size', str(config["world_size"]),
        '--token_ids', json_data,
        '--max_tokens', str(config["max_tokens"]),
    ]

    process = subprocess.Popen(
        command,
        stdout=sys.stdout,
        stderr=sys.stderr,
        start_new_session=False
    )
    def signal_handler(sig, frame):
        process.send_signal(signal.SIGINT)
        process.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    process.wait()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_ids_path = os.path.join(current_dir, "outputIds.txt")
    # wait for outputIDs file
    while not os.path.exists(output_ids_path):
        time.sleep(0.1)

    with open(output_ids_path, 'r') as f:
        output_ids = json.load(f)

    def detokenize_output(output_ids) -> list[str]:
        decoded_texts = []
        for ids in output_ids:
            text = tokenizer.decode(ids, skip_special_tokens=True)
            decoded_texts.append(text)
        return decoded_texts

    # Detokenize
    decoded_texts = detokenize_output(output_ids)
    for decoded_text in decoded_texts:
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("decoded_texts: ", decoded_text)


def main():
    parser = argparse.ArgumentParser(description="Launch the LLM inference backend")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--data_path", type=str, default="", help="Path to the load dataset")
    parser.add_argument("--batchSize", type=int, default=-1, help="The batch Size for sigle inference")
    parser.add_argument("--input-text", type=str, nargs="+", default=[])

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    try:
        dp_size = config["data_parallel"]
    except KeyError:
        dp_size = 1

    if dp_size > 1:
        assert config["tensor_parallel"] * dp_size == config["expert_parallel"]

    # get port
    num_process = len(config["devices"]) // config["world_size"]

    # prepare prompts
    assert not (args.batchSize > 0 and len(args.input_text) > 0), "args.batchSize and len(args.input_text) cannot both be greater than 0 at the same time."

    if (args.batchSize > 0):
        assert len(args.data_path), "DataSet path is Null"
        questions = []
        with open(args.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                questions.append(data['prompt'])

        prompts = questions[:args.batchSize]
    elif len(args.input_text) > 0:
        prompts = args.input_text
    else:
        assert 0, "args.batchSize must > 0 or len(args.input_text) > 0"

    print(prompts)

    # launch sub-process
    processes = []
    for i in range(num_process):
        p = mp.Process(
            target=run,
            args=(config, prompts)
        )
        p.start()
        processes.append(p)

if __name__ == "__main__":
    main()
