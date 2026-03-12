# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
import yaml
from executor.core import InferenceConfig, OfflineInference
from executor.utils.data_utils import generate_default_prompt, load_longbench_dataset, build_dataset_input


def parse_args():
    parser = argparse.ArgumentParser(description="llm run parameters")
    parser.add_argument('--yaml_file_path', type=str, required=True, help="inference configurations")
    return parser.parse_args()

def generate_prompt(dataset, dataset_path):
    if dataset == "default":
        preset_prompts = generate_default_prompt(dataset_path)
    elif dataset == "LongBench":
        dataset_path = os.path.abspath(os.path.join(dataset_path, f"{dataset}"))
        if os.path.isdir(dataset_path): # use local LongBench dataset first
            dataset = dataset_path
        else:
            dataset = "THUDM/LongBench"
        preset_prompts = load_longbench_dataset(dataset)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return preset_prompts

def preprocess_prompts_for_scheduler(prompts, tokenizer, scheduler_config):
    bsz = scheduler_config.batch_size
    prompts = prompts * (bsz // len(prompts) + 1)
    prompts = prompts[:bsz]
    return build_dataset_input(tokenizer, prompts, scheduler_config.input_max_len,
                               scheduler_config.max_new_tokens, False)

def main():
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    rank_offset = int(os.getenv("RANK_OFFSET", "0"))
    global_rank = local_rank + rank_offset

    args = parse_args()
    with open(args.yaml_file_path, 'r') as f:
        yaml_dict = yaml.safe_load(f)
    config = InferenceConfig.from_dict(yaml_dict, global_rank=global_rank, local_rank=local_rank)
    if config.model_config.output_path == "":
        config.model_config.output_path = os.path.dirname(args.yaml_file_path)

    dataset_path = os.path.join(os.path.dirname(__file__), f"../dataset")
    if config.data_config.dataset_path != "":
        dataset_path = config.data_config.dataset_path

    attn_tp_size = config.parallel_config.attn_tp_size
    attn_dp_size = config.parallel_config.attn_dp_size
    batch_size = config.scheduler_config.batch_size

    if attn_dp_size > 1:
        if batch_size % attn_dp_size != 0:
            raise ValueError(f"batch_size ({batch_size}) must be divisible by attn_dp_size ({attn_dp_size})")
        batch_size_per_rank = batch_size // attn_dp_size
        # For BSH/BSND format, some modules within TP group split along batch dimension,
        # so each TP rank must process the same number of samples.
        # Note: TND format splits by token count, this validation does not apply.
        if batch_size_per_rank % attn_tp_size != 0:
            raise ValueError(
                f"batch_size_per_rank ({batch_size_per_rank}) "
                f"must be divisible by attn_tp_size ({attn_tp_size})"
            )
        global_dp_rank = global_rank // attn_tp_size
        all_prompts = generate_prompt(config.data_config.dataset, dataset_path)
        all_prompts = all_prompts * (batch_size // len(all_prompts) + 1)
        prompts = all_prompts[
            global_dp_rank * batch_size_per_rank:(global_dp_rank + 1) * batch_size_per_rank
        ]
    else:
        prompts = generate_prompt(config.data_config.dataset, dataset_path)

    llm = OfflineInference(config)

    if config.data_config.dataset != "default":
        prompts = preprocess_prompts_for_scheduler(prompts, llm.engine.tokenizer, config.scheduler_config)
    results = llm.generate(prompts)
    for res in results:
        print(f"outputs: {res.output_text}")


if __name__ == "__main__":
    main()
