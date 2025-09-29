# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import json
import os
import logging
from datasets import load_dataset  # requires version == 3.6.0


def load_infinitebench_dataset(data_path):
    prompts = []
    datasets = ["longbook_qa_eng.jsonl"]
    data = load_dataset(data_path, data_files=datasets, split="train", trust_remote_code=True)
    for d in data:
        prompts.append(d['context'])
    return prompts


def load_longbench_dataset(data_path):
    prompts = []
    datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
    datasets_e = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", "trec", \
                  "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    datasets_e = [item + "_e" for item in datasets_e]

    for dataset in datasets + datasets_e:
        data = load_dataset(data_path, dataset, split='test', trust_remote_code=True)
        for d in data:
            prompts.append(d['context'])
    return prompts


def generate_default_prompt(dataset_dir):
    json_path = os.path.join(dataset_dir, "default_prompt.json")
    json_path = os.path.abspath(json_path)
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            text = data["text"]
    except FileNotFoundError as e:
        raise FileNotFoundError(f"prompt error: prompt file({json_path}) not find.") from e
    except json.JSONDecodeError as e:
        logging.error(f"prompt error: the json format of prompt file({json_path}) is incorrect.")
        raise e
    except Exception as e:
        raise e
    preset_prompts = [
        text,
    ]
    return preset_prompts


def get_prompts_for_cur_rank(preset_prompts, global_bs, batch_size_per_rank, global_dp_rank):
    preset_prompts = preset_prompts * (global_bs // len(preset_prompts) + 1)
    preset_prompts = preset_prompts[global_dp_rank * batch_size_per_rank: (global_dp_rank + 1) * batch_size_per_rank]
    query_id_list = list(range(global_dp_rank * batch_size_per_rank, (global_dp_rank + 1) * batch_size_per_rank))
    logging.info(f"prompt batch size: {len(preset_prompts)}/{global_bs}, {query_id_list=}")
    return (preset_prompts, query_id_list)


def generate_prompt(runner_settings, tp_size=None):
    batch_size = runner_settings.get("data_config").get("batch_size", 1)
    attn_tp_size = runner_settings.get("parallel_config").get("attn_tp_size")
    cp_size = runner_settings.get("parallel_config").get("cp_size", 1)
    global_rank = int(os.getenv("RANK_ID", 0))
    global_dp_rank = global_rank // cp_size // attn_tp_size
    bs_per_cp_group = runner_settings.get("data_config").get("bs_per_cp_group", 1)
    batch_size_per_rank = bs_per_cp_group if cp_size > 1 \
        else runner_settings.get("data_config").get("batch_size_per_rank", 1)
    dataset = runner_settings.get("data_config").get("dataset", "default")

    cur_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(cur_dir, "../../dataset")
    if dataset == "default":
        preset_prompts = generate_default_prompt(dataset_path)
    elif dataset == "LongBench":
        dataset_path = os.path.abspath(os.path.join(dataset_path, f"{dataset}"))
        if os.path.isdir(dataset_path): # use local LongBench dataset first
            dataset = dataset_path
        preset_prompts = load_longbench_dataset(dataset)
    elif dataset == "InfiniteBench":
        dataset_path = os.path.abspath(os.path.join(dataset_path, f"{dataset}"))
        if os.path.isdir(dataset_path): # use local InfiniteBench dataset first
            dataset = dataset_path
        preset_prompts = load_infinitebench_dataset(dataset)
    else:
        raise Exception(f"your dataset {dataset} is not supported, dataset supported: LongBench, InfiniteBench")
    return get_prompts_for_cur_rank(preset_prompts, batch_size, batch_size_per_rank, global_dp_rank)


def build_dataset_input(tokenizer, prompts, input_max_len):
    prompts_inputids = tokenizer(prompts).input_ids
    out_prompts = []
    for prompt_inputids in prompts_inputids:
        prompt = "Please read a part of the book below, and then give me the summary.\n[start of the book]\n" + \
            tokenizer.decode(prompt_inputids[:input_max_len - 70], skip_special_tokens=True) + \
            "\n[end of the book]\n\nNow you have read it. Please summarize it for me. " + \
            "First, tell me the title and the author, and then tell the story in 400 words.\n\n"
        out_prompts.append(prompt)
    return out_prompts