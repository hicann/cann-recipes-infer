# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import json
import os
import logging
import math
import glob
from pathlib import Path
from urllib.parse import urlparse
from datasets import load_dataset  # requires version == 3.6.0

logger = logging.getLogger(__name__)

def load_infinitebench_dataset(data_path):
    prompts = []
    datasets = ["longbook_qa_eng.jsonl"]
    data = load_dataset(data_path, data_files=datasets, split="train", trust_remote_code=True)
    for d in data:
        prompts.append(d['context'])
    return prompts

MMMU_SUBJECTS = (
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art",
    "Art_Theory", "Basic_Medical_Science", "Biology", "Chemistry",
    "Clinical_Medicine", "Computer_Science", "Design",
    "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics",
    "Energy_and_Power", "Finance", "Geography", "History", "Literature",
    "Manage", "Marketing", "Materials", "Math", "Mechanical_Engineering",
    "Music", "Pharmacy", "Physics", "Psychology", "Public_Health", "Sociology",
)

def load_mmmu_dataset(data_path, max_samples=0):
    from PIL import Image
    import io
    is_local = os.path.isdir(data_path)
    tmp_image_dir = (os.path.join(data_path, ".tmp_images") if is_local
                     else os.path.join(os.getcwd(),".mmmu_tmp_images"))
    os.makedirs(tmp_image_dir,exist_ok=True)

    prompts = []
    sample_ids=[]
    ground_truths=[]
    skipped_subjects=[]
    for subject in MMMU_SUBJECTS:
        try:
            if is_local:
                subject_dir = os.path.join(data_path, subject)
                validation_files = sorted(
                    glob.glob(
                        os.path.join(
                            subject_dir,
                            "validation-*.parquet",
                        )
                    )
                )

                if not validation_files:
                    raise FileNotFoundError(
                        f"validation parquet not found: {subject_dir}"
                    )

                data = load_dataset(
                    "parquet",
                    data_files=validation_files,
                    split="train",
                )
            else:
                data = load_dataset(
                    data_path,
                    subject,
                    split="validation",
                    trust_remote_code=True,
                )
        except Exception as e:
            logger.warning("MMMU: skip subject %s (validation split failed to load: %s)", subject, e)
            skipped_subjects.append(subject)
            continue
        for d in data:
            sample_id = str(d["id"])
            question = d["question"]
            options = d.get("options") or ""

            if isinstance(options, list):
                options = "\n".join(
                    f"({chr(65 + index)}) {option}"
                    for index, option in enumerate(options)
                )
            answer = d.get("answer") or ""
            image_urls=[]
            for i in range(1,8):
                img=d.get(f"image_{i}")
                if img is None:
                    continue
                if isinstance(img, dict):
                    if img.get("bytes") is not None:
                        img = Image.open(io.BytesIO(img["bytes"]))
                    elif img.get("path"):
                        img = Image.open(img["path"])
                    else:
                        raise ValueError(f"image_{i} of {sample_id} has neither bytes nor path")
                img_path = os.path.join(tmp_image_dir,f"{sample_id}_img{i}.png")
                img.save(img_path)
                image_urls.append(Path(img_path).resolve().as_uri())
            text = question + (f"\n{options}" if options.strip() else "")
            content= [{"type":"image_url", "image_url":{"url":url}} for url in image_urls]
            content.append({"type": "text","text":text})
            prompts.append([{"role": "user","content":content}])
            sample_ids.append(sample_id)
            ground_truths.append(answer)
            if max_samples > 0 and len(prompts)>= max_samples:
                break
        if max_samples > 0 and len(prompts)>= max_samples:
            break
    if skipped_subjects:
        logger.warning(
            "MMMU: skipped %d/%d subjects: %s",
            len(skipped_subjects),
            len(MMMU_SUBJECTS),
            ", ".join(skipped_subjects),
        )
    loaded_subject_count = len(MMMU_SUBJECTS) - len(skipped_subjects)
    logger.info(
        "MMMU: loaded %d samples from %d subjects; temporary images: %s",
        len(prompts),
        loaded_subject_count,
        tmp_image_dir,
    )
    if not prompts:
        failed_subjects = ", ".join(skipped_subjects) or "none"
        raise ValueError(
            f"MMMU: no samples loaded from {data_path!r}; "
            f"{len(skipped_subjects)}/{len(MMMU_SUBJECTS)} subjects failed "
            f"({failed_subjects})"
        )
    return prompts,sample_ids,ground_truths
 
def export_mmmu_results(results, sample_ids, ground_truths, output_path, suffix=""):
    """Write MMMU predictions and references as a JSON array."""
    mmmu_results=[]
    for result, sid, gt in zip(results, sample_ids,ground_truths):
        answer = result.output_text if result.output_text else ""
        mmmu_results.append({"id": sid, "answer": answer, "ground_truth": gt})
    res_file = os.path.join(output_path, f"mmmu_results{suffix}.json")
    with open(res_file, 'w', encoding='utf-8') as f:
        json.dump(mmmu_results, f, ensure_ascii=False)
    return res_file


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


def _resolve_relative_image_urls(value, prompt_dir):
    if isinstance(value, list):
        for item in value:
            _resolve_relative_image_urls(item, prompt_dir)
        return
    if not isinstance(value, dict):
        return

    if value.get("type") == "image_url":
        image_url = value.get("image_url")
        if isinstance(image_url, dict):
            url = image_url.get("url")
            if isinstance(url, str) and not urlparse(url).scheme:
                image_url["url"] = (prompt_dir / url).resolve().as_uri()

    for item in value.values():
        _resolve_relative_image_urls(item, prompt_dir)


def generate_default_prompt(dataset_dir, prompt_filename="default_prompt.json"):
    json_path = os.path.join(dataset_dir, prompt_filename)
    json_path = os.path.abspath(json_path)
    try:
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            text = data["text"]
    except FileNotFoundError as e:
        raise FileNotFoundError(f"prompt error: prompt file({json_path}) not find.") from e
    except json.JSONDecodeError as e:
        logger.error(f"prompt error: the json format of prompt file({json_path}) is incorrect.")
        raise e
    except Exception as e:
        raise e
    _resolve_relative_image_urls(text, Path(json_path).parent)
    if isinstance(text, list):
        preset_prompts = text
    else:
        preset_prompts = [text]
    return preset_prompts


def get_prompts_for_cur_rank(preset_prompts, global_bs, batch_size_per_rank, global_dp_rank):
    preset_prompts = preset_prompts * (global_bs // len(preset_prompts) + 1)
    preset_prompts = preset_prompts[global_dp_rank * batch_size_per_rank: (global_dp_rank + 1) * batch_size_per_rank]
    query_id_list = list(range(global_dp_rank * batch_size_per_rank, (global_dp_rank + 1) * batch_size_per_rank))
    logger.info(f"prompt batch size: {len(preset_prompts)}/{global_bs}, {query_id_list=}")
    return (preset_prompts, query_id_list)


def generate_prompt(runner_settings):
    batch_size = runner_settings.get("data_config").get("batch_size", 1)
    attn_tp_size = runner_settings.get("parallel_config").get("attn_tp_size", 1)
    cp_size = runner_settings.get("parallel_config").get("cp_size", 1)
    global_rank = int(os.getenv("RANK_ID", 0))
    bs_per_cp_group = runner_settings.get("data_config").get("bs_per_cp_group", 1)
    dataset = runner_settings.get("data_config").get("dataset", "default")
    kvp_size = runner_settings.get("parallel_config").get("kvp_size", 1)
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    global_dp_rank = global_rank // max(cp_size, kvp_size) // attn_tp_size
    if cp_size > 1:
        batch_size_per_rank = bs_per_cp_group
    else:
        batch_size_per_rank = runner_settings.get("data_config").get("batch_size_per_rank", 1)

    cur_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(cur_dir, "../../dataset")
    if dataset == "default":
        preset_prompts = generate_default_prompt(dataset_path)
    elif dataset == "LongBench":
        dataset_path = os.path.abspath(os.path.join(dataset_path, f"{dataset}"))
        if os.path.isdir(dataset_path): # use local LongBench dataset first
            dataset = dataset_path
        else:
            dataset = "THUDM/LongBench"
        preset_prompts = load_longbench_dataset(dataset)
    elif dataset == "InfiniteBench":
        dataset_path = os.path.abspath(os.path.join(dataset_path, f"{dataset}"))
        if os.path.isdir(dataset_path): # use local InfiniteBench dataset first
            dataset = dataset_path
        preset_prompts = load_infinitebench_dataset(dataset)
    else:
        raise Exception(f"your dataset {dataset} is not supported, dataset supported: LongBench, InfiniteBench")
    return get_prompts_for_cur_rank(preset_prompts, batch_size, batch_size_per_rank, global_dp_rank)


def tokenizer_in_loop(tokenizer, prompts, input_max_len=32, section_size=1024):
    total_num = len(prompts)
    if section_size <= 0:
        raise ValueError(f"section_size must be positive, but got {section_size}.")
    section_num = math.ceil(total_num / section_size)
    input_ids_list = []
    for i in range(section_num):
        start_index = i * section_size
        end_index = (i + 1) * section_size
        each_prompts = prompts[start_index: end_index]
        prompts_input_ids = tokenizer(each_prompts, truncation=True, max_length=input_max_len,
                                      return_attention_mask=False).input_ids
        input_ids_list.extend(prompts_input_ids)
        logger.info(f"{len(input_ids_list)} of {total_num} prompts have been tokenized...")
    return input_ids_list


def build_dataset_input(tokenizer, prompts, input_max_len, max_new_tokens=32, is_chat=False):
    # Provide system prompt for the text; the default is aritcle continuation, which can be modified as needed.
    prefix = "Please read a part of the book below, and then give me the summary.\n[start of the book]\n"
    suffix = "[end of the book]\n\n" + \
            "Now you have read it. Please summarize it for me. " + \
            f"First, tell me the title and the author, and then tell the story in {max_new_tokens} words.\n\n "
    if is_chat:
        system_prompt_chat = [{"role": "user", "content": prefix + suffix}]
        system_prompt_len = len(tokenizer.apply_chat_template(
                                system_prompt_chat,
                                add_generation_prompt=True,
                                return_dict=False,
                                tokenize=True
                            ))
    else:
        system_prompt_len = len(tokenizer(prefix + suffix).input_ids)
    if system_prompt_len > input_max_len:
        logger.info("The parameter 'input_max_len' should be greater than the length of system prompt. " + \
         "Please modify the input_max_len in the YAML file or modify system prompt in executor/utils/data_utils.py.")

    # use tokenizer loop to avoid host oom when query num is large
    input_ids_list = tokenizer_in_loop(tokenizer, prompts, input_max_len)

    out_prompts = []
    for prompt_input_ids in input_ids_list:
        prompt = prefix + \
            tokenizer.decode(prompt_input_ids[:input_max_len - system_prompt_len], skip_special_tokens=True) + \
            suffix
        out_prompts.append(prompt)
    return out_prompts
