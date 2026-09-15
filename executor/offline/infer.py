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
import logging
import yaml
from executor.core.config import InferenceConfig, SpeculativeConfig
from executor.offline.offline_inference import OfflineInference
from executor.utils.data_utils import generate_default_prompt, load_longbench_dataset, build_dataset_input, \
    load_infinitebench_dataset, load_mmmu_dataset, export_mmmu_results
from executor.utils.common_utils import process_infer_time
from executor.utils.logging_config import setup_logging
from executor.core.forward_data_info import SamplingParams

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="llm run parameters")
    parser.add_argument('--yaml_file_path', type=str, required=True, help="inference configurations")
    return parser.parse_args()


def generate_prompt(dataset, dataset_path):
    if dataset in ("default", "default_multimodal"):
        prompt_filename = f"{dataset}_prompt.json"
        preset_prompts = generate_default_prompt(dataset_path, prompt_filename)
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
        raise Exception(
            f"your dataset {dataset} is not supported, dataset supported: "
            "default, default_multimodal, LongBench, InfiniteBench, MMMU"
        )
    return preset_prompts


def preprocess_prompts_for_scheduler(prompts, tokenizer, scheduler_config, data_config):
    bsz = scheduler_config.batch_size
    prompts = prompts * (bsz // len(prompts) + 1)
    prompts = prompts[:bsz]
    return build_dataset_input(tokenizer, prompts, data_config.input_truncated_len,
                               scheduler_config.max_new_tokens, is_chat=True)


def _get_prompt_dp_rank(config, global_rank):
    parallel_config = config.parallel_config
    if config.model_config.custom_params.get("enable_afd", False):
        ffn_world_size = parallel_config.world_size // 2
        global_rank = global_rank % ffn_world_size

    return global_rank // parallel_config.attn_tp_size


def log_results(results, speculative_stats, infer_time, model_name, speculative_config: SpeculativeConfig,
                *, enable_mm_encode=False):
    """Log inference results and calculate draft acceptance rate when enabled.

    Args:
        results: List of GenerationOutput objects containing output_text.
        speculative_stats: Draft metrics (None if speculative inference is disabled).
                   Contain 'spec_num_accepted_tokens' and 'spec_num_forward_ct'.
        infer_time: [prefill, decode...] or [encode, prefill, decode...] when MM Encode is enabled.
        model_name: Name of the model used for logging.
        speculative_config: Proposal width and backend used for speculative statistics.
        enable_mm_encode: Whether infer_time includes an Encode entry before Prefill.
    """
    next_n = speculative_config.num_speculative_tokens
    speculative_method = speculative_config.method or "none"
    # Log output text for each request
    for i, res in enumerate(results):
        logger.info("Request %s: outputs: %s\n", i, res.output_text)

    if enable_mm_encode and infer_time:
        logger.info("%s encode total inference time cost is %.2f ms", model_name, infer_time[0] * 1000)
        logger.info("%s prefill total inference time cost is %.2f ms", model_name, infer_time[1] * 1000)
    decode_infer_time = infer_time[2:] if enable_mm_encode else infer_time[1:]
    # Calculate and log total draft acceptance rate if speculative inference is enabled
    if next_n > 0:
        spec_num_forward_ct = sum(speculative_stats['spec_num_forward_ct'])
        draft_token_counts = speculative_stats.get('spec_num_draft_tokens')
        total_spec_tokens = sum(draft_token_counts) if draft_token_counts is not None \
            else spec_num_forward_ct * next_n
        total_accept_tokens = sum(speculative_stats['spec_num_accepted_tokens'])
        valid_output_len = sum(speculative_stats['valid_output_len'])
        draft_label = {"mtp": "MTP", "dspark": "DSpark"}.get(
            speculative_method, speculative_method
        )

        avg_accept_rate = total_accept_tokens / total_spec_tokens if total_spec_tokens > 0 else 0.0
        avg_accept_length = total_accept_tokens / spec_num_forward_ct + 1 if spec_num_forward_ct > 0 else 1.0
        decode_execution_times = speculative_stats.get("decode_execution_time")
        if decode_execution_times is not None:
            total_decode_time = sum(decode_execution_times)
            avg_decode_time = total_decode_time / spec_num_forward_ct if spec_num_forward_ct > 0 else 0.0
        else:
            avg_decode_time = process_infer_time(decode_infer_time, len(decode_infer_time))
        avg_equivalent_time = avg_decode_time / avg_accept_length

        logger.info(f"Finished inference, the number of valid output tokens is {valid_output_len}, "
                    f"total number of draft tokens is {total_spec_tokens}, "
                    f"total accepted number is {total_accept_tokens}")
        logger.info(
            f"{model_name} main and {draft_label} model average inference time cost "
            f"is {avg_decode_time*1000:.2f} ms")
        logger.info(
            f"{model_name} model average equivalent latency of {draft_label}{next_n}"
            f" is {avg_equivalent_time*1000:.2f} ms")
        logger.info("The speculation accept length: %.4f", avg_accept_length)
        logger.info("The speculation accept rate: %.4f", avg_accept_rate)
        decode_output_counts = speculative_stats.get("decode_output_tokens")
        if decode_execution_times is not None and decode_output_counts is not None:
            total_decode_outputs = sum(decode_output_counts)
            if total_decode_outputs > 0:
                logger.info(
                    "Decode model execution time per delivered token (excluding scheduling): %.2f ms",
                    total_decode_time / total_decode_outputs * 1000,
                )
    else:
        avg_decode_time = process_infer_time(decode_infer_time, len(decode_infer_time))
        logger.info(
            "%s decode average inference time cost is %.2f ms",
            model_name,
            avg_decode_time * 1000,
        )


def main():
    setup_logging()
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    rank_offset = int(os.getenv("RANK_OFFSET", "0"))
    global_rank = local_rank + rank_offset

    args = parse_args()
    with open(args.yaml_file_path, 'r') as f:
        yaml_dict = yaml.safe_load(f)
    config = InferenceConfig.from_dict(yaml_dict, global_rank=global_rank, local_rank=local_rank)
    if config.model_config.output_path == "":
        config.model_config.output_path = os.path.join(os.getenv("WORK_DIR", "."), os.getenv("RES_PATH", ""))
    logger.info("Inference Configuration")
    logger.info(config)

    dataset_path = os.path.join(os.path.dirname(__file__), "../../dataset")
    if config.data_config.dataset_path != "":
        dataset_path = config.data_config.dataset_path

    attn_dp_size = config.parallel_config.attn_dp_size
    cp_size = config.parallel_config.cp_size
    batch_size = config.scheduler_config.batch_size
    mmmu_ids = None
    mmmu_ground_truths = None
    if config.data_config.dataset == "MMMU":
        if cp_size>1:
                raise ValueError(
                "MMMU dataset does not support cp_size>1"
            )
        mmmu_path = os.path.abspath(os.path.join(dataset_path, "MMMU"))
        if os.path.isdir(mmmu_path):
            mmmu_source = mmmu_path
        else:
            mmmu_source = "MMMU/MMMU"
        prompts, mmmu_ids, mmmu_ground_truths = load_mmmu_dataset(mmmu_source, batch_size)
        global_dp_rank = 0
        if attn_dp_size > 1:
            if batch_size % attn_dp_size !=0:
                raise ValueError(
                f"batch_size ({batch_size}) must be divisible by attn_dp_size ({attn_dp_size})"
            )
            per_rank = config.scheduler_config.batch_size_per_dp_rank
            global_dp_rank = _get_prompt_dp_rank(config, global_rank)
            lo = global_dp_rank * per_rank
            hi = (global_dp_rank+1)* per_rank
            prompts =prompts[lo:hi]
            mmmu_ids = mmmu_ids[lo:hi]
            mmmu_ground_truths =mmmu_ground_truths[lo:hi]
    elif cp_size > 1:
        if batch_size % attn_dp_size != 0:
            raise ValueError(f"batch_size ({batch_size}) must be divisible by attn_dp_size ({attn_dp_size})")
        all_prompts = generate_prompt(config.data_config.dataset, dataset_path)
        all_prompts = all_prompts * (batch_size // len(all_prompts) + 1)
        prompts = all_prompts[:batch_size]
    elif attn_dp_size > 1:
        if batch_size % attn_dp_size != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by attn_dp_size ({attn_dp_size})"
            )
        batch_size_per_rank = batch_size // attn_dp_size
        global_dp_rank = _get_prompt_dp_rank(config, global_rank)
        all_prompts = generate_prompt(config.data_config.dataset, dataset_path)
        all_prompts = all_prompts * (batch_size // len(all_prompts) + 1)
        prompts = all_prompts[
            global_dp_rank * batch_size_per_rank:(global_dp_rank + 1) * batch_size_per_rank
        ]
    else:
        prompts = generate_prompt(config.data_config.dataset, dataset_path)

    llm = OfflineInference(config)

    if config.data_config.dataset not in ("default", "default_multimodal", "MMMU"):
        prompts = preprocess_prompts_for_scheduler(
            prompts, llm.engine.tokenizer, config.scheduler_config, config.data_config)
    sampling_params = SamplingParams(
        temperature=config.data_config.temperature,
        top_p=config.data_config.top_p,
        top_k=config.data_config.top_k,
        seed=config.data_config.seed,
        top_logprobs=0,
        logprobs=False
    )
    results, speculative_stats, infer_time = llm.generate(prompts=prompts, sampling_params=sampling_params)
    if llm.engine.is_afd_ffn_rank:
        return
    log_results(
        results,
        speculative_stats,
        infer_time,
        llm.engine.main_worker.model_name,
        config.speculative_config,
        enable_mm_encode=llm.engine.enable_mm_encode,
    )

    if config.data_config.dataset == "MMMU" and mmmu_ids is not None:
        suffix = f"_dp{global_dp_rank}" if attn_dp_size>1 else ""
        export_mmmu_results(results, mmmu_ids, mmmu_ground_truths,config.model_config.output_path, suffix=suffix)

if __name__ == "__main__":
    main()
