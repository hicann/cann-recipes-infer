# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import argparse
import logging
import os

import torch

from executor.utils import align_up, read_yaml, update_settings
from executor.utils.data_utils import generate_prompt
from models.configuration_dspark import KimiK3DSparkConfig
from models.configuration_kimi_k3 import KimiLinearConfig
from models.model_infer import KimiK3Infer
from runner_kimi_k3 import KimiK3DSparkRunner, KimiK3Runner


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - [LLM](%(filename)s:%(lineno)d): %(message)s",
    level=logging.INFO,
)
torch.manual_seed(42)
torch.npu.manual_seed_all(42)


def parse_args():
    parser = argparse.ArgumentParser(description="Kimi K3 offline inference")
    parser.add_argument("--yaml_file_path", required=True)
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_args()


def check_settings(world_size, settings):
    parallel = settings.get("parallel_config", {})
    model = settings.get("model_config", {})
    data = settings.get("data_config", {})
    attn_tp = parallel.get("attn_tp_size", 1)
    if attn_tp <= 0:
        raise ValueError("attn_tp_size must be greater than 0")
    if world_size <= 0 or world_size % attn_tp:
        raise ValueError(f"world_size={world_size} must be divisible by attn_tp_size={attn_tp}")
    if attn_tp not in (1, world_size):
        raise ValueError("Kimi K3 supports attention TP over the full world only")
    if parallel.get("moe_tp_size", 1) != 1:
        raise ValueError("Kimi K3 requires moe_tp_size=1")
    for label in ("dense_tp_size", "embed_tp_size", "lmhead_tp_size", "oproj_tp_size"):
        size = parallel.get(label, 1)
        if size <= 0 or world_size % size:
            raise ValueError(f"world_size={world_size} must be divisible by {label}={size}")
    if parallel.get("oproj_tp_size", attn_tp) != attn_tp:
        raise ValueError("Kimi K3 requires oproj_tp_size=attn_tp_size")
    batch_size = data.get("batch_size", 1)
    attn_dp = world_size // attn_tp
    if batch_size <= 0 or batch_size % attn_dp:
        raise ValueError("batch_size must be divisible by attention DP size")
    batch_size_per_rank = batch_size // attn_dp
    if batch_size_per_rank % attn_tp:
        raise ValueError("batch_size_per_rank must be divisible by attn_tp_size")
    model_path = settings.get("model_path")
    if not model_path:
        raise ValueError("model_path must be set")
    if world_size > 1 and not model.get("enable_online_split_weight", False):
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        rank_offset = int(os.getenv("RANK_OFFSET", "0"))
        model_path = os.path.join(model_path, f"rank_{local_rank + rank_offset}")
    target_config = KimiLinearConfig.from_pretrained(
        model_path, runner_settings=settings
    )
    for label in ("embed_tp_size", "lmhead_tp_size"):
        size = parallel.get(label, 1)
        if target_config.vocab_size % size:
            raise ValueError(
                f"vocab_size={target_config.vocab_size} must be divisible by "
                f"{label}={size}"
            )
    if target_config.num_attention_heads % attn_tp:
        raise ValueError(
            f"num_attention_heads={target_config.num_attention_heads} must be "
            f"divisible by attn_tp_size={attn_tp}"
        )
    kda_num_heads = target_config.linear_attn_config["num_heads"]
    if kda_num_heads % attn_tp:
        raise ValueError(
            f"KDA num_heads={kda_num_heads} must be divisible by "
            f"attn_tp_size={attn_tp}"
        )
    oproj_tp = parallel.get("oproj_tp_size", 1)
    mla_output_width = target_config.num_attention_heads * target_config.v_head_dim
    if mla_output_width % oproj_tp:
        raise ValueError(
            f"MLA output width={mla_output_width} must be divisible by "
            f"oproj_tp_size={oproj_tp}"
        )
    if target_config.num_experts % world_size:
        raise ValueError(
            f"num_experts={target_config.num_experts} must be divisible by "
            f"moe_ep_size={world_size}"
        )
    prefill_mini_batch_size = model.get("prefill_mini_batch_size", 0)
    if prefill_mini_batch_size < 0:
        raise ValueError("prefill_mini_batch_size must be greater than or equal to 0")
    if prefill_mini_batch_size > 0 and (
        prefill_mini_batch_size > batch_size_per_rank
        or batch_size_per_rank % prefill_mini_batch_size
    ):
        raise ValueError(
            f"batch_size_per_rank={batch_size_per_rank} must be divisible by "
            f"prefill_mini_batch_size={prefill_mini_batch_size}"
        )
    if not isinstance(model.get("skip_warm_up", True), bool):
        raise ValueError("skip_warm_up must be a boolean")
    max_new_tokens = data.get("max_new_tokens", 128)
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than 0")
    if data.get("temperature", 1.0) < 0:
        raise ValueError("temperature must be greater than or equal to 0")
    draft_model_type = model.get("draft_model_type", "none")
    next_n = model.get("next_n", 0)
    if draft_model_type not in ("none", "dspark"):
        raise ValueError("draft_model_type must be 'none' or 'dspark'")
    if draft_model_type == "none" and next_n != 0:
        raise ValueError("next_n must be 0 when draft_model_type is none")
    if draft_model_type == "dspark":
        draft_model_path = settings.get("draft_model_path")
        if not draft_model_path:
            raise ValueError("draft_model_path must be set when DSpark is enabled")
        if not 1 <= next_n <= 16:
            raise ValueError("DSpark GQA requires next_n in [1, 16]")
        if model.get("pa_block_size", 128) not in (16, 128):
            raise ValueError("DSpark GQA requires pa_block_size 16 or 128")
        draft_config = KimiK3DSparkConfig.from_pretrained(draft_model_path)
        if draft_config.block_size != next_n:
            raise ValueError("DSpark checkpoint block_size must equal next_n")
        if draft_config.num_attention_heads != 64:
            raise ValueError("DSpark GQA requires num_attention_heads=64")
        if draft_config.num_key_value_heads != 16:
            raise ValueError("DSpark GQA requires num_key_value_heads=16")
        if draft_config.head_dim != 64:
            raise ValueError("DSpark GQA requires head_dim=64")
        if batch_size_per_rank % 8:
            raise ValueError("batch_size_per_rank must be divisible by DSpark tp_size=8")
        if draft_config.intermediate_size % 8:
            raise ValueError("DSpark intermediate_size must be divisible by tp_size=8")
        if draft_config.vocab_size % 8:
            raise ValueError("DSpark vocab_size must be divisible by tp_size=8")
        if draft_config.vocab_size != target_config.vocab_size:
            raise ValueError("draft and target vocab_size must match")
    if parallel.get("cp_size", 1) != 1:
        raise ValueError("Kimi K3 offline mode does not support context parallel")
    block_size = model.get("pa_block_size", 128)
    if block_size % 16:
        raise ValueError("pa_block_size must be divisible by 16 for the NZ MLA cache")
    if block_size < attn_tp:
        raise ValueError("pa_block_size must be at least attn_tp_size")
    if settings.get("exe_mode") not in ("eager", "ge_graph", "npugraph_ex"):
        raise ValueError("exe_mode must be eager, ge_graph, or npugraph_ex")


def update_vars(world_size, settings):
    parallel = settings.get("parallel_config", {})
    data = settings.get("data_config", {})
    model = settings.get("model_config", {})
    attn_tp = parallel.get("attn_tp_size", 1)
    attn_dp = world_size // attn_tp
    moe_ep = world_size // parallel.get("moe_tp_size", 1)
    batch_size = data.get("batch_size", 1)
    settings = update_settings(settings, "parallel_config", "attn_dp_size", attn_dp)
    settings = update_settings(settings, "parallel_config", "moe_dp_size", moe_ep)
    settings = update_settings(settings, "parallel_config", "moe_ep_size", moe_ep)
    settings = update_settings(
        settings,
        "parallel_config",
        "embed_dp_size",
        world_size // parallel.get("embed_tp_size", 1),
    )
    batch_size_per_rank = batch_size // attn_dp
    settings = update_settings(
        settings, "data_config", "batch_size_per_rank", batch_size_per_rank
    )
    settings = update_settings(
        settings,
        "data_config",
        "mla_batch_per_rank",
        batch_size_per_rank // attn_tp,
    )
    max_total_len = (
        data.get("input_max_len", 128)
        + data.get("max_new_tokens", 128)
        + model.get("next_n", 0)
    )
    block_size = model.get("pa_block_size", 128)
    settings = update_settings(settings, "model_config", "pa_max_length", align_up(max_total_len, block_size))
    settings = update_settings(settings, "data_config", "max_position_embeddings", max_total_len)
    return settings


def run_kimi_k3(settings):
    prompts, _ = generate_prompt(settings)
    runner = KimiK3Runner(settings)
    torch.npu.set_compile_mode(jit_compile=False)
    runner.init_model()
    draft_runner = None
    if settings.get("model_config", {}).get("draft_model_type", "none") == "dspark":
        draft_runner = KimiK3DSparkRunner(settings, runner)
        draft_runner.init_model()
    infer = KimiK3Infer(settings, runner, draft_runner)
    cache_data = None
    draft_cache_data = None
    if settings.get("model_config", {}).get("skip_warm_up", True):
        logging.warning(
            "Warm-up is disabled; the first formal inference includes graph "
            "compilation and NPU operator cold-start overhead"
        )
    else:
        warmup_state = infer.model_generate(prompts, warm_up=True)
        cache_data = warmup_state["cache_data"]
        draft_cache_data = warmup_state.get("draft_cache_data")
        infer.cache_manager.reset_cache(cache_data)
        if draft_cache_data is not None:
            infer.cache_manager.reset_cache(draft_cache_data)
    infer.model_generate(
        prompts,
        cache_data=cache_data,
        draft_cache_data=draft_cache_data,
        warm_up=False,
    )


if __name__ == "__main__":
    args = parse_args()
    runner_settings = read_yaml(args.yaml_file_path)
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    check_settings(world_size, runner_settings)
    runner_settings = update_vars(world_size, runner_settings)
    logging.info("runner_settings is: %s", runner_settings)
    run_kimi_k3(runner_settings)
    logging.info("model run success")
