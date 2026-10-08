# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import argparse
import logging
import math
import os

import torch

from executor.utils import align_up, read_yaml, update_settings
from executor.utils.data_utils import generate_prompt
from models.configuration_kimi_k3 import KimiLinearConfig
from models.dspark_registry import (
    DRAFT_MODEL_DSPARK_GQA,
    DRAFT_MODEL_DSPARK_MLA,
    DSPARK_DRAFT_MODEL_TYPES,
    get_draft_model_type,
    get_dspark_config_class,
)
from models.model_infer import KimiK3Infer
from models.modules.attention_data import validate_mega_kda_replayssm_switch
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
    skip_prefill = model.get("skip_prefill", False)
    if not isinstance(skip_prefill, bool):
        raise ValueError("skip_prefill must be a boolean")
    prefill_stub_token_id = model.get("prefill_stub_token_id")
    if prefill_stub_token_id is not None and (
        isinstance(prefill_stub_token_id, bool)
        or not isinstance(prefill_stub_token_id, int)
        or prefill_stub_token_id < 0
    ):
        raise ValueError("prefill_stub_token_id must be a non-negative integer")
    attn_tp = parallel.get("attn_tp_size", 1)
    if attn_tp <= 0:
        raise ValueError("attn_tp_size must be greater than 0")
    if world_size <= 0 or world_size % attn_tp:
        raise ValueError(f"world_size={world_size} must be divisible by attn_tp_size={attn_tp}")
    if parallel.get("moe_tp_size", 1) != 1:
        raise ValueError("Kimi K3 requires moe_tp_size=1")
    for label in ("dense_tp_size", "embed_tp_size", "lmhead_tp_size", "oproj_tp_size"):
        size = parallel.get(label, 1)
        if size <= 0 or world_size % size:
            raise ValueError(f"world_size={world_size} must be divisible by {label}={size}")
        if attn_tp % size:
            raise ValueError(
                f"attn_tp_size={attn_tp} must be divisible by {label}={size}"
            )
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
    if (
        prefill_stub_token_id is not None
        and prefill_stub_token_id >= target_config.vocab_size
    ):
        raise ValueError(
            "prefill_stub_token_id must be smaller than "
            f"vocab_size={target_config.vocab_size}"
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
    if model.get("prefill_chunk_size", 0) < 0:
        raise ValueError("prefill_chunk_size must be greater than or equal to 0")
    if not isinstance(model.get("skip_warm_up", True), bool):
        raise ValueError("skip_warm_up must be a boolean")
    if not isinstance(model.get("force_eplb", False), bool):
        raise ValueError("force_eplb must be a boolean")
    custom_params = model.get("custom_params") or {}
    for name in (
        "enable_prefill_mega_moe",
        "enable_multi_streams",
        "enable_superkernel",
        "enable_dspark_confidence_head",
        "enable_mega_kda",
        "enable_mega_kda_replayssm",
    ):
        if not isinstance(custom_params.get(name, False), bool):
            raise ValueError(f"{name} must be a boolean")
    max_new_tokens = data.get("max_new_tokens", 128)
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be greater than 0")
    if skip_prefill and max_new_tokens < 2:
        raise ValueError("skip_prefill requires max_new_tokens to be at least 2")
    if data.get("temperature", 1.0) < 0:
        raise ValueError("temperature must be greater than or equal to 0")
    draft_model_type = get_draft_model_type(settings)
    next_n = model.get("next_n", 0)
    if draft_model_type == "none" and next_n != 0:
        raise ValueError("next_n must be 0 when draft_model_type is none")
    enable_mega_kda_replayssm = validate_mega_kda_replayssm_switch(
        custom_params,
        draft_model_type,
        next_n,
    )
    if enable_mega_kda_replayssm:
        linear = target_config.linear_attn_config
        local_kda_heads = linear["num_heads"] // attn_tp
        if model.get("platform_version") != "950":
            raise ValueError(
                "enable_mega_kda_replayssm requires platform_version='950'"
            )
        if not 1 <= batch_size_per_rank <= 16:
            raise ValueError(
                "enable_mega_kda_replayssm requires batch_size_per_rank in [1, 16]"
            )
        if (
            target_config.hidden_size != 7168
            or local_kda_heads != 6
            or linear["head_dim"] != 128
            or linear["short_conv_kernel_size"] != 4
        ):
            raise ValueError(
                "enable_mega_kda_replayssm requires hidden_size=7168, six "
                "local KDA heads, head_dim=128 and ShortConv kernel=4"
            )
        if not linear.get("use_full_rank_gate", False):
            raise ValueError(
                "enable_mega_kda_replayssm requires a full-rank output gate"
            )
        rms_norm_eps = target_config.rms_norm_eps
        if (
            isinstance(rms_norm_eps, bool)
            or not isinstance(rms_norm_eps, (int, float))
            or not math.isfinite(rms_norm_eps)
            or rms_norm_eps <= 0
        ):
            raise ValueError(
                "enable_mega_kda_replayssm requires finite rms_norm_eps > 0"
            )
        lower_bound = linear.get("gate_lower_bound")
        if (
            isinstance(lower_bound, bool)
            or not isinstance(lower_bound, (int, float))
            or not math.isfinite(lower_bound)
            or not -5 <= lower_bound <= 0
        ):
            raise ValueError(
                "enable_mega_kda_replayssm requires gate_lower_bound in [-5, 0]"
            )
    if draft_model_type in DSPARK_DRAFT_MODEL_TYPES:
        dspark_tp = model.get("dspark_tp_size", 8)
        if dspark_tp <= 0 or world_size % dspark_tp:
            raise ValueError(
                f"world_size={world_size} must be divisible by "
                f"dspark_tp_size={dspark_tp}"
            )
        if attn_tp % dspark_tp:
            raise ValueError(
                f"attn_tp_size={attn_tp} must be divisible by "
                f"dspark_tp_size={dspark_tp}"
            )
        draft_model_path = settings.get("draft_model_path")
        if not draft_model_path:
            raise ValueError("draft_model_path must be set when DSpark is enabled")
        if not 1 <= next_n <= 16:
            raise ValueError("DSpark requires next_n in [1, 16]")
        if model.get("pa_block_size", 128) not in (16, 128):
            raise ValueError("DSpark requires pa_block_size 16 or 128")
        draft_config_cls = get_dspark_config_class(draft_model_type)
        draft_config = draft_config_cls.from_pretrained(draft_model_path)
        if (
            custom_params.get("enable_dspark_confidence_head", False)
            and not draft_config.enable_confidence_head
        ):
            raise ValueError(
                "enable_dspark_confidence_head requires checkpoint "
                "enable_confidence_head=True"
            )
        if draft_config.block_size != next_n:
            raise ValueError("DSpark checkpoint block_size must equal next_n")
        if draft_model_type == DRAFT_MODEL_DSPARK_GQA:
            if draft_config.num_attention_heads != 64:
                raise ValueError("DSpark GQA requires num_attention_heads=64")
            if draft_config.num_key_value_heads != 16:
                raise ValueError("DSpark GQA requires num_key_value_heads=16")
            if draft_config.head_dim != 64:
                raise ValueError("DSpark GQA requires head_dim=64")
        elif draft_model_type == DRAFT_MODEL_DSPARK_MLA:
            expected_mla_dims = {
                "num_attention_heads": 64,
                "num_key_value_heads": 64,
                "q_lora_rank": 1536,
                "kv_lora_rank": 512,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": 64,
                "v_head_dim": 128,
            }
            for name, expected in expected_mla_dims.items():
                actual = int(getattr(draft_config, name))
                if actual != expected:
                    raise ValueError(
                        f"DSpark MLA requires {name}={expected}, got {actual}"
                    )
            unsupported_mla_features = {
                "mla_use_nope": draft_config.mla_use_nope,
                "mla_use_output_gate": draft_config.mla_use_output_gate,
                "mla_use_qk_norm": draft_config.mla_use_qk_norm,
                "dspark_bonus_anchor": draft_config.dspark_bonus_anchor,
            }
            enabled_unsupported = [
                name
                for name, enabled in unsupported_mla_features.items()
                if enabled
            ]
            if enabled_unsupported:
                raise ValueError(
                    "DSpark MLA does not support "
                    + ", ".join(enabled_unsupported)
                )
            if draft_config.kv_lora_rank % 16 or draft_config.qk_rope_head_dim % 16:
                raise ValueError(
                    "DSpark MLA latent cache dimensions must be divisible by 16"
                )
            if draft_config.target_num_hidden_layers != target_config.num_hidden_layers:
                raise ValueError(
                    "DSpark MLA target_num_hidden_layers must match the target model"
                )
            if len(draft_config.target_layer_ids) != draft_config.num_target_layers:
                raise ValueError(
                    "DSpark MLA target_layer_ids must match num_target_layers"
                )
            if len(set(draft_config.target_layer_ids)) != draft_config.num_target_layers:
                raise ValueError("DSpark MLA target_layer_ids must be unique")
            if (
                not draft_config.target_layer_ids
                or min(draft_config.target_layer_ids) < 0
                or max(draft_config.target_layer_ids)
                >= draft_config.target_num_hidden_layers
            ):
                raise ValueError(
                    "DSpark MLA target_layer_ids are outside the target model"
                )
            if draft_config.draft_vocab_size != target_config.vocab_size:
                raise ValueError(
                    "DSpark MLA draft_vocab_size and target vocab_size must match"
                )
        if batch_size_per_rank % dspark_tp:
            raise ValueError(
                "batch_size_per_rank must be divisible by "
                f"dspark_tp_size={dspark_tp}"
            )
        if draft_config.intermediate_size % dspark_tp:
            raise ValueError(
                "DSpark intermediate_size must be divisible by "
                f"dspark_tp_size={dspark_tp}"
            )
        if draft_config.vocab_size != target_config.vocab_size:
            raise ValueError("draft and target vocab_size must match")
    if parallel.get("cp_size", 1) != 1:
        raise ValueError("Kimi K3 offline mode does not support context parallel")
    if custom_params.get("enable_superkernel", False):
        if settings.get("exe_mode") != "npugraph_ex":
            raise ValueError("enable_superkernel requires exe_mode npugraph_ex")
        if not model.get("enable_static_kernel", False):
            raise ValueError("enable_superkernel requires enable_static_kernel")
        if not custom_params.get("enable_multi_streams", False):
            raise ValueError("enable_superkernel requires enable_multi_streams")
    block_size = model.get("pa_block_size", 128)
    if block_size % 16:
        raise ValueError("pa_block_size must be divisible by 16 for the NZ MLA cache")
    if block_size < attn_tp:
        raise ValueError("pa_block_size must be at least attn_tp_size")
    if settings.get("exe_mode") not in ("eager", "npugraph_ex"):
        raise ValueError("Kimi K3 exe_mode must be eager or npugraph_ex")


def update_vars(world_size, settings):
    parallel = settings.get("parallel_config", {})
    data = settings.get("data_config", {})
    model = settings.get("model_config", {})
    settings = update_settings(
        settings,
        "model_config",
        "draft_model_type",
        get_draft_model_type(settings),
    )
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
    if (
        settings.get("model_config", {}).get("draft_model_type", "none")
        in DSPARK_DRAFT_MODEL_TYPES
    ):
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
