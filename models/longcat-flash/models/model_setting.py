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

import logging
import os
from executor.utils import update_settings, align_up
from executor.utils.common_utils import update_common_vars, check_common_parallel_settings


def update_vars(world_size, runner_settings):
    update_common_vars(world_size, runner_settings)
    kvp_size = runner_settings.get("parallel_config").get("kvp_size", 1)
    batch_size = runner_settings.get("data_config").get("batch_size", 1)
    max_position_embeddings = runner_settings.get("data_config").get("max_position_embeddings", 64)
    pa_block_size = runner_settings.get("model_config").get("pa_block_size", 128)
    if kvp_size > 1:
        batch_size_per_rank = (batch_size * kvp_size) // world_size
    else:
        batch_size_per_rank = max(batch_size // world_size, 1)
    runner_settings = update_settings(runner_settings, "data_config", "batch_size_per_rank", batch_size_per_rank)
    runner_settings = update_settings(runner_settings, "model_config", "pa_max_length",
                                        align_up(align_up(max_position_embeddings, kvp_size) // kvp_size, pa_block_size)
                                        )

    enable_multi_stream = runner_settings.get("model_config").get("enable_multi_stream", 0)
    exe_mode = runner_settings.get("exe_mode", "eager")
    # multi_stream is only supported for ge_graph mode in this model
    if enable_multi_stream and exe_mode == "eager":
        runner_settings = update_settings(runner_settings, "model_config", "enable_multi_stream", 0)
        logging.warning(f"{exe_mode=} doesn't support {enable_multi_stream=}, force set {enable_multi_stream=} to run")
        enable_multi_stream = 0

    enable_prefetch = runner_settings.get("model_config").get("enable_prefetch", False)
    # prefetch is only supported on the multi stream branch in this model
    if enable_prefetch and not enable_multi_stream:
        runner_settings = update_settings(runner_settings, "model_config", "enable_prefetch", False)
        enable_prefetch = False
        logging.warning(f"Prefetch is only supported when enable_multi_stream > 0, force set {enable_prefetch=} to run")


def check_model_settings(world_size, runner_settings):
    exe_mode = runner_settings.get("exe_mode")
    enable_cache_compile = runner_settings.get("model_config").get("enable_cache_compile", False)
    moe_chunk_max_len = runner_settings.get("model_config").get("moe_chunk_max_len", 65536)
    enable_multi_stream = runner_settings.get("model_config").get("enable_multi_stream", 0)
    enable_superkernel = runner_settings.get("model_config").get("enable_superkernel", False)
    next_n = runner_settings.get("model_config").get("next_n", 0)
    kvp_size = runner_settings.get("parallel_config").get("kvp_size", 1)

    if exe_mode not in ["ge_graph", "eager"]:
        raise ValueError(f"{exe_mode=} does not supported!")
    if moe_chunk_max_len <= 0:
        raise ValueError(f"{moe_chunk_max_len=} should be a positive integer.")
    dynamo_feat = (enable_cache_compile or enable_multi_stream or enable_superkernel)
    if exe_mode == "eager" and dynamo_feat:
        raise ValueError(f"{exe_mode=} does not support cache compile, aclgraph, multi_streams or superkernel!")
    if next_n > 2:
        raise ValueError(f"{next_n=}, currently only support 0 or 1 or 2")
    if kvp_size > 1 and next_n > 0:
        raise ValueError(f"{kvp_size=}, currently only support next_n = 0")


def check_parallel_settings(world_size, runner_settings):
    check_common_parallel_settings(world_size, runner_settings)
    attn_tp_size = runner_settings.get("parallel_config").get("attn_tp_size")
    o_proj_tp_size = runner_settings.get("parallel_config").get("o_proj_tp_size")
    cp_size = runner_settings.get("parallel_config").get("cp_size", 1)
    batch_size = runner_settings.get("data_config").get("batch_size", 1)
    kvp_size = runner_settings.get("parallel_config").get("kvp_size", 1)

    if attn_tp_size > 1 and attn_tp_size != o_proj_tp_size:
        raise ValueError(f"when attn_tp_size > 1, {attn_tp_size=} must be equal to {o_proj_tp_size=}")
    if kvp_size > 1 and batch_size % (world_size // kvp_size) != 0:
        raise ValueError(f"when kvp_size > 1, {batch_size=} is not divisible by {world_size // kvp_size=}")
    if kvp_size > 1 and attn_tp_size != 1:
        raise ValueError(f"when kvp_size > 1, {attn_tp_size=} must be equal to 1")
    if kvp_size > 1 and o_proj_tp_size != kvp_size:
        raise ValueError(f"when kvp_size > 1, {o_proj_tp_size=} must be equal to {kvp_size=}")
    if kvp_size > 1 and cp_size != 1:
        raise ValueError(f"when kvp_size > 1, {cp_size=} must be equal to 1")


def check_vars(world_size, runner_settings):
    check_parallel_settings(world_size, runner_settings)
    check_model_settings(world_size, runner_settings)


def update_afd_settings(runner_settings):
    enable_afd = runner_settings.get("model_config").get("enable_afd", False)
    global_world_size = runner_settings.get("world_size", 1)
    if enable_afd and global_world_size % 2 != 0:
        raise ValueError(f"Afd is only supported when global_world_size % 2 == 0, but now {global_world_size=}!")

    # If not in AFD scenario, every rank is treated as an attention rank.
    ffn_world_size = global_world_size // 2 if enable_afd else 0
    runner_settings.update({"ffn_world_size": ffn_world_size})
    logging.info(f"add (ffn_world_size: {ffn_world_size}) to runner_settings.")


def check_and_update_vars(runner_settings):
    update_afd_settings(runner_settings)
    world_size = runner_settings.get("world_size", 1) - runner_settings.get("ffn_world_size", 0)
    check_vars(world_size, runner_settings)
    update_vars(world_size, runner_settings)


def check_is_attn_rank(runner_settings):
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    rank_offset = int(os.getenv("RANK_OFFSET", "0"))
    global_rank = local_rank + rank_offset
    ffn_world_size = runner_settings.get("ffn_world_size", 0)
    # In AFD scenario, the first half of the ranks will serve as FFN, and the second half will serve as Attn.
    return False if global_rank < ffn_world_size else True
