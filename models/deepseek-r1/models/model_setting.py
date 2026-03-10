# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
from executor.utils import update_settings, align_up, MicroBatchMode
from executor.utils.common_utils import update_common_vars, check_common_parallel_settings


def update_vars(world_size, runner_settings):
    update_common_vars(world_size, runner_settings)
    max_position_embeddings = runner_settings.get("data_config").get("max_position_embeddings", 64)
    pa_block_size = runner_settings.get("model_config").get("pa_block_size", 128)
    runner_settings = update_settings(runner_settings, "model_config", "pa_max_length",
                                      align_up(max_position_embeddings, pa_block_size))
    attn_tp_size = runner_settings.get("parallel_config").get("attn_tp_size", 1)
    moe_ep_size = runner_settings.get("parallel_config").get("moe_ep_size", 1)
    # Enable TP for attention + EP for MoE → activate SP sharding.
    if attn_tp_size > 1 and moe_ep_size > 1:
        input_max_len = runner_settings.get("data_config").get("input_max_len", 32)
        max_new_tokens = runner_settings.get("data_config").get("max_new_tokens", 32)
        next_n = runner_settings.get("model_config").get("next_n", 0)
        input_max_len = align_up(input_max_len, attn_tp_size)
        max_position_embeddings = max_new_tokens * (next_n + 1) + input_max_len
        runner_settings = update_settings(runner_settings, "data_config", "max_position_embeddings",
                                          max_position_embeddings)

    if runner_settings.get("exe_mode") == "acl_graph" and os.getenv("TASK_QUEUE_ENABLE", "2") != "1":
        os.environ["TASK_QUEUE_ENABLE"] = "1"  # aclgraph only supports TASK_QUEUE_ENABLE 0 or 1
    else:
        os.environ["TASK_QUEUE_ENABLE"] = "2"  # 2: default value, opt host perf in eager


def check_model_settings(world_size, runner_settings):
    exe_mode = runner_settings.get("exe_mode")
    enable_cache_compile = runner_settings.get("model_config").get("enable_cache_compile", False)
    moe_chunk_max_len = runner_settings.get("model_config").get("moe_chunk_max_len", 65536)
    enable_multi_streams = runner_settings.get("model_config").get("enable_multi_streams", False)
    enable_superkernel = runner_settings.get("model_config").get("enable_superkernel", False)
    next_n = runner_settings.get("model_config").get("next_n", 0)
    enable_pa = runner_settings.get("model_config").get("enable_pa", False)
    micro_batch_mode = runner_settings.get("model_config").get("micro_batch_mode", 0)
    enable_prefill_multi_cycle = runner_settings.get("model_config").get("enable_prefill_multi_cycle", False)
    moe_tp_size = runner_settings.get("parallel_config").get("moe_tp_size")
    atten_tp_size = runner_settings.get("parallel_config").get("attn_tp_size")

    if exe_mode not in ["ge_graph", "acl_graph", "eager"]:
        raise ValueError(f"{exe_mode=} does not supported!")

    enable_aclgraph = exe_mode == "acl_graph"

    if moe_chunk_max_len <= 0:
        raise ValueError(f"{moe_chunk_max_len=} should be a positive integer.")
    dynamo_feat = (enable_cache_compile or enable_multi_streams or enable_superkernel)
    if exe_mode == "eager" and dynamo_feat:
        raise ValueError(f"{exe_mode=} does not support cache compile, aclgraph, multi_streams or superkernel!")
    if enable_aclgraph and enable_superkernel:
        raise ValueError(f"aclgraph dose not support superkernel!")
    if next_n > 3:
        raise ValueError(f"{next_n=}, currently only support 0, 1, 2, 3")
    try:
        micro_batch_mode = MicroBatchMode(micro_batch_mode)
    except Exception as e:
        new_e = ValueError(f" invalid micro_batch_mode, micro_batch_mode can only be int 0, 1, 2 !")
        raise new_e from e
    if micro_batch_mode != MicroBatchMode.DISABLE:
        if not enable_pa:
            raise ValueError(f" micro_batch is only supported when pa is enabled!")
        if enable_prefill_multi_cycle:
            raise ValueError(f" micro_batch requires more then one batch per rank, so {enable_prefill_multi_cycle=}"
                                f"with mini-batch 1 is not supported!")
        if micro_batch_mode == MicroBatchMode.PREFILL_MICRO_BATCH_DP_EP:
            if moe_tp_size != 1 or atten_tp_size != 1:
                raise ValueError(f"  micro_batch_mode 1 can only be enabled when atten only DP and moe only EP!")


def check_parallel_settings(world_size, runner_settings):
    check_common_parallel_settings(world_size, runner_settings)
    attn_tp_size = runner_settings.get("parallel_config").get("attn_tp_size")
    embed_tp_size = runner_settings.get("parallel_config").get("embed_tp_size")
    attn_dp_size = world_size // attn_tp_size
    batch_size = runner_settings.get("data_config").get("batch_size", 1)

    if embed_tp_size < attn_tp_size:
        raise ValueError(f"{embed_tp_size=} should not be smaller than {attn_tp_size=}")
    elif embed_tp_size % attn_tp_size != 0:
        raise ValueError(f"{embed_tp_size=} should be a multiple of {attn_tp_size=}")
    if batch_size % attn_dp_size != 0:
        raise ValueError(f"{batch_size=} is not divisible by {attn_dp_size=}")


def check_vars(world_size, runner_settings):
    check_parallel_settings(world_size, runner_settings)
    check_model_settings(world_size, runner_settings)
