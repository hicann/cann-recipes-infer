# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import logging

from executor.utils.common_utils import check_common_parallel_settings, update_common_vars
from executor.utils.common_utils import update_settings

logger = logging.getLogger(__name__)


def update_vars(world_size, runner_settings):
    update_common_vars(world_size, runner_settings)
    embed_dp_size = world_size // runner_settings.get("parallel_config").get("embed_tp_size", 1)
    batch_size = runner_settings.get("data_config").get("batch_size", 1)
    batch_size_per_rank = batch_size // embed_dp_size
    runner_settings = update_settings(runner_settings, "data_config", "batch_size_per_rank", batch_size_per_rank)
    if runner_settings.get("exe_mode") == "acl_graph" and os.getenv("TASK_QUEUE_ENABLE", "2") != "1":
        os.environ["TASK_QUEUE_ENABLE"] = "1"  # aclgraph only supports TASK_QUEUE_ENABLE 0 or 1
    else:
        os.environ["TASK_QUEUE_ENABLE"] = "2"  # 2: default value, opt host perf in eager


def check_model_settings(world_size, runner_settings):
    del world_size
    exe_mode = runner_settings.get("exe_mode")
    enable_cache_compile = runner_settings.get("model_config").get("enable_cache_compile", False)

    if exe_mode not in ["ge_graph", "eager", "acl_graph"]:
        raise ValueError(
            f"{exe_mode=} does not supported! "
            "Only the eager, ge_graph and acl_graph mode are supported!"
        )

    dynamo_feat = enable_cache_compile
    if exe_mode == "eager" and dynamo_feat:
        logger.info("%s does not support cache compile!", exe_mode)


def check_parallel_settings(world_size, runner_settings):
    moe_ep_size = world_size // runner_settings.get("parallel_config").get("moe_tp_size", 1)
    batch_size = runner_settings.get("data_config").get("batch_size", 1)
    if batch_size % moe_ep_size != 0:
        raise ValueError(f"{batch_size=} is not divisible by {moe_ep_size=}")
    check_common_parallel_settings(world_size, runner_settings)


def check_vars(world_size, runner_settings):
    check_parallel_settings(world_size, runner_settings)
    check_model_settings(world_size, runner_settings)
