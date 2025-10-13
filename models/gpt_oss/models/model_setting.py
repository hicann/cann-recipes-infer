# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from executor.utils import update_settings
from executor.utils.common_utils import check_common_parallel_settings

def check_vars(world_size, runner_settings):
    check_common_parallel_settings(world_size, runner_settings)
    attn_tp_size = runner_settings.get("parallel_config").get("attn_tp_size", 1)
    moe_tp_size = runner_settings.get("parallel_config").get("moe_tp_size", 1)
    lmhead_tp_size = runner_settings.get("parallel_config").get("lmhead_tp_size", 1)
    if not world_size == attn_tp_size == moe_tp_size == lmhead_tp_size:
        raise ValueError("The values of world_size, attn_tp_size, moe_tp_size and lmhead_tp_size must be equal.")


def update_vars(world_size, runner_settings):
    input_max_len = runner_settings.get("data_config").get("input_max_len", 32)
    max_new_tokens = runner_settings.get("data_config").get("max_new_tokens", 32)
    max_len_bound = max_new_tokens + input_max_len
    runner_settings = update_settings(runner_settings, "data_config", "max_position_embeddings", max_len_bound)