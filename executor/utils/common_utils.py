# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging
from functools import wraps
from typing import Dict
from enum import Enum
import yaml
import torch
import torch_npu
import numpy as np
import torchair as tng


def read_yaml(yaml_file_path):
    try:
        with open(yaml_file_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"No such yaml file: {yaml_file_path}")
    except yaml.YAMLERROR as e:
        logging.error(f"Load yaml file failed: {e}")
    return data


class FakeContextManager:
    def __init__(self) -> None:
        pass
    
    def __enter__(self):
        pass
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass


def superkernel_scope(enable: bool, scope: str, options: str = None):
    if enable:
        return tng.scope.super_kernel(scope, options)
    else:
        return FakeContextManager()


def align_up(a, b):
    if b <= 0:
        raise ValueError("b should be larger then zero!")                            
    return (a + b - 1) // b * b


def ceil_div(a, b):
    return (a + b - 1) // b


def update_settings(runner_settings: Dict, module_name: str, key: str, value):
    if runner_settings.get(module_name) is None:
        raise Exception(f"runner_settings doesn't have submodule ({module_name})!")
    module = runner_settings.get(module_name)
    module.update({key: value})
    logging.info(f"add ({key}: {value}) to runner_settings.")
    return runner_settings


def override(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def get_init_attn_mask(mask_length, device, valid_len=None):
    share_mask_tril = ~torch.tril(
        torch.ones((mask_length, mask_length),
                   dtype=torch.bool, device=device))
    if valid_len is not None:
        share_mask_tril[-valid_len:, :] = torch.zeros(valid_len, mask_length)
    return share_mask_tril


def get_decode_mask(mask_length, device, position):
    decode_mask = torch.zeros((1, mask_length), device=device)
    decode_mask[0, :position] = 1
    return decode_mask


def npu_stream_switch(switch_flag: bool, stream_tag: str, stream_priority: int = 0):
    if switch_flag:
        return tng.scope.npu_stream_switch(stream_tag, stream_priority)
    else:
        return FakeContextManager()
    

def process_infer_time(infer_time_rec, token_count):
    if len(infer_time_rec) == 0: # no time recorded
        logging.info(f"precoss infer time receives empty time record")
        return 0
    elif len(infer_time_rec) == 1 or (token_count <= 1): # only prefill
        return infer_time_rec[0]
    else: # obtain average time for decode
        avg_token_per_round = token_count / len(infer_time_rec) # mtp steps may take more than one token

        # skip the time cost for prefill step
        infer_time_rec = infer_time_rec[1:]
        token_count -= 1

    q1 = np.percentile(infer_time_rec, 25)
    q3 = np.percentile(infer_time_rec, 75)
    iqr_upper_threshold = q3 + 1.5 * (q3 - q1)
    total_time = 0
    for t in infer_time_rec:
        if t > iqr_upper_threshold:
            token_count -= avg_token_per_round
            continue
        total_time += t
    if token_count == 0:
        return infer_time_rec[0]
    avg_infer_time = total_time / token_count

    return avg_infer_time


class MicroBatchMode(Enum):
    DISABLE = 0
    PREFILL_MICRO_BATCH_DP_EP = 1
    PREFILL_MICRO_BATCH_SP_TP_EP = 2


def remove_padding_left(tensor, pad_id):
    if tensor.shape[0] == 1:
        return [tensor[0]]
    if tensor.dim() != 2:
        raise ValueError("remove padding func input dim must be 2")
    batch_size, seq_len = tensor.shape
    output_tensorlist = []

    for i in range(batch_size):
        row = tensor[i]
        mask = (row != pad_id)
        if mask.any():
            first_valid_token = torch.argmax(mask.float())
            processed_row = row[first_valid_token:]
        else:
            processed_row = row
        output_tensorlist.append(processed_row)

    return output_tensorlist


def check_common_parallel_settings(world_size, runner_settings):
    if world_size <= 0:
        raise ValueError(f"{world_size=} must greater than 0")
    parallel_config = runner_settings.get("parallel_config", {})
    batch_size = runner_settings.get("data_config").get("batch_size", 1)
    for key, value in parallel_config.items():
        if ("tp_size" in key or "ep_size" in key) and world_size % value != 0:
            raise ValueError(f"{world_size=} is not divisible by {key}={value}")
        if "dp_size" in key and batch_size % value != 0:
            raise ValueError(f"{batch_size=} is not divisible by {key}={value}")


def update_common_vars(world_size, runner_settings):
    attn_dp_size = world_size // runner_settings.get("parallel_config").get("attn_tp_size", 1)
    moe_dp_size = world_size // runner_settings.get("parallel_config").get("moe_tp_size", 1)
    moe_ep_size = moe_dp_size
    embed_dp_size = world_size // runner_settings.get("parallel_config").get("embed_tp_size", 1)

    batch_size = runner_settings.get("data_config").get("batch_size", 1)
    batch_size_per_rank = batch_size // attn_dp_size

    runner_settings = update_settings(runner_settings, "data_config", "batch_size_per_rank", batch_size_per_rank)
    runner_settings = update_settings(runner_settings, "parallel_config", "attn_dp_size", attn_dp_size)
    runner_settings = update_settings(runner_settings, "parallel_config", "moe_dp_size", moe_dp_size)
    runner_settings = update_settings(runner_settings, "parallel_config", "moe_ep_size", moe_ep_size)
    runner_settings = update_settings(runner_settings, "parallel_config", "embed_dp_size", embed_dp_size)

    input_max_len = runner_settings.get("data_config").get("input_max_len", 32)
    max_new_tokens = runner_settings.get("data_config").get("max_new_tokens", 32)
    next_n = runner_settings.get("model_config").get("next_n", 0)
    max_len_bound = max_new_tokens * (next_n + 1) + input_max_len
    runner_settings = update_settings(runner_settings, "data_config", "max_position_embeddings", max_len_bound)