# coding=utf-8
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging
import math
import torch
import torch_npu
import torch.distributed as dist
from torch.distributed.distributed_c10d import _world

from executor.utils import align_up


def get_default_group():
    # 避免重复显式调用私有变量
    return _world._default_pg


def get_group_name(comm_group, global_rank):
    return None if comm_group is None\
        else comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(global_rank)

    
def init_comm_group(
    global_rank,
    group_num, 
    world_size,
    group_stride=1,
    group_name="default",
    hccl_buffer_size=200,
    return_name=False
):
    # 预期创建 group_num 个通信域, 每个通信域的大小是 group_size
    ## 默认 group_num * group_size == world_size
    # 每个通信域中, rank的跨度是group_stride
    group_size = world_size // group_num
    default_pg = get_default_group()

    cur_group_set = None
    for group_id in range(group_num):
        # 构建每个通信域的首rank
        if group_stride == 1:
            # 连续通信域场景，起始地址为 0, 4, 8, 12。每个通信域内为 [0, 1, 2, 3]
            start_rank_id = group_id * group_size
            init_rank_id = global_rank // group_size * group_size
        else:
            # 跳跃通信域场景，起始地址为 0, 1, 2, 3。每个通信域内为 [0, 4, 8, 12]
            start_rank_id = group_id
            init_rank_id = global_rank % group_num

        cur_group_list = [start_rank_id + i * group_stride for i in range(group_size)]
        if default_pg is not None:
            if group_num == world_size:
                cur_group = None
            else:
                options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
                cur_group = dist.new_group(cur_group_list, pg_options=options)
        else:
            cur_group = None

        if start_rank_id == init_rank_id:
            cur_group_set = cur_group
            logging.info(f"group_name is {group_name}, group_list: {cur_group_list}")
    logging.info(f"{group_name} hccl comm init rank_id: {global_rank}")
    if not return_name:
        return cur_group_set
    else:
        logging.info(f"{group_name} hccl comm init in else branch rank_id: {global_rank}")
        comm_name = get_group_name(cur_group_set, global_rank)
        logging.info(f"{group_name} rank_{global_rank} hccl comm init in else branch comm_name: {comm_name}")
        return cur_group_set, comm_name


def calc_moe_hccl_buffer_size(runner_settings, config):
    """
    calc hccl buffer size (MB) for MoE Dispatch and Combine ops.
    formula:
        (localMoeExpertNum * maxBs * ep_worldsize * align512(align32(2*h)+64) +
         (top_k + shardExpertNum) * maxBs * align512(2*h)) * 2 / 1024 / 1024
    """
    default_hccl_buffsize = 200 # MB
    world_size = runner_settings.get("world_size", 16)
    batch_size = runner_settings.get("data_config").get("batch_size", 16)
    next_n = runner_settings.get("model_config").get("next_n", 0)
    spec_len = next_n + 1
    moe_ep_size = runner_settings.get("parallel_config").get("moe_ep_size", 1)

    experts_per_rank = config.n_routed_experts // moe_ep_size
    hidden_size = config.hidden_size
    top_k = config.num_experts_per_tok
    shared_expert_rank_num = 0 # route and share on same card = 0

    bs_per_rank = batch_size // world_size * spec_len
    dispatch_size = experts_per_rank * bs_per_rank * world_size * \
                    align_up(align_up(2 * hidden_size, 32) + 64, 512)
    combine_size = (top_k + shared_expert_rank_num) * bs_per_rank * \
                    align_up(2 * hidden_size, 512)
    moe_buffer_size = (dispatch_size + combine_size) * 2 / 1024 / 1024 # MB
    moe_buffer_size = math.ceil(moe_buffer_size)

    # use default value if moe_buffer_size is small than default_hccl_buffersize
    if moe_buffer_size <= default_hccl_buffsize:
        hccl_buffer_size = default_hccl_buffsize
    else:
        hccl_buffer_size = moe_buffer_size

    logging.info(f"batch_size:{batch_size} world_size:{world_size} moe_ep_size:{moe_ep_size}")
    logging.info(f"experts_per_rank:{experts_per_rank} hidden_size:{hidden_size} spec_len:{spec_len}")
    logging.info(f"dispatch_size:{dispatch_size} combine_size:{combine_size}")
    logging.info(f"hccl_buffer_size:{hccl_buffer_size} (MB) moe_buffer_size:{moe_buffer_size} (MB)")

    return hccl_buffer_size