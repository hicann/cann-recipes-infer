
# coding=utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging
import math
import os
import torch
import torch_npu
import torch.distributed as dist
from torch.distributed.distributed_c10d import _world

from executor.utils import align_up

WIN_ADDR_ALIGN = 512
MB_SIZE = 1024 * 1024
SCALE_EXPAND_IDX_BUFFER = 44
UB_ALIGN = 32
FULL_MESH_DATA_ALIGN = 480


def get_default_group():
    return _world._default_pg


def get_group_name(comm_group, global_rank):
    return None if comm_group is None\
        else comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(global_rank)

created_group = {}

def init_comm_group(
    global_rank,
    group_num,
    world_size,
    group_stride=1,
    group_name="default",
    hccl_buffer_size=200,
    return_name=False,
    group_type=None, # 1：hostcpu_ts 2 aicpu_ts 3:aiv 4:aiv_only 5:ccu_ms 6:ccu_sch 7 aicpu_ub 0 default
    platform_version="A3",
):
    group_size = world_size // group_num
    default_pg = get_default_group()

    cur_group_set = None
    for group_id in range(group_num):
        if group_stride == 1:
            start_rank_id = group_id * group_size
            init_rank_id = global_rank // group_size * group_size
        else:
            start_rank_id = group_id
            init_rank_id = global_rank % group_num

        cur_group_list = [start_rank_id + i * group_stride for i in range(group_size)]
        if default_pg is not None and group_type is None:
            # fullmesh v2, communication for Expert Parallelism shall be exclusively occupied
            if group_num == world_size and "moe_ep_group" not in group_name:
                cur_group = None
            elif group_num == 1 and "moe_ep_group" not in group_name:
                cur_group = default_pg
            else:
                logging.info(f"group:{group_name} create default type comm group")
                options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
                options.hccl_config = {"hccl_buffer_size": hccl_buffer_size}
                if platform_version == "950":
                    os.environ["HCCL_BUFFSIZE"] = str(hccl_buffer_size)
                cur_group = dist.new_group(cur_group_list, pg_options=options)
        elif group_type is not None:
            global created_group
            if group_type == 0:
                logging.info(f"group:{group_name} create default type comm group")
                cur_group = default_pg
            else:
                if group_name not in created_group.keys():
                    logging.info(f"group:{group_name} create type {group_type} comm group")
                    options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
                    options.hccl_config = {"hccl_op_expansion_mode" : group_type}
                    options.hccl_config = {"hccl_buffer_size": hccl_buffer_size}
                    if platform_version == "950":
                        os.environ["HCCL_BUFFSIZE"] = str(hccl_buffer_size)
                    cur_group = dist.new_group(cur_group_list, pg_options=options)
                    created_group[group_name] = cur_group
                else:
                    logging.info(f"group:{group_name} has already been created")
                    cur_group = created_group.get(group_name, None)
                    assert cur_group is not None
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


def calc_moe_hccl_buffer_size(
        runner_settings,
        config,
        is_full_mesh_v2=False):
    """
    calc hccl buffer size (MB) for MoE Dispatch and Combine ops.
    formula:
      not full_mesh_v2:
        (localMoeExpertNum * maxBs * ep_worldsize * align512(ceil480(align32(2*h)+44)) +
         (top_k + shardExpertNum) * maxBs * align512(2*h)) * 2 / 1024 / 1024
      full_mesh_v2:
        (localMoeExpertNum * maxBs * ep_worldsize * align512(align32(2*h)+44) +
         (top_k + shardExpertNum) * maxBs * align512(2*h)) * 2 / 1024 / 1024
    """
    default_hccl_buffsize = 200 # MB
    world_size = runner_settings.get("world_size", 16)
    batch_size = runner_settings.get("data_config").get("batch_size", 16)
    next_n = runner_settings.get("model_config").get("next_n", 0)
    spec_len = next_n + 1
    moe_ep_size = runner_settings.get("parallel_config").get("moe_ep_size", 1)
    platform_version = runner_settings.get("model_config").get("platform_version", "A3")

    experts_per_rank = config.n_routed_experts // moe_ep_size
    hidden_size = config.hidden_size
    top_k = config.num_experts_per_tok
    shared_expert_rank_num = 0 # route and share on same card = 0

    bs_per_rank = batch_size // world_size * spec_len
    if not is_full_mesh_v2:
        token_need_size_dispatch = align_up(align_up(2 * hidden_size, UB_ALIGN) +
                                 SCALE_EXPAND_IDX_BUFFER, WIN_ADDR_ALIGN)
    else:
        token_need_size_dispatch = math.ceil(2 * hidden_size / FULL_MESH_DATA_ALIGN) * WIN_ADDR_ALIGN
    dispatch_size = experts_per_rank * bs_per_rank * world_size * token_need_size_dispatch
    combine_size = (top_k + shared_expert_rank_num) * bs_per_rank * \
                    align_up(2 * hidden_size, WIN_ADDR_ALIGN)
    moe_buffer_size = (dispatch_size + combine_size) * 2 / MB_SIZE  # MB
    moe_buffer_size = math.ceil(moe_buffer_size) + 1  # add win size

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
