# coding=utf-8
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

__all__ = ["update_settings", "init_comm_group", "get_group_name", "get_default_group", "read_yaml",
           "override", "get_init_attn_mask", "get_decode_mask", "npu_stream_switch", "align_up",
           "superkernel_scope", "ceil_div", "process_infer_time", "build_dataset_input",
           "calc_moe_hccl_buffer_size", "MicroBatchMode", "remove_padding_left"]

from .common_utils import (update_settings, override, get_init_attn_mask, get_decode_mask,
                           npu_stream_switch, align_up, read_yaml, superkernel_scope, ceil_div,
                           process_infer_time, MicroBatchMode, remove_padding_left
                          )
from .hccl_utils import init_comm_group, get_group_name, get_default_group, calc_moe_hccl_buffer_size
from .data_utils import build_dataset_input