# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import pkgutil
import warnings

__all__ = list(module for _, module, _ in pkgutil.iter_modules([os.path.dirname(__file__)]))

# 导入so 和 python
from . import custom_ops_lib
from .converter import npu_moe_gating_top_k, npu_indexer_compress_epilog, npu_moe_init_routing_group_quant, npu_hc_post, npu_hc_pre, npu_swiglu_clip_quant, npu_swiglu_group_quant, npu_gather_selection_kv_cache, npu_quant_lightning_indexer, npu_sparse_attn_sharedkv, npu_quant_lightning_indexer_metadata, npu_sparse_attn_sharedkv_metadata, npu_hc_pre_sinkhorn, npu_hc_pre_inv_rms, npu_kv_quant_sparse_attn_sharedkv, npu_kv_quant_sparse_attn_sharedkv_metadata, npu_dequant_swiglu_clamp_quant

"""
import custom ops as torch_npu ops to support the following usage:
'torch.ops.custom.npu_selected_flash_attention()'
'torch_npu.npu_selected_flash_attention()'
"""

# Ensure that the torch and torch_npu has been successfully imported to avoid subsequent mount operation failures
import torch
import torch_npu

# get torch.ops.custom module
custom_ops_module = getattr(torch.ops, 'custom', None)

if custom_ops_module is not None:
    for op_name in dir(custom_ops_module):
        if op_name.startswith('_'):
            # skip built-in method, such as __name__, __doc__
            continue

        # get custom ops and set to torch_npu
        custom_op_func = getattr(custom_ops_module, op_name)
        setattr(torch_npu, op_name, custom_op_func)

else:
    warn_msg = "torch.ops.custom module is not found, mount custom ops to torch_npu failed." \
               "Calling by torch_npu.xxx for custom ops is unsupported, please use torch.ops.custom.xxx."
    warnings.warn(warn_msg)
    warnings.filterwarnings("ignore", message=warn_msg)