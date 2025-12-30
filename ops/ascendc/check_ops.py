#!/usr/bin/env python3
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 China Mobile Limited. All rights reserved 
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# =========================================================================================================

# check_lightning_ops.py
import logging
import torch
import torch_npu
import custom_ops

# -------------------------------------------------
# 0. 日志初始化（如主入口已配置，可删除）
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# -------------------------------------------------
# 1. 检查单个算子是否存在
# -------------------------------------------------
LIGHTNING_OPS = {
    "npu_lightning_indexer",
    "npu_lightning_indexer_quant",
    "npu_gather_selection_kv_cache",
    "npu_mla_prolog_v3",
    "npu_sparse_flash_attention",
    "npu_sparse_flash_attention_antiquant",
    "npu_swiglu_clip_quant",
}


def check_single_ops():
    ops = torch.ops.custom
    logging.info("------------ 1. 单个算子存在性检查 ------------")
    for name in sorted(LIGHTNING_OPS):
        exist = hasattr(ops, name)
        logging.info("%-40s exist: %s", name, exist)


# -------------------------------------------------
# 2. 枚举 torch.ops.custom 下所有已注册算子
# -------------------------------------------------
def list_all_custom_ops():
    ops = torch.ops.custom
    logging.info("\n------------ 2. 当前 custom_ops 全部算子 ------------")
    all_ops = [attr for attr in dir(ops) if not attr.startswith("_")]
    if not all_ops:
        logging.warning("（暂无算子）")
    else:
        for op in sorted(all_ops):
            logging.info(op)


# -------------------------------------------------
# 3. 一键运行
# -------------------------------------------------
if __name__ == "__main__":
    check_single_ops()
    list_all_custom_ops()
