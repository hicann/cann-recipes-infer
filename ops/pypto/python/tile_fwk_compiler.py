#!/usr/bin/env python
# -*- coding:utf-8 -*-
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

"""
tile fwk ops compile script
"""
import ctypes
import os
import stat
from shutil import copy as file_copy
import tbe.common.context.op_context as op_context
from tbe.common.buildcfg import get_current_build_config
from tbe.common.platform.platform_info import get_soc_spec


def load_rt_lib():
    ascend_home_path = os.getenv("ASCEND_HOME_PATH")
    so_lib_path = os.path.join(ascend_home_path, "tile-framework-devkit", "lib64", "libtile_fwk_compiler.so")
    librt = ctypes.CDLL(so_lib_path)
    return librt


def copy_file_to_output(src_file_path, output_path):
    try:
        aicpu_file_path = os.path.join(os.path.dirname(src_file_path),
            "kernel_meta", "libNativeSparseAttention_machine.so")
        aicpu_file_path = os.path.realpath(aicpu_file_path)
        if os.path.exists(aicpu_file_path):
            os.chmod(aicpu_file_path, stat.S_IWUSR + stat.S_IRGRP + stat.S_IRUSR)
            parent_path = os.path.abspath(os.path.dirname(output_path))
            aicpu_path = os.path.join(parent_path, "tile_fwk_machine")
            os.makedirs(aicpu_path, exist_ok=True)
            file_copy(aicpu_file_path, aicpu_path)
    except Exception as e:
        raise RuntimeError("Copy [%s] to [%s] failed, reason: %s." % (aicpu_file_path, output_path, str(e))) from e
    finally:
        pass


def ascendcpp_compile_op(*args):
    kernel_name = args[-1]
    cur_context = op_context.get_context()

    if cur_context is None:
        return False
    op_infos = cur_context.get_op_info()
    if op_infos is None or len(op_infos) == 0:
        return False
    op_info = op_infos[0]
    if op_info is None:
        return False
    op_type_c = op_info.op_type.encode('utf_8')
    soc_version = get_soc_spec("SOC_VERSION")
    soc_version_c = soc_version.encode('utf_8')
    dump_path = get_current_build_config("kernel_meta_parent_dir") + "/kernel_meta"
    dump_path_c = dump_path.encode('utf_8')
    kernel_name_c = kernel_name.encode('utf_8')
    try:
        librt = load_rt_lib()
        if librt is None:
            return False
        res = librt.TileFwkCompileFatbin(op_type_c, soc_version_c, dump_path_c, kernel_name_c)
        print("TileFwkCompileFatbin result is : ", res, flush=True)
    except Exception as e:
        raise RuntimeError("Exception: Fail to call compile func, reason is %s." % str(e)) from e
    if bool(res) is not True:
        return False
    output_path = cur_context.get_addition("output")
    if output_path is None:
        return False
    copy_file_to_output(dump_path, output_path)
    return True
