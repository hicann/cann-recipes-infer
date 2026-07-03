# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import importlib
import pkgutil
from transformers.utils import logging

logger = logging.get_logger(__name__)
SUPPORT_PLATFORM = ["A3", "950"]


class OpKernel:
    OP_TYPE = []
    KERNEL_MAP = {}

    @classmethod
    def op_impl_apply(cls, op_type, used_kernel):
        if used_kernel not in cls.KERNEL_MAP:
            logger.warning(f"{used_kernel} is not support")
            return
        used_kernel_func = cls.KERNEL_MAP[used_kernel]
        setattr(cls, op_type, staticmethod(used_kernel_func))
        return


def register_op_impl(op_type, func_key: str = None):
    def decorator(func):
        if op_type not in OpKernel.OP_TYPE:
            OpKernel.OP_TYPE.append(op_type)
            logger.info(f"register optype {op_type} success.")
        func_name = func_key if func_key else func.__name__
        support_platform = tuple([platform.lower() for platform in SUPPORT_PLATFORM])
        if not func_name.endswith(support_platform):
            for platform in support_platform:
                func_name_update = func_name + "_" + platform
                if func_name_update in OpKernel.KERNEL_MAP:
                    logger.warning(f"func {func_name_update} is registed, and will be overlaped.")
                OpKernel.KERNEL_MAP[func_name_update] = func
        else:
            if func_name in OpKernel.KERNEL_MAP:
                logger.warning(f"func {func_name} is registed, and will be overlaped.")
            OpKernel.KERNEL_MAP[func_name] = func

        @functools.wraps(func)
        def wrapper(*args, **kargs):
            return func(*args, **kargs)
        return wrapper
    return decorator


def auto_import_modules(pkg_name):
    package = importlib.import_module(pkg_name)
    for _, module_name, is_pkg in pkgutil.walk_packages(
        package.__path__,
        package.__name__ + "."
    ):
        if not is_pkg:
            try:
                importlib.import_module(module_name)
            except Exception as e:
                logger.warning(f"import mudule {module_name} failed: {str(e)}")