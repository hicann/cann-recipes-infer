#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os
import logging
import sys
import glob
import shutil
import subprocess
import sysconfig
from pathlib import Path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from setuptools import Command


class CleanCommand(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # 删除构建目录
        if os.path.exists('build'):
            shutil.rmtree('build')
            logging.info("Removed build/")

        # 删除dist目录
        if os.path.exists('dist'):
            shutil.rmtree('dist')
            logging.info("Removed dist/")

        # 删除egg-info目录
        egg_info_dir = f"{self.distribution.get_name().replace('-', '_')}.egg-info"
        if os.path.exists(egg_info_dir):
            shutil.rmtree(egg_info_dir)
            logging.info(f"Removed {egg_info_dir}/")

        # 删除.pyc文件和__pycache__目录
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.pyc'):
                    os.remove(os.path.join(root, file))
                    logging.info(f"Removed {os.path.join(root, file)}")

            for dir_name in dirs:
                if dir_name == '__pycache__':
                    shutil.rmtree(os.path.join(root, dir_name))
                    logging.info(f"Removed {os.path.join(root, dir_name)}/")


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        cmake_path = shutil.which("cmake")
        if cmake_path is None:
            raise RuntimeError("CMake must be installed to build the extensions")

        try:
            subprocess.check_output([cmake_path, "--version"])
        except OSError as e:
            raise RuntimeError(
                f"Failed to execute CMake: {e}\n"
                "Please ensure CMake is properly installed."
            ) from e

        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        python_include = sysconfig.get_path('include')
        python_libs = sysconfig.get_config_var('LIBDIR')

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DBUILD_TORCH_OPS=ON",
            f"-DPYTHON_EXTENSION_USE_ABI3=ON",
            f"-DPYTHON_INCLUDE_DIR={python_include}",
            f"-DPYTHON_LIBRARIES={python_libs}",
            f"-DPy_LIMITED_API_VERSION=0x03080000",

        ]

        build_type = "Debug" if self.debug else "Release"
        build_args = ["--config", build_type]

        cpu_count = os.cpu_count() or 1
        parallel_jobs = max(16, cpu_count // 2)
        build_args += ["--", f"-j{parallel_jobs}"]

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        subprocess.check_call(["cmake", ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_temp)


setup(
    name='npu_ops_transformer_ext',
    version='0.0.1',
    packages=find_packages(),
    ext_modules=[CMakeExtension("npu_ops_transformer_ext._C", sourcedir=".")],
    cmdclass={'build_ext': CMakeBuild, 'clean': CleanCommand},
    zip_safe=False,
    install_requires=["torch"],
    options={"bdist_wheel": {"py_limited_api": "cp38"}},
    description="Example of PyTorch C++ and Ascend extensions (with CMake)",
)
