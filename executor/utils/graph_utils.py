# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

"""
Graph compilation utilities for NPU inference.
"""

import os
import logging
import torch
from torchair.configs.compiler_config import CompilerConfig


def compile_model_forward(
    model_forward,
    exe_mode="ge_graph",
    enable_cache_compile=False,
    cache_dir=None,
    frozen_parameter=True,
    tiling_schedule_optimize=True,
    topology_sorting_strategy="StableRDFS",
):
    """
    Compile a model.forward method with torchair backend.

    Args:
        model_forward: The model.forward callable to compile
        exe_mode: Execution mode ("ge_graph", "acl_graph", etc.)
        enable_cache_compile: Whether to use cache compilation
        cache_dir: Directory for cache compilation
        frozen_parameter: Whether to freeze parameters
        tiling_schedule_optimize: Enable tiling schedule optimization
        topology_sorting_strategy: Strategy for topology sorting

    Returns:
        Compiled forward function
    """
    import torchair as tng
    import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce
    tng.patch_for_hcom()

    # Create and configure CompilerConfig
    compiler_config = CompilerConfig()
    compiler_config.experimental_config.frozen_parameter = frozen_parameter
    compiler_config.experimental_config.tiling_schedule_optimize = tiling_schedule_optimize
    compiler_config.experimental_config.topology_sorting_strategy = topology_sorting_strategy

    use_aclgraph = exe_mode == "acl_graph"
    if use_aclgraph:
        compiler_config.mode = "reduce-overhead"
        if torch.__version__ < "2.5.0":
            compiler_config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True

    # Compile model forward
    dynamic = exe_mode == "acl_graph"
    if enable_cache_compile:
        if cache_dir is None:
            raise ValueError("cache_dir must be provided when enable_cache_compile=True")

        compiled = tng.inference.cache_compile(
            model_forward,
            cache_dir=cache_dir,
            config=compiler_config,
            dynamic=dynamic,
            fullgraph=True,
            ge_cache=not use_aclgraph
        )
    else:
        npu_backend = tng.get_npu_backend(compiler_config=compiler_config)
        compiled = torch.compile(model_forward, dynamic=dynamic, fullgraph=True, backend=npu_backend)

    return compiled
